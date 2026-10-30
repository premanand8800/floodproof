"""Candidate discovery adapters. They only PROPOSE URLs; nothing counts until the matcher proves it.

Order used by pipeline.investigate(): memory lookup -> Yandex reverse image search (2-3 keyframes, scripted
Playwright; a captcha means we skip, we never try to bypass it) -> Tavily text search from Muse queries.
`verify_candidates` downloads each candidate (<=360p, first <=150 s), runs the matcher, and keeps matcher-proven ones.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import requests

from . import CACHE, load_env
from .matcher import ClipFeatures, match
from . import media

VIDEO_HOSTS = ("youtube.com/watch", "youtube.com/shorts", "youtu.be", "instagram.com/reel", "instagram.com/p/",
               "facebook.com/watch", "facebook.com/reel", "x.com/", "twitter.com/", "tiktok.com", "dailymotion.com/video",
               "vimeo.com")
IMG_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?|$)", re.I)


@dataclass
class Candidate:
    url: str
    title: str = ""
    adapter: str = ""            # memory | yandex | tavily
    hint: str = ""               # which keyframe / query produced it
    status: str = "pending"      # proven | not_proven | unavailable | too_long | later_than_query | download_failed | error
    upload_date: str | None = None
    date_source: str | None = None   # where upload_date came from: 'yt-dlp metadata' | 'memory entry' (never guessed)
    retrieved_at: str | None = None  # UTC timestamp of the probe/lookup
    uploader: str | None = None
    proven: bool = False
    match: dict | None = None
    evidence_png: str | None = None
    memory_entry: dict | None = None

    def to_dict(self):
        return asdict(self)


def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _clean_url(u):
    return re.sub(r"[?&]utm_[^&]*", "", u).rstrip("?&")


# ---------------------------------------------------------------- memory
def memory_candidates(mem, feats: ClipFeatures, evidence_dir: Path | None = None, **kw):
    res = mem.lookup(feats, evidence_dir=evidence_dir, **kw)
    cands = []
    for h in res.hits:
        e = h.entry
        cands.append(Candidate(e["source_url"] or f"memory:{e['id']}", e.get("title") or "", "memory", f"entry {e['id']}",
                               "proven", e.get("upload_date"), "memory entry" if e.get("upload_date") else None,
                               _now(), e.get("uploader"), True, h.match.to_dict(),
                               str(Path(evidence_dir) / f"memory_{e['id']}.png") if evidence_dir else None, e))
    return cands, res


# ---------------------------------------------------------------- yandex
def yandex_candidates(feats: ClipFeatures, n_frames=3, workdir: Path | None = None, headless=True, max_links=40):
    """Scripted Yandex Images reverse search. Returns (candidates, info). A captcha => skipped (never bypassed)."""
    import numpy as np
    from playwright.sync_api import sync_playwright
    wd = workdir or (CACHE / "yandex"); wd.mkdir(parents=True, exist_ok=True)
    # sharpest frames, spread out
    sharp = sorted(((cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(), i) for i, f in enumerate(feats.frames)), reverse=True)
    pick = []
    for _, i in sharp:
        if all(abs(i - j) > max(2, len(feats) * 0.15) for j in pick):
            pick.append(i)
        if len(pick) == n_frames:
            break
    info = dict(frames=pick, captcha=False, searched=0, errors=[])
    seen = {}
    with sync_playwright() as p:
        b = p.chromium.launch(headless=headless)
        for i in pick:
            img = wd / f"{hashlib.sha1(feats.path.encode()).hexdigest()[:8]}_{i}.jpg"
            cv2.imwrite(str(img), feats.frames[i])
            pg = b.new_context(viewport={"width": 1280, "height": 800}, locale="en-US").new_page()
            try:
                pg.goto("https://yandex.com/images/", timeout=40000); time.sleep(2)
                if "captcha" in pg.url:
                    info["captcha"] = True; break
                pg.locator("input[type=file]").first.set_input_files(str(img)); time.sleep(7)
                if "captcha" in pg.url or "showcaptcha" in pg.url:
                    info["captcha"] = True; break          # skip; never bypass
                try:
                    pg.get_by_text("Sites", exact=True).first.click(); time.sleep(4)
                except Exception:
                    pass
                links = pg.evaluate("""()=>[...document.querySelectorAll('a[href^=http]')].map(a=>[a.href,(a.innerText||'').trim().slice(0,120)]).filter(x=>!x[0].includes('yandex.'))""")
                info["searched"] += 1
                for u, t in links:
                    if "ytimg.com" in u or IMG_RE.search(u) or "googleusercontent" in u or len(t) <= 12 or t.endswith(".com"):
                        continue
                    seen.setdefault(_clean_url(u), (t, f"frame {i}"))
            except Exception as e:
                info["errors"].append(str(e)[:120])
            finally:
                pg.context.close(); img.unlink(missing_ok=True)
            time.sleep(2)
        b.close()
    items = sorted(seen.items(), key=lambda kv: not any(h in kv[0] for h in VIDEO_HOSTS))[:max_links]
    return [Candidate(u, t, "yandex", h) for u, (t, h) in items], info


# ---------------------------------------------------------------- tavily
def tavily_candidates(queries, per_query=6, max_queries=6, max_links=30):
    load_env()
    key = os.environ.get("TAVILY_API_KEY")
    info = dict(queries=0, errors=[])
    if not key:
        return [], dict(info, errors=["TAVILY_API_KEY missing"])
    seen = {}
    for q in queries[:max_queries]:
        for kw in ({"include_domains": ["youtube.com"]}, {"include_domains": ["instagram.com", "facebook.com", "x.com", "tiktok.com"]}, {"topic": "news"}):
            try:
                r = requests.post("https://api.tavily.com/search", json={"query": q, "max_results": per_query, **kw},
                                  headers={"Authorization": f"Bearer {key}"}, timeout=60).json()
                info["queries"] += 1
            except Exception as e:
                info["errors"].append(str(e)[:100]); continue
            for x in r.get("results", []):
                seen.setdefault(_clean_url(x["url"]), (x.get("title", ""), q))
    items = sorted(seen.items(), key=lambda kv: not any(h in kv[0] for h in VIDEO_HOSTS))[:max_links]
    return [Candidate(u, t, "tavily", f"query: {q[:60]}") for u, (t, q) in items], info


# ---------------------------------------------------------------- verification
def verify_candidate(c: Candidate, q: ClipFeatures, query_date: str | None = None, self_url: str | None = None,
                     evidence_dir: Path | None = None) -> Candidate:
    """Probe -> download -> matcher. Mutates and returns c; c.proven only if the matcher decision is positive."""
    if c.proven:                                   # already proven by memory
        return c
    if self_url and _clean_url(self_url) == c.url:
        c.status = "skipped_self"; return c
    meta = media.probe_url(c.url)
    if not meta:
        c.status = "unavailable"; return c
    c.upload_date, c.uploader = meta["upload_date"], meta["uploader"]
    c.date_source = "yt-dlp metadata" if c.upload_date else None
    c.retrieved_at = _now()
    if not c.title:
        c.title = meta.get("title") or ""
    if query_date and c.upload_date and c.upload_date > query_date:
        c.status = "later_than_query"; return c       # cannot be the earlier source of the query clip
    dur = meta.get("duration") or 60
    if dur > 600:
        c.status = "too_long"; return c
    try:
        clip = media.fetch(c.url, max_seconds=150 if dur > 120 else 120, max_height=360 if dur <= 120 else 240)
    except Exception as e:
        c.status = "download_failed"; c.hint += f" ({str(e)[:60]})"; return c
    try:
        B = ClipFeatures.from_path(clip.path, maxdur=150, fps=2.0 if dur > 120 else 3.0, maxframes=400)
        m = match(q, B, coarse=12)
        c.match = m.to_dict(); c.proven = m.decision
        c.status = "proven" if m.decision else "not_proven"
        if m.decision and evidence_dir:
            tag = hashlib.sha1(c.url.encode()).hexdigest()[:8]
            c.evidence_png = m.evidence(Path(evidence_dir) / f"{c.adapter}_{tag}.png", f"{c.adapter} candidate {c.url[-50:]} date={c.upload_date}")
    except Exception as e:
        c.status = "error"; c.hint += f" ({str(e)[:60]})"
    finally:
        media.cleanup(clip)
    return c


def verify_candidates(cands, q, query_date=None, self_url=None, evidence_dir=None, max_n=14, workers=3):
    todo = [c for c in cands if not c.proven][:max_n]
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda c: verify_candidate(c, q, query_date, self_url, evidence_dir), todo))
    return cands


def earliest_proven_copy(cands):
    """Earliest matcher-proven copy by platform upload date (dated copies only). A copy, not necessarily the source."""
    dated = [c for c in cands if c.proven and c.upload_date]
    return min(dated, key=lambda c: c.upload_date) if dated else None
