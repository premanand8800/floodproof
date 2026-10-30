"""Candidate discovery adapters. They only PROPOSE URLs; nothing counts until the matcher proves it.

Order used by pipeline.investigate(): memory lookup -> Google Vision web detection (GOOGLE_API_KEY) -> YouTube Data API
search (GOOGLE_API_KEY) -> Yandex reverse image search (2-3 keyframes, scripted Playwright; a captcha means we skip, we
never try to bypass it) -> SerpApi Google Lens (SERPAPI_KEY, only if nothing is proven yet, 1 call) -> Tavily text search
from Muse queries. Key-gated adapters are inert ("skipped (no key)") until their key exists.
`verify_candidates` downloads each candidate (<=360p, first <=150 s), runs the matcher, and keeps matcher-proven ones.
"""
from __future__ import annotations

import concurrent.futures as cf
import base64
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import requests

from . import CACHE, load_env
from .matcher import ClipFeatures, match, proof_artifacts
from . import media

VIDEO_HOSTS = ("youtube.com/watch", "youtube.com/shorts", "youtu.be", "instagram.com/reel", "instagram.com/p/",
               "facebook.com/watch", "facebook.com/reel", "x.com/", "twitter.com/", "tiktok.com", "dailymotion.com/video",
               "vimeo.com")
log = logging.getLogger("floodproof.sources")
SKIP_NO_KEY = "skipped (no key)"
LENS_HOSTS = ("instagram.com", "tiktok.com", "facebook.com", "fb.watch", "youtube.com", "youtu.be")
PUBLIC_BASE = "https://floodproof.aether-soft.tech"
IMG_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?|$)", re.I)


@dataclass
class Candidate:
    url: str
    title: str = ""
    adapter: str = ""            # memory | gvision | youtube_api | yandex | lens | tavily
    hint: str = ""               # which keyframe / query produced it
    status: str = "pending"      # proven | not_proven | unavailable | too_long | later_than_query | download_failed | error
    upload_date: str | None = None
    date_source: str | None = None   # where upload_date came from: 'yt-dlp metadata' | 'memory entry' (never guessed)
    retrieved_at: str | None = None  # UTC timestamp of the probe/lookup
    uploader: str | None = None
    proven: bool = False
    match: dict | None = None
    evidence_png: str | None = None
    proof: dict | None = None          # clean best-pair frames + RANSAC inlier coordinates (+ optional Muse explanation)
    memory_entry: dict | None = None
    stored_date: str | None = None          # date inherited from a memory entry (NOT evidence by itself)
    independent_date: str | None = None       # date independently re-fetched (live metadata / page meta / Wayback)
    independent_date_source: str | None = None
    date_independent: bool = False          # True only if upload_date came from a live fetch in THIS investigation
    date_note: str | None = None
    date_alternatives: list = field(default_factory=list)
    also_from: list = field(default_factory=list)   # other adapters that proposed the same URL (merged by dedupe)

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
                               "proven", e.get("upload_date"), "memory entry (inherited)" if e.get("upload_date") else None,
                               _now(), e.get("uploader"), True, h.match.to_dict(),
                               str(Path(evidence_dir) / f"memory_{e['id']}.png") if evidence_dir else None, None, e,
                               stored_date=e.get("upload_date"), date_independent=False))
        if evidence_dir:
            try:
                cands[-1].proof = proof_artifacts(h.match, evidence_dir, f"memory_{e['id']}")
            except Exception:
                pass
    return cands, res


# ---------------------------------------------------------------- key-gated adapters (inert without a key)
def _post_json(url, params=None, body=None, timeout=60):
    """POST json -> dict; isolated so tests can monkeypatch (no network in tests)."""
    r = requests.post(url, params=params, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get_json(url, params=None, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _spread_frames(feats: ClipFeatures, n):
    """n sharp keyframe ids, spread out in time."""
    sharp = sorted(((cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(), i) for i, f in enumerate(feats.frames)), reverse=True)
    pick = []
    for _, i in sharp:
        if all(abs(i - j) > max(1, len(feats) * 0.12) for j in pick):
            pick.append(i)
        if len(pick) == n:
            break
    return pick or list(range(min(n, len(feats))))


def _yt_from_thumb(u):
    m = re.search(r"i\d?\.ytimg\.com/vi(?:_webp)?/([\w-]{11})/", u)
    return f"https://www.youtube.com/watch?v={m.group(1)}" if m else None


def gvision_candidates(feats: ClipFeatures, n_frames=4, max_links=40, key=None, frames: list | None = None):
    """Google Cloud Vision WEB_DETECTION on 3-4 keyframes. Candidates only; the matcher still has to prove each."""
    load_env()
    key = key or os.environ.get("GOOGLE_API_KEY")
    info = dict(source="gvision", frames=[], calls=0, errors=[])
    if not key:
        log.info("gvision: %s", SKIP_NO_KEY)
        return [], dict(info, skipped=SKIP_NO_KEY)
    pick = list(frames) if frames else _spread_frames(feats, min(max(n_frames, 3), 4))
    reqs = []
    for i in pick:
        ok, buf = cv2.imencode(".jpg", feats.frames[i], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            reqs.append(dict(image=dict(content=base64.b64encode(buf.tobytes()).decode()),
                             features=[dict(type="WEB_DETECTION", maxResults=30)]))
            info["frames"].append(int(i))
    if not reqs:
        return [], dict(info, errors=["no frames"])
    try:
        resp = _post_json("https://vision.googleapis.com/v1/images:annotate", params={"key": key}, body=dict(requests=reqs))
        info["calls"] = 1
    except Exception as e:
        info["errors"].append(type(e).__name__ + ": " + str(e)[:100].replace(key, "***"))
        return [], info
    seen = {}
    for fi, r in zip(info["frames"], resp.get("responses", [])):
        if r.get("error"):
            info["errors"].append(str(r["error"].get("message"))[:100]); continue
        wd = r.get("webDetection") or {}
        for pg in wd.get("pagesWithMatchingImages", []):
            u = pg.get("url")
            if u:
                seen.setdefault(_clean_url(u), (pg.get("pageTitle") or "", f"frame {fi} page match"))
        for kind in ("fullMatchingImages", "partialMatchingImages"):
            for im in wd.get(kind, []):
                u = im.get("url") or ""
                u = _yt_from_thumb(u) or (None if IMG_RE.search(u) or "ytimg.com" in u else u)
                if u:
                    seen.setdefault(_clean_url(u), ("", f"frame {fi} {kind}"))
    items = sorted(seen.items(), key=lambda kv: not any(h in kv[0] for h in VIDEO_HOSTS))[:max_links]
    return [Candidate(u, re.sub(r"<[^>]+>", "", t), "gvision", h) for u, (t, h) in items], info


def youtube_api_candidates(queries, claim_date=None, max_queries=3, max_results=15, key=None):
    """YouTube Data API v3 search.list: videos published on/before claim date + 1 day, oldest-relevant first (order=date).
    The snippet publishedAt is kept only as stored_date; verify_candidate still re-fetches the date independently."""
    import datetime as dt
    load_env()
    key = key or os.environ.get("GOOGLE_API_KEY")
    info = dict(source="youtube_api", queries=0, errors=[])
    if not key:
        log.info("youtube_api: %s", SKIP_NO_KEY)
        return [], dict(info, skipped=SKIP_NO_KEY)
    params0 = dict(part="snippet", type="video", order="date", maxResults=max_results, key=key)
    if claim_date:
        try:
            before = dt.date.fromisoformat(claim_date) + dt.timedelta(days=1)
            params0["publishedBefore"] = f"{before.isoformat()}T00:00:00Z"
        except ValueError:
            pass
    seen = {}
    for q in list(queries)[:max_queries]:
        try:
            r = _get_json("https://www.googleapis.com/youtube/v3/search", params=dict(params0, q=q[:200]))
            info["queries"] += 1
        except Exception as e:
            info["errors"].append(type(e).__name__ + ": " + str(e)[:100].replace(key, "***")); continue
        for it in r.get("items", []):
            vid = (it.get("id") or {}).get("videoId")
            sn = it.get("snippet") or {}
            if vid:
                seen.setdefault(f"https://www.youtube.com/watch?v={vid}", (sn.get("title", ""), (sn.get("publishedAt") or "")[:10] or None, q))
    out = []
    for u, (t, d, q) in seen.items():
        c = Candidate(u, t, "youtube_api", f"query: {q[:60]}")
        c.stored_date = d
        out.append(c)
    return out, info


def public_frame_url(img_path: Path, out_dir: Path, runs_dir: Path, base: str | None = None):
    """Public URL of a keyframe JPEG, only if it lives inside a served run dir (<runs>/<id>/...). Else None."""
    try:
        img_path, out_dir, runs_dir = Path(img_path).resolve(), Path(out_dir).resolve(), Path(runs_dir).resolve()
        if out_dir.parent != runs_dir or out_dir not in img_path.parents or img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            return None
        return f"{(base or os.environ.get('FLOODPROOF_PUBLIC_BASE') or PUBLIC_BASE).rstrip('/')}/files/run/{out_dir.name}/{img_path.relative_to(out_dir).as_posix()}"
    except Exception:
        return None


def lens_candidates(image_url: str | None, max_links=30, key=None):
    """SerpApi google_lens on ONE public image URL (max 1 call per investigation). Keeps Instagram/TikTok/Facebook/YouTube links."""
    load_env()
    key = key or os.environ.get("SERPAPI_KEY")
    info = dict(source="lens", calls=0, errors=[])
    if not key:
        log.info("lens: %s", SKIP_NO_KEY)
        return [], dict(info, skipped=SKIP_NO_KEY)
    if not image_url:
        log.info("lens: skipped (no public image URL)")
        return [], dict(info, skipped="no public image URL")
    try:
        r = _get_json("https://serpapi.com/search.json", params=dict(engine="google_lens", url=image_url, api_key=key))
        info["calls"] = 1
    except Exception as e:
        info["errors"].append(type(e).__name__ + ": " + str(e)[:100].replace(key, "***"))
        return [], info
    seen = {}
    for m in r.get("visual_matches", []):
        u = m.get("link") or ""
        if any(h in u for h in LENS_HOSTS):
            seen.setdefault(_clean_url(u), m.get("title") or "")
    return [Candidate(u, t, "lens", "google lens best keyframe") for u, t in list(seen.items())[:max_links]], info


def merge_and_prioritize(cands):
    """Dedupe by URL (remember every adapter that proposed it) and order for the verify budget: already proven first,
    then URLs proposed by several sources, then video-platform URLs. Stable otherwise."""
    uniq = {}
    for c in cands:
        k = uniq.get(c.url)
        if k is None:
            uniq[c.url] = c
        elif c.adapter != k.adapter and c.adapter not in k.also_from:
            k.also_from.append(c.adapter)
    out = list(uniq.values())
    out.sort(key=lambda c: (not c.proven, -len(c.also_from), not any(h in c.url for h in VIDEO_HOSTS)))
    return out


def source_stats(cands):
    """Per-adapter {candidates, checked, proven}; a URL proposed by two adapters counts for both."""
    st = {}
    for c in cands:
        for a in [c.adapter] + list(c.also_from):
            d = st.setdefault(a, dict(candidates=0, checked=0, proven=0))
            d["candidates"] += 1
            d["checked"] += c.status != "pending"
            d["proven"] += bool(c.proven)
    return st


# ---------------------------------------------------------------- yandex
def yandex_candidates(feats: ClipFeatures, n_frames=3, workdir: Path | None = None, headless=True, max_links=40,
                      frames: list | None = None, crop: tuple | None = None):
    """Scripted Yandex Images reverse search. Returns (candidates, info). A captcha => skipped (never bypassed)."""
    import numpy as np
    from playwright.sync_api import sync_playwright
    wd = workdir or (CACHE / "yandex"); wd.mkdir(parents=True, exist_ok=True)
    # sharpest frames, spread out
    sharp = sorted(((cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(), i) for i, f in enumerate(feats.frames)), reverse=True)
    pick = []
    if frames:                       # caller (the agent) chose the keyframes; ids are validated here
        pick = [int(i) for i in frames if 0 <= int(i) < len(feats)][:4]
    for _, i in (sharp if not pick else []):
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
            fr = feats.frames[i]
            if crop:                 # normalized 0-1000 box (x0, y0, x1, y1)
                h, w = fr.shape[:2]; x0, y0, x1, y1 = crop
                fr = fr[int(y0 * h / 1000):max(int(y1 * h / 1000), int(y0 * h / 1000) + 8), int(x0 * w / 1000):max(int(x1 * w / 1000), int(x0 * w / 1000) + 8)]
            cv2.imwrite(str(img), fr)
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


def tavily_search(query, site=None, after=None, before=None, max_results=8):
    """One agent-written Tavily query with optional site / date filters. Returns (results, info); results are
    {url,title,snippet,published_date} and are only PROPOSALS (untrusted text)."""
    load_env()
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return [], dict(error="TAVILY_API_KEY missing")
    body = {"query": query[:300], "max_results": max_results}
    if site:
        body["include_domains"] = [site] if isinstance(site, str) else list(site)[:5]
    if after:
        body["start_date"] = after
    if before:
        body["end_date"] = before
    try:
        r = requests.post("https://api.tavily.com/search", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60).json()
    except Exception as e:
        return [], dict(error=str(e)[:100])
    return [dict(url=_clean_url(x["url"]), title=x.get("title", ""), snippet=(x.get("content") or "")[:160],
                 published_date=x.get("published_date")) for x in r.get("results", [])], dict(error=r.get("detail") and str(r["detail"])[:100])


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
    c.date_independent = bool(c.upload_date); c.independent_date = c.upload_date; c.independent_date_source = c.date_source
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
            c.proof = proof_artifacts(m, evidence_dir, f"{c.adapter}_{tag}")
    except Exception as e:
        c.status = "error"; c.hint += f" ({str(e)[:60]})"
    finally:
        media.cleanup(clip)
    return c


def verify_candidates(cands, q, query_date=None, self_url=None, evidence_dir=None, max_n=14, workers=3):
    todo = [c for c in cands if not c.proven and c.status == "pending"][:max_n]
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda c: verify_candidate(c, q, query_date, self_url, evidence_dir), todo))
    return cands


def earliest_proven_copy(cands):
    """Earliest matcher-proven copy by platform upload date (dated copies only). A copy, not necessarily the source."""
    dated = [c for c in cands if c.proven and c.upload_date]
    return min(dated, key=lambda c: c.upload_date) if dated else None


def enrich_dates(cands, refetch=None):
    """For proven copies whose date is only inherited (memory), re-fetch it independently. Sets upload_date/date_source to the
    re-fetched value (inherited date kept in stored_date and, if different, in date_alternatives); on failure the inherited
    date stays and date_independent stays False (the verdict layer then refuses RECYCLED on it). Never raises."""
    from .provenance import refetch_date
    refetch = refetch or refetch_date
    for c in cands:
        if not c.proven or c.date_independent:
            continue
        try:
            r = refetch(c.url)
        except Exception as e:                       # belt and braces: provenance never raises, but a patched one might
            r = dict(date=None, status="failed", errors=[type(e).__name__])
        if r.get("date"):
            c.independent_date, c.independent_date_source = r["date"], r["source"]
            if c.stored_date and c.stored_date != r["date"]:
                c.date_alternatives = [dict(date=c.stored_date, source="memory entry (inherited)")]
            c.upload_date, c.date_source, c.date_independent = r["date"], r["source"], True
            c.retrieved_at = r.get("retrieved_at") or c.retrieved_at
            c.date_note = ("Wayback snapshot date is an upper bound: the page existed by then, it may be older."
                           if str(r["source"]).startswith("wayback") else None)
        else:
            c.date_independent = False
            c.date_note = "date re-fetch failed (" + "; ".join(r.get("errors") or ["no reason"]) + "); inherited date not independently verified"
    return cands
