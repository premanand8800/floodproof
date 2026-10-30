"""Relevance gate (step [0/6]): is this video about a disaster/emergency at all?

Cheap and conservative. Metadata (yt-dlp, no download) + 4 stream-seeked frames -> one small Muse call.
Only a confident "no" (confidence >= 0.8, cited frames, no disaster keywords in the metadata) stops the pipeline;
"yes" and "unsure" always run. The caller can override with force=True ("Run anyway").
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from . import CACHE
from . import llm

PRIMARY_MODEL = "meta/muse-glimmer-30b"
FALLBACK_MODEL = "meta/muse-spark-1.3"
N_FRAMES = 4
MIN_CONF_TO_STOP = 0.8
MAX_TOKENS = 1500
GATE_CACHE = CACHE / "gate"
KINDS = ("disaster_footage", "news_about_disaster", "music_video", "gaming", "ad", "talking_head", "other")
DISASTER_RE = re.compile(
    r"flood|tsunami|earthquake|quake|hurricane|cyclone|typhoon|tornado|storm|wildfire|bushfire|landslide|avalanche|"
    r"glacier|glof|eruption|volcano|disaster|catastroph|rescue|evacuat|emergency|survivor|victims|death toll|collapse|"
    r"cloudburst|flash flood|tragedy|devastat|drought|heatwave", re.I)

PROMPT = """You are a relevance filter for a disaster-footage verification tool. You see {n} frames (numbered 1..{n}, in time order) sampled from one video, plus its metadata.
Question: Does this video show, or claim to show, a disaster or emergency (flood, quake, fire, storm, tsunami, landslide, accident, rescue, aftermath, etc.)?
Rules: a music, tribute, news or explainer video built from disaster footage IS relevant. A film trailer, game, ad, vlog, lecture, cooking, sports, unboxing or podcast with no real or claimed disaster footage is NOT relevant. If unsure, say unsure.
Metadata (signals only, may be misleading): {meta}
Reply with ONLY a JSON object:
{{"relevant":"yes|no|unsure","kind":"disaster_footage|news_about_disaster|music_video|gaming|ad|talking_head|other","shows_disaster_or_emergency":true|false,"reason":"one short sentence","refs":[frame numbers that justify the answer],"confidence":0.0-1.0}}"""


def _meta_signals(info: dict) -> dict:
    title = info.get("title") or ""
    tags = [t for t in (info.get("tags") or [])][:25]
    cats = info.get("categories") or []
    blob = " ".join([title, " ".join(tags), (info.get("description") or "")[:300]])
    return dict(category=cats, title=title[:150], tags=tags[:12], duration=info.get("duration"),
                disaster_keywords=sorted({m.group(0).lower() for m in DISASTER_RE.finditer(blob)})[:8])


def _stream_url(info: dict) -> str | None:
    if info.get("url"):
        return info["url"]
    for f in info.get("requested_formats") or []:
        if f.get("vcodec") not in (None, "none") and f.get("url"):
            return f["url"]
    return None


def probe(url: str) -> tuple[dict, str | None, str | None]:
    """(metadata_signals, direct_stream_url, video_id). Metadata only, nothing downloaded."""
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True, "socket_timeout": 30,
            "format": "bv*[height<=360][vcodec!^=av01]/b[height<=360][vcodec!^=av01]/worst[vcodec!^=av01]/worst",
            "cachedir": str(CACHE / "ytdlp")}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
    return _meta_signals(info), _stream_url(info), info.get("id")


def _grab(src: str, t: float) -> bytes | None:
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", src, "-frames:v", "1", "-vf", "scale=-2:360",
                        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "-"], capture_output=True, timeout=60)
    return r.stdout if r.returncode == 0 and len(r.stdout) > 1000 else None


def grab_frames(src: str, duration: float | None, n=N_FRAMES) -> list[bytes]:
    """n frames at evenly spaced fractions of the duration. src may be a stream URL (seeked, not downloaded) or a file."""
    if not duration or duration <= 0:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                           capture_output=True, text=True, timeout=60)
        try:
            duration = float(r.stdout.strip())
        except ValueError:
            duration = 20.0
    from concurrent.futures import ThreadPoolExecutor
    ts = [min(duration * (i + 0.5) / n, max(duration - 1.0, 0)) for i in range(n)]

    def one(t):
        try:
            return _grab(src, t)
        except Exception:
            return None
    with ThreadPoolExecutor(n) as ex:            # independent ffmpeg stream seeks; time order is kept by map()
        return [b for b in ex.map(one, ts) if b]


def _file_key(path: str) -> str:
    p = Path(path)
    h = hashlib.sha1(str(p.stat().st_size).encode())
    with open(p, "rb") as fh:
        h.update(fh.read(1 << 20))
    return "file_" + h.hexdigest()[:16]


def _ask(frames: list[bytes], meta: dict):
    content = [{"type": "text", "text": PROMPT.format(n=len(frames), meta=json.dumps(meta, ensure_ascii=False)[:900])}]
    for i, b in enumerate(frames, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}})
    cost, last = 0.0, None
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            txt, c = llm.chat(model, content, max_tokens=MAX_TOKENS if model == PRIMARY_MODEL else 3000, timeout=90)
            cost += c
            return llm.parse_json(txt), model, cost
        except Exception as e:                                   # fall through to the next model
            last = e
    raise RuntimeError(f"gate model call failed: {str(last)[:150]}")


def decide(ans: dict, meta: dict) -> tuple[str, dict]:
    """Return ('stop'|'run', normalised answer). Only a confident, frame-cited 'no' with no disaster keywords stops."""
    kind = ans.get("kind") if ans.get("kind") in KINDS else "other"
    try:
        conf = max(0.0, min(1.0, float(ans.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0
    refs = [int(x) for x in (ans.get("refs") or []) if str(x).lstrip("-").isdigit()][:4]
    rel = str(ans.get("relevant", "unsure")).lower()
    rel = rel if rel in ("yes", "no", "unsure") else "unsure"
    shows = bool(ans.get("shows_disaster_or_emergency"))
    norm = dict(relevant=rel, kind=kind, reason=str(ans.get("reason") or "")[:240], refs=refs, confidence=round(conf, 2), shows=shows)
    stop = (rel == "no" and conf >= MIN_CONF_TO_STOP and bool(refs) and not shows and not meta.get("disaster_keywords"))
    return ("stop" if stop else "run"), norm


def check(source: str, is_url: bool | None = None, force=False, log=print) -> dict:
    """Run the gate. Returns the report 'gate' section. Never raises: any failure means decision='run' (fail open)."""
    t0 = time.time()
    is_url = bool(re.match(r"https?://", source)) if is_url is None else is_url
    sec = dict(decision="run", kind=None, reason="", refs=[], confidence=None, metadata_signals={}, cost=0.0, seconds=0.0)
    try:
        if is_url:
            meta, stream, vid = probe(source)
            key, src = f"yt_{vid}" if vid else "url_" + hashlib.sha1(source.encode()).hexdigest()[:12], stream
            dur = meta.get("duration")
        else:
            meta, key, src, dur = {}, _file_key(source), source, None
        sec["metadata_signals"] = meta
        cf = GATE_CACHE / f"{re.sub(r'[^A-Za-z0-9_-]', '_', key)}.json"
        if cf.exists():
            c = json.loads(cf.read_text())
            c.update(cost=0.0, seconds=round(time.time() - t0, 1), cached=True)
            if force and c["decision"] == "stop":
                c["decision"] = "run_forced"
            return c
        if not src:
            raise RuntimeError("no direct stream url")
        frames = grab_frames(src, dur)
        if len(frames) < 2:
            raise RuntimeError(f"only {len(frames)} frames could be grabbed")
        ans, model, cost = _ask(frames, meta)
        verdict, n = decide(ans, meta)
        sec.update(decision=verdict, kind=n["kind"], reason=n["reason"], refs=n["refs"], confidence=n["confidence"],
                   relevant=n["relevant"], shows_disaster_or_emergency=n["shows"], model=model, cost=round(cost, 5))
        sec["seconds"] = round(time.time() - t0, 1)
        GATE_CACHE.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(sec))
        if force and sec["decision"] == "stop":
            sec["decision"] = "run_forced"
        return sec
    except Exception as e:
        log("gate skipped (fail open):", f"{type(e).__name__}: {str(e)[:150]}")
        sec.update(error=f"{type(e).__name__}: {str(e)[:150]}", seconds=round(time.time() - t0, 1))
        return sec


def headline(g: dict) -> str:
    refs = ",".join(str(r) for r in g.get("refs") or [])
    kind = (g.get("kind") or "other").replace("_", " ")
    return f"This looks like a {kind} (frames {refs}: {g.get('reason', '').rstrip('.')}). FloodProof checks disaster footage."
