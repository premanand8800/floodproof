"""Fetch a clip (local file or URL via yt-dlp), read metadata, extract keyframes. Everything lives in ./.cache."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

from . import CACHE

MEDIA_DIR = CACHE / "dl"          # downloads cached by URL hash (sha1 of url|max_seconds|max_height) and reused across runs
MEDIA_DIR.mkdir(exist_ok=True)
DL_CACHE_MAX_BYTES = 1_500_000_000
DL_META_TTL_S = 7 * 86400
MAX_SECONDS = 120
MAX_HEIGHT = 480
# AV1 is skipped: OpenCV's ffmpeg build cannot decode it everywhere.
FMT = "bv*[height<={h}][vcodec!^=av01]+ba/b[height<={h}][vcodec!^=av01]/bv*[height<={h}][vcodec!^=av01]/worst[vcodec!^=av01]"


@dataclass
class Clip:
    path: str
    url: str | None = None          # canonical source URL (None for a local file)
    uploader: str | None = None
    upload_date: str | None = None  # ISO YYYY-MM-DD when known (UTC calendar date from yt-dlp)
    title: str | None = None
    duration: float | None = None   # seconds of the *source* video (may exceed the downloaded excerpt)
    has_audio: bool = False
    truncated: bool = False         # downloaded excerpt shorter than source
    description: str | None = None  # post caption/description (yt-dlp), used for claim inference
    upload_timestamp: str | None = None   # exact UTC publication time (ISO, from yt-dlp timestamp/release_timestamp) when the platform gives it

    def to_dict(self):
        return asdict(self)


def iso_date(d):
    if not d or d in ("NA", "None"):
        return None
    d = str(d)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if re.fullmatch(r"\d{8}", d) else d[:10]


def iso_timestamp(info):
    """Exact UTC publication time from yt-dlp metadata (timestamp, else release_timestamp) as ISO 'YYYY-MM-DDTHH:MM:SSZ', or None."""
    import datetime
    for k in ("timestamp", "release_timestamp"):
        v = (info or {}).get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            try:
                return datetime.datetime.fromtimestamp(v, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (OverflowError, OSError, ValueError):
                continue
    return None


def is_url(s: str) -> bool:
    return bool(re.match(r"https?://", s))


def _ytdlp_opts(out_tpl, max_seconds, max_height, quiet=True):
    from yt_dlp.utils import download_range_func
    return {"quiet": quiet, "no_warnings": True, "format": FMT.format(h=max_height),
            "download_ranges": download_range_func(None, [(0, max_seconds)]),
            "merge_output_format": "mp4", "outtmpl": out_tpl, "max_filesize": 60_000_000,
            "noplaylist": True, "socket_timeout": 30, "retries": 2, "cachedir": str(CACHE / "ytdlp")}


def probe_url(url: str) -> dict | None:
    """Metadata without downloading (None if unavailable). Keys: upload_date (ISO), uploader, duration, title."""
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True,
                               "socket_timeout": 30, "cachedir": str(CACHE / "ytdlp")}) as y:
            i = y.extract_info(url, download=False)
    except Exception:
        return None
    if not i:
        return None
    return dict(upload_date=iso_date(i.get("upload_date")), upload_timestamp=iso_timestamp(i), uploader=i.get("uploader") or i.get("channel"),
                duration=i.get("duration"), title=i.get("title"), url=i.get("webpage_url") or url)


def has_audio_stream(path) -> bool:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30)
        return "audio" in r.stdout
    except Exception:
        return False


def fetch(source: str, max_seconds=MAX_SECONDS, max_height=MAX_HEIGHT, workdir: Path | None = None) -> Clip:
    """Local file -> Clip (metadata empty); URL -> yt-dlp download (<=max_height, first max_seconds s)."""
    if not is_url(source):
        p = Path(source).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(source)
        cap = cv2.VideoCapture(str(p)); n = cap.get(cv2.CAP_PROP_FRAME_COUNT); f = cap.get(cv2.CAP_PROP_FPS) or 25; cap.release()
        dur = n / f if n else None
        return Clip(str(p), None, None, None, p.name, dur, has_audio_stream(p), bool(dur and dur > max_seconds))
    import yt_dlp
    d = (workdir or MEDIA_DIR) / hashlib.sha1(f"{source}|{max_seconds}|{max_height}".encode()).hexdigest()[:12]
    meta_f = d / "meta.json"
    if meta_f.exists() and time.time() - meta_f.stat().st_mtime < DL_META_TTL_S:     # stale metadata (title, dates) is not reused forever
        m = json.loads(meta_f.read_text())
        if Path(m["path"]).exists():
            d.touch()
            return Clip(**{k: v for k, v in m.items() if k in Clip.__dataclass_fields__})
    d.mkdir(parents=True, exist_ok=True)
    with yt_dlp.YoutubeDL(_ytdlp_opts(str(d / "clip.%(ext)s"), max_seconds, max_height)) as y:
        info = y.extract_info(source, download=True)
    files = [f for f in d.glob("clip.*") if f.suffix not in (".part", ".json")]
    if not files:
        raise RuntimeError(f"download produced no file for {source}")
    path = files[0]
    dur = info.get("duration")
    clip = Clip(str(path), info.get("webpage_url") or source, info.get("uploader") or info.get("channel"),
                iso_date(info.get("upload_date")), info.get("title"), dur, has_audio_stream(path),
                bool(dur and dur > max_seconds), (info.get("description") or None) and info["description"][:2000],
                upload_timestamp=iso_timestamp(info))
    meta_f.write_text(json.dumps(clip.to_dict()))
    return clip


def keyframes(path, **kw):
    """Keyframes (scene change + uniform subsample) as ClipFeatures; see matcher.ClipFeatures.from_path."""
    from .matcher import ClipFeatures
    return ClipFeatures.from_path(path, **kw)


def save_keyframe_jpgs(feats, out_dir: Path, indices=None, quality=80):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in (indices if indices is not None else range(len(feats))):
        p = out_dir / f"frame_{i:02d}_t{feats.times[i]:.1f}.jpg"
        cv2.imwrite(str(p), feats.frames[i], [cv2.IMWRITE_JPEG_QUALITY, quality])
        paths.append(str(p))
    return paths


def extract_audio(path, out_mp3: Path, max_seconds=60) -> Path | None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vn", "-t", str(max_seconds), "-ac", "1",
                        "-ar", "16000", "-b:a", "32k", str(out_mp3)], capture_output=True)
    return out_mp3 if r.returncode == 0 and out_mp3.exists() and out_mp3.stat().st_size > 1000 else None


def _prune_cache(keep: Path | None = None, max_bytes: int = DL_CACHE_MAX_BYTES):
    """Least-recently-used eviction so .cache/dl stays bounded. Never raises."""
    try:
        dirs = [d for d in MEDIA_DIR.iterdir() if d.is_dir() and d != keep]
        sizes = {d: sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) for d in dirs}
        total = sum(sizes.values())
        for d in sorted(dirs, key=lambda d: d.stat().st_mtime):
            if total <= max_bytes:
                break
            shutil.rmtree(d, ignore_errors=True)
            total -= sizes[d]
    except Exception:
        pass


def cleanup(clip: Clip):
    """Downloads are cached by URL hash in .cache/dl and reused across runs, so this only touches the cache's LRU bound (it never deletes
    a user-supplied local file)."""
    if clip.url:
        try:
            Path(clip.path).parent.touch()
        except OSError:
            pass
        _prune_cache(keep=Path(clip.path).parent)
