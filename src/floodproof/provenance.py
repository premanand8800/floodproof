"""Independent re-fetch of a copy's publication date.

A date inherited from memory (a seeded/stored entry) is a claim from an earlier session, not evidence. Before a copy can
support RECYCLED_FOOTAGE its date must be re-fetched independently:
  1. yt-dlp --skip-download metadata of the live page ('yt-dlp live metadata')
  2. page meta tags (article:published_time / datePublished) for non-video pages ('page meta')
  3. Internet Archive CDX: earliest snapshot ('wayback_cdx_earliest_snapshot'; an UPPER BOUND, the page existed by then)
Network failure never raises: the result has date=None and a reason, and the inherited date stays 'not independently verified'.
Results are cached in .cache/refetch.json (successes indefinitely, failures for FAIL_TTL_S).
"""
from __future__ import annotations

import json
import re
import time

import requests

from . import CACHE

FAIL_TTL_S = 6 * 3600
_FILE = CACHE / "refetch.json"
_META = re.compile(r'(?:article:published_time|datePublished|og:video:release_date|uploadDate)["\']?\s*(?:content=|:)\s*["\'](\d{4}-\d{2}-\d{2})', re.I)


def _load():
    try:
        return json.loads(_FILE.read_text())
    except Exception:
        return {}


def _save(d):
    try:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        _FILE.write_text(json.dumps(d, indent=1))
    except Exception:
        pass


def _ytdlp(url):
    from . import media
    m = media.probe_url(url) or {}
    return (m.get("upload_date"), m.get("upload_timestamp")) if m.get("upload_date") else None


def _page_meta(url):
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 FloodProof"})
    r.raise_for_status()
    m = _META.search(r.text[:300000])
    return m.group(1) if m else None


def _wayback(url):
    r = requests.get("https://web.archive.org/cdx/search/cdx", timeout=25,
                     params=dict(url=url, output="json", limit=1, fl="timestamp,statuscode", filter="statuscode:200"))
    r.raise_for_status()
    rows = r.json()
    if len(rows) >= 2:
        ts = rows[1][0]
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
    return None


METHODS = (("yt-dlp live metadata", _ytdlp), ("page meta", _page_meta), ("wayback_cdx_earliest_snapshot", _wayback))


def refetch_date(url: str, methods=None, now=time.time) -> dict:
    """-> dict(date|None, source|None, status 'ok'|'failed', errors[list], cached bool, retrieved_at). Never raises."""
    cache = _load()
    hit = cache.get(url)
    if hit and (hit["status"] == "ok" or now() - hit.get("ts", 0) < FAIL_TTL_S):
        return dict(hit, cached=True)
    errs = []
    for name, fn in (methods or METHODS):
        try:
            d = fn(url)
        except Exception as e:
            errs.append(f"{name}: {type(e).__name__}"); continue
        stamp = None
        if isinstance(d, tuple):                     # a method may return (date, exact UTC timestamp)
            d, stamp = d
        if d:
            out = dict(date=d, source=name, status="ok", errors=errs, cached=False, ts=now(), timestamp=stamp,
                       retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()))
            cache[url] = out; _save(cache)
            return out
        errs.append(f"{name}: no date")
    out = dict(date=None, source=None, status="failed", errors=errs, cached=False, ts=now(),
               retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()))
    cache[url] = out; _save(cache)
    return out
