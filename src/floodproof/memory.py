"""Investigation memory: an on-disk index of known originals and previously investigated footage.

Storage (default ./data/memory):  index.sqlite  (entries + per-frame pHash/dHash + colour histogram)
                                  features/<id>.npz (keyframe thumbnails + SIFT keypoints/descriptors)

lookup(clip) is a two-stage search:
  1. CHEAP PREFILTER: rank entries by (a) smallest pHash+dHash Hamming distance over frame pairs, mirrored query
     included, and (b) global colour-histogram distance; shortlist the top_k by combined rank.
  2. PROOF: run the SIFT/RANSAC matcher (with its unchanged decision rule) on each shortlisted entry. Only
     matcher-proven entries are returned.

HONESTY NOTE: hashes only catch near-identical frames. A fresh crop, heavy overlay or re-edit can push the true
original out of the hash shortlist. `exhaustive=True` proves against EVERY entry (slow, O(entries)); the default
shortlist is a speed/recall trade-off, and bench/evaluate.py measures it rather than assuming it.
"""
from __future__ import annotations

import io
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import ROOT
from .matcher import ClipFeatures, MatchResult, hashes, match

DEFAULT_DIR = ROOT / "data" / "memory"

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries(
  id INTEGER PRIMARY KEY AUTOINCREMENT, clip_key TEXT UNIQUE, source_url TEXT, uploader TEXT, upload_date TEXT,
  title TEXT, event_context TEXT, investigation_id TEXT, role TEXT, origin TEXT, n_frames INTEGER, hist BLOB,
  feat_file TEXT, added_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS frames(entry_id INTEGER, idx INTEGER, t REAL, phash TEXT, dhash TEXT);
CREATE INDEX IF NOT EXISTS frames_e ON frames(entry_id);
"""

ROLES = ("fact_check_cited_original", "investigated_input", "matcher_proven_copy", "manual")


@dataclass
class MemoryHit:
    entry: dict
    match: MatchResult

    def to_dict(self):
        return dict(entry=self.entry, match=self.match.to_dict())


@dataclass
class LookupResult:
    hits: list            # MemoryHit, proven only, best first
    n_entries: int
    shortlisted: list     # entry ids sent to the matcher
    mode: str
    seconds: float

    @property
    def hit(self):
        return bool(self.hits)


def _save_features(path: Path, f: ClipFeatures):
    pts = np.concatenate([x[0] for x in f.feats]) if f.feats else np.zeros((0, 2), np.float32)
    sz = np.concatenate([x[1] for x in f.feats])
    desc = np.concatenate([np.zeros((0, 128), np.float32) if x[2] is None else x[2] for x in f.feats]).astype(np.uint8)
    counts = np.array([len(x[0]) for x in f.feats])
    thumbs, tcounts = [], []
    for fr in f.frames:
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        thumbs.append(buf.tobytes()); tcounts.append(len(thumbs[-1]))
    np.savez_compressed(path, times=np.array(f.times), pts=pts, sz=sz, desc=desc, counts=counts,
                        thumbs=np.frombuffer(b"".join(thumbs), np.uint8), tcounts=np.array(tcounts), hist=f.hist)


def _load_features(path: Path, source: str) -> ClipFeatures:
    z = np.load(path)
    counts = z["counts"]; off = np.concatenate([[0], np.cumsum(counts)])
    feats = []
    for i in range(len(counts)):
        a, b = off[i], off[i + 1]
        d = z["desc"][a:b].astype(np.float32)
        feats.append((z["pts"][a:b], z["sz"][a:b], d if len(d) else None))
    tb = z["thumbs"].tobytes(); to = np.concatenate([[0], np.cumsum(z["tcounts"])])
    frames = [cv2.imdecode(np.frombuffer(tb[to[i]:to[i + 1]], np.uint8), cv2.IMREAD_COLOR) for i in range(len(counts))]
    ph, dh = zip(*[hashes(f) for f in frames])
    return ClipFeatures(source, z["times"].tolist(), frames, list(ph), list(dh), feats, z["hist"])


def _hex2hash(h):
    import imagehash
    return imagehash.hex_to_hash(h)


class Memory:
    def __init__(self, root: Path | str = DEFAULT_DIR):
        self.root = Path(root)
        (self.root / "features").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "index.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = [r[1] for r in self.db.execute("PRAGMA table_info(entries)")]
        if "origin" not in cols:   # migrate older index
            self.db.execute("ALTER TABLE entries ADD COLUMN origin TEXT")
        self.db.execute("UPDATE entries SET origin='seed:benchmark' WHERE origin IS NULL AND investigation_id LIKE 'bench:%'")
        self.db.execute("UPDATE entries SET origin='manual' WHERE origin IS NULL")
        self.db.commit()
        self._cache: dict[int, ClipFeatures] = {}

    # ---------------------------------------------------------------- write
    def add(self, feats: ClipFeatures, source_url: str | None, upload_date: str | None = None, event_context=None,
            investigation_id: str | None = None, uploader: str | None = None, title: str | None = None,
            role: str = "manual", clip_key: str | None = None, origin: str | None = None) -> int:
        """origin marks contamination risk: 'seed:benchmark' | 'discovered:<investigation id>' | 'manual'."""
        assert role in ROLES, role
        origin = origin or (f"discovered:{investigation_id}" if investigation_id else "manual")
        key = clip_key or source_url or f"local:{feats.path}"
        old = self.db.execute("SELECT id, feat_file FROM entries WHERE clip_key=?", (key,)).fetchone()
        if old:
            self.remove(old["id"])
        cur = self.db.execute(
            "INSERT INTO entries(clip_key,source_url,uploader,upload_date,title,event_context,investigation_id,role,origin,n_frames,hist,feat_file)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, source_url, uploader, upload_date, title, json.dumps(event_context or {}, ensure_ascii=False),
             investigation_id, role, origin, len(feats), feats.hist.astype(np.float32).tobytes(), ""))
        eid = cur.lastrowid
        ff = f"features/{eid}.npz"
        _save_features(self.root / ff, feats)
        self.db.execute("UPDATE entries SET feat_file=? WHERE id=?", (ff, eid))
        self.db.executemany("INSERT INTO frames VALUES(?,?,?,?,?)",
                            [(eid, i, feats.times[i], str(feats.ph[i]), str(feats.dh[i])) for i in range(len(feats))])
        self.db.commit()
        return eid

    def remove(self, eid: int):
        r = self.db.execute("SELECT feat_file FROM entries WHERE id=?", (eid,)).fetchone()
        if r and r["feat_file"]:
            (self.root / r["feat_file"]).unlink(missing_ok=True)
        self.db.execute("DELETE FROM frames WHERE entry_id=?", (eid,))
        self.db.execute("DELETE FROM entries WHERE id=?", (eid,))
        self.db.commit(); self._cache.pop(eid, None)

    # ----------------------------------------------------------------- read
    def entries(self):
        return [self._row(r) for r in self.db.execute("SELECT * FROM entries ORDER BY id")]

    @staticmethod
    def _row(r):
        d = {k: r[k] for k in r.keys() if k not in ("hist", "feat_file")}
        d["event_context"] = json.loads(d["event_context"] or "{}")
        return d

    def get(self, eid):
        r = self.db.execute("SELECT * FROM entries WHERE id=?", (eid,)).fetchone()
        return self._row(r) if r else None

    def features(self, eid) -> ClipFeatures:
        if eid not in self._cache:
            r = self.db.execute("SELECT feat_file, source_url FROM entries WHERE id=?", (eid,)).fetchone()
            self._cache[eid] = _load_features(self.root / r["feat_file"], r["source_url"] or "")
        return self._cache[eid]

    def stats(self):
        n = self.db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        nf = self.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        by = {r[0]: r[1] for r in self.db.execute("SELECT role, COUNT(*) FROM entries GROUP BY role")}
        size = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        return dict(entries=n, frames=nf, by_role=by, disk_mb=round(size / 1e6, 1), path=str(self.root))

    # -------------------------------------------------------------- lookup
    def shortlist(self, q: ClipFeatures, top_k=6, exclude=()):
        """Stage 1 only. Returns [(entry_id, hash_dist, colour_dist)] best first (cheap: no SIFT)."""
        qh = [(a, b) for a, b in zip(q.ph, q.dh)]
        qf = [hashes(cv2.flip(f, 1)) for f in q.frames]       # mirrored query hashes
        rows = {}
        for r in self.db.execute("SELECT entry_id, phash, dhash FROM frames"):
            if r["entry_id"] in exclude:
                continue
            rows.setdefault(r["entry_id"], []).append((_hex2hash(r["phash"]), _hex2hash(r["dhash"])))
        hist = {r["id"]: np.frombuffer(r["hist"], np.float32) for r in self.db.execute("SELECT id, hist FROM entries")}
        scored = []
        for eid, fr in rows.items():
            best = min((a - p) + (b - d) for (a, b) in qh + qf for (p, d) in fr)
            col = float(cv2.compareHist(q.hist.astype(np.float32), hist[eid], cv2.HISTCMP_BHATTACHARYYA))
            scored.append((eid, int(best), col))
        r_h = {e: i for i, (e, _, _) in enumerate(sorted(scored, key=lambda x: x[1]))}
        r_c = {e: i for i, (e, _, _) in enumerate(sorted(scored, key=lambda x: x[2]))}
        scored.sort(key=lambda x: min(r_h[x[0]], r_c[x[0]]) * 1000 + r_h[x[0]] + r_c[x[0]])  # best of either signal first
        return scored[:top_k]

    def lookup(self, q: ClipFeatures, top_k=6, exhaustive=False, exclude=(), coarse=12, evidence_dir=None, origin_prefix=None) -> LookupResult:
        t0 = time.time()
        if origin_prefix:   # e.g. 'seed:' or 'discovered:' -> search only that part of memory (evaluation uses this)
            exclude = set(exclude) | {r[0] for r in self.db.execute("SELECT id FROM entries WHERE origin NOT LIKE ?", (origin_prefix + "%",))}
        n = self.db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if exhaustive:
            ids = [r[0] for r in self.db.execute("SELECT id FROM entries") if r[0] not in exclude]
            mode = "exhaustive-match"
        else:
            ids = [e for e, _, _ in self.shortlist(q, top_k, exclude)]
            mode = f"hash-shortlist(top_{top_k})+match"
        hits = []
        for eid in ids:
            m = match(q, self.features(eid), coarse=coarse)
            if m.decision:
                ent = self.get(eid)
                if evidence_dir:
                    m.evidence(Path(evidence_dir) / f"memory_{eid}.png", f"memory hit #{eid} {ent['source_url']}")
                hits.append(MemoryHit(ent, m))
        hits.sort(key=lambda h: -h.match.score)
        return LookupResult(hits, n, ids, mode, round(time.time() - t0, 2))

    def lookup_path(self, path, **kw):
        return self.lookup(ClipFeatures.from_path(path), **kw)


# ------------------------------------------------------------------ seeding
def _benchmark_originals(cases_path: Path, avail_path: Path):
    """Group benchmark cases by original URL -> merged fact-check context."""
    avail = json.loads(avail_path.read_text()) if avail_path.exists() else {}
    out = {}
    for line in cases_path.read_text().splitlines():
        c = json.loads(line)
        for u in c["original_urls"]:
            e = out.setdefault(u, dict(url=u, cases=[], claimed_events=[], descs=[], factchecks=[], verdicts=[],
                                       fc_date=c["original_date"]))
            e["cases"].append(c["id"]); e["claimed_events"].append(c["claimed_event"]); e["descs"].append(c["original_desc"])
            e["factchecks"].append(c["factcheck_url"]); e["verdicts"].append(c["verdict"])
            a = avail.get(u)
            e["available"] = bool(a and a[0])
            if a and a[0]:
                parts = str(a[1]).split("|")
                if len(parts) >= 3 and len(parts[1]) == 8:
                    e["date"] = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}"; e["uploader"] = parts[2]
    return out


def seed_benchmark(mem: Memory, videos_dir: Path | None = None, download_missing=True, log=print):
    """Index the benchmark's fact-check-cited originals. Uses local files in data/videos when present, else yt-dlp
    (<=480p, <=120 s, deleted after indexing). Context is what the fact-checker said, stored verbatim."""
    from . import media
    videos_dir = videos_dir or ROOT / "data" / "videos"
    keymap = {u: k for k, u in json.loads((ROOT / "bench" / "local_keys.json").read_text()).items()}
    orig = _benchmark_originals(ROOT / "bench" / "cases.jsonl", ROOT / "bench" / "availability.json")
    added = skipped = 0
    for u, e in orig.items():
        if not e.get("available") and u not in keymap:
            log(f"skip (unavailable): {u}"); skipped += 1; continue
        local = None
        k = keymap.get(u)
        if k:
            g = [p for p in videos_dir.glob(f"{k}.*") if p.suffix != ".part"]
            local = g[0] if g else None
        clip = None
        try:
            if local is None:
                if not download_missing:
                    skipped += 1; continue
                clip = media.fetch(u); local = Path(clip.path)
            feats = ClipFeatures.from_path(local)
        except Exception as ex:
            log(f"skip (fetch/decode failed): {u} {str(ex)[:80]}"); skipped += 1; continue
        finally:
            if clip is not None:
                pass
        ctx = dict(source="benchmark fact-check", claimed_events=sorted(set(e["claimed_events"])), original_desc=e["descs"][0],
                   fact_check_urls=sorted(set(e["factchecks"])), fact_check_verdicts=sorted(set(e["verdicts"])),
                   note="original as cited by fact-checkers; not independently verified as the first upload")
        mem.add(feats, u, e.get("date") or e["fc_date"], ctx, "bench:" + ",".join(sorted(set(e["cases"]))),
                e.get("uploader"), None, "fact_check_cited_original", origin="seed:benchmark")
        added += 1; log(f"indexed {u} ({len(feats)} keyframes) cases={sorted(set(e['cases']))}")
        if clip is not None:
            media.cleanup(clip)
    return dict(added=added, skipped=skipped)
