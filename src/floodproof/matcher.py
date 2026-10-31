"""Code-verified video matching (validated engine, packaged as a library).

Pipeline: keyframes (scene change + uniform) -> SIFT per keyframe -> ratio test + RANSAC homography on every
(frame_a, frame_b) pair, both orientations -> temporal-offset consensus.

Decision rule (unchanged from the validated prototype, 0/16 false matches on hard negatives):
    MATCH  <=>  >=2 temporally consistent frame matches  OR  one frame pair with >=60 RANSAC inliers.
Each accepted frame pair also needs >=15 inliers whose convex hull covers >=8% of BOTH frames (rejects logos).

Mirroring is tested by flipping clip A's features. Evidence PNGs draw the RANSAC inlier lines.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import imagehash
import numpy as np
from PIL import Image

W = 360  # working width
MIN_INL, MIN_SPREAD = 15, 0.08
STRONG_SINGLE, MIN_CONSISTENT = 60, 2
_SIFT_N = 600
cv2.setNumThreads(1)  # we parallelise over frame pairs ourselves


def read_frames(path, fps=3.0, maxdur=75, maxframes=240):
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(vfps / fps)))
    out, i = [], 0
    while True:
        if not cap.grab():
            break
        t = i / vfps
        if t > maxdur or len(out) >= maxframes:
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                h, w = f.shape[:2]
                out.append((t, cv2.resize(f, (W, max(2, int(h * W / w))), interpolation=cv2.INTER_AREA)))
        i += 1
    cap.release()
    return out


def hashes(f):
    im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return imagehash.phash(im), imagehash.dhash(im)


def select_keyframes(frames, maxk=48, mind=5):
    """Scene-change selection (dHash distance to last kept > mind), then uniform subsample to maxk."""
    kept, last = [], None
    for t, f in frames:
        _, d = hashes(f)
        if last is None or d - last > mind:
            kept.append((t, f))
            last = d
    if len(kept) > maxk:
        idx = np.linspace(0, len(kept) - 1, maxk).astype(int)
        kept = [kept[i] for i in idx]
    return kept


def colour_hist(frames):
    """Global HSV histogram averaged over frames (cheap, crop-tolerant-ish prefilter signal)."""
    acc = np.zeros(8 * 4 * 4, np.float32)
    for f in frames:
        h = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        acc += h.ravel()
    return acc / max(acc.sum(), 1)


def _sift():
    return cv2.SIFT_create(nfeatures=_SIFT_N)


def _feat(gray, sift, mask=None):
    kp, d = sift.detectAndCompute(gray, mask)
    pts = np.float32([k.pt for k in kp]).reshape(-1, 2)
    sz = np.float32([k.size for k in kp])
    return pts, sz, d


@dataclass
class ClipFeatures:
    """Keyframes + hashes + SIFT features of one clip. Flipped features are computed lazily (query side only)."""
    path: str
    times: list
    frames: list                 # BGR, width W
    ph: list                     # imagehash.ImageHash
    dh: list
    feats: list                  # per frame: (pts Nx2, size N, desc Nx128 float32 | None)
    hist: np.ndarray = None
    masks: list = field(default=None, repr=False)   # optional per-frame SIFT keep-masks (uint8, 0 = overlay); None = unmasked
    _flip: list = field(default=None, repr=False)

    @classmethod
    def from_path(cls, path, fps=3.0, maxdur=120, maxframes=360, maxk=48, masker=None):
        kf = select_keyframes(read_frames(path, fps=fps, maxdur=maxdur, maxframes=maxframes), maxk=maxk)
        if not kf:
            raise ValueError(f"no decodable frames in {path}")
        if masker is None and os.environ.get("FLOODPROOF_SAM") == "1":   # opt-in, default OFF
            from . import sam
            masker = sam.masker()
        return cls.from_frames(str(path), [t for t, _ in kf], [f for _, f in kf], masker=masker)

    @classmethod
    def from_frames(cls, path, times, frames, masker=None):
        """masker: optional callable frame -> uint8 keep-mask or None (see floodproof.sam). Frames themselves are never altered."""
        sift = _sift()
        masks = [masker(f) for f in frames] if masker else None
        ph, dh, feats = [], [], []
        for k, f in enumerate(frames):
            p, d = hashes(f)
            ph.append(p); dh.append(d)
            feats.append(_feat(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), sift, masks[k] if masks else None))
        return cls(path, list(times), list(frames), ph, dh, feats, colour_hist(frames), masks)

    @property
    def flipped(self):
        if self._flip is None:
            sift = _sift()
            ms = self.masks or [None] * len(self.frames)
            self._flip = [_feat(cv2.flip(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 1), sift, None if m is None else cv2.flip(m, 1))
                          for f, m in zip(self.frames, ms)]
        return self._flip

    def __len__(self):
        return len(self.frames)


def _hull_frac(pts, shape):
    if len(pts) < 3:
        return 0.0
    return cv2.contourArea(cv2.convexHull(pts.astype(np.float32))) / float(shape[0] * shape[1])


def _pair_match(fa, fb, shape_a, shape_b, want_pts=False):
    (pa_all, _, da), (pb_all, _, db) = fa, fb
    if da is None or db is None or len(pa_all) < 12 or len(pb_all) < 12:
        return 0, 0.0, None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 10:
        return 0, 0.0, None
    pa = np.float32([pa_all[g.queryIdx] for g in good]); pb = np.float32([pb_all[g.trainIdx] for g in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    if H is None:
        return 0, 0.0, None
    mask = mask.ravel().astype(bool); n = int(mask.sum())
    if n < 10:
        return 0, 0.0, None
    if not (0.1 < abs(np.linalg.det(H[:2, :2])) < 10):
        return 0, 0.0, None
    spread = min(_hull_frac(pa[mask], shape_a), _hull_frac(pb[mask], shape_b))
    return n, spread, ((pa, pb, mask) if want_pts else None)


@dataclass
class MatchResult:
    decision: bool
    consistent_frames: int
    max_inliers: int
    mirrored: bool
    temporal_offset: float | None   # seconds: time in B minus time in A
    n_pairs: int
    score: float
    pairs: list                     # best frame pairs: (ia, ib, inliers, spread, flip)
    a: ClipFeatures = field(default=None, repr=False)
    b: ClipFeatures = field(default=None, repr=False)
    stage: str = "full"
    consistent_pairs: list = field(default_factory=list, repr=False)   # EVERY pair of the consensus set (ia, ib, inliers, flip)

    def matched_frames(self, cap=60):
        """One entry per matched input frame (its best pair inside the consensus set): the per-frame data behind report.source_map."""
        best = {}
        for (i, j, n, f) in self.consistent_pairs:
            if i not in best or n > best[i][1]:
                best[i] = (j, n, f)
        return [dict(a_frame=i, a_t=round(self.a.times[i], 2) if self.a else None, b_frame=j,
                     b_t=round(self.b.times[j], 2) if self.b else None, inliers=n, flip=bool(f))
                for i, (j, n, f) in sorted(best.items())][:cap]

    @property
    def weak(self) -> bool:
        """A match that rests ONLY on the single-frame path (< MIN_CONSISTENT consistent frames, >= STRONG_SINGLE inliers).
        Shown, but never enough on its own for RECYCLED_FOOTAGE (see report.landmark_veto / proven_before)."""
        return bool(self.decision and self.consistent_frames < MIN_CONSISTENT)

    def to_dict(self):
        return dict(decision=self.decision, weak_match=self.weak, consistent_frames=self.consistent_frames, max_inliers=self.max_inliers,
                    mirrored=self.mirrored, temporal_offset=self.temporal_offset, n_pairs=self.n_pairs,
                    score=round(self.score, 2), stage=self.stage, matched_frames=self.matched_frames(),
                    best_pairs=[dict(a_frame=i, b_frame=j, a_t=self.a.times[i] if self.a else None,
                                     b_t=self.b.times[j] if self.b else None, inliers=n, flip=bool(f))
                                for (i, j, n, sp, f) in self.pairs])

    def evidence(self, out_png, title=""):
        return evidence_png(self, out_png, title)


def decide(consistent, max_inliers):
    return bool(consistent >= MIN_CONSISTENT or max_inliers >= STRONG_SINGLE)


def _score_pairs(A, B, a_idx, workers):
    jobs = [(i, j, flip) for i in a_idx for j in range(len(B)) for flip in (0, 1)]

    def run(job):
        i, j, flip = job
        fa = A.flipped[i] if flip else A.feats[i]
        n, sp, _ = _pair_match(fa, B.feats[j], A.frames[i].shape, B.frames[j].shape)
        return (i, j, n, sp, flip) if (n >= MIN_INL and sp >= MIN_SPREAD) else None

    _ = A.flipped  # compute once, before threads
    with ThreadPoolExecutor(workers) as ex:
        return [r for r in ex.map(run, jobs, chunksize=32) if r]


def _consensus(A, B, res):
    out = MatchResult(False, 0, 0, False, None, len(res), 0.0, [], A, B)
    if not res:
        return out
    best = []
    for flip in (0, 1):
        r = [x for x in res if x[4] == flip]
        for x in r:
            off = B.times[x[1]] - A.times[x[0]]
            s = [y for y in r if abs((B.times[y[1]] - A.times[y[0]]) - off) <= 1.0]
            if (len({y[0] for y in s}) > len({y[0] for y in best}) or
                    (len({y[0] for y in s}) == len({y[0] for y in best}) and sum(y[2] for y in s) > sum(y[2] for y in best))):
                best = s
    if not best:
        return out
    best.sort(key=lambda x: -x[2])
    nd = len({y[0] for y in best})
    out.consistent_frames, out.max_inliers, out.mirrored = nd, best[0][2], bool(best[0][4])
    out.temporal_offset = B.times[best[0][1]] - A.times[best[0][0]]
    out.pairs = best[:3]
    out.consistent_pairs = [(y[0], y[1], y[2], y[4]) for y in best]
    out.score = float(nd) + min(best[0][2], 100) / 100.0
    out.decision = decide(nd, best[0][2])
    return out


def match(a: ClipFeatures, b: ClipFeatures, coarse: int | None = None, workers: int | None = None) -> MatchResult:
    """Prove (or fail to prove) that clips a and b share footage.

    coarse=N: first compare only N evenly spaced frames of `a` against all of `b`; if no frame pair even reaches
    the per-frame bar, stop early (stage='coarse-reject'). This is a speed-up for screening many candidates; the
    decision for anything that passes is made by the full all-pairs run, with the unchanged rule.
    """
    workers = workers or min(8, os.cpu_count() or 4)
    if coarse and len(a) > coarse:
        idx = sorted(set(np.linspace(0, len(a) - 1, coarse).astype(int).tolist()))
        if not _score_pairs(a, b, idx, workers):
            r = MatchResult(False, 0, 0, False, None, 0, 0.0, [], a, b, stage="coarse-reject")
            return r
    return _consensus(a, b, _score_pairs(a, b, range(len(a)), workers))


def phash_distance(a: ClipFeatures, b: ClipFeatures) -> int:
    """Smallest pHash Hamming distance over all frame pairs (informational; a crop can defeat it)."""
    return int(min(x - y for x in a.ph for y in b.ph))


def evidence_png(r: MatchResult, out_png, title=""):
    """Side-by-side frames with the RANSAC inlier correspondences drawn as lines (up to 3 frame pairs)."""
    A, B = r.a, r.b
    rows = []
    for (i, j, n, sp, flip) in r.pairs[:3]:
        fa = A.frames[i]; fa_feat = A.feats[i]
        if flip:
            fa = cv2.flip(fa, 1); fa_feat = A.flipped[i]
        fb, fb_feat = B.frames[j], B.feats[j]
        (pa_all, sa, da), (pb_all, sb, db) = fa_feat, fb_feat
        m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        pa = np.float32([pa_all[g.queryIdx] for g in good]); pb = np.float32([pb_all[g.trainIdx] for g in good])
        H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
        mk = mask.ravel().tolist() if mask is not None else None
        ka = [cv2.KeyPoint(float(x), float(y), float(s)) for (x, y), s in zip(pa_all, sa)]
        kb = [cv2.KeyPoint(float(x), float(y), float(s)) for (x, y), s in zip(pb_all, sb)]
        img = cv2.drawMatches(fa, ka, fb, kb, good, None, matchColor=(0, 255, 0), singlePointColor=None,
                              matchesMask=mk, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.putText(img, f"A t={A.times[i]:.1f}s | B t={B.times[j]:.1f}s{' (A mirrored)' if flip else ''} | inliers={n}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        rows.append(img)
    if not rows:
        return None
    w = max(x.shape[1] for x in rows)
    rows = [cv2.copyMakeBorder(x, 0, 4, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in rows]
    img = np.vstack(rows)
    if title:
        band = np.full((28, img.shape[1], 3), 255, np.uint8)
        cv2.putText(band, title[:120], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        img = np.vstack([band, img])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(str(out_png), img)
    return str(out_png)


PROOF_INLIERS = 40


def proof_inliers(frame_a, frame_b, flip=False, top=PROOF_INLIERS):
    """RANSAC inliers of one frame pair as normalised 0-1 coordinates [[x1,y1,x2,y2],...], strongest SIFT response first.
    frame_a is the UNFLIPPED viral frame; when the match was found against its mirror, x1 is mapped back (1-x)."""
    sift = _sift()
    ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    if flip:
        ga = cv2.flip(ga, 1)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return []
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 10:
        return []
    pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    if H is None:
        return []
    idx = [k for k, ok in enumerate(mask.ravel()) if ok]
    idx.sort(key=lambda k: -ka[good[k].queryIdx].response)
    (ha, wa), (hb, wb) = ga.shape[:2], frame_b.shape[:2]
    out = []
    for k in idx[:top]:
        x1 = pa[k][0] / wa
        out.append([round(float(1 - x1 if flip else x1), 4), round(float(pa[k][1] / ha), 4),
                    round(float(pb[k][0] / wb), 4), round(float(pb[k][1] / hb), 4)])
    return [[min(1.0, max(0.0, v)) for v in row] for row in out]


def proof_artifacts(r: MatchResult, out_dir, tag):
    """For a PROVEN match: save the clean best-pair frames (no lines drawn) and the inlier coordinates.
    Returns {viral_frame, copy_frame, viral_t, copy_t, mirrored, inliers} (frame paths absolute), or None."""
    if not (r and r.decision and r.pairs and r.a and r.b):
        return None
    i, j, n, sp, flip = r.pairs[0]
    out_dir = os.path.abspath(out_dir); os.makedirs(out_dir, exist_ok=True)
    pv, pc = os.path.join(out_dir, f"proof_{tag}_viral.jpg"), os.path.join(out_dir, f"proof_{tag}_copy.jpg")
    cv2.imwrite(pv, r.a.frames[i], [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(pc, r.b.frames[j], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return dict(viral_frame=pv, copy_frame=pc, viral_t=round(float(r.a.times[i]), 2), copy_t=round(float(r.b.times[j]), 2),
                mirrored=bool(flip), inliers=proof_inliers(r.a.frames[i], r.b.frames[j], bool(flip)))


def make_variant(src, dst, vf, extra=()):
    """Synthetic recycled variant (crop / mirror / overlay / re-encode) via ffmpeg, for evaluation and tests."""
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-vf", vf, "-c:v", "libx264",
                    "-crf", "30", "-preset", "veryfast", *extra, str(dst)], check=True)


VARIANTS = {
    "crop70": "crop=iw*0.7:ih*0.7:iw*0.15:ih*0.2,scale=360:-2",
    "mirror": "hflip",
    "overlay": "drawbox=x=0:y=ih*0.75:w=iw:h=ih*0.2:color=black@0.9:t=fill,drawtext=text='BREAKING NEWS NEPAL FLOOD 2026':fontcolor=white:fontsize=18:x=10:y=h*0.8",
    "lowres": "scale=180:-2,scale=360:-2,eq=contrast=1.2:saturation=0.8",
    "combo": "hflip,crop=iw*0.8:ih*0.8:iw*0.1:ih*0.1,scale=300:-2,drawbox=x=0:y=0:w=iw:h=24:color=red@0.8:t=fill",
}
