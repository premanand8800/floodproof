"""Code-verified video matching: keyframes -> pHash/dHash prefilter + SIFT/RANSAC homography -> temporal offset consensus."""
import cv2, numpy as np, imagehash, subprocess, os, json
from PIL import Image

W = 360  # working width

def read_frames(path, fps=3.0, maxdur=75, maxframes=240):
    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(vfps / fps)))
    out, i = [], 0
    while True:
        ok = cap.grab()
        if not ok: break
        t = i / vfps
        if t > maxdur or len(out) >= maxframes: break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                h, w = f.shape[:2]
                f = cv2.resize(f, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
                out.append((t, f))
        i += 1
    cap.release()
    return out

def hashes(f):
    im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return imagehash.phash(im), imagehash.dhash(im)

def keyframes(frames, maxk=48, mind=5):
    """scene-change style selection: keep frame when dHash differs from last kept; then uniform subsample."""
    kept, last = [], None
    for t, f in frames:
        p, d = hashes(f)
        if last is None or d - last > mind:
            kept.append((t, f)); last = d
    if len(kept) > maxk:
        idx = np.linspace(0, len(kept) - 1, maxk).astype(int)
        kept = [kept[i] for i in idx]
    return kept

class Prep:
    def __init__(self, path, **kw):
        self.path = path
        self.kf = keyframes(read_frames(path, **kw))
        self.sift = cv2.SIFT_create(nfeatures=600)
        self.ph, self.dh, self.feat, self.featflip = [], [], [], []
        for t, f in self.kf:
            p, d = hashes(f); self.ph.append(p); self.dh.append(d)
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            self.feat.append(self.sift.detectAndCompute(g, None))
            self.featflip.append(self.sift.detectAndCompute(cv2.flip(g, 1), None))

bf = cv2.BFMatcher(cv2.NORM_L2)

def hull_frac(pts, shape):
    if len(pts) < 3: return 0.0
    h = cv2.convexHull(pts.astype(np.float32))
    return cv2.contourArea(h) / float(shape[0] * shape[1])

def pair_match(fa, fb, shape_a, shape_b):
    ka, da = fa; kb, db = fb
    if da is None or db is None or len(ka) < 12 or len(kb) < 12: return 0, 0.0, None
    m = bf.knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 10: return 0, 0.0, None
    pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    if H is None: return 0, 0.0, None
    mask = mask.ravel().astype(bool); n = int(mask.sum())
    if n < 10: return 0, 0.0, None
    # sanity: non-degenerate transform
    det = np.linalg.det(H[:2, :2])
    if not (0.1 < abs(det) < 10): return 0, 0.0, None
    # spatial spread of inliers in BOTH frames (rejects tiny logos / watermark-only matches)
    spread = min(hull_frac(pa[mask], shape_a), hull_frac(pb[mask], shape_b))
    return n, spread, (pa[mask], pb[mask], good, mask)

def compare(A, B, min_inl=15, min_spread=0.08):
    """returns dict with score and matched frame pairs. A,B: Prep"""
    res = []  # (ia, ib, inl, spread, flip)
    for i, (ta, fa) in enumerate(A.kf):
        for j, (tb, fb) in enumerate(B.kf):
            best = (0, 0, 0)
            for flip, fbfeat in ((0, B.feat[j]), (1, B.featflip[j])):
                n, sp, _ = pair_match(A.feat[i], fbfeat, fa.shape, fb.shape)
                if n > best[0]: best = (n, sp, flip)
            if best[0] >= min_inl and best[1] >= min_spread:
                res.append((i, j, best[0], best[1], best[2]))
    # temporal consensus: offset tb - ta; allow clip to be sped/trimmed -> use offset histogram (1.0s bins) per flip
    out = dict(n_pairs=len(res), consistent=0, max_inliers=0, offset=None, flip=None, pairs=[], score=0.0)
    if not res: return out
    best_set = []
    for flip in (0, 1):
        r = [x for x in res if x[4] == flip]
        for x in r:
            off = B.kf[x[1]][0] - A.kf[x[0]][0]
            sset = [y for y in r if abs((B.kf[y[1]][0] - A.kf[y[0]][0]) - off) <= 1.0]
            # distinct frames of A only
            if len({y[0] for y in sset}) > len({y[0] for y in best_set}) or (
               len({y[0] for y in sset}) == len({y[0] for y in best_set}) and sum(y[2] for y in sset) > sum(y[2] for y in best_set)):
                best_set = sset
    if not best_set: return out
    best_set.sort(key=lambda x: -x[2])
    nd = len({y[0] for y in best_set})
    out.update(consistent=nd, max_inliers=best_set[0][2], flip=best_set[0][4],
               offset=B.kf[best_set[0][1]][0] - A.kf[best_set[0][0]][0], pairs=best_set[:3])
    # score: distinct temporally consistent matched frames, weighted by inlier strength
    out["score"] = float(nd) + min(best_set[0][2], 100) / 100.0
    return out

def phash_best(A, B):
    d = min(min(a - b, a - imagehash.phash(Image.fromarray(cv2.cvtColor(cv2.flip(fb, 1), cv2.COLOR_BGR2RGB)))) for a in A.ph for b, (t, fb) in zip(B.ph, B.kf))
    return int(d)

def verdict(r):
    # code-judged: >=2 temporally consistent frame matches (or 1 very strong one) with spatially spread RANSAC inliers
    return bool(r["consistent"] >= 2 or r["max_inliers"] >= 60)

def evidence_png(A, B, r, out_png, title=""):
    rows = []
    for (i, j, n, sp, flip) in r["pairs"][:3]:
        fa = A.kf[i][1]; fb = B.kf[j][1]
        if flip: fb = cv2.flip(fb, 1)
        ga = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY); gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
        ka, da = A.feat[i]
        kb, db = (B.feat[j] if not flip else B.featflip[j])
        m = bf.knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
        H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
        mk = mask.ravel().tolist() if mask is not None else None
        img = cv2.drawMatches(fa, ka, fb, kb, good, None, matchColor=(0, 255, 0), singlePointColor=None,
                              matchesMask=mk, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.putText(img, f"A t={A.kf[i][0]:.1f}s | B t={B.kf[j][0]:.1f}s{' (mirrored)' if flip else ''} | inliers={n}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        rows.append(img)
    if not rows: return None
    w = max(r_.shape[1] for r_ in rows)
    rows = [cv2.copyMakeBorder(r_, 0, 4, 0, w - r_.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r_ in rows]
    img = np.vstack(rows)
    if title:
        band = np.full((28, img.shape[1], 3), 255, np.uint8)
        cv2.putText(band, title[:120], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        img = np.vstack([band, img])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cv2.imwrite(out_png, img)
    return out_png

# ---- synthetic recycled-variant generator (crop / mirror / overlay / re-encode) ----
def make_variant(src, dst, vf, extra=()):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-an", "-vf", vf, "-c:v", "libx264", "-crf", "30", "-preset", "veryfast", *extra, dst], check=True)
