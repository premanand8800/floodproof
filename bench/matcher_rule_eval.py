"""Collect (consistent_frames, max_inliers, best-pair spread) for variants / hard negatives / known true pairs / H04, so match
rules can be compared offline.  Output: bench/matcher_rule_data.json.  Run from the MAIN checkout data dir:
    FP_DATA=/home/samir-kumal/hackethon/floodproof/data python bench/matcher_rule_eval.py
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodproof import CACHE
from floodproof.matcher import ClipFeatures, VARIANTS, make_variant, match

DATA = Path(os.environ.get("FP_DATA", Path(__file__).resolve().parents[1] / "data"))
VID = DATA / "videos"
B = Path(__file__).parent
LOO = ["ig_leigh", "ig_sainj", "yt_sun_italy", "yt_ie_nalagarh", "yt_dawn", "yt_tauktae1", "yt_ramban", "yt_japan2015",
       "ig_pak2022", "yt_sikkim_austria"]
NEG = [("ig_leigh", "yt_ie_nalagarh"), ("ig_sainj", "ig_pak2022"), ("ig_viral_italy", "ig_dharali_accu"), ("yt_ie_nalagarh", "yt_sun_italy"),
       ("x_iambmane", "yt_croc"), ("yts_nepal_vanTV", "yts_viral_nalagarh"), ("yt_dawn", "yt_tauktae1"), ("yt_hum", "yts_sainj"),
       ("ig_wayanad_el", "yt_tribune_nalagarh"), ("yt_leigh", "ig_viral_italy"), ("ig_pak2022", "yt_sun_italy"),
       ("yts_ajesh_buddha", "ig_dharali_accu"), ("x_dmuppavarapu", "yt_dawn"), ("yt_tauktae2", "yt_wayanad_el"),
       ("ig_dharali_accu", "yts_sainj"), ("yt_hum", "yt_ie_nalagarh")]
# same footage uploaded twice (confirmed by eye in earlier work): not negatives
SIBLINGS = [{"ig_sainj", "yts_sainj"}, {"yt_tauktae1", "yt_tauktae2"}, {"ig_wayanad_el", "yt_wayanad_el"}, {"yt_ie_nalagarh", "yt_indexp_punjab"},
            {"yt_tribune_nalagarh", "yt_indexp_punjab"}, {"ig_leigh", "yt_leigh"}, {"ig_sainj", "yt_sainj"}]


def vp(k):
    g = [p for p in VID.glob(f"{k}.*") if p.suffix != ".part"]
    return g[0] if g else None


_c = {}


def feats(path):
    path = str(path)
    if path not in _c:
        _c[path] = ClipFeatures.from_path(path)
    return _c[path]


def row(kind, a, b, fa, fb):
    t = time.time()
    r = match(fa, fb)
    sp = r.pairs[0][3] if r.pairs else 0.0
    out = dict(kind=kind, a=a, b=b, consistent=r.consistent_frames, max_inliers=r.max_inliers, spread=round(float(sp), 3), old_rule=r.decision,
               sec=round(time.time() - t, 1))
    print(json.dumps(out), flush=True)
    return out


rows = []
vd = CACHE / "var"; vd.mkdir(exist_ok=True)
for k in LOO:
    if not vp(k):
        continue
    for n in ("crop70", "mirror", "combo"):
        dst = vd / f"{k}_{n}.mp4"
        if not dst.exists():
            make_variant(vp(k), dst, VARIANTS[n])
        rows.append(row("variant", k, n, feats(vp(k)), feats(dst)))
done = set()
for a, b in NEG + list(itertools.combinations([k for k in LOO if vp(k)], 2)):
    key = frozenset((a, b))
    if key in done or any(key == frozenset(s) for s in SIBLINGS) or not vp(a) or not vp(b):
        continue
    done.add(key)
    rows.append(row("hard_negative", a, b, feats(vp(a)), feats(vp(b))))
json.dump(rows, open(B / "matcher_rule_data.json", "w"), indent=1)
