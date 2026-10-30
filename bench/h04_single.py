"""Recompute the match stats (incl. spread) of the one single-frame-only proven match found in saved runs: H04 vs youtube 5eJJpmYmJQU."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodproof import media
from floodproof.matcher import ClipFeatures, match

a = ClipFeatures.from_path("/home/samir-kumal/hackethon/floodproof/data/heldout/H04.mp4")
clip = media.fetch("https://www.youtube.com/watch?v=5eJJpmYmJQU")
r = match(a, ClipFeatures.from_path(clip.path))
out = dict(consistent=r.consistent_frames, max_inliers=r.max_inliers, spread=[round(float(p[3]), 3) for p in r.pairs],
           pairs=[(p[0], p[1], p[2]) for p in r.pairs], decision=r.decision)
print(json.dumps(out))
json.dump(out, open(Path(__file__).parent / "h04_single.json", "w"))
