"""Muse Spark explains WHAT is shared between a matcher-proven pair of frames (viral = A, earlier copy = B).

The proof is the geometric SIFT/RANSAC match. This text only points a human at the shared visual elements; it is
labelled as an AI explanation, describes only what is visible, and is optional (a failure never affects the report).
"""
from __future__ import annotations

import base64
from pathlib import Path

from .llm import chat, parse_json

MODEL = "meta/muse-spark-1.3"
LABEL = "AI explanation of a matcher-proven match: the proof is the geometric match, not this text"
REGIONS = ("top-left", "top-center", "top-right", "middle-left", "center", "middle-right", "bottom-left", "bottom-center", "bottom-right")

PROMPT = f"""Two video frames follow. Image A is a frame from a viral video. Image B is a frame from an EARLIER copy that a
computer-vision matcher has ALREADY PROVEN to share footage with A. Your job is only to point out, for a human reader, the
specific visible elements that make A and B the same footage.

Rules:
- Describe ONLY what you can actually see in the images. Do not guess locations, dates, events or who filmed it.
- Give concrete, specific elements (e.g. "the same red-roofed building", "the same broken railing", "identical river bend shape"),
  3 to 6 items.
- For each item give a rough region in A and in B, chosen from: {", ".join(REGIONS)}.
- List differences between A and B too (crop, mirror, caption bar, watermark, colour or brightness, resolution). Use [] if none.
Reply with ONE JSON object and nothing else:
{{"items":[{{"what":"...","region_a":"top-left","region_b":"top-left"}}],"differences":["..."]}}"""


def _part(path):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()}}


def clean_explanation(d: dict, model=MODEL, cost=0.0):
    """Normalise a model reply into {items, differences, model, label, cost_usd}; None if nothing usable."""
    items = []
    for it in (d.get("items") or [])[:8]:
        if isinstance(it, dict) and str(it.get("what") or "").strip():
            items.append(dict(what=str(it["what"]).strip()[:240], region_a=str(it.get("region_a") or "")[:30],
                              region_b=str(it.get("region_b") or "")[:30]))
    if not items:
        return None
    diffs = [str(x).strip()[:200] for x in (d.get("differences") or []) if str(x).strip()][:8]
    return dict(items=items, differences=diffs, model=model, label=LABEL, cost_usd=round(float(cost), 5))


def explain_proof(viral_path, copy_path, chat_fn=chat):
    """One Muse Spark call on the two CLEAN frames. Returns the explanation dict, or None on any failure."""
    try:
        txt, cost = chat_fn(MODEL, [{"type": "text", "text": PROMPT}, {"type": "text", "text": "Image A (viral video):"}, _part(viral_path),
                                    {"type": "text", "text": "Image B (earlier copy):"}, _part(copy_path)], max_tokens=4000)
        return clean_explanation(parse_json(txt), cost=cost)
    except Exception:
        return None
