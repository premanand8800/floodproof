"""Optional SAM 3.1 overlay masking (OFF unless FLOODPROOF_SAM=1).

SAM 3.1 (Meta Model API, POST https://api.meta.ai/v1/responses, model "sam-3.1", text prompt + input_image) finds overlay regions
(caption text, logo, watermark, banner, ticker). The matcher then drops SIFT keypoints inside them (mask argument of detectAndCompute).
Only boxes are used: the line grammar is `<Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;...|>`. Proof images are never altered.
Masks are cached per frame (sha1 of the frame bytes). Any API failure means "no mask" (matching proceeds unmasked).
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import time

import cv2
import numpy as np

URL = "https://api.meta.ai/v1/responses"
PROMPTS = ("caption text", "logo", "watermark", "text banner", "news ticker")
MAX_COVER = 0.6   # if overlay boxes cover more than this fraction of the frame, ignore them (SAM misfire; keep the frame matchable)
_BOX = re.compile(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)")
_cache: dict = {}
stats = dict(calls=0, errors=0, regions=0, frames=0, seconds=0.0)


def enabled() -> bool:
    return os.environ.get("FLOODPROOF_SAM") == "1"


def parse_boxes(text: str):
    """Boxes (x1,y1,x2,y2,w,h) from a SAM output stream (all frames/ids)."""
    return [tuple(int(v) for v in m.groups()) for m in _BOX.finditer(text or "")]


def _call(jpeg: bytes, prompt: str) -> str:
    import requests
    body = {"model": "sam-3.1", "stream": False, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}]}]}
    r = requests.post(URL, json=body, headers={"Authorization": "Bearer " + os.environ["META_API_KEY"]}, timeout=60)
    r.raise_for_status()
    out = []
    for item in r.json().get("output", []):
        for c in item.get("content", []) or []:
            out.append(c.get("text", ""))
    return "".join(out)


def overlay_boxes(frame, call=_call, prompts=PROMPTS):
    """Overlay boxes (in frame pixels) for one BGR frame, cached by content."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    jpeg = buf.tobytes()
    key = hashlib.sha1(jpeg).hexdigest()
    if key in _cache:
        return _cache[key]
    t = time.time()
    h, w = frame.shape[:2]
    boxes = []
    for p in prompts:
        try:
            for (x1, y1, x2, y2, bw, bh) in parse_boxes(call(jpeg, p)):
                sx, sy = w / max(bw, 1), h / max(bh, 1)
                boxes.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)))
            stats["calls"] += 1
        except Exception:
            stats["errors"] += 1
    stats["seconds"] += time.time() - t
    _cache[key] = boxes
    return boxes


def keep_mask(frame, call=_call, pad=4):
    """uint8 mask for SIFT (255 = keep, 0 = overlay), or None when nothing was found / too much was found."""
    boxes = overlay_boxes(frame, call)
    h, w = frame.shape[:2]
    if not boxes:
        return None
    m = np.full((h, w), 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = 0
    if (m == 0).mean() > MAX_COVER:
        return None
    stats["frames"] += 1
    stats["regions"] += len(boxes)
    return m


def masker():
    """Frame -> mask callable for ClipFeatures, or None when the flag is off."""
    return keep_mask if enabled() else None
