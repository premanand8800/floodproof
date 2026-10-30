"""Muse Spark investigator: structured, frame-referenced clues from keyframes (+ narration audio if present).

Models (OpenRouter): meta/muse-spark-1.3 for visual reasoning (max_tokens >= 4000, it spends tokens thinking);
meta/muse-spark-open for audio (it accepts audio input). Every clue must cite a keyframe index or an audio timestamp;
clues without a valid reference are moved to `rejected_clues` and are NOT passed on as evidence.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import cv2
import numpy as np

from . import CACHE
from .llm import chat, parse_json
from .matcher import ClipFeatures
from .media import Clip, extract_audio

VISION_MODEL = "meta/muse-spark-1.3"
AUDIO_MODEL = "meta/muse-spark-open"
MAX_TOKENS = 5000

SCHEMA = """Return ONLY one JSON object (no prose) with exactly these keys:
{
 "shows": [{"text": str, "refs": [frame index ints]}],              // what the footage actually shows
 "scene_type": {"value": str, "refs": [..]},                        // e.g. "debris-covered glacier", "dense urban street", "coastal"
 "visible_text": [{"text": str, "language": str, "refs": [..]}],   // OCR of every legible caption, sign, logo, watermark
 "likely_location": {"place": str, "country": str, "confidence": float 0-1, "lat": float|null, "lon": float|null,
                     "reasons": [{"text": str, "refs": [..]}]},
 "contradictions_with_claim": [{"text": str, "refs": [..]}],       // visible cues that argue against the claim; [] if none
 "claim_parsed": {"place": str|null, "country_iso2": str|null, "event_type": one of "flood","flash_flood","glof","landslide","cyclone","storm_surge","unknown"},  // what the CLAIM names
 "queries": [{"q": str, "language": str, "why": str}]             // 8 search queries to find the earliest upload / true source
}
RULES: 'refs' are the frame numbers printed before each image (e.g. [3,7]) or audio timestamps as strings like "audio@12.5".
Every item MUST have at least one ref. Report only what is visible/audible; if unsure say so in confidence, do not invent.
Do NOT trust the claim: the footage may be recycled from an earlier event, possibly in another country. Include queries in the
likely local language(s) and neutral English ones that avoid the claimed event name."""


def _jpeg_part(img, q=70):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}}


def pick_frames(feats: ClipFeatures, n=8):
    return sorted(set(np.linspace(0, len(feats) - 1, min(n, len(feats))).round().astype(int).tolist()))


def analyze_audio(clip: Clip, workdir: Path | None = None) -> dict:
    """Narration analysis with Muse Spark Open. Returns dict with status; never raises."""
    if not clip.has_audio:
        return dict(status="no_audio_track")
    wd = workdir or (CACHE / "audio"); mp3 = extract_audio(clip.path, wd / (Path(clip.path).stem + ".mp3"))
    if not mp3:
        return dict(status="audio_extract_failed")
    prompt = ("Listen to this audio from a viral disaster video. Return ONLY JSON: {\"has_speech\": bool, \"language\": str|null, "
              "\"transcript_summary\": str, \"spoken_place_names\": [{\"name\": str, \"audio_t\": float}], \"other_sounds\": [str]}. "
              "If it is only music or noise, has_speech=false. Never invent words.")
    try:
        txt, cost = chat(AUDIO_MODEL, [{"type": "text", "text": prompt},
                                       {"type": "input_audio", "input_audio": {"data": base64.b64encode(mp3.read_bytes()).decode(), "format": "mp3"}}],
                         max_tokens=3000)
        d = parse_json(txt); d.update(status="ok", model=AUDIO_MODEL, cost_usd=round(cost, 5))
        return d
    except Exception as e:
        return dict(status="error", error=str(e)[:200], model=AUDIO_MODEL)


def _valid_refs(refs, nframes, audio_ok):
    good = []
    for r in refs or []:
        if isinstance(r, bool):
            continue
        if isinstance(r, (int, float)) and 0 <= int(r) < nframes:
            good.append(int(r))
        elif isinstance(r, str) and re.fullmatch(r"audio@\d+(\.\d+)?", r.strip()) and audio_ok:
            good.append(r.strip())
        elif isinstance(r, str) and r.strip().isdigit() and 0 <= int(r) < nframes:
            good.append(int(r))
    return good


def _clean(plan: dict, feats: ClipFeatures, audio_ok: bool):
    """Drop clue items that lack a valid frame/timestamp reference; attach timestamps to frame refs."""
    rejected = []
    n = len(feats)

    def fix(item, kind):
        if not isinstance(item, dict):
            rejected.append(dict(kind=kind, item=item, reason="not an object")); return None
        refs = _valid_refs(item.get("refs"), n, audio_ok)
        if not refs:
            rejected.append(dict(kind=kind, item=item, reason="no valid frame/timestamp reference")); return None
        item["refs"] = refs
        item["timestamps_s"] = [round(feats.times[r], 2) if isinstance(r, int) else float(r.split("@")[1]) for r in refs]
        return item

    out = {}
    out["shows"] = [x for x in (fix(i, "shows") for i in plan.get("shows", [])) if x]
    st = plan.get("scene_type")
    out["scene_type"] = fix(st, "scene_type") if isinstance(st, dict) else None
    out["visible_text"] = [x for x in (fix(i, "visible_text") for i in plan.get("visible_text", [])) if x]
    ll = plan.get("likely_location") or {}
    if isinstance(ll, dict):
        ll["reasons"] = [x for x in (fix(i, "location_reason") for i in ll.get("reasons", [])) if x]
        if not ll["reasons"]:           # a location guess with no cited evidence is not a clue
            rejected.append(dict(kind="likely_location", item=dict(ll), reason="no cited reasons with valid refs"))
            ll = dict(place=None, country=None, confidence=0.0, lat=None, lon=None, reasons=[], note="uncited guess rejected",
                      uncited_guess=dict(place=ll.get("place"), country=ll.get("country")))
    out["likely_location"] = ll
    out["contradictions_with_claim"] = [x for x in (fix(i, "contradiction") for i in plan.get("contradictions_with_claim", [])) if x]
    out["claim_parsed"] = plan.get("claim_parsed") or {}
    out["queries"] = [q for q in plan.get("queries", []) if isinstance(q, dict) and q.get("q")][:10]
    out["rejected_clues"] = rejected
    return out


def investigate_clip(clip: Clip, feats: ClipFeatures, claim: str | None = None, claim_date: str | None = None,
                     n_frames=8, use_audio=True, frames_dir: Path | None = None) -> dict:
    """Run Muse over keyframes (+ narration). Returns clue dict (see SCHEMA) plus cost/model/frames metadata."""
    idx = pick_frames(feats, n_frames)
    audio = analyze_audio(clip) if (use_audio and clip.has_audio) else dict(status="skipped" if clip.has_audio else "no_audio_track")
    audio_ok = audio.get("status") == "ok" and audio.get("has_speech") is not False
    ctx = f"It is being shared with this claim: '{claim}'" + (f" (claimed event date {claim_date})" if claim_date else "") if claim else \
        "No claim text was supplied."
    atext = ""
    if audio.get("status") == "ok":
        atext = ("\nAudio analysis (from a separate audio model, timestamps in seconds): " +
                 f"speech={audio.get('has_speech')}, language={audio.get('language')}, summary={audio.get('transcript_summary')}, "
                 f"spoken place names={audio.get('spoken_place_names')}. Cite as \"audio@<seconds>\".\n")
    content = [{"type": "text", "text": f"These are {len(idx)} keyframes (numbered) from a viral disaster video. {ctx}.{atext}\n" + SCHEMA}]
    for i in idx:
        content.append({"type": "text", "text": f"Frame {i} (t={feats.times[i]:.1f}s):"})
        content.append(_jpeg_part(feats.frames[i]))
    txt, cost = chat(VISION_MODEL, content, max_tokens=MAX_TOKENS)
    try:
        plan = parse_json(txt)
    except Exception as e:
        return dict(status="error", error=f"unparseable reply: {e}", raw=txt[:500], model=VISION_MODEL, cost_usd=round(cost, 5),
                    frames_shown=idx, audio=audio)
    out = _clean(plan, feats, audio_ok)
    out.update(status="ok", model=VISION_MODEL, cost_usd=round(cost + audio.get("cost_usd", 0), 5), frames_shown=idx,
               frame_times_s={str(i): round(feats.times[i], 2) for i in idx}, audio=audio)
    if frames_dir:
        from .media import save_keyframe_jpgs
        out["frame_images"] = save_keyframe_jpgs(feats, frames_dir, idx)
    return out
