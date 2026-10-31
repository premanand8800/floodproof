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
from .llm import chat, parse_json, last_provider, meta_enabled, transcribe_meta, TRANSCRIBE_MODEL
from .matcher import ClipFeatures
from .media import Clip, extract_audio, extract_wav

VISION_MODEL = "meta/muse-spark-1.3"
AUDIO_MODEL = "meta/muse-spark-open"
EXTRACT_MODEL = "meta/muse-spark-open"       # text extraction from the transcript: Baseten first, OpenRouter fallback
MAX_TOKENS = 5000

SCHEMA = """Return ONLY one JSON object (no prose) with exactly these keys:
{
 "shows": [{"text": str, "refs": [frame index ints]}],              // what the footage actually shows
 "scene_type": {"value": str, "refs": [..]},                        // e.g. "debris-covered glacier", "dense urban street", "coastal"
 "visible_text": [{"text": str, "language": str, "is_place": bool (true only if the text is a geographic place name), "refs": [..]}],   // OCR of every legible caption, sign, logo, watermark
 "likely_location": {"place": str, "country": str, "confidence": float 0-1, "lat": float|null, "lon": float|null,
                     "reasons": [{"text": str, "refs": [..]}]},
 "contradictions_with_claim": [{"text": str, "refs": [..]}],       // visible cues that argue against the claim; [] if none
 "claim_parsed": {"place": str|null, "country_iso2": str|null, "event_type": one of "flood","flash_flood","glof","landslide","cyclone","storm_surge","tsunami","earthquake","wildfire","unknown"},  // what the CLAIM names
 "media_nature": {"value": one of "camera_footage","news_compilation","animation_or_cgi","likely_ai_generated","screen_recording","unknown",
                  "confidence": float 0-1, "refs": [..], "reasons": [{"text": str, "refs": [..]}]},   // what KIND of media this is, judged from the frames
 "frame_labels": [{"frame": int, "label": one of "camera_footage","likely_ai_generated","animation_or_cgi","graphic_or_text","screen_recording","unknown",
                   "confidence": float 0-1, "reason": str}],       // ONE entry for EVERY frame shown, judged on that frame alone
 "on_screen_dates": [{"date": "YYYY-MM-DD", "text": str, "refs": [..]}],   // dates legible on title cards/captions/timestamps (full date with year only); [] if none
 "queries": [{"q": str, "language": str, "why": str}]             // 8 search queries to find the earliest upload / true source
}
media_nature: camera_footage = raw/phone/CCTV/drone footage; news_compilation = an edited news report/montage that contains third-party
clips, with presenter/graphics/logos/captions; animation_or_cgi = cartoon, animation, 3D render or reenactment; likely_ai_generated =
photoreal but synthetic-looking (warping, impossible physics); screen_recording = recording of a screen/app. Cite the frames that show it
(for camera_footage cite a few representative frames). This is an observation from the frames, not proof; use "unknown" if unsure.
frame_labels: a video can MIX real camera clips with AI-generated shots, so label each frame separately. Tells of AI generation to look for:
overly smooth or plastic textures, impossible physics (water, ice or debris moving wrongly), morphing or melting details, garbled or
nonsense text, too-perfect cinematic lighting, and the generic staged-analyst / stock-footage look. Real camera footage shows
natural noise, shake, compression artefacts and incidental detail. graphic_or_text = title card, infographic, map or chart overlay.
If a frame is uncertain use "unknown" with low confidence. These are observations, never certainty; do not claim proof.
RULES: 'refs' are the frame numbers printed before each image (e.g. [3,7]) or audio timestamps as strings like "audio@12.5".
Every item MUST have at least one ref. Report only what is visible/audible; if unsure say so in confidence, do not invent.
Do NOT trust the claim: the footage may be recycled from an earlier event, possibly in another country. Include queries in the
likely local language(s) and neutral English ones that avoid the claimed event name."""


def _jpeg_part(img, q=70):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}}


def pick_frames(feats: ClipFeatures, n=8):
    return sorted(set(np.linspace(0, len(feats) - 1, min(n, len(feats))).round().astype(int).tolist()))


AUDIO_MAX_S = 90


def _lenient_json(txt):
    """parse_json, else strip code fences / take the first balanced {...}; None if nothing parses."""
    import json
    t = re.sub(r"```(?:json)?", "", txt or "")
    try:
        return parse_json(t)
    except Exception:
        pass
    i = t.find("{")
    while i != -1:
        depth = 0
        for j in range(i, len(t)):
            depth += (t[j] == "{") - (t[j] == "}")
            if depth == 0:
                try:
                    return json.loads(t[i:j + 1])
                except Exception:
                    break
        i = t.find("{", i + 1)
    return None


def _clean_audio(d: dict) -> dict:
    """Keep only items with a numeric audio_t (never invent); normalise spoken dates to ISO."""
    def t_of(x):
        v = x.get("audio_t") if isinstance(x, dict) else None
        if isinstance(v, str):
            m = re.fullmatch(r"\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*s?\s*", v)
            v = (int(m[1] or 0) * 60 + float(m[2])) if m else None
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else None
    places = [dict(name=str(x["name"]).strip(), audio_t=t_of(x)) for x in (d.get("spoken_place_names") or [])
              if isinstance(x, dict) and str(x.get("name") or "").strip() and t_of(x) is not None]
    dates = []
    for x in (d.get("spoken_dates") or []):
        iso = _iso_ok(x.get("date")) if isinstance(x, dict) else None
        if iso and t_of(x) is not None:
            dates.append(dict(date=iso, text=str(x.get("text") or "")[:120], audio_t=t_of(x)))
    partial = []
    for x in (d.get("partial_dates") or []):
        if not isinstance(x, dict) or t_of(x) is None:
            continue
        try:
            mo, dy = int(x.get("month")), int(x.get("day"))
        except (TypeError, ValueError):
            continue
        if 1 <= mo <= 12 and 1 <= dy <= 31:
            partial.append(dict(month=mo, day=dy, text=str(x.get("text") or "")[:120], audio_t=t_of(x)))
    d["dropped_untimed"] = len(d.get("spoken_place_names") or []) - len(places) + len(d.get("spoken_dates") or []) - len(dates)
    d["spoken_place_names"], d["spoken_dates"], d["partial_dates"] = places, dates, partial
    return d


def analyze_audio_transcript(clip: Clip, workdir: Path | None = None, transcribe=transcribe_meta, chat_fn=chat) -> dict | None:
    """Primary narration path: Muse Voice Transcribe (turn-level timestamps) -> Muse Spark extracts places/dates, each citing the timestamp
    of the turn it was said in. Returns the audio dict, or None on any failure/empty speech so the caller can fall back to Spark Open audio."""
    try:
        wd = workdir or (CACHE / "audio")
        wav = extract_wav(clip.path, wd / (Path(clip.path).stem + ".wav"), max_seconds=AUDIO_MAX_S)
        if not wav:
            return None
        tr = transcribe(wav)
        turns = tr.get("turns") or []
        if not turns:                       # plain text only: no timestamps to cite, so use the fallback path
            return None
        dur = getattr(clip, "duration", None)
        truncated = bool(dur and dur > AUDIO_MAX_S)
        lines = "\n".join(f"[{t['start_s']:.1f}s] {t['text']}" for t in turns)
        prompt = ("Below is a timestamped transcript (turn start seconds in brackets) of narration from a viral disaster video. Return ONLY JSON: "
                  "{\"language\": str|null, \"spoken_place_names\": [{\"name\": str, \"audio_t\": float}], "
                  "\"spoken_dates\": [{\"date\": \"YYYY-MM-DD\", \"text\": str, \"audio_t\": float}], "
                  "\"partial_dates\": [{\"month\": int 1-12, \"day\": int, \"text\": str, \"audio_t\": float}]}. "
                  "spoken_dates: only full calendar dates with a year that the speaker says. partial_dates: day and month but NO year. "
                  "Place names include cities, regions, countries and borders. audio_t MUST be the bracketed start time of the turn where it is said. "
                  "Only use words present in the transcript; never invent.\n\nTRANSCRIPT:\n" + lines[:6000])
        raw, cost = chat_fn(EXTRACT_MODEL, [{"type": "text", "text": prompt}], max_tokens=4000, **({"baseten": True} if chat_fn is chat else {}))
        prov = last_provider() if chat_fn is chat else "injected"
        d = _lenient_json(raw)
        if not isinstance(d, dict):
            return None
        starts = [t["start_s"] for t in turns]
        d = _clean_audio(d)
        for k in ("spoken_place_names", "spoken_dates", "partial_dates"):      # snap to the real turn start at or before the cited time
            for x in d[k]:
                x["audio_t"] = max([s for s in starts if s <= x["audio_t"] + 0.5] or [starts[0]])
        d.update(status="ok", has_speech=True, transcript_summary=" ".join(t["text"] for t in turns)[:600], other_sounds=[],
                 model=EXTRACT_MODEL, provider=prov, transcriber=TRANSCRIBE_MODEL, transcriber_provider="meta_api",
                 timestamps="turn-level", turns=turns[:60], cost_usd=round(cost, 5), truncated=truncated,
                 analyzed_seconds=AUDIO_MAX_S if truncated else dur, attempts=1)
        return d
    except Exception:
        return None


def analyze_audio(clip: Clip, workdir: Path | None = None) -> dict:
    """Narration analysis. Primary: Muse Voice Transcribe + Muse Spark extraction (timestamps from transcript turns).
    Fallback: Muse Spark Open listens to the audio directly. Returns dict with status; never raises.
    Sends at most the first AUDIO_MAX_S seconds; retries once with a stricter prompt if no JSON came back."""
    if not clip.has_audio:
        return dict(status="no_audio_track")
    res = analyze_audio_transcript(clip, workdir) if meta_enabled() else None
    if res:
        return res
    out = _analyze_audio_spark_open(clip, workdir)
    if out.get("status") == "ok":
        out.update(provider="openrouter", transcriber=None)
    return out


def _analyze_audio_spark_open(clip: Clip, workdir: Path | None = None) -> dict:
    wd = workdir or (CACHE / "audio"); mp3 = extract_audio(clip.path, wd / (Path(clip.path).stem + ".mp3"), max_seconds=AUDIO_MAX_S)
    if not mp3:
        return dict(status="audio_extract_failed")
    dur = getattr(clip, "duration", None)
    truncated = bool(dur and dur > AUDIO_MAX_S)
    prompt = ("Listen to this audio from a viral disaster video. Return ONLY JSON: {\"has_speech\": bool, \"language\": str|null, "
              "\"transcript_summary\": str, \"spoken_place_names\": [{\"name\": str, \"audio_t\": float}], "
              "\"spoken_dates\": [{\"date\": \"YYYY-MM-DD\", \"text\": str, \"audio_t\": float}], "
              "\"partial_dates\": [{\"month\": int 1-12, \"day\": int, \"text\": str, \"audio_t\": float}], \"other_sounds\": [str]}. "
              "spoken_dates: only full calendar dates (with year) that the speaker says. partial_dates: dates spoken with a day and month but NO year "
              "(e.g. 'August 26'). Place names include countries and borders (e.g. 'Nepal', 'Nepal-China border'). audio_t = seconds from the start of the audio. "
              "transcript_summary must mention every date and place the speaker names. If it is only music or noise, has_speech=false. Never invent words.")
    strict = ("Reply with ONE JSON object and nothing else: no prose, no markdown, no thinking. Keep transcript_summary under 60 words. "
              "Keys: has_speech, language, transcript_summary, spoken_place_names [{name, audio_t}], spoken_dates [{date, text, audio_t}], partial_dates [{month, day, text, audio_t}], other_sounds. "
              "Every place and date MUST have audio_t, a number of seconds from the start; omit any you cannot time. ")
    data = base64.b64encode(mp3.read_bytes()).decode()
    audio_part = {"type": "input_audio", "input_audio": {"data": data, "format": "mp3"}}
    cost, raw, err, best = 0.0, "", None, None
    for attempt, (p, mt) in enumerate(((prompt, 6000), (strict + prompt, 8000))):
        try:
            raw, c = chat(AUDIO_MODEL, [{"type": "text", "text": p}, audio_part], max_tokens=mt)
            cost += c
        except Exception as e:
            err = str(e)[:200]; continue
        d = _lenient_json(raw)
        if isinstance(d, dict):
            d = _clean_audio(d)
            if attempt == 0 and d["dropped_untimed"] and not (d["spoken_place_names"] or d["spoken_dates"]):
                best = d; err = "items had no audio_t"; continue      # retry once asking explicitly for timestamps
            d.update(status="ok", model=AUDIO_MODEL, cost_usd=round(cost, 5), truncated=truncated, analyzed_seconds=AUDIO_MAX_S if truncated else dur,
                     attempts=attempt + 1)
            return d
        err = "no JSON object in model reply" + (" (empty reply)" if not (raw or "").strip() else "")
    if best:
        best.update(status="ok", model=AUDIO_MODEL, cost_usd=round(cost, 5), truncated=truncated, analyzed_seconds=AUDIO_MAX_S if truncated else dur, attempts=2)
        return best
    return dict(status="error", error=err, model=AUDIO_MODEL, cost_usd=round(cost, 5), raw_reply=(raw or "")[:300], attempts=2, truncated=truncated)


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


NATURES = ("camera_footage", "news_compilation", "animation_or_cgi", "likely_ai_generated", "screen_recording", "mixed", "unknown")
FRAME_LABELS = ("camera_footage", "likely_ai_generated", "animation_or_cgi", "graphic_or_text", "screen_recording", "unknown")
CLASS_NAMES = {"camera_footage": "camera footage", "likely_ai_generated": "likely AI-generated shots", "animation_or_cgi": "animation/CGI shots",
               "screen_recording": "screen recording"}


def _clean_frame_labels(items, feats, rejected):
    """Per-frame media labels: each must cite a real frame (0 <= frame < len) and a known label, else it is rejected."""
    out, seen = [], set()
    for x in items or []:
        f = x.get("frame") if isinstance(x, dict) else None
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not (0 <= int(f) < len(feats)) or int(f) in seen:
            rejected.append(dict(kind="frame_label", item=x, reason="no valid frame reference")); continue
        lab = x.get("label") if x.get("label") in FRAME_LABELS else "unknown"
        try:
            conf = max(0.0, min(1.0, float(x.get("confidence"))))
        except (TypeError, ValueError):
            conf = 0.0
        f = int(f); seen.add(f)
        out.append(dict(frame=f, t=round(feats.times[f], 2), label=lab, confidence=conf if lab != "unknown" else 0.0,
                        reason=str(x.get("reason") or "")[:200]))
    return sorted(out, key=lambda d: d["frame"])


def derive_media_nature(frame_labels, overall, feats):
    """Overall media_nature from the per-frame labels. >=2 distinct real classes -> 'mixed'; one class -> that value (a Muse
    'news_compilation' is kept for uniform camera clips); none -> Muse's own overall value. Always an AI observation."""
    by = {}
    for x in frame_labels:
        if x["label"] != "unknown":
            by.setdefault(x["label"], []).append(x["frame"])
    classes = [k for k in by if k in CLASS_NAMES]
    if not frame_labels or not classes:
        return dict(overall, per_frame=frame_labels, by_class=by)
    conf = lambda fr: sum(x["confidence"] for x in frame_labels if x["frame"] in fr) / len(fr)
    if len(classes) >= 2:
        value = "mixed"
        used = sorted(f for k in classes for f in by[k])
        summary = "Mixed: " + " + ".join(f"{CLASS_NAMES[k]} (frames {','.join(map(str, by[k]))})"
                                         for k in sorted(classes, key=lambda k: -len(by[k]))) + " (AI observation)"
        c = sum(conf(by[k]) for k in classes) / len(classes)
    else:
        value, used = classes[0], sorted(by[classes[0]])
        if value == "camera_footage" and overall.get("value") == "news_compilation":
            value = "news_compilation"
        summary = f"{CLASS_NAMES[classes[0]]} (frames {','.join(map(str, used))}) (AI observation)"
        c = conf(by[classes[0]])
    reasons = [dict(text=x["reason"], refs=[x["frame"]], timestamps_s=[x["t"]]) for x in frame_labels if x["label"] in classes and x["reason"]][:6]
    return dict(value=value, confidence=round(c, 2), cited_frames=used, cited_timestamps_s=[round(feats.times[f], 2) for f in used],
                reasons=reasons, status="AI_OBSERVATION_HYPOTHESIS", per_frame=frame_labels, by_class=by, summary=summary,
                note="Per-frame model observation from sampled keyframes. Not proof; it never changes the verdict label by itself.")


def camera_frame_ids(clues, n):
    """Keyframe ids (0..n-1) whose nearest labelled frame is camera_footage; [] if no per-frame labels or none are camera.
    Real clips are the ones worth tracing to earlier copies; AI-generated shots have no original."""
    labs = ((clues or {}).get("media_nature") or {}).get("per_frame") if (clues or {}).get("status") == "ok" else None
    if not labs or not any(x["label"] == "camera_footage" for x in labs):
        return []
    return [i for i in range(n) if min(labs, key=lambda x: abs(x["frame"] - i))["label"] == "camera_footage"]


def _clean_nature(mn, fix, feats):
    """media_nature is an AI observation (hypothesis): needs cited frames and a known enum value, else it is 'unknown'."""
    mn = mn if isinstance(mn, dict) else {}
    val = mn.get("value") if mn.get("value") in NATURES else "unknown"
    item = fix(dict(refs=mn.get("refs")), "media_nature") if val != "unknown" else None
    if val != "unknown" and not item:
        val = "unknown"
    refs = sorted({r for r in (item or {}).get("refs", []) if isinstance(r, int)})
    try:
        conf = max(0.0, min(1.0, float(mn.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0
    reasons = [x for x in (fix(i, "media_nature_reason") for i in mn.get("reasons", []) if isinstance(i, dict)) if x]
    return dict(value=val, confidence=conf if val != "unknown" else 0.0, cited_frames=refs,
                cited_timestamps_s=[round(feats.times[r], 2) for r in refs], reasons=reasons, status="AI_OBSERVATION_HYPOTHESIS",
                note="Model observation from sampled keyframes. Not proof; it never changes the verdict label by itself.")


def _iso_ok(d):
    import datetime
    try:
        return datetime.date.fromisoformat(str(d)[:10]).isoformat() if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(d)) else None
    except ValueError:
        return None


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
    # Location guesses are HYPOTHESES (model inference from visual cues), never evidence and never contradictions.
    out["location_hypotheses"] = []
    if isinstance(ll, dict) and ll.get("place") and ll.get("reasons"):
        refs = sorted({r for x in ll["reasons"] for r in x["refs"] if isinstance(r, int)})
        out["location_hypotheses"].append(dict(
            place=ll.get("place"), country=ll.get("country"), confidence=ll.get("confidence"), lat=ll.get("lat"), lon=ll.get("lon"),
            cited_frames=refs, cited_timestamps_s=[round(feats.times[r], 2) for r in refs],
            reasons=ll["reasons"], status="UNVERIFIED_HYPOTHESIS",
            note="Model inference from visual cues. Not verified by any independent source; not evidence for or against the claim."))
    # What the model says visibly clashes with the claim: an unverified observation, NOT a contradiction. Contradictions
    # (report['contradictions']) require independent evidence (physical-world data, an independently dated earlier copy).
    out["observations_against_claim"] = [dict(x, status="UNVERIFIED_MODEL_OBSERVATION")
                                         for x in (fix(i, "observation") for i in plan.get("contradictions_with_claim", [])) if x]
    out["media_nature"] = derive_media_nature(_clean_frame_labels(plan.get("frame_labels"), feats, rejected),
                                              _clean_nature(plan.get("media_nature"), fix, feats), feats)
    out["on_screen_dates"] = [dict(x, date=_iso_ok(x.get("date"))) for x in (fix(i, "on_screen_date") for i in plan.get("on_screen_dates", []) if isinstance(i, dict)) if x and _iso_ok(x.get("date"))]
    out["claim_parsed"] = plan.get("claim_parsed") or {}
    out["queries"] = [q for q in plan.get("queries", []) if isinstance(q, dict) and q.get("q")][:10]
    out["rejected_clues"] = rejected
    return out


def investigate_clip(clip: Clip, feats: ClipFeatures, claim: str | None = None, claim_date: str | None = None,
                     n_frames=8, use_audio=True, frames_dir: Path | None = None, post_text: str | None = None) -> dict:
    """Run Muse over keyframes (+ narration). Returns clue dict (see SCHEMA) plus cost/model/frames metadata."""
    idx = pick_frames(feats, n_frames)
    audio = analyze_audio(clip) if (use_audio and clip.has_audio) else dict(status="skipped" if clip.has_audio else "no_audio_track")
    audio_ok = audio.get("status") == "ok" and audio.get("has_speech") is not False
    ctx = f"It is being shared with this claim: '{claim}'" + (f" (claimed event date {claim_date})" if claim_date else "") if claim else \
        "No claim text was supplied."
    if post_text and not claim:
        ctx += (f" The post itself carries this title/caption (unverified, written by the uploader): {post_text!r}. "
                "Treat it as the claim for 'claim_parsed' only; judge media_nature from the frames alone.")
    atext = ""
    if audio.get("status") == "ok":
        atext = ("\nAudio analysis (from a separate audio model, timestamps in seconds): " +
                 f"speech={audio.get('has_speech')}, language={audio.get('language')}, summary={audio.get('transcript_summary')}, "
                 f"spoken place names={audio.get('spoken_place_names')}, spoken dates={audio.get('spoken_dates')}. Cite as \"audio@<seconds>\".\n")
    content = [{"type": "text", "text": f"These are {len(idx)} keyframes (numbered) from a viral disaster video. {ctx}.{atext}\n" + SCHEMA}]
    for i in idx:
        content.append({"type": "text", "text": f"Frame {i} (t={feats.times[i]:.1f}s):"})
        content.append(_jpeg_part(feats.frames[i]))
    txt, cost = chat(VISION_MODEL, content, max_tokens=MAX_TOKENS)
    provider = last_provider()
    try:
        plan = parse_json(txt)
    except Exception as e:
        return dict(status="error", error=f"unparseable reply: {e}", raw=txt[:500], model=VISION_MODEL, provider=provider, cost_usd=round(cost, 5),
                    frames_shown=idx, audio=audio)
    out = _clean(plan, feats, audio_ok)
    out.update(status="ok", model=VISION_MODEL, provider=provider, cost_usd=round(cost + audio.get("cost_usd", 0), 5), frames_shown=idx,
               frame_times_s={str(i): round(feats.times[i], 2) for i in idx}, audio=audio)
    if frames_dir:
        from .media import save_keyframe_jpgs
        out["frame_images"] = save_keyframe_jpgs(feats, frames_dir, idx)
    return out
