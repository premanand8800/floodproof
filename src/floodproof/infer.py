"""Inferred claim: when the user leaves claim/date/place empty and the input is a URL, read them from the post itself.

Every inferred field is labelled `inferred_from_post` in report['claim']['source']; user-entered values are never overridden.
Date preference: explicit date in title > description > Muse-read on-screen text > upload date ("date posted", weakest: it
only says when this copy was posted, so it can never by itself show recycling or consistency).
"""
from __future__ import annotations

import datetime
import re

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_PATTERNS = (
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), lambda m: (m[1], m[2], m[3])),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MON + r"[a-z]*\.?,?\s+(\d{4})\b", re.I), lambda m: (m[3], MONTHS[m[2][:3].lower()], m[1])),
    (re.compile(r"\b" + _MON + r"[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I), lambda m: (m[3], MONTHS[m[1][:3].lower()], m[2])),
)


def parse_explicit_date(text) -> str | None:
    """First explicit, unambiguous calendar date WITH a year in text -> ISO, else None (ambiguous d/m/y numerics are ignored)."""
    if not text:
        return None
    for rx, fn in _PATTERNS:
        for m in rx.finditer(str(text)):
            y, mo, d = (int(x) for x in fn(m))
            try:
                return datetime.date(y, mo, d).isoformat()
            except ValueError:
                continue
    return None


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def place_in_text(place, *texts) -> bool:
    p = _norm(place)
    return bool(p) and any(p in _norm(t) for t in texts if t)


def _spoken_match(place, spoken_places):
    p = _norm(place)
    for x in spoken_places:
        n = _norm(x.get("name"))
        if n and p and (n in p or p in n):
            return x
    return None


AI_LOC = "ai_location_hypothesis (used only to look up data, not as proof)"


def fallback_places(claim_d, clues, country_name=None):
    """Ordered [(place, source_label)] to try when the claimed place does not geocode. Narration, then on-screen text, then Muse's
    top location hypothesis (only if confidence >= 0.6 and it cites frames). Every entry keeps its label; none is proof."""
    ok = (clues or {}).get("status") == "ok"
    if not ok:
        return []
    skip = {_norm(country_name), _norm(claim_d.get("place"))} - {""}
    claimed = _norm(claim_d.get("place"))
    seen, out = set(), []

    def add(name, label):
        k = _norm(name)
        if k and k not in skip and k not in seen and not (claimed and k in claimed):   # a coarser part of the claimed place adds nothing
            seen.add(k); out.append((str(name).strip(), label))
    aud = clues.get("audio") or {}
    if aud.get("status") == "ok" and aud.get("has_speech") is not False:
        for x in aud.get("spoken_place_names") or []:
            if x.get("audio_t") is not None:
                add(x.get("name"), "narration")
    for x in clues.get("visible_text") or []:
        if x.get("refs") and x.get("text") and len(str(x["text"])) <= 60 and x.get("is_place") is True:
            add(x["text"], "on_screen_text")
    ll = clues.get("likely_location") or {}
    try:
        conf = float(ll.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    if ll.get("place") and conf >= 0.6 and ll.get("reasons"):
        add(ll["place"], AI_LOC)
    return out


def infer_claim(clip, user_claim, user_date, user_place, clues) -> dict:
    """-> dict(text, event_date, place, source{text,event_date,place}, event_date_basis, inferred_note). Only fills EMPTY fields."""
    is_url = bool(getattr(clip, "url", None))
    src = dict(text="user" if user_claim else None, event_date="user" if user_date else None, place="user" if user_place else None)
    out = dict(text=user_claim, event_date=user_date, place=user_place, source=src,
               event_date_basis="user" if user_date else None, event_date_basis_detail=None, inferred_note=None)
    cp = (clues.get("claim_parsed") or {}) if (clues or {}).get("status") == "ok" else {}
    title, desc = getattr(clip, "title", None), getattr(clip, "description", None)
    ok = (clues or {}).get("status") == "ok"
    screen = [x for x in (clues.get("visible_text") or [])] if ok else []
    screen_dates = [x for x in (clues.get("on_screen_dates") or []) if x.get("date")] if ok else []
    aud = (clues.get("audio") or {}) if ok else {}
    aud_ok = aud.get("status") == "ok" and aud.get("has_speech") is not False
    spoken_dates = [x for x in (aud.get("spoken_dates") or []) if x.get("date") and x.get("audio_t") is not None] if aud_ok else []
    spoken_places = [x for x in (aud.get("spoken_place_names") or []) if x.get("name") and x.get("audio_t") is not None] if aud_ok else []
    if not is_url:
        if user_claim and not user_place and cp.get("place"):
            out["place"], src["place"] = cp["place"], "user_claim_text"
        return out
    if not user_claim and title:
        out["text"], src["text"] = title, "inferred_from_post"
    if not user_date:
        d = parse_explicit_date(title)
        if d:
            basis, detail = "title", f"explicit date in the post title ({title!r})"
        elif (d := parse_explicit_date(desc)):
            basis, detail = "description", "explicit date in the post description"
        elif screen_dates:
            d, basis = screen_dates[0]["date"], "on_screen_text"
            detail = f"date read from on-screen text ({screen_dates[0].get('text')!r}, frames {screen_dates[0].get('refs')})"
        elif spoken_dates:
            d, basis = spoken_dates[0]["date"], "spoken_in_narration"
            detail = f"date spoken in the narration ({spoken_dates[0].get('text')!r}, audio@{spoken_dates[0]['audio_t']}); AI transcription, may contain errors"
        elif getattr(clip, "upload_date", None):
            d, basis, detail = clip.upload_date, "date_posted", "no explicit date in title, caption or on-screen text; using the date this post was uploaded (claimed date = date posted)"
        else:
            d = basis = detail = None
        if d:
            out["event_date"], src["event_date"], out["event_date_basis"], out["event_date_basis_detail"] = d, "inferred_from_post", basis, detail
    if not user_place and cp.get("place") and place_in_text(cp["place"], title, desc, *[x.get("text") for x in screen]):
        out["place"], src["place"] = cp["place"], "inferred_from_post"
    elif not user_place and cp.get("place") and (sp := _spoken_match(cp["place"], spoken_places)):
        out["place"], src["place"] = cp["place"], "inferred_from_narration"
        out["place_basis"] = f"spoken in narration (audio@{sp['audio_t']})"
    elif not user_place and user_claim and cp.get("place"):
        out["place"], src["place"] = cp["place"], "user_claim_text"
    if "inferred_from_post" in src.values():
        out["inferred_note"] = "Claim inferred from the post's title/caption (you did not enter one)"
    return out
