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
        elif getattr(clip, "upload_date", None):
            d, basis, detail = clip.upload_date, "date_posted", "no explicit date in title, caption or on-screen text; using the date this post was uploaded (claimed date = date posted)"
        else:
            d = basis = detail = None
        if d:
            out["event_date"], src["event_date"], out["event_date_basis"], out["event_date_basis_detail"] = d, "inferred_from_post", basis, detail
    if not user_place and cp.get("place") and place_in_text(cp["place"], title, desc, *[x.get("text") for x in screen]):
        out["place"], src["place"] = cp["place"], "inferred_from_post"
    elif not user_place and user_claim and cp.get("place"):
        out["place"], src["place"] = cp["place"], "user_claim_text"
    if "inferred_from_post" in src.values():
        out["inferred_note"] = "Claim inferred from the post's title/caption (you did not enter one)"
    return out
