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


_PARTIAL_PATTERNS = (
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MON + r"[a-z]*\b\.?(?!\s*,?\s*\d{4})", re.I),
    re.compile(r"\b" + _MON + r"[a-z]*\b\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{4})(?!\s*[:/.]\d)", re.I),
)


def resolve_partial_date(month, day, upload_date) -> str | None:
    """Day+month without a year -> ISO using the upload year; the previous year if that date would fall AFTER the upload date."""
    try:
        up = datetime.date.fromisoformat(str(upload_date)[:10])
        for y in (up.year, up.year - 1):
            try:
                d = datetime.date(y, int(month), int(day))
            except ValueError:
                continue
            if d <= up:
                return d.isoformat()
    except (TypeError, ValueError):
        pass
    return None


def parse_partial_date(text, upload_date) -> dict | None:
    """First 'August 26' / '26 Aug' (no year) in text -> dict(date, month, day, text) with the year resolved from upload_date."""
    if not text or not upload_date:
        return None
    for i, rx in enumerate(_PARTIAL_PATTERNS):
        for m in rx.finditer(str(text)):
            day, mon = (m[1], m[2]) if i == 0 else (m[2], m[1])
            iso = resolve_partial_date(MONTHS[mon[:3].lower()], day, upload_date)
            if iso:
                return dict(date=iso, month=MONTHS[mon[:3].lower()], day=int(day), text=m[0])
    return None


PARTIAL_NOTE = "year taken from post date"

# country name (as written in a post) -> ISO2, for country-level place inference. Country-level only: the physical check
# already treats a whole country as too coarse to corroborate a local event.
COUNTRIES = dict(nepal="NP", india="IN", pakistan="PK", bangladesh="BD", bhutan="BT", china="CN", afghanistan="AF", myanmar="MM",
                 sri_lanka="LK", indonesia="ID", philippines="PH", thailand="TH", vietnam="VN", malaysia="MY", japan="JP",
                 turkey="TR", iran="IR", iraq="IQ", syria="SY", yemen="YE", egypt="EG", sudan="SD", nigeria="NG", kenya="KE",
                 ethiopia="ET", somalia="SO", uganda="UG", tanzania="TZ", mozambique="MZ", malawi="MW", zambia="ZM",
                 zimbabwe="ZW", ghana="GH", niger="NE", mali="ML", cameroon="CM", italy="IT", france="FR", germany="DE",
                 spain="ES", greece="GR", poland="PL", romania="RO", ukraine="UA", russia="RU", brazil="BR", colombia="CO",
                 ecuador="EC", peru="PE", chile="CL", argentina="AR", bolivia="BO", mexico="MX", guatemala="GT", haiti="HT",
                 honduras="HN", australia="AU", canada="CA", fiji="FJ", libya="LY", lebanon="LB", korea="KR", laos="LA",
                 cambodia="KH", madagascar="MG")
_EXTRA_NAMES = {"united states": "US", "south korea": "KR", "usa": "US"}
_DISPLAY = {"US": "United States", "KR": "South Korea"}


def country_name(iso2):
    if iso2 in _DISPLAY:
        return _DISPLAY[iso2]
    k = next((k for k, v in COUNTRIES.items() if v == iso2), None)
    return k.replace("_", " ").title() if k else None


def find_country(iso2_hint, *texts):
    """Country named in any text (word match, so "Nepal's" matches). The hint (claim_parsed country) wins, else first mention."""
    blob = [(" " + _norm(t) + " ") for t in texts if t]
    names = {k.replace("_", " "): v for k, v in COUNTRIES.items()}
    names.update(_EXTRA_NAMES)
    if iso2_hint:
        for n, v in names.items():
            if v == iso2_hint and any(f" {n} " in b for b in blob):
                return v
    hits = [(b.find(f" {n} "), v) for n, v in names.items() for b in blob if f" {n} " in b]
    return min(hits)[1] if hits else None


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


def _partial_from_sources(title, desc, screen, partial_spoken, aud, upload_date):
    """(iso, basis, detail) for the first day+month-without-year date: title, description, on-screen text, narration."""
    if not upload_date:
        return None
    for label, basis, text in (("post title", "title_partial_year", title), ("post description", "description_partial_year", desc)):
        p = parse_partial_date(text, upload_date)
        if p:
            return p["date"], basis, f"date in the {label} without year ({p['text']!r}); {PARTIAL_NOTE}"
    for x in screen:
        p = parse_partial_date(x.get("text"), upload_date)
        if p:
            return p["date"], "on_screen_text_partial_year", f"date read from on-screen text without year ({p['text']!r}, frames {x.get('refs')}); {PARTIAL_NOTE}"
    for x in partial_spoken:
        iso = resolve_partial_date(x["month"], x["day"], upload_date)
        if iso:
            return iso, "spoken_in_narration_partial_year", (f"date spoken in narration without year ({x.get('text') or ''!r}, audio@{x['audio_t']}); "
                                                            f"{PARTIAL_NOTE}; AI transcription, may contain errors")
    p = parse_partial_date(aud.get("transcript_summary"), upload_date)
    if p:
        return p["date"], "spoken_in_narration_partial_year", (f"date mentioned in the AI narration summary without year ({p['text']!r}); {PARTIAL_NOTE}; "
                                                              "AI transcription, may contain errors")
    return None


def _infer_place(out, src, cp, title, desc, screen, spoken_places, aud):
    """Fallbacks when claim_parsed.place is a longer phrase than the post wording ("near Nepal-China border, high Himalayas"):
    (1) a comma-separated part of that phrase that the post states, (2) a country named in the post, (3) a country named in the
    narration. Every result is labelled; country-level is fine (the physical check treats it as too coarse)."""
    post_texts = [title, desc] + [x.get("text") for x in screen]
    spoken = [x.get("name") for x in spoken_places] + [aud.get("transcript_summary")]
    for part in re.split(r"[,;()]", cp.get("place") or ""):
        part = re.sub(r"^\s*(?:near|in|at|the|around|close to)\s+", "", part, flags=re.I).strip()
        if re.search(r"\bborder\b", part, re.I):      # "Nepal-China border" names two countries: not one place, fall through to a country
            continue
        core = re.sub(r"\s+(?:district|valley|region|province|area|city|village|border|municipality|gorge)$", "", part, flags=re.I).strip()
        if len(core) >= 4 and place_in_text(core, *post_texts):
            out["place"], src["place"], out["place_basis"] = part, "inferred_from_post", f"part of the claimed place stated in the post ({core!r})"
            return
    iso = find_country(cp.get("country_iso2"), *post_texts)
    if iso:
        out["place"], src["place"] = country_name(iso), "inferred_from_post"
        out["place_basis"] = "country named in the post title/caption/on-screen text (country-level)"
        out["country_iso2"] = iso
        return
    iso = find_country(cp.get("country_iso2"), *spoken)
    if iso:
        out["place"], src["place"] = country_name(iso), "inferred_from_narration"
        out["place_basis"] = "country named in the narration (AI transcription, country-level)"
        out["country_iso2"] = iso


# ---------------------------------------------------------------- self-disclosure and claim presence
# The uploader's OWN statement that the video is synthetic / animated / staged. Deterministic regexes, conservative: model
# names count only in a "made with ..." construction ("Runway" alone is an airport word). It is evidence of what the uploader
# says, not a model guess, and it never changes the verdict label. "tribute" alone is NOT a disclosure (it feeds claim_presence).
_AI_MODELS = (r"(?:sora|veo\s?\d?(?:\.\d)?|runway(?:\s?(?:gen\s?-?\d|ml))?|kling(?:\s?ai)?|midjourney|pika(?:\s?labs)?|"
              r"luma(?:\s?dream machine)?|hailuo|dall-?e|stable\s?diffusion|gemini|firefly|heygen|synthesia)")
_NEG = re.compile(r"(?:\bnot|\bno|\bnever|\bwithout|n't|\bnon)[\s\-]*(?:\w+\s){0,2}$", re.I)
DISCLOSURE_PATTERNS = (
    ("ai_generated", re.compile(
        r"\bai[\s\-]?(?:generated|created|made|produced|animated|rendered)\b|\bgenerated\s+(?:by|with|using)\s+ai\b|"
        r"\bai[\s\-]?(?:video|art|footage|animation|clip|image)s?\b|#\s?ai(?:art|generated|video|animation)\b|"
        r"#\s?(?:sora|veo\d?|kling|midjourney|runwayml)\b|"
        r"\b(?:made|created|generated|produced|animated|rendered)\s+(?:with|using|by|in|via)\s+" + _AI_MODELS + r"\b|"
        r"\b(?:synthetic|deep\s?fake)\s+(?:video|footage)\b", re.I)),
    ("animation", re.compile(
        r"\banimat(?:ion|ed)\b|\b[23]d\s+(?:animation|render(?:ing)?)\b|\bcgi\b|\bmotion graphics\b|#\s?animation\b", re.I)),
    ("reenactment", re.compile(
        r"\bre-?enact(?:ment|ed|ing)?\b|\billustrat(?:ive|ion|ed)\b|\bsimulat(?:ion|ed)\b|\bdramati[sz](?:ation|ed)\b|"
        r"\bartist(?:'s|s')?\s+(?:impression|rendering|concept)\b|\brecreat(?:ion|ed)\b", re.I)),
)
DISCLOSURE_PRIORITY = ("ai_generated", "animation", "reenactment")


def _snip(text, m, width=40):
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    while a > 0 and not text[a - 1].isspace():      # never cut a word in half
        a += 1
        if a >= m.start():
            a = m.start(); break
    while b < len(text) and not text[b].isspace():
        b -= 1
        if b <= m.end():
            b = m.end(); break
    return re.sub(r"\s+", " ", text[a:b]).strip()


def detect_disclosure(title=None, description=None, visible_text=None) -> dict:
    """-> dict(type ai_generated|animation|reenactment|none, evidence [{field, text}]). Scans the title, the description
    (hashtags included) and Muse's OCR of on-screen text. Negations ("not AI-generated") are skipped."""
    fields = [("title", title), ("description", description)]
    for x in (visible_text or []):
        if isinstance(x, dict) and x.get("text"):
            fields.append(("on_screen_text", str(x["text"])))
    found = {}
    for field, text in fields:
        if not text:
            continue
        text = str(text)
        for typ, rx in DISCLOSURE_PATTERNS:
            for m in rx.finditer(text):
                if _NEG.search(text[max(0, m.start() - 24):m.start()]):
                    continue
                ev = found.setdefault(typ, [])
                sn = _snip(text, m)
                if len(ev) < 4 and not any(e["field"] == field and e["text"] == sn for e in ev):
                    ev.append(dict(field=field, text=sn))
    for typ in DISCLOSURE_PRIORITY:
        if found.get(typ):
            return dict(type=typ, evidence=found[typ])
    return dict(type="none", evidence=[])


_TRIBUTE = re.compile(
    r"\btribute\b|\bexplain(?:ed|er|s)\b|\bdocumentary\b|\bvisuali[sz]ation\b|\bremember(?:ing)?\b|\bin memory\b|"
    r"\bbefore\s*(?:and|&)\s*after\b|\bhow (?:did|does|it|the)\b|\bwhy (?:did|does|the)\b|\bwhat (?:triggered|caused|happened)\b|"
    r"\bhistory of\b|\bin memoriam\b|\blessons?\b|\breconstruct", re.I)
_ASSERTS = re.compile(
    r"\bbreaking\b|\blive\b|\bcaught on (?:camera|video)\b|\bfootage\b|\bvideo shows\b|\brare video\b|\bhappening now\b|"
    r"\bjust now\b|\bviral\b|\blatest\b|\bshocking\b|\bcctv\b|\bdrone (?:video|footage)\b|\bwatch\b", re.I)


def claim_presence(title=None, description=None, claim_text=None, disclosure=None) -> dict:
    """-> dict(value asserts_real_event|tribute_or_explainer|no_claim, reason). A title such as "Tribute" / "Explained" that
    does not assert this is footage of the event is tribute_or_explainer. Descriptive only; never changes the label."""
    head = " ".join(x for x in (claim_text, title) if x).strip()
    if not head:
        return dict(value="no_claim", reason="no title or claim text names an event")
    asserts = _ASSERTS.search(head)
    trib = _TRIBUTE.search(head)
    disc = (disclosure or {}).get("type", "none")
    if trib and not asserts:
        return dict(value="tribute_or_explainer", reason=f"the title is framed as a tribute/explainer ({trib.group(0)!r}), not as footage of the event")
    if disc != "none" and not asserts:
        return dict(value="tribute_or_explainer", reason=f"the uploader labels it {disc.replace('_', ' ')}, so it does not present itself as footage of the event")
    if asserts:
        return dict(value="asserts_real_event", reason=f"the title/claim presents this as footage ({asserts.group(0)!r})")
    return dict(value="asserts_real_event", reason="the post presents the video together with a claim about an event")


def parse_all_dates(text) -> list:
    """Every explicit full date in text as sorted unique ISO strings."""
    out = set()
    if not text:
        return []
    for rx, fn in _PATTERNS:
        for m in rx.finditer(str(text)):
            try:
                y, mo, d = (int(x) for x in fn(m))
                out.add(datetime.date(y, mo, d).isoformat())
            except ValueError:
                continue
    return sorted(out)


PUBLISH_WINDOW_DAYS = 45


def _days(a, b):
    return (datetime.date.fromisoformat(str(a)[:10]) - datetime.date.fromisoformat(str(b)[:10])).days


def earlier_event_date(posted_date, upload_date, desc, screen_dates, spoken_dates, partial_spoken=()):
    """If `posted_date` (a title/description date) is within PUBLISH_WINDOW_DAYS of the upload date it is probably a publishing
    date; look for an EARLIER date named by the narration, on-screen text or description and return (iso, basis, detail), else None."""
    if not posted_date or not upload_date:
        return None
    try:
        if abs(_days(upload_date, posted_date)) > PUBLISH_WINDOW_DAYS:
            return None
    except ValueError:
        return None
    cands = []
    for x in spoken_dates:
        cands.append((x["date"], "spoken_in_narration", f"date spoken in the narration ({x.get('text')!r}, audio@{x['audio_t']}); AI transcription, may contain errors"))
    for x in screen_dates:
        cands.append((x["date"], "on_screen_text", f"date read from on-screen text ({x.get('text')!r}, frames {x.get('refs')})"))
    for x in partial_spoken:
        iso = resolve_partial_date(x["month"], x["day"], upload_date)
        if iso:
            cands.append((iso, "spoken_in_narration", f"date spoken in the narration without year ({x.get('text')!r}, audio@{x['audio_t']}); year taken from post date"))
    for d in parse_all_dates(desc):
        cands.append((d, "description", "an earlier explicit date in the post description"))
    p = parse_partial_date(desc, upload_date)
    if p:
        cands.append((p["date"], "description", f"an earlier date in the description without year ({p['text']!r}); {PARTIAL_NOTE}"))
    for d, basis, detail in cands:
        try:
            if _days(posted_date, d) >= 1:
                return d, basis, detail
        except ValueError:
            continue
    return None


def is_country_level_place(place, iso2=None) -> bool:
    """True when `place` is just a country name ("Japan"), so a specific place inside it is more useful."""
    n = _norm(place)
    if not n:
        return False
    names = {k.replace("_", " ") for k in COUNTRIES} | set(_EXTRA_NAMES)
    if iso2:
        cn = country_name(iso2)
        if cn:
            names.add(_norm(cn))
    return n in names


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
    partial_spoken = [x for x in (aud.get("partial_dates") or []) if x.get("month") and x.get("day") and x.get("audio_t") is not None] if aud_ok else []
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
        elif (pd_ := _partial_from_sources(title, desc, screen, partial_spoken, aud, getattr(clip, "upload_date", None))):
            d, basis, detail = pd_
        elif getattr(clip, "upload_date", None):
            d, basis, detail = clip.upload_date, "date_posted", "no explicit date in title, caption or on-screen text; using the date this post was uploaded (claimed date = date posted)"
        else:
            d = basis = detail = None
        if d and basis in ("title", "description", "title_partial_year", "description_partial_year"):
            ev = earlier_event_date(d, getattr(clip, "upload_date", None), desc, screen_dates, spoken_dates, partial_spoken)
            if ev:       # the post's own date is a publishing date; the narration / on-screen text names the real event date
                out["publication_date"] = dict(date=d, basis=basis, detail=detail)
                which = "description" if "description" in basis else "title"
                src_name = {"spoken_in_narration": "narration", "on_screen_text": "on-screen text", "description": "the description"}[ev[1]]
                d, basis = ev[0], ev[1]
                detail = f"event date from {src_name}; the {which} date is a publishing date. " + ev[2]
        if d:
            out["event_date"], src["event_date"], out["event_date_basis"], out["event_date_basis_detail"] = d, "inferred_from_post", basis, detail
    if not user_place and cp.get("place") and place_in_text(cp["place"], title, desc, *[x.get("text") for x in screen]):
        out["place"], src["place"] = cp["place"], "inferred_from_post"
    elif not user_place and cp.get("place") and (sp := _spoken_match(cp["place"], spoken_places)):
        out["place"], src["place"] = cp["place"], "inferred_from_narration"
        out["place_basis"] = f"spoken in narration (audio@{sp['audio_t']})"
    elif not user_place and user_claim and cp.get("place"):
        out["place"], src["place"] = cp["place"], "user_claim_text"
    if not user_place and not out["place"]:
        _infer_place(out, src, cp, title, desc, screen, spoken_places, aud)
    if "inferred_from_post" in src.values():
        out["inferred_note"] = "Claim inferred from the post's title/caption (you did not enter one)"
    return out
