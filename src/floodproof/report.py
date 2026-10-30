"""Investigation report: SEPARATE evidence layers, never merged into one score.

Layers: visual_matching | clues | physical_world_check | memory | claim | final_verdict.

Verdict discipline
  RECYCLED_FOOTAGE        a matcher-proven copy was published (platform upload date) strictly BEFORE the claimed event
                          date. This shows the footage pre-dates the claim; it does not say the footage is fake.
  CONSISTENT_WITH_CLAIM   no contradiction found in the available evidence (no earlier proven copy, physical check
                          supports, no cited contradicting clue). NOT a statement that the footage is authentic.
  UNVERIFIED              everything else, including "no copy found" (a valid outcome) and "visual match but dates cannot
                          show it predates the claim".
A visual match alone proves "previously published footage", never "the claim is false".
"""
from __future__ import annotations

import datetime
import json
import re
import uuid
from pathlib import Path

LABELS = ("RECYCLED_FOOTAGE", "CONSISTENT_WITH_CLAIM", "UNVERIFIED")
SCHEMA_VERSION = "1.5"   # 1.5 also carries optional input_publication_before_claim finding and optional upload_timestamp (exact UTC time) on copies, and also carries optional disclosure, claim_presence, signals, headline (none changes the label); 1.5: optional proven_matches[].proof (clean frames, inliers, explanation); 1.4: media_nature.per_frame + value "mixed", claim date bases *_partial_year; 1.3 reports stay valid; 1.3: optional report["debunk"] (see debunk.py); 1.2 reports stay valid
REQUIRED_KEYS = ("schema_version", "investigation_id", "created_at", "input", "claim", "memory", "visual_matching",
                 "clues", "physical_world_check", "final_verdict")


EVENT_TYPES = ("flood", "flash_flood", "glof", "landslide", "cyclone", "storm_surge", "tsunami", "earthquake", "wildfire", "unknown")


def run_physical(place, lat, lon, date, window_days=3, event_type="unknown", country=None):
    """physical.check(place, lat, lon, date, window_days, event_type, country) -> dict, if importable; else NOT_RUN.
    event_type/country come from the claim as parsed by Muse (country = ISO-3166 alpha-2)."""
    try:
        from . import physical  # written by another module/agent
    except Exception as e:
        return dict(status="NOT_RUN", reason=f"physical module not importable ({type(e).__name__})")
    try:
        d = physical.check(place, lat, lon, date, window_days=window_days,
                           event_type=event_type if event_type in EVENT_TYPES else 'unknown', country=country)
        return annotate_physical(d) if isinstance(d, dict) else dict(status="ERROR", reason="physical.check did not return a dict")
    except Exception as e:
        return dict(status="ERROR", reason=str(e)[:200])


INSUFFICIENT = "Insufficient corroboration \u2014 this does not show the event didn't happen"


def annotate_physical(p: dict) -> dict:
    """NO_EVIDENCE keeps its enum value; the user-facing wording is 'insufficient corroboration', never 'no evidence'."""
    if isinstance(p, dict) and p.get("verdict") == "NO_EVIDENCE":
        p["verdict_display"] = INSUFFICIENT
        lim = p.setdefault("limits", [])
        if INSUFFICIENT not in lim:
            lim.append(INSUFFICIENT + " (public hydrology/activation data cannot resolve flash floods, GLOFs or landslides).")
    return p


def physical_label(p: dict) -> str:
    """User-facing text for a physical check dict."""
    p = p or {}
    if p.get("verdict") == "NO_EVIDENCE":
        return INSUFFICIENT
    return str(p.get("verdict") or p.get("status") or "NOT_RUN")


def physical_supports(p: dict):
    """True / False / None from a physical.check dict. Accepts consistent_with_claim: bool, or status in
    SUPPORTS|CONSISTENT / CONTRADICTS|INCONSISTENT."""
    if not isinstance(p, dict):
        return None
    v = p.get("consistent_with_claim")
    if isinstance(v, bool):
        return v
    s = str(p.get("status") or p.get("verdict") or "").upper()
    if s in ("SUPPORTS", "CONSISTENT", "CONSISTENT_WITH_CLAIM"):
        return True
    if s in ("CONTRADICTS", "INCONSISTENT", "NOT_SUPPORTED"):
        return False
    return None


def all_dates(m):
    """Every date reported for one proven copy: its main date plus conflicting alternatives (kept visible, never merged)."""
    ds = [m["upload_date"]] if m.get("upload_date") else []
    ds += [a["date"] for a in m.get("date_alternatives", []) if a.get("date")]
    return ds


def date_conflicts(proven):
    """URLs for which sources disagree on the publication date."""
    return [dict(url=m["url"], dates=[dict(date=m.get("upload_date"), source=m.get("date_source"))] + list(m.get("date_alternatives", [])))
            for m in proven if len(set(all_dates(m))) > 1]


def is_independent(m) -> bool:
    """True if the copy's date was obtained live in this investigation (yt-dlp metadata, page meta, Wayback). A date that is
    only inherited from memory is not independent. Legacy dicts: inferred from date_source."""
    if "date_independent" in m:
        return bool(m["date_independent"])
    return not str(m.get("date_source") or "").startswith("memory")


LANDMARK_RE = re.compile(r"landmark|monument|tower|bridge|cathedral|church|temple|mosque|statue|skyline|famous|palace|castle|stadium|"
                         r"waterfall|falls\b|colosseum|opera house|big ben|eiffel", re.I)


def landmark_scene(clues):
    """True if Muse's scene description names a landmark-type scene (landmarks are filmed by many people: a lone matching frame is weak)."""
    st = ((clues or {}).get("scene_type") or {}).get("value") if (clues or {}).get("status") == "ok" else None
    return bool(st and LANDMARK_RE.search(str(st)))


def mark_weak_blocked(proven, clues):
    """Single-frame-only matches (match.weak_match) cannot support RECYCLED when Muse calls the scene a landmark; flag them in place."""
    lm = landmark_scene(clues)
    for m in proven:
        mm = m.get("match") or {}
        if mm.get("weak_match") and lm:
            mm["weak_blocked"] = "scene described as a landmark scene; a single matching frame is not enough"
    return proven


MIN_DATE_MARGIN_DAYS = 2      # date-only comparisons: the copy must be at least this many calendar days before the claimed date
TZ_NOTE = "copy dated within a day of the claimed event; time-zone uncertainty"
EARLIEST_DAY_START = datetime.timezone(datetime.timedelta(hours=14))   # the calendar day D starts first at UTC+14 (D-1 10:00 UTC)


def _day(s):
    return datetime.date.fromisoformat(str(s)[:10])


def date_precedes(date, claim_date, timestamp=None):
    """True only if a copy published at `date` (a UTC calendar date from yt-dlp upload_date) certainly predates the claimed event day,
    whatever the time zone of the event (Nepal is UTC+5:45, so a clip uploaded the same UTC day may be hours AFTER the event).
      - exact UTC timestamp known: it must be before the moment the claimed calendar day first begins anywhere on Earth (UTC+14);
      - date only: the date must be at least MIN_DATE_MARGIN_DAYS calendar days before the claimed date."""
    if not date or not claim_date:
        return False
    if timestamp:
        try:
            t = datetime.datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=datetime.timezone.utc)
            return t < datetime.datetime.combine(_day(claim_date), datetime.time(0), tzinfo=EARLIEST_DAY_START)
        except ValueError:
            pass
    try:
        return (_day(claim_date) - _day(date)).days >= MIN_DATE_MARGIN_DAYS
    except ValueError:
        return False


def copy_precedes(m, claim_date):
    """date_precedes for every date reported for one copy (conflicting alternatives are date-only and must ALL predate the claim)."""
    ds = all_dates(m)
    if not ds or not claim_date:
        return False
    main = m.get("upload_date")
    return all(date_precedes(d, claim_date, m.get("upload_timestamp") if d == main else None) for d in ds)


def proven_before(proven, claim_date, independent_only=False):
    """Matcher-proven copies that certainly predate the claimed event date (see date_precedes for the time-zone margin). If sources
    disagree on a copy's date, ALL reported dates must predate the claim (conservative: a conflict that straddles the claim date
    does not count)."""
    if not claim_date:
        return []
    ok = [m for m in proven if all_dates(m) and copy_precedes(m, claim_date) and (is_independent(m) or not independent_only)
          and not (m.get("match") or {}).get("weak_blocked")]
    return sorted(ok, key=lambda m: min(all_dates(m)))


def proven_within_margin(proven, claim_date):
    """Independently dated copies that are earlier than the claimed date on the calendar but NOT by the time-zone margin."""
    if not claim_date:
        return []
    return [m for m in proven if all_dates(m) and max(all_dates(m)) < claim_date and is_independent(m)
            and not copy_precedes(m, claim_date) and not (m.get("match") or {}).get("weak_blocked")]


def input_publication(visual):
    """The input URL's own publication record. It is the video itself, NEVER an 'earlier copy': it is used only for the separate
    input_publication_before_claim finding and the (circularity-guarded) timing check of CONSISTENT_WITH_CLAIM."""
    ip = (visual or {}).get("input_publication")
    if ip and ip.get("upload_date"):
        return dict(ip, adapter="input_url", date_independent=True, match=dict(identity=True, consistent_frames=None, max_inliers=None,
                                                                              temporal_offset=0.0, mirrored=False, best_pairs=[]))
    return None


INPUT_BEFORE_TEXT = "This video was posted before the event it is claimed to show."


def input_before_claim(input_pub, claim):
    """Own finding (not a matcher result): the input video's publication date certainly predates the claimed event date.
    -> dict(finding, text, input_date, claim_date, ...) or None. Needs a real claimed event date (not a date-posted fallback)."""
    cd = claim_date_for_logic(claim)
    if not input_pub or not input_pub.get("upload_date") or not cd:
        return None
    if not date_precedes(input_pub["upload_date"], cd, input_pub.get("upload_timestamp")):
        return None
    return dict(finding="input_publication_before_claim", text=INPUT_BEFORE_TEXT, input_date=input_pub["upload_date"], claim_date=cd,
                url=input_pub.get("url"), date_source=input_pub.get("date_source"), upload_timestamp=input_pub.get("upload_timestamp"),
                precision="timestamp" if input_pub.get("upload_timestamp") else "date",
                margin_days=(_day(cd) - _day(input_pub["upload_date"])).days,
                note="This is the video's own publication date, not an earlier copy found by the matcher. A post published before the claimed "
                     "event cannot show that event; it does not show the claimed event did not happen.")


NATURE_CHIP = {
    "camera_footage": "Looks like camera footage (AI observation)",
    "news_compilation": "Edited news report containing third-party clips \u2014 earlier copies of individual clips are expected and do not by themselves mean recycling.",
    "animation_or_cgi": "Not camera footage: animated / reenactment (AI observation)",
    "likely_ai_generated": "Not camera footage: likely AI-generated (AI observation)",
    "screen_recording": "Screen recording, not original camera footage (AI observation)",
    "mixed": "Mixed: AI-generated shots + real footage clips (AI observation)",
}


def nature_sentence(mn):
    """One sentence for the verdict summary, or '' when unknown/absent. Always flagged as an AI observation."""
    v = (mn or {}).get("value")
    if v not in NATURE_CHIP or v == "camera_footage":
        return ""
    fr = ",".join(str(x) for x in (mn.get("cited_frames") or []))
    chip = mn.get("summary") if v == "mixed" and mn.get("summary") else NATURE_CHIP[v]
    return f" Media nature (AI observation from frames {fr}, not proof; does not change the label): " + chip + (" " if not NATURE_CHIP[v].endswith(".") else " ")


def claim_date_for_logic(claim):
    """A claimed date that is only 'date posted' says nothing about the event, so it never drives RECYCLED/CONSISTENT."""
    c = claim or {}
    return None if c.get("event_date_basis") == "date_posted" else c.get("event_date")


def decide_verdict(claim, proven, clues, physical, input_meta=None, input_pub=None, media_nature=None):
    posted = (claim or {}).get("event_date_basis") == "date_posted"
    post_date = (claim or {}).get("event_date") if posted else None
    compilation = (media_nature or {}).get("value") == "news_compilation"
    claim_date = claim_date_for_logic(claim)
    proven_real = list(proven)           # matcher-proven copies of OTHER uploads; the input's own publication is never one of them
    limits = [
        "A visual match proves shared footage ('previously published'), not that a claim is false or that the footage is fake.",
        "Upload dates are platform metadata (or an archive snapshot date, an upper bound) for the copy found; the earliest copy found is not necessarily the original source.",
        "Search is not exhaustive: memory covers only indexed clips, reverse search rarely finds originals, and heavy crops/edits can evade hash shortlisting.",
        "'Not found' means not found by these adapters, not that no earlier copy exists.",
        f"Dates are UTC calendar dates from the platform. Without an exact timestamp a copy must be dated at least {MIN_DATE_MARGIN_DAYS} days before the claimed date to count as earlier (time-zone margin: Nepal is UTC+5:45).",
    ]
    earlier = proven_before(proven_real, claim_date, independent_only=True)
    unverified_earlier = [] if earlier else proven_before(proven_real, claim_date)    # earlier only on an inherited date
    ibc = input_before_claim(input_pub, claim)
    within = [] if (earlier or ibc) else proven_within_margin(proven_real + ([input_pub] if input_pub else []), claim_date)
    conflicts = date_conflicts(proven_real)
    cl_ok = (clues or {}).get("status") == "ok"
    contrad = list((clues or {}).get("observations_against_claim", (clues or {}).get("contradictions_with_claim", []))) if cl_ok else []
    phys = physical_supports(physical)
    ibc_s = (f"This video was posted on {ibc['input_date']}, before the claimed event on {ibc['claim_date']} (publication date per {ibc.get('date_source')}"
             + (", exact timestamp" if ibc["precision"] == "timestamp" else "") + "). ") if ibc else ""
    if earlier:
        e = earlier[0]
        label = "RECYCLED_FOOTAGE"
        ev_txt = (f"Matcher evidence: {e['match']['consistent_frames']} temporally consistent frame(s), up to {e['match']['max_inliers']} inliers"
                  f"{', mirrored' if e['match'].get('mirrored') else ''}.")
        s = (f"The {'video contains a clip published earlier: it matches' if compilation else 'footage matches'} the earliest copy found, published on {e['upload_date']} ({e['url']}, date independently fetched per {e.get('date_source')}), before the claimed event date {claim_date}. "
             f"{ev_txt} This shows the footage was previously published; it does not "
             "by itself show the footage is fake or that the claimed event did not happen.")
        if ibc:
            s += " Separately, " + ibc_s[0].lower() + ibc_s[1:].strip() + " (the input's own publication, not a matcher result.)"
    elif ibc:
        label = "RECYCLED_FOOTAGE"
        s = (ibc_s + "A post published before the claimed event cannot show that event. This is the input video's own publication date, "
             "not an earlier copy found by the matcher, and it does not show that the claimed event did not happen.")
        if proven_real:
            s += f" The frame matcher also proved {len(proven_real)} other copy(ies), none dated before the claim by the required margin."
    else:
        if input_pub and not proven_real:
            s = (f"The input URL was itself published on {input_pub['upload_date']} (per {input_pub.get('date_source')}), which is not before "
                 "the claimed event date, so recycling relative to the claim is not established. ")
        elif proven_real:
            e = min([m for m in proven_real if m.get("upload_date")] or proven_real, key=lambda m: m.get("upload_date") or "9999")
            s = (("The video contains a clip published earlier (" if compilation else "The footage matches a previously published copy (") + f"{e['url']}"
                 f"{', uploaded ' + e['upload_date'] if e.get('upload_date') else ', upload date unavailable'}), but "
                 + (("that date is too close to the claimed event date to count as earlier (time-zone margin, see below)"
                     if within and e.get("upload_date") and e["upload_date"][:10] < claim_date[:10]
                     else "that date is not earlier than the claimed event date") if claim_date and e.get("upload_date")
                    else ("the claimed date is only the date this post was published (" + str(post_date) + "), which says nothing about when the event happened"
                          if posted else "no claimed event date was given") if not claim_date else "its date is unknown")
                 + ", so recycling relative to the claim is not established. ")
        else:
            s = "No matcher-proven earlier copy was found by memory lookup, reverse image search or text search. "
        if within:
            w = within[0]
            s += (f"A copy dated {w['upload_date']} ({w['url']}) is within a day of the claimed event date {claim_date}: " + TZ_NOTE
                  + f". Upload dates are UTC and the event may be in a time zone such as Nepal (UTC+5:45), so the copy may have been posted after the event; an earlier copy needs to be dated at least {MIN_DATE_MARGIN_DAYS} days before the claim or carry an exact timestamp. ")
        if unverified_earlier:
            u = unverified_earlier[0]
            unverified_reason = "earlier copy found, date not independently verified"
            s += (f"An earlier copy was found ({u['url']}, dated {u['upload_date']}), but that date is only inherited from memory and could not be "
                  "re-fetched independently, so it is not used as proof of recycling (" + unverified_reason + "). ")
        # Physical-world SUPPORTS is context only and can NEVER raise the verdict (C2 H05 bug: a debunked clip judged consistent).
        # CONSISTENT_WITH_CLAIM needs provenance evidence: an independently dated copy (input publication or proven copy) that
        # falls within a few days of the claimed event date, no earlier independent copy, no observations against the claim.
        ind = [m for m in proven_real + ([input_pub] if input_pub else []) if is_independent(m) and m.get("upload_date") and not (m.get("match") or {}).get("weak_match")]
        # Circularity guard: when the claimed date was inferred from the post itself (title/caption/narration/post date),
        # the post's own publication date cannot confirm it; only OTHER independently dated copies count.
        if ((claim or {}).get("source") or {}).get("event_date") == "inferred_from_post":
            ind = [m for m in ind if m.get("adapter") != "input_url"]
        timing_ok = bool(claim_date and ind and not within and
                         0 <= (datetime.date.fromisoformat(min(m["upload_date"] for m in ind)[:10]) - datetime.date.fromisoformat(claim_date[:10])).days <= 3)
        if timing_ok and not contrad and not earlier and not unverified_earlier and phys is not False:
            label = "CONSISTENT_WITH_CLAIM"
            only_input = all(m.get("adapter") == "input_url" for m in ind)
            s += (("The input video itself was posted within days of the claimed event date (its own publication date, not independent corroboration of the claim)"
                   if only_input else "The earliest independently dated copy was published within days of the claimed event date")
                  + ", and no earlier copy or cited contradiction was found"
                  + (", and the physical-world check adds supporting context (context only; it did not change the verdict)" if phys else "")
                  + ". This means no contradiction was found in the available evidence; it is not a finding that the footage is authentic.")
        else:
            label = "UNVERIFIED"
            if phys is False:
                s += "The physical-world check did not support the claimed place/date. "
            elif (physical or {}).get("verdict") == "NO_EVIDENCE":
                s += INSUFFICIENT + " (physical-world check). "
            elif phys is None:
                s += "The physical-world check was not available or inconclusive. "
            s += "The claim could not be verified from the available evidence."
    s += nature_sentence(media_nature)
    if (claim or {}).get("inferred_note"):
        s = f"[{claim['inferred_note']}; date basis: {claim.get('event_date_basis')}.] " + s
    if conflicts:
        s += " Sources disagree on the publication date of: " + "; ".join(
            c["url"] + " (" + ", ".join(f"{d['date']} per {d['source']}" for d in c["dates"]) + ")" for c in conflicts) + \
            ". Both dates are reported; the verdict uses the later one."
    if contrad:
        s += (" Separately, the AI visual clues report " + str(len(contrad)) + " unverified observation(s) that clash with the claim (e.g. \""
              + contrad[0]["text"][:140] + "\", frame ref " + str(contrad[0]["refs"]) + "); these are model observations, not independent evidence "
              "and not part of the visual-match proof.")
    if (physical or {}).get("verdict") == "NO_EVIDENCE":
        limits.append(INSUFFICIENT + ".")
    basis = dict(earlier_proven_copies=len(earlier), proven_copies=len(proven_real), physical_supports=phys,
                 unverified_model_observations=len(contrad), date_conflicts=len(conflicts),
                 media_nature=(media_nature or {}).get("value"), claim_date_basis=(claim or {}).get("event_date_basis"),
                 input_posted_before_claim=bool(ibc))
    if unverified_earlier and label == "UNVERIFIED":
        basis["unverified_reason"] = "earlier copy found, date not independently verified"
    if within and label == "UNVERIFIED":
        basis["unverified_reason"] = TZ_NOTE
        basis["time_zone_margin"] = dict(note=TZ_NOTE, copies=[m.get("url") for m in within], margin_days=MIN_DATE_MARGIN_DAYS)
    return dict(label=label, summary=s, limits=limits, basis=basis)


def independent_contradictions(claim, proven, physical, input_pub=None):
    """input_pub is accepted for compatibility and ignored: the input's own publication is a separate finding, never a contradiction."""
    """Claim-vs-evidence conflicts backed by INDEPENDENT evidence only. Muse observations and location hypotheses never qualify."""
    out = []
    cd = claim_date_for_logic(claim)
    for m in proven_before(list(proven), cd, independent_only=True)[:1]:
        out.append(dict(kind="independently_dated_earlier_copy", text=f"Footage was already published on {m['upload_date']} (earliest copy found), before the claimed event date {cd}.",
                        evidence=[m["url"]], date_source=m.get("date_source")))
    if (physical or {}).get("verdict") == "CONTRADICTS":
        out.append(dict(kind="physical_world_data", text="; ".join((physical.get("reasons") or [])[:2]) or "physical-world data contradict the claim",
                        evidence=list(physical.get("links") or [])[:3], note="weak: no physical trace found at model resolution"))
    return out


def nature_layer(clues):
    mn = (clues or {}).get("media_nature") if (clues or {}).get("status") == "ok" else None
    return mn if isinstance(mn, dict) else dict(value="unknown", confidence=0.0, cited_frames=[], cited_timestamps_s=[], reasons=[],
                                                 status="NOT_RUN", note="Muse clues were not available, so the media nature was not assessed.")


# ---------------------------------------------------------------- clarity layer: disclosure, signals, headline
# NONE of this changes final_verdict.label. Physical / official-event support, in-clip signals and the uploader's own words
# are context: real events are exactly what recycled fakes get attached to (global G08: physical SUPPORTS on a debunked
# clip; held-out C2 H05: a wrong CONSISTENT verdict). Only a matcher-proven, independently dated earlier copy can produce
# RECYCLED_FOOTAGE, and CONSISTENT_WITH_CLAIM still needs an independently dated copy near the claimed date.
DISCLOSURE_TYPES = ("ai_generated", "animation", "reenactment", "none")
CLAIM_PRESENCE = ("asserts_real_event", "tribute_or_explainer", "no_claim")
TONES = ("recycled", "ai_disclosed", "signals_agree", "unverified", "consistent")
SIGNALS_NEEDED = 3

_BROADCASTERS_CI = re.compile(r"\b(NHK|BBC|Reuters|AFP|CNN|Al Jazeera|Sky News|NDTV|Kyodo|Xinhua|CGTN|France ?24|DW News|ABC News|NBC News|"
                              r"CBS News|Fox News|Getty Images|National Geographic|Fuji TV|TV Asahi|Nippon TV|TRT World|Channel 4 News)\b", re.I)
_BROADCASTERS_CS = re.compile(r"\b(AP|ANI|PTI|TBS|ITV)\b")


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def disclosure_layer(input_meta, clues):
    from .infer import detect_disclosure
    vt = (clues or {}).get("visible_text") if (clues or {}).get("status") == "ok" else None
    return detect_disclosure((input_meta or {}).get("title"), (input_meta or {}).get("description"), vt)


def claim_presence_layer(input_meta, claim, disclosure):
    from .infer import claim_presence
    return claim_presence((input_meta or {}).get("title"), (input_meta or {}).get("description"), (claim or {}).get("text"), disclosure)


def event_phrase(claim):
    c = claim or {}
    et = c.get("event_type") if c.get("event_type") not in (None, "unknown") else None
    place = c.get("place")
    if et:
        return "the " + (f"{place} " if place else "") + et.replace("_", " ")
    return "the event" + (f" in {place}" if place else "")


def _within_days(a, b, n):
    try:
        return abs((datetime.date.fromisoformat(str(a)[:10]) - datetime.date.fromisoformat(str(b)[:10])).days) <= n
    except (TypeError, ValueError):
        return False


def signals_assessment(claim, clues, physical, proven, input_pub, media_nature, disclosure):
    """Count independent-ish signals that AGREE with the claim, and anything that contradicts. Context only; never the label."""
    c = claim or {}
    cd = claim_date_for_logic(c)
    ok = (clues or {}).get("status") == "ok"
    aud = ((clues or {}).get("audio") or {}) if ok else {}
    speech = aud.get("status") == "ok" and aud.get("has_speech") is not False
    agree, against = [], []
    # 1. date in narration / on-screen text equals the claimed date
    if cd and ok:
        hits = []
        for x in (aud.get("spoken_dates") or []) if speech else []:
            if x.get("date") == cd:
                hits.append(f"narration says {x.get('text') or x['date']}")
        for x in (aud.get("partial_dates") or []) if speech else []:
            if x.get("month") and x.get("day") and cd[5:7] == f"{int(x['month']):02d}" and cd[8:10] == f"{int(x['day']):02d}":
                hits.append(f"narration says {x.get('text') or 'the same day and month'}")
        for x in clues.get("on_screen_dates") or []:
            if x.get("date") == cd:
                hits.append(f"on-screen text shows {x.get('text') or x['date']}")
        if hits:
            agree.append(dict(key="date", label="Date named in the clip", detail=hits[0] + f" (claimed {cd})", field="clues.audio.spoken_dates"))
    # 2. place named in the narration / on-screen text is inside the claimed region
    if ok:
        from .infer import country_name
        claimed = {_norm(c.get("place")), _norm((c.get("specific_place") or {}).get("name")), _norm(country_name(c.get("country_iso2")))} - {""}
        names = [x.get("name") for x in (aud.get("spoken_place_names") or []) if speech and x.get("name")] + \
                [x.get("text") for x in clues.get("visible_text") or [] if x.get("is_place") and x.get("text")]
        for n in names:
            nn = _norm(n)
            if nn and any(nn == k or nn in k or k in nn for k in claimed):
                agree.append(dict(key="place", label="Place named in the clip", detail=f"{n} (claimed place: {c.get('place')})", field="clues.audio.spoken_place_names"))
                break
    # 3. broadcaster logo or credit visible in the video
    if ok:
        for x in clues.get("visible_text") or []:
            t = str(x.get("text") or "")
            m = _BROADCASTERS_CI.search(t) or _BROADCASTERS_CS.search(t)
            if m:
                agree.append(dict(key="broadcaster", label="Broadcaster logo or credit", detail=f"{m.group(0)} visible on screen (a logo can be added to a copy; context only)", field="clues.visible_text"))
                break
    # 4. an official event record at a specific location within +-2 days
    loc = (physical or {}).get("location") or {}
    oe = (physical or {}).get("official_event") or {}
    if cd and loc.get("lat") is not None and not loc.get("country_level"):
        rec = None
        for a in oe.get("activations") or []:
            if _within_days(a.get("event_time") or a.get("activation_time"), cd, 2):
                rec = f"Copernicus EMS {a.get('code')}: {a.get('title')}, {a.get('distance_km')} km away"
                break
        us = oe.get("usgs") or {}
        if not rec and us.get("status") == "MATCH" and us.get("events"):
            e = us["events"][0]
            rec = f"USGS M{e.get('mag')} {e.get('place')}, {e.get('distance_km')} km away"
        if rec:
            agree.append(dict(key="official_event", label="Official event record", detail=rec, field="physical_world_check.official_event"))
    # 5. no proven earlier copy (independent or inherited-date)
    allp = list(proven)          # matcher-proven copies only; the input's own publication is not a matcher signal
    ibc = input_before_claim(input_pub, c)
    if cd and not proven_before(allp, cd) and not ibc:
        agree.append(dict(key="no_earlier_copy", label="No earlier copy proven", detail="the frame matcher found no earlier copy (absence of evidence, not proof)", field="visual_matching.status"))
    # contradictions: anything here blocks the signals-agree headline
    if proven_before(allp, cd, independent_only=True):
        against.append("a proven earlier copy exists")
    if ibc:
        against.append(f"this video was posted on {ibc['input_date']}, before the claimed event on {ibc['claim_date']}")
    if physical_supports(physical) is False:
        against.append("the physical-world check does not support the claimed place/date")
    if (disclosure or {}).get("type", "none") != "none":
        against.append("the uploader labels the video " + disclosure["type"].replace("_", " "))
    mv = (media_nature or {}).get("value")
    if mv in ("animation_or_cgi", "likely_ai_generated", "screen_recording", "mixed"):
        against.append(f"the frames look like {mv.replace('_', ' ')} (AI observation)")
    return dict(agree=agree, against=against, count=len(agree), needed=SIGNALS_NEEDED,
                qualifies=bool(len(agree) >= SIGNALS_NEEDED and not against),
                note="Context only: these signals never change the label. Real events are what fakes get attached to, so signals that agree with a claim do not prove the footage is genuine.")


def _host(u):
    return re.sub(r"^https?://(?:www\.)?", "", str(u or "")).split("/")[0]


def build_headline(rep) -> dict:
    """Deterministic headline: precedence proven earlier copy > uploader self-disclosure > input posted before the claim > consistent > signals agree > plain.
    -> dict(sentence, facts [{text, field}] (<= 3), tone). Never changes the label."""
    fv, claim, vis = rep["final_verdict"], rep.get("claim") or {}, rep.get("visual_matching") or {}
    disc, presence, sig = rep.get("disclosure") or {}, rep.get("claim_presence") or {}, rep.get("signals") or {}
    event = event_phrase(claim)
    label = fv["label"]
    proven = vis.get("proven_matches") or []
    ip = input_publication(vis)
    cd = claim_date_for_logic(claim)
    ibc = input_before_claim(ip, claim)
    earlier = proven_before(proven, cd, independent_only=True)
    if label == "RECYCLED_FOOTAGE" and earlier:
        e = earlier[0]
        m = e.get("match") or {}
        f2 = f"The frame matcher links the two: {m.get('consistent_frames')} consistent frames, up to {m.get('max_inliers')} inliers"
        return dict(tone="recycled",
                    sentence=f"This footage was already published on {e['upload_date']}, before the claimed event date {cd}.",
                    facts=[dict(text=f"Earliest copy found: {_host(e.get('url'))}, dated {e['upload_date']} (date from {e.get('date_source')})", field="visual_matching.earliest_proven_copy"),
                           dict(text=f2, field="visual_matching.proven_matches"),
                           dict(text="Recycled does not mean fake, and the claimed event itself may be real", field="final_verdict.label")])
    if disc.get("type", "none") != "none":
        t = disc["type"]
        what = {"ai_generated": "AI-generated", "animation": "an animation", "reenactment": "a reenactment or illustration"}[t]
        verb = "labels this video" if t == "ai_generated" else "describes this video as"
        ev = (disc.get("evidence") or [{}])[0]
        facts = [dict(text=f"The uploader's own words ({ev.get('field', '?').replace('_', ' ')}): “{ev.get('text', '')}”", field="disclosure.evidence")]
        if presence.get("reason"):
            facts.append(dict(text=presence["reason"][0].upper() + presence["reason"][1:], field="claim_presence"))
        mn = rep.get("media_nature") or {}
        if mn.get("value") in ("animation_or_cgi", "likely_ai_generated", "mixed"):
            facts.append(dict(text="Muse reads the frames as " + mn["value"].replace("_", " ") + " too (AI observation, not proof)", field="media_nature.value"))
        else:
            facts.append(dict(text=f"No earlier copy was proven among {vis.get('candidates_checked', 0)} candidates checked", field="visual_matching.candidates_checked"))
        return dict(tone="ai_disclosed", sentence=f"The uploader {verb} {what}. It is not documentary footage of {event}.", facts=facts[:3])
    if label == "RECYCLED_FOOTAGE" and ibc:
        return dict(tone="recycled",
                    sentence=f"This video was posted on {ibc['input_date']}, before the claimed event on {ibc['claim_date']}.",
                    facts=[dict(text=f"Its own publication date: {ibc['input_date']} (from {ibc.get('date_source')}" + (", exact timestamp" if ibc["precision"] == "timestamp" else "") + ")", field="visual_matching.input_publication"),
                           dict(text=f"Claimed event date: {ibc['claim_date']} ({ibc['margin_days']} days later)", field="input_publication_before_claim.claim_date"),
                           dict(text="This is the video's own publication date, not an earlier copy found by the matcher; the claimed event itself may be real", field="input_publication_before_claim.note")])
    if label == "CONSISTENT_WITH_CLAIM":
        ind = [m for m in proven + ([ip] if ip else []) if is_independent(m) and m.get("upload_date")]
        d0 = min((m["upload_date"] for m in ind), default="?")
        only_input = bool(ind) and all(m.get("adapter") == "input_url" for m in ind)
        return dict(tone="consistent",
                    sentence=(f"No contradiction found: this video was posted on {d0}, within days of the claimed event date {cd} (its own publication date, not independent corroboration). This is not proof the footage is authentic." if only_input else
                              f"No contradiction found: the earliest independently dated copy ({d0}) falls within days of the claimed event date {cd}. This is not proof the footage is authentic."),
                    facts=[dict(text=(f"The video's own publication date: {d0}" if only_input else f"Earliest independently dated copy: {d0}"), field="visual_matching.input_publication" if only_input else "visual_matching.earliest_proven_copy"),
                           dict(text="No earlier copy and no independent contradiction were found", field="contradictions"),
                           dict(text="Physical-world data is context only and did not change the label", field="physical_world_check.verdict")])
    if sig.get("qualifies"):
        return dict(tone="signals_agree",
                    sentence=f"Signals point to genuine footage of {event} ({sig['count']} signals agree). Not proven by a dated copy.",
                    facts=[dict(text=f"{s['label']}: {s['detail']}", field=s["field"]) for s in sig["agree"][:3]])
    n_checked = vis.get("candidates_checked", 0)
    basis = claim.get("event_date_basis")
    cd = claim.get("event_date") or ""
    close = [m for m in proven if (m.get("upload_date") or "")[:10] and cd and (m.get("upload_date") or "")[:10] < cd[:10]]
    f1 = (dict(text=f"No matcher-proven earlier copy among {n_checked} candidates checked", field="visual_matching.candidates_checked") if not proven else
          dict(text=f"A copy dated {close[0]['upload_date'][:10]} matched, but that is too close to the claimed date to rule out a time-zone difference",
               field="visual_matching.proven_matches") if close else
          dict(text=f"{len(proven)} copy(ies) matched, but none is independently dated before the claim", field="visual_matching.proven_matches"))
    f2 = (dict(text="No event date is known; only the date this video was posted", field="claim.event_date_basis") if (basis == "date_posted" or not claim.get("event_date")) else
          dict(text=f"Claimed event date {claim['event_date']} (from {str(basis).replace('_', ' ')})", field="claim.event_date_basis"))
    f3 = dict(text="Physical-world check: " + physical_label(rep.get("physical_world_check") or {}), field="physical_world_check.verdict")
    return dict(tone="unverified", sentence="We could not verify this video from the available evidence. That does not mean it is fake, and it does not mean it is genuine.",
                facts=[f1, f2, f3])


# ---------------------------------------------------------------- source map: a timeline per segment of the INPUT video
SEGMENT_KINDS = ("matched_source", "ai_generated", "animation", "graphic", "camera_unmatched", "unknown")
_MUSE_KIND = {"likely_ai_generated": "ai_generated", "animation_or_cgi": "animation", "graphic_or_text": "graphic",
              "screen_recording": "graphic", "camera_footage": "camera_unmatched"}
PLATFORMS = {"instagram.com": "Instagram", "youtube.com": "YouTube", "youtu.be": "YouTube", "facebook.com": "Facebook", "fb.watch": "Facebook",
             "x.com": "X", "twitter.com": "X", "tiktok.com": "TikTok", "vimeo.com": "Vimeo"}


def platform_of(url):
    h = _host(url)
    return PLATFORMS.get(h) or PLATFORMS.get(h.split(":")[0]) or h or "unknown"


def build_source_map(rep) -> list:
    """Per-segment provenance of the input video. A segment gets a SOURCE only from a matcher proof (the matched input frames of a
    proven copy); frames without a proof get Muse's per-frame media label (an AI observation) or 'unknown'. Contiguous frames that
    share the same sources are merged. Never guesses: no proof, no source."""
    inp = rep.get("input") or {}
    times = inp.get("keyframe_times_s") or []
    n = len(times)
    proven = (rep.get("visual_matching") or {}).get("proven_matches") or []
    if not n:
        return []
    by_frame = {i: [] for i in range(n)}
    for m in proven:
        mt = m.get("match") or {}
        mf = mt.get("matched_frames")
        if mf is None:                                   # older reports: only the best pairs were stored
            mf = [dict(a_frame=p["a_frame"], b_frame=p["b_frame"], a_t=p.get("a_t"), b_t=p.get("b_t"), inliers=p.get("inliers")) for p in mt.get("best_pairs") or []]
        for x in mf:
            f = x.get("a_frame")
            if isinstance(f, int) and 0 <= f < n:
                by_frame[f].append((m, x))
    labels = {x["frame"]: x["label"] for x in ((rep.get("media_nature") or {}).get("per_frame") or []) if isinstance(x.get("frame"), int)}
    dur = inp.get("duration")
    gaps = sorted(float(times[i + 1]) - float(times[i]) for i in range(n - 1)) or [1.0]
    end_total = float(times[-1]) + gaps[len(gaps) // 2]           # the last keyframe covers one typical gap, never more than is evidenced
    if isinstance(dur, (int, float)) and dur:
        end_total = min(end_total, float(dur))
    end_total = max(end_total, float(times[-1]))
    bounds = [0.0] + [round((float(times[i]) + float(times[i + 1])) / 2, 2) for i in range(n - 1)] + [round(end_total, 2)]
    segs = []
    for i in range(n):
        hits = by_frame[i]
        urls = tuple(sorted({m["url"] for m, _ in hits}))
        kind = "matched_source" if hits else _MUSE_KIND.get(labels.get(i), "unknown")
        key = (kind, urls)
        if segs and segs[-1]["_key"] == key:
            segs[-1]["end_s"] = bounds[i + 1]
            segs[-1]["frames"].append(i)
            segs[-1]["_hits"].extend(hits)
        else:
            segs.append(dict(_key=key, start_s=bounds[i], end_s=bounds[i + 1], kind=kind, frames=[i], _hits=list(hits)))
    out = []
    for sg in segs:
        srcs = []
        for url in sg["_key"][1]:
            hs = [(m, x) for m, x in sg["_hits"] if m["url"] == url]
            m = hs[0][0]
            best = max(hs, key=lambda mx: mx[1].get("inliers") or 0)[1]
            pr = m.get("proof") or {}
            srcs.append(dict(url=url, date=m.get("upload_date"), date_source=m.get("date_source"), date_independent=bool(m.get("date_independent")),
                             platform=platform_of(url), frames_matched=len({x["a_frame"] for _, x in hs}),
                             copy_t_range=[min(x["b_t"] for _, x in hs if x.get("b_t") is not None), max(x["b_t"] for _, x in hs if x.get("b_t") is not None)]
                             if any(x.get("b_t") is not None for _, x in hs) else None,
                             best_pair=dict(a_frame=best["a_frame"], b_frame=best.get("b_frame"), inliers=best.get("inliers")),
                             evidence_png=m.get("evidence_png"),
                             proof=dict(viral_frame=pr["viral_frame"], copy_frame=pr["copy_frame"], viral_t=pr["viral_t"], copy_t=pr["copy_t"]) if pr.get("viral_frame") else None))
        out.append(dict(start_s=sg["start_s"], end_s=sg["end_s"], kind=sg["kind"], sources=srcs, frames=sg["frames"],
                        basis="matcher proof" if sg["kind"] == "matched_source" else
                        ("Muse per-frame media label (AI observation, not proof)" if sg["kind"] != "unknown" else "no source found")))
    return out


def validate_source_map(r):
    sm = r["source_map"]
    assert isinstance(sm, list), "source_map must be a list"
    urls = {m["url"]: m for m in (r.get("visual_matching") or {}).get("proven_matches", [])}
    nk = (r.get("input") or {}).get("n_keyframes")
    prev_end = None
    for sg in sm:
        assert sg.get("kind") in SEGMENT_KINDS, "source_map kind invalid"
        assert isinstance(sg.get("start_s"), (int, float)) and isinstance(sg.get("end_s"), (int, float)) and sg["start_s"] <= sg["end_s"], "segment times invalid"
        assert prev_end is None or sg["start_s"] >= prev_end - 1e-6, "segments must be ordered and not overlap"
        prev_end = sg["end_s"]
        assert isinstance(sg.get("frames"), list) and all(isinstance(f, int) and f >= 0 and (nk is None or f < nk) for f in sg["frames"]), "segment frames invalid"
        assert (sg["kind"] == "matched_source") == bool(sg.get("sources")), "a segment has sources iff it is matched_source (no proof, no source)"
        for s in sg.get("sources") or []:
            m = urls.get(s.get("url"))
            assert m is not None, "a source must be a matcher-proven copy"
            assert s.get("date") == m.get("upload_date") and s.get("date_source") == m.get("date_source"), "source date must be the proven copy's date"
            assert isinstance(s.get("frames_matched"), int) and 1 <= s["frames_matched"] <= len(sg["frames"]), "frames_matched invalid"


def add_clarity(rep):
    """Compute (or recompute) the clarity layers of a finished report from the report itself. Context only: the label is never touched."""
    label = rep["final_verdict"]["label"]
    vis = rep.get("visual_matching") or {}
    rep["disclosure"] = disclosure_layer(rep.get("input"), rep.get("clues"))
    rep["claim_presence"] = claim_presence_layer(rep.get("input"), rep.get("claim"), rep["disclosure"])
    rep["signals"] = signals_assessment(rep.get("claim"), rep.get("clues"), rep.get("physical_world_check"), vis.get("proven_matches", []),
                                        input_publication(vis), rep.get("media_nature"), rep["disclosure"])
    ibc = input_before_claim(input_publication(vis), rep.get("claim"))
    if ibc:
        rep["input_publication_before_claim"] = ibc
    else:
        rep.pop("input_publication_before_claim", None)
    rep["headline"] = build_headline(rep)
    rep["source_map"] = build_source_map(rep)
    assert rep["final_verdict"]["label"] == label
    return rep


def build_report(*, input_meta, claim, memory, visual, clues, physical, investigation_id=None):
    proven = mark_weak_blocked(visual.get("proven_matches", []), clues)
    rep = dict(schema_version=SCHEMA_VERSION, investigation_id=investigation_id or uuid.uuid4().hex[:12],
               created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               input=input_meta, claim=claim, memory=memory, visual_matching=visual, clues=clues,
               physical_world_check=physical)
    rep["media_nature"] = nature_layer(clues)
    rep["contradictions"] = independent_contradictions(claim, proven, physical, input_publication(visual))
    rep["final_verdict"] = decide_verdict(claim, proven, clues, physical, input_meta, input_publication(visual), rep["media_nature"])
    add_clarity(rep)      # context layers (disclosure, claim presence, signals, headline, source map); never touches the label
    return rep


def validate_proof(p):
    """Schema 1.5 proof block of a proven match (optional; older reports have none)."""
    assert isinstance(p, dict), "proof must be an object"
    for k in ("viral_frame", "copy_frame"):
        assert isinstance(p.get(k), str) and p[k] and not p[k].startswith("/") and ".." not in p[k].split("/"), f"proof.{k} must be a relative path"
    for k in ("viral_t", "copy_t"):
        assert isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool), f"proof.{k} must be a number"
    inl = p.get("inliers")
    assert isinstance(inl, list) and len(inl) <= 40, "proof.inliers must be a list of at most 40 points"
    for q in inl:
        assert isinstance(q, (list, tuple)) and len(q) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1 for x in q), \
            "each inlier must be [x1,y1,x2,y2] normalised to 0-1"
    ex = p.get("explanation")          # optional
    if ex is not None:
        assert isinstance(ex, dict) and isinstance(ex.get("items"), list) and ex["items"], "explanation.items must be a non-empty list"
        for it in ex["items"]:
            assert isinstance(it, dict) and str(it.get("what") or "").strip(), "explanation item needs 'what'"
        assert isinstance(ex.get("differences", []), list), "explanation.differences must be a list"
        assert ex.get("model"), "explanation needs model"


GATE_DECISIONS = ("run", "stop", "run_forced")


def validate_gate(g):
    assert isinstance(g, dict) and g.get("decision") in GATE_DECISIONS, "gate.decision invalid"
    assert isinstance(g.get("metadata_signals", {}), dict) and isinstance(g.get("refs", []), list), "gate signals/refs invalid"
    for k in ("cost", "seconds"):
        assert isinstance(g.get(k), (int, float)) and not isinstance(g.get(k), bool), f"gate.{k} must be a number"
    if g["decision"] == "stop":
        assert g.get("kind") and g.get("reason") and g["refs"], "a gate stop needs kind, reason and cited frames"
        assert isinstance(g.get("confidence"), (int, float)) and g["confidence"] >= 0.8, "a gate stop needs confidence >= 0.8"


def validate_report(r: dict):
    """Raise AssertionError if the report violates the schema/rules; returns r."""
    for k in REQUIRED_KEYS:
        assert k in r, f"missing {k}"
    v = r["visual_matching"]
    assert v["status"] in ("FOUND", "NOT_FOUND")
    assert "physical_world_check" in r and isinstance(r["physical_world_check"], dict)
    assert (v["status"] == "FOUND") == bool(v.get("proven_matches")), "FOUND iff proven matches present"
    for a, st in (v.get("source_stats") or {}).items():
        assert st["proven"] <= st["checked"] <= st["candidates"], f"source_stats inconsistent for {a}"
    for m in v.get("proven_matches", []):
        for k in ("url", "upload_date", "date_source", "retrieved_at", "adapter", "match"):
            assert k in m, f"proven match missing {k}"
        for k in ("consistent_frames", "max_inliers", "temporal_offset", "best_pairs"):
            assert k in m["match"], f"match evidence missing {k}"
        if m.get("proof") is not None:
            validate_proof(m["proof"])
        assert isinstance(m.get("date_independent"), bool), "proven match missing date_independent"
        if not m["date_independent"]:
            assert str(m["date_source"] or "").startswith("memory") or not m.get("upload_date"), "non-independent date must be labelled as inherited"
        if m.get("stored_date") and m.get("independent_date"):
            assert m.get("independent_date_source"), "refetched date needs independent_date_source"
    cl = r["clues"]
    if cl.get("status") == "ok":
        assert "contradictions_with_claim" not in cl, "model observations must not be filed as contradictions"
        for h in cl.get("location_hypotheses", []):
            assert h.get("status") == "UNVERIFIED_HYPOTHESIS" and h.get("cited_frames") is not None, "location guess must be a cited hypothesis"
    for c in r.get("contradictions", []):
        assert c.get("evidence") and c.get("kind") in ("independently_dated_earlier_copy", "physical_world_data"), "contradiction needs independent evidence"
    if r["schema_version"] >= "1.2":      # 1.1 demo reports stay valid without media_nature / claim.source
        mn = r.get("media_nature")
        assert isinstance(mn, dict) and mn.get("value") in ("camera_footage", "news_compilation", "animation_or_cgi", "likely_ai_generated",
                                                          "screen_recording", "mixed", "unknown"), "media_nature layer missing/invalid"
        assert mn.get("status") in ("AI_OBSERVATION_HYPOTHESIS", "NOT_RUN"), "media_nature must be labelled an AI observation"
        if mn["value"] != "unknown":
            assert mn.get("cited_frames"), "media_nature needs cited frames"
        nk = (r.get("input") or {}).get("n_keyframes")
        for x in mn.get("per_frame") or []:      # schema 1.4: every per-frame label must cite a real frame
            assert isinstance(x.get("frame"), int) and not isinstance(x.get("frame"), bool) and x["frame"] >= 0 and (nk is None or x["frame"] < nk), \
                "per-frame media label must cite a real frame"
            assert x.get("label") in ("camera_footage", "likely_ai_generated", "animation_or_cgi", "graphic_or_text", "screen_recording", "unknown"), \
                "unknown per-frame media label"
        if mn["value"] == "mixed":
            assert len({x["label"] for x in mn.get("per_frame") or [] if x["label"] not in ("unknown", "graphic_or_text")}) >= 2, \
                "mixed needs at least two per-frame classes"
        cs = r["claim"].get("source")
        if cs is not None:
            for k in ("text", "event_date", "place"):
                if cs.get(k) == "inferred_from_post":
                    assert (r["input"] or {}).get("url"), "inferred claim requires a URL input"
            if cs.get("event_date") == "inferred_from_post":
                assert r["claim"].get("event_date_basis") in ("title", "description", "on_screen_text", "date_posted", "spoken_in_narration",
                                                                "title_partial_year", "description_partial_year", "on_screen_text_partial_year",
                                                                "spoken_in_narration_partial_year")
    assert r["memory"]["status"] in ("HIT", "MISS", "NOT_RUN")
    if r.get("debunk") is not None:      # optional (schema 1.3); older reports simply do not have it
        from .debunk import validate_debunk
        validate_debunk(r["debunk"])
    if r.get("disclosure") is not None:       # optional clarity layer
        d = r["disclosure"]
        assert d.get("type") in DISCLOSURE_TYPES and isinstance(d.get("evidence"), list), "disclosure invalid"
        assert (d["type"] == "none") == (not d["evidence"]), "disclosure needs evidence iff its type is not none"
        for e in d["evidence"]:
            assert e.get("field") and e.get("text"), "disclosure evidence needs field and text"
        assert (r.get("claim_presence") or {}).get("value") in CLAIM_PRESENCE, "claim_presence invalid"
    if r.get("headline") is not None:
        hl = r["headline"]
        assert hl.get("tone") in TONES and str(hl.get("sentence") or "").strip(), "headline needs tone and sentence"
        assert isinstance(hl.get("facts"), list) and len(hl["facts"]) <= 3 and all(f.get("text") and f.get("field") for f in hl["facts"]), \
            "headline facts: at most 3, each citing a report field"
        assert hl["tone"] != "recycled" or r["final_verdict"]["label"] == "RECYCLED_FOOTAGE", "recycled headline needs the RECYCLED label"
        assert hl["tone"] != "signals_agree" or (r.get("signals") or {}).get("qualifies"), "signals_agree headline needs a qualifying assessment"
    if r.get("source_map") is not None:
        validate_source_map(r)
    if r.get("input_publication_before_claim") is not None:       # optional finding (schema 1.6): must match the report's own data
        f = r["input_publication_before_claim"]
        want = input_before_claim(input_publication(v), r["claim"])
        assert want and f.get("finding") == "input_publication_before_claim" and f.get("input_date") == want["input_date"] and f.get("claim_date") == want["claim_date"], \
            "input_publication_before_claim must follow from input_publication and the claimed event date"
        assert not any(m.get("adapter") == "input_url" for m in v.get("proven_matches", [])) and \
            (v.get("earliest_proven_copy") or {}).get("adapter") != "input_url", "the input's own publication is never a proven copy"
    if r.get("gate") is not None:          # optional relevance gate (older reports have none)
        validate_gate(r["gate"])
    fv = r["final_verdict"]
    assert fv["label"] in LABELS and fv["summary"] and fv["limits"]
    if fv["label"] == "RECYCLED_FOOTAGE":
        cd = claim_date_for_logic(r["claim"])
        assert cd and (proven_before(v["proven_matches"], cd, independent_only=True) or input_before_claim(input_publication(v), r["claim"])), \
            "RECYCLED requires a proven copy with an independently re-fetched date predating the claimed date (by the time-zone margin), or the input video posted before the claimed event"
    return r


def write_report(r: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "report.json"
    p.write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    return p


def render_text(r: dict) -> str:
    v, fv, c = r["visual_matching"], r["final_verdict"], r["claim"]
    L = [f"FloodProof investigation {r['investigation_id']}",
         f"Claim: {c.get('text')} (event date {c.get('event_date')}, place {c.get('place')})"
         + (f"  [{c['inferred_note']}; date basis: {c.get('event_date_basis')}]" if c.get("inferred_note") else ""),
         f"Media nature (AI observation): {(r.get('media_nature') or {}).get('value')} frames {(r.get('media_nature') or {}).get('cited_frames')}",
         f"Memory: {r['memory']['status']}   Visual matching: {v['status']} ({len(v.get('proven_matches', []))} proven)"]
    for m in v.get("proven_matches", []):
        dt = f"{m['upload_date']} ({m['date_source']}" + ("" if m.get("date_independent") else "; NOT independently verified") + ")"
        if m.get("stored_date") and m["stored_date"] != m["upload_date"]:
            dt += f" [memory said {m['stored_date']}]"
        L.append(f"  - [{m['adapter']}] {m['url']} dated {dt}; "
                 f"{m['match']['consistent_frames']} frames, {m['match']['max_inliers']} inliers; evidence: {m.get('evidence_png')}")
    ep = v.get("earliest_proven_copy")
    L.append(f"Earliest copy found: {ep['url'] + ' ' + str(ep['upload_date']) if ep else 'none'}   Source claimed by user: {(v.get('claimed_original') or {}).get('url')}")
    cl = r["clues"]
    if cl.get("status") == "ok":
        for h in cl.get("location_hypotheses", []):
            L.append(f"Hypothesis (unverified): location {h.get('place')}, {h.get('country')} (conf {h.get('confidence')}), cited frames {h.get('cited_frames')}")
        L.append(f"Muse clues: {len(cl.get('observations_against_claim', []))} unverified observation(s) clashing with the claim; "
                 f"scene: {(cl.get('scene_type') or {}).get('value')}")
    else:
        L.append(f"Muse clues: {cl.get('status')}")
    L.append(f"Physical-world check: {physical_label(r['physical_world_check'])}")
    for c_ in r.get("contradictions", []):
        L.append(f"Contradiction (independent evidence): {c_['text']}")
    L.append(f"VERDICT: {fv['label']}\n{fv['summary']}\nLimits:\n" + "\n".join(f"  * {x}" for x in fv["limits"]))
    return "\n".join(L)
