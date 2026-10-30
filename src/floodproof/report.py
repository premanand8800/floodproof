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
import uuid
from pathlib import Path

LABELS = ("RECYCLED_FOOTAGE", "CONSISTENT_WITH_CLAIM", "UNVERIFIED")
SCHEMA_VERSION = "1.5"
REQUIRED_KEYS = ("schema_version", "investigation_id", "created_at", "input", "claim", "memory", "visual_matching",
                 "clues", "physical_world_check", "final_verdict")


EVENT_TYPES = ("flood", "flash_flood", "glof", "landslide", "cyclone", "storm_surge", "unknown")


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


def proven_before(proven, claim_date, independent_only=False):
    """Matcher-proven copies strictly earlier than the claimed event date. If sources disagree on a copy's date, ALL
    reported dates must predate the claim (conservative: a conflict that straddles the claim date does not count)."""
    if not claim_date:
        return []
    ok = [m for m in proven if all_dates(m) and max(all_dates(m)) < claim_date and (is_independent(m) or not independent_only)]
    return sorted(ok, key=lambda m: min(all_dates(m)))


def input_publication(visual):
    """The input URL's own publication record, usable as 'previously published' evidence (it is the footage itself)."""
    ip = (visual or {}).get("input_publication")
    if ip and ip.get("upload_date"):
        return dict(ip, adapter="input_url", match=dict(identity=True, consistent_frames=None, max_inliers=None,
                                                        temporal_offset=0.0, mirrored=False, best_pairs=[]))
    return None


NATURE_CHIP = {
    "camera_footage": "Looks like camera footage (AI observation)",
    "news_compilation": "Edited news report containing third-party clips \u2014 earlier copies of individual clips are expected and do not by themselves mean recycling.",
    "animation_or_cgi": "Not camera footage: animated / reenactment (AI observation)",
    "likely_ai_generated": "Not camera footage: likely AI-generated (AI observation)",
    "screen_recording": "Screen recording, not original camera footage (AI observation)",
}


def nature_sentence(mn):
    """One sentence for the verdict summary, or '' when unknown/absent. Always flagged as an AI observation."""
    v = (mn or {}).get("value")
    if v not in NATURE_CHIP or v == "camera_footage":
        return ""
    fr = ",".join(str(x) for x in (mn.get("cited_frames") or []))
    return f" Media nature (AI observation from frames {fr}, not proof; does not change the label): " + NATURE_CHIP[v] + (" " if not NATURE_CHIP[v].endswith(".") else " ")


def claim_date_for_logic(claim):
    """A claimed date that is only 'date posted' says nothing about the event, so it never drives RECYCLED/CONSISTENT."""
    c = claim or {}
    return None if c.get("event_date_basis") == "date_posted" else c.get("event_date")


def decide_verdict(claim, proven, clues, physical, input_meta=None, input_pub=None, media_nature=None):
    posted = (claim or {}).get("event_date_basis") == "date_posted"
    post_date = (claim or {}).get("event_date") if posted else None
    compilation = (media_nature or {}).get("value") == "news_compilation"
    claim_date = claim_date_for_logic(claim)
    proven_real = list(proven)           # matcher-proven copies of OTHER uploads
    if input_pub:
        proven = proven_real + [input_pub]   # the input's own publication also counts as 'previously published' evidence
    limits = [
        "A visual match proves shared footage ('previously published'), not that a claim is false or that the footage is fake.",
        "Upload dates are platform metadata (or an archive snapshot date, an upper bound) for the copy found; the earliest copy found is not necessarily the original source.",
        "Search is not exhaustive: memory covers only indexed clips, reverse search rarely finds originals, and heavy crops/edits can evade hash shortlisting.",
        "'Not found' means not found by these adapters, not that no earlier copy exists.",
    ]
    earlier = proven_before(proven, claim_date, independent_only=True)
    unverified_earlier = [] if earlier else proven_before(proven, claim_date)    # earlier only on an inherited date
    conflicts = date_conflicts(proven)
    cl_ok = (clues or {}).get("status") == "ok"
    contrad = list((clues or {}).get("observations_against_claim", (clues or {}).get("contradictions_with_claim", []))) if cl_ok else []
    phys = physical_supports(physical)
    ev = []   # evidence actually present, cited in the summary
    if earlier:
        e = earlier[0]
        label = "RECYCLED_FOOTAGE"
        ev_txt = ("The input is itself that publication." if e["match"].get("identity") else
                  f"Matcher evidence: {e['match']['consistent_frames']} temporally consistent frame(s), up to {e['match']['max_inliers']} inliers"
                  f"{', mirrored' if e['match'].get('mirrored') else ''}.")
        s = (f"The {'video contains a clip published earlier: it matches' if compilation else 'footage matches'} the earliest copy found, published on {e['upload_date']} ({e['url']}, date independently fetched per {e.get('date_source')}), before the claimed event date {claim_date}. "
             f"{ev_txt} This shows the footage was previously published; it does not "
             "by itself show the footage is fake or that the claimed event did not happen.")
    else:
        if input_pub and not proven_real:
            s = (f"The input URL was itself published on {input_pub['upload_date']} (per {input_pub.get('date_source')}), which is not earlier "
                 "than the claimed event date, so recycling relative to the claim is not established. ")
        elif proven_real:
            e = min([m for m in proven_real if m.get("upload_date")] or proven_real, key=lambda m: m.get("upload_date") or "9999")
            s = (("The video contains a clip published earlier (" if compilation else "The footage matches a previously published copy (") + f"{e['url']}"
                 f"{', uploaded ' + e['upload_date'] if e.get('upload_date') else ', upload date unavailable'}), but "
                 + ("that date is not earlier than the claimed event date" if claim_date and e.get("upload_date")
                    else ("the claimed date is only the date this post was published (" + str(post_date) + "), which says nothing about when the event happened"
                          if posted else "no claimed event date was given") if not claim_date else "its date is unknown")
                 + ", so recycling relative to the claim is not established. ")
        else:
            s = "No matcher-proven earlier copy was found by memory lookup, reverse image search or text search. "
        if unverified_earlier:
            u = unverified_earlier[0]
            unverified_reason = "earlier copy found, date not independently verified"
            s += (f"An earlier copy was found ({u['url']}, dated {u['upload_date']}), but that date is only inherited from memory and could not be "
                  "re-fetched independently, so it is not used as proof of recycling (" + unverified_reason + "). ")
        # Physical-world SUPPORTS is context only and can NEVER raise the verdict (C2 H05 bug: a debunked clip judged consistent).
        # CONSISTENT_WITH_CLAIM needs provenance evidence: an independently dated copy (input publication or proven copy) that
        # falls within a few days of the claimed event date, no earlier independent copy, no observations against the claim.
        ind = [m for m in proven if is_independent(m) and m.get("upload_date")]
        timing_ok = bool(claim_date and ind and
                         -1 <= (datetime.date.fromisoformat(min(m["upload_date"] for m in ind)[:10]) - datetime.date.fromisoformat(claim_date[:10])).days <= 3)
        if timing_ok and not contrad and not earlier and not unverified_earlier and phys is not False:
            label = "CONSISTENT_WITH_CLAIM"
            s += ("The earliest independently dated copy was published within days of the claimed event date, and no earlier copy or cited contradiction was found"
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
                 media_nature=(media_nature or {}).get("value"), claim_date_basis=(claim or {}).get("event_date_basis"))
    if unverified_earlier and label == "UNVERIFIED":
        basis["unverified_reason"] = "earlier copy found, date not independently verified"
    return dict(label=label, summary=s, limits=limits, basis=basis)


def independent_contradictions(claim, proven, physical, input_pub=None):
    """Claim-vs-evidence conflicts backed by INDEPENDENT evidence only. Muse observations and location hypotheses never qualify."""
    out = []
    cd = claim_date_for_logic(claim)
    for m in proven_before(list(proven) + ([input_pub] if input_pub else []), cd, independent_only=True)[:1]:
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


def build_report(*, input_meta, claim, memory, visual, clues, physical, investigation_id=None):
    proven = visual.get("proven_matches", [])
    rep = dict(schema_version=SCHEMA_VERSION, investigation_id=investigation_id or uuid.uuid4().hex[:12],
               created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               input=input_meta, claim=claim, memory=memory, visual_matching=visual, clues=clues,
               physical_world_check=physical)
    rep["media_nature"] = nature_layer(clues)
    rep["contradictions"] = independent_contradictions(claim, proven, physical, input_publication(visual))
    rep["final_verdict"] = decide_verdict(claim, proven, clues, physical, input_meta, input_publication(visual), rep["media_nature"])
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
                                                          "screen_recording", "unknown"), "media_nature layer missing/invalid"
        assert mn.get("status") in ("AI_OBSERVATION_HYPOTHESIS", "NOT_RUN"), "media_nature must be labelled an AI observation"
        if mn["value"] != "unknown":
            assert mn.get("cited_frames"), "media_nature needs cited frames"
        cs = r["claim"].get("source")
        if cs is not None:
            for k in ("text", "event_date", "place"):
                if cs.get(k) == "inferred_from_post":
                    assert (r["input"] or {}).get("url"), "inferred claim requires a URL input"
            if cs.get("event_date") == "inferred_from_post":
                assert r["claim"].get("event_date_basis") in ("title", "description", "on_screen_text", "date_posted")
    assert r["memory"]["status"] in ("HIT", "MISS", "NOT_RUN")
    fv = r["final_verdict"]
    assert fv["label"] in LABELS and fv["summary"] and fv["limits"]
    if fv["label"] == "RECYCLED_FOOTAGE":
        cd = claim_date_for_logic(r["claim"])
        assert cd and proven_before(v["proven_matches"] + ([ip] if (ip := input_publication(v)) else []), cd, independent_only=True), \
            "RECYCLED requires a proven copy with an independently re-fetched date predating the claimed date"
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
