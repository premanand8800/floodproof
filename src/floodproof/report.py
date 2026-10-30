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
SCHEMA_VERSION = "1.0"
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
        return d if isinstance(d, dict) else dict(status="ERROR", reason="physical.check did not return a dict")
    except Exception as e:
        return dict(status="ERROR", reason=str(e)[:200])


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


def proven_before(proven, claim_date):
    """Matcher-proven copies strictly earlier than the claimed event date. If sources disagree on a copy's date, ALL
    reported dates must predate the claim (conservative: a conflict that straddles the claim date does not count)."""
    if not claim_date:
        return []
    ok = [m for m in proven if all_dates(m) and max(all_dates(m)) < claim_date]
    return sorted(ok, key=lambda m: min(all_dates(m)))


def input_publication(visual):
    """The input URL's own publication record, usable as 'previously published' evidence (it is the footage itself)."""
    ip = (visual or {}).get("input_publication")
    if ip and ip.get("upload_date"):
        return dict(ip, adapter="input_url", match=dict(identity=True, consistent_frames=None, max_inliers=None,
                                                        temporal_offset=0.0, mirrored=False, best_pairs=[]))
    return None


def decide_verdict(claim, proven, clues, physical, input_meta=None, input_pub=None):
    claim_date = (claim or {}).get("event_date")
    proven_real = list(proven)           # matcher-proven copies of OTHER uploads
    if input_pub:
        proven = proven_real + [input_pub]   # the input's own publication also counts as 'previously published' evidence
    limits = [
        "A visual match proves shared footage ('previously published'), not that a claim is false or that the footage is fake.",
        "Upload dates are platform metadata for the copy found; the earliest copy found is not necessarily the true source.",
        "Search is not exhaustive: memory covers only indexed clips, reverse search rarely finds originals, and heavy crops/edits can evade hash shortlisting.",
        "'Not found' means not found by these adapters, not that no earlier copy exists.",
    ]
    earlier = proven_before(proven, claim_date)
    conflicts = date_conflicts(proven)
    contrad = [c for c in (clues or {}).get("contradictions_with_claim", [])] if (clues or {}).get("status") == "ok" else []
    phys = physical_supports(physical)
    ev = []   # evidence actually present, cited in the summary
    if earlier:
        e = earlier[0]
        label = "RECYCLED_FOOTAGE"
        ev_txt = ("The input is itself that publication." if e["match"].get("identity") else
                  f"Matcher evidence: {e['match']['consistent_frames']} temporally consistent frame(s), up to {e['match']['max_inliers']} inliers"
                  f"{', mirrored' if e['match'].get('mirrored') else ''}.")
        s = (f"The footage matches a copy published on {e['upload_date']} ({e['url']}, date per {e.get('date_source')}), before the claimed event date {claim_date}. "
             f"{ev_txt} This shows the footage was previously published; it does not "
             "by itself show the footage is fake or that the claimed event did not happen.")
    else:
        if input_pub and not proven_real:
            s = (f"The input URL was itself published on {input_pub['upload_date']} (per {input_pub.get('date_source')}), which is not earlier "
                 "than the claimed event date, so recycling relative to the claim is not established. ")
        elif proven_real:
            e = min([m for m in proven_real if m.get("upload_date")] or proven_real, key=lambda m: m.get("upload_date") or "9999")
            s = (f"The footage matches a previously published copy ({e['url']}"
                 f"{', uploaded ' + e['upload_date'] if e.get('upload_date') else ', upload date unavailable'}), but "
                 + ("that date is not earlier than the claimed event date" if claim_date and e.get("upload_date")
                    else "no claimed event date was given" if not claim_date else "its date is unknown")
                 + ", so recycling relative to the claim is not established. ")
        else:
            s = "No matcher-proven earlier copy was found by memory lookup, reverse image search or text search. "
        if phys is True and not contrad and not earlier:
            label = "CONSISTENT_WITH_CLAIM"
            s += ("The physical-world check supports the claimed place/date and no cited contradiction was found. This means no "
                  "contradiction was found in the available evidence; it is not a finding that the footage is authentic.")
        else:
            label = "UNVERIFIED"
            if phys is False:
                s += "The physical-world check did not support the claimed place/date. "
            elif phys is None:
                s += "The physical-world check was not available or inconclusive. "
            s += "The claim could not be verified from the available evidence."
    if conflicts:
        s += " Sources disagree on the publication date of: " + "; ".join(
            c["url"] + " (" + ", ".join(f"{d['date']} per {d['source']}" for d in c["dates"]) + ")" for c in conflicts) + \
            ". Both dates are reported; the verdict uses the later one."
    if contrad:
        s += (" Separately, the AI visual clues cite " + str(len(contrad)) + " cue(s) that argue against the claim (e.g. \""
              + contrad[0]["text"][:140] + "\", frame ref " + str(contrad[0]["refs"]) + "); these are model observations, not part of the visual-match proof.")
    return dict(label=label, summary=s, limits=limits, basis=dict(
        earlier_proven_copies=len(earlier), proven_copies=len(proven_real), physical_supports=phys, cited_contradictions=len(contrad), date_conflicts=len(conflicts)))


def build_report(*, input_meta, claim, memory, visual, clues, physical, investigation_id=None):
    proven = visual.get("proven_matches", [])
    rep = dict(schema_version=SCHEMA_VERSION, investigation_id=investigation_id or uuid.uuid4().hex[:12],
               created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               input=input_meta, claim=claim, memory=memory, visual_matching=visual, clues=clues,
               physical_world_check=physical)
    rep["final_verdict"] = decide_verdict(claim, proven, clues, physical, input_meta, input_publication(visual))
    return rep


def validate_report(r: dict):
    """Raise AssertionError if the report violates the schema/rules; returns r."""
    for k in REQUIRED_KEYS:
        assert k in r, f"missing {k}"
    v = r["visual_matching"]
    assert v["status"] in ("FOUND", "NOT_FOUND")
    assert "physical_world_check" in r and isinstance(r["physical_world_check"], dict)
    assert (v["status"] == "FOUND") == bool(v.get("proven_matches")), "FOUND iff proven matches present"
    for m in v.get("proven_matches", []):
        for k in ("url", "upload_date", "date_source", "retrieved_at", "adapter", "match"):
            assert k in m, f"proven match missing {k}"
        for k in ("consistent_frames", "max_inliers", "temporal_offset", "best_pairs"):
            assert k in m["match"], f"match evidence missing {k}"
    assert r["memory"]["status"] in ("HIT", "MISS", "NOT_RUN")
    fv = r["final_verdict"]
    assert fv["label"] in LABELS and fv["summary"] and fv["limits"]
    if fv["label"] == "RECYCLED_FOOTAGE":
        cd = r["claim"].get("event_date")
        assert cd and proven_before(v["proven_matches"] + ([ip] if (ip := input_publication(v)) else []), cd), "RECYCLED requires a proven copy predating the claimed date"
    return r


def write_report(r: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "report.json"
    p.write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    return p


def render_text(r: dict) -> str:
    v, fv, c = r["visual_matching"], r["final_verdict"], r["claim"]
    L = [f"FloodProof investigation {r['investigation_id']}",
         f"Claim: {c.get('text')} (event date {c.get('event_date')})",
         f"Memory: {r['memory']['status']}   Visual matching: {v['status']} ({len(v.get('proven_matches', []))} proven)"]
    for m in v.get("proven_matches", []):
        L.append(f"  - [{m['adapter']}] {m['url']} uploaded {m['upload_date']} ({m['date_source']}); "
                 f"{m['match']['consistent_frames']} frames, {m['match']['max_inliers']} inliers; evidence: {m.get('evidence_png')}")
    ep = v.get("earliest_proven_copy")
    L.append(f"Earliest proven copy: {ep['url'] + ' ' + str(ep['upload_date']) if ep else 'none'}   Claimed original: {(v.get('claimed_original') or {}).get('url')}")
    cl = r["clues"]
    if cl.get("status") == "ok":
        ll = cl["likely_location"]
        L.append(f"Muse clues: location guess {ll.get('place')}, {ll.get('country')} (conf {ll.get('confidence')}); "
                 f"{len(cl['contradictions_with_claim'])} contradiction(s) cited; scene: {(cl.get('scene_type') or {}).get('value')}")
    else:
        L.append(f"Muse clues: {cl.get('status')}")
    L.append(f"Physical-world check: {r['physical_world_check'].get('verdict') or r['physical_world_check'].get('status')}")
    L.append(f"VERDICT: {fv['label']}\n{fv['summary']}\nLimits:\n" + "\n".join(f"  * {x}" for x in fv["limits"]))
    return "\n".join(L)
