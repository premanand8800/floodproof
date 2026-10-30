"""Report schema + verdict rules. No media needed: proven copies are synthetic dicts shaped like pipeline output."""
import copy
import pytest
from floodproof.report import build_report, validate_report, decide_verdict, proven_before, LABELS


def proven(url="https://youtu.be/orig", date="2024-09-28", alt=None, src="yt-dlp metadata"):
    m = dict(url=url, upload_date=date, date_source=src, retrieved_at="2026-10-30T00:00:00+00:00", adapter="memory",
             match=dict(consistent_frames=40, max_inliers=300, temporal_offset=0.0, mirrored=False,
                        best_pairs=[dict(a_frame=0, b_frame=0, inliers=300, flip=False)]), evidence_png="evidence/x.png")
    if alt:
        m["date_alternatives"] = alt
    return m


CLUES_OK = dict(status="ok", contradictions_with_claim=[], likely_location=dict(place="x", country="y", confidence=.5),
                scene_type=dict(value="river"), queries=[])


def rep(claim_date, matches, phys=None, clues=CLUES_OK):
    visual = dict(status="FOUND" if matches else "NOT_FOUND", proven_matches=matches)
    return build_report(input_meta=dict(source="x"), claim=dict(text="Nepal flood", event_date=claim_date),
                        memory=dict(status="MISS"), visual=visual, clues=clues,
                        physical=phys or dict(status="NOT_RUN"))


def test_recycled_only_when_proven_copy_predates_claim():
    r = validate_report(rep("2026-08-26", [proven(date="2024-09-28")]))
    assert r["final_verdict"]["label"] == "RECYCLED_FOOTAGE"
    assert "does not" in r["final_verdict"]["summary"] and "fake" in r["final_verdict"]["summary"]


def test_copy_after_claim_is_not_recycled():
    r = validate_report(rep("2024-09-28", [proven(date="2024-09-29")]))
    assert r["final_verdict"]["label"] == "UNVERIFIED"
    assert "not established" in r["final_verdict"]["summary"]


def test_same_day_is_not_strictly_earlier():
    assert not proven_before([proven(date="2026-08-26")], "2026-08-26")


def test_no_claim_date_never_recycled():
    assert validate_report(rep(None, [proven()]))["final_verdict"]["label"] == "UNVERIFIED"


def test_insufficient_evidence_is_unverified():
    r = validate_report(rep("2026-08-26", [], dict(status="NOT_RUN"), clues=dict(status="not_run")))
    fv = r["final_verdict"]
    assert fv["label"] == "UNVERIFIED" and r["visual_matching"]["status"] == "NOT_FOUND"
    assert "No matcher-proven earlier copy" in fv["summary"] and fv["limits"]


def test_physical_support_alone_gives_consistent_with_claim_wording():
    r = validate_report(rep("2026-08-26", [], dict(verdict="SUPPORTS")))
    fv = r["final_verdict"]
    assert fv["label"] == "CONSISTENT_WITH_CLAIM"
    assert "no contradiction was found" in fv["summary"] and "not a finding that the footage is authentic" in fv["summary"]


def test_physical_contradiction_or_unknown_is_unverified():
    for p in (dict(verdict="CONTRADICTS"), dict(verdict="NO_EVIDENCE"), dict(verdict="UNKNOWN")):
        assert rep("2026-08-26", [], p)["final_verdict"]["label"] == "UNVERIFIED"


def test_clue_contradiction_blocks_consistent_and_is_not_merged_into_visual():
    cl = dict(CLUES_OK, contradictions_with_claim=[dict(text="Alaska glacier, not Nepal", refs=[3])])
    r = validate_report(rep("2026-08-26", [], dict(verdict="SUPPORTS"), clues=cl))
    assert r["final_verdict"]["label"] == "UNVERIFIED"
    assert "model observations" in r["final_verdict"]["summary"]
    assert r["visual_matching"]["status"] == "NOT_FOUND"        # layers stay separate


def test_genuine_footage_correct_claim_not_recycled():
    """Genuine clip, correct claim date: its own earlier-or-same-day uploads must not trigger RECYCLED."""
    r = validate_report(rep("2024-09-28", [proven(date="2024-09-28"), proven(url="https://x.com/a", date="2024-09-29")]))
    assert r["final_verdict"]["label"] != "RECYCLED_FOOTAGE"


def test_genuine_footage_wrong_claim_shows_context_mismatch_without_calling_footage_fake():
    r = validate_report(rep("2026-08-26", [proven(date="2024-09-28")]))
    s = r["final_verdict"]["summary"]
    assert r["final_verdict"]["label"] == "RECYCLED_FOOTAGE"
    assert "2024-09-28" in s and "does not by itself show the footage is fake" in s


def test_misleading_timestamp_claim_vs_real_date():
    """Footage genuinely from 2024 presented with an Aug-2026 date: the report must show the 2024 copy date."""
    r = rep("2026-08-26", [proven(date="2024-09-28")])
    assert r["final_verdict"]["basis"]["earlier_proven_copies"] == 1
    assert r["visual_matching"]["proven_matches"][0]["upload_date"] == "2024-09-28"


def test_conflicting_sources_are_both_reported_and_not_silently_resolved():
    alt = [dict(date="2026-09-01", source="page text")]
    r = validate_report(rep("2026-08-26", [proven(date="2024-09-28", alt=alt)]))
    fv = r["final_verdict"]
    assert fv["label"] == "UNVERIFIED"                       # a conflict that straddles the claim date does not prove recycling
    assert "2024-09-28" in fv["summary"] and "2026-09-01" in fv["summary"] and "disagree" in fv["summary"]
    assert fv["basis"]["date_conflicts"] == 1
    # conflict that does not straddle: both dates still shown, verdict uses the later one
    r2 = rep("2026-08-26", [proven(date="2024-09-28", alt=[dict(date="2025-01-01", source="page text")])])
    assert r2["final_verdict"]["label"] == "RECYCLED_FOOTAGE" and "2025-01-01" in r2["final_verdict"]["summary"]


def test_schema_rejects_bad_reports():
    good = rep("2026-08-26", [proven()])
    for mut in (lambda r: r.pop("clues"), lambda r: r["visual_matching"].update(status="NOT_FOUND"),
                lambda r: r["final_verdict"].update(label="FAKE"),
                lambda r: r["visual_matching"]["proven_matches"][0].pop("date_source")):
        bad = copy.deepcopy(good); mut(bad)
        with pytest.raises(AssertionError):
            validate_report(bad)
    bad = copy.deepcopy(good); bad["claim"]["event_date"] = "2020-01-01"
    with pytest.raises(AssertionError):   # RECYCLED without a copy predating the claim
        validate_report(bad)
    assert set(LABELS) == {"RECYCLED_FOOTAGE", "CONSISTENT_WITH_CLAIM", "UNVERIFIED"}


def test_physical_layer_not_run_when_module_missing(monkeypatch):
    import sys
    import floodproof
    from floodproof import report
    monkeypatch.delattr(floodproof, "physical", raising=False)
    monkeypatch.setitem(sys.modules, "floodproof.physical", None)
    p = report.run_physical("Nepal", None, None, "2026-08-26")
    assert p["status"] == "NOT_RUN"


def test_physical_interface_is_called_with_new_signature(monkeypatch):
    import types, sys
    seen = {}
    fake = types.ModuleType("floodproof.physical")
    fake.check = lambda place=None, lat=None, lon=None, date="", window_days=3, event_type="flood", country=None: seen.update(
        place=place, date=date, event_type=event_type, country=country, window_days=window_days) or dict(verdict="SUPPORTS")
    import floodproof
    monkeypatch.setattr(floodproof, "physical", fake, raising=False)
    monkeypatch.setitem(sys.modules, "floodproof.physical", fake)
    from floodproof import report
    out = report.run_physical("Kathmandu", None, None, "2024-09-28", 3, "flash_flood", "NP")
    assert out["verdict"] == "SUPPORTS" and seen["event_type"] == "flash_flood" and seen["country"] == "NP"


def test_input_url_own_publication_date_counts_as_previously_published():
    visual = dict(status="NOT_FOUND", proven_matches=[], input_publication=dict(
        url="https://youtu.be/in", upload_date="2024-09-28", date_source="yt-dlp metadata", retrieved_at="t"))
    r = build_report(input_meta={}, claim=dict(text="Nepal flood", event_date="2026-08-26"), memory=dict(status="MISS"),
                     visual=visual, clues=dict(status="not_run"), physical=dict(status="NOT_RUN"))
    validate_report(r)
    assert r["final_verdict"]["label"] == "RECYCLED_FOOTAGE" and "itself that publication" in r["final_verdict"]["summary"]
    r["claim"]["event_date"] = "2024-09-28"
    r2 = build_report(input_meta={}, claim=r["claim"], memory=dict(status="MISS"), visual=visual,
                      clues=dict(status="not_run"), physical=dict(status="NOT_RUN"))
    assert r2["final_verdict"]["label"] == "UNVERIFIED"
