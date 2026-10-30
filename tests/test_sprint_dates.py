"""A1 (input's own publication is a separate finding, never an earlier copy) and A2 (time-zone margin for date-only verdicts)."""
import copy
import pytest
from floodproof.report import (build_report, validate_report, proven_before, date_precedes, input_before_claim, TZ_NOTE)
from test_report import proven, CLUES_OK


def rep(claim_date, matches=(), ip=None, claim_extra=None, basis=None, clues=CLUES_OK):
    visual = dict(status="FOUND" if matches else "NOT_FOUND", proven_matches=list(matches), input_publication=ip)
    claim = dict(text="Nepal flood", event_date=claim_date, **(claim_extra or {}))
    if basis:
        claim["event_date_basis"] = basis
    return build_report(input_meta=dict(source="x"), claim=claim, memory=dict(status="MISS"), visual=visual, clues=clues,
                        physical=dict(status="NOT_RUN"))


def ip(date, ts=None):
    return dict(url="https://youtu.be/in", upload_date=date, date_source="yt-dlp metadata", retrieved_at="t", **({"upload_timestamp": ts} if ts else {}))


# ---------------------------------------------------------------- A1
def test_input_publication_is_never_a_proven_copy_or_contradiction():
    r = validate_report(rep("2026-08-26", ip=ip("2024-09-28")))
    v = r["visual_matching"]
    assert v["proven_matches"] == [] and v.get("earliest_proven_copy") in (None, {}) and v["status"] == "NOT_FOUND"
    assert r["contradictions"] == []
    f = r["input_publication_before_claim"]
    assert f["finding"] == "input_publication_before_claim" and f["input_date"] == "2024-09-28" and f["claim_date"] == "2026-08-26"
    assert "posted before the event it is claimed to show" in f["text"]
    assert r["final_verdict"]["label"] == "RECYCLED_FOOTAGE"
    assert r["final_verdict"]["basis"]["earlier_proven_copies"] == 0 and r["final_verdict"]["basis"]["input_posted_before_claim"] is True
    assert r["headline"]["sentence"] == "This video was posted on 2024-09-28, before the claimed event on 2026-08-26."
    assert "earlier copy" not in r["headline"]["sentence"].lower()


def test_input_publication_not_a_matcher_signal_and_not_independent_support():
    r = validate_report(rep("2026-08-26", ip=ip("2024-09-28")))
    sig = r["signals"]
    assert not any(a["key"] == "no_earlier_copy" for a in sig["agree"]) and not sig["qualifies"]
    assert any("posted on 2024-09-28" in a for a in sig["against"])


def test_validator_rejects_input_url_as_proven_match():
    r = validate_report(rep("2026-08-26", ip=ip("2024-09-28")))
    bad = copy.deepcopy(r)
    bad["visual_matching"]["proven_matches"] = [dict(proven(), adapter="input_url", url="https://youtu.be/in")]
    bad["visual_matching"]["status"] = "FOUND"
    with pytest.raises(AssertionError):
        validate_report(bad)
    bad = copy.deepcopy(r)
    bad["input_publication_before_claim"]["input_date"] = "2020-01-01"
    with pytest.raises(AssertionError):
        validate_report(bad)


def test_real_earlier_copy_wins_and_input_finding_is_separate():
    r = validate_report(rep("2026-08-26", [proven(date="2024-01-01")], ip=ip("2024-09-28")))
    assert r["final_verdict"]["label"] == "RECYCLED_FOOTAGE"
    assert r["headline"]["sentence"].startswith("This footage was already published on 2024-01-01")
    assert r["visual_matching"]["proven_matches"][0]["adapter"] == "memory"
    assert "Separately, this video was posted on 2024-09-28" in r["final_verdict"]["summary"]


def test_input_posted_after_or_on_claim_is_no_finding():
    for d in ("2026-08-26", "2026-08-27", "2026-09-30"):
        r = validate_report(rep("2026-08-26", ip=ip(d)))
        assert "input_publication_before_claim" not in r and r["final_verdict"]["label"] != "RECYCLED_FOOTAGE"


def test_date_posted_claim_never_triggers_input_finding():
    r = validate_report(rep("2026-08-28", ip=ip("2026-08-28"), basis="date_posted"))
    assert "input_publication_before_claim" not in r


def test_consistent_from_input_alone_is_not_called_independent():
    r = validate_report(rep("2026-08-26", ip=ip("2026-08-26")))
    fv = r["final_verdict"]
    assert fv["label"] == "CONSISTENT_WITH_CLAIM" and "not independent corroboration" in fv["summary"]


# ---------------------------------------------------------------- A2
def test_date_only_needs_two_days():
    assert date_precedes("2026-08-24", "2026-08-26")
    assert not date_precedes("2026-08-25", "2026-08-26")          # one day: time-zone uncertainty
    assert not date_precedes("2026-08-26", "2026-08-26")
    assert not proven_before([proven(date="2026-08-25")], "2026-08-26")
    assert proven_before([proven(date="2026-08-24")], "2026-08-26")


def test_clip_posted_hours_after_a_nepal_event_is_unverified():
    # Nepal flood on 2026-08-26 (UTC+5:45). A copy uploaded 2026-08-25 20:30 UTC is 02:15 on the 26th in Nepal: after the event began.
    m = proven(date="2026-08-25")
    r = validate_report(rep("2026-08-26", [m]))
    fv = r["final_verdict"]
    assert fv["label"] == "UNVERIFIED" and TZ_NOTE in fv["summary"] and fv["basis"]["unverified_reason"] == TZ_NOTE
    assert r["contradictions"] == []
    m["upload_timestamp"] = "2026-08-25T20:30:00Z"
    assert validate_report(rep("2026-08-26", [m]))["final_verdict"]["label"] == "UNVERIFIED"
    assert not date_precedes("2026-08-25", "2026-08-26", "2026-08-25T20:30:00Z")


def test_exact_timestamp_allows_a_close_but_certainly_earlier_copy():
    # 2026-08-25 06:00 UTC is before the 26th began anywhere (UTC+14: 08-25 10:00 UTC), so it counts even though the date is one day before.
    assert date_precedes("2026-08-25", "2026-08-26", "2026-08-25T06:00:00Z")
    m = proven(date="2026-08-25")
    m["upload_timestamp"] = "2026-08-25T06:00:00Z"
    assert validate_report(rep("2026-08-26", [m]))["final_verdict"]["label"] == "RECYCLED_FOOTAGE"


def test_input_posted_within_a_day_of_claim_needs_margin():
    r = validate_report(rep("2026-08-26", ip=ip("2026-08-25")))
    assert r["final_verdict"]["label"] == "UNVERIFIED" and TZ_NOTE in r["final_verdict"]["summary"]
    assert "input_publication_before_claim" not in r
    r2 = validate_report(rep("2026-08-26", ip=ip("2026-08-24")))
    assert r2["final_verdict"]["label"] == "RECYCLED_FOOTAGE"
    assert input_before_claim(ip("2026-08-25", "2026-08-25T06:00:00Z"), dict(event_date="2026-08-26"))


def test_one_day_before_is_not_consistent_either():
    m = proven(date="2026-08-25")
    assert validate_report(rep("2026-08-26", [m]))["final_verdict"]["label"] == "UNVERIFIED"
