"""Single-frame-only matches are 'weak_match': shown, but a landmark-scene weak match cannot yield RECYCLED_FOOTAGE."""
import copy
from floodproof import matcher
from floodproof.report import validate_report
from test_report import proven, rep, CLUES_OK


def weak(**kw):
    m = proven(**kw)
    m["match"].update(consistent_frames=1, max_inliers=73, weak_match=True)
    return m


def test_matcher_weak_flag():
    r = matcher.MatchResult(True, 1, 70, False, 0.0, 3, 1.7, [])
    assert r.weak and r.to_dict()["weak_match"] is True
    assert not matcher.MatchResult(True, 2, 20, False, 0.0, 3, 2.2, []).weak
    assert not matcher.MatchResult(False, 1, 30, False, 0.0, 3, 1.3, []).weak


def test_weak_match_with_independent_date_non_landmark_still_recycled():
    r = validate_report(rep("2026-08-26", [weak(date="2024-09-28")]))
    assert r["final_verdict"]["label"] == "RECYCLED_FOOTAGE"


def test_weak_match_on_landmark_scene_is_not_recycled():
    clues = copy.deepcopy(CLUES_OK); clues["scene_type"] = dict(value="Eiffel Tower and city skyline")
    r = validate_report(rep("2026-08-26", [weak(date="2024-09-28")], clues=clues))
    assert r["final_verdict"]["label"] == "UNVERIFIED"
    assert r["visual_matching"]["proven_matches"][0]["match"]["weak_blocked"]


def test_strong_match_on_landmark_scene_unaffected():
    clues = copy.deepcopy(CLUES_OK); clues["scene_type"] = dict(value="Eiffel Tower")
    assert rep("2026-08-26", [proven(date="2024-09-28")], clues=clues)["final_verdict"]["label"] == "RECYCLED_FOOTAGE"
