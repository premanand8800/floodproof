"""Clue hygiene (no network): every clue needs a valid frame/timestamp reference, uncited location guesses are rejected."""
from types import SimpleNamespace
from floodproof.investigator import _clean, _valid_refs

class _F:
    times = [0.0, 1.5, 3.0, 4.5]
    frames = [None] * 4

    def __len__(self):
        return 4


F = _F()


def test_refs_validation():
    assert _valid_refs([0, 2, 9, "3", True], 4, False) == [0, 2, 3]
    assert _valid_refs(["audio@12.5"], 4, True) == ["audio@12.5"]
    assert _valid_refs(["audio@12.5"], 4, False) == []


def test_uncited_clues_are_rejected_not_reported():
    plan = dict(shows=[dict(text="river", refs=[1]), dict(text="no ref")],
                scene_type=dict(value="glacier", refs=[]),
                visible_text=[dict(text="BREAKING", language="en", refs=[2])],
                likely_location=dict(place="Alaska", country="US", confidence=.9, reasons=[dict(text="guess", refs=[])]),
                contradictions_with_claim=[dict(text="no Nepali signage", refs=[0, 3])],
                claim_parsed=dict(place="Nepal"), queries=[dict(q="q1"), dict(nope=1)])
    out = _clean(plan, F, False)
    assert [s["text"] for s in out["shows"]] == ["river"] and out["shows"][0]["timestamps_s"] == [1.5]
    assert out["scene_type"] is None
    assert out["likely_location"]["place"] is None and out["likely_location"]["uncited_guess"]["place"] == "Alaska"
    assert out["observations_against_claim"][0]["timestamps_s"] == [0.0, 4.5]
    assert out["observations_against_claim"][0]["status"] == "UNVERIFIED_MODEL_OBSERVATION" and "contradictions_with_claim" not in out
    assert out["location_hypotheses"] == []        # uncited guess is rejected, not a hypothesis
    assert len(out["queries"]) == 1
    assert {r["kind"] for r in out["rejected_clues"]} >= {"shows", "scene_type", "likely_location"}


def test_cited_location_guess_becomes_unverified_hypothesis():
    plan = dict(likely_location=dict(place="Alaska", country="US", confidence=.4, reasons=[dict(text="black ash glacier", refs=[1, 3])]))
    out = _clean(plan, F, False)
    h = out["location_hypotheses"][0]
    assert h["status"] == "UNVERIFIED_HYPOTHESIS" and h["cited_frames"] == [1, 3] and h["confidence"] == .4
