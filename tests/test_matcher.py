import pytest
from floodproof.matcher import match, decide, phash_distance

# saved pairs from bench/true_pairs.json that the validated engine proves
TRUE_PAIRS = [("x_iambmane", "ig_leigh"), ("ig_sainj", "yts_sainj"), ("yt_tauktae1", "yt_tauktae2")]
# hard negatives from bench/neg_pairs.json (same genre, different footage)
NEG_PAIRS = [("ig_leigh", "yt_ie_nalagarh"), ("x_iambmane", "yt_croc"), ("yt_hum", "yts_sainj")]


def test_decision_rule():
    assert decide(2, 0) and decide(5, 20)
    assert decide(1, 60) and not decide(1, 59)
    assert not decide(0, 0) and not decide(1, 30)


@pytest.mark.parametrize("a,b", TRUE_PAIRS)
def test_true_pairs_match(feats, a, b):
    r = match(feats(a), feats(b))
    assert r.decision and (r.consistent_frames >= 2 or r.max_inliers >= 60)
    assert not r.mirrored


@pytest.mark.parametrize("a,b", NEG_PAIRS)
def test_hard_negatives_do_not_match(feats, a, b):
    r = match(feats(a), feats(b), coarse=12)
    assert not r.decision


@pytest.mark.parametrize("name", ["crop70", "mirror", "overlay", "lowres", "combo"])
def test_near_duplicates_match(feats, variant, name):
    """Same video at a different compression/resolution/crop/caption/mirror must still match."""
    r = match(feats("yt_leigh"), variant("yt_leigh", name))
    assert r.decision, r.to_dict()
    assert r.mirrored == (name in ("mirror", "combo"))


def test_evidence_png(feats, tmp_path):
    r = match(feats("ig_sainj"), feats("yts_sainj"))
    p = r.evidence(tmp_path / "e.png", "t")
    assert p and (tmp_path / "e.png").stat().st_size > 5000
    d = r.to_dict()
    assert d["best_pairs"] and d["temporal_offset"] is not None
