import pytest
from floodproof.memory import Memory


def test_round_trip_and_origin(feats, variant, tmp_path):
    mem = Memory(tmp_path / "m")
    orig = feats("yt_leigh")
    eid = mem.add(orig, "https://www.youtube.com/watch?v=E5TOaoetGDo", "2025-08-16", dict(note="ctx"), "bench:C07",
                  "Leigh Coates", None, "fact_check_cited_original", origin="seed:benchmark")
    mem.add(feats("yts_sainj"), "https://www.youtube.com/shorts/k95Yfi__4h0", "2024-09-09", {}, "inv123", role="investigated_input")
    st = mem.stats()
    assert st["entries"] == 2 and st["frames"] > 20
    assert {e["origin"] for e in mem.entries()} == {"seed:benchmark", "discovered:inv123"}
    for name in ("mirror", "crop70"):                      # a re-encoded/cropped/mirrored copy is found AND proven
        res = mem.lookup(variant("yt_leigh", name))
        assert res.hit and res.hits[0].entry["id"] == eid
        assert res.hits[0].entry["event_context"] == dict(note="ctx")
        assert res.hits[0].match.decision
    # contamination control: search only the discovered part -> original must not be returned
    assert not mem.lookup(variant("yt_leigh", "mirror"), origin_prefix="discovered:").hit


def test_unrelated_clip_is_a_miss(feats, tmp_path):
    mem = Memory(tmp_path / "m")
    mem.add(feats("yt_leigh"), "u1", "2025-08-16", role="manual")
    assert not mem.lookup(feats("yt_croc")).hit


def test_exhaustive_matches_without_hash_shortlist(feats, tmp_path):
    mem = Memory(tmp_path / "m")
    eid = mem.add(feats("ig_sainj"), "u1", "2024-09-08")
    assert mem.lookup(feats("yts_sainj"), exhaustive=True).hits[0].entry["id"] == eid


def test_replace_same_key(feats, tmp_path):
    mem = Memory(tmp_path / "m")
    a = mem.add(feats("yt_leigh"), "u1"); b = mem.add(feats("yt_leigh"), "u1")
    assert mem.stats()["entries"] == 1 and a != b
