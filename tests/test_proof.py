"""Clean proof frames, RANSAC inlier coordinates, Muse explanation (mocked) and schema 1.5 validation."""
import copy
import json

import cv2
import numpy as np
import pytest

from floodproof import explain
from floodproof.matcher import ClipFeatures, match, proof_artifacts, proof_inliers
from floodproof.pipeline import attach_explanation
from floodproof.report import build_report, validate_report


def scene(seed=0, w=360, h=240):
    rng = np.random.RandomState(seed)
    img = np.full((h, w, 3), 120, np.uint8)
    for _ in range(90):
        c = tuple(int(x) for x in rng.randint(0, 255, 3))
        p = (int(rng.randint(0, w)), int(rng.randint(0, h)))
        if rng.rand() < .5:
            cv2.circle(img, p, int(rng.randint(4, 25)), c, -1)
        else:
            cv2.rectangle(img, p, (p[0] + int(rng.randint(6, 40)), p[1] + int(rng.randint(6, 40))), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def clip(frames):
    return ClipFeatures.from_frames("x", [float(i) for i in range(len(frames))], frames)


@pytest.fixture(scope="module")
def pair():
    a = [scene(s) for s in (1, 2, 3)]
    b = [cv2.convertScaleAbs(f, alpha=1.1, beta=8) for f in a]       # same footage, different brightness
    return clip(a), clip(b)


def test_inliers_normalised_and_capped(pair):
    A, B = pair
    inl = proof_inliers(A.frames[0], B.frames[0])
    assert 10 <= len(inl) <= 40
    assert all(len(q) == 4 and all(0.0 <= x <= 1.0 for x in q) for q in inl)
    # identical-geometry copy: matched points sit at (almost) the same normalised place
    assert np.mean([abs(q[0] - q[2]) + abs(q[1] - q[3]) for q in inl]) < 0.02


def test_mirrored_inliers_map_back_to_unflipped_frame(pair):
    A, B = pair
    flipped = clip([cv2.flip(f, 1) for f in B.frames])
    r = match(A, flipped)
    assert r.decision and r.mirrored
    inl = proof_inliers(A.frames[r.pairs[0][0]], flipped.frames[r.pairs[0][1]], flip=True)
    assert inl and all(0 <= x <= 1 for q in inl for x in q)
    assert np.mean([abs((1 - q[0]) - q[2]) for q in inl]) < 0.03 or np.mean([abs(q[0] - (1 - q[2])) for q in inl]) < 0.03


def test_proof_artifacts_clean_frames(pair, tmp_path):
    A, B = pair
    r = match(A, B)
    assert r.decision
    p = proof_artifacts(r, tmp_path / "ev", "t")
    for k in ("viral_frame", "copy_frame"):
        im = cv2.imread(p[k]); assert im is not None and im.shape[:2] == A.frames[0].shape[:2]
    i, j = r.pairs[0][:2]
    # clean: no lines drawn, the saved JPG equals the stored frame (up to JPEG loss)
    assert np.abs(cv2.imread(p["viral_frame"]).astype(int) - A.frames[i].astype(int)).mean() < 6
    assert p["viral_t"] == A.times[i] and p["copy_t"] == B.times[j] and p["inliers"]


def test_no_proof_when_not_proven(pair, tmp_path):
    A, _ = pair
    r = match(A, clip([scene(99), scene(98)]))
    assert not r.decision and proof_artifacts(r, tmp_path, "n") is None


# ---------------------------------------------------------------- explanation (LLM mocked)
GOOD = json.dumps({"items": [{"what": "the same red-roofed building", "region_a": "top-left", "region_b": "top-left"},
                             {"what": "identical river bend", "region_a": "center", "region_b": "center"}],
                   "differences": ["B is cropped at the right", "B has a caption bar"]})


def test_explain_proof_mocked(pair, tmp_path):
    A, B = pair
    p = proof_artifacts(match(A, B), tmp_path, "t")
    seen = {}

    def fake(model, content, max_tokens=0):
        seen["model"] = model; seen["images"] = sum(c["type"] == "image_url" for c in content)
        return "```json\n" + GOOD + "\n```", 0.012
    ex = explain.explain_proof(p["viral_frame"], p["copy_frame"], chat_fn=fake)
    assert seen == {"model": "meta/muse-spark-1.3", "images": 2}
    assert ex["model"] == "meta/muse-spark-1.3" and len(ex["items"]) == 2 and ex["differences"] and ex["cost_usd"] == 0.012
    assert "not this text" in ex["label"]


@pytest.mark.parametrize("reply", ["not json", '{"items": []}', "{}"])
def test_explanation_is_optional_on_bad_reply(pair, tmp_path, reply):
    A, B = pair
    p = proof_artifacts(match(A, B), tmp_path, "t")
    assert explain.explain_proof(p["viral_frame"], p["copy_frame"], chat_fn=lambda *a, **k: (reply, 0.0)) is None


def test_explanation_is_optional_on_llm_failure(pair, tmp_path):
    A, B = pair
    p = proof_artifacts(match(A, B), tmp_path, "t")

    def boom(*a, **k):
        raise RuntimeError("OpenRouter down")
    assert explain.explain_proof(p["viral_frame"], p["copy_frame"], chat_fn=boom) is None


# ---------------------------------------------------------------- schema 1.5
def entry(url, date, proof=None):
    m = dict(url=url, upload_date=date, date_source="yt-dlp live metadata", retrieved_at="t", adapter="memory",
             independent_date=date, independent_date_source="yt-dlp live metadata", date_independent=True,
             match=dict(consistent_frames=9, max_inliers=120, temporal_offset=0.0, mirrored=False, best_pairs=[]))
    if proof is not None:
        m["proof"] = proof
    return m


def proof_block(**kw):
    p = dict(viral_frame="evidence/v.jpg", copy_frame="evidence/c.jpg", viral_t=1.0, copy_t=1.3, inliers=[[.1, .2, .12, .21], [.9, .8, .91, .79]])
    p.update(kw)
    return p


def report(*entries):
    return build_report(input_meta=dict(source="x"), claim=dict(text="Nepal flood", event_date="2026-08-26"), memory=dict(status="MISS"),
                        visual=dict(status="FOUND", proven_matches=list(entries)), clues=dict(status="not_run"), physical=dict(status="NOT_RUN"))


def test_schema_15_and_backward_compat():
    r = report(entry("https://y/1", "2026-08-21"))                      # older style: no proof block at all
    assert r["schema_version"] == "1.5" and validate_report(r)
    ex = dict(items=[dict(what="x", region_a="top-left", region_b="top-left")], differences=[], model="meta/muse-spark-1.3")
    assert validate_report(report(entry("https://y/1", "2026-08-21", proof_block())))               # proof without explanation
    assert validate_report(report(entry("https://y/1", "2026-08-21", proof_block(explanation=ex))))


@pytest.mark.parametrize("bad", [dict(inliers=[[.1, .2, 1.5, .3]]), dict(inliers=[[.1, .2, .3]]), dict(inliers=[[.1, .2, .3, .4]] * 41),
                                 dict(viral_frame="/etc/passwd"), dict(copy_frame="../x.jpg"), dict(viral_t="a"),
                                 dict(explanation=dict(items=[], differences=[], model="m")),
                                 dict(explanation=dict(items=[dict(what="x")], differences=[]))])
def test_schema_rejects_bad_proof(bad):
    with pytest.raises(AssertionError):
        validate_report(report(entry("https://y/1", "2026-08-21", proof_block(**bad))))


def test_attach_explanation_only_earliest_one_call(tmp_path):
    es = [entry("https://y/old", "2026-08-21", proof_block()), entry("https://y/new", "2026-08-24", proof_block())]
    calls = []
    ex = dict(items=[dict(what="x", region_a="a", region_b="b")], differences=[], model="m", cost_usd=0.01)
    got = attach_explanation(es, "https://y/old", tmp_path, explain_fn=lambda v, c: (calls.append((str(v), str(c))), ex)[1], log=lambda *_: None)
    assert got == ex and len(calls) == 1 and es[0]["proof"]["explanation"] == ex and "explanation" not in es[1]["proof"]
    assert calls[0][0].endswith("evidence/v.jpg")
    es2 = copy.deepcopy(es)
    for e in es2:
        e["proof"].pop("explanation", None)
    assert attach_explanation(es2, "https://y/old", tmp_path, explain_fn=lambda v, c: None, log=lambda *_: None) is None
    assert "explanation" not in es2[0]["proof"]
    assert attach_explanation([entry("https://y/old", "2026-08-21")], "https://y/old", tmp_path, explain_fn=lambda v, c: ex, log=lambda *_: None) is None
