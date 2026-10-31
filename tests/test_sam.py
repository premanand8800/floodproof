import cv2
import numpy as np

from floodproof import sam
from floodproof.matcher import ClipFeatures

STREAM = "<0f>1<|box;x1=0;y1=150;x2=180;y2=200;w=180;h=200|><|mask;x=0;y=0;data=2,2,~abcd|>,2<|box;x1=10;y1=10;x2=40;y2=30;w=180;h=200|>"


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (200, 180, 3), dtype=np.uint8), (0, 0), 1.5)


def test_parse_boxes():
    b = sam.parse_boxes(STREAM)
    assert b == [(0, 150, 180, 200, 180, 200), (10, 10, 40, 30, 180, 200)]
    assert sam.parse_boxes("") == []


def test_keep_mask_and_cache():
    sam._cache.clear()
    n = []

    def fake(jpeg, prompt):
        n.append(prompt)
        return STREAM if prompt == "logo" else ""

    f = _frame(1)
    m = sam.keep_mask(f, call=fake)
    assert m.shape == f.shape[:2] and m[180, 90] == 0 and m[100, 90] == 255 and m[20, 20] == 0
    sam.keep_mask(f, call=fake)
    assert len(n) == len(sam.PROMPTS)   # second call served from the cache


def test_failure_means_no_mask():
    sam._cache.clear()

    def boom(jpeg, prompt):
        raise RuntimeError("402")

    assert sam.keep_mask(_frame(2), call=boom) is None


def test_overlay_covering_too_much_is_ignored():
    sam._cache.clear()
    assert sam.keep_mask(_frame(3), call=lambda j, p: "<0f>1<|box;x1=0;y1=0;x2=180;y2=190;w=180;h=200|>") is None


def test_masked_sift_has_no_keypoints_in_overlay_and_frames_untouched():
    f = _frame(4)
    orig = f.copy()
    mask = np.full(f.shape[:2], 255, np.uint8)
    mask[100:, :] = 0
    plain = ClipFeatures.from_frames("x", [0.0], [f])
    masked = ClipFeatures.from_frames("x", [0.0], [f], masker=lambda fr: mask)
    assert len(plain.feats[0][0]) > 0 and (plain.feats[0][0][:, 1] >= 100).any()
    assert (masked.feats[0][0][:, 1] < 100).all()
    assert (masked.flipped[0][0][:, 1] < 100).all()
    assert np.array_equal(masked.frames[0], orig)   # proof frames never altered


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("FLOODPROOF_SAM", raising=False)
    assert sam.masker() is None
    monkeypatch.setenv("FLOODPROOF_SAM", "1")
    assert sam.masker() is sam.keep_mask
