"""Key-gated discovery adapters + candidate merge/stats. No network: every HTTP call is monkeypatched."""
import logging

import numpy as np
import pytest

from floodproof import sources as S


class FakeFeats:
    def __init__(self, n=8):
        rng = np.random.default_rng(0)
        self.frames = [rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(n)]
        self.times = list(range(n))
        self.path = "fake.mp4"

    def __len__(self):
        return len(self.frames)


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.setattr(S, "load_env", lambda: None)
    for k in ("GOOGLE_API_KEY", "SERPAPI_KEY"):
        monkeypatch.delenv(k, raising=False)

    def nonet(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(S, "_post_json", nonet)
    monkeypatch.setattr(S, "_get_json", nonet)


# ---------------------------------------------------------------- inert without key
def test_all_adapters_skip_without_key(caplog):
    caplog.set_level(logging.INFO, logger="floodproof.sources")
    for fn, args in ((S.gvision_candidates, (FakeFeats(),)), (S.youtube_api_candidates, (["q"], "2024-09-28")),
                     (S.lens_candidates, ("https://x/y.jpg",))):
        c, info = fn(*args)
        assert c == [] and info["skipped"] == "skipped (no key)"
    assert caplog.text.count("skipped (no key)") == 3


# ---------------------------------------------------------------- gvision
def test_gvision_collects_pages_and_images(monkeypatch):
    seen = {}

    def post(url, params=None, body=None, timeout=60):
        seen.update(url=url, body=body)
        r = {"webDetection": {
            "pagesWithMatchingImages": [{"url": "https://www.youtube.com/watch?v=aaaaaaaaaaa", "pageTitle": "<b>Flood</b>"},
                                        {"url": "https://news.example/story?utm_source=x"}],
            "fullMatchingImages": [{"url": "https://i.ytimg.com/vi/bbbbbbbbbbb/hqdefault.jpg"}, {"url": "https://cdn.example/a.jpg"}],
            "partialMatchingImages": [{"url": "https://www.tiktok.com/@u/video/1"}]}}
        return {"responses": [r for _ in body["requests"]]}
    monkeypatch.setattr(S, "_post_json", post)
    c, info = S.gvision_candidates(FakeFeats(), key="K")
    assert "images:annotate" in seen["url"]
    assert 3 <= len(seen["body"]["requests"]) <= 4
    assert seen["body"]["requests"][0]["features"][0]["type"] == "WEB_DETECTION"
    assert seen["body"]["requests"][0]["image"]["content"]            # base64 content, not a URL
    urls = {x.url for x in c}
    assert urls == {"https://www.youtube.com/watch?v=aaaaaaaaaaa", "https://news.example/story",
                    "https://www.youtube.com/watch?v=bbbbbbbbbbb", "https://www.tiktok.com/@u/video/1"}
    assert all(x.adapter == "gvision" and not x.proven for x in c)
    assert [x.title for x in c if "aaaa" in x.url] == ["Flood"]


def test_gvision_error_does_not_leak_key(monkeypatch):
    def post(*a, **k):
        raise RuntimeError("403 for key=SECRETKEY")
    monkeypatch.setattr(S, "_post_json", post)
    c, info = S.gvision_candidates(FakeFeats(), key="SECRETKEY")
    assert c == [] and "SECRETKEY" not in str(info)


# ---------------------------------------------------------------- youtube
def test_youtube_api_params_and_stored_date(monkeypatch):
    calls = []

    def get(url, params=None, timeout=60):
        calls.append(params)
        return {"items": [{"id": {"videoId": "vid00000001"}, "snippet": {"title": "T", "publishedAt": "2024-09-27T10:00:00Z"}},
                          {"id": {"kind": "youtube#channel"}, "snippet": {}}]}
    monkeypatch.setattr(S, "_get_json", get)
    c, info = S.youtube_api_candidates(["kathmandu flood", "bagmati river"], "2024-09-28", key="K")
    p = calls[0]
    assert p["type"] == "video" and p["order"] == "date" and p["maxResults"] == 15
    assert p["publishedBefore"] == "2024-09-29T00:00:00Z"
    assert len(c) == 1 and c[0].adapter == "youtube_api" and c[0].url == "https://www.youtube.com/watch?v=vid00000001"
    assert c[0].stored_date == "2024-09-27" and c[0].date_independent is False and c[0].upload_date is None
    assert not c[0].proven


# ---------------------------------------------------------------- lens
def test_lens_needs_public_url():
    S.os.environ["SERPAPI_KEY"] = "K"
    try:
        c, info = S.lens_candidates(None)
    finally:
        del S.os.environ["SERPAPI_KEY"]
    assert c == [] and info["skipped"] == "no public image URL"


def test_lens_filters_platform_links(monkeypatch):
    calls = []

    def get(url, params=None, timeout=60):
        calls.append(params)
        return {"visual_matches": [{"link": "https://www.instagram.com/reel/abc/", "title": "ig"},
                                   {"link": "https://shop.example/x", "title": "shop"},
                                   {"link": "https://www.tiktok.com/@a/video/9", "title": "tt"},
                                   {"link": "https://www.facebook.com/watch/?v=1"}, {"link": "https://youtu.be/xyz"}]}
    monkeypatch.setattr(S, "_get_json", get)
    c, info = S.lens_candidates("https://floodproof.aether-soft.tech/files/run/abc/frames/lens.jpg", key="K")
    assert len(calls) == 1 and calls[0]["engine"] == "google_lens" and calls[0]["url"].endswith("lens.jpg")
    assert len(c) == 4 and all(x.adapter == "lens" for x in c)


def test_public_frame_url(tmp_path):
    runs = tmp_path / "runs"
    out = runs / "abc123"
    (out / "frames").mkdir(parents=True)
    img = out / "frames" / "lens.jpg"
    img.write_bytes(b"x")
    assert S.public_frame_url(img, out, runs) == "https://floodproof.aether-soft.tech/files/run/abc123/frames/lens.jpg"
    other = tmp_path / "elsewhere" / "r1"
    (other / "frames").mkdir(parents=True)
    img2 = other / "frames" / "lens.jpg"
    img2.write_bytes(b"x")
    assert S.public_frame_url(img2, other, runs) is None          # run dir not served -> cannot upload


# ---------------------------------------------------------------- merge / stats / proof rule
def test_merge_prioritizes_multi_source_and_video():
    a = S.Candidate("https://blog.example/a", "", "gvision")
    b = S.Candidate("https://www.youtube.com/watch?v=1", "", "yandex")
    c = S.Candidate("https://news.example/c", "", "youtube_api")
    c2 = S.Candidate("https://news.example/c", "", "lens")
    m = S.Candidate("memory:1", "", "memory", proven=True, status="proven")
    out = S.merge_and_prioritize([a, b, c, c2, m])
    assert [x.url for x in out] == ["memory:1", "https://news.example/c", "https://www.youtube.com/watch?v=1", "https://blog.example/a"]
    assert out[1].also_from == ["lens"]


def test_source_stats_counts_proven_only_when_matcher_proved():
    a = S.Candidate("u1", "", "gvision", status="proven", proven=True)
    b = S.Candidate("u2", "", "gvision", status="not_proven")
    c = S.Candidate("u3", "", "lens", status="pending")
    a.also_from = ["yandex"]
    st = S.source_stats([a, b, c])
    assert st["gvision"] == dict(candidates=2, checked=2, proven=1)
    assert st["yandex"] == dict(candidates=1, checked=1, proven=1)
    assert st["lens"] == dict(candidates=1, checked=0, proven=0)


def test_verify_candidates_skips_already_checked(monkeypatch):
    seen = []
    monkeypatch.setattr(S, "verify_candidate", lambda c, *a, **k: seen.append(c.url) or setattr(c, "status", "not_proven"))
    cs = [S.Candidate("a", "", "x"), S.Candidate("b", "", "x", status="unavailable")]
    S.verify_candidates(cs, None, max_n=5, workers=1)
    S.verify_candidates(cs, None, max_n=5, workers=1)
    assert seen == ["a"]
