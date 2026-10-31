"""YouTube Data API quota guard: ledger, cache, quota-error handling and the keyless yt-dlp fallback (no network)."""
import requests

from floodproof import sources as S


def _item(vid, day="2026-08-20"):
    return {"id": {"videoId": vid}, "snippet": {"title": vid, "publishedAt": f"{day}T00:00:00Z"}}


def _quota_error():
    r = requests.Response()
    r.status_code = 403
    r._content = b'{"error":{"errors":[{"reason":"quotaExceeded"}]}}'
    return requests.HTTPError("403 Client Error: Forbidden", response=r)


def test_spends_100_units_per_search_and_caches_repeats(monkeypatch):
    calls = []
    monkeypatch.setattr(S, "_get_json", lambda url, params=None, timeout=60: calls.append(params) or {"items": [_item("AAAAAAAAAAA")]})
    S.youtube_api_candidates(["flood nepal"], "2026-08-26", key="K")
    assert len(calls) == 2 and S.yt_quota_state()["units"] == 200          # date pass + relevance pass
    c, info = S.youtube_api_candidates(["flood nepal"], "2026-08-26", key="K")
    assert len(calls) == 2 and info["cached"] == 2 and S.yt_quota_state()["units"] == 200   # identical searches: no new units
    assert c[0].url == "https://www.youtube.com/watch?v=AAAAAAAAAAA"


def test_quota_error_marks_exhausted_and_falls_back_to_ytsearch(monkeypatch):
    api, kl = [], []

    def get(url, params=None, timeout=60):
        api.append(params)
        raise _quota_error()
    monkeypatch.setattr(S, "_get_json", get)
    monkeypatch.setattr(S, "ytdlp_search", lambda q, n=10: kl.append(q) or [("BBBBBBBBBBB", "t", None), ("LATELATELAT", "t", "2026-09-01")])
    c, info = S.youtube_api_candidates(["glacier collapse"], "2026-08-26", key="K")
    assert S.yt_quota_state()["exhausted"] and "keyless" in info["quota"] and kl == ["glacier collapse"]
    assert [x.url for x in c] == ["https://www.youtube.com/watch?v=BBBBBBBBBBB"]   # a hit known to be after the cutoff is dropped
    assert c[0].hint.startswith("ytsearch:") and c[0].adapter == "youtube_api" and not c[0].proven
    n_api = len(api)
    S.youtube_api_candidates(["another query"], "2026-08-26", key="K")
    assert len(api) == n_api                                                   # exhausted for the day: the API is not called again


def test_budget_stops_api_before_google_does(monkeypatch):
    monkeypatch.setattr(S, "YT_DAILY_BUDGET", 300)
    api = []
    monkeypatch.setattr(S, "_get_json", lambda url, params=None, timeout=60: api.append(params) or {"items": [_item(f"C{len(api):010d}")]})
    monkeypatch.setattr(S, "ytdlp_search", lambda q, n=10: [("DDDDDDDDDDD", "t", None)])
    _, i1 = S.youtube_api_candidates(["q one"], "2026-08-26", key="K")
    _, i2 = S.youtube_api_candidates(["q two"], "2026-08-26", key="K")
    assert len(api) == 3 and S.yt_quota_state()["units"] == 300
    assert i2["fallback_queries"] == 1 and "budget" in i2["quota"]


def test_other_errors_are_not_quota(monkeypatch):
    def get(url, params=None, timeout=60):
        raise requests.ConnectionError("boom")
    monkeypatch.setattr(S, "_get_json", get)
    monkeypatch.setattr(S, "ytdlp_search", lambda q, n=10: (_ for _ in ()).throw(AssertionError("no fallback on network errors")))
    c, info = S.youtube_api_candidates(["x"], "2026-08-26", key="K")
    assert c == [] and not S.yt_quota_state()["exhausted"] and info["errors"]


def test_new_pacific_day_resets_the_ledger(monkeypatch):
    monkeypatch.setattr(S, "_pacific_day", lambda: "2026-11-05")
    S._yt_quota_spend(9000); S._yt_quota_mark_exhausted()
    assert not S.yt_quota_available()
    monkeypatch.setattr(S, "_pacific_day", lambda: "2026-11-06")
    assert S.yt_quota_available() and S.yt_quota_state()["units"] == 0
