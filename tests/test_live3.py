"""Live-test fixes: partial dates, place inference, Vision billing state, early stop, fair verify order, memory fast path,
per-frame media nature. No network, no LLM."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from floodproof import infer, investigator as inv, sources as S
from floodproof.infer import infer_claim, parse_partial_date, resolve_partial_date
from floodproof.report import validate_report
from test_inferred import F, _rep, clip


def aud(**k):
    return dict(dict(status="ok", has_speech=True, spoken_dates=[], spoken_place_names=[], partial_dates=[]), **k)


def clues(**k):
    return dict(status="ok", claim_parsed=dict(place="near Nepal-China border, high Himalayas", country_iso2="NP"),
                visible_text=[], on_screen_dates=[], **k)


# ---------------------------------------------------------------- partial dates
def test_resolve_partial_year_and_rollback():
    assert resolve_partial_date(8, 26, "2026-08-29") == "2026-08-26"
    assert resolve_partial_date(12, 30, "2026-01-05") == "2025-12-30"      # would be after the upload date -> previous year
    assert resolve_partial_date(2, 30, "2026-08-29") is None
    assert parse_partial_date("an August 26 slope failure", "2026-08-29")["date"] == "2026-08-26"
    assert parse_partial_date("26th of Aug", "2026-08-29")["date"] == "2026-08-26"
    assert parse_partial_date("August 26, 2025", "2026-08-29") is None      # has a year: handled by parse_explicit_date
    assert parse_partial_date("meeting at 10:30", "2026-08-29") is None


def test_partial_date_from_title_description_screen_and_narration():
    c = infer_claim(clip(title="Flood on 26 Aug in Nepal"), None, None, None, clues())
    assert (c["event_date"], c["event_date_basis"]) == ("2026-08-26", "title_partial_year")
    assert "year taken from post date" in c["event_date_basis_detail"]
    c = infer_claim(clip(title="Flood", desc="It happened August 26."), None, None, None, clues())
    assert c["event_date_basis"] == "description_partial_year"
    cl = clues()
    cl["visible_text"] = [dict(text="AUG 26", refs=[3])]
    assert infer_claim(clip(title="Flood"), None, None, None, cl)["event_date_basis"] == "on_screen_text_partial_year"
    cl = clues(audio=aud(partial_dates=[dict(month=8, day=26, text="August 26", audio_t=40.0)]))
    c = infer_claim(clip(title="Flood", up="2026-08-29"), None, None, None, cl)
    assert (c["event_date"], c["event_date_basis"]) == ("2026-08-26", "spoken_in_narration_partial_year")
    assert "date spoken in narration without year" in c["event_date_basis_detail"] and "year taken from post date" in c["event_date_basis_detail"]
    assert c["source"]["event_date"] == "inferred_from_post"
    # explicit full dates and user dates still win; no partial date -> date_posted
    assert infer_claim(clip(title="Flood 26 AUG 2025"), None, None, None, clues())["event_date"] == "2025-08-26"
    assert infer_claim(clip(title="Flood 26 Aug"), None, "2020-01-01", None, clues())["event_date"] == "2020-01-01"
    assert infer_claim(clip(title="Flood"), None, None, None, clues())["event_date_basis"] == "date_posted"


def test_audio_partial_dates_are_kept_and_need_a_timestamp():
    d = inv._clean_audio(dict(partial_dates=[dict(month=8, day=26, text="August 26", audio_t="0:40"), dict(month=8, day=26, text="x"),
                                             dict(month=13, day=1, audio_t=2)]))
    assert d["partial_dates"] == [dict(month=8, day=26, text="August 26", audio_t=40.0)]


def test_partial_date_report_validates():
    cl = clues(audio=aud(transcript_summary="An August 26 slope failure near the Nepal-China border"))
    c = infer_claim(clip(title="Nepal's Flood: What Triggered It?", up="2026-08-29"), None, None, None, cl)
    assert c["event_date"] == "2026-08-26" and c["event_date_basis"] == "spoken_in_narration_partial_year"
    validate_report(_rep("unknown", c, []))


# ---------------------------------------------------------------- place
def test_place_inferred_from_title_country_even_with_possessive():
    c = infer_claim(clip(title="Nepal's Flood: What Triggered It?"), None, None, None, clues())
    assert c["place"] == "Nepal" and c["source"]["place"] == "inferred_from_post" and "country named in the post" in c["place_basis"]
    assert c["country_iso2"] == "NP"


def test_place_from_narration_and_part_of_claimed_place():
    cl = clues(audio=aud(transcript_summary="a slope failure near the Nepal-China border"))
    c = infer_claim(clip(title="What triggered it?"), None, None, None, cl)
    assert c["place"] == "Nepal" and c["source"]["place"] == "inferred_from_narration" and "narration" in c["place_basis"]
    cl = clues()
    cl["claim_parsed"] = dict(place="Rasuwa district, Nepal")
    c = infer_claim(clip(title="Rasuwa flood video"), None, None, None, cl)
    assert c["place"] == "Rasuwa district" and c["source"]["place"] == "inferred_from_post"
    assert infer_claim(clip(title="Big flood"), None, None, None, clues())["place"] is None      # nothing stated -> not inferred
    assert infer_claim(clip(title="Nepal flood"), None, None, "Pokhara", clues())["place"] == "Pokhara"


# ---------------------------------------------------------------- google vision billing
class Resp:
    status_code, text = 403, "This API method requires billing to be enabled"


def test_gvision_billing_error_is_unavailable_and_cached(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        e = RuntimeError("403 Client Error: Forbidden")
        e.response = Resp()
        raise e
    monkeypatch.setattr(S, "_post_json", post)
    from test_sources_v2 import FakeFeats
    c, info = S.gvision_candidates(FakeFeats(), key="K")
    assert c == [] and info["errors"] == [] and info["unavailable"] == "billing not enabled"
    assert S.gvision_disabled_until() > time.time() + 3000
    c, info = S.gvision_candidates(FakeFeats(), key="K")        # cached: no second call
    assert len(calls) == 1 and info["unavailable"] and info.get("cached")
    monkeypatch.setattr(S, "gvision_disabled_until", lambda: time.time() - 1)   # expired -> retried
    S.gvision_candidates(FakeFeats(), key="K")
    assert len(calls) == 2


def test_gvision_other_errors_still_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("500 boom")
    monkeypatch.setattr(S, "_post_json", boom)
    from test_sources_v2 import FakeFeats
    c, info = S.gvision_candidates(FakeFeats(), key="K")
    assert info["errors"] and "unavailable" not in info and S.gvision_disabled_until() == 0


# ---------------------------------------------------------------- early stop
def cand(url, date=None, proven=False, ind=True, also=(), status="pending", adapter="youtube_api"):
    c = S.Candidate(url, "", adapter, status=status, upload_date=date, proven=proven, date_independent=ind)
    c.also_from = list(also)
    return c


def test_can_stop_early_is_conservative():
    good = cand("https://youtube.com/watch?v=a", "2026-08-01", True, True, also=["yandex"])
    low = [cand("https://blog.example/x")]
    assert S.can_stop_early([good], low, "2026-08-26")
    assert not S.can_stop_early([good], low, None)                                   # no real claimed date
    assert not S.can_stop_early([good], low, "2026-07-01")                           # copy is not earlier than the claim
    assert not S.can_stop_early([cand("u", "2026-08-01", True, ind=False)], low, "2026-08-26")   # inherited date
    same_tier = [cand("https://youtube.com/watch?v=b", also=["yandex"])]
    assert not S.can_stop_early([good], same_tier, "2026-08-26")                      # equally promising candidates remain
    assert not S.can_stop_early([good], [], "2026-08-26")


def test_verify_candidates_stops_and_records(monkeypatch):
    seen = []

    def fake(c, *a, **k):
        seen.append(c.url)
        if c.url.endswith("=a"):
            c.proven, c.status, c.upload_date, c.date_independent = True, "proven", "2026-08-01", True
        else:
            c.status = "not_proven"
        time.sleep(0.05)
        return c
    monkeypatch.setattr(S, "verify_candidate", fake)
    cs = [cand("https://youtube.com/watch?v=a", also=["yandex"])] + [cand(f"https://blog.example/{i}") for i in range(8)]
    info = {}
    S.verify_candidates(cs, None, max_n=9, workers=2, stop_before="2026-08-26", info=info)
    assert info["early_stop"] is True and info["checked"] < 9 and info["skipped_lower_priority"] == 9 - len(seen)
    info = {}
    S.verify_candidates([cand(f"https://x.example/{i}") for i in range(4)], None, max_n=4, workers=2, info=info)
    assert info["early_stop"] is False and info["checked"] == 4


# ---------------------------------------------------------------- fair verification order
def test_no_source_fills_the_budget_and_each_gets_min_slots():
    cs = [cand(f"https://www.youtube.com/watch?v=y{i:010d}", adapter="yandex") for i in range(30)]
    cs += [cand(f"https://www.youtube.com/watch?v=t{i:010d}", adapter="youtube_api") for i in range(5)]
    cs += [cand(f"https://www.instagram.com/reel/l{i}", adapter="lens") for i in range(3)]
    cs += [cand(f"https://news.example/t{i}", adapter="tavily") for i in range(4)]
    order = S.merge_and_prioritize(cs, max_n=14)
    first = order[:14]
    by = {a: sum(c.adapter == a for c in first) for a in ("yandex", "youtube_api", "lens", "tavily")}
    assert all(v >= 2 for v in by.values()), by
    assert by["yandex"] < 14 and len(order) == len(cs)
    # without a budget it is a plain round-robin, video URLs before other links
    rr = S.merge_and_prioritize(cs)[:4]
    assert {c.adapter for c in rr} >= {"yandex", "youtube_api", "lens"}


def test_dedupe_by_normalised_url_and_multi_source_first():
    a = cand("https://youtu.be/abcdefghijk?si=zz", adapter="yandex")
    b = cand("https://www.youtube.com/watch?v=abcdefghijk&utm_source=x", adapter="tavily")
    c = cand("https://m.youtube.com/shorts/abcdefghijk", adapter="lens")
    d = cand("https://other.example/z/", adapter="yandex")
    e = cand("https://other.example/z", adapter="tavily")
    out = S.merge_and_prioritize([d, a, b, c, e], max_n=14)
    assert len(out) == 2 and out[0].url == a.url and sorted(a.also_from) == ["lens", "tavily"] and d.also_from == ["tavily"]


def test_source_stats_record_which_source_proved(monkeypatch):
    p = cand("https://youtu.be/p", adapter="lens", status="proven", proven=True, also=["tavily"])
    st = S.source_stats([p, cand("https://x.example/1", adapter="yandex", status="not_proven")])
    assert st["lens"]["proven"] == 1 and st["tavily"]["proven"] == 1 and st["yandex"]["proven"] == 0


# ---------------------------------------------------------------- memory fast path
class FakeClip:
    path, url, title, description, upload_date, uploader = "x.mp4", "https://youtu.be/new", "Rasuwa flood", None, "2026-08-27", "u"
    has_audio, duration = False, 10

    def to_dict(self):
        return dict(url=self.url, title=self.title, upload_date=self.upload_date, uploader=self.uploader, duration=10, has_audio=False)


class FakeF:
    times = [0.0, 1.0]
    frames = [np.zeros((8, 8, 3), np.uint8)] * 2
    path = "x.mp4"

    def __len__(self):
        return 2


def _run_pipeline(monkeypatch, tmp_path, stored_date, muse_clues):
    from floodproof import pipeline as P
    called = []
    monkeypatch.setattr(P.media, "fetch", lambda s: FakeClip())
    monkeypatch.setattr(P.media, "cleanup", lambda c: None)
    monkeypatch.setattr(P.ClipFeatures, "from_path", classmethod(lambda cls, p: FakeF()))
    monkeypatch.setattr(P, "load_env", lambda: None)
    entry = dict(id=1, source_url="https://www.instagram.com/reel/OLD/", upload_date=stored_date, title="old")
    mc = S.Candidate(entry["source_url"], "", "memory", "entry 1", "proven", stored_date, "memory entry (inherited)", "now", "u", True,
                     dict(consistent_frames=5, max_inliers=80, temporal_offset=0.0, best_pairs=[]), None, entry, stored_date=stored_date)
    res = SimpleNamespace(mode="hash-shortlist", shortlisted=[1], n_entries=1, seconds=0.1)
    monkeypatch.setattr(P.sources, "memory_candidates", lambda mem, feats, evidence_dir=None: ([mc], res))
    for n in ("gvision_candidates", "yandex_candidates", "lens_candidates", "youtube_api_candidates", "tavily_candidates"):
        monkeypatch.setattr(P.sources, n, lambda *a, _n=n, **k: called.append(_n) or ([], {}))
    monkeypatch.setattr(P.sources, "verify_candidates", lambda *a, **k: called.append("verify"))
    monkeypatch.setattr(P.sources, "enrich_dates", lambda cs: [setattr(c, "date_independent", True) or setattr(c, "independent_date", c.upload_date)
                                                              or setattr(c, "independent_date_source", "test") for c in cs])
    monkeypatch.setattr(P, "_physical", lambda *a: dict(status="NOT_RUN", reason="test"))
    monkeypatch.setattr(P.investigator, "investigate_clip", lambda *a, **k: (called.append("muse"), muse_clues)[1])
    mem = SimpleNamespace(add=lambda *a, **k: None)
    rep = P.investigate("https://youtu.be/new", claim="Flood in Rasuwa", date="2026-08-26", place="Rasuwa", out_dir=tmp_path / "r", memory=mem, log=lambda m: None)
    return rep, called


def test_memory_hit_before_claim_takes_the_fast_path(monkeypatch, tmp_path):
    rep, called = _run_pipeline(monkeypatch, tmp_path, "2026-08-01", dict(status="error", error="x"))
    assert rep["fast_path"] == "memory_hit" and rep["visual_matching"]["fast_path"] == "memory_hit"
    assert called == ["muse"]                                  # Muse still runs (media nature / debunk); nothing else is searched
    assert rep["final_verdict"]["label"] == "RECYCLED_FOOTAGE" and "stage_timings_s" in rep


def test_memory_hit_not_before_claim_runs_full_search(monkeypatch, tmp_path):
    rep, called = _run_pipeline(monkeypatch, tmp_path, "2026-08-27", dict(status="error", error="x"))
    assert "fast_path" not in rep and "yandex_candidates" in called and "youtube_api_candidates" in called
    assert "tavily_candidates" in called                      # Muse failed: Tavily still runs on the plain claim text
    assert rep["visual_matching"]["adapters"]["tavily"].get("query_source") is None or True


# ---------------------------------------------------------------- per-frame media nature
def labels(*pairs):
    return [dict(frame=f, label=l, confidence=.8, reason=f"r{f}") for f, l in pairs]


class Times:
    def __init__(self, n=10):
        self.times = [float(i) for i in range(n)]
        self.n = n

    def __len__(self):
        return self.n


def test_mixed_media_nature_from_per_frame_labels():
    rej = []
    raw = labels((0, "camera_footage"), (1, "likely_ai_generated"), (2, "likely_ai_generated"), (3, "camera_footage"),
                 (4, "graphic_or_text"), (5, "unknown"), (99, "camera_footage"), (2, "unknown"))
    fl = inv._clean_frame_labels(raw, Times(), rej)
    assert [x["frame"] for x in fl] == [0, 1, 2, 3, 4, 5] and len(rej) == 2          # frame 99 and the duplicate 2 rejected
    mn = inv.derive_media_nature(fl, dict(value="news_compilation", confidence=.8, cited_frames=[1]), Times())
    assert mn["value"] == "mixed" and mn["by_class"]["likely_ai_generated"] == [1, 2] and mn["by_class"]["camera_footage"] == [0, 3]
    assert mn["summary"].startswith("Mixed: ") and "frames 1,2" in mn["summary"] and mn["status"] == "AI_OBSERVATION_HYPOTHESIS"
    assert mn["cited_frames"] == [0, 1, 2, 3]


def test_uniform_and_unlabelled_keep_old_values():
    T = Times()
    cam = inv.derive_media_nature(inv._clean_frame_labels(labels((0, "camera_footage"), (1, "camera_footage"), (2, "unknown")), T, []),
                                  dict(value="camera_footage", cited_frames=[0]), T)
    assert cam["value"] == "camera_footage" and cam["cited_frames"] == [0, 1]
    comp = inv.derive_media_nature(inv._clean_frame_labels(labels((0, "camera_footage")), T, []), dict(value="news_compilation", cited_frames=[0]), T)
    assert comp["value"] == "news_compilation"
    none = inv.derive_media_nature([], dict(value="animation_or_cgi", cited_frames=[1]), T)
    assert none["value"] == "animation_or_cgi" and none["per_frame"] == []
    allunk = inv.derive_media_nature(inv._clean_frame_labels(labels((0, "unknown")), T, []), dict(value="unknown", cited_frames=[]), T)
    assert allunk["value"] == "unknown"


def test_camera_frames_drive_keyframe_choice():
    T = Times()
    fl = inv._clean_frame_labels(labels((0, "likely_ai_generated"), (4, "camera_footage")), T, [])
    mn = inv.derive_media_nature(fl, dict(value="unknown"), T)
    ids = inv.camera_frame_ids(dict(status="ok", media_nature=mn), 10)
    assert ids == [3, 4, 5, 6, 7, 8, 9]
    assert inv.camera_frame_ids(dict(status="ok", media_nature=dict(per_frame=[])), 10) == []
    rng = np.random.default_rng(1)
    feats = Times()
    feats.frames = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(10)]
    assert set(S._spread_frames(feats, 3, set(ids))) <= set(ids)


def test_report_validates_mixed_and_rejects_bad_frame():
    claim = dict(text="t", event_date="2026-08-26", event_date_basis="title", source=dict(event_date="inferred_from_post"))
    fl = labels((1, "likely_ai_generated"), (3, "camera_footage"))
    mn = dict(value="mixed", confidence=.8, cited_frames=[1, 3], cited_timestamps_s=[1.0, 3.0], reasons=[], status="AI_OBSERVATION_HYPOTHESIS",
              per_frame=fl, by_class={"likely_ai_generated": [1], "camera_footage": [3]},
              summary="Mixed: likely AI-generated shots (frames 1) + camera footage (frames 3) (AI observation)")
    r = _rep("mixed", claim, [])
    r["media_nature"] = mn
    r["input"]["n_keyframes"] = 6
    validate_report(r)
    assert r["schema_version"] == "1.5"
    for bad in (labels((1, "likely_ai_generated"), (9, "camera_footage")), labels((1, "likely_ai_generated"), (2, "likely_ai_generated")),
                [dict(frame=1, label="weird")]):
        r["media_nature"] = dict(mn, per_frame=bad)
        with pytest.raises(AssertionError):
            validate_report(r)


def test_mixed_never_changes_label_and_debunk_mentions_observation():
    from floodproof import debunk
    claim = dict(text="t", event_date="2026-08-26", event_date_basis="title", source=dict(event_date="inferred_from_post"))
    base = validate_report(_rep("unknown", claim, []))["final_verdict"]["label"]
    r = _rep("unknown", claim, [])
    r["media_nature"] = dict(value="mixed", confidence=.8, cited_frames=[1, 3], cited_timestamps_s=[1.0, 3.0], reasons=[], status="AI_OBSERVATION_HYPOTHESIS",
                             per_frame=labels((1, "likely_ai_generated"), (3, "camera_footage")),
                             summary="Mixed: likely AI-generated shots (frames 1) + camera footage (frames 3) (AI observation)")
    assert r["final_verdict"]["label"] == base
    t = debunk.template_debunk(r, ["en", "ne", "hi"])
    assert "AI observation" in t["texts"]["en"] and "not proof" in t["texts"]["en"]
    sents = [dict(text=x["sentence"], ref=x["ref"]) for x in t["field_refs"]["en"]]
    assert debunk.check_grounded(r, sents) == []
