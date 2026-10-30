"""Snowball search and the spread timeline: no network, no API (every external call is injected)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from floodproof import snowball, spread, sources
from floodproof.report import validate_report, add_clarity


def _frame(tmp_path, name="f.jpg"):
    p = tmp_path / name
    cv2.imwrite(str(p), np.full((90, 160, 3), 120, np.uint8))
    return str(p)


def _cand(url, date, tmp_path, adapter="yandex", proven=True, uploader="up"):
    c = sources.Candidate(url, "t", adapter, "h", "proven" if proven else "pending")
    c.proven, c.upload_date, c.date_independent, c.date_source, c.uploader = proven, date, bool(date), "yt-dlp metadata", uploader
    c.independent_date, c.independent_date_source = date, "yt-dlp metadata"
    if proven:
        fp = _frame(tmp_path, f"{abs(hash(url))}.jpg")
        c.proof = dict(viral_frame=fp, copy_frame=fp, viral_t=1.0, copy_t=1.0, mirrored=False, inliers=[])
    return c


def _chat(queries):
    def f(model, content, max_tokens=0):
        return json.dumps(dict(visible_text=[dict(text="@SourceChan", kind="watermark")], queries=queries)), 0.004
    return f


Q = [dict(q="original video glacier", language="en", rationale="title says original", source_field="title"),
     dict(q="glacier collapse footage", language="ja", rationale="caption is Japanese", source_field="caption")]


def test_clean_queries_caps_dedupes_and_needs_rationale():
    qs = [dict(q=f"query {i}", rationale="why", source_field="title") for i in range(9)]
    qs += [dict(q="no rationale", source_field="title"), dict(q="bad field", rationale="x", source_field="nonsense")]
    out = snowball.clean_queries(qs)
    assert len(out) == snowball.MAX_QUERIES and all(x["rationale"] and x["source_field"] in snowball.SOURCE_FIELDS for x in out)
    assert snowball.clean_queries([dict(q="query 0", rationale="why", source_field="title")], used=["QUERY 0"]) == []


def test_snowball_round_finds_new_copy_then_stops(tmp_path, monkeypatch):
    seed = _cand("https://www.youtube.com/watch?v=AAAAAAAAAAA", "2026-08-21", tmp_path)
    calls = {"yt": [], "lens": 0}

    def yt(qs, d):
        calls["yt"].append((qs, d))
        return [sources.Candidate("https://www.youtube.com/watch?v=BBBBBBBBBBB", "x", "youtube_api", "q")], dict(queries=2)

    def lens(u):
        calls["lens"] += 1
        return [sources.Candidate("https://www.instagram.com/reel/XYZ/", "y", "lens", "l")], dict(calls=1)

    def verify(c, f, qd, su, ev):
        c.proven = c.url.endswith("BBBBBBBBBBB") or c.url.endswith("XYZ/")
        c.status = "proven" if c.proven else "not_proven"
        c.upload_date = "2026-08-10" if "BBBB" in c.url else "2026-08-15"
        c.date_independent = True
        c.proof = dict(copy_frame=_frame(tmp_path, f"p{len(c.url)}.jpg"), viral_frame="x", viral_t=1, copy_t=1, mirrored=False, inliers=[])
        return c
    monkeypatch.setattr(snowball, "_lens_url", lambda frame, out: ("https://pub/x.jpg", None))
    new, block, reads = snowball.run(feats=None, proven=[seed], out_dir=tmp_path, ev_dir=tmp_path, chat_fn=_chat(Q), meta_fn=lambda u: dict(title="orig?", uploader="up"),
                                     fns=dict(youtube=yt, lens=lens), verify_fn=verify, log=lambda *a: None)
    assert {c.url for c in new} == {"https://www.youtube.com/watch?v=BBBBBBBBBBB", "https://www.instagram.com/reel/XYZ/"}
    r1 = block["rounds"][0]
    assert r1["seed_copy"] == seed.url and len(r1["queries"]) == 2 and all(q["rationale"] and q["source_field"] for q in r1["queries"])
    assert calls["yt"][0][1] == "2026-08-20" and calls["lens"] >= 1          # publishedBefore the earliest proven day
    assert block["rounds"][-1]["proven_new"] == [] and "found no new proven copy" in block["stopped_reason"]
    assert block["cost"]["verifications"] <= snowball.MAX_EXTRA_VERIFY and block["cost"]["openrouter_usd"] > 0
    assert all(c.snowball_round == 1 for c in new)
    assert reads[seed.url]["usernames"] == ["SourceChan"]


def test_snowball_verification_cap_and_round_cap(tmp_path):
    seed = _cand("https://www.youtube.com/watch?v=AAAAAAAAAAA", "2026-08-21", tmp_path)
    n = [0]

    def yt(qs, d):
        out = []
        for _ in range(20):
            n[0] += 1
            out.append(sources.Candidate(f"https://www.youtube.com/watch?v={n[0]:011d}", "x", "youtube_api", "q"))
        return out, dict(queries=1)

    def verify(c, f, qd, su, ev):
        c.proven = True; c.status = "proven"; c.upload_date = "2026-08-01"; c.date_independent = True
        c.proof = dict(copy_frame=_frame(tmp_path, f"{c.url[-5:]}.jpg"), viral_frame="x", viral_t=1, copy_t=1, mirrored=False, inliers=[])
        return c
    new, block, _ = snowball.run(feats=None, proven=[seed], out_dir=tmp_path, ev_dir=tmp_path, chat_fn=_chat(Q), meta_fn=lambda u: None,
                                 fns=dict(youtube=yt), verify_fn=verify, log=lambda *a: None)
    assert block["cost"]["verifications"] <= snowball.MAX_EXTRA_VERIFY and len(block["rounds"]) <= snowball.MAX_ROUNDS
    assert sum(r["verified"] for r in block["rounds"]) == block["cost"]["verifications"]


def test_snowball_time_cap_and_no_seed(tmp_path):
    seed = _cand("https://www.youtube.com/watch?v=AAAAAAAAAAA", "2026-08-21", tmp_path)
    new, block, _ = snowball.run(feats=None, proven=[seed], out_dir=tmp_path, ev_dir=tmp_path, chat_fn=_chat(Q), fns={}, verify_fn=None, max_seconds=5, log=lambda *a: None)
    assert not new and block["rounds"] == [] and block["stopped_reason"] == "time budget reached"
    nofr = _cand("https://www.youtube.com/watch?v=CCCCCCCCCCC", "2026-08-21", tmp_path)
    nofr.proof = None
    _, b2, _ = snowball.run(feats=None, proven=[nofr], out_dir=tmp_path, ev_dir=tmp_path, chat_fn=_chat(Q), fns={}, log=lambda *a: None)
    assert b2["rounds"] == []


def test_unparseable_muse_reply_is_harmless(tmp_path):
    seed = _cand("https://www.youtube.com/watch?v=AAAAAAAAAAA", "2026-08-21", tmp_path)
    new, block, _ = snowball.run(feats=None, proven=[seed], out_dir=tmp_path, ev_dir=tmp_path, chat_fn=lambda *a, **k: ("not json", 0.0), meta_fn=lambda u: None,
                                 fns={}, log=lambda *a: None)
    assert not new and block["rounds"][0]["queries"] == [] and block["rounds"][0]["muse_error"]


# ---------------------------------------------------------------- spread
def _inliers(x0, y0, x1, y1, n=30):
    """Input-frame points of a copy that shows the input rectangle (x0,y0)-(x1,y1) of the frame."""
    rng = np.random.default_rng(0)
    pts = rng.uniform([x0, y0], [x1, y1], (n, 2))
    return [[float(x), float(y), float((x - x0) / (x1 - x0)), float((y - y0) / (y1 - y0))] for x, y in pts]


def _entry(url, date, inl, uploader="u", found_by="yandex", mirrored=False, read=None, ind=True):
    e = dict(url=url, upload_date=date, date_source="yt-dlp metadata", date_independent=ind, uploader=uploader, adapter=found_by, found_by=found_by,
             proof=dict(viral_frame="a.jpg", copy_frame="b.jpg", viral_t=1.0, copy_t=1.0, mirrored=mirrored, inliers=inl))
    if read:
        e["muse_read"] = read
    return e


def _rep(entries):
    return dict(visual_matching=dict(proven_matches=entries))


def test_crop_makes_wide_copy_the_likely_parent():
    wide = _entry("https://www.youtube.com/watch?v=WIDE0000000", "2026-08-10", _inliers(0, 0, 1, 1))
    crop = _entry("https://www.instagram.com/reel/CROP/", "2026-08-20", _inliers(0.2, 0.2, 0.8, 0.8))
    sp = spread.build_spread(_rep([crop, wide]))
    assert [e["url"] for e in sp] == [wide["url"], crop["url"]]
    assert sp[0]["likely_parent"] is None
    lp = sp[1]["likely_parent"]
    assert lp["url"] == wide["url"] and lp["label"] == "likely derived, not proof"
    assert any("crop or zoom" in r for r in lp["reasons"]) and "dated earlier" in lp["reasons"][0]


def test_wider_copy_is_not_derived_from_a_narrower_one():
    narrow = _entry("https://www.youtube.com/watch?v=NARROW00000", "2026-08-10", _inliers(0.2, 0.2, 0.8, 0.8))
    wide = _entry("https://www.instagram.com/reel/WIDE/", "2026-08-20", _inliers(0, 0, 1, 1))
    assert spread.build_spread(_rep([narrow, wide]))[1]["likely_parent"] is None


def test_date_order_alone_is_not_enough_and_same_day_has_no_order():
    a = _entry("https://www.youtube.com/watch?v=AAAAAAAAAAA", "2026-08-10", [])
    b = _entry("https://www.instagram.com/reel/B/", "2026-08-20", [])
    assert all(e["likely_parent"] is None for e in spread.build_spread(_rep([a, b])))
    c = _entry("https://www.instagram.com/reel/C/", "2026-08-20", _inliers(0.1, 0.1, 0.9, 0.9))
    d = _entry("https://www.youtube.com/watch?v=DDDDDDDDDDD", "2026-08-20", _inliers(0, 0, 1, 1))
    assert spread.build_spread(_rep([d, c]))[1]["likely_parent"] is None            # same day: order unknown
    e = _entry("https://www.instagram.com/reel/E/", "2026-08-25", _inliers(0.1, 0.1, 0.9, 0.9), ind=False)    # inherited date: never gets a parent
    w = _entry("https://www.youtube.com/watch?v=WWWWWWWWWWW", "2026-08-10", _inliers(0, 0, 1, 1))
    assert spread.build_spread(_rep([w, e]))[1]["likely_parent"] is None


def test_username_watermark_names_the_parent():
    def rd(t, kind="watermark"):
        v = [dict(text=t, kind=kind)]
        return dict(visible_text=v, usernames=snowball.usernames_in(v))
    src = _entry("https://www.youtube.com/watch?v=SRC00000000", "2026-08-10", [], uploader="SourceChan", read=dict(visible_text=[], usernames=[]))
    other = _entry("https://www.instagram.com/reel/O/", "2026-08-12", [], uploader="Other", read=dict(visible_text=[], usernames=[]))
    kid = _entry("https://www.tiktok.com/@kid/video/1", "2026-08-20", [], uploader="kid", read=rd("@SourceChan"))
    sp = spread.build_spread(_rep([kid, other, src]))
    lp = sp[2]["likely_parent"]
    assert lp and lp["url"] == src["url"] and any("@SourceChan" in r for r in lp["reasons"])


def test_shared_overlay_not_on_earlier_copy_links_copies():
    def rd(*ts):
        return dict(visible_text=[dict(text=t, kind="caption") for t in ts], usernames=[])
    first = _entry("https://www.youtube.com/watch?v=FIRST000000", "2026-08-01", [], read=rd("river camera"))
    mid = _entry("https://www.youtube.com/watch?v=MID00000000", "2026-08-10", [], read=rd("BREAKING NEWS Nepal flood"))
    last = _entry("https://www.instagram.com/reel/LAST/", "2026-08-20", [], read=rd("BREAKING NEWS Nepal flood"))
    sp = spread.build_spread(_rep([last, mid, first]))
    assert sp[2]["likely_parent"]["url"] == mid["url"] and any("overlay" in r for r in sp[2]["likely_parent"]["reasons"])
    assert sp[1]["likely_parent"] is None


def test_equal_evidence_for_two_parents_leaves_parent_null():
    w1 = _entry("https://www.youtube.com/watch?v=W1000000000", "2026-08-05", _inliers(0, 0, 1, 1))
    w2 = _entry("https://www.youtube.com/watch?v=W2000000000", "2026-08-06", _inliers(0, 0, 1, 1))
    crop = _entry("https://www.instagram.com/reel/CROP/", "2026-08-20", _inliers(0.2, 0.2, 0.8, 0.8))
    e = spread.build_spread(_rep([w1, w2, crop]))[2]
    assert e["likely_parent"] is None and "equally well" in e["parent_note"]


def test_mirroring_only_is_not_enough():
    a = _entry("https://www.youtube.com/watch?v=AAAAAAAAAAA", "2026-08-01", [], mirrored=True)
    b = _entry("https://www.instagram.com/reel/B/", "2026-08-10", [], mirrored=True)
    assert spread.build_spread(_rep([a, b]))[1]["likely_parent"] is None


# ---------------------------------------------------------------- report validation
def _demo():
    p = Path(__file__).resolve().parents[1] / "demo" / "success_recycled" / "report.json"
    return json.loads(p.read_text())


def test_demo_reports_stay_valid_without_new_fields():
    old = json.loads((Path(__file__).resolve().parents[1] / "demo" / "genuine_wrong_date" / "report.json").read_text())
    assert "snowball" not in old
    validate_report(old)
    validate_report(_demo())


def test_validate_snowball_and_spread_rules():
    r = _demo()
    m = r["visual_matching"]["proven_matches"][0]
    r["snowball"] = dict(rounds=[dict(round=1, seed_copy=m["url"], queries=[dict(q="x y z", rationale="why", source_field="title")], candidates=3, verified=3, proven_new=[])],
                         stopped_reason="round 1 found no new proven copy", cost=dict(openrouter_usd=0.01, lens_calls=1, verifications=3), seconds=40.0)
    validate_report(r)
    bad = json.loads(json.dumps(r))
    bad["snowball"]["rounds"][0]["queries"][0]["rationale"] = ""
    with pytest.raises(AssertionError):
        validate_report(bad)
    bad = json.loads(json.dumps(r))
    bad["snowball"]["rounds"][0]["proven_new"] = ["https://not.proven/x"]
    with pytest.raises(AssertionError):
        validate_report(bad)
    bad = json.loads(json.dumps(r))
    bad["snowball"]["cost"]["verifications"] = 13
    bad["snowball"]["rounds"][0]["verified"] = 13
    with pytest.raises(AssertionError):
        validate_report(bad)
    r2 = json.loads(json.dumps(r))
    add_clarity(r2)
    assert r2["spread"][0]["url"] == m["url"] and r2["spread"][0]["likely_parent"] is None
    validate_report(r2)
    r2["spread"][0]["likely_parent"] = dict(url=m["url"], label="likely derived, not proof", reasons=["a", "b"])
    with pytest.raises(AssertionError):
        validate_report(r2)
