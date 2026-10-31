"""Clarity layer: self-disclosure, claim presence, headline, signals-agree, event-date and place inference, USGS, GDELT, YouTube.
No network, no LLM. None of it may change the verdict label."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from floodproof import infer, pipeline, physical as P, sources as S
from floodproof.infer import claim_presence, detect_disclosure, infer_claim
from floodproof.report import build_report, validate_report

STATIC = Path(__file__).resolve().parents[1] / "src" / "floodproof" / "static" / "index.html"


# ---------------------------------------------------------------- A. disclosure and claim presence
def test_disclosure_title_ai_generated_and_tribute():
    t = "Nepal Before & After the Flash Flood | AI-Generated Video | A Tribute to Nepal"
    d = detect_disclosure(t, "Nepal, a land of mountains.", [])
    assert d["type"] == "ai_generated" and d["evidence"][0]["field"] == "title" and "AI-Generated" in d["evidence"][0]["text"]
    cp = claim_presence(t, None, t, d)
    assert cp["value"] == "tribute_or_explainer" and cp["reason"]


@pytest.mark.parametrize("desc,typ", [("Great clip #aiart #flood", "ai_generated"), ("Made with Sora from a prompt", "ai_generated"),
                                      ("created using Veo 3", "ai_generated"), ("a 2D animation documentary", "animation"),
                                      ("a reenactment of the day", "reenactment"), ("illustrative footage only", "reenactment"),
                                      ("computer simulation of the dam", "reenactment")])
def test_disclosure_patterns(desc, typ):
    assert detect_disclosure(None, desc, [])["type"] == typ


@pytest.mark.parametrize("txt", ["This is not AI-generated, it is real", "The runway was flooded at the airport", "A tribute to the victims",
                                 "Rare Video: Japan Tsunami | National Geographic", "no animation, real camera"])
def test_disclosure_conservative(txt):
    assert detect_disclosure(txt, txt, [])["type"] == "none"


def test_disclosure_visible_text_and_priority():
    d = detect_disclosure("Flood", "an animation", [dict(text="AI generated video", refs=[1])])
    assert d["type"] == "ai_generated"                              # ai_generated outranks animation
    assert {e["field"] for e in d["evidence"]} == {"on_screen_text"}
    assert detect_disclosure(None, None, [dict(text="Made with Kling AI")])["type"] == "ai_generated"


def test_claim_presence_values():
    assert claim_presence("Nepal GLOF Explained", None, None, dict(type="none"))["value"] == "tribute_or_explainer"
    assert claim_presence("BREAKING: flood caught on camera in Kathmandu", None, None, dict(type="none"))["value"] == "asserts_real_event"
    assert claim_presence("Rare Video: Japan Tsunami", None, "Rare Video: Japan Tsunami", dict(type="none"))["value"] == "asserts_real_event"
    assert claim_presence(None, None, None, dict(type="none"))["value"] == "no_claim"


# ---------------------------------------------------------------- report helpers
def proven_copy(date="2020-01-01", independent=True, adapter="yandex"):
    return dict(url="https://youtu.be/early", upload_date=date, date_source="yt-dlp live metadata" if independent else "memory entry (inherited)", retrieved_at="t", adapter=adapter,
                stored_date=None, independent_date=date, independent_date_source="yt-dlp live metadata", date_independent=independent,
                match=dict(consistent_frames=9, max_inliers=120, temporal_offset=0.0, mirrored=False, best_pairs=[]))


def japan_clues(extra=None):
    c = dict(status="ok", media_nature=dict(value="news_compilation", confidence=.8, cited_frames=[1], reasons=[], status="AI_OBSERVATION_HYPOTHESIS"),
             visible_text=[dict(text="NHK", refs=[2], is_place=False), dict(text="LOCATION WAVE AT SEA", refs=[3], is_place=False)],
             on_screen_dates=[], observations_against_claim=[], claim_parsed=dict(place="Japan", country_iso2="JP", event_type="tsunami"),
             audio=dict(status="ok", has_speech=True, spoken_dates=[dict(date="2011-03-11", text="March 11th, 2011", audio_t=8.5)],
                        partial_dates=[], spoken_place_names=[dict(name="Kesennuma", audio_t=32.0), dict(name="Japan", audio_t=71.0)]))
    c.update(extra or {})
    return c


JAPAN_CLAIM = dict(text="Rare Video: Japan Tsunami", event_date="2011-03-11", place="Japan", event_type="tsunami", country_iso2="JP",
                   event_date_basis="spoken_in_narration", source=dict(text="inferred_from_post", event_date="inferred_from_post", place="inferred_from_post"),
                   specific_place=dict(name="Kesennuma", source="narration"))
JAPAN_PHYS = dict(verdict="SUPPORTS", reasons=["USGS"], links=[], limits=["x"],
                  location=dict(lat=38.9, lon=141.6, country_level=False, name="Kesennuma"),
                  official_event=dict(status="NO_RECORD", activations=[], usgs=dict(status="MATCH", events=[dict(mag=9.1, place="off Tohoku", distance_km=130)])))


def report(claim=JAPAN_CLAIM, clues=None, physical=JAPAN_PHYS, proven=(), title="Rare Video: Japan Tsunami | National Geographic", desc="June 9, 2011 - The March 11 earthquake"):
    proven = list(proven)
    visual = dict(status="FOUND" if proven else "NOT_FOUND", proven_matches=proven, candidates_checked=14)
    r = build_report(input_meta=dict(source="https://youtu.be/x", title=title, description=desc, url="https://youtu.be/x"), claim=claim,
                     memory=dict(status="MISS"), visual=visual, clues=japan_clues() if clues is None else clues, physical=physical)
    return validate_report(r)


# ---------------------------------------------------------------- B. headline precedence
def test_ai_disclosed_headline_label_unchanged():
    t = "Nepal Before & After the Flash Flood | AI-Generated Video | A Tribute to Nepal"
    claim = dict(text=t, event_date="2026-09-08", place="Nepal", event_type="flash_flood", event_date_basis="date_posted")
    r = report(claim, clues=dict(status="not_run"), physical=dict(status="NOT_RUN"), title=t, desc="a land of mountains")
    assert r["final_verdict"]["label"] == "UNVERIFIED"
    hl = r["headline"]
    assert hl["tone"] == "ai_disclosed" and hl["sentence"].startswith("The uploader labels this video AI-generated. It is not documentary footage of")
    assert "flash flood" in hl["sentence"] and 1 <= len(hl["facts"]) <= 3 and all(f["field"] and f["text"] for f in hl["facts"])
    assert r["disclosure"]["type"] == "ai_generated" and r["claim_presence"]["value"] == "tribute_or_explainer"


def test_recycled_beats_disclosure():
    claim = dict(text="flood", event_date="2024-09-28", place="Nepal", event_type="flood", event_date_basis="user")
    r = report(claim, clues=dict(status="not_run"), physical=dict(status="NOT_RUN"), proven=[proven_copy("2020-01-01")], title="AI generated flood", desc="")
    assert r["final_verdict"]["label"] == "RECYCLED_FOOTAGE" and r["headline"]["tone"] == "recycled"
    assert "2020-01-01" in r["headline"]["sentence"] and r["disclosure"]["type"] == "ai_generated"


def test_unverified_plain_headline_and_facts_cap():
    claim = dict(text="flood", event_date="2024-09-28", place="Kathmandu", event_type="flood", event_date_basis="user")
    r = report(claim, clues=dict(status="not_run"), physical=dict(status="NOT_RUN"), title="Flood video", desc="")
    assert r["headline"]["tone"] == "unverified" and len(r["headline"]["facts"]) == 3


def test_headline_validation_rejects_bad_shapes():
    r = report()
    r["headline"]["facts"] = r["headline"]["facts"] + [dict(text="x", field="y")] * 3
    with pytest.raises(AssertionError):
        validate_report(r)
    r = report(); r["headline"]["tone"] = "recycled"
    with pytest.raises(AssertionError):
        validate_report(r)


# ---------------------------------------------------------------- C. signals-agree
def test_signals_agree_japan_label_stays_unverified():
    r = report()
    sg = r["signals"]
    assert sg["qualifies"] and sg["count"] >= 3
    assert {s["key"] for s in sg["agree"]} >= {"date", "place", "broadcaster", "official_event", "no_earlier_copy"}
    assert all(s["label"] and s["detail"] for s in sg["agree"])
    assert r["final_verdict"]["label"] == "UNVERIFIED"             # signals and physical SUPPORTS never raise the label
    hl = r["headline"]
    assert hl["tone"] == "signals_agree" and hl["sentence"] == \
        f"Signals point to genuine footage of the Japan tsunami ({sg['count']} signals agree). Not proven by a dated copy."
    assert len(hl["facts"]) == 3


def test_two_signals_are_not_enough():
    clues = japan_clues(dict(visible_text=[], audio=dict(status="ok", has_speech=True, spoken_dates=[], partial_dates=[], spoken_place_names=[])))
    r = report(clues=clues, physical=dict(status="NOT_RUN"))
    assert r["signals"]["count"] == 1 and not r["signals"]["qualifies"] and r["headline"]["tone"] == "unverified"


def test_signals_blocked_by_contradiction_or_disclosure():
    r = report(physical=dict(JAPAN_PHYS, verdict="CONTRADICTS", links=["https://x.org/d"]))
    assert not r["signals"]["qualifies"] and r["signals"]["against"]
    r = report(clues=japan_clues(dict(media_nature=dict(value="likely_ai_generated", confidence=.9, cited_frames=[1], reasons=[], status="AI_OBSERVATION_HYPOTHESIS"))))
    assert not r["signals"]["qualifies"] and r["headline"]["tone"] != "signals_agree"


def test_signals_never_produce_consistent_or_recycled():
    r = report()
    assert r["final_verdict"]["label"] not in ("CONSISTENT_WITH_CLAIM", "RECYCLED_FOOTAGE")
    # an earlier INHERITED-date copy stops the absence signal and still cannot create RECYCLED
    r2 = report(proven=[proven_copy("2005-01-01", independent=False, adapter="memory")])
    assert r2["final_verdict"]["label"] == "UNVERIFIED"


# ---------------------------------------------------------------- D. event date from narration
def clip(title, desc, up="2011-06-13", url="https://youtu.be/x"):
    return SimpleNamespace(title=title, description=desc, upload_date=up, url=url)


def test_event_date_from_narration_beats_description_publish_date():
    cl = dict(status="ok", claim_parsed=dict(place="Japan", country_iso2="JP"), visible_text=[], on_screen_dates=[],
              audio=dict(status="ok", has_speech=True, spoken_dates=[dict(date="2011-03-11", text="March 11th, 2011", audio_t=8.5)],
                         partial_dates=[], spoken_place_names=[]))
    c = infer_claim(clip("Rare Video: Japan Tsunami", "June 9, 2011 - The March 11 earthquake and tsunami left 28,000 dead."), None, None, None, cl)
    assert c["event_date"] == "2011-03-11" and c["event_date_basis"] == "spoken_in_narration"
    assert "event date from narration; the description date is a publishing date" in c["event_date_basis_detail"]
    assert c["publication_date"]["date"] == "2011-06-09" and c["publication_date"]["basis"] == "description"      # both kept


def test_event_date_from_description_partial_date_when_no_narration():
    cl = dict(status="ok", claim_parsed=dict(place="Japan"), visible_text=[], on_screen_dates=[])
    c = infer_claim(clip("Rare Video: Japan Tsunami", "June 9, 2011 - The March 11 earthquake and tsunami."), None, None, None, cl)
    assert c["event_date"] == "2011-03-11" and c["event_date_basis"] == "description" and c["publication_date"]["date"] == "2011-06-09"


def test_old_description_date_is_kept_when_far_from_upload():
    cl = dict(status="ok", claim_parsed=dict(place="Japan"), visible_text=[], on_screen_dates=[],
              audio=dict(status="ok", has_speech=True, spoken_dates=[dict(date="2011-03-11", text="x", audio_t=1.0)], partial_dates=[], spoken_place_names=[]))
    c = infer_claim(clip("Tsunami", "Filmed on January 5, 2011", up="2012-06-13"), None, None, None, cl)
    assert c["event_date"] == "2011-01-05" and c["event_date_basis"] == "description" and "publication_date" not in c


def test_later_narration_date_is_not_used():
    cl = dict(status="ok", claim_parsed=dict(place="Japan"), visible_text=[], on_screen_dates=[],
              audio=dict(status="ok", has_speech=True, spoken_dates=[dict(date="2011-07-01", text="x", audio_t=1.0)], partial_dates=[], spoken_place_names=[]))
    c = infer_claim(clip("Tsunami", "June 9, 2011"), None, None, None, cl)
    assert c["event_date"] == "2011-06-09"


# ---------------------------------------------------------------- E. specific place inside a country
def test_country_level_helper():
    assert infer.is_country_level_place("Japan", "JP") and infer.is_country_level_place("japan") and not infer.is_country_level_place("Kesennuma", "JP")


def test_physical_uses_specific_place_named_in_narration(monkeypatch):
    seen = []
    monkeypatch.setattr(P, "geocode", lambda place, cc=None: (seen.append(place), dict(name=place + ", Japan", lat=39.0, lon=141.6, confidence="high", country_level=False))[1])
    monkeypatch.setattr(pipeline, "run_physical", lambda place, lat, lon, date, w, et, cc: dict(status="OK", verdict="NO_EVIDENCE", location=dict(lat=lat, lon=lon, name=place, country_level=False),
                                                                                           reasons=[], limits=[], links=[]))
    clues = japan_clues()
    claim = dict(place="Japan", country_iso2="JP", event_type="tsunami", text="x")
    res = pipeline._physical(claim, clues, "2011-03-11")
    assert seen == ["Kesennuma"]                                  # "Japan" itself is never the physical-check location
    assert res["location"]["source"] == "narration" and res["location"]["fallback_from"] == "Japan"
    assert claim["specific_place"]["name"] == "Kesennuma" and claim["specific_place"]["source"] == "narration"


def test_physical_keeps_country_when_nothing_specific(monkeypatch):
    monkeypatch.setattr(pipeline, "run_physical", lambda place, *a: dict(status="OK", verdict="UNKNOWN", location=dict(lat=36.0, lon=139.0, name=place, country_level=True),
                                                                          reasons=[], limits=[], links=[]))
    res = pipeline._physical(dict(place="Japan", country_iso2="JP", event_type="tsunami"), dict(status="not_run"), "2011-03-11")
    assert res["location"]["country_level"] is True and "specific_place" not in res


# ---------------------------------------------------------------- F. USGS and new event types
USGS_QUAKE = {"features": [{"id": "official20110311054624120_30", "properties": {"mag": 9.1, "place": "near the east coast of Honshu, Japan",
                                                                                   "time": 1299822384120, "url": "https://earthquake.usgs.gov/earthquakes/eventpage/official20110311054624120_30", "tsunami": 1},
                            "geometry": {"type": "Point", "coordinates": [142.37, 38.3, 29.0]}}]}


@pytest.fixture
def ph(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    return P


def test_usgs_events_query_and_cache(ph, monkeypatch):
    calls = []

    def http(url, params):
        calls.append((url, params))
        return USGS_QUAKE
    monkeypatch.setattr(P, "_http_json", http)
    from datetime import date
    r = P.usgs_events(38.9, 141.6, date(2011, 3, 11), 2)
    assert r["status"] == "MATCH" and r["events"][0]["mag"] == 9.1 and r["events"][0]["distance_km"] < 500 and r["events"][0]["url"].startswith("https://earthquake.usgs.gov/")
    u, p = calls[0]
    assert u == "https://earthquake.usgs.gov/fdsnws/event/1/query" and p["format"] == "geojson" and p["minmagnitude"] == 5.0 and p["maxradiuskm"] == 500
    assert p["starttime"] == "2011-03-09" and p["endtime"] == "2011-03-14"
    P.usgs_events(38.9, 141.6, date(2011, 3, 11), 2)
    assert len(calls) == 1                                         # second call served from the cache


def test_usgs_error_never_raises(ph, monkeypatch):
    def boom(url, params):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(P, "_http_json", boom)
    from datetime import date
    r = P.usgs_events(1.0, 1.0, date(2011, 3, 11))
    assert r["status"] == "ERROR" and "secret" not in json.dumps(r)


def stub_other_sources(monkeypatch, events=USGS_QUAKE):
    monkeypatch.setattr(P, "official_event", lambda *a: dict(source="copernicus_ems", status="NO_RECORD", activations=[]))
    monkeypatch.setattr(P, "reliefweb", lambda *a, **k: dict(source="reliefweb", status="SKIPPED", disasters=[], window_days=7))
    monkeypatch.setattr(P, "news_coverage", lambda *a, **k: dict(source="gdelt", status="SKIPPED"))
    monkeypatch.setattr(P, "hydrology", lambda *a: (_ for _ in ()).throw(AssertionError("river discharge must be skipped")))
    monkeypatch.setattr(P, "_http_json", lambda url, params: events)


def test_tsunami_supports_via_usgs_and_skips_discharge(ph, monkeypatch):
    stub_other_sources(monkeypatch)
    r = P.check("Kesennuma", 38.9, 141.6, "2011-03-11", event_type="tsunami", country="JP")
    assert r["verdict"] == "SUPPORTS" and r["official_event"]["usgs"]["status"] == "MATCH"
    assert any("USGS earthquake M9.1" in x and "https://earthquake.usgs.gov/" in x for x in r["reasons"])
    assert any(l.startswith("https://earthquake.usgs.gov/") for l in r["links"])
    assert r["hydrology"]["status"] == "SKIPPED" and not any("flood-api" in l for l in r["links"])


def test_wildfire_discharge_not_applicable_and_no_evidence(ph, monkeypatch):
    stub_other_sources(monkeypatch)
    r = P.check("Paradise", 39.7, -121.6, "2018-11-08", event_type="wildfire", country="US")
    assert r["hydrology"]["status"] == "NOT_APPLICABLE" and r["verdict"] == "NO_EVIDENCE" and "usgs" not in r["official_event"]


def test_earthquake_no_match_is_no_evidence_never_contradicts(ph, monkeypatch):
    stub_other_sources(monkeypatch, events={"features": []})
    r = P.check("Kesennuma", 38.9, 141.6, "2011-03-11", event_type="earthquake", country="JP")
    assert r["verdict"] == "NO_EVIDENCE" and r["official_event"]["usgs"]["status"] == "NO_MATCH"


def test_new_event_types_registered():
    from floodproof.report import EVENT_TYPES
    from floodproof import investigator
    for t in ("tsunami", "earthquake", "wildfire"):
        assert t in EVENT_TYPES and t in P.EVENT_TYPES and f'"{t}"' in investigator.SCHEMA and t in P.NOT_RESOLVED_BY_GLOFAS


# ---------------------------------------------------------------- G. YouTube search
def test_youtube_query_diversity_relevance_pass_and_short_fallback(monkeypatch):
    calls = []

    def get(url, params=None, timeout=60):
        calls.append(dict(params))
        q = params["q"]
        if q == "釜石市 津波 防波堤 越波 2011 映像 NHK":      # over-specific local query: nothing
            return {"items": []}
        if params.get("order") == "relevance":
            return {"items": [{"id": {"videoId": "REL00000001"}, "snippet": {"title": "t", "publishedAt": "2011-03-12T01:00:00Z"}}]}
        return {"items": []}
    monkeypatch.setattr(S, "_get_json", get)
    qs = ["釜石市 津波 防波堤 越波 2011 映像 NHK", "気仙沼 津波 2011", "Japan tsunami March 11 footage", "Kamaishi tsunami seawall"]
    c, info = S.youtube_api_candidates(qs, "2011-03-11", key="K")
    assert [x["q"] for x in calls[:2]] == [qs[0], qs[0]] and calls[0]["order"] == "date" and calls[1]["order"] == "relevance"
    assert any(x["q"] == "Japan tsunami March 11 footage" for x in calls)       # an English query is inside the budget of 3
    assert all(x["publishedBefore"] == "2011-03-12T00:00:00Z" for x in calls)
    assert info["zero_result_queries"] == [qs[0][:80]] and any(x["q"] == "釜石市 津波 防波堤" for x in calls)   # shortened retry
    assert [x.url for x in c] == ["https://www.youtube.com/watch?v=REL00000001"] and c[0].stored_date == "2011-03-12"


# ---------------------------------------------------------------- H. debunk bullets (data vs UI field names)
def test_debunk_refs_never_render_empty_bullets():
    import shutil, subprocess
    html = STATIC.read_text()
    assert "x.sentence" in html and "function refItems" in html
    node = shutil.which("node")
    if node:
        fn = re.search(r"function refItems\(d, lg\) \{.*?\n\}\n", html, re.S).group(0)
        js = fn + """
const cases = [
  {field_refs: {en: [{sentence: "A.", ref: "claim.text"}, {text: "B.", ref: "x"}, "C.", {ref: "only-ref"}, {sentence: "  "}]}, texts: {en: "unused."}},
  {field_refs: {en: []}, texts: {en: "First one. Second one!"}},
  {texts: {en: "Only text."}},
];
console.log(JSON.stringify(cases.map((c) => refItems(c, "en").map((x) => x.sentence))));"""
        out = subprocess.run([node, "-e", js], capture_output=True, text=True, check=True).stdout.strip()
        assert json.loads(out) == [["A.", "B.", "C."], ["First one.", "Second one!"], ["Only text."]]
    d = json.loads((Path(__file__).resolve().parents[1] / "demo" / "success_recycled" / "report.json").read_text()).get("debunk")
    assert d and all(x["sentence"].strip() and x["ref"] for refs in d["field_refs"].values() for x in refs)


# ---------------------------------------------------------------- I. unconfigured / failing sources
def test_unconfigured_sources_marked_not_configured(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(S, "load_env", lambda: None)
    c, info = S.gvision_candidates(SimpleNamespace(frames=[], times=[]))
    assert c == [] and info["status"] == "not_configured"


def test_reliefweb_403_is_not_configured(ph, monkeypatch):
    def deny(url, body):
        raise httpx.HTTPStatusError("403 Forbidden", request=httpx.Request("POST", url), response=httpx.Response(403))
    monkeypatch.setattr(P, "_http_post_json", deny)
    from datetime import date
    r = P.reliefweb("JP", date(2011, 3, 11), 7, "tsunami")
    assert r["status"] == "not_configured"


def gd_status(code):
    return httpx.HTTPStatusError(str(code), request=httpx.Request("GET", "u"), response=httpx.Response(code))


def test_gdelt_retries_on_429_and_empty_body_then_succeeds(ph, monkeypatch):
    seq = [gd_status(429), ValueError("empty"), {"articles": [{"url": "https://a/1", "title": "t", "domain": "a"}]}]
    sleeps = []

    def http(url, params):
        v = seq.pop(0) if len(seq) else {"articles": []}
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr(P, "_http_json", http)
    monkeypatch.setattr(P.time, "sleep", lambda s: sleeps.append(s))
    n, arts = P._gdelt_count("tsunami", P.datetime(2011, 3, 9), P.datetime(2011, 3, 14))
    assert n == 1 and len([s for s in sleeps if s >= 6]) == 2             # two backoff waits


def test_gdelt_failure_shows_friendly_note_and_caches_success(ph, monkeypatch):
    monkeypatch.setattr(P, "_http_json", lambda u, p: (_ for _ in ()).throw(gd_status(429)))
    from datetime import date
    r = P.news_coverage("Kesennuma", date(2011, 3, 11), 3, "tsunami")
    assert r["status"] == "UNAVAILABLE" and r["reason"] == "news coverage unavailable right now" and "429" not in json.dumps(r)
    calls = []
    monkeypatch.setattr(P, "_http_json", lambda u, p: (calls.append(1), {"articles": []})[1])
    P._gdelt_count("zz", P.datetime(2011, 3, 9), P.datetime(2011, 3, 14))
    P._gdelt_count("zz", P.datetime(2011, 3, 9), P.datetime(2011, 3, 14))
    assert len(calls) == 1 and P.GDELT_TTL_S == 12 * 3600


def test_ui_hides_unconfigured_and_raw_errors():
    html = STATIC.read_text()
    for needle in ('"not_configured"', "news coverage unavailable right now", "The physical-world check could not be completed right now.",
                   "Signals checked (", "Full summary and what this does NOT mean", "Limitations ("):
        assert needle in html, needle
    assert "(cl.error ||" not in html and '"Failed: " + j.error' not in html and "au.error ||" not in html


# ---------------------------------------------------------------- K. source map
def sm_report(proven, labels, n=6, claim_date="2026-08-26"):
    times = [float(i * 2) for i in range(n)]
    claim = dict(text="x", event_date=claim_date, place="Nepal", event_type="glof", event_date_basis="user")
    clues = dict(status="ok", media_nature=dict(value="camera_footage", confidence=.8, cited_frames=[0, 5], reasons=[], status="AI_OBSERVATION_HYPOTHESIS",
                                                 per_frame=[dict(frame=i, t=times[i], label=l, confidence=.7, reason="") for i, l in enumerate(labels)]),
                 visible_text=[], on_screen_dates=[], observations_against_claim=[], claim_parsed={})
    r = build_report(input_meta=dict(source="u", url="https://youtu.be/in", n_keyframes=n, keyframe_times_s=times, duration=12.0), claim=claim,
                     memory=dict(status="MISS"), visual=dict(status="FOUND" if proven else "NOT_FOUND", proven_matches=proven), clues=clues,
                     physical=dict(status="NOT_RUN"))
    return validate_report(r)


def copy_with_frames(url, frames, date="2026-08-21"):
    m = proven_copy(date)
    m["url"] = url
    m["match"] = dict(m["match"], matched_frames=[dict(a_frame=f, a_t=f * 2.0, b_frame=f, b_t=f * 2.0 + 1, inliers=50, flip=False) for f in frames])
    return m


def test_source_map_merges_segments_and_never_guesses():
    labels = ["camera_footage", "camera_footage", "camera_footage", "likely_ai_generated", "likely_ai_generated", "graphic_or_text"]
    r = sm_report([copy_with_frames("https://www.instagram.com/reel/A/", [0, 1]), copy_with_frames("https://www.tiktok.com/@u/video/1", [2], "2026-08-22")], labels)
    sm = r["source_map"]
    assert [s["kind"] for s in sm] == ["matched_source", "matched_source", "ai_generated", "graphic"]
    assert sm[0]["frames"] == [0, 1] and sm[0]["sources"][0]["platform"] == "Instagram" and sm[0]["sources"][0]["frames_matched"] == 2
    assert sm[1]["sources"][0]["platform"] == "TikTok" and sm[1]["sources"][0]["date"] == "2026-08-22"
    assert not sm[2]["sources"] and not sm[3]["sources"]                # Muse kinds never carry a source
    assert sm[0]["start_s"] == 0.0 and sm[-1]["end_s"] <= 12.0


def test_source_map_no_match_uses_muse_label_or_unknown():
    r = sm_report([], ["camera_footage", "unknown", "unknown", "animation_or_cgi", "animation_or_cgi", "unknown"])
    assert [s["kind"] for s in r["source_map"]] == ["camera_unmatched", "unknown", "animation", "unknown"]
    assert all(not s["sources"] for s in r["source_map"])


def test_source_map_validation_rejects_invented_source():
    r = sm_report([copy_with_frames("https://www.instagram.com/reel/A/", [0])], ["camera_footage"] * 6)
    r["source_map"][1]["kind"] = "matched_source"
    r["source_map"][1]["sources"] = [dict(url="https://invented.example/x", date="2020-01-01", date_source="x", frames_matched=1)]
    with pytest.raises(AssertionError):
        validate_report(r)
    r2 = sm_report([], ["camera_footage"] * 6)
    r2["source_map"][0]["kind"] = "ai_generated"; r2["source_map"][0]["sources"] = [dict(url="u")]
    with pytest.raises(AssertionError):
        validate_report(r2)


def test_source_map_backward_compatible_and_label_untouched():
    r = report()
    r.pop("source_map", None)
    assert validate_report(r)
    from floodproof.matcher import MatchResult
    assert "matched_frames" in MatchResult(False, 0, 0, False, None, 0, 0.0, []).to_dict()


def test_ui_has_source_map():
    html = STATIC.read_text()
    for needle in ("function sourceMap", "AI observation (Muse per-frame label)", "No source found"):
        assert needle in html


def test_reconstructed_is_not_a_disclosure_and_snippets_keep_whole_words():
    assert detect_disclosure("Nepal flood reconstructed minute by minute | FRANCE 24", "", [])["type"] == "none"
    d = detect_disclosure("Nepal Before & After the Flash Flood | AI-Generated Video | A Tribute to Nepal", "", [])
    assert not d["evidence"][0]["text"].startswith("al ")


def test_specific_place_suffix_is_stripped_for_geocoding(monkeypatch):
    def geo(place, cc=None):
        return dict(name="Kamaishi, Japan", lat=39.27, lon=141.8, confidence="high", country_level=False) if place == "Kamaishi" else dict(lat=None, lon=None, confidence="low")
    monkeypatch.setattr(P, "geocode", geo)
    monkeypatch.setattr(pipeline, "run_physical", lambda place, lat, lon, *a: dict(status="OK", verdict="NO_EVIDENCE", location=dict(lat=lat, lon=lon), reasons=[], limits=[], links=[]))
    clues = japan_clues(dict(audio=dict(status="ok", has_speech=True, spoken_dates=[], partial_dates=[], spoken_place_names=[dict(name="Northern Japan", audio_t=2.5), dict(name="Kamaishi City", audio_t=38.5)])))
    res = pipeline._physical(dict(place="Japan", country_iso2="JP", event_type="tsunami", text="x"), clues, "2011-03-11")
    assert res["location"]["lat"] == 39.27 and res["location"]["query"] == "Kamaishi City"


# ---------------------------------------------------------------- I. investigation replay
def test_replay_is_built_from_report_fields_and_autoplays():
    html = STATIC.read_text()
    assert "function replaySteps(rep, ref)" in html and "function replayCard(rep, ref)" in html
    for field in ("stage_timings_s", "candidate_status_counts", "source_stats", "proven_matches", "rep.snowball", "rep.gate"):
        assert field in html                                   # every step reads a saved report field
    assert "Math.random" not in html.split("function replaySteps")[1].split("function render(")[0]   # nothing simulated
    assert html.count("AUTOPLAY = true") == 3                  # showcase card, the Nepal story button, a finished live run
    assert "rk-proof" in html and ".rp-step.proof" not in html  # step classes cannot collide with the .proof image style
