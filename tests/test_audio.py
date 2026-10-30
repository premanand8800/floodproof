"""Narration analysis, spoken-date/place inference and physical-location fallback. LLM and geocoder mocked."""
from pathlib import Path
from types import SimpleNamespace
from floodproof import investigator, pipeline, infer
from floodproof.infer import infer_claim


def mk_clip(tmp_path, dur=200):
    p = tmp_path / "c.mp4"; p.write_bytes(b"x")
    return SimpleNamespace(path=str(p), has_audio=True, duration=dur, title="Nepal flood", description=None, upload_date="2026-08-28", url="https://youtu.be/x")


def patch(monkeypatch, tmp_path, replies):
    mp3 = tmp_path / "a.mp3"; mp3.write_bytes(b"y" * 2000)
    seen = {}
    monkeypatch.setattr(investigator, "extract_audio", lambda path, out, max_seconds=60: seen.update(max=max_seconds) or mp3)
    it = iter(replies); calls = []
    def fake(model, content, max_tokens=0, **k):
        calls.append(content[0]["text"]); r = next(it)
        if isinstance(r, Exception): raise r
        return r, 0.01
    monkeypatch.setattr(investigator, "chat", fake)
    return seen, calls


GOOD = ('Sure:\n```json\n{"has_speech": true, "language": "hi", "transcript_summary": "s", '
        '"spoken_place_names": [{"name": "Rasuwa", "audio_t": 10.5}, {"name": "Ghost"}], '
        '"spoken_dates": [{"date": "2026-08-26", "text": "26 August 2026", "audio_t": 4}, {"date": "2026-08-27", "text": "no t"}, {"date": "bad", "audio_t": 1}]}\n```')


def test_retry_then_lenient_and_cleaning(monkeypatch, tmp_path):
    seen, calls = patch(monkeypatch, tmp_path, ["", GOOD])
    a = investigator.analyze_audio(mk_clip(tmp_path), tmp_path)
    assert a["status"] == "ok" and a["attempts"] == 2 and len(calls) == 2 and "ONE JSON object" in calls[1]
    assert a["truncated"] is True and seen["max"] == 90
    assert a["spoken_place_names"] == [dict(name="Rasuwa", audio_t=10.5)]          # no audio_t -> dropped
    assert a["spoken_dates"] == [dict(date="2026-08-26", text="26 August 2026", audio_t=4.0)]


def test_two_failures_report_error(monkeypatch, tmp_path):
    patch(monkeypatch, tmp_path, ["no json", ""])
    a = investigator.analyze_audio(mk_clip(tmp_path, dur=30), tmp_path)
    assert a["status"] == "error" and a["truncated"] is False and "no JSON" in a["error"]


def test_spoken_date_order_and_place_grounding():
    aud = dict(status="ok", has_speech=True, spoken_dates=[dict(date="2026-08-26", text="x", audio_t=4.0)],
               spoken_place_names=[dict(name="Rasuwa", audio_t=10.5)])
    cl = dict(status="ok", claim_parsed=dict(place="Rasuwa district"), visible_text=[], on_screen_dates=[], audio=aud)
    c = infer_claim(SimpleNamespace(title="Nepal flood", description=None, upload_date="2026-08-28", url="u"), None, None, None, cl)
    assert (c["event_date"], c["event_date_basis"]) == ("2026-08-26", "spoken_in_narration")
    assert c["source"]["place"] == "inferred_from_narration" and c["place_basis"] == "spoken in narration (audio@10.5)"
    cl["on_screen_dates"] = [dict(date="2026-08-20", text="t", refs=[1])]
    assert infer_claim(SimpleNamespace(title="n", description=None, upload_date="2026-08-28", url="u"), None, None, None, cl)["event_date_basis"] == "on_screen_text"
    cl["claim_parsed"] = dict(place="Atlantis")                                       # not in text or narration
    assert infer_claim(SimpleNamespace(title="n", description=None, upload_date=None, url="u"), None, None, None, cl)["place"] is None


def test_fallback_order_and_ai_threshold():
    aud = dict(status="ok", has_speech=True, spoken_place_names=[dict(name="Nepal", audio_t=1.0), dict(name="Trishuli", audio_t=2.0)])
    ll = dict(place="Langtang", confidence=0.7, reasons=[dict(text="r", refs=[1])])
    cl = dict(status="ok", audio=aud, likely_location=ll, visible_text=[dict(text="Dhunche", refs=[2], is_place=True), dict(text="BREAKING", refs=[2])])
    fb = infer.fallback_places(dict(place="Nepal's Himalayan region"), cl, "Nepal")
    assert [x[0] for x in fb] == ["Trishuli", "Dhunche", "Langtang"]
    assert fb[0][1] == "narration" and fb[1][1] == "on_screen_text" and fb[2][1].startswith("ai_location_hypothesis")
    assert infer.fallback_places({}, dict(cl, likely_location=dict(ll, confidence=0.5)), "Nepal")[-1][1] == "on_screen_text"
    assert not [x for x in infer.fallback_places({}, dict(cl, likely_location=dict(ll, reasons=[])), "Nepal") if x[0] == "Langtang"]


def test_pipeline_physical_fallback_label(monkeypatch):
    from floodproof import physical
    def geo(place, cc=None):
        return dict(lat=28.1, lon=85.3, name="Trishuli", confidence="high", source="nominatim") if place == "Trishuli" else dict(lat=None, lon=None, source="nominatim", confidence="low")
    monkeypatch.setattr(physical, "geocode", geo)
    monkeypatch.setattr(physical, "official_event", lambda *a, **k: dict(status="NONE", activations=[]))
    monkeypatch.setattr(physical, "hydrology", lambda *a, **k: dict(river_discharge=None, rain=None, status="UNKNOWN"))
    aud = dict(status="ok", has_speech=True, spoken_place_names=[dict(name="Trishuli", audio_t=2.0)])
    r = pipeline._physical(dict(place="Nepal's Himalayan region", event_type="glof", country_iso2="NP"), dict(status="ok", audio=aud), "2026-08-26")
    assert r["location"]["source"] == "narration" and r["location"]["lat"] == 28.1 and r["location"]["fallback_from"]


def test_string_timestamps_accepted():
    d = investigator._clean_audio(dict(spoken_place_names=[dict(name="A", audio_t="7.5"), dict(name="B", audio_t="1:05"), dict(name="C", audio_t="x")]))
    assert [x["audio_t"] for x in d["spoken_place_names"]] == [7.5, 65.0] and d["dropped_untimed"] == 1


def test_untimed_first_reply_triggers_retry(monkeypatch, tmp_path):
    untimed = '{"has_speech": true, "language": "hi", "transcript_summary": "s", "spoken_place_names": [{"name": "Rasuwa"}]}'
    _, calls = patch(monkeypatch, tmp_path, [untimed, GOOD])
    a = investigator.analyze_audio(mk_clip(tmp_path), tmp_path)
    assert len(calls) == 2 and a["spoken_place_names"][0]["name"] == "Rasuwa"
