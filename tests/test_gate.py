"""Relevance gate: decision rule, fail-open, caching, stop report validity, web force flag. No network."""
import json

import pytest

from floodproof import gate, pipeline, report


def _ans(**kw):
    a = dict(relevant="no", kind="music_video", shows_disaster_or_emergency=False, reason="a pop video", refs=[1, 3], confidence=0.95)
    a.update(kw)
    return a


def test_confident_no_stops():
    v, n = gate.decide(_ans(), {})
    assert v == "stop" and n["kind"] == "music_video" and n["refs"] == [1, 3]


@pytest.mark.parametrize("kw", [dict(confidence=0.79), dict(refs=[]), dict(relevant="unsure"), dict(relevant="yes"),
                                dict(shows_disaster_or_emergency=True), dict(confidence="high")])
def test_everything_else_runs(kw):
    assert gate.decide(_ans(**kw), {})[0] == "run"


def test_disaster_keywords_in_metadata_veto_a_stop():
    assert gate.decide(_ans(), {"disaster_keywords": ["flood"]})[0] == "run"


def test_bad_kind_normalised():
    assert gate.decide(_ans(kind="weird"), {})[1]["kind"] == "other"


def test_meta_keywords():
    m = gate._meta_signals({"title": "Nepal flood tribute", "tags": ["song"], "categories": ["Music"], "duration": 200})
    assert "flood" in m["disaster_keywords"] and m["category"] == ["Music"]
    assert gate._meta_signals({"title": "Cooking pasta"})["disaster_keywords"] == []


def _patch(monkeypatch, tmp_path, answer, calls):
    monkeypatch.setattr(gate, "GATE_CACHE", tmp_path / "gate")
    monkeypatch.setattr(gate, "probe", lambda url: ({"title": "x", "duration": 100, "disaster_keywords": []}, "http://stream", "VID123"))
    monkeypatch.setattr(gate, "grab_frames", lambda src, dur, n=4: [b"a" * 2000] * 4)

    def ask(frames, meta):
        calls.append(1)
        return answer, "meta/muse-glimmer-30b", 0.001
    monkeypatch.setattr(gate, "_ask", ask)


def test_check_stop_cache_and_force(monkeypatch, tmp_path):
    calls = []
    _patch(monkeypatch, tmp_path, _ans(), calls)
    g = gate.check("https://youtu.be/VID123")
    assert g["decision"] == "stop" and g["cost"] > 0 and len(calls) == 1
    g2 = gate.check("https://youtu.be/VID123")                    # cached by video id: no second model call
    assert g2["decision"] == "stop" and g2["cost"] == 0 and g2["cached"] and len(calls) == 1
    g3 = gate.check("https://youtu.be/VID123", force=True)
    assert g3["decision"] == "run_forced"


def test_check_fails_open(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "GATE_CACHE", tmp_path / "gate")
    monkeypatch.setattr(gate, "probe", lambda url: (_ for _ in ()).throw(RuntimeError("net down")))
    g = gate.check("https://youtu.be/zzz", log=lambda *a: None)
    assert g["decision"] == "run" and "net down" in g["error"]


def test_model_fallback(monkeypatch):
    seen = []

    def chat(model, content, max_tokens=0, timeout=0):
        seen.append(model)
        if model == gate.PRIMARY_MODEL:
            raise RuntimeError("down")
        return json.dumps(_ans()), 0.002
    monkeypatch.setattr(gate.llm, "chat", chat)
    ans, model, cost = gate._ask([b"x" * 10] * 4, {})
    assert seen == [gate.PRIMARY_MODEL, gate.FALLBACK_MODEL] and model == gate.FALLBACK_MODEL and ans["relevant"] == "no"


def test_stop_report_is_valid_and_headlined(tmp_path):
    g = dict(decision="stop", kind="music_video", reason="a pop video", refs=[1, 3], confidence=0.95, cost=0.001, seconds=3.0,
             metadata_signals={"title": "t"})
    rep = pipeline._gate_stop_report("https://youtu.be/x", g, "id1", tmp_path, 0)
    report.validate_report(rep)
    assert rep["headline"]["sentence"] == "This looks like a music video (frames 1,3: a pop video). FloodProof checks disaster footage."
    assert rep["final_verdict"]["label"] == "UNVERIFIED" and rep["gate"]["decision"] == "stop"


def test_old_reports_without_gate_still_validate_and_bad_gate_rejected():
    with pytest.raises(AssertionError):
        report.validate_gate(dict(decision="stop", kind="x", reason="y", refs=[1], confidence=0.5, cost=0, seconds=0))
    with pytest.raises(AssertionError):
        report.validate_gate(dict(decision="maybe", cost=0, seconds=0))
    report.validate_gate(dict(decision="run", cost=0, seconds=0))


def test_web_passes_force(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from floodproof import web
    got = {}
    monkeypatch.setattr(web, "UPLOADS", tmp_path)
    monkeypatch.setattr(web, "_run_job", lambda *a, **k: got.update(a=a))
    TestClient(web.create_app()).post("/api/investigate", data={"url": "https://youtu.be/x", "force": "1"})
    import time
    time.sleep(0.2)
    web.JOBS.clear()
    assert got["a"][-1] is True
