"""Meta Model API routing (mocked HTTP): success, fallback to OpenRouter, key never leaks; transcript -> extraction narration path."""
import json
from types import SimpleNamespace
import pytest
from floodproof import llm, investigator

KEY = "sk-meta-SECRET-1234567890"
OR_KEY = "sk-or-SECRET-0987654321"


class Resp:
    def __init__(self, status, body):
        self.status_code, self._b = status, body
    def json(self):
        return self._b


def ok_body(txt="hello"):
    return {"choices": [{"message": {"content": txt}}], "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.001}}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("FLOODPROOF_META_API", "1")
    monkeypatch.setenv("META_API_KEY", KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_KEY)
    monkeypatch.setattr(llm, "load_env", lambda: None)


def test_meta_success(env, monkeypatch):
    calls = []
    monkeypatch.setattr(llm.requests, "post", lambda url, **k: calls.append((url, k)) or Resp(200, ok_body("from meta")))
    txt, cost = llm.chat("meta/muse-spark-1.3", [{"type": "text", "text": "x"}])
    assert txt == "from meta" and llm.last_provider() == "meta_api" and cost > 0
    assert calls[0][0].startswith("https://api.meta.ai/v1/chat/completions") and calls[0][1]["json"]["model"] == "muse-spark-1.3"


@pytest.mark.parametrize("bad", [Resp(402, {"error": {"code": "billing_not_configured", "message": "pay " + KEY}}), Resp(429, {"error": "slow"}),
                                 Resp(200, {"choices": [{"message": {"content": ""}}]}), RuntimeError("timeout " + KEY)])
def test_meta_error_falls_back_to_openrouter(env, monkeypatch, bad):
    urls = []
    def post(url, **k):
        urls.append(url)
        if "api.meta.ai" in url:
            if isinstance(bad, Exception): raise bad
            return bad
        return Resp(200, ok_body("from openrouter"))
    monkeypatch.setattr(llm.requests, "post", post)
    txt, _ = llm.chat("meta/muse-spark-1.3", [{"type": "text", "text": "x"}])
    assert txt == "from openrouter" and llm.last_provider() == "openrouter" and "openrouter.ai" in urls[-1]


def test_other_models_and_switch_stay_on_openrouter(env, monkeypatch):
    urls = []
    monkeypatch.setattr(llm.requests, "post", lambda url, **k: urls.append(url) or Resp(200, ok_body()))
    llm.chat("meta/muse-spark-open", [{"type": "text", "text": "x"}])
    monkeypatch.setenv("FLOODPROOF_META_API", "0")
    llm.chat("meta/muse-spark-1.3", [{"type": "text", "text": "x"}])
    assert all("openrouter.ai" in u for u in urls) and llm.last_provider() == "openrouter"


def test_key_never_in_error_or_log(env, monkeypatch, capsys):
    monkeypatch.setattr(llm.requests, "post", lambda url, **k: Resp(401, {"error": "bad " + (KEY if "meta" in url else OR_KEY)}))
    with pytest.raises(Exception) as e:
        llm.chat("meta/muse-spark-1.3", [{"type": "text", "text": "x"}])
    out = capsys.readouterr()
    assert KEY not in str(e.value) and KEY not in out.out + out.err
    with pytest.raises(Exception) as e2:
        llm.transcribe_meta(__file__)
    assert KEY not in str(e2.value)


def test_provider_in_clues_and_explanation(env, monkeypatch, tmp_path):
    import numpy as np
    from floodproof import explain
    monkeypatch.setattr(llm.requests, "post", lambda url, **k: Resp(200, ok_body('{"items":[{"what":"logo","region_a":"tl","region_b":"tl"}],"differences":[]}')))
    import cv2
    p = tmp_path / "a.jpg"; cv2.imwrite(str(p), np.zeros((20, 20, 3), np.uint8))
    ex = explain.explain_proof(p, p)
    assert ex and ex["provider"] == "meta_api" and KEY not in json.dumps(ex)


TURNS = {"transcript": "On March 11th, 2011 ... Kesennuma ... Kamaishi", "audioDurationMs": 30000,
         "turns": [{"turnId": 1, "startMs": 1000, "endMs": 6000, "transcript": "On March 11th, 2011 a huge earthquake struck."},
                   {"turnId": 2, "startMs": 8200, "endMs": 14000, "transcript": "The tsunami hit Kesennuma and Kamaishi."}]}


def test_transcribe_parses_turns(env, monkeypatch, tmp_path):
    w = tmp_path / "a.wav"; w.write_bytes(b"RIFF")
    seen = {}
    monkeypatch.setattr(llm.requests, "post", lambda url, **k: seen.update(url=url, k=k) or Resp(200, TURNS))
    t = llm.transcribe_meta(w)
    assert seen["url"].endswith("/asr/transcribe") and t["turns"][1]["start_s"] == 8.2 and len(t["turns"]) == 2


def test_transcript_to_extraction_path(monkeypatch, tmp_path):
    clip = SimpleNamespace(path=str(tmp_path / "c.mp4"), has_audio=True, duration=30)
    wav = tmp_path / "a.wav"; wav.write_bytes(b"y" * 2000)
    monkeypatch.setattr(investigator, "extract_wav", lambda *a, **k: wav)
    monkeypatch.setattr(investigator, "meta_enabled", lambda: True)
    prompts = []
    def fake_chat(model, content, max_tokens=0, **k):
        prompts.append(content[0]["text"]); assert model == "meta/muse-spark-open"
        return json.dumps({"language": "en", "spoken_place_names": [{"name": "Kesennuma", "audio_t": 9.0}, {"name": "Kamaishi", "audio_t": 8.4}],
                           "spoken_dates": [{"date": "2011-03-11", "text": "March 11th, 2011", "audio_t": 1.2}]}), 0.002
    a = investigator.analyze_audio_transcript(clip, tmp_path, transcribe=lambda w: {"turns": [dict(start_s=1.0, end_s=6.0, text="On March 11th, 2011 a huge earthquake struck."),
                                              dict(start_s=8.2, end_s=14.0, text="The tsunami hit Kesennuma and Kamaishi.")]}, chat_fn=fake_chat)
    assert a["status"] == "ok" and a["transcriber"] == "muse-voice-transcribe-1.0" and a["timestamps"] == "turn-level"
    assert [(x["name"], x["audio_t"]) for x in a["spoken_place_names"]] == [("Kesennuma", 8.2), ("Kamaishi", 8.2)]     # snapped to the turn start
    assert a["spoken_dates"][0]["date"] == "2011-03-11" and a["spoken_dates"][0]["audio_t"] == 1.0
    assert "[8.2s] The tsunami" in prompts[0]


def test_transcript_failure_falls_back_to_spark_open(monkeypatch, tmp_path):
    clip = SimpleNamespace(path=str(tmp_path / "c.mp4"), has_audio=True, duration=30)
    mp3 = tmp_path / "a.mp3"; mp3.write_bytes(b"y" * 2000)
    monkeypatch.setattr(investigator, "meta_enabled", lambda: True)
    monkeypatch.setattr(investigator, "extract_wav", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(investigator, "extract_audio", lambda *a, **k: mp3)
    monkeypatch.setattr(investigator, "chat", lambda m, c, max_tokens=0, **k: ('{"has_speech": true, "spoken_place_names": [{"name": "Kesennuma", "audio_t": 3}]}', 0.01))
    a = investigator.analyze_audio(clip, tmp_path)
    assert a["status"] == "ok" and a["provider"] == "openrouter" and a["transcriber"] is None and a["model"] == investigator.AUDIO_MODEL


def test_baseten_first_then_openrouter(env, monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "bt-SECRET-555")
    urls = []
    def post(url, **k):
        urls.append(url)
        if "baseten" in url:
            assert k["json"]["model"] == "meta-models/muse-spark-open-1026"
            return Resp(200, ok_body("from baseten"))
        return Resp(200, ok_body("from openrouter"))
    monkeypatch.setattr(llm.requests, "post", post)
    assert llm.chat("meta/muse-spark-open", [{"type": "text", "text": "x"}], baseten=True)[0] == "from baseten" and llm.last_provider() == "baseten"
    llm.chat("meta/muse-spark-open", [{"type": "text", "text": "x"}])          # default: stays on OpenRouter
    assert llm.last_provider() == "openrouter"
    monkeypatch.setattr(llm.requests, "post", lambda url, **k: Resp(500, {"e": "bt-SECRET-555"}) if "baseten" in url else Resp(200, ok_body("or")))
    assert llm.chat("meta/muse-spark-open", [{"type": "text", "text": "x"}], baseten=True)[0] == "or" and llm.last_provider() == "openrouter"
