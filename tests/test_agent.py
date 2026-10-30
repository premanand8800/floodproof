"""Agent guardrails with a MOCKED LLM and mocked tools: the agent can choose tools but cannot put anything into the
report that a validated tool output does not back."""
import json
import cv2
import numpy as np
import pytest

from floodproof import agent, sources
from floodproof.memory import Memory


@pytest.fixture
def clip_path(tmp_path):
    p = str(tmp_path / "clip.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 120))
    rng = np.random.default_rng(0)
    for i in range(20):
        w.write((rng.random((120, 160, 3)) * 255).astype(np.uint8))
    w.release()
    return p


@pytest.fixture
def mem(tmp_path):
    return Memory(tmp_path / "mem")


def script(*actions):
    """Mock LLM: replays the given actions (dicts), then keeps asking for the last one."""
    it = iter(actions); last = [None]

    def llm(model, text, system):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        a = last[0]
        return (a if isinstance(a, str) else json.dumps(a)), 0.001
    return llm


def mock_tools(monkeypatch, proven_urls=(), date="2020-01-01"):
    monkeypatch.setattr(sources, "tavily_search", lambda q, s=None, a=None, b=None, n=8: (
        [dict(url="https://youtu.be/fake1", title="x", snippet="y", published_date="1999-01-01")], dict(error=None)))

    def vc(c, q, query_date=None, self_url=None, evidence_dir=None):
        if c.url in proven_urls:
            c.status, c.proven, c.upload_date, c.date_source, c.retrieved_at = "proven", True, date, "yt-dlp metadata", "now"
            c.match = dict(decision=True, consistent_frames=33, max_inliers=210, temporal_offset=0.0, mirrored=False,
                           best_pairs=[dict(a_frame=0, b_frame=0, inliers=210, flip=False)], score=1.0)
        else:
            c.status = "not_proven"
        return c
    monkeypatch.setattr(sources, "verify_candidate", vc)
    monkeypatch.setattr(sources, "yandex_candidates", lambda *a, **k: ([], dict(frames=[], captcha=False, searched=0, errors=[])))


def run(clip_path, mem, tmp_path, llm, **kw):
    return agent.run_agent(clip_path, "Nepal flood", "2026-08-26", "Kathmandu", out_dir=tmp_path / "out", memory=mem, llm=llm, **kw)


def test_unverified_agent_claims_cannot_enter_report(clip_path, mem, tmp_path, monkeypatch):
    mock_tools(monkeypatch)
    llm = script({"thought": "search", "tool": "text_search", "args": {"query": "nepal flood"}},
                 {"thought": "done", "tool": "finish", "args": {
                     "summary": "The original is https://youtu.be/fake1 uploaded 2019-05-05, proven match with 99 frames.",
                     "original_url": "https://youtu.be/fake1"}})
    r = run(clip_path, mem, tmp_path, llm)
    assert r["visual_matching"]["status"] == "NOT_FOUND" and r["visual_matching"]["proven_matches"] == []
    assert r["final_verdict"]["label"] == "UNVERIFIED"
    assert "fake1" not in r["final_verdict"]["summary"] and "2019-05-05" not in r["final_verdict"]["summary"]
    kinds = {p["kind"] for p in r["agent_trace"]["summary_audit"]}
    assert {"original_url_not_matcher_proven", "date_not_in_tool_output", "claims_proof_without_any_proven_match",
            "match_number_not_in_tool_output"} <= kinds


def test_only_matcher_proven_candidates_become_matches(clip_path, mem, tmp_path, monkeypatch):
    mock_tools(monkeypatch, proven_urls=("https://youtu.be/real",))
    llm = script({"tool": "fetch_and_verify", "args": {"url": "https://youtu.be/nope"}, "thought": "try"},
                 {"tool": "fetch_and_verify", "args": {"url": "https://youtu.be/real"}, "thought": "try"},
                 {"tool": "finish", "args": {"summary": "Proven copy https://youtu.be/real dated 2020-01-01 with 33 frames.",
                                             "original_url": "https://youtu.be/real"}})
    r = run(clip_path, mem, tmp_path, llm)
    urls = [m["url"] for m in r["visual_matching"]["proven_matches"]]
    assert urls == ["https://youtu.be/real"]
    assert r["final_verdict"]["label"] == "RECYCLED_FOOTAGE"          # from the validated date, not from agent text
    assert r["agent_trace"]["summary_audit"] == []


def test_trace_recorded_every_step(clip_path, mem, tmp_path, monkeypatch):
    mock_tools(monkeypatch)
    llm = script({"thought": "a", "tool": "memory_lookup", "args": {}}, {"thought": "b", "tool": "text_search", "args": {"query": "q"}},
                 {"thought": "c", "tool": "finish", "args": {"summary": "Nothing was proven."}})
    r = run(clip_path, mem, tmp_path, llm)
    steps = r["agent_trace"]["steps"]
    assert [s["tool"] for s in steps] == ["memory_lookup", "text_search", "finish"]
    for s in steps:
        assert {"observation", "tool", "reason", "result", "decision"} <= set(s)
    assert steps[-1]["decision"] == "stop" and steps[0]["decision"] == "continue"
    assert "NOT TRUSTED" in r["agent_trace"]["note"]


def test_budget_respected(clip_path, mem, tmp_path, monkeypatch):
    mock_tools(monkeypatch)
    n = [0]

    def llm(model, text, system):
        n[0] += 1
        return json.dumps({"tool": "text_search", "args": {"query": f"q{n[0]}"}, "thought": "again"}), 0.0
    r = run(clip_path, mem, tmp_path, llm, max_calls=4)
    assert r["agent_trace"]["tool_calls"] == 4 and r["agent_trace"]["stop_reason"] == "budget_exhausted"


def test_duplicate_calls_blocked(clip_path, mem, tmp_path, monkeypatch):
    mock_tools(monkeypatch)
    llm = script(*[{"tool": "text_search", "args": {"query": "same"}, "thought": "x"}] * 3, {"tool": "finish", "args": {"summary": "none"}})
    r = run(clip_path, mem, tmp_path, llm)
    assert [s.get("duplicate") for s in r["agent_trace"]["steps"][:3]] == [False, True, True]


def test_unknown_tools_and_garbage_do_not_crash_or_execute(clip_path, mem, tmp_path, monkeypatch):
    mock_tools(monkeypatch)
    llm = script("not json", {"tool": "write_report", "args": {"verdict": "RECYCLED_FOOTAGE"}}, "???", "again", {"tool": "finish", "args": {"summary": "x"}})
    r = run(clip_path, mem, tmp_path, llm)
    assert r["agent_trace"]["stop_reason"] == "invalid_actions"
    assert r["final_verdict"]["label"] == "UNVERIFIED"


def test_physical_comes_only_from_tool(clip_path, mem, tmp_path, monkeypatch):
    mock_tools(monkeypatch)
    monkeypatch.setattr(agent, "run_physical", lambda *a, **k: dict(status="CONTRADICTS", consistent_with_claim=False, reason="no rain"))
    llm = script({"tool": "physical_check", "args": {"place": "Kathmandu", "date": "2026-08-26", "event_type": "flood", "country": "NP"}},
                 {"tool": "finish", "args": {"summary": "The physical check supports the claim."}})
    r = run(clip_path, mem, tmp_path, llm)
    assert r["physical_world_check"]["status"] == "CONTRADICTS"
    assert r["final_verdict"]["label"] != "CONSISTENT_WITH_CLAIM"


def test_tool_args_sanitised(clip_path, mem, tmp_path, monkeypatch):
    mock_tools(monkeypatch)
    seen = {}
    monkeypatch.setattr(sources, "yandex_candidates", lambda feats, n_frames=3, frames=None, crop=None, **k: (seen.update(frames=frames, crop=crop) or [], dict(searched=1, captcha=False, errors=[])))
    llm = script({"tool": "reverse_search", "args": {"frames": [999, -1, 2, 3, 4, 5], "crop": [0, 0, 5, 5]}}, {"tool": "finish", "args": {"summary": "x"}})
    run(clip_path, mem, tmp_path, llm)
    assert seen["frames"] == [2, 3, 4] and seen["crop"] is None
    from types import SimpleNamespace as NS
    st = agent.AgentState(NS(url=None), [0] * 4, "c", None, None, None, tmp_path, tmp_path)
    assert st.fetch_and_verify("javascript:alert(1)").get("error") and st.text_search("") .get("error")


def test_same_video_url_matching():
    assert agent.same_video_url("https://www.youtube.com/watch?v=abcDEF12345&utm_source=x", "https://youtu.be/abcDEF12345")
    assert not agent.same_video_url("https://youtu.be/abcDEF12345", "https://youtu.be/zzzzzzzzzzz")
