"""Web app: renders only validated reports, serves proof images safely, static page carries the required honesty text."""
import json
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from floodproof import web
from floodproof.report import build_report


def good_report():
    m = dict(url="https://youtu.be/o", upload_date="2024-09-28", date_source="yt-dlp live metadata", retrieved_at="t", adapter="memory",
             stored_date="2024-09-28", stored_date_source="memory entry (inherited)", independent_date="2024-09-28",
             independent_date_source="yt-dlp live metadata", date_independent=True, evidence_png="evidence/e.png",
             match=dict(consistent_frames=9, max_inliers=120, temporal_offset=0.0, mirrored=False, best_pairs=[]))
    return build_report(input_meta=dict(source="x"), claim=dict(text="Nepal flood", event_date="2026-08-26"), memory=dict(status="MISS"),
                        visual=dict(status="FOUND", proven_matches=[m]), clues=dict(status="not_run"), physical=dict(status="NOT_RUN"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    d = tmp_path / "demo"
    (d / "ok" / "evidence").mkdir(parents=True)
    (d / "ok" / "report.json").write_text(json.dumps(good_report()))
    (d / "ok" / "evidence" / "e.png").write_bytes(b"\x89PNG")
    (d / "ok" / "secret.txt").write_text("nope")
    bad = good_report(); bad["final_verdict"]["label"] = "FAKE"
    (d / "bad").mkdir(); (d / "bad" / "report.json").write_text(json.dumps(bad))
    monkeypatch.setattr(web, "DEMO_DIR", d)
    return TestClient(web.create_app())


def test_demo_list_validates(client):
    got = {x["name"]: x for x in client.get("/api/demos").json()}
    assert got["ok"]["label"] == "RECYCLED_FOOTAGE" and got["bad"]["label"] == "INVALID"


def test_report_served_only_if_valid(client):
    assert client.get("/api/report/demo/ok").json()["final_verdict"]["label"] == "RECYCLED_FOOTAGE"
    assert client.get("/api/report/demo/bad").status_code == 422
    assert client.get("/api/report/demo/missing").status_code == 404
    assert client.get("/api/report/demo/..%2f..").status_code == 404


def test_files_are_images_only_and_confined(client):
    assert client.get("/files/demo/ok/evidence/e.png").status_code == 200
    assert client.get("/files/demo/ok/secret.txt").status_code == 404
    assert client.get("/files/demo/ok/../ok/report.json").status_code == 404


def test_index_has_required_honesty_ui(client):
    html = client.get("/").text
    for s in ("What this does NOT mean", "Deep investigate (experimental)", "Extra investigation does not guarantee a better verdict",
              "Hypothesis (unverified)", "Stored date", "Independent date", "Limitations (", "Show full trace"):
        assert s in html
    import re
    assert not re.search(r"<script[^>]+src=", html)


def test_investigate_requires_input(client):
    assert client.post("/api/investigate", data={"claim": "x"}).status_code == 400


def test_file_upload_starts_job(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from floodproof import web
    monkeypatch.setattr(web, "UPLOADS", tmp_path)
    monkeypatch.setattr(web, "_run_job", lambda *a, **k: None)
    web.JOBS.clear()
    r = TestClient(web.create_app()).post("/api/investigate", files={"file": ("clip.mp4", b"\x00" * 100, "video/mp4")})
    assert r.status_code == 200 and "job" in r.json()
    web.JOBS.clear()


def test_index_script_parses(tmp_path):
    """Guard against merge damage in the inline UI script (a missing brace froze the page once)."""
    import re, shutil, subprocess
    from floodproof.web import STATIC
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    js = re.findall(r"<script>(.*?)</script>", (STATIC / "index.html").read_text(), re.S)
    for i, code in enumerate(js):
        f = tmp_path / f"s{i}.js"; f.write_text(code)
        assert subprocess.run([node, "--check", str(f)], capture_output=True).returncode == 0, f"script {i} has a syntax error"


def test_job_api_exposes_live_events_and_run_media(tmp_path, monkeypatch):
    """Structured '§' progress lines go to events (not the log); the job API lists keyframes and proof pairs written so far."""
    from fastapi.testclient import TestClient
    from floodproof import web
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    d = tmp_path / "abc123"; (d / "frames").mkdir(parents=True); (d / "evidence").mkdir()
    (d / "frames" / "frame_00_t0.0.jpg").write_bytes(b"x"); (d / "frames" / "lens.jpg").write_bytes(b"x")
    (d / "evidence" / "proof_yandex_1_viral.jpg").write_bytes(b"x"); (d / "evidence" / "proof_yandex_1_copy.jpg").write_bytes(b"x")

    def fake_investigate(source, claim, date, place, out_dir=None, force=False, log=print):
        log("[1/6] fetching clip + keyframes")
        log('§{"type": "search", "source": "yandex", "candidates": 40, "seconds": 60.1}')
        raise RuntimeError("stop here")
    monkeypatch.setattr("floodproof.pipeline.investigate", fake_investigate)
    web.JOBS["abc123"] = dict(status="running", log=[], events=[], started=0, deep=False)
    web._run_job("abc123", "https://youtu.be/x", None, None, None, False)
    j = TestClient(web.create_app()).get("/api/job/abc123").json()
    assert j["events"] == [{"type": "search", "source": "yandex", "candidates": 40, "seconds": 60.1}]
    assert all(not x.startswith("§") for x in j["log"])
    assert j["media"] == {"frames": ["frames/frame_00_t0.0.jpg"],
                          "proofs": [{"viral": "evidence/proof_yandex_1_viral.jpg", "copy": "evidence/proof_yandex_1_copy.jpg", "source": "yandex"}]}


def test_pipeline_event_helper_never_raises():
    from floodproof.pipeline import _event
    out = []
    _event(out.append, "gate", kind="music_video", decision="stop")
    assert out == ['§{"type": "gate", "kind": "music_video", "decision": "stop"}']
    _event(lambda m: (_ for _ in ()).throw(RuntimeError("log broke")), "x")     # swallowed
