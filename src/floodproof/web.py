"""Local web app. Everything shown is rendered from a validated report.json (the single source of truth).

  uv run floodproof serve [--port 8000]      then open http://127.0.0.1:8000
Endpoints: GET / | GET /api/demos | GET /api/report/{kind}/{name} (kind = demo|run) | GET /files/{kind}/{name}/{path}
           POST /api/investigate (multipart: file | url, claim, date, place, deep) -> {job} | GET /api/job/{id}
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import UploadFile  # module level: needed to resolve the postponed "UploadFile | None" annotation

from . import CACHE, ROOT
from .report import validate_report

DEMO_DIR = ROOT / "demo"
RUNS_DIR = ROOT / "runs"
UPLOADS = CACHE / "uploads"
STATIC = Path(__file__).parent / "static"
DEMO_TITLES = {
    "success_recycled": "Recycled: Alaska glacier shared as Nepal GLOF",
    "reshare_memory": "Re-share caught by memory",
    "genuine_correct_claim": "Genuine footage, correct date",
    "genuine_wrong_date": "Genuine footage, wrong date claimed",
    "failed_unverified": "Unverified: no earlier copy found",
}
DEMO_BLURBS = {
    "success_recycled": "A proven earlier copy, posted 5 days before the claimed event.",
    "reshare_memory": "A mirrored, re-edited re-share, recognised from a previously verified clip.",
    "genuine_correct_claim": "The earliest copy is dated at the claimed event: nothing contradicts the claim.",
    "genuine_wrong_date": "Real footage, but the claimed date is not supported.",
    "failed_unverified": "No earlier copy found. A normal outcome, not a finding of fakery.",
}
DEMO_ORDER = list(DEMO_TITLES)
JOBS: dict[str, dict] = {}
# Public-deployment guard: demos are free; live runs cost API money and CPU.
# FLOODPROOF_PASSCODE (in .env) lifts the daily cap and unlocks Deep investigate.
DAILY_PUBLIC_RUNS = int(os.environ.get("FLOODPROOF_DAILY_RUNS", "15"))
MAX_UPLOAD = 80 * 1024 * 1024
_RUNS_TODAY: dict[str, int] = {}
_LOCK = threading.Lock()


def _passcode_ok(code: str) -> bool:
    want = os.environ.get("FLOODPROOF_PASSCODE", "")
    return bool(want) and code.strip() == want


def _safe_name(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in "_-")


def _dir(kind: str, name: str) -> Path:
    base = {"demo": DEMO_DIR, "run": RUNS_DIR}.get(kind)
    if base is None or _safe_name(name) != name or not name:
        raise FileNotFoundError("bad report id")
    return base / name


def load_report(kind: str, name: str) -> dict:
    """Read and VALIDATE a report. The UI never sees a report that fails validate_report."""
    p = _dir(kind, name) / "report.json"
    r = json.loads(p.read_text())
    validate_report(r)
    return r


def list_demos() -> list[dict]:
    out = []
    for d in sorted(DEMO_DIR.glob("*/report.json"), key=lambda p: (DEMO_ORDER.index(p.parent.name) if p.parent.name in DEMO_ORDER else 99, p.parent.name)):
        try:
            r = load_report("demo", d.parent.name)
        except Exception as e:                      # an invalid demo is reported, not rendered
            out.append(dict(name=d.parent.name, title=d.parent.name, label="INVALID", error=str(e)[:120])); continue
        ep = (r["visual_matching"].get("earliest_proven_copy") or {})
        thumb = (ep.get("proof") or {}).get("viral_frame") or next(iter((r.get("clues") or {}).get("frame_images") or []), None)
        if thumb and thumb.startswith("frames/") is False and "/" in thumb and not thumb.startswith("evidence/"):
            thumb = "frames/" + thumb.split("/")[-1]
        out.append(dict(name=d.parent.name, title=DEMO_TITLES.get(d.parent.name, d.parent.name), label=r["final_verdict"]["label"],
                        claim=r["claim"].get("text"), blurb=DEMO_BLURBS.get(d.parent.name), thumb=thumb))
    return out


# Per-check cost is stated in the frozen bench write-ups (system B only): heldout $0.153 over 9 cases, global $0.156 over 11 cases.
BENCH_COST_USD = {"heldout": 0.153, "global": 0.156}


def bench_headline() -> dict:
    """Headline numbers computed from bench/heldout_rescored.json (pipeline B rows only) and bench/global_rescored.json."""
    bench = ROOT / "bench"
    ho = [x for x in json.loads((bench / "heldout_rescored.json").read_text()) if x.get("system") == "B"]
    gl_all = json.loads((bench / "global_rescored.json").read_text())
    gl = [x for x in gl_all if x.get("kind") == "debunked"]
    proven = sum(bool(x["found_earlier_copy"]) for x in ho + gl)
    false_acc = sum(bool(x.get("false_attribution_corrected") or x.get("false_attribution") or x.get("wrong_verdict")) for x in ho + gl)
    genuine_fp = sum(bool(x.get("genuine_false_positive")) for x in gl_all if x.get("kind") == "genuine")
    checks = len(ho) + len(gl_all)
    cents = round(100 * sum(BENCH_COST_USD.values()) / checks, 1)
    return dict(clips=len(ho) + len(gl), earlier_copy_proven=proven, false_accusations=false_acc, genuine_false_positives=genuine_fp,
                genuine_clips=len(gl_all) - len(gl), cents_per_check=cents, checks_costed=checks,
                sources=["bench/heldout_rescored.json (system B)", "bench/global_rescored.json", "bench/heldout_RESULTS.md and bench/global_RESULTS.md (cost)"])


def _run_job(job_id, source, claim, date, place, deep):
    job = JOBS[job_id]
    log = lambda *a: job["log"].append(" ".join(str(x) for x in a)[:300])
    out = RUNS_DIR / job_id
    try:
        if deep:
            from .agent import run_agent
            rep = run_agent(source, claim=claim, date=date, place=place, out_dir=out, log=log)
        else:
            from .pipeline import investigate
            rep = investigate(source, claim, date, place, out_dir=out, log=log)
        validate_report(rep)
        try:                                        # best-effort shareable debunk card; never blocks the report
            from .debunk import attach
            attach(out / "report.json")
        except Exception as e:
            log("debunk skipped:", f"{type(e).__name__}: {e}")
        job.update(status="done", report=job_id)
    except Exception as e:
        job.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")


def create_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    app = FastAPI(title="FloodProof")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/headline")
    def headline():
        return bench_headline()

    @app.get("/api/demos")
    def demos():
        return list_demos()

    @app.get("/api/report/{kind}/{name}")
    def report(kind: str, name: str):
        try:
            return load_report(kind, name)
        except FileNotFoundError:
            raise HTTPException(404, "report not found")
        except AssertionError as e:
            raise HTTPException(422, f"report failed validation: {e}")

    @app.get("/files/{kind}/{name}/{path:path}")
    def files(kind: str, name: str, path: str):
        try:
            base = _dir(kind, name).resolve()
        except FileNotFoundError:
            raise HTTPException(404)
        p = (base / path).resolve()
        if base not in p.parents or not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise HTTPException(404)
        return FileResponse(p)

    @app.post("/api/investigate")
    async def investigate_(file: UploadFile | None = File(None), url: str = Form(""), claim: str = Form(""), date: str = Form(""),
                           place: str = Form(""), deep: str = Form(""), passcode: str = Form("")):
        trusted = _passcode_ok(passcode)
        want_deep = deep in ("1", "true", "on")
        if want_deep and not trusted:
            raise HTTPException(403, "Deep investigate needs the judge passcode.")
        if any(j["status"] == "running" for j in JOBS.values()):
            raise HTTPException(429, "Another investigation is running. Try again in a few minutes.")
        today = time.strftime("%Y-%m-%d")
        if not trusted and _RUNS_TODAY.get(today, 0) >= DAILY_PUBLIC_RUNS:
            raise HTTPException(429, "Daily public run limit reached. Try the demos, or use the judge passcode.")
        if file is not None and file.filename:
            UPLOADS.mkdir(parents=True, exist_ok=True)
            src = UPLOADS / (uuid.uuid4().hex[:8] + "_" + _safe_name(Path(file.filename).stem) + Path(file.filename).suffix[:6])
            data = await file.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "Video too large (max 80 MB).")
            src.write_bytes(data); source = str(src)
        elif url.strip().startswith(("http://", "https://")):
            source = url.strip()
        else:
            raise HTTPException(400, "give a video file or an http(s) URL")
        with _LOCK:
            if not trusted:
                _RUNS_TODAY[today] = _RUNS_TODAY.get(today, 0) + 1
        job_id = uuid.uuid4().hex[:10]
        JOBS[job_id] = dict(status="running", log=[], started=time.time(), deep=want_deep)
        threading.Thread(target=_run_job, args=(job_id, source, claim or None, date or None, place or None, JOBS[job_id]["deep"]),
                         daemon=True).start()
        return dict(job=job_id)

    @app.get("/api/job/{job_id}")
    def job(job_id: str):
        j = JOBS.get(job_id)
        if not j:
            raise HTTPException(404)
        return JSONResponse(dict(status=j["status"], log=j["log"][-30:], report=j.get("report"), error=j.get("error"),
                                 seconds=round(time.time() - j["started"])))

    return app


def serve(host="127.0.0.1", port=8000):
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
