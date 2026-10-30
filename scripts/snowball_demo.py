"""Regenerate demo/success_recycled with the snowball on (a full investigation; memory is a read-only copy).
usage: PYTHONPATH=src python scripts/snowball_demo.py <memory_copy_dir> <served_runs_dir>
The run is written under <served_runs_dir> (a directory the public site serves) so Google Lens can fetch the seed frame, then copied to demo/."""
import json, shutil, sys
from pathlib import Path
from floodproof import ROOT, load_env
from floodproof.memory import Memory
from floodproof import pipeline
from floodproof.report import validate_report

load_env()
mem = Memory(sys.argv[1])
runs = Path(sys.argv[2]); out = runs / "snowdemo_success"
shutil.rmtree(out, ignore_errors=True)
import os
os.environ["FLOODPROOF_SNOWBALL_RUNS_DIR"] = str(runs)
rep = pipeline.investigate(str(ROOT / "data/videos/x_iambmane.mp4"), claim="Nepal GLOF", date="2026-08-26", place="Nepal", out_dir=out, memory=mem,
                           remember=False, use_snowball=True, log=print)
validate_report(rep)
from floodproof.debunk import attach
attach(out / "report.json")                       # the shareable card, as the web app does after a run
dst = ROOT / "demo" / "success_recycled"
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(out, dst, ignore=shutil.ignore_patterns("snowball"))
print(rep["final_verdict"]["label"], json.dumps(rep.get("snowball"), indent=1)[:3000])
print(json.dumps(rep.get("spread"), indent=1, ensure_ascii=False))
