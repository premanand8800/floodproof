"""Muse ablation: pipeline B with use_muse=False on the frozen cases (H01-H09 + G02,G03,G05-G10,N01-N03).
Muse-on rows are NOT rerun: they come from bench/eval_heldout.json / eval_global.json (+ the saved reports in the main checkout's
.cache/{heldout,global}_runs/B). To match the original runs (which predate the YouTube API / Lens / Google Vision adapters) the new
sources are switched off here: use_gvision=False, use_youtube=False, use_lens=False. Memory is the same 37 benchmark originals,
read-only (remember=False). Everything else (Yandex, max_verify=14, same claim/date/place) is unchanged.
usage: uv run python bench/ablation_muse.py run ID[,ID...]   -> bench/ablation_runs/<ID>.json
"""
import json, sys, time, shutil
from pathlib import Path
from floodproof import ROOT, load_env
from floodproof.memory import Memory
from floodproof import pipeline

load_env()
B = ROOT / "bench"
OUTD = B / "ablation_runs"; OUTD.mkdir(exist_ok=True)
CASES = {}
for f in ("heldout.jsonl", "global.jsonl"):
    for l in (B / f).read_text().splitlines():
        if l.strip():
            c = json.loads(l); CASES[c["id"]] = c
IDS = [f"H0{i}" for i in range(1, 10)] + ["G02", "G03", "G05", "G06", "G07", "G08", "G09", "G10", "N01", "N02", "N03"]


def run(ids):
    mem = Memory()
    for i in ids:
        if (OUTD / f"{i}.json").exists():
            continue
        c = CASES[i]; d = ROOT / ".cache" / "ablation_runs" / i; shutil.rmtree(d, ignore_errors=True)
        t = time.time()
        try:
            rep = pipeline.investigate(str(ROOT / c["viral_clip"]), claim=c["claimed_event"], date=c.get("claimed_date"),
                                       place=c.get("claimed_place"), out_dir=d, memory=mem, remember=False, log=lambda *a: None,
                                       use_muse=False, use_gvision=False, use_youtube=False, use_lens=False)
            rep["_wall_s"] = round(time.time() - t, 1)
        except Exception as e:
            rep = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
        (OUTD / f"{i}.json").write_text(json.dumps(rep, indent=1, default=str))
        print(i, rep.get("final_verdict", {}).get("label"), rep.get("error"), flush=True)


if __name__ == "__main__":
    run(sys.argv[2].split(",") if len(sys.argv) > 2 else IDS)
