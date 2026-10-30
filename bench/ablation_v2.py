"""Ablation v2: Muse on vs off with the CURRENT pipeline (fair round-robin verify, Tavily on plain claim text when Muse is off).
Modes (all: pipeline B, same claim/date/place, memory = read-only copy, max_verify=14, gvision off):
  muse_on   use_muse=True,  Yandex+Tavily on, youtube_api/lens OFF
  muse_off  use_muse=False, Yandex+Tavily on, youtube_api/lens OFF
  all_on    use_muse=True,  Yandex+Tavily+YouTube+Lens on (Muse clues replayed from the muse_on run: no second Muse bill; Lens capped)
usage: uv run python bench/ablation_v2.py run MODE [ID,ID...]  -> bench/ablation_v2_runs/<MODE>/<ID>.json
Env: ABL_MEMORY (memory dir copy), ABL_LENS_CAP (default 20), ABL_BUDGET (Muse $ cap, default 0.42)
"""
import json, os, sys, time, shutil, threading
from pathlib import Path
from floodproof import ROOT, load_env
from floodproof.memory import Memory
from floodproof import pipeline, sources, investigator

load_env()
B = ROOT / "bench"
CASES = {}
for f in ("heldout.jsonl", "global.jsonl"):
    for l in (B / f).read_text().splitlines():
        if l.strip():
            c = json.loads(l); CASES[c["id"]] = c
IDS = [f"H0{i}" for i in range(1, 10)] + ["G02", "G03", "G05", "G06", "G07", "G08", "G09", "G10", "N01", "N02", "N03"]
MODES = dict(muse_on=dict(use_muse=True, use_youtube=False, use_lens=False),
             muse_off=dict(use_muse=False, use_youtube=False, use_lens=False),
             all_on=dict(use_muse=True, use_youtube=True, use_lens=True))
CLUES = ROOT / ".cache" / "ablation_v2_clues"; CLUES.mkdir(parents=True, exist_ok=True)
LENS_CAP = int(os.environ.get("ABL_LENS_CAP", 20)); lens_calls = [0]; _lock = threading.Lock()
_orig_lens = sources.lens_candidates


def _capped_lens(url, *a, **k):
    with _lock:
        if lens_calls[0] >= LENS_CAP:
            return [], dict(source="lens", calls=0, errors=[], skipped="ablation Lens cap reached")
        if url:
            lens_calls[0] += 1
    return _orig_lens(url, *a, **k)


def run(mode, ids):
    sources.lens_candidates = _capped_lens
    out_d = B / "ablation_v2_runs" / mode; out_d.mkdir(parents=True, exist_ok=True)
    mem = Memory(os.environ.get("ABL_MEMORY", str(ROOT / ".cache" / "mem_copy")))
    cur = [None]; orig_inv = investigator.investigate_clip

    live_budget = float(os.environ.get("ABL_BUDGET", 0.41))     # real OpenRouter $ for live Muse calls in this ablation
    spent = [sum((json.loads(f.read_text()).get("cost_usd") or 0) for f in CLUES.glob("*.json")
                 if not json.loads(f.read_text()).get("_replayed_v1"))]

    def inv(*a, **k):
        cf = CLUES / f"{cur[0]}.json"
        if cf.exists():                                           # paid once, reused by every later mode
            return json.loads(cf.read_text())
        if mode == "muse_on" and spent[0] >= live_budget:
            # budget exhausted: replay the clues saved by the ORIGINAL (v1) Muse-on run of this case ($0). Labelled; no per-frame labels.
            v1 = ROOT / ".cache" / ("heldout_runs" if cur[0].startswith("H") else "global_runs") / "B" / cur[0] / "report.json"
            if not v1.exists():
                v1 = Path("/home/samir-kumal/hackethon/floodproof") / v1.relative_to(ROOT)
            r = dict(json.loads(v1.read_text())["clues"], _replayed_v1=True)
        else:
            r = orig_inv(*a, **k); spent[0] += r.get("cost_usd") or 0
        cf.write_text(json.dumps(r, default=str))
        return r
    investigator.investigate_clip = inv
    for i in ids:
        p = out_d / f"{i}.json"
        if p.exists():
            continue
        c = CASES[i]; cur[0] = i; d = ROOT / ".cache" / "ablation_v2" / mode / i; shutil.rmtree(d, ignore_errors=True)
        t = time.time()
        try:
            rep = pipeline.investigate(str(ROOT / c["viral_clip"]), claim=c["claimed_event"], date=c.get("claimed_date"),
                                       place=c.get("claimed_place"), out_dir=d, memory=mem, remember=False, log=lambda *a: None,
                                       max_verify=14, use_gvision=False, **MODES[mode])
            rep["_wall_s"] = round(time.time() - t, 1)
            rep["_clues_source"] = "v1_replay" if (CLUES / f"{i}.json").exists() and json.loads((CLUES / f"{i}.json").read_text()).get("_replayed_v1") else "live"
        except Exception as e:
            rep = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
        rep["_lens_calls_so_far"] = lens_calls[0]
        p.write_text(json.dumps(rep, indent=1, default=str))
        print(mode, i, rep.get("final_verdict", {}).get("label"), rep.get("error"), f"muse$={spent[0]:.3f} lens={lens_calls[0]}", flush=True)


if __name__ == "__main__":
    run(sys.argv[2], sys.argv[3].split(",") if len(sys.argv) > 3 else IDS)
