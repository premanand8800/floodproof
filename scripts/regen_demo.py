"""Recompute the deterministic layers (verdict text, headline, signals, source map) of saved demo reports from their own stored evidence
and regenerate the grounded debunk card. No investigation is re-run and the label rules are the current ones; the label is printed
before/after so a change is never silent.   usage: python scripts/regen_demo.py demo/<name> [...]  [--no-debunk]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodproof import load_env  # noqa: E402
from floodproof.report import build_report, validate_report, write_report  # noqa: E402


def regen(d: Path, debunk=True):
    r = json.loads((d / "report.json").read_text())
    old_label = r["final_verdict"]["label"]
    n = build_report(input_meta=r["input"], claim=r["claim"], memory=r["memory"], visual=r["visual_matching"], clues=r["clues"],
                     physical=r["physical_world_check"], investigation_id=r["investigation_id"])
    n["created_at"] = r["created_at"]
    for k, v in r.items():                       # keep everything the rebuild does not own (gate, timings, fast_path, ...)
        if k not in n and k != "debunk":
            n[k] = v
    validate_report(n)
    write_report(n, d)
    print(f"{d.name}: {old_label} -> {n['final_verdict']['label']}")
    if debunk:
        from floodproof.debunk import attach
        x = attach(d / "report.json")
        print(f"  debunk: model={x['model']} grounded={x['grounded']} cost={x.get('cost_usd')}\n  first sentence: {x['field_refs']['en'][0]['sentence']}")


if __name__ == "__main__":
    load_env()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in args:
        regen(Path(a), debunk="--no-debunk" not in sys.argv)
