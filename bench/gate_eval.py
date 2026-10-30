"""Run the relevance gate alone on bench/gate.jsonl; writes bench/gate_results.json. Usage: python bench/gate_eval.py [ids...]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodproof import gate, load_env

load_env()
rows = [json.loads(l) for l in open(Path(__file__).parent / "gate.jsonl")]
want = set(sys.argv[1:])
out = []
for r in rows:
    if want and r["id"] not in want:
        continue
    g = gate.check(r["url"])
    stopped = g["decision"] == "stop"
    out.append(dict(id=r["id"], expected=r["expected"], note=r["note"], stopped=stopped,
                    **{k: g.get(k) for k in ("decision", "kind", "reason", "refs", "confidence", "relevant", "model", "cost", "seconds", "error")},
                    kw=g.get("metadata_signals", {}).get("disaster_keywords")))
    print(json.dumps(out[-1]), flush=True)
if not want:
    json.dump(out, open(Path(__file__).parent / "gate_results.json", "w"), indent=1)
