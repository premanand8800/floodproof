"""Re-score saved bench/eval_heldout.json rows with a corrected scorer (no API calls, no re-runs).
Original eval_heldout.py numbers are left untouched. usage: python3 bench/rescore_heldout.py  -> bench/heldout_rescored.json

Why: eval_heldout.false_attribution flagged (a) agents whose named 'original' was not the exact ground-truth URL even when it
was a proven same-event copy dated on/before the original (H03 C/C2, H06 C), and (b) B/agents whose earliest PROVEN copy was
merely later than the true original (H01/H02/H05/H09 B) although nothing was attributed.
Corrected false attribution = the system NAMES a source as the original/origin (agent_original_url) that is dated more than
1 day AFTER the ground-truth original, or says RECYCLED_FOOTAGE with no proven copy dated before the claimed date.
Wrong verdict = label CONSISTENT_WITH_CLAIM on a debunked clip (reported separately).
"""
import json, datetime
from pathlib import Path
B = Path(__file__).parent
cases = {json.loads(l)["id"]: json.loads(l) for l in (B / "heldout.jsonl").read_text().splitlines() if l.strip()}
d = json.loads((B / "eval_heldout.json").read_text())
D = lambda s: datetime.date.fromisoformat(s[:10])


def rescore(r):
    c = cases[r["id"]]; od = D(c["original_date"]); proven = r.get("proven_urls") or []
    named = r.get("agent_original_url")
    named_date = next((u[1] for u in proven if u[0] == named), None) if named else None
    fa = False; why = ""
    if named:
        if named_date is None: fa, why = True, "named origin not among proven matches"
        elif (D(named_date) - od).days > 1: fa, why = True, f"named origin dated {named_date} after original {od}"
    if r.get("label") == "RECYCLED_FOOTAGE" and c.get("claimed_date") and not any(D(u[1]) < D(c["claimed_date"]) for u in proven if u[1]):
        fa, why = True, "RECYCLED without copy before claimed date"
    wrong_verdict = r.get("label") == "CONSISTENT_WITH_CLAIM"
    return dict(id=r["id"], system=r["system"], orig_flag=bool(r.get("false_attribution")), false_attribution_corrected=fa, why=why,
                wrong_verdict=wrong_verdict, correct_abstention=r.get("label") == "UNVERIFIED",
                found_original=bool(r.get("found_original")), found_earlier_copy=bool(r.get("found_earlier_copy")))


if __name__ == "__main__":
    out = [rescore(r) for r in d["rows"] if r["id"] in cases]
    (B / "heldout_rescored.json").write_text(json.dumps(out, indent=1))
    for o in out:
        if o["orig_flag"] or o["false_attribution_corrected"] or o["wrong_verdict"]: print(o)
