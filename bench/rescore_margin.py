"""Re-score every saved benchmark row under the time-zone margin rule (A2). No API calls, no re-runs.
usage: python3 bench/rescore_margin.py -> bench/margin_rescored.json (and a printed summary)

Rule (src/floodproof/report.py date_precedes): yt-dlp upload dates are UTC calendar dates, Nepal is UTC+5:45, so a date-only copy counts as
EARLIER than the claimed event only if it is at least 2 calendar days before the claimed date. Saved rows carry dates only (no exact
timestamps), so the date-only rule is applied to every proven copy. A saved RECYCLED_FOOTAGE label whose only earlier copies fall inside
the margin would now be UNVERIFIED; a saved found_earlier_copy that relied only on such a copy would now be False.
Also reported: rows whose saved label rested on the input URL's own publication (none of the saved rows record one, so none can be affected
by the A1 change; the check is on proven_urls only).
"""
import datetime
import json
from pathlib import Path

B = Path(__file__).parent
D = lambda s: datetime.date.fromisoformat(s[:10])
MARGIN = 2


def load_cases():
    cases = {}
    for f in ("heldout.jsonl", "global.jsonl"):
        for line in (B / f).read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                cases[c["id"]] = c
    return cases


def judge(row, case):
    cd = case.get("claimed_date")
    pr = [u for u in (row.get("proven_urls") or []) if u and u[1]]
    old = bool(cd) and any(D(u[1]) < D(cd) for u in pr)
    new = bool(cd) and any((D(cd) - D(u[1])).days >= MARGIN for u in pr)
    label = row.get("label")
    new_label = "UNVERIFIED" if (label == "RECYCLED_FOOTAGE" and not new) else label
    return dict(old_earlier=old, new_earlier=new, label=label, new_label=new_label,
                label_changed=new_label != label, found_earlier_changed=(old != new))


def main():
    cases = load_cases()
    sources = [("eval_global.json", None), ("eval_heldout.json", None), ("ablation_v2.json", None)]
    out, n = [], 0
    for fname, _ in sources:
        d = json.loads((B / fname).read_text())
        for r in d["rows"]:
            c = cases.get(r.get("id"))
            if not c:
                continue
            n += 1
            j = judge(r, c)
            out.append(dict(file=fname, id=r["id"], system=r.get("system") or r.get("arm"), claimed_date=c.get("claimed_date"), **j))
    changed = [o for o in out if o["label_changed"] or o["found_earlier_changed"]]
    (B / "margin_rescored.json").write_text(json.dumps(dict(rows_checked=n, changed=changed, rows=out), indent=1))
    print(f"rows checked: {n}; label or earlier-copy changes under the 2-day margin: {len(changed)}")
    for o in changed:
        print(o)


if __name__ == "__main__":
    main()
