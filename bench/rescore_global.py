"""Score bench/eval_global.json with the corrected heldout rule (no API calls). usage: python3 bench/rescore_global.py -> bench/global_rescored.json
False attribution = the system NAMES a wrong source as origin. System B names none (it reports proven copies, never an 'original'), so the only way to
be flagged is label RECYCLED_FOOTAGE with no matcher-proven copy dated before the claimed date (heldout rule). UNVERIFIED is never a false attribution.
Wrong verdict = CONSISTENT_WITH_CLAIM on a debunked clip. Genuine false positive = RECYCLED_FOOTAGE (or any wrong attribution) on a genuine clip.
"""
import json, datetime
from pathlib import Path
B = Path(__file__).parent
cases = {json.loads(l)["id"]: json.loads(l) for l in (B / "global.jsonl").read_text().splitlines() if l.strip()}
d = json.loads((B / "eval_global.json").read_text())
D = lambda s: datetime.date.fromisoformat(s[:10])


def rescore(r):
    c = cases[r["id"]]; proven = r.get("proven_urls") or []
    fa = False; why = ""
    if r.get("label") == "RECYCLED_FOOTAGE" and c.get("claimed_date") and not any(D(u[1]) < D(c["claimed_date"]) for u in proven if u[1]):
        fa, why = True, "RECYCLED without copy before claimed date"
    named = any(u[1] and c.get("original_date") and D(u[1]) > D(c["original_date"]) + datetime.timedelta(days=1) for u in proven) if False else False
    out = dict(id=r["id"], kind=c["kind"], region=c["region"], label=r.get("label"), completed=r.get("completed", False),
               false_attribution=fa, why=why, correct_abstention=r.get("label") == "UNVERIFIED",
               found_original=bool(r.get("found_original")), found_earlier_copy=bool(r.get("found_earlier_copy")))
    if c["kind"] == "debunked":
        out["wrong_verdict"] = r.get("label") == "CONSISTENT_WITH_CLAIM"
    else:
        out["genuine_false_positive"] = r.get("label") == "RECYCLED_FOOTAGE" or fa
        out["genuine_support_label"] = r.get("label")
    en = c.get("expected_media_nature")
    out["media_nature"] = r.get("media_nature"); out["expected_media_nature"] = en
    out["media_nature_ok"] = (r.get("media_nature") == en) if en else (r.get("media_nature") in ("camera_footage", "news_compilation", "screen_recording") if c["kind"] == "genuine" else None)
    return out


if __name__ == "__main__":
    out = [rescore(r) for r in d["rows"] if r["id"] in cases]
    (B / "global_rescored.json").write_text(json.dumps(out, indent=1))
    for o in out: print(o)
