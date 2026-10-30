"""Score ablation v2 (muse_on / muse_off / all_on) from bench/ablation_v2_runs/<mode>/<ID>.json. No API calls.
usage: python3 bench/ablation_v2_analyze.py -> bench/ablation_v2.json (+ prints a markdown table)
"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodproof.agent import same_video_url

B = Path(__file__).parent
CASES = {}
for f in ("heldout.jsonl", "global.jsonl"):
    for l in (B / f).read_text().splitlines():
        if l.strip():
            c = json.loads(l); CASES[c["id"]] = c
IDS = [f"H0{i}" for i in range(1, 10)] + ["G02", "G03", "G05", "G06", "G07", "G08", "G09", "G10", "N01", "N02", "N03"]
GENUINE = {"N01", "N02", "N03"}
MODES = ["muse_on", "muse_off", "all_on"]


def load(mode, i):
    p = B / "ablation_v2_runs" / mode / f"{i}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return None if "error" in r else r


def score(i, rep):
    c = CASES[i]; v = rep["visual_matching"]; pr = v["proven_matches"]
    cd = c.get("claimed_date")
    earlier = bool(cd and any(m.get("upload_date") and m["upload_date"][:10] < cd for m in pr))
    label = rep["final_verdict"]["label"]
    by = collections.Counter()
    only_tavily = []
    for m in pr:
        srcs = [m["adapter"]] + list(m.get("also_found_by") or [])
        for a in srcs:
            by[a] += 1
        if srcs == ["tavily"]:
            only_tavily.append(m["url"])
    ad = v.get("adapters") or {}
    return dict(label=label, earlier_copy=earlier, n_proven=len(pr), proven_urls=sorted({m["url"] for m in pr}),
                found_original=bool(c.get("original_url") and any(same_video_url(m["url"], c["original_url"]) for m in pr)),
                false_attribution=bool(label == "RECYCLED_FOOTAGE" and cd and not earlier),
                wrong_verdict=bool(i not in GENUINE and label == "CONSISTENT_WITH_CLAIM"),
                genuine_false_positive=bool(i in GENUINE and label == "RECYCLED_FOOTAGE"),
                abstained=label == "UNVERIFIED", proven_by_source=dict(by), proven_only_tavily=sorted(only_tavily),
                candidates_total=v.get("candidates_total"), candidates_checked=v.get("candidates_checked"),
                source_stats=v.get("source_stats"), tavily_label=(ad.get("tavily") or {}).get("queries"),
                lens=(ad.get("lens") or {}), sec=rep.get("timing_s") or rep.get("_wall_s"),
                muse_cost=((rep.get("clues") or {}).get("cost_usd") or 0) if rep["clues"].get("status") == "ok" else 0)


def main():
    rows, summ = [], {}
    for i in IDS:
        row = dict(id=i, kind="genuine" if i in GENUINE else "debunked")
        for m in MODES:
            r = load(m, i); row[m] = score(i, r) if r else None
        rows.append(row)
        on, off = row["muse_on"], row["muse_off"]
        if on and off:
            row["muse_only_proven"] = sorted(set(on["proven_only_tavily"]) - set(off["proven_urls"]))
            row["same_proven_set"] = on["proven_urls"] == off["proven_urls"]
    for m in MODES:
        ok = [r for r in rows if r[m]]; deb = [r for r in ok if r["kind"] == "debunked"]; gen = [r for r in ok if r["kind"] == "genuine"]
        f = lambda rs, k: sum(bool(r[m][k]) for r in rs)
        by = collections.Counter()
        for r in ok:
            by.update(r[m]["proven_by_source"])
        summ[m] = dict(n_cases=len(ok), n_debunked=len(deb), n_genuine=len(gen), earlier_copy_found=f(deb, "earlier_copy"),
                       original_found=f(deb, "found_original"), false_attribution=f(ok, "false_attribution"),
                       wrong_verdict=f(deb, "wrong_verdict"), abstention_debunked=f(deb, "abstained"),
                       genuine_false_positive=f(gen, "genuine_false_positive"), proven_total=sum(r[m]["n_proven"] for r in ok),
                       proven_by_source=dict(by), mean_sec=round(sum(r[m]["sec"] or 0 for r in ok) / max(1, len(ok)), 1),
                       muse_cost_usd=round(sum(r[m]["muse_cost"] for r in ok), 4),
                       mean_candidates_total=round(sum(r[m]["candidates_total"] or 0 for r in ok) / max(1, len(ok)), 1))
    both = [r for r in rows if r["muse_on"] and r["muse_off"]]
    summ["identical_labels_on_vs_off"] = f"{sum(r['muse_on']['label'] == r['muse_off']['label'] for r in both)}/{len(both)}"
    summ["identical_proven_sets_on_vs_off"] = f"{sum(r['same_proven_set'] for r in both)}/{len(both)}"
    summ["muse_only_proven_copies"] = {r["id"]: r["muse_only_proven"] for r in both if r["muse_only_proven"]}
    (B / "ablation_v2.json").write_text(json.dumps(dict(rows=rows, summary=summ), indent=1, default=str))
    print(json.dumps(summ, indent=1))
    for r in rows:
        print(r["id"], *[(r[m]["label"][:5], r[m]["n_proven"], r[m]["proven_by_source"]) if r[m] else None for m in MODES], r.get("muse_only_proven"))


if __name__ == "__main__":
    main()
