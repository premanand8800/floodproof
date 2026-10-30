"""Compare Muse-on (original saved reports) with Muse-off (bench/ablation_runs). No API calls.
usage: python3 bench/ablation_analyze.py [MAIN_CACHE]  -> bench/ablation_muse.json
Muse-on saved reports: <MAIN_CACHE>/{heldout,global}_runs/B/<ID>/report.json (the exact runs behind eval_heldout.json / eval_global.json).
"""
import json, sys, datetime, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodproof.agent import same_video_url

B = Path(__file__).parent
MAIN = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/samir-kumal/hackethon/floodproof/.cache")
CASES = {}
for f in ("heldout.jsonl", "global.jsonl"):
    for l in (B / f).read_text().splitlines():
        if l.strip():
            c = json.loads(l); CASES[c["id"]] = c
IDS = [f"H0{i}" for i in range(1, 10)] + ["G02", "G03", "G05", "G06", "G07", "G08", "G09", "G10", "N01", "N02", "N03"]
GENUINE = {"N01", "N02", "N03"}


def load_on(i):
    p = MAIN / ("heldout_runs" if i.startswith("H") else "global_runs") / "B" / i / "report.json"
    return json.loads(p.read_text())


def score(i, rep):
    c = CASES[i]; v = rep["visual_matching"]; pr = v["proven_matches"]
    dated = [m for m in pr if m.get("upload_date")]
    cd = c.get("claimed_date")
    earlier = bool(cd and any(m["upload_date"][:10] < cd for m in dated))
    label = rep["final_verdict"]["label"]
    cl = rep["clues"]; ok = cl.get("status") == "ok"
    mn = (cl.get("media_nature") or {}).get("value") if ok else None
    au = (cl.get("audio") or {}) if ok else {}
    ad = v.get("adapters") or {}
    by = collections.Counter(m["adapter"] for m in pr)
    return dict(
        label=label, n_proven=len(pr), proven_urls=sorted({m["url"] for m in pr}), earlier_copy=earlier,
        found_original=bool(c.get("original_url") and any(same_video_url(m["url"], c["original_url"]) for m in pr)),
        false_attribution=bool(label == "RECYCLED_FOOTAGE" and cd and not earlier),
        wrong_verdict=bool(i not in GENUINE and label == "CONSISTENT_WITH_CLAIM"),
        genuine_false_positive=bool(i in GENUINE and label == "RECYCLED_FOOTAGE"),
        abstained=label == "UNVERIFIED",
        candidates_total=v.get("candidates_total"), candidates_checked=v.get("candidates_checked"),
        proven_by_adapter=dict(by), source_stats=v.get("source_stats"),
        tavily_queries=(ad.get("tavily") or {}).get("queries"), muse_queries=len(cl.get("queries") or []) if ok else 0,
        media_nature=mn, on_screen_dates=len(cl.get("on_screen_dates") or []) if ok else 0,
        narration_places=[p["name"] for p in (au.get("spoken_place_names") or [])] if au.get("status") == "ok" else [],
        narration_status=au.get("status"), claim_inferred=bool((rep.get("claim") or {}).get("inferred_note")),
        observations_against_claim=len(cl.get("observations_against_claim") or []) if ok else 0,
        physical_verdict=(rep.get("physical_world_check") or {}).get("verdict"),
        sec=rep.get("timing_s") or rep.get("_wall_s"), muse_cost=(cl.get("cost_usd") or 0) + (au.get("cost_usd") or 0) if ok else 0)


def main():
    rows = []
    for i in IDS:
        on = score(i, load_on(i))
        p = B / "ablation_runs" / f"{i}.json"
        off_rep = json.loads(p.read_text()) if p.exists() else None
        off = score(i, off_rep) if off_rep and "error" not in off_rep else dict(error=(off_rep or {}).get("error", "not run"))
        row = dict(id=i, kind="genuine" if i in GENUINE else "debunked", on=on, off=off)
        if "error" not in off:
            row["same_proven_set"] = set(on["proven_urls"]) == set(off["proven_urls"])
            row["on_only_proven"] = sorted(set(on["proven_urls"]) - set(off["proven_urls"]))
            row["off_only_proven"] = sorted(set(off["proven_urls"]) - set(on["proven_urls"]))
        rows.append(row)
    ok = [r for r in rows if "error" not in r["off"]]; deb = [r for r in ok if r["kind"] == "debunked"]; gen = [r for r in ok if r["kind"] == "genuine"]
    f = lambda rs, side, k: sum(bool(r[side][k]) for r in rs)
    summ = {}
    for side in ("on", "off"):
        summ[side] = dict(
            n_debunked=len(deb), n_genuine=len(gen),
            earlier_copy_found=f(deb, side, "earlier_copy"), original_found=f(deb, side, "found_original"),
            false_attribution=f(ok, side, "false_attribution"), wrong_verdict=f(deb, side, "wrong_verdict"),
            abstention=f(deb, side, "abstained"), genuine_false_positive=f(gen, side, "genuine_false_positive"),
            genuine_abstention=f(gen, side, "abstained"),
            mean_candidates_total=round(sum(r[side]["candidates_total"] or 0 for r in ok) / max(1, len(ok)), 1),
            mean_candidates_checked=round(sum(r[side]["candidates_checked"] or 0 for r in ok) / max(1, len(ok)), 1),
            total_proven_urls=sum(r[side]["n_proven"] for r in ok),
            cases_with_media_nature=sum(bool(r[side]["media_nature"]) for r in ok),
            cases_with_on_screen_dates=sum(r[side]["on_screen_dates"] > 0 for r in ok),
            cases_with_narration_places=sum(bool(r[side]["narration_places"]) for r in ok),
            cases_with_observations_against=sum(r[side]["observations_against_claim"] > 0 for r in ok),
            muse_cost_usd=round(sum(r[side]["muse_cost"] for r in ok), 4),
            mean_sec=round(sum(r[side]["sec"] or 0 for r in ok) / max(1, len(ok)), 1))
    summ["identical_proven_sets"] = f"{sum(r['same_proven_set'] for r in ok)}/{len(ok)}"
    summ["identical_labels"] = f"{sum(r['on']['label'] == r['off']['label'] for r in ok)}/{len(ok)}"
    proven_by = collections.Counter(a for r in ok for a, n in r["on"]["proven_by_adapter"].items() for _ in range(n))
    summ["on_proven_by_adapter"] = dict(proven_by)
    out = dict(rows=rows, summary=summ,
               method=dict(muse_off="pipeline.investigate(use_muse=False, use_gvision=False, use_youtube=False, use_lens=False), Yandex ON, memory 37 entries read-only, max_verify=14",
                           muse_on="saved reports of the original B runs (Yandex + Tavily + Muse, no new sources, same memory)"))
    (B / "ablation_muse.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(summ, indent=1))
    for r in rows:
        o, w = r["on"], r["off"]
        print(r["id"], o["label"][:6], "|", w.get("label", w.get("error"))[:6] if w else "", "on", o["n_proven"], "off", w.get("n_proven"), "tot", o["candidates_total"], w.get("candidates_total"),
              "same" if r.get("same_proven_set") else "DIFF", "mn", o["media_nature"], w.get("media_nature"))


if __name__ == "__main__":
    main()
