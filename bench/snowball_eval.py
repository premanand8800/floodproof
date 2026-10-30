"""Snowball evaluation (paired: pipeline proven set vs the same set extended by the snowball). See bench/snowball_RESULTS.md.
usage: PYTHONPATH=src uv run python bench/snowball_eval.py run [ID,ID...]     -> bench/snowball_runs/<ID>.json (resumable)
       PYTHONPATH=src uv run python bench/snowball_eval.py score               -> bench/snowball_results.json (no API calls)
Env: SNOW_MUSE_CAP (default 0.72 USD), SNOW_LENS_CAP (38), SNOW_TAVILY_CAP (58), SNOW_YT_UNITS (7000).
Lens needs a public image URL: frames are written under the served runs dir of the live site and deleted afterwards.
"""
import json, os, sys, time, shutil, threading
from pathlib import Path

MAIN_RUNS = Path("/home/samir-kumal/hackethon/floodproof/runs")
os.environ["FLOODPROOF_SNOWBALL_RUNS_DIR"] = str(MAIN_RUNS)
from floodproof import ROOT, load_env, sources, snowball, llm
from floodproof.matcher import ClipFeatures
import concurrent.futures as cf

load_env()
B = ROOT / "bench"
RUNS = B / "snowball_runs"; RUNS.mkdir(exist_ok=True)
LEDGER = B / "snowball_ledger.json"
IDS = ["H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09", "G02", "G03", "G05", "G06", "G07", "G08", "G09", "G10"]
CAP = dict(muse=float(os.environ.get("SNOW_MUSE_CAP", 0.72)), lens=int(os.environ.get("SNOW_LENS_CAP", 38)),
           tavily=int(os.environ.get("SNOW_TAVILY_CAP", 58)), yt=int(os.environ.get("SNOW_YT_UNITS", 7000)))
_lock = threading.Lock()


def led():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(muse=0.0, lens=0, tavily=0, yt=0)


def bump(**kw):
    with _lock:
        d = led()
        for k, v in kw.items():
            d[k] += v
        LEDGER.write_text(json.dumps(d))
        return d


def chat_capped(model, content, max_tokens=4000, **k):
    if led()["muse"] >= CAP["muse"]:
        raise RuntimeError("evaluation Muse budget reached")
    t, c = llm.chat(model, content, max_tokens=max_tokens, **k)
    bump(muse=c)
    return t, c


def lens_capped(url):
    if led()["lens"] >= CAP["lens"]:
        return [], dict(calls=0, skipped="evaluation Lens cap reached")
    cs, info = sources.lens_candidates(url)
    bump(lens=int(info.get("calls") or 0))
    return cs, info


def tavily_capped(queries, before=None):
    n = min(len(queries), snowball.TAVILY_QUERIES_PER_ROUND, max(0, CAP["tavily"] - led()["tavily"]))
    if n <= 0:
        return [], dict(calls=0, skipped="evaluation Tavily cap reached")
    bump(tavily=n)
    return snowball.tavily_snowball(queries[:n], before)


def yt_capped(qs, d):
    n = min(len(qs), max(0, (CAP["yt"] - led()["yt"]) // 100))
    if n <= 0:
        return [], dict(queries=0, skipped="evaluation YouTube budget reached")
    cs, info = sources.youtube_api_candidates(qs[:n], d, max_queries=n, max_results=10, relevance_only=True)
    bump(yt=100 * int(info.get("queries") or 0))
    return cs, info


def cases():
    out = {}
    for f in ("heldout.jsonl", "global.jsonl"):
        for l in (B / f).read_text().splitlines():
            if l.strip():
                c = json.loads(l); out[c["id"]] = c
    return out


def reconstitute(case_id, feats, ev_dir):
    """Re-verify the baseline's proven copies against the input (cached downloads). Returns proven Candidates."""
    rep = json.loads((B / "ablation_v2_runs" / "all_on" / f"{case_id}.json").read_text())
    pm = (rep.get("visual_matching") or {}).get("proven_matches") or []
    seen, cs = set(), []
    for m in pm:
        if m["url"] in seen:
            continue
        seen.add(m["url"]); cs.append(sources.Candidate(m["url"], m.get("title") or "", m.get("adapter") or "baseline", "baseline", "pending"))
    cs = cs[:8]
    with cf.ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda c: sources.verify_candidate(c, feats, None, None, ev_dir), cs))
    return [c for c in cs if c.proven], len(cs)


def run_case(cid, case):
    out_dir = MAIN_RUNS / f"snoweval_{cid}"
    shutil.rmtree(out_dir, ignore_errors=True); ev = out_dir / "evidence"; ev.mkdir(parents=True)
    t = time.time()
    feats = ClipFeatures.from_path(str(ROOT / case["viral_clip"]))
    base, n_base = reconstitute(cid, feats, ev)
    rec_s = round(time.time() - t, 1)
    res = dict(id=cid, baseline_reported=n_base, baseline_reproven=len(base),
               baseline=[dict(url=c.url, date=c.upload_date, adapter=c.adapter, independent=c.date_independent) for c in base])
    if base:
        before = led()
        new, block, reads = snowball.run(feats=feats, proven=base, seen_urls=[], query_date=None, self_url=None, out_dir=out_dir, ev_dir=ev, log=lambda *a: print(cid, *a, flush=True),
                                         chat_fn=chat_capped, fns=dict(youtube=yt_capped, tavily=tavily_capped, yandex=snowball._yandex_one, lens=lens_capped))
        res.update(snowball=block, new=[dict(url=c.url, date=c.upload_date, adapter=c.adapter, independent=c.date_independent, uploader=c.uploader, round=c.snowball_round,
                                             seed=c.snowball_seed, consistent_frames=(c.match or {}).get("consistent_frames"), max_inliers=(c.match or {}).get("max_inliers"))
                                        for c in new], ledger_after=led())
    shutil.rmtree(out_dir, ignore_errors=True)
    res["reconstitute_seconds"] = rec_s
    return res


def run(ids):
    C = cases()
    for cid in ids:
        p = RUNS / f"{cid}.json"
        if p.exists():
            continue
        try:
            r = run_case(cid, C[cid])
        except Exception as e:
            r = dict(id=cid, error=f"{type(e).__name__}: {str(e)[:300]}")
        p.write_text(json.dumps(r, indent=1, default=str))
        s = r.get("snowball") or {}
        print("DONE", cid, "base", r.get("baseline_reproven"), "new", len(r.get("new") or []), s.get("stopped_reason"), s.get("seconds"), led(), r.get("error"), flush=True)


def score():
    C = cases(); rows = []
    for cid in IDS:
        p = RUNS / f"{cid}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text()); c = C[cid]
        base = r.get("baseline") or []
        if not base:
            rows.append(dict(id=cid, in_scope=False, note=r.get("error") or "no baseline copy re-proven")); continue
        new = r.get("new") or []
        nu = lambda u: sources.norm_url(u)
        dated = lambda xs: [x["date"] for x in xs if x.get("date") and x.get("independent")]
        e_off, e_on = (min(dated(base)) if dated(base) else None), (min(dated(base + new)) if dated(base + new) else None)
        orig = nu(c["original_url"]) if c.get("original_url") else None
        found_off = any(nu(x["url"]) == orig for x in base)
        found_on = any(nu(x["url"]) == orig for x in base + new)
        s = r.get("snowball") or {}
        before_orig = [x for x in new if x.get("date") and c.get("original_date") and x["date"] < c["original_date"]]
        rows.append(dict(id=cid, in_scope=True, baseline_copies=len(base), new_copies=len(new), earliest_off=e_off, earliest_on=e_on,
                         earliest_improved=bool(e_off and e_on and e_on < e_off), original_date=c.get("original_date"), original_found_off=found_off,
                         original_found_on=found_on, original_gained=bool(found_on and not found_off), rounds=len(s.get("rounds") or []),
                         stopped=s.get("stopped_reason"), seconds=s.get("seconds"), cost=s.get("cost"),
                         new_before_ground_truth_original=[dict(url=x["url"], date=x["date"]) for x in before_orig], error=r.get("error")))
    ins = [x for x in rows if x["in_scope"]]
    summ = dict(cases_in_scope=len(ins), earliest_improved=sum(x["earliest_improved"] for x in ins), originals_gained=sum(x["original_gained"] for x in ins),
                originals_off=sum(x["original_found_off"] for x in ins), originals_on=sum(x["original_found_on"] for x in ins),
                new_copies=sum(x["new_copies"] for x in ins), cases_with_new=sum(x["new_copies"] > 0 for x in ins),
                max_seconds=max([x["seconds"] or 0 for x in ins] or [0]), mean_seconds=round(sum((x["seconds"] or 0) for x in ins) / max(1, len(ins)), 1),
                openrouter_usd=round(sum(((x["cost"] or {}).get("openrouter_usd") or 0) for x in ins), 4),
                lens_calls=sum(((x["cost"] or {}).get("lens_calls") or 0) for x in ins), tavily_calls=sum(((x["cost"] or {}).get("tavily_calls") or 0) for x in ins),
                youtube_search_calls=sum(((x["cost"] or {}).get("youtube_search_calls") or 0) for x in ins), ledger=led())
    (B / "snowball_results.json").write_text(json.dumps(dict(summary=summ, cases=rows), indent=1, default=str))
    print(json.dumps(summ, indent=1))
    for x in rows:
        print(x)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2].split(",") if len(sys.argv) > 2 else IDS)
    else:
        score()
