"""Held-out comparison: (A) memory+matcher only, (B) fixed pipeline, (C) agent. Same input per case (local viral clip +
claim text/date/place), same memory (benchmark originals only), same sources (Yandex, Tavily, yt-dlp, matcher, physical),
same time cap, per-case cost cap. bench/heldout.jsonl is FROZEN: never tune on it.

usage: uv run python bench/eval_heldout.py unseen | run [A,B,C] | summary
"""
import json, sys, time, re, datetime, shutil, urllib.request, os
from pathlib import Path
from floodproof import ROOT, load_env
from floodproof.matcher import ClipFeatures
from floodproof.memory import Memory
from floodproof import sources, pipeline, agent

load_env()
B = ROOT / "bench"
OUT = B / "eval_heldout.json"
TIME_CAP = agent.TIME_CAP_S      # 420 s per case for every system
COST_CAP = 0.25                  # USD per case (enforced inside the agent; B's only paid call is one Muse clue call)
ALL_CASES = [json.loads(l) for l in (B / "heldout.jsonl").read_text().splitlines() if l.strip()]
_uc = B / "heldout_unseen_check.json"
SEEN = {r["id"] for r in json.loads(_uc.read_text())["rows"] if not r["unseen"]} if _uc.exists() else set()   # visible in the JSON, excluded from scoring
CASES = [c for c in ALL_CASES if c["id"] not in SEEN] if sys.argv[1:2] != ["unseen"] else ALL_CASES


def usage():
    r = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
    return json.load(urllib.request.urlopen(r))["data"]["usage"]


def day_before(d):
    return d and (datetime.date.fromisoformat(d[:10]) - datetime.timedelta(days=1)).isoformat()


def unseen():
    """Held-out cases must not be in the benchmark/memory: by URL, by video id, AND by keyframe matching against memory."""
    mem = Memory(); rows = []
    bench_urls = set()
    for l in (B / "cases.jsonl").read_text().splitlines():
        c = json.loads(l); bench_urls |= {agent._vid_key(u) for u in c["original_urls"] + c["viral_urls"]}
    mem_urls = {agent._vid_key(e["source_url"]) for e in mem.entries() if e.get("source_url")}
    for c in CASES:
        keys = {agent._vid_key(u) for u in (c["original_url"], c.get("viral_source_url")) if u}
        feats = ClipFeatures.from_path(ROOT / c["viral_clip"])
        ex = mem.lookup(feats, exhaustive=True)          # prove against EVERY memory entry, not a hash shortlist
        rows.append(dict(id=c["id"], url_in_bench=bool(keys & bench_urls), url_in_memory=bool(keys & mem_urls),
                         memory_visual_hits=[h.entry["source_url"] for h in ex.hits], entries_matched_against=len(ex.shortlisted),
                         unseen=not (keys & bench_urls or keys & mem_urls or ex.hits)))
    (B / "heldout_unseen_check.json").write_text(json.dumps(dict(memory_entries=mem.stats()["entries"], rows=rows), indent=1))
    print(json.dumps(rows, indent=1)); print("all unseen:", all(r["unseen"] for r in rows))


def common(rep, c, system, t, cost):
    v = rep["visual_matching"]
    proven = v["proven_matches"]
    truth_ok = [m for m in proven if agent.same_video_url(m["url"], c["original_url"])]
    dated = [m for m in proven if m.get("upload_date")]
    earliest = min(dated, key=lambda m: m["upload_date"]) if dated else None
    od = c["original_date"]
    row = dict(id=c["id"], system=system, sec=round(t, 1), cost=round(cost, 4), completed=True, label=rep["final_verdict"]["label"],
               n_proven=len(proven), proven_urls=[(m["url"], m["upload_date"]) for m in proven],
               found_original=bool(truth_ok), earliest_proven=earliest and (earliest["url"], earliest["upload_date"]),
               found_earlier_copy=any(m["upload_date"] < c["claimed_date"] for m in dated) if c.get("claimed_date") else False,
               earliest_is_original=bool(earliest and (agent.same_video_url(earliest["url"], c["original_url"]) or earliest["upload_date"] <= day_before(od) or earliest["upload_date"][:10] <= od[:10] and bool(truth_ok))),
               abstained=not proven, over_time_cap=t > TIME_CAP, over_cost_cap=cost > COST_CAP)
    # false attribution: the report's earliest copy is a LATER upload than the confirmed original and is not the original
    row["false_attribution"] = bool(earliest and not truth_ok and earliest["upload_date"][:10] > od[:10])
    row["later_copy_as_earliest"] = bool(earliest and truth_ok and earliest["upload_date"][:10] > od[:10] and not agent.same_video_url(earliest["url"], c["original_url"]))
    return row


def run_case(sysname, c, mem):
    src, d = str(ROOT / c["viral_clip"]), ROOT / ".cache" / "heldout_runs" / sysname / c["id"]
    shutil.rmtree(d, ignore_errors=True)
    t = time.time()
    kw = dict(claim=c["claimed_event"], date=c.get("claimed_date"), place=c.get("claimed_place"), out_dir=d, memory=mem)
    if sysname == "A":
        feats = ClipFeatures.from_path(src)
        mc, res = sources.memory_candidates(mem, feats)
        proven = [dict(url=x.url, upload_date=x.upload_date) for x in mc]
        fake = dict(final_verdict=dict(label="RECYCLED_FOOTAGE" if proven else "UNVERIFIED"), visual_matching=dict(proven_matches=proven))
        return common(fake, c, "A", time.time() - t, 0.0)
    if sysname == "B":
        rep = pipeline.investigate(src, log=lambda *a: None, **kw)
        cost = (rep["clues"].get("cost_usd") or 0) if rep["clues"].get("status") == "ok" else 0
        row = common(rep, c, "B", time.time() - t, cost)
        row["captcha"] = bool(rep["visual_matching"]["adapters"].get("yandex", {}).get("captcha"))
        return row
    # C = agent as first frozen (stops after 4 unparseable replies). C2 = same agent, tolerates 10 unparseable replies
    # (harness robustness change made AFTER seeing 2 C traces cut off by unparseable model replies; disclosed in the report)
    rep = agent.run_agent(src, cost_cap=COST_CAP, log=lambda *a: None, invalid_limit=10 if sysname == "C2" else 4, **kw)
    at = rep["agent_trace"]; steps = at["steps"]
    row = common(rep, c, sysname, time.time() - t, at["cost_usd"])
    truth_idx = None
    for s_ in steps:                       # first step after which the true original was proven
        if s_.get("tool") == "fetch_and_verify" and '"proven": true' in (s_.get("result") or "") and agent.same_video_url((s_.get("args") or {}).get("url", ""), c["original_url"]):
            truth_idx = s_["step"]; break
        if s_.get("tool") == "memory_lookup" and agent.same_video_url(s_.get("result", ""), c["original_url"]):
            truth_idx = s_["step"]; break
    calls = [s_ for s_ in steps if s_.get("tool") in agent.TOOL_NAMES]
    dups = sum(bool(s_.get("duplicate")) for s_ in calls)
    after = sum(1 for s_ in calls if truth_idx and s_["step"] > truth_idx)
    # failed search = a reverse/text search call made while nothing was proven yet
    failed = 0; proven_any = False
    for s_ in calls:
        if s_["tool"] in ("reverse_search", "text_search") and not proven_any:
            failed += 1
        if s_["tool"] in ("fetch_and_verify", "memory_lookup") and '"proven": true' in (s_.get("result") or "") or (s_["tool"] == "memory_lookup" and '"status": "HIT"' in (s_.get("result") or "")):
            proven_any = True
    orig_claim = at.get("agent_original_url")
    row.update(tool_calls=len(calls), unnecessary_calls=dups + after, duplicates=dups, calls_after_truth=after,
               failed_searches_before_first_proof=failed, recovered_after_failures=bool(failed >= 1 and row["found_original"]),
               stop_reason=at["stop_reason"], completed=at["stop_reason"] in ("agent_finished", "budget_exhausted"),
               unsupported_claims=len(at["summary_audit"]), summary_audit=at["summary_audit"],
               agent_original_url=orig_claim,
               agent_original_wrong=bool(orig_claim and not agent.same_video_url(orig_claim, c["original_url"])),
               agent_summary=at.get("agent_summary"))
    row["false_attribution"] = row["false_attribution"] or row["agent_original_wrong"]
    return row


def run(systems):
    mem = Memory()
    data = json.loads(OUT.read_text()) if OUT.exists() else dict(rows=[])
    done = {(r["id"], r["system"]) for r in data["rows"]}
    u0 = usage(); data.setdefault("usage_start", u0)
    for c in CASES:
        for s in systems:
            if (c["id"], s) in done:
                continue
            if usage() - data["usage_start"] > 2.6:
                print("budget guard hit"); OUT.write_text(json.dumps(data, indent=1, default=str)); return
            try:
                row = run_case(s, c, mem)
            except Exception as e:
                row = dict(id=c["id"], system=s, completed=False, error=f"{type(e).__name__}: {str(e)[:200]}", found_original=False,
                           found_earlier_copy=False, earliest_is_original=False, false_attribution=False, abstained=True, cost=0, sec=0)
            data["rows"].append(row)
            OUT.write_text(json.dumps(data, indent=1, default=str))
            print(c["id"], s, {k: row.get(k) for k in ("found_original", "found_earlier_copy", "false_attribution", "tool_calls", "sec", "cost", "error")}, flush=True)
    data["usage_end"] = usage(); OUT.write_text(json.dumps(data, indent=1, default=str))


def summary():
    d = json.loads(OUT.read_text()); rows = d["rows"]; out = {}
    for s in ("A", "B", "C", "C2"):
        r = [x for x in rows if x["system"] == s]
        if not r:
            continue
        n = len(r); f = lambda k: sum(bool(x.get(k)) for x in r)
        o = dict(n=n, found_original=f"{f('found_original')}/{n}", found_earlier_copy=f"{f('found_earlier_copy')}/{n}",
                 earliest_is_original=f"{f('earliest_is_original')}/{n}", false_attribution=f"{f('false_attribution')}/{n}",
                 abstained_not_found=f"{f('abstained')}/{n}", completed=f"{f('completed')}/{n}",
                 avg_sec=round(sum(x['sec'] for x in r) / n, 1), avg_cost=round(sum(x['cost'] for x in r) / n, 4),
                 over_time_cap=f("over_time_cap"), over_cost_cap=f("over_cost_cap"))
        if s in ("C", "C2"):
            o.update(avg_tool_calls=round(sum(x.get("tool_calls", 0) for x in r) / n, 1), unnecessary_calls=sum(x.get("unnecessary_calls", 0) for x in r),
                     unsupported_claims=sum(x.get("unsupported_claims", 0) for x in r),
                     recovery=f"{sum(bool(x.get('recovered_after_failures')) for x in r)}/{sum(x.get('failed_searches_before_first_proof', 0) >= 1 for x in r)} cases with >=1 failed search")
        out[s] = o
    d["summary"] = out; OUT.write_text(json.dumps(d, indent=1, default=str)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    m = sys.argv[1]
    {"unseen": unseen, "run": lambda: run(sys.argv[2].split(",") if len(sys.argv) > 2 else ["A", "B", "C"]), "summary": summary}[m]()
