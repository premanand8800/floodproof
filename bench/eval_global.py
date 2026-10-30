"""Global (non-South-Asia) benchmark: system B (fixed pipeline floodproof.pipeline.investigate) on bench/global.jsonl
(G01-G10 debunked, N01-N03 genuine). bench/global.jsonl is FROZEN: never tune on it. No --remember; memory is read-only here.
usage: uv run python bench/eval_global.py unseen | run | score
"""
import json, sys, time, shutil, datetime, urllib.request, os
from pathlib import Path
from floodproof import ROOT, load_env
from floodproof.matcher import ClipFeatures
from floodproof.memory import Memory
from floodproof import pipeline, agent

load_env()
B = ROOT / "bench"
OUT = B / "eval_global.json"
SPEND_CAP = 0.40                  # hard cap on OpenRouter usage delta for the whole run
ALL = [json.loads(l) for l in (B / "global.jsonl").read_text().splitlines() if l.strip()]
_uc = B / "global_unseen_check.json"
SEEN = {r["id"] for r in json.loads(_uc.read_text())["rows"] if not r["unseen"]} if _uc.exists() else set()
CASES = [c for c in ALL if c["id"] not in SEEN] if sys.argv[1:2] != ["unseen"] else ALL


def usage():
    r = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
    return json.load(urllib.request.urlopen(r))["data"]["usage"]


def unseen():
    mem = Memory(); rows = []
    bench_urls = set()
    for l in (B / "cases.jsonl").read_text().splitlines():
        c = json.loads(l); bench_urls |= {agent._vid_key(u) for u in c["original_urls"] + c["viral_urls"]}
    for l in (B / "heldout.jsonl").read_text().splitlines():
        c = json.loads(l); bench_urls |= {agent._vid_key(u) for u in (c.get("original_url"), c.get("viral_source_url")) if u}
    mem_urls = {agent._vid_key(e["source_url"]) for e in mem.entries() if e.get("source_url")}
    for c in ALL:
        keys = {agent._vid_key(u) for u in (c.get("original_url"), c.get("viral_source_url")) if u}
        feats = ClipFeatures.from_path(ROOT / c["viral_clip"])
        ex = mem.lookup(feats, exhaustive=True)
        rows.append(dict(id=c["id"], url_in_bench=bool(keys & bench_urls), url_in_memory=bool(keys & mem_urls),
                         memory_visual_hits=[h.entry["source_url"] for h in ex.hits], entries_matched_against=len(ex.shortlisted),
                         unseen=not (keys & bench_urls or keys & mem_urls or ex.hits)))
    (B / "global_unseen_check.json").write_text(json.dumps(dict(memory_entries=mem.stats()["entries"], rows=rows), indent=1))
    print(json.dumps(rows, indent=1)); print("all unseen:", all(r["unseen"] for r in rows))


def run():
    mem = Memory()
    data = json.loads(OUT.read_text()) if OUT.exists() else dict(rows=[])
    done = {r["id"] for r in data["rows"]}
    data.setdefault("usage_start", usage())
    for c in CASES:
        if c["id"] in done:
            continue
        if usage() - data["usage_start"] > SPEND_CAP - 0.04:
            print("budget guard hit"); break
        d = ROOT / ".cache" / "global_runs" / "B" / c["id"]; shutil.rmtree(d, ignore_errors=True)
        t = time.time()
        try:
            rep = pipeline.investigate(str(ROOT / c["viral_clip"]), claim=c["claimed_event"], date=c.get("claimed_date"),
                                       place=c.get("claimed_place"), out_dir=d, memory=mem, remember=False, log=lambda *a: None)
            row = extract(c, rep, time.time() - t)
        except Exception as e:
            row = dict(id=c["id"], kind=c["kind"], completed=False, error=f"{type(e).__name__}: {str(e)[:300]}", sec=round(time.time() - t, 1), cost=0)
        data["rows"].append(row)
        OUT.write_text(json.dumps(data, indent=1, default=str))
        print(c["id"], {k: row.get(k) for k in ("label", "n_proven", "sec", "cost", "error")}, flush=True)
    data["usage_end"] = usage(); OUT.write_text(json.dumps(data, indent=1, default=str))


def extract(c, rep, t):
    v = rep["visual_matching"]; proven = v["proven_matches"]
    cl = rep["clues"]; cost = (cl.get("cost_usd") or 0) if cl.get("status") == "ok" else 0
    ph = rep.get("physical_world_check") or {}; loc = ph.get("location") or {}
    mn = (cl.get("media_nature") or {}) if cl.get("status") == "ok" else {}
    dated = [m for m in proven if m.get("upload_date")]
    earliest = min(dated, key=lambda m: m["upload_date"]) if dated else None
    return dict(id=c["id"], kind=c["kind"], completed=True, sec=round(t, 1), cost=round(cost, 4), label=rep["final_verdict"]["label"],
                n_proven=len(proven), proven_urls=[(m["url"], m.get("upload_date")) for m in proven],
                earliest_proven=earliest and (earliest["url"], earliest["upload_date"]),
                found_original=bool(c.get("original_url") and any(agent.same_video_url(m["url"], c["original_url"]) for m in proven)),
                found_earlier_copy=bool(c.get("claimed_date") and any(m["upload_date"][:10] < c["claimed_date"] for m in dated)),
                media_nature=mn.get("value"), clues_status=cl.get("status"), clues_model=cl.get("model"),
                physical_status=ph.get("status"), physical_verdict=ph.get("verdict"), physical_ran=bool(ph) and ph.get("status") != "NOT_RUN",
                physical_located=loc.get("lat") is not None, physical_loc_name=(loc.get("name") or "")[:90], physical_loc_source=loc.get("source"),
                physical_loc_conf=loc.get("confidence"), physical_reasons=(ph.get("reasons") or [])[:3],
                captcha=bool(v["adapters"].get("yandex", {}).get("captcha")), adapters={k: (a.get("status") if isinstance(a, dict) else a) for k, a in v["adapters"].items()},
                candidates_checked=v.get("candidates_checked"), candidates_total=v.get("candidates_total"),
                memory_status=(rep.get("memory") or {}).get("status"), summary=rep["final_verdict"].get("summary", "")[:400])


if __name__ == "__main__":
    {"unseen": unseen, "run": run}[sys.argv[1]]()
