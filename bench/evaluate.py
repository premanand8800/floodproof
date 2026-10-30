"""Evaluation. Metrics are reported SEPARATELY (no single accuracy number):

  memory   : seeded-original recognition (variants + real viral clips), false-attribution rate (leave-one-out),
             time per lookup.  Memory origin = seed:benchmark. NOT discovery.
  genuine  : genuine-footage false-positive rate and context-mismatch detection rate (from the memory hits).
  discovery: previously-unseen-original discovery (memory OFF; Yandex + Tavily + Muse queries, matcher-verified).
  clues    : Muse location accuracy vs the fact-check's true location (country / place level).

usage:  uv run python bench/evaluate.py memory|genuine|discovery|clues|all   (results -> bench/eval_<name>.json)
"""
import json, sys, time, glob
from pathlib import Path

from floodproof import ROOT, load_env, CACHE
from floodproof.matcher import ClipFeatures, VARIANTS, make_variant
from floodproof.memory import Memory
from floodproof.report import decide_verdict

load_env()
B = ROOT / "bench"
KEYS = json.loads((B / "local_keys.json").read_text())      # key -> url
URL2KEY = {u: k for k, u in KEYS.items()}
VID = ROOT / "data" / "videos"
WRONG_DATE = "2026-08-26"

LOO_ORIGINALS = ["ig_leigh", "ig_sainj", "yt_sun_italy", "yt_ie_nalagarh", "yt_dawn", "yt_tauktae1", "yt_ramban",
                 "yt_japan2015", "ig_pak2022", "yt_sikkim_austria"]
LOO_VARIANTS = ["crop70", "mirror", "combo"]
# same footage uploaded twice (matcher-proven siblings in bench/true_pairs.json): a LOO hit on these is not a false attribution
SIBLINGS = [{"ig_sainj", "yts_sainj"}, {"yt_tauktae1", "yt_tauktae2"}, {"ig_wayanad_el", "yt_wayanad_el"}]
# real viral clips -> keys of their fact-checked originals (bench/cases.jsonl via search_loop.CASES)
REAL = {"x_iambmane": ["ig_leigh", "yt_leigh"], "x_dmuppavarapu": ["yt_leigh", "ig_leigh"],
        "yts_viral_nalagarh": ["yt_ie_nalagarh", "yt_tribune_nalagarh"], "ig_viral_italy": ["yt_sun_italy"]}


def vp(k):
    g = [p for p in VID.glob(f"{k}.*") if p.suffix != ".part"]
    return g[0] if g else None


def entry_by_key(mem):
    return {URL2KEY[e["source_url"]]: e for e in mem.entries() if e["source_url"] in URL2KEY}


def variant_path(key, name):
    d = CACHE / "var"; d.mkdir(exist_ok=True)
    dst = d / f"{key}_{name}.mp4"
    if not dst.exists():
        make_variant(vp(key), dst, VARIANTS[name])
    return dst


def run_memory():
    mem = Memory(); ek = entry_by_key(mem)
    out = dict(variants=[], real=[], loo=[])
    for key in LOO_ORIGINALS:
        if key not in ek or not vp(key):
            continue
        src = ek[key]
        for name in LOO_VARIANTS:
            q = ClipFeatures.from_path(variant_path(key, name))
            t = time.time(); res = mem.lookup(q, origin_prefix="seed:"); dt = time.time() - t
            ids = [h.entry["id"] for h in res.hits]
            short = [e for e, _, _ in mem.shortlist(q, 6)]
            row = dict(key=key, variant=name, hit=src["id"] in ids and ids[0] == src["id"], returned=[h.entry["source_url"] for h in res.hits],
                       truth_in_hash_shortlist=src["id"] in short, sec=round(dt, 1), orig_date=src["upload_date"],
                       orig_url=src["source_url"], stage_frames=res.hits[0].match.consistent_frames if res.hits else 0)
            out["variants"].append(row)
            t = time.time(); r2 = mem.lookup(q, exclude={src["id"]}, origin_prefix="seed:")   # leave-one-out: true original removed
            sib = [h for h in r2.hits if any(key in s and ek_key(mem, h.entry) in s for s in SIBLINGS)]
            out["loo"].append(dict(key=key, variant=name, false_hits=[h.entry["source_url"] for h in r2.hits if h not in sib],
                                   sibling_hits=[h.entry["source_url"] for h in sib], sec=round(time.time() - t, 1)))
            print(row, out["loo"][-1], flush=True)
    for vk, truths in REAL.items():
        if not vp(vk):
            continue
        q = ClipFeatures.from_path(vp(vk)); t = time.time()
        res = mem.lookup(q, origin_prefix="seed:")
        tids = {ek[k]["id"] for k in truths if k in ek}
        out["real"].append(dict(viral=vk, truth=truths, hit=any(h.entry["id"] in tids for h in res.hits),
                                returned=[h.entry["source_url"] for h in res.hits], sec=round(time.time() - t, 1)))
        print(out["real"][-1], flush=True)
    v, l, r = out["variants"], out["loo"], out["real"]
    out["summary"] = dict(
        seeded_original_recognition_variants=f"{sum(x['hit'] for x in v)}/{len(v)}",
        by_variant={n: f"{sum(x['hit'] for x in v if x['variant'] == n)}/{sum(1 for x in v if x['variant'] == n)}" for n in LOO_VARIANTS},
        truth_in_hash_shortlist=f"{sum(x['truth_in_hash_shortlist'] for x in v)}/{len(v)}",
        seeded_original_recognition_real_viral=f"{sum(x['hit'] for x in r)}/{len(r)}",
        false_attribution_loo=f"{sum(bool(x['false_hits']) for x in l)}/{len(l)}",
        sibling_hits_loo=sum(bool(x['sibling_hits']) for x in l),
        mean_sec_per_lookup=round(sum(x['sec'] for x in v) / max(1, len(v)), 1),
        memory_entries=mem.stats()["entries"], origin="seed:benchmark (NOT discovery)")
    (B / "eval_memory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


def ek_key(mem, entry):
    return URL2KEY.get(entry["source_url"])


def _proven(entry, m):
    return dict(url=entry["source_url"], upload_date=entry["upload_date"], date_source="memory entry", retrieved_at="eval",
                adapter="memory", match=m.to_dict())


def run_genuine():
    """From memory hits: label with the CORRECT claim date (original's own date) and with a WRONG later date."""
    d = json.loads((B / "eval_memory.json").read_text())
    ok_rows = [x for x in d["variants"] if x["hit"] and x["orig_date"]]
    correct_recycled = wrong_detected = n = 0
    rows = []
    for x in ok_rows:
        pv = dict(url=x["orig_url"], upload_date=x["orig_date"], date_source="memory entry", retrieved_at="eval", adapter="memory",
                  match=dict(consistent_frames=x["stage_frames"], max_inliers=60, temporal_offset=0, mirrored=False, best_pairs=[]))
        c = decide_verdict(dict(event_date=x["orig_date"]), [pv], dict(status="not_run"), dict(status="NOT_RUN"))["label"]
        wd = WRONG_DATE if x["orig_date"] < WRONG_DATE else None
        w = decide_verdict(dict(event_date=wd), [pv], dict(status="not_run"), dict(status="NOT_RUN"))["label"] if wd else None
        n += 1; correct_recycled += c == "RECYCLED_FOOTAGE"; wrong_detected += (w == "RECYCLED_FOOTAGE")
        rows.append(dict(key=x["key"], variant=x["variant"], label_correct_claim=c, label_wrong_date=w))
    out = dict(rows=rows, summary=dict(
        genuine_footage_false_positive_rate=f"{correct_recycled}/{n} (genuine re-shares with the CORRECT claim date labelled RECYCLED_FOOTAGE)",
        correct_context_mismatch_detection_rate=f"{wrong_detected}/{n} (same footage, WRONG later claim date, labelled RECYCLED_FOOTAGE)",
        caveat="synthetic re-shares of benchmark originals; 'correct claim date' = the original's own upload date. Not an independent set."))
    (B / "eval_genuine.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out["summary"], indent=1))


def run_discovery(keys=None):
    """Memory OFF, web adapters ON. Hit = a matcher-proven copy that is (or is the same URL as) a fact-check-cited original,
    or any proven copy dated before the query's claimed date."""
    from floodproof.pipeline import investigate
    CASES = {"x_iambmane": ("Nepal GLOF Aug 2026", "2026-08-26", ["DcUFQZhvGzS", "E5TOaoetGDo"]),
             "yts_sainj": ("Flooding in Himachal Pradesh 2024", "2024-09-09", ["C_pHwZgvMXg"]),
             "ig_viral_italy": ("Uttarakhand flash flood Aug 2025", "2025-08-05", ["Nsb9eSzAEgQ"]),
             "yt_tauktae2": ("Cyclone Tauktae May 2021", "2021-05-18", ["bSkLZSRyRu0"]),
             "yts_viral_nalagarh": ("Uttarkashi cloudburst Aug 2025", "2025-08-06", ["gG8y--iMl04", "MBLVrGHbJAc"]),
             "yt_wayanad_el": ("Wayanad landslide July 2024", "2024-07-30", ["C2AMQyYpf_f"])}
    rows = []
    for k, (claim, date, truth) in CASES.items():
        if keys and k not in keys or not vp(k):
            continue
        t = time.time()
        rep = investigate(str(vp(k)), claim, date, out_dir=CACHE / "eval_runs" / k, use_memory=False, use_muse=True, log=lambda s: None)
        pm = rep["visual_matching"]["proven_matches"]
        rows.append(dict(key=k, found_truth=any(any(t_ in m["url"] for t_ in truth) for m in pm), proven=[m["url"] for m in pm],
                         label=rep["final_verdict"]["label"], candidates=rep["visual_matching"]["candidates_total"],
                         adapters={a: v for a, v in rep["visual_matching"]["adapters"].items()}, sec=round(time.time() - t),
                         clue_location=rep["clues"].get("likely_location", {}).get("place") if rep["clues"].get("status") == "ok" else None))
        print(rows[-1], flush=True)
        (B / "eval_discovery.json").write_text(json.dumps(dict(rows=rows), indent=1, default=str))
    s = dict(previously_unseen_original_discovery_rate=f"{sum(r['found_truth'] for r in rows)}/{len(rows)}",
             any_proven_copy=f"{sum(bool(r['proven']) for r in rows)}/{len(rows)}",
             prototype_baseline="bench/search_S*.json + rev_S*.json: 0/8 originals found (Yandex rarely returns originals)",
             origin="memory disabled")
    (B / "eval_discovery.json").write_text(json.dumps(dict(rows=rows, summary=s), indent=1, default=str)); print(json.dumps(s, indent=1))


# ------------------------------------------------------------------ Muse clue accuracy
TRUTH = {  # key: (country aliases, place keywords or None)   source: fact-check original_desc in bench/cases.jsonl
    "ig_leigh": (["united states", "usa", "us", "alaska"], ["columbia", "alaska", "valdez"]),
    "yt_leigh": (["iceland"], ["iceland", "jokul", "skaftá", "skafta", "vatnaj"]),
    "ig_dharali_accu": (["india"], ["dharali", "uttarkashi", "uttarakhand", "harsil"]),
    "ig_pak2022": (["pakistan"], None),
    "yt_sun_italy": (["italy"], ["bardonecchia", "piedmont", "piemonte", "turin"]),
    "yt_ie_nalagarh": (["india"], ["nalagarh", "himachal", "solan", "baddi"]),
    "yt_tribune_nalagarh": (["india"], ["nalagarh", "himachal", "solan", "baddi"]),
    "yt_indexp_punjab": (["india"], ["punjab"]),
    "ig_sainj": (["india"], ["sainj", "himachal", "kullu"]),
    "yts_sainj": (["india"], ["sainj", "himachal", "kullu"]),
    "yt_dawn": (["pakistan"], None), "yt_hum": (["pakistan"], None),
    "fb_kolkata": (["india"], ["kolkata", "calcutta", "west bengal"]),
    "yt_michael": (["united states", "usa", "us"], ["florida", "mexico beach", "panama city", "michael"]),
    "yt_tauktae1": (["india"], ["gujarat", "mumbai", "kerala", "goa", "tauktae"]),
    "yt_tauktae2": (["india"], ["gujarat", "mumbai", "kerala", "goa", "tauktae"]),
    "ig_wayanad_el": (["india"], ["wayanad", "kerala"]), "yt_wayanad_el": (["india"], ["wayanad", "kerala"]),
    "yt_manorama_dog": (["india"], ["kerala", "kottayam"]),
    "ig_wayanad_china": (["china"], None),
    "yt_sikkim_austria": (["austria"], ["austria", "tyrol", "salzburg", "carinthia", "lake"]),
    "yt_ramban": (["india"], ["ramban", "jammu", "kashmir"]),
    "yt_sankei": (["japan"], ["atami", "shizuoka"]),
    "yt_japan2015": (["japan"], None),
    "yt_kompas": (["indonesia"], ["nusa tenggara", "ntt", "flores", "adonara"]),
    "yt_croc": (["india"], ["vadodara", "gujarat", "baroda"]),
    "yts_viral_nalagarh": (["india"], ["nalagarh", "himachal"]),
}
DEMO_CLAIM = "Nepal flood, August 2026"   # every benchmark case is a false claim about a different event; Muse is NOT told the truth


def run_clues(keys=None):
    from floodproof.investigator import investigate_clip
    from floodproof.media import Clip
    rows = []
    for k, (cc, pk) in TRUTH.items():
        if keys and k not in keys or not vp(k):
            continue
        f = ClipFeatures.from_path(vp(k))
        try:
            c = investigate_clip(Clip(str(vp(k))), f, DEMO_CLAIM, "2026-08-26", use_audio=False)
        except Exception as e:
            rows.append(dict(key=k, error=str(e)[:100])); continue
        if c.get("status") != "ok":
            rows.append(dict(key=k, error=c.get("error"))); continue
        ll = c["likely_location"]; txt = f"{ll.get('place') or ''} {ll.get('country') or ''}".lower()
        cok = any(a in txt.replace(",", " ").split() or a in txt for a in cc if len(a) > 3) or any(txt.split() and a == w for a in cc for w in txt.replace(",", " ").split())
        pok = bool(pk) and any(x in txt for x in pk)
        rows.append(dict(key=k, truth=cc[0], guess=f"{ll.get('place')} / {ll.get('country')}", conf=ll.get("confidence"),
                         country_ok=cok, place_ok=pok if pk else None, cited_contradiction=len(c["contradictions_with_claim"]) > 0,
                         n_rejected_uncited=len(c["rejected_clues"]), all_cited=all(x.get("refs") for x in c["shows"]), cost=c["cost_usd"]))
        print(rows[-1], flush=True)
        (B / "eval_clues.json").write_text(json.dumps(dict(rows=rows), indent=1))
    ok = [r for r in rows if "guess" in r]; pl = [r for r in ok if r["place_ok"] is not None]
    s = dict(n=len(ok), country_level_accuracy=f"{sum(r['country_ok'] for r in ok)}/{len(ok)}",
             place_level_accuracy=f"{sum(r['place_ok'] for r in pl)}/{len(pl)} (only clips whose fact-check names a place)",
             cited_contradiction_with_false_Nepal_claim=f"{sum(r['cited_contradiction'] for r in ok)}/{len(ok)}",
             errors=len(rows) - len(ok), muse_cost_usd=round(sum(r["cost"] for r in ok), 3),
             note="truth = fact-check original_desc (country/place keyword match, automated, see TRUTH table); audio not used here")
    (B / "eval_clues.json").write_text(json.dumps(dict(rows=rows, summary=s), indent=1)); print(json.dumps(s, indent=1))


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["all"]):
        if w in ("memory", "all"): run_memory()
        if w in ("genuine", "all"): run_genuine()
        if w in ("discovery", "all"): run_discovery()
        if w in ("clues", "all"): run_clues()
