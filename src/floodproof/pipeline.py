"""investigate(): the end-to-end entry point used by the CLI and the web app."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from . import ROOT, load_env, media
from .matcher import ClipFeatures
from .memory import Memory
from .report import build_report, run_physical, write_report, validate_report
from . import sources, investigator, infer


def _proven_entry(c: sources.Candidate, out_dir: Path):
    ev = c.evidence_png
    if ev:
        try:
            ev = str(Path(ev).resolve().relative_to(out_dir.resolve()))
        except ValueError:
            pass
    return dict(url=c.url, title=c.title, uploader=c.uploader, upload_date=c.upload_date, date_source=c.date_source,
                retrieved_at=c.retrieved_at, adapter=c.adapter, found_via=c.hint, match=c.match, evidence_png=ev,
                stored_date=c.stored_date, stored_date_source=("memory entry (inherited)" if c.stored_date else None), independent_date=c.independent_date, independent_date_source=c.independent_date_source,
                date_independent=bool(c.date_independent), date_note=c.date_note,
                date_alternatives=list(c.date_alternatives or []),
                memory_origin=(c.memory_entry or {}).get("origin"),
                memory_context=(c.memory_entry or {}).get("event_context"))


def investigate(source: str, claim: str | None = None, date: str | None = None, place: str | None = None,
                claimed_source: str | None = None, out_dir: str | Path | None = None, memory: Memory | None = None,
                use_memory=True, use_yandex=True, use_tavily=True, use_muse=True, remember=False,
                max_verify=14, log=print) -> dict:
    """Full investigation. Returns the report dict (also written to out_dir/report.json with evidence/ PNGs)."""
    load_env()
    t0 = time.time()
    inv_id = uuid.uuid4().hex[:10]
    out = Path(out_dir or ROOT / "runs" / inv_id); out.mkdir(parents=True, exist_ok=True)
    ev_dir = out / "evidence"; ev_dir.mkdir(exist_ok=True)

    log("[1/6] fetching clip + keyframes")
    clip = media.fetch(source)
    feats = ClipFeatures.from_path(clip.path)
    self_url = clip.url or (source if media.is_url(source) else None)
    input_meta = dict(source=source, **clip.to_dict(), n_keyframes=len(feats), keyframe_times_s=[round(t, 2) for t in feats.times])
    input_meta["path"] = clip.path if not clip.url else "(downloaded excerpt removed after run)"

    mem = memory or (Memory() if use_memory else None)
    memory_layer = dict(status="NOT_RUN")
    cands: list[sources.Candidate] = []
    if use_memory and mem is not None:
        log("[2/6] memory lookup")
        mc, res = sources.memory_candidates(mem, feats, evidence_dir=ev_dir)
        mc = [c for c in mc if not (self_url and c.url == sources._clean_url(self_url))]   # never "find" the input itself
        cands += mc
        memory_layer = dict(status="HIT" if mc else "MISS", mode=res.mode, entries_searched=len(res.shortlisted),
                            entries_total=res.n_entries, seconds=res.seconds,
                            hits=[dict(url=c.url, stored_date=c.stored_date, stored_date_source=("memory entry (inherited)" if c.stored_date else None), origin=(c.memory_entry or {}).get("origin"),
                                       context=(c.memory_entry or {}).get("event_context")) for c in mc],
                            note="memory is a hash-shortlist + matcher proof, not an exhaustive scan; a fresh crop may need full matching")

    clues = dict(status="not_run")
    if use_muse:
        log("[3/6] Muse Spark clues")
        try:
            clues = investigator.investigate_clip(clip, feats, claim, date, frames_dir=out / "frames",
                                                  post_text=(clip.title if (clip.url and not claim) else None) and
                                                  ((clip.title or "") + (" | " + clip.description[:500] if clip.description else "")))
        except Exception as e:
            clues = dict(status="error", error=str(e)[:200])
    cp = (clues.get("claim_parsed") or {}) if clues.get("status") == "ok" else {}
    claim_d = infer.infer_claim(clip, claim, date, place, clues)
    claim, date, place = claim_d["text"], claim_d["event_date"], claim_d["place"]
    if claim_d.get("inferred_note"):
        log(f"claim inferred from post: {claim_d['text']!r} date={date} ({claim_d['event_date_basis']}) place={place}")
    claim_d["country_iso2"] = cp.get("country_iso2"); claim_d["event_type"] = cp.get("event_type") or "unknown"

    web = {}
    if use_yandex:
        log("[4/6] Yandex reverse image search")
        try:
            yc, yi = sources.yandex_candidates(feats); cands += yc; web["yandex"] = yi
        except Exception as e:
            web["yandex"] = dict(status="error", error=str(e)[:150])
    if use_tavily and clues.get("status") == "ok":
        log("[5/6] Tavily text search")
        tc, ti = sources.tavily_candidates([q["q"] for q in clues["queries"]]); cands += tc; web["tavily"] = ti
    seen, uniq = set(), []
    for c in cands:
        if c.url not in seen:
            seen.add(c.url); uniq.append(c)
    log(f"[6/6] verifying {min(len(uniq), max_verify)} of {len(uniq)} candidates with the matcher")
    sources.verify_candidates(uniq, feats, query_date=clip.upload_date, self_url=self_url, evidence_dir=ev_dir, max_n=max_verify)

    proven = [c for c in uniq if c.proven]
    log(f"re-fetching dates independently for {sum(not c.date_independent for c in proven)} inherited-date copies")
    sources.enrich_dates(proven)
    by_url = {c.url: c for c in proven}
    for h in memory_layer.get("hits", []):
        c = by_url.get(h["url"])
        if c:
            h.update(independent_date=c.independent_date, independent_date_source=c.independent_date_source, date_independent=c.date_independent)
    earliest = sources.earliest_proven_copy(uniq)
    claimed_orig = None
    if claimed_source:
        cs = sources._clean_url(claimed_source)
        claimed_orig = dict(url=cs, matcher_proven=any(c.url == cs for c in proven),
                            note="the original as asserted by the user/claim; only 'matcher_proven' if a candidate with this URL was verified")
    visual = dict(status="FOUND" if proven else "NOT_FOUND", proven_matches=[_proven_entry(c, out) for c in proven],
                  earliest_proven_copy=_proven_entry(earliest, out) if earliest else None, claimed_original=claimed_orig,
                  input_publication=(dict(url=clip.url, upload_date=clip.upload_date, date_source="yt-dlp metadata",
                                          retrieved_at=sources._now(), uploader=clip.uploader)
                                     if clip.url and clip.upload_date else None),
                  provenance_note="earliest_proven_copy is the earliest matcher-proven copy by platform upload date; it is the earliest copy found and is not asserted to be the original source.",
                  candidates_checked=len([c for c in uniq if c.status not in ("pending",)]), candidates_total=len(uniq),
                  candidate_status_counts={s: sum(c.status == s for c in uniq) for s in {c.status for c in uniq}},
                  adapters=web)

    phys_place = claim_d.get("place")
    physical = run_physical(phys_place, None, None, date, 3, claim_d.get("event_type") or "unknown", claim_d.get("country_iso2")) if date else dict(status="NOT_RUN", reason="no claim date supplied")
    rep = build_report(input_meta=input_meta, claim=claim_d, memory=memory_layer, visual=visual, clues=clues,
                       physical=physical, investigation_id=inv_id)
    rep["timing_s"] = round(time.time() - t0, 1)
    validate_report(rep)
    write_report(rep, out)
    if remember and mem is not None:
        mem.add(feats, self_url, clip.upload_date, dict(claim=claim, verdict=rep["final_verdict"]["label"]), inv_id,
                clip.uploader, clip.title, "investigated_input", clip_key=self_url or f"local:{Path(source).name}:{inv_id}")
    media.cleanup(clip)
    return rep
