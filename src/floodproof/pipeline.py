"""investigate(): the end-to-end entry point used by the CLI and the web app."""
from __future__ import annotations

import time
import uuid
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import ROOT, load_env, media
from .matcher import ClipFeatures
from .memory import Memory
from .report import build_report, run_physical, write_report, validate_report
from . import sources, investigator, infer


VERIFY_WORKERS = int(os.environ.get("FLOODPROOF_VERIFY_WORKERS", "3"))   # matcher is CPU-bound and the box also serves the site


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


def _physical(claim_d, clues, date):
    """Physical check; if the claimed place does not geocode, try narration / on-screen / Muse-hypothesis places (labelled)."""
    if not date:
        return dict(status="NOT_RUN", reason="no claim date supplied")
    et, cc, place = claim_d.get("event_type") or "unknown", claim_d.get("country_iso2"), claim_d.get("place")
    res = run_physical(place, None, None, date, 3, et, cc)
    loc = res.get("location") or {}
    if loc.get("lat") is not None or res.get("status") in ("NOT_RUN",) :
        return res
    from . import physical as ph
    tried = [place] if place else []
    for cand, label in infer.fallback_places(claim_d, clues)[:5]:
        try:
            g = ph.geocode(cand, cc)
        except Exception:
            continue
        tried.append(cand)
        if g.get("lat") is None:
            continue
        r2 = run_physical(cand, g["lat"], g["lon"], date, 3, et, cc)
        if isinstance(r2.get("location"), dict):
            r2["location"].update(source=label, name=g.get("name") or cand, query=cand, confidence=g.get("confidence", "low"),
                                  fallback_from=place, note="claimed place did not resolve; this location comes from a fallback")
        return r2
    if tried and isinstance(res.get("location"), dict):
        res["location"]["fallback_tried"] = tried
    return res


def sources_stop_date(claim_d):
    """Claimed date usable as the early-stop reference: never a bare date-posted fallback."""
    return None if claim_d.get("event_date_basis") == "date_posted" else claim_d.get("event_date")


def _best_frame(feats, out: Path, allowed=None):
    """Save the sharpest keyframe as frames/lens.jpg (served at /files/run/<id>/frames/...) and return its path."""
    import cv2
    ids = sources._spread_frames(feats, 1, allowed)
    if not ids:
        return None
    p = out / "frames" / "lens.jpg"
    p.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(p), feats.frames[ids[0]], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return p


def investigate(source: str, claim: str | None = None, date: str | None = None, place: str | None = None,
                claimed_source: str | None = None, out_dir: str | Path | None = None, memory: Memory | None = None,
                use_memory=True, use_yandex=True, use_tavily=True, use_muse=True, remember=False,
                use_gvision=True, use_youtube=True, use_lens=True,
                max_verify=14, log=print) -> dict:
    """Full investigation. Returns the report dict (also written to out_dir/report.json with evidence/ PNGs)."""
    load_env()
    t0 = time.time()
    stage, t_mark = {}, [t0]
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
    stage["fetch_and_keyframes"] = round(time.time() - t0, 1); t_mark[0] = time.time()
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
    claim_d["country_iso2"] = cp.get("country_iso2") or claim_d.get("country_iso2"); claim_d["event_type"] = cp.get("event_type") or "unknown"

    web = {}
    claim_queries = [q["q"] for q in clues["queries"]] if clues.get("status") == "ok" else ([claim] if claim else [])
    stage["memory_muse_claim"] = round(time.time() - t_mark[0], 1); t_mark[0] = time.time()

    # Real camera clips are the ones worth tracing to earlier copies (AI-generated shots have no original): when Muse labelled any
    # frame camera_footage, reverse-search keyframes are picked only among those.
    cam = set(investigator.camera_frame_ids(clues, len(feats)))
    cam_note = f"{len(cam)} of {len(feats)} keyframes are near frames labelled camera_footage" if cam else None
    pick = lambda n: sources._spread_frames(feats, n, cam or None)

    def _timed(name, fn):
        t = time.time()
        try:
            return name, fn(), None, round(time.time() - t, 1)
        except Exception as e:
            return name, None, e, round(time.time() - t, 1)

    def _lens():
        best = _best_frame(feats, out, cam or None)
        url = sources.public_frame_url(best, out, ROOT / "runs") if best else None
        return sources.lens_candidates(url)

    jobs = []
    if use_gvision:
        jobs.append(("gvision", lambda: sources.gvision_candidates(feats, frames=pick(4) if cam else None)))
    if use_yandex:
        jobs.append(("yandex", lambda: sources.yandex_candidates(feats, frames=pick(3) if cam else None)))
    if use_lens and not any(c.proven for c in cands):     # a copy already proven by memory makes the Lens quota pointless
        jobs.append(("lens", _lens))
    elif use_lens:
        web["lens"] = dict(skipped="not needed (a copy is already proven)")
    if use_youtube:
        jobs.append(("youtube_api", lambda: sources.youtube_api_candidates(claim_queries, date)))
    if use_tavily and clues.get("status") == "ok":
        jobs.append(("tavily", lambda: sources.tavily_candidates(claim_queries)))
    log(f"[4/6] discovery in parallel: {', '.join(n for n, _ in jobs)}")
    disc_t = {}
    results = {}
    if jobs:
        with ThreadPoolExecutor(len(jobs)) as ex:
            for name, res, err, secs in ex.map(lambda j: _timed(*j), jobs):
                results[name], disc_t[name] = (res, err), secs
    for name in ("gvision", "yandex", "lens", "youtube_api", "tavily"):      # reverse-image sources first, then text search
        if name not in results:
            continue
        res, err = results[name]
        if err is not None:
            web[name] = dict(status="error", error=(str(err)[:150] if name == "yandex" else type(err).__name__))
        else:
            cs, info = res; cands += cs; web[name] = info
    _drop_self()
    stage["discovery"] = round(time.time() - t_mark[0], 1); stage["discovery_by_source"] = disc_t; t_mark[0] = time.time()

    todo = sources.merge_and_prioritize(cands)
    n = min(max_verify, sum(1 for c in todo if not c.proven and c.status == "pending"))
    vinfo = {}
    if n > 0:
        log(f"verifying up to {n} candidates with the matcher ({VERIFY_WORKERS} workers)")
        sources.verify_candidates(todo, feats, query_date=clip.upload_date, self_url=self_url, evidence_dir=ev_dir, max_n=n,
                                  workers=VERIFY_WORKERS, stop_before=sources_stop_date(claim_d), info=vinfo)
    uniq = todo
    stage["verify"] = round(time.time() - t_mark[0], 1); t_mark[0] = time.time()
    log(f"checked {len([c for c in uniq if c.status != 'pending'])} of {len(uniq)} candidates"
        + (" (stopped early: independently dated earlier copy proven)" if vinfo.get("early_stop") else ""))

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
                  adapters=web, source_stats=sources.source_stats(uniq),
                  early_stop=(dict(stopped=True, skipped_lower_priority=vinfo.get("skipped_lower_priority"),
                                   reason="a matcher-proven copy with an independently fetched upload date before the claimed date was found, and every remaining candidate was in a lower priority tier")
                              if vinfo.get("early_stop") else dict(stopped=False)),
                  keyframe_selection=dict(preferred="camera_footage" if cam else None, note=cam_note))

    phys_place = claim_d.get("place")
    physical = _physical(claim_d, clues, date)
    rep = build_report(input_meta=input_meta, claim=claim_d, memory=memory_layer, visual=visual, clues=clues,
                       physical=physical, investigation_id=inv_id)
    stage["dates_physical_report"] = round(time.time() - t_mark[0], 1)
    rep["timing_s"] = round(time.time() - t0, 1)
    rep["stage_timings_s"] = dict(stage, verify_workers=VERIFY_WORKERS, verify=vinfo.get("seconds", stage.get("verify")))
    validate_report(rep)
    write_report(rep, out)
    if remember and mem is not None:
        mem.add(feats, self_url, clip.upload_date, dict(claim=claim, verdict=rep["final_verdict"]["label"]), inv_id,
                clip.uploader, clip.title, "investigated_input", clip_key=self_url or f"local:{Path(source).name}:{inv_id}")
    media.cleanup(clip)
    return rep
