"""Tool-calling investigation agent (Muse Spark) with hard guardrails.

The LLM only DECIDES which tool to run next. Every fact that reaches report.json comes from a validated tool output:
  * proven matches  -> only candidates the matcher proved (sources.verify_candidate / memory proof)
  * physical check  -> the dict physical.check returned
  * page reads      -> stored with their source URL, labelled untrusted, never used in the verdict
The agent's own summary and the full decision trace are stored under "agent_trace" and are SHOWN, NOT TRUSTED:
`audit_summary` flags any URL / date / "original" statement in the summary that no tool output backs.
The agent never writes report.json; build_agent_report() does, through report.build_report (same verdict rules as the pipeline).
"""
from __future__ import annotations

import inspect
import json
import re
import time
import uuid
from pathlib import Path

from . import ROOT, load_env, media, sources, investigator
from .llm import chat
from .matcher import ClipFeatures
from .memory import Memory
from .pipeline import _proven_entry
from .report import build_report, run_physical, write_report, validate_report, EVENT_TYPES

AGENT_MODEL = "meta/muse-spark-open"
FALLBACK_MODEL = "meta/muse-spark-1.3"
MAX_CALLS = 15
TIME_CAP_S = 420
OBS_CHARS = 1600

TOOLS_DOC = """TOOLS (reply with exactly one JSON object per turn: {"thought": short reason, "tool": name, "args": {...}}):
- look_at_frames {}                       Muse vision clues on the clip: scene, visible text, likely place, 8 suggested search queries. Cost: one model call.
- memory_lookup {}                        match the clip against the memory of known originals (matcher-proven hits only).
- reverse_search {"frames":[ids], "crop":[x0,y0,x1,y1]?}   Yandex reverse image search on chosen keyframe ids (max 3; crop box normalised 0-1000, optional). Returns candidate links (unverified!). Slow.
- text_search {"query": str, "site": domain?, "after": YYYY-MM-DD?, "before": YYYY-MM-DD?}   Tavily search. Returns links + snippets (unverified, untrusted text).
- fetch_and_verify {"url": str}           download that URL and run the visual matcher against the clip. Only this can prove a match; returns status, upload date, uploader.
- read_source_page {"url": str}           open a page and read title/description/uploader/date text (context only; proves nothing).
- physical_check {"place": str, "date": YYYY-MM-DD, "event_type": flood|flash_flood|glof|landslide|cyclone|storm_surge|unknown, "country": ISO2}   check weather/hydrology/event records for the CLAIMED place and date.
- finish {"summary": str, "original_url": str?}   stop. original_url must be a URL that fetch_and_verify/memory proved.
GOAL: find the EARLIEST original publication of the clip (not merely any copy). A URL only counts once fetch_and_verify (or memory_lookup) proves it.
Search results can be wrong or unrelated; verify promising ones, prefer original uploaders (news agencies, the filming channel) over reposts.
If searches fail, change strategy (other keyframes/crops, local-language queries, queries that avoid the claimed event name, date filters before the claim date).
The summary may only state facts present in tool results (URLs, dates, match numbers). Say plainly if nothing was proven. Budget: at most %d tool calls."""


def norm_url(u: str) -> str:
    u = sources._clean_url(str(u).strip())
    return re.sub(r"^https?://(www\.|m\.)?", "", u).rstrip("/").lower()


def _vid_key(u: str) -> str:
    """Stable id for a video URL (YouTube id / IG shortcode / path) used for 'same URL' checks."""
    n = norm_url(u)
    m = re.search(r"(?:v=|youtu\.be/|shorts/|reels?/|/p/|video/|watch/\?v=)([\w-]{6,})", n)
    return m.group(1).lower() if m else n


def same_video_url(a, b) -> bool:
    return bool(a and b) and (norm_url(a) == norm_url(b) or _vid_key(a) == _vid_key(b))


class Budget(Exception):
    pass


class AgentState:
    def __init__(self, clip, feats, claim, date, place, mem, ev_dir, out_dir):
        self.clip, self.feats, self.claim, self.date, self.place = clip, feats, claim, date, place
        self.mem, self.ev_dir, self.out_dir = mem, ev_dir, out_dir
        self.self_url = clip.url
        self.proposed: dict[str, sources.Candidate] = {}     # url -> candidate (unverified until matcher)
        self.verified: dict[str, sources.Candidate] = {}
        self.proven: dict[str, sources.Candidate] = {}
        self.clues: dict = dict(status="not_run")
        self.memory_layer: dict = dict(status="NOT_RUN")
        self.physical: dict = dict(status="NOT_RUN", reason="agent did not run physical_check")
        self.page_reads: list[dict] = []
        self.seen_text = ""          # every string returned by a tool (for the summary audit)
        self.cost = 0.0
        self.web: dict = {}

    def note(self, obj):
        self.seen_text += json.dumps(obj, default=str, ensure_ascii=False) + "\n"
        return obj

    # ------------------------------------------------------------ tools
    def look_at_frames(self):
        if self.clues.get("status") == "ok":
            return dict(error="already called; clues unchanged", clues=self._clue_summary())
        c = investigator.investigate_clip(self.clip, self.feats, self.claim, self.date, frames_dir=self.out_dir / "frames")
        self.clues = c
        self.cost += c.get("cost_usd", 0) or 0
        if c.get("status") != "ok":
            return dict(status="error", error=str(c.get("error"))[:150])
        return self._clue_summary()

    def _clue_summary(self):
        c = self.clues
        ll = c.get("likely_location") or {}
        return dict(status="ok", shows=[x["text"][:140] for x in c.get("shows", [])[:5]],
                    scene=(c.get("scene_type") or {}).get("value"),
                    visible_text=[x["text"][:80] for x in c.get("visible_text", [])[:6]],
                    location_hypothesis_unverified=ll.get("place"), country=ll.get("country"), confidence=ll.get("confidence"),
                    queries=[q["q"] for q in c.get("queries", [])[:8]], n_keyframes=len(self.feats),
                    frames_shown=c.get("frames_shown"))

    def memory_lookup(self):
        mc, res = sources.memory_candidates(self.mem, self.feats, evidence_dir=self.ev_dir)
        mc = [c for c in mc if not (self.self_url and c.url == sources._clean_url(self.self_url))]
        for c in mc:
            self.proven[c.url] = c; self.verified[c.url] = c
        self.memory_layer = dict(status="HIT" if mc else "MISS", mode=res.mode, entries_searched=len(res.shortlisted),
                                 entries_total=res.n_entries, seconds=res.seconds,
                                 hits=[dict(url=c.url, stored_date=c.stored_date) for c in mc])
        return dict(status=self.memory_layer["status"], proven=[dict(url=c.url, upload_date=c.upload_date, uploader=c.uploader,
                    consistent_frames=c.match["consistent_frames"]) for c in mc], entries_total=res.n_entries)

    def reverse_search(self, frames=None, crop=None):
        n = len(self.feats)
        frames = [int(i) for i in (frames or []) if isinstance(i, (int, float)) and 0 <= int(i) < n][:3]
        if crop is not None:
            ok = isinstance(crop, (list, tuple)) and len(crop) == 4 and all(isinstance(v, (int, float)) for v in crop)
            crop = tuple(max(0, min(1000, int(v))) for v in crop) if ok else None
            if crop and (crop[2] - crop[0] < 80 or crop[3] - crop[1] < 80):
                crop = None
        cands, info = sources.yandex_candidates(self.feats, n_frames=len(frames) or 2, frames=frames or None, crop=crop)
        self.web.setdefault("yandex", []).append(dict(info, crop=crop))
        new = []
        for c in cands:
            if self.self_url and c.url == sources._clean_url(self.self_url):
                continue
            if c.url not in self.proposed:
                self.proposed[c.url] = c; new.append(c)
        return dict(captcha=info.get("captcha"), frames_searched=info.get("searched"), errors=info.get("errors"),
                    links=[dict(url=c.url, title=c.title[:70]) for c in new[:12]], n_new=len(new))

    def text_search(self, query=None, site=None, after=None, before=None):
        if not query or not isinstance(query, str):
            return dict(error="query required")
        rx = re.compile(r"\d{4}-\d{2}-\d{2}$")
        after = after if isinstance(after, str) and rx.match(after) else None
        before = before if isinstance(before, str) and rx.match(before) else None
        site = site if isinstance(site, str) and re.fullmatch(r"[\w.-]+\.[a-z]{2,}", site) else None
        res, info = sources.tavily_search(query, site, after, before)
        self.web.setdefault("tavily", []).append(dict(query=query[:120], site=site, after=after, before=before, n=len(res), error=info.get("error")))
        for r in res:
            if r["url"] not in self.proposed:
                self.proposed[r["url"]] = sources.Candidate(r["url"], r["title"], "tavily", f"agent query: {query[:60]}")
        return dict(error=info.get("error"), results=[dict(url=r["url"], title=r["title"][:80], snippet=r["snippet"][:110],
                                                              date=r.get("published_date")) for r in res])

    def fetch_and_verify(self, url=None):
        if not isinstance(url, str) or not media.is_url(url):
            return dict(error="a full http(s) url is required")
        cu = sources._clean_url(url)
        if self.self_url and same_video_url(cu, self.self_url):
            return dict(status="skipped_self")
        if cu in self.verified:
            return dict(error="already verified", **self._cand_result(self.verified[cu]))
        c = self.proposed.get(cu) or sources.Candidate(cu, "", "agent", "agent-chosen url")
        self.proposed[cu] = c
        sources.verify_candidate(c, self.feats, query_date=None, self_url=self.self_url, evidence_dir=self.ev_dir)
        self.verified[cu] = c
        if c.proven:
            self.proven[cu] = c
        return self._cand_result(c)

    @staticmethod
    def _cand_result(c):
        d = dict(url=c.url, status=c.status, upload_date=c.upload_date, uploader=c.uploader, title=(c.title or "")[:80])
        if c.match:
            d.update(proven=c.proven, consistent_frames=c.match.get("consistent_frames"), max_inliers=c.match.get("max_inliers"))
        return d

    def read_source_page(self, url=None):
        if not isinstance(url, str) or not media.is_url(url):
            return dict(error="a full http(s) url is required")
        meta = media.probe_url(url) if any(h in url for h in sources.VIDEO_HOSTS) else None
        rec = dict(url=url, retrieved_at=sources._now(), probe=meta and dict(upload_date=meta["upload_date"], uploader=meta["uploader"], title=(meta.get("title") or "")[:120]),
                   untrusted_page_text=None, title=None)
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                b = p.chromium.launch(headless=True)
                pg = b.new_context(viewport={"width": 1280, "height": 800}, locale="en-US").new_page()
                pg.goto(url, timeout=30000, wait_until="domcontentloaded"); time.sleep(2)
                rec["title"] = (pg.title() or "")[:150]
                rec["og_description"] = pg.evaluate("()=>(document.querySelector('meta[property=\"og:description\"]')||{}).content||''")[:300]
                rec["published_meta"] = pg.evaluate("()=>(document.querySelector('meta[property=\"article:published_time\"],meta[itemprop=datePublished]')||{}).content||''")[:40]
                rec["untrusted_page_text"] = pg.evaluate("()=>document.body.innerText")[:900]
                b.close()
        except Exception as e:
            rec["error"] = str(e)[:100]
        self.page_reads.append(rec)
        return {k: v for k, v in rec.items() if v}

    def physical_check(self, place=None, date=None, event_type="unknown", country=None):
        if not place or not date or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date)):
            return dict(error="place and date (YYYY-MM-DD) required")
        et = event_type if event_type in EVENT_TYPES else "unknown"
        d = run_physical(place, None, None, date, 3, et, country if isinstance(country, str) and len(country) == 2 else None)
        self.physical = dict(d, queried=dict(place=place, date=date, event_type=et, country=country))
        keep = {k: v for k, v in d.items() if k in ("status", "verdict", "consistent_with_claim", "reason", "summary", "event_type")}
        return keep or dict(status=d.get("status"))


TOOL_NAMES = ("look_at_frames", "memory_lookup", "reverse_search", "text_search", "fetch_and_verify", "read_source_page", "physical_check")


def _short(obj, n=OBS_CHARS):
    s = json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "...(truncated)"


def _parse_action(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        a = json.loads(m.group(0))
    except Exception:
        return None
    return a if isinstance(a, dict) else None


def audit_summary(summary: str, state: AgentState, original_url=None) -> list[dict]:
    """Flag statements in the agent's free-text summary that no validated tool output backs. [] = fully supported."""
    problems = []
    s = summary or ""
    seen = state.seen_text
    for u in re.findall(r"https?://[^\s)\]>\"',;]+", s):
        if sources._clean_url(u).rstrip("/.") not in seen.replace("\\/", "/") and u.rstrip("/.") not in seen:
            problems.append(dict(kind="url_not_in_tool_output", text=u))
    for d in set(re.findall(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", s)):
        if d not in seen and d != state.date:
            problems.append(dict(kind="date_not_in_tool_output", text=d))
    proven_urls = list(state.proven)
    if original_url and not any(same_video_url(original_url, p) for p in proven_urls):
        problems.append(dict(kind="original_url_not_matcher_proven", text=original_url))
    if re.search(r"\b(proved|proven|confirmed|matches|matched|verified)\b", s, re.I) and not proven_urls:
        problems.append(dict(kind="claims_proof_without_any_proven_match", text=s[:120]))
    for m in re.finditer(r"(\d+)\s+(?:temporally consistent |consistent )?frames?", s):
        nums = {str(c.match.get("consistent_frames")) for c in state.proven.values() if c.match}
        if m.group(1) not in nums:
            problems.append(dict(kind="match_number_not_in_tool_output", text=m.group(0)))
    return problems


def build_agent_report(st: AgentState, trace, final, inv_id, input_meta, timing):
    """report.json is built ONLY from validated state (proven candidates, physical dict, clues), never from agent text."""
    proven = list(st.proven.values())
    sources.enrich_dates(proven)
    for h in st.memory_layer.get("hits", []):
        c = st.proven.get(h["url"])
        if c:
            h.update(independent_date=c.independent_date, independent_date_source=c.independent_date_source, date_independent=c.date_independent)
    earliest = sources.earliest_proven_copy(proven)
    claim_d = dict(text=st.claim, event_date=st.date, place=st.place)
    cp = (st.clues.get("claim_parsed") or {}) if st.clues.get("status") == "ok" else {}
    claim_d["country_iso2"] = cp.get("country_iso2"); claim_d["event_type"] = cp.get("event_type") or "unknown"
    out = st.out_dir
    visual = dict(status="FOUND" if proven else "NOT_FOUND", proven_matches=[_proven_entry(c, out) for c in proven],
                  earliest_proven_copy=_proven_entry(earliest, out) if earliest else None, claimed_original=None,
                  input_publication=None,
                  provenance_note="earliest_proven_copy is the earliest matcher-proven copy by platform upload date; it is the earliest copy found and is not asserted to be the original source.",
                  candidates_checked=len(st.verified), candidates_total=len(st.proposed) + len(st.proven),
                  candidate_status_counts={s: sum(c.status == s for c in st.verified.values()) for s in {c.status for c in st.verified.values()}},
                  adapters=st.web, agent_page_reads=st.page_reads)
    rep = build_report(input_meta=input_meta, claim=claim_d, memory=st.memory_layer, visual=visual, clues=st.clues,
                       physical=st.physical, investigation_id=inv_id)
    rep["agent_trace"] = dict(note="SHOWN, NOT TRUSTED: the trace and the agent's summary never feed final_verdict or proven_matches.",
                              model=final["model"], steps=trace, agent_summary=final.get("summary"),
                              agent_original_url=final.get("original_url"), summary_audit=final["audit"],
                              stop_reason=final["stop_reason"], tool_calls=len([t for t in trace if t["tool"] in TOOL_NAMES]),
                              cost_usd=round(st.cost, 4))
    rep["timing_s"] = timing
    validate_report(rep)
    return rep


def run_agent(source: str, claim=None, date=None, place=None, out_dir=None, memory: Memory | None = None,
              max_calls=MAX_CALLS, time_cap=TIME_CAP_S, cost_cap=0.25, invalid_limit=4, model=AGENT_MODEL, llm=None, log=print) -> dict:
    """Run the agent on one clip. `llm(model, text, system) -> (reply_text, cost)` can be injected (tests mock it)."""
    load_env()
    llm = llm or (lambda m, t, s: chat(m, [{"type": "text", "text": t}], max_tokens=1800, system=s, timeout=120))
    t0 = time.time()
    inv_id = uuid.uuid4().hex[:10]
    out = Path(out_dir or ROOT / "runs" / inv_id); out.mkdir(parents=True, exist_ok=True)
    ev_dir = out / "evidence"; ev_dir.mkdir(exist_ok=True)
    clip = media.fetch(source)
    feats = ClipFeatures.from_path(clip.path)
    input_meta = dict(source=source, **clip.to_dict(), n_keyframes=len(feats), keyframe_times_s=[round(t, 2) for t in feats.times])
    input_meta["path"] = clip.path if not clip.url else "(downloaded excerpt removed after run)"
    mem = memory if memory is not None else Memory()
    st = AgentState(clip, feats, claim, date, place, mem, ev_dir, out)
    system = "You are an investigator agent for debunking recycled disaster videos. " + TOOLS_DOC % max_calls
    trace, hist, calls, last_obs, seen_calls = [], [], 0, "(start)", set()
    stop_reason, final = "budget", dict(summary=None, original_url=None)
    cur_model = model
    header = (f"CLIP: {len(feats)} keyframes (ids 0-{len(feats) - 1}), duration {clip.duration and round(clip.duration)}s. "
              f"CLAIM attached to the clip: {claim!r}; claimed event date {date}; claimed place {place}. "
              "The clip is possibly recycled from an earlier event; the claim may be wrong.\n")
    step = 0
    while True:
        step += 1
        if calls >= max_calls:
            stop_reason = "budget_exhausted"; break
        if time.time() - t0 > time_cap:
            stop_reason = "time_cap"; break
        if st.cost >= cost_cap:
            stop_reason = "cost_cap"; break
        prompt = header + f"CALLS USED {calls}/{max_calls}. PROVEN SO FAR: " + (
            "; ".join(f"{u} ({c.upload_date})" for u, c in st.proven.items()) or "nothing") + "\nHISTORY:\n" + \
            ("\n".join(hist[-14:]) or "(none)") + "\nChoose the next tool (JSON only)."
        try:
            txt, cost = llm(cur_model, prompt, system)
        except Exception as e:
            if cur_model == model and model != FALLBACK_MODEL:
                cur_model = FALLBACK_MODEL; trace.append(dict(step=step, tool="llm_fallback", reason=str(e)[:100])); continue
            stop_reason = "llm_error"; break
        st.cost += cost
        a = _parse_action(txt)
        if not a or a.get("tool") not in TOOL_NAMES + ("finish",):
            hist.append(f"{step}: invalid reply; reply with one JSON tool call")
            trace.append(dict(step=step, observation=last_obs[:300], tool=None, reason=(txt or "")[:200], result="invalid action", decision="continue"))
            if sum(1 for t in trace if t.get("result") == "invalid action") >= invalid_limit:
                stop_reason = "invalid_actions"; break
            continue
        tool, args, why = a["tool"], a.get("args") if isinstance(a.get("args"), dict) else {}, str(a.get("thought", ""))[:300]
        if tool == "finish":
            final = dict(summary=str(args.get("summary", ""))[:1500], original_url=args.get("original_url"))
            trace.append(dict(step=step, observation=last_obs[:300], tool="finish", args=args, reason=why, result="stop", decision="stop"))
            stop_reason = "agent_finished"; break
        key = tool + json.dumps(args, sort_keys=True, default=str)
        calls += 1
        ts = time.time()
        if key in seen_calls:
            res = dict(error="duplicate call blocked (loop-breaker); choose a different tool or arguments"); dup = True
        else:
            seen_calls.add(key); dup = False
            try:
                fn = getattr(st, tool)
                res = fn(**{k: v for k, v in args.items() if k in inspect.signature(fn).parameters})
            except Exception as e:
                res = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
        st.note(res)
        last_obs = _short(res, 500)
        hist.append(f"{step}: {tool} {_short(args, 160)} -> {_short(res)}")
        trace.append(dict(step=step, observation=trace[-1]["result"] if trace and "result" in trace[-1] else "(start)", tool=tool, args=args, reason=why,
                          result=_short(res, 700), duplicate=dup, seconds=round(time.time() - ts, 1), decision="continue"))
    if stop_reason != "agent_finished":
        trace.append(dict(step=step, tool="stop", reason=stop_reason, decision="stop"))
    final.update(model=cur_model, stop_reason=stop_reason, audit=audit_summary(final.get("summary") or "", st, final.get("original_url")))
    rep = build_agent_report(st, trace, final, inv_id, input_meta, round(time.time() - t0, 1))
    write_report(rep, out)
    media.cleanup(clip)
    return rep
