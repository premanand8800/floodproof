# Muse ablation (pipeline B, use_muse=False) and memory re-share demo

Raw: `bench/ablation_muse.json` (per-case, both sides), per-run reports `bench/ablation_runs/<ID>.json`. Scripts: `bench/ablation_muse.py` (run), `bench/ablation_analyze.py` (compare, no API calls). Cost of the ablation: $0.00 (no Muse call, no paid API; Yandex/yt-dlp only).

## Setup and fairness
- Cases: the frozen debunked + genuine sets, 20 scored: H01-H09, G02, G03, G05-G10 (17 debunked), N01-N03 (3 genuine).
- **Muse ON** = the original B runs. Not rerun: their saved reports (`.cache/{heldout,global}_runs/B/<ID>/report.json`) and eval_heldout.json / eval_global.json.
- **Muse OFF** = `investigate(use_muse=False)` with the same claim/date/place, same memory (37 benchmark originals, identical entry count to the original unseen checks, read-only, `remember=False`), same `max_verify=14`, Yandex ON.
- **New sources disabled** so the comparison matches the original environment: `use_gvision=False, use_youtube=False, use_lens=False` (the original runs predate YouTube API, Lens, Vision). No source code changed.
- Structural fact: in `pipeline.py` Tavily runs only if Muse clues succeeded (its queries are Muse's). So `use_muse=False` also removes Tavily. "Muse off" here = "no Muse and no Tavily". A Tavily-with-plain-claim-query variant was not run (it would need a code change).
- Not controlled: Yandex results vary run to run (see H02, G06 below); Muse-on candidate counts come from saved reports without per-source stats, so Tavily's share is inferred from the difference in `candidates_total`.

## Table (17 debunked + 3 genuine)
| Metric | With Muse | Without Muse |
|---|---|---|
| Earlier copy found (debunked) | 10/17 | 10/17 |
| Original found | 0/17 | 0/17 |
| False attribution | 0 | 0 |
| Wrong verdict (CONSISTENT on debunked) | 0 | 0 |
| Abstention UNVERIFIED (debunked) | 7/17 | 7/17 |
| Genuine false positives | 0/3 | 0/3 |
| Labels identical per case | 20/20 | |
| Identical proven-URL sets | 18/20 | |
| Candidates generated per case (mean) | 68.6 (Yandex ~38.5 + Tavily ~30) | 38.5 (Yandex only) |
| Candidates verified per case | 14.0 | 14.0 |
| Proven copies, total | 67 | 67 |
| Proven per source | Yandex 67, Tavily 0 | Yandex 67 |
| Mean time per case | 226 s | 202 s |
| Muse cost per case | $0.017 | $0 |

The two set differences are Yandex variance, not Muse: H02 gained one YouTube short without Muse, G06 lost one with Muse off (one URL each, no label change).

## Honest contribution of Muse
**Discovery: about zero in these runs.** All 67 proven copies came from Yandex reverse image search (memory missed in every case). Muse-driven Tavily added ~30 candidates per case, but the verify budget (14) is filled by Yandex candidates first (priority order, Yandex always yields 24-40), so Tavily candidates were effectively never checked. Same earlier-copy rate (10/17), same 0/17 originals, same labels (20/20). Do not claim that Muse finds footage. Caveat: with the cap lifted or Yandex blocked (captcha), the Muse queries might matter; not measured.

**Where Muse adds value (verdict-side information that exists only with Muse):**
- `media_nature`: 11 of 20 cases with Muse, 0 without; includes G09 and G03 flagged `likely_ai_generated` (G09 is the AI/animation test clip) and N01 `news_compilation`. This is a hypothesis chip, never changes the label. Misses: G10 (CGI tornado) called camera footage.
- Observations that clash with the claim: 5 cases (G02, G03, G06, G07, G09; 2-4 each), 0 without.
- Narration places (Muse Spark Open): 2 cases (H02 two Nepal village names, H09 "Canada"), 0 without.
- Physical check: 3 cases (H04, H07, H09) have a different physical verdict without Muse (UNKNOWN instead of NO_EVIDENCE / SUPPORTS): Muse parses country and event type from the claim, which the physical layer needs. Label unchanged (physical never decides the verdict).
- On-screen dates: 0 cases in either mode (none found in this set). Inferred claims: not exercised, every case supplied its claim, so no ablation evidence either way.
- Muse also reads on-screen text (e.g. "House wall collapses in florida hurricane", TikTok handle) and writes the report summary lines, not measured here.

Net: Muse is an explanation/triage layer (what the clip is, why it clashes, AI-likeness, parsed place/event type), not a discovery engine. It costs $0.017 and ~25 s per case.

## Memory re-share demo ("Fakes recycle. So does our memory.")
`scripts/demo_reshare.py` -> `bench/reshare_demo.json`, `demo/reshare_memory/report.json` (passes `validate_report`).
- Source: `data/videos/x_iambmane.mp4` (the Alaska glacier clip behind demo/success_recycled), 320x568, 593 kb/s.
- Variant (ffmpeg): crop to 80%, hflip, black caption bar with Nepali text "बाढीको भिडियो: रासुवा, आजै", H.264 at 150 kb/s, AAC 48 kb/s -> 256x454, 147 kb/s.
- Run: memory ON (a copy of data/memory, 37 entries, unchanged), Yandex/Tavily/YouTube/Lens/Vision/Muse all OFF, $0.
- Result: memory HIT, verdict RECYCLED_FOOTAGE. Proven copy `instagram.com/reel/DcUFQZhvGzS/` (date 2026-08-21, independently re-fetched via yt-dlp live metadata): 35 temporally consistent frames, max 175 inliers, `mirrored: true`, stage full.
- Time: **75.7 s and 99.9 s** (two runs, idle machine) vs 264.5 s for the first full run = **2.6-3.5x faster**. This is not "seconds": memory matching alone is ~31-33 s (hash shortlist top 6 plus full matcher on a mirrored crop), the rest is clip load, independent date re-fetch and the physical check. Say "about a minute and a half, with a proven match and no external search", not "instantly".
- The memory hit only works because the original was already in memory; a re-share of a clip never seen is a full run.
