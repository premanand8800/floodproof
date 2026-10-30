# Snowball search evaluation (pre-registered)

Written and committed BEFORE any snowball evaluation run.

## What is evaluated
`src/floodproof/snowball.py`: after the pipeline has proven at least one copy, Muse (meta/muse-spark-1.3) reads that copy's metadata and visible watermarks, proposes at most 5 queries aimed at earlier uploads, the queries and the copy's frame go to YouTube Data API (publishedBefore the earliest proven day), Tavily (1 call per query, 2 queries per round), Yandex and Google Lens (at most 1 call per round), and every candidate is verified by the matcher against the INPUT clip. At most 3 rounds, at most 12 extra verifications, at most 90 s. Snowball never names an original; it only extends the list of proven copies.

## Design
- Cases: the 17 frozen debunked cases (H01-H09 from `bench/heldout.jsonl`, G02, G03, G05-G10 from `bench/global.jsonl`), restricted to those where the pipeline proved at least one copy in the saved all-sources baseline (`bench/ablation_v2_runs/all_on`, Muse + YouTube + Lens + Yandex + Tavily). Cases with 0 proven copies are out of scope (snowball does not run on them by design).
- Paired design: "off" is the pipeline's proven set; "on" is the same set extended by the snowball. The baseline copies are re-verified against the input from the cached downloads (same matcher, no API) so that their proof frames exist; a copy that no longer proves is dropped from BOTH arms. This removes Yandex/Lens run-to-run variance from the comparison. It is not a fresh end-to-end rerun of the pipeline.
- Memory is not used (read-only, never written).
- Caps for the whole evaluation: OpenRouter $0.80, Google Lens <= 40 calls, Tavily <= 60 calls, YouTube Data API within the daily quota.

## Metrics
1. Earliest proven date improved: the earliest independently-dated proven copy in "on" is earlier than in "off".
2. Originals found: a proven copy whose URL equals the ground-truth `original_url` (exact normalised match) in "on" but not in "off".
3. New proven copies (count).
4. False attribution: must stay 0. Snowball names no original, so this is checked by auditing every new copy that is dated earlier than the ground-truth original date by looking at its proof.
5. Extra cost (OpenRouter, Lens, Tavily calls) and extra time per case.

## Decision rule (fixed before the run)
Ship snowball as default ON only if ALL hold: it improves the earliest date or finds an original in at least 2 cases; false attributions = 0; added time <= 90 s per investigation. Otherwise default OFF behind a flag (`FLOODPROOF_SNOWBALL=1` or `use_snowball=True`), and the spread timeline still uses the pipeline copies.

---
# Results (run after the section above was committed)

Raw per-case output: `bench/snowball_runs/<ID>.json`; scorer: `bench/snowball_eval.py score` (no API calls) -> `bench/snowball_results.json`. Harness: `bench/snowball_eval.py run`.

## Table (15 cases in scope; H09 and G03 had no proven copy in the baseline, so snowball does not run on them)
| ID | Pipeline copies | New proven | Earliest date off -> on | Original (ground truth) found | Rounds | Snowball time | Stopped |
|---|---|---|---|---|---|---|---|
| H01 | 3 | 1 | 2025-02-26 -> **2024-09-07** | no (true original 2024-04-06) | 2 | 80 s | no new copy in round 2 |
| H02 | 2 | 0 | 2025-04-20 | no | 1 | 52 s | no new copy |
| H03 | 5 | 4 | 2021-07-03 (same) | no | 1 | 84 s | time budget |
| H04 | 1 | 1 | 2017-06-21 (same) | no | 2 | 82 s | no new copy |
| H05 | 1 | 0 | 2026-08-12 | no | 1 | 71 s | no new copy |
| H06 | 1 | 0 | 2024-04-17 | no | 1 | 55 s | no new copy |
| H07 | 3 | 0 | 2026-09-01 | no | 1 | 40 s | no new copy |
| H08 | 1 | 1 | 2026-08-09 | no | 2 | 72 s | no new copy |
| G02 | 4 | 4 | 2021-09-03 (same) | no | 1 | 77 s | time budget |
| G05 | 4 | 3 | 2020-10-30 (same) | no | 2 | 82 s | no new copy |
| G06 | 3 | 1 | 2021-04-09 | no | 2 | 64 s | no new copy |
| G07 | 3 | 1 | 2026-06-09 | no | 2 | 70 s | no new copy |
| G08 | 2 | 0 | 2023-01-28 | no | 1 | 41 s | no new copy |
| G09 | 1 | 0 | 2025-07-30 | no | 1 | 45 s | no new copy |
| G10 | 8 | 5 | 2023-09-04 (same) | no | 2 | 69 s | no new copy |

## Summary against the metrics
| Metric | Result |
|---|---|
| Earliest proven date moved earlier | **1 of 15** (H01: 2025-02-26 -> 2024-09-07; still 5 months after the true original, which was not found) |
| Originals found (exact URL match with ground truth) | **0 of 15** (0 before, 0 after) |
| New proven copies | **21 in 9 cases** (matcher-proven against the input, independently dated) |
| False attribution | **0** (snowball names no original; the one case where a new copy predates the ground-truth original date, H04, is a 2017 copy of the same footage that the pipeline baseline already had, a known fact-check date issue from the held-out audit, not an attribution) |
| Extra time | mean 65.5 s, max 84.0 s per investigation (cap 90 s); 2 cases stopped on the time budget |
| Extra cost | OpenRouter $0.194 total (about $0.013 per case); Lens 5 calls by the harness ledger (4 inside cases, 1 smoke test, plus 1 diagnostic call outside it); Tavily 44 ledger calls (plus 2 diagnostic calls), under the 60 cap; YouTube Data API 0 calls (see below) |

## Decision (rule fixed above)
The rule needs the earliest date improved OR an original found in at least 2 cases. Result: 1. **The rule is not met, so snowball ships default OFF**, behind `FLOODPROOF_SNOWBALL=1`, `investigate(use_snowball=True)` or `floodproof investigate --snowball`. False attribution stayed 0 and time stayed under 90 s, but the benefit criterion failed. The spread timeline does not depend on the snowball: it is built from whatever proven copies the report has (pipeline copies by default).

## Honest caveats
- **YouTube Data API was unavailable during the evaluation (HTTP 429, daily quota already spent)**, so the planned YouTube search with `publishedBefore` ran in 0 rounds. The result is for snowball through Tavily, Yandex and Lens only. A rerun with YouTube quota could differ, particularly since the queries Muse writes are most useful as YouTube search text.
- Tavily mostly returns news pages; only video-platform links are kept, so it produced few candidates (24 video-platform candidates over all rounds, 1 of the 21 new proven copies).
- Lens returned no usable results in most rounds (4 calls billed in 15 cases, harness cap not reached). The reason was not logged per round; it was not a quota stop.
- Paired design on saved baselines (see Design): not a fresh end-to-end run, so Yandex variance is excluded, but so is any effect of the snowball on the pipeline's own timing.
- Most new copies were new Yandex frame results (round 1), i.e. a second Yandex search on a different frame (the seed copy's frame) rather than anything Muse read. Of the 21 new proven copies, 19 came from Yandex on the seed frame, 1 from Lens and 1 from Tavily (a Muse query): Muse's own queries contributed one proven copy.
- One run per case, 15 cases, a small sample; H01's improvement is real (27 consistent frames, 122 inliers) but a single case.
- Time caps are wall-clock and the evaluation host was also running other jobs.

## Demo regenerated with snowball on (`demo/success_recycled`)
Full investigation (memory copy read-only, `scripts/snowball_demo.py`): RECYCLED_FOOTAGE, 1 proven copy (the memory-proven Instagram reel dated 2026-08-21, date re-fetched live). Snowball ran 1 round (47 s): Muse read the reel (title credits a handle, description names Columbia Glacier, watermark "GeliPro"), proposed 5 queries, 28 new candidates, 6 verified, 0 newly proven. So the glacier demo's spread timeline has one real entry and no likely parent; this is the honest result, not a staged one. Multi-copy timeline rendering was checked on a synthetic report (not committed). Demo spend: $0.010 Muse (snowball) plus the run's own Muse calls, 1 Lens call, 2 Tavily calls.

## Spend for this task
OpenRouter: $0.194 evaluation plus about $0.07 for the demo run and debunk card (well under $0.80). Lens: 4 in-case calls + 1 smoke test + 1 diagnostic + 1 demo = 7 of the 40 cap. Tavily: 44 + 2 diagnostic + 2 demo = 48 of 60. YouTube Data API: quota was exhausted (429), 0 snowball searches.
