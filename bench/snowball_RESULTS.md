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
