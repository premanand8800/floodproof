# Audit of the single-frame match rule

Rule today (`matcher.decide`): MATCH = >= 2 temporally consistent frames OR 1 frame with >= 60 inliers.

## 1. How many proven matches rely ONLY on the single-frame path?
Scanned every saved `report.json` (main checkout `runs/`, `demo/`, ablation runs `ablv2_*`, other worktree copies): 74 proven-match
entries in 36 reports. 66 have >= 5 consistent frames, 6 have 2, and 2 have 1 (the same match, duplicated in a worktree copy): **one distinct
single-frame-only match**. `bench/eval_heldout.json` / `eval_global.json` store only URLs and verdicts (no per-match frame counts), so the
per-match stats come from the reports.

- ablv2_H04 vs youtube 5eJJpmYmJQU (yandex, 2017-06-21): 1 consistent frame, 73 inliers, inlier hull 0.10 (viral) / 0.33 (copy) of the frame.
- **It was correct**: H04 is CCTV footage of the 2015 Gorkha earthquake (`bench/heldout.jsonl` notes; Newschecker cites a 2015 reupload).
  It is the heldout case where system B proved an earlier copy that A/C/C2 missed (`bench/heldout_RESULTS.md`).
- Wrong single-frame matches in saved data: **0**. Re-running that pair today (`bench/h04_single.py`) gives 1 frame / 54 inliers /
  spread 0.14, i.e. below 60 (different download/keyframes): this match sits at the edge of the rule.

So n = 1. The single-frame path is rarely used, but the one time it was used it carried a true match with a LOW inlier spread (0.10).

## 2. Options
- **A: single frame needs >= 60 inliers AND spread >= 20%.** Would have rejected the H04 true match (spread 0.10), losing the only known
  true single-frame match. Rejected.
- **B (chosen): a single-frame-only match is `weak_match`.** Still shown (decision and evidence unchanged), but it cannot by itself yield
  RECYCLED_FOOTAGE: it needs an independently verified date (already required for any RECYCLED) AND Muse must not describe the scene as
  a landmark (`scene_type` matching landmark/monument/tower/bridge/cathedral/temple/statue/skyline/famous/waterfall/...). A weak match also
  never counts toward CONSISTENT_WITH_CLAIM. Weak matches carry `match.weak_match=true` (plus `match.weak_blocked` with the reason when
  vetoed) and the UI prints "WEAK MATCH". H04 still resolves to RECYCLED_FOOTAGE (YouTube date fetched by yt-dlp, scene "dense urban street").

Caveat: the landmark veto relies on Muse's free-text scene_type, not a dedicated landmark classifier. It is a heuristic guard.

## 3. Re-run (matcher alone, `bench/matcher_rule_eval.py`, raw rows in `bench/matcher_rule_data.json`)
| set | old rule | A | B |
|---|---|---|---|
| 30 synthetic variants (10 originals x crop70, mirror, combo): recall | 30/30 | 30/30 | 30/30 |
| 56 hard negatives (16 curated same-genre pairs from `run_neg.py` + 40 extra pairs among the 10 flood originals; known sibling uploads excluded): false matches | 0 | 0 | 0 |

The README's "29/30" is the memory-lookup path (one mirrored variant ranked second); the matcher itself recognises all 30.
Every variant matches with >= 9 consistent frames, so none exercises the single-frame path; the strongest negative is 1 frame / 19 inliers,
far below 60. This test therefore cannot distinguish the rules; the choice rests on the H04 audit.
Honest limit: besides H04 there is no stored case that depends on the single-frame path, so the reviewers' scenario (a famous landmark
filmed by different people) is guarded by the veto but NOT measured here.
