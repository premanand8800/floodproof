# SAM 3.1 overlay masking: pre-registration (written before any run)

Date written: 2026-10-31. Deadline 2026-11-02 12:00 Nepal time. Nothing below may be changed after the first masked run.

Question: does removing SIFT keypoints that fall inside overlay regions (captions, logos, watermarks, tickers), found by SAM 3.1 with text prompts,
make the matcher prove more true recycled copies without making it accept false ones?

## Cases (recall side)
1. Synthetic overlay variants of the 10 leave-one-out originals used by `bench/matcher_rule_eval.py`
   (`ig_leigh, ig_sainj, yt_sun_italy, yt_ie_nalagarh, yt_dawn, yt_tauktae1, yt_ramban, yt_japan2015, ig_pak2022, yt_sikkim_austria`),
   built with `floodproof.matcher.VARIANTS`:
   - `overlay` (black caption box over the bottom 20% plus "BREAKING NEWS" text) x 10
   - `combo` (hflip + 80% crop + rescale + red top banner) x 10
   That is 20 pairs. (`crop70`, `mirror` have no overlay and are not scored; `lowres` is not scored.)
2. Global case G03 (`bench/global.jsonl`, the mirrored "Tornado in Texas" repost with an added tornado) against its named original
   (YouTube n3vidANS1-c, Hurricane Matthew compilation, building at about 4:25). The earlier matcher run found 0 frames for the segments tried.
   If the original cannot be downloaded this is reported as "not testable", not as a failure or success.

## Metric
Number of the 21 pairs for which `matcher.match` returns decision=True, WITHOUT masking vs WITH masking. Everything else is identical
(same keyframes, same SIFT, same ratio test, RANSAC, consensus, the unchanged rule `decide`). Masks apply to the viral clip A and the candidate B.
Also reported: consistent frames and max inliers per pair, SAM calls, cost, and latency per clip.

## Safety (hard negatives)
The 56 hard-negative pairs of `bench/matcher_rule_RESULTS.md` (16 curated same-genre pairs from `run_neg.py`, 40 pairs among the 10 flood
originals, known sibling uploads excluded; list built by `bench/matcher_rule_eval.py`). Run WITH masking on both sides. Allowed false matches: 0.
Baseline for this set is 0 false matches (`bench/matcher_rule_data.json`).

## Decision rule
Ship masking ON by default ONLY IF both hold:
- masked matches proven >= unmasked matches proven + 2 on the 21 recall pairs, and
- masked hard-negative false matches = 0.
Otherwise ship OFF: code stays behind the flag `FLOODPROOF_SAM=1` (default unset), results are reported honestly.

## Known limit stated in advance
The unmasked matcher already recognises all 30 earlier synthetic variants, so the 20 synthetic pairs are probably saturated and cannot show a gain of 2.
G03 is the only pair likely to move. We do not add easier-to-win cases after seeing results. If the gain is below 2 the answer is OFF.

## Method notes
- SAM 3.1 called through the Meta Model API (https://api.meta.ai/v1, model `sam-3.1`) with text prompts: "caption text", "logo", "watermark",
  "text banner", "news ticker". Masks cached per frame (content hash). Proof images saved to reports are never modified.
- Masks are applied only as the `mask` argument of SIFT `detectAndCompute`.
