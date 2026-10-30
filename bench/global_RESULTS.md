# FloodProof global (non-South-Asia) results: system B only

Frozen set: bench/global.jsonl (commit f02aa39, before any B run). Raw: bench/eval_global.json. Scoring: bench/rescore_global.py (corrected rule: false attribution only when a wrong source is named as origin or RECYCLED with no copy before the claimed date; UNVERIFIED is never a false attribution) -> bench/global_rescored.json. Memory read-only, no --remember.

## What was scored
13 cases built, **11 scored**: G01 and G04 were excluded by the pre-registered unseen check (bench/global_unseen_check.json), not by download failure. G04's viral clip matched a memory entry (Full Fact Apr-2016 Saudi flood original, already indexed); G01 matched 10 memory entries (likely spurious, rule applied as written). So **Europe has no scored case**; scored regions: Americas (G02, G03, N02), Middle East (G05, N01), East Asia/Russia Far East (G06, G09), Southeast Asia (G07), Africa (G10, N03), Oceania (G08). 8 debunked + 3 genuine.

## Table (8 debunked, 3 genuine)
| Metric | Result |
|---|---|
| Found earlier copy (matcher-proven, dated before claim) | 5/8 (G02, G05, G06, G07, G10) |
| Found the named original (exact URL) | 0/8 (G02 and G05 found a same-day re-upload of the same event, 2021-09-03 and 2020-10-30; not the exact URL) |
| False attribution (corrected) | 0/8 |
| Wrong verdict (CONSISTENT_WITH_CLAIM on debunked) | 0/8 |
| Correct abstention (UNVERIFIED) | 3/8 (G03, G08, G09); N03 genuine also UNVERIFIED |
| Genuine false positives (RECYCLED on genuine) | 0/3 (N01, N02 CONSISTENT_WITH_CLAIM; N03 UNVERIFIED) |
| media_nature | G09 AI correctly "likely_ai_generated"; G10 (CGI tornado) called camera_footage = wrong; genuine N02/N03 camera_footage ok, N01 news_compilation (CNN report, reasonable); G03 called likely_ai_generated (no expected label, mirrored real footage) |
| Cost / time | $0.156 total ($0.014/case, Muse clue call); mean 218 s/case (173-247). OpenRouter usage 4.855 -> 5.011, under the $0.40 cap |

## Physical check per region
Ran and resolved a location in 11/11 (Nominatim on the claimed place). But the location is only country/region level in 4 cases (Turkey, Indonesia, Libya low confidence), so the check there returned UNKNOWN; NO_EVIDENCE for Florida, Texas, Kamchatka, Jerusalem. Only G08 (New York) and N03 (Durban) returned SUPPORTS. G08 SUPPORTS is on a debunked clip: the physical check confirms rain on the claimed date, not that the clip is from there. The label stayed UNVERIFIED, so no wrong verdict, but physical data does not discriminate recycled clips.

## Failures and causes
- 0/8 originals: the originals are mostly TikTok/X/YouTube-shorts copies the search adapters did not surface; Yandex and Tavily were queried but no exact-URL original was proven. Candidates checked 14 of ~58-70 per case (max_verify=14 cap).
- G03, G08 UNVERIFIED: no proven copy (G03 altered/mirrored with added tornado, matcher could not tie it even to its own original; G08 original URL unknown). G09: AI clip, nothing to find, correct.
- G10 media_nature missed CGI.
- G07 RECYCLED relies on a 2026-06 copy of a compilation that is partly genuine Indonesia footage; the label is right for the clips but not a clean score.
- Ground truth caveats: G03, G08, G10 not matcher-validated, G03/G10 share one original, G06 original_date is a later re-upload.
- Selection bias: only fact-checks with a still-downloadable X/Instagram viral post; Facebook/TikTok mostly blocked, Africa Check 403.

## Honest read
Outside South Asia the pipeline did not make a false attribution or wrong verdict and caught 5/8 recycled clips by finding an earlier dated copy, at $0.014/case. It does not find originals (0/8, same as the South Asia held-out set) and n=8 with six regions is small; Europe is untested after exclusion.
