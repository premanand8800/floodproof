# Muse ablation v2 (fair re-run). v1 is kept unchanged in `ablation_RESULTS.md`

Raw: `bench/ablation_v2.json` (per case, all three modes), per-run reports `bench/ablation_v2_runs/<mode>/<ID>.json`. Scripts: `bench/ablation_v2.py` (run), `bench/ablation_v2_analyze.py` (score, no API calls).

## Why a v2
v1 found that the 14-slot verify budget was always filled by Yandex, so Muse-query (Tavily) candidates were never checked. Master now verifies round-robin (at least 2 slots per source) and runs Tavily on the plain claim text when Muse is off. v2 re-measures with that code.

## Setup
- Same 20 cases: H01-H09, G02, G03, G05-G10 (17 debunked), N01-N03 (3 genuine). Pipeline B, same claim/date/place, `max_verify=14`.
- Memory: a copy of `data/memory` (read-only, `remember=False`). Google Vision, YouTube API and Lens disabled in the two ablation modes; Yandex and Tavily on in both.
- **muse_on**: live Muse clues (vision + narration), Tavily on Muse queries. **muse_off**: `use_muse=False`, Tavily on the plain claim text.
- **all_on**: Muse on plus YouTube Data API plus Google Lens (SerpApi). Muse clues were replayed from the muse_on run (same clues, no second bill). Lens capped at 20 calls; used exactly 20 (one per case).
- `explain.py` (a Muse call per proof) was disabled in every mode so cost and comparison stay fair.
- Code: the Muse on/off arms ran on the branch before merging master `ad7e52b`; the merge only adds proof frames, schema 1.5, `explain.py` and UI, and does not touch discovery, merge/prioritise or verify (checked by diff). The all_on arm ran after the merge.
- Muse spend (live, once): $0.393. No other paid API. Lens: 20 of the 250/month quota.

## Table (17 debunked + 3 genuine)
| Metric | Muse on | Muse off | All sources on (Muse + YouTube + Lens) |
|---|---|---|---|
| Earlier copy found (debunked) | 8/17 | 7/17 | 11/17 |
| Original found | 0/17 | 0/17 | 0/17 |
| False attribution | 0 | 0 | 0 |
| Wrong verdict (CONSISTENT on debunked) | 0 | 0 | 0 |
| Abstention UNVERIFIED (debunked) | 9/17 | 10/17 | 6/17 |
| Genuine false positives | 0/3 | 0/3 | 0/3 |
| Proven copies, total | 37 | 34 | 54 |
| Proven per source (a URL found by two sources counts for both) | Yandex 26, Tavily 12 | Yandex 26, Tavily 8 | Lens 31, Yandex 24, Tavily 4, YouTube API 1 |
| Candidates per case (mean) | 67.0 | 52.8 | 66.3 |
| Mean time per case | 128 s | 176 s | 154 s |

Labels identical between Muse on and off: 16/20. Identical proven-URL sets: 10/20.

## Proven copies that came ONLY from Muse-generated queries
Defined strictly: found by Tavily alone in the Muse-on run, and not in the Muse-off proven set (so also not found by Yandex or by Tavily on plain claim text).
| Case | Muse-only proven copies | Effect |
|---|---|---|
| H06 | 3 YouTube videos | Muse off: UNVERIFIED; Muse on: RECYCLED_FOOTAGE with an earlier copy. The only case where Muse changed the outcome. |
| H08 | 2 YouTube shorts | Label RECYCLED in both modes (other copies already found). |
| G06 | 1 YouTube video | Label RECYCLED in both modes. |
| N01 | 2 YouTube videos | Genuine case, CONSISTENT in both modes. |
| N03 | 1 YouTube video | Genuine case, CONSISTENT in both modes. |

9 URLs in 5 cases; 1 case with a different label. This reverses v1's "Muse discovery is about zero": once Tavily candidates get verification slots, Muse queries do surface copies Yandex and plain-text search do not.

## Honest caveats
- One run per arm. Yandex is not stable: H03 and N02 (Yandex copies found only in the Muse-on run) and H09 (a Yandex copy found only in the Muse-off run) differ with no Muse-only copy behind them, so they are run-to-run Yandex variance, not a Muse effect. The 8/17 vs 7/17 headline difference is within that noise; only H06 is attributable to Muse.
- Yandex returned fewer proven copies than in v1 (v1: 67 proven in each mode; v2: 26 Yandex in each mode). v1 and v2 are different days and different Yandex behaviour, so v1 and v2 absolute numbers must not be compared. Earlier-copy rate with Muse went 10/17 (v1) to 8/17 (v2).
- Time: Muse-on is faster per case (128 s vs 176 s) mostly because Tavily candidates fill the budget with fast, dated YouTube pages and the early-stop fires; not a controlled timing result.
- All sources on: Lens contributed most extra proven copies (31) and turned H01, G02, G08 from abstain to RECYCLED. YouTube API proved 1 copy. Earlier copies 11/17. Still 0/17 originals, 0 false attributions, 0/3 genuine false positives.
- Lens requires a publicly served keyframe URL; the run dirs for this arm were written under the served `runs/` directory and deleted afterwards.
- Net: Muse is mainly a triage/explanation layer (v1 findings stand), plus a small, real discovery contribution through Tavily queries; Lens is the largest single recall gain.
