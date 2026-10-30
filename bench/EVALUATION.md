# FloodProof evaluation (one page)

Task: given a viral disaster clip and its claim, find earlier copies with matcher proof and say whether the claim is contradicted. All numbers are from `bench/`; scorers make no API calls. Nothing was tuned on the frozen sets.

**Plain statement: originals found 0/17 on the pipeline across the frozen sets (held-out 0/9, global 0/8). We prove that footage was published earlier; we do not find where it came from.** False attributions: 0 in every experiment below.

| Experiment | Earlier copy found | Original found | False attribution | Wrong verdict / genuine false positives |
|---|---|---|---|---|
| Held-out (pre-registered), 9 debunked | 5/9 | 0/9 | 0 (audited) | 0 / n.a. |
| Global (outside South Asia), 8 debunked + 3 genuine | 5/8 | 0/8 | 0 | 0 / 0 of 3 |
| Synthetic variants (crop, mirror, caption, re-encode) | 29/30 recognised | n.a. | 0/30 leave-one-out | n.a. / 0/29 |
| Ablation v2, Muse on / off / all sources, 17 debunked + 3 genuine | 8 / 7 / 11 of 17 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |

## Pre-registered held-out set (`heldout_RESULTS.md`)
Set frozen in commit `f908b10` before any system run; H10 dropped by the pre-run unseen check (it was already in memory). Systems A (memory), B (fixed pipeline), C (agent). Pipeline B: 5/9 earlier copies, 0/9 originals, $0.153 total. The original scorer raised 7 false-attribution flags across systems; reading each case showed all 7 were scorer bugs (same-event copies, or an earlier copy dated after the true original), so the audited count is 0. Raw and audited numbers are both kept.

## Global set (`global_RESULTS.md`)
13 cases built, 11 scored (G01 and G04 removed by the unseen check, so Europe has no scored case). 5/8 earlier copies, 0/8 originals, 0/3 genuine false positives, $0.156. The physical check did not discriminate recycled clips. Selection bias: only fact-checks whose viral post was still downloadable.

## Agent vs pipeline (`heldout_RESULTS.md`)
Pre-registered: the tool-calling agent (C) completed 4/9 and found 2/9 earlier copies against 5/9 for the pipeline, at 4x the cost ($0.604 vs $0.153). A post-hoc rerun (C2, retry limit raised after seeing failures, so not a clean comparison) found one original (H09) and 4/9 earlier copies, plus one wrong verdict. The agent did not beat the pipeline, so the pipeline is the default and the agent is labelled experimental.

## Ablation v1 and v2 (`ablation_RESULTS.md`, `ablation_v2_RESULTS.md`)
v1 (Muse off vs on, old verify budget): identical outcomes, 10/17 earlier copies both ways, because Yandex filled all 14 verify slots and Muse-query candidates were never checked. v2 (fair round-robin verify): Muse on 8/17, off 7/17, all sources 11/17. 9 proven copies in 5 cases came only from Muse queries; they changed the label in one case (H06). The other differences are Yandex run-to-run variance, so do not read 8 vs 7 as an effect. Lens added the most recall (31 proven copies, 20 calls). Absolute v1 and v2 numbers are not comparable (Yandex varied between days).

## Memory re-share demo (`ablation_RESULTS.md`, `reshare_demo.json`)
A cropped, mirrored, re-captioned, re-encoded copy of a clip already in memory: memory hit, RECYCLED_FOOTAGE with 35 consistent frames, in 75.7 s and 99.9 s versus 264.5 s for the first full run (2.6-3.5x). That is about a minute and a half with no external search, not instant. It works only for clips already in memory.

## Limits
Small sets (9 and 8 debunked cases); one run per arm; Yandex and platform blocking vary between runs; ground truth for some global cases rests on fact-check text. "Earliest copy found" is not the original source.
