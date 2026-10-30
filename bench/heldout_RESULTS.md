# FloodProof held-out results (H01-H09; frozen bench/heldout.jsonl)

Systems: A memory+matcher, B fixed pipeline, C agent (4 invalid-reply limit), C2 = POST-HOC RERUN (diagnostic, not a clean comparison; see below).

**C2 status: post-hoc.** The pre-registered protocol (bench/eval_heldout.py header, heldout_README.txt, commit f908b10 which froze the set before any run) names three systems: A, B, C. C2 appears only in commit 388408b, and eval_heldout.py's own comment says the change was made AFTER seeing 2 C traces cut off by unparseable model replies. What C2 changed vs C: exactly one harness parameter, `invalid_limit` 4 -> 10 (the agent aborts after 10 instead of 4 invalid model replies). Same model, prompt, tools, caps, cases. C2 therefore got more chances on the same frozen cases after we saw C fail, and it is a diagnostic rerun of an altered harness, not a clean comparison against A/B/C. Any C2 advantage is not evidence for promotion. The pre-registered C row (completed 4/9) is the clean agent result.

Raw numbers: bench/eval_heldout.json (original scorer, untouched). Corrected scoring: bench/rescore_heldout.py -> bench/heldout_rescored.json (no API calls).

## H10 missing
H10 was excluded by the pre-run unseen check (bench/heldout_unseen_check.json): the viral clip visually matched two entries already in memory
(instagram DafoQ6vAJCR, youtube 1bI4bXO_E2w), so it was not unseen and eval_heldout.py drops it from CASES (SEEN set). Excluded by design, not a failed download.

## Headline
Discovery of true originals: A 0/9, B 0/9, C 0/9, C2 (post-hoc) 1/9 (H09, Euronews Storm Ivar 2013). Independently dated earlier copies (proof of recycling): B 5/9, C2 4/9, C 2/9, A 0/9.
Raw (original-scorer) false-attribution flags: A 0, B 4, C 2, C2 1 = 7 flags. Audited: all 7 were scorer bugs, audited false attribution is 0 for every system. Both numbers are kept in the table below; the raw ones are not hidden or overwritten.

## Table
| System | Found original | Earlier copy (before claim date) | False attr. (orig scorer -> corrected) | Unsupported claims | Wrong verdict (false support) | Correct abstentions (UNVERIFIED) | Completed | Mean tool calls | Unnecessary calls | Recovery (proof after failed search) | Mean s | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0/9 | 0/9 | 0/9 -> 0/9 | n/a | 0 | 9/9 | 9/9 | n/a | n/a | n/a | 13 | $0.000 |
| B | 0/9 | 5/9 | 4/9 -> 0/9 | n/a | 0 | 4/9 | 9/9 | n/a | n/a | n/a | 236 | $0.153 |
| C | 0/9 | 2/9 | 2/9 -> 0/9 | 0 | 0 | 7/9 | 4/9 | 8.8 | 7 | 2/7 | 191 | $0.604 |
| C2 (post-hoc) | 1/9 | 4/9 | 1/9 -> 0/9 | 1 | 1 | 4/9 | 8/9 | 13.3 | 19 | 4/9 | 265 | $0.807 |

Per case (orig/earlier = found original / earlier copy; FA = original flag -> corrected):

| Case | A | B | C | C2 |
|---|---|---|---|---|
| H01 | UNVER o=0 e=0 FA=0->0 | RECYC o=0 e=1 FA=1->0 | UNVER o=0 e=0 FA=0->0 calls=15 | UNVER o=0 e=0 FA=0->0 calls=5 |
| H02 | UNVER o=0 e=0 FA=0->0 | UNVER o=0 e=0 FA=1->0 | UNVER o=0 e=0 FA=0->0 calls=8 | UNVER o=0 e=0 FA=0->0 calls=14 |
| H03 | UNVER o=0 e=0 FA=0->0 | UNVER o=0 e=0 FA=0->0 | RECYC o=0 e=1 FA=1->0 calls=14 | RECYC o=0 e=1 FA=1->0 calls=14 |
| H04 | UNVER o=0 e=0 FA=0->0 | RECYC o=0 e=1 FA=0->0 | UNVER o=0 e=0 FA=0->0 calls=15 | UNVER o=0 e=0 FA=0->0 calls=15 |
| H05 | UNVER o=0 e=0 FA=0->0 | UNVER o=0 e=0 FA=1->0 | UNVER o=0 e=0 FA=0->0 calls=5 | CONSI o=0 e=0 FA=0->0 calls=15 |
| H06 | UNVER o=0 e=0 FA=0->0 | RECYC o=0 e=1 FA=0->0 | RECYC o=0 e=1 FA=1->0 calls=14 | RECYC o=0 e=1 FA=0->0 calls=15 |
| H07 | UNVER o=0 e=0 FA=0->0 | UNVER o=0 e=0 FA=0->0 | UNVER o=0 e=0 FA=0->0 calls=5 | UNVER o=0 e=0 FA=0->0 calls=15 |
| H08 | UNVER o=0 e=0 FA=0->0 | RECYC o=0 e=1 FA=0->0 | UNVER o=0 e=0 FA=0->0 calls=2 | RECYC o=0 e=1 FA=0->0 calls=15 |
| H09 | UNVER o=0 e=0 FA=0->0 | RECYC o=0 e=1 FA=1->0 | UNVER o=0 e=0 FA=0->0 calls=1 | RECYC o=1 e=1 FA=0->0 calls=12 |

Definitions: earlier copy = matcher-proven copy dated before the claimed date. Correct abstention = label UNVERIFIED (nothing over-claimed).
Recovery = proof obtained after >=1 failed search (trace-derived; C2 H09 switched from reverse_search to event-specific text_search "Storm Ivar Norway..." and proved the Euronews original; H03/H06/H08 likewise reached copies after failures). C stopped early on invalid replies in 5/9 cases (completion 4/9).

## Audit
All 7 raw false-attribution flags (B H01, H02, H05, H09; C H03, H06; C2 H03) were scorer bugs, resolved by reading each case's saved row against the ground truth (details below): 3 were agents naming a proven same-event copy that was not the exact ground-truth URL (or was dated one day before the original), 4 were B reporting an earlier copy dated later than the true original while asserting no origin. None was a system naming a wrong source as the original. The scorer was not "fixed to look better": bench/rescore_heldout.py re-scores the saved rows with the corrected rule and writes bench/heldout_rescored.json only; it makes no API calls, does not rerun anything, and never overwrites bench/eval_heldout.json, so the raw numbers remain reproducible. Real problems the raw flag missed are listed at the end of the list below and count against the system.

### Per-flag resolution
Scorer cause: for agents `agent_original_wrong` = named URL != exact ground-truth URL; for B/others `earliest proven copy dated later than true original`.
- C H03: BUG. Named watchers.news Atami mudslide 2021-07-03 (same event, same date as truth, matcher-proven); copy not the exact URL.
- C2 H03: BUG. Named YouTube jCn45LkTxNs Atami 2021-07-03, matcher-proven same event/date.
- C H06: BUG. Named YouTube short 2024-04-16 (Dubai airport), one day before the 9 News original 2024-04-17; earliest_is_original was already true.
- B H01: BUG. Reported only "copy published 2024-09-07 before claimed date"; no origin claimed; truth 2024-04-06 just not found.
- B H02: BUG. Label UNVERIFIED; copy dated 2025-07-09 later than claim, so no recycling asserted.
- B H05: BUG. Label UNVERIFIED; copy 2026-08-12, no claimed date; nothing asserted.
- B H09: BUG. Reported 2014 copy as a prior copy; truth 2013-12-13 not found; no origin claim.
Real problems the old flag missed: C2 H05 label CONSISTENT_WITH_CLAIM on a debunked clip (physical check "supports place/date", hedged wording, but wrong verdict); C2 H02 one audited unsupported claim ("likely recycled" with no proven match).

## Promotion rule (agent promoted only if discovery > B AND 0 false attributions AND 0 unsupported claims)
Discovery of originals: C2 1/9 vs B 0/9 (better, n=1 case); recycling detection: C2 4/9 vs B 5/9 (not better). False attributions 0. Unsupported claims: C2 1 (C 0). Rule FAILS for C2 (1 unsupported claim, 1 wrong verdict) and C (discovery 0/9 = B, 4/9 completion). Agent NOT promoted.

## Recommendation: pipeline (B) as default
B gives the best recycling detection (5/9) at $0.017/case with no unsupported claims. The agent costs 4-5x more ($0.07-0.09/case), is slower, makes more unnecessary calls (C2 19 over 9 cases), and its only unique win is one original (1/9, n too small to generalise) bought with one unsupported claim and one false-support verdict. Originals are essentially undiscoverable with current sources (0/9 for A, B, C). A hybrid (B first, agent escalation only on UNVERIFIED, with hard claim audit) is plausible since C2 found copies on H03 where B abstained, but that is a post-hoc idea that this frozen set cannot validate; test on a new held-out set before adopting.

## Spend
OpenRouter usage 2.70 -> 4.30 USD = about $1.60 across the experiment (row costs: A 0, B $0.153, C $0.604, C2 $0.807 = $1.56). Small n (9 cases), selection bias per bench/heldout_README.txt.
