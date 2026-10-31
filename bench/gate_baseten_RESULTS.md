# Relevance gate on Muse Spark Open (Baseten): results on the frozen set

Pre-registered in `bench/gate_baseten_README.md` (committed before the run). Same 20 clips (`bench/gate.jsonl`), one run, same prompt/frames/scorer,
decision cache bypassed. Raw rows: `bench/gate_baseten_results.json`. All 20 calls were served by Baseten (`provider=baseten`); no fallback used.

| | Spark Open (Baseten) | Glimmer (previous) |
|---|---|---|
| Relevant clips wrongly stopped | **0 / 10** | 0 / 10 |
| Irrelevant clips stopped | **10 / 10** | 9 / 10 |
| Cost, all 20 clips | $0.103 (mean $0.0052) | $0.047 (mean $0.0024) |
| Latency per check | median 7.6 s, mean 11.7 s (range 5-30 s) | median 32 s, mean 36.6 s |

**Rule applied: 0/10 wrongly stopped AND >= 9/10 irrelevant stopped -> met. Default switched to Spark Open; Glimmer is the fallback
(and `FLOODPROOF_GATE_MODEL=glimmer` restores it as primary).** Keyword veto unchanged.

| id | expected | decision | kind | model said (conf) | cost | s |
|---|---|---|---|---|---|---|
| I01 | irrelevant | stop | music_video | no (1.0) | .0036 | 5.2 |
| I02 | irrelevant | stop | gaming | no (.99) | .0032 | 8.2 |
| I03 | irrelevant | stop | other | no (.98) | .0059 | 29.9 |
| I04 | irrelevant | stop | ad | no (.99) | .0046 | 27.8 |
| I05 | irrelevant | stop | other | no (1.0) | .0039 | 6.9 |
| I06 | irrelevant | stop | other | no (1.0) | .0031 | 5.1 |
| I07 | irrelevant | stop | other | no (.99) | .0034 | 7.9 |
| I08 | irrelevant | stop | other | no (1.0) | .0042 | 24.1 |
| I09 | irrelevant | stop | talking_head | no (.93) | .0082 | 13.7 |
| I10 | irrelevant | stop | other | no (.99) | .0055 | 5.9 |
| R01 | relevant | run | music_video | yes (.95) | .0046 | 5.9 |
| R02 | relevant | run | music_video | unsure (.6) | .0120 | 17.1 |
| R03 | relevant | run | other | **no (.88)** | .0084 | 10.4 |
| R04 | relevant | run | news_about_disaster | yes (.98) | .0045 | 8.3 |
| R05 | relevant | run | disaster_footage | yes (.99) | .0036 | 7.1 |
| R06 | relevant | run | disaster_footage | yes (.97) | .0048 | 6.2 |
| R07 | relevant | run | news_about_disaster | yes (.93) | .0068 | 7.3 |
| R08 | relevant | run | disaster_footage | yes (.99) | .0043 | 5.7 |
| R09 | relevant | run | disaster_footage | yes (.95) | .0040 | 25.0 |
| R10 | relevant | run | news_about_disaster | yes (.99) | .0044 | 5.9 |

## Honest caveats
- **R03 was judged "no" at 0.88 and was saved only by the metadata keyword veto** (title/tags carry flash flood, rescue, survivor...). As with R02
  under Glimmer, the veto is load-bearing: a relevant clip with a bland title is at risk with either model. R02 this time was "unsure" (0.6), a better answer than Glimmer's fallback.
- Spark Open is ~2x the cost per check (it is a reasoning model and spends more tokens) but ~4x faster at the median.
- Irrelevant-side gain (I04 now stopped, with cited frames) is one clip; n = 20, one run, a smoke test, not a precision/recall estimate.
- Gate failures still fail open; unparseable replies (think blocks, chat-template tokens) are parsed tolerantly, else Glimmer then Spark 1.3 are tried.
