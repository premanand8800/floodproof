# Relevance gate: results on the frozen set (bench/gate.jsonl, committed before any run)

Gate run alone (no pipeline) on 20 public YouTube videos: 10 irrelevant, 10 relevant. Raw rows: `bench/gate_results.json`,
runner `bench/gate_eval.py`. Model: meta/muse-glimmer-30b (4 frames seeked from the stream URL, metadata prompt, max_tokens 1500),
fallback meta/muse-spark-1.3 (used on 2 clips). Stop rule: relevant=no, confidence >= 0.8, cited frames, no shows_disaster flag,
and no disaster keyword in title/tags/description.

| | count |
|---|---|
| Relevant clips wrongly rejected | **0 / 10** |
| Irrelevant clips stopped | **9 / 10** |
| Cost per check | mean $0.0024 (median ~$0.001; $0.047 for all 20; the two fallback calls cost ~$0.013-0.015 each) |
| Time per check | mean 36.6 s, median 32 s (range 11-97 s); dominated by the model call, not frame grabbing |

Stopped (9): music video, gaming, vlog, lecture, cooking, sports, unboxing, podcast, trailer.
Not stopped (1): I04 (Super Bowl ad compilation). The model said "no / ad" at 0.95 but cited no frames, so the rule (cited frames required)
let it run. Safe direction: it just means the normal pipeline runs.

## Honest caveats
- **R02 (tsunami tribute video) was judged "no, music_video, 0.85" by the fallback model.** It was saved only by the metadata keyword veto
  (title has "tsunami", "earthquake"). Without that veto the gate would have wrongly rejected a relevant clip. The keyword veto is therefore
  load-bearing, not decoration. A relevant clip with a bland title and no disaster tags would be at risk. The threshold was NOT tuned on
  this set (0.8 as specified); raising it to 0.9 would also have saved R02 but would un-stop I09 (0.85).
- Glimmer failed or returned unparsable output on R02 and R03 and the fallback was used (slower: 73 s and 97 s).
- n = 20, one run, public clips picked by search before the run. This is a smoke test, not a precision/recall estimate.
- Gate failures (network, no stream URL, model error) fail open: the pipeline runs.
- Decisions are cached by video id (URL) or file hash (upload); a cached check costs $0. "Run anyway" (force=1) overrides a stop.
