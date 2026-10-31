# Pre-registration: relevance gate on Muse Spark Open (Baseten)

Written and committed before any run.

- Question: can the gate's model call move from Glimmer (OpenRouter) to Muse Spark Open (open weights, Baseten), with Glimmer as fallback?
- Data: the same frozen 20-clip set, `bench/gate.jsonl` (10 irrelevant, 10 relevant). One run. Same scorer (`gate.decide`, same prompt, same 4 frames, same 0.8 confidence threshold, same metadata keyword veto, unchanged). The gate's decision cache is bypassed so it is a real re-run.
- Ship rule: switch the default ONLY if 0/10 relevant clips are wrongly stopped AND >= 9/10 irrelevant clips are stopped (matching the current Glimmer result of 0/10 and 9/10).
- Otherwise: Glimmer stays the default and Spark Open stays available behind `FLOODPROOF_GATE_MODEL=spark-open`.
- Reported regardless of outcome: per-clip table, counts, cost, median latency vs Glimmer (median 32 s, mean $0.0024).
- No threshold or prompt tuning after seeing results.
