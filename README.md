# FloodProof

**Where did this disaster video really come from, and does it match the claim?**

During disasters, old or foreign footage goes viral with false captions. During the August 2026 Nepal floods, an Alaska glacier clip was shared as the Rasuwa GLOF. FloodProof takes a video (a link or an upload) plus the claim, and returns an evidence report: matcher-proven earlier copies with frame-by-frame proof, independently re-checked dates, AI clues from the frames and the narration, and a physical-world check against disaster records. The verdict is one of three labels, never a single "fake score".

Live: https://floodproof.aether-soft.tech · Built for the Meta Global AI Developer Hackathon with Meta Muse models.

## How it works

1. **Input.** A video URL or file, plus an optional claim, date and place. Empty fields are inferred from the post's title, caption, on-screen text and narration, and each inferred field is labelled with its source.
2. **Keyframes and features.** About 50 frames are extracted, and SIFT features are computed for each.
3. **Memory.** The clip is compared against previously verified clips (hash shortlist, then matcher proof).
4. **Muse Spark (Meta).** It reads the frames and extracts on-screen text, scene description, location *hypotheses*, the media type (camera footage, news compilation, animation/CGI, likely AI-generated, screen recording) and search queries. Every clue must cite frame numbers.
5. **Muse Spark Open (Meta).** It analyses the narration: language, spoken place names and spoken dates, each with a timestamp.
6. **Discovery.** Candidates come from memory, YouTube Data API (uploads before the claimed date), Yandex reverse image search, Google Lens (SerpApi) and Tavily text search.
7. **Matcher proof.** Every candidate must pass SIFT + RANSAC homography with temporal consistency (≥2 consistent frames, or one frame with ≥60 inliers). Only proven matches count. The upload date of each proven copy is re-fetched independently (live metadata or the Wayback Machine).
8. **Physical-world check.** Copernicus EMS activations, GloFAS river discharge, archive rainfall and elevation (Open-Meteo), ReliefWeb (pending appname approval) and GDELT news coverage. This layer is context only: it never changes the verdict on its own.
9. **Verdict.**
   - `RECYCLED_FOOTAGE`: a proven copy exists with an independently verified date earlier than the claim.
   - `CONSISTENT_WITH_CLAIM`: no contradiction was found. This does not mean the claim is confirmed.
   - `UNVERIFIED`: there is not enough evidence. This does not mean the video is fake.

   Every report shows its limitations.
10. **Report.** `report.json` is the single source of truth, and the web UI renders only validated reports. An optional **Deep investigate (experimental)** mode runs a Muse Spark Open tool-calling agent and shows its full trace.

## Evaluation (honest numbers)

| Set | Earlier copy found | Original found | False attribution | Genuine false positives |
|---|---|---|---|---|
| Held-out, 9 debunked (frozen before runs) | 5/9 (pipeline) | 0/9 (pipeline), 1/9 (agent, post-hoc rerun) | 0 (after audit) | – |
| Global, outside South Asia: 8 debunked + 3 genuine | 5/8 | 0/8 | 0 | 0/3 |
| Synthetic variants (crop, mirror, caption, re-encode) | 29/30 recognised | – | 0/30 leave-one-out | 0/29 |

- **Strength:** proving footage was published earlier, with zero false attributions.
- **Weakness:** finding the true original. Originals mostly live on TikTok, X and Instagram, which search engines rarely index.
- **Agent vs pipeline:** a pre-registered experiment showed the agent did not beat the fixed pipeline (`bench/heldout_RESULTS.md`), so the pipeline is the default.
- All raw results, audits and scorers are in `bench/`.

## Run it

```bash
uv sync
cp .env.example .env      # add OPENROUTER_API_KEY (Muse models); other keys are optional
uv run floodproof investigate https://youtu.be/... --claim "Flood in Rasuwa" --date 2026-08-26
uv run floodproof serve --port 8000     # web UI
PYTHONPATH=src uv run --with pytest pytest -q    # 116 tests
```

## Layout

- `src/floodproof/`:
  - `pipeline.py`: orchestration
  - `matcher.py`: SIFT/RANSAC proof
  - `sources.py`: discovery
  - `investigator.py`: Muse clues and narration
  - `infer.py`: inferred claim
  - `provenance.py`: independent dates
  - `physical.py`: physical-world check
  - `report.py`: schema and verdict rules
  - `agent.py`: Deep investigate
  - `memory.py`: verified-clip memory
  - `web.py` and `static/`: the UI
- `bench/`: frozen benchmarks, eval scripts and results
- `demo/`: four reproducible demo reports (recycled, unverified, genuine with the correct date, genuine with the wrong date)
- `tests/`: the test suite
- Files at the top level (`t*.py`, `rev_*.py`, `cu_test/`, …) are early prototypes kept for history.

## Limits

- A visual match proves shared footage. It does not prove that a claim is false or that the footage is fake.
- "Earliest copy found" is not necessarily the original source.
- Search is not exhaustive. Downloads from Facebook and TikTok are often blocked.
- AI clues (location, media type) are hypotheses, never proof.
- The river model misses flash floods and GLOFs. Country-level locations are too coarse to corroborate a local event.
