# SAM 3.1 overlay masking: results

**The pre-registered evaluation (`bench/sam_README.md`) could NOT be run. No masked-vs-unmasked numbers exist. Decision: ship OFF behind `FLOODPROOF_SAM=1`.**

## What blocked it
- `GET https://api.meta.ai/v1/models` works with the project key and lists `sam-3.1`.
- Every `POST /v1/responses` call with `model: sam-3.1` (two probes, prompts "caption text" and "logo", one overlay-variant frame each) returned
  HTTP 402 `billing_not_configured` ("your team needs to resolve a billing issue"). It is an account billing state, not a request error; it needs the
  account owner to fix billing at dev.meta.ai/billing. Latency of the failed call: 1.4-2.4 s. Cost: $0 (nothing was served).
- Local open-weights fallback: `facebook/sam3` and `facebook/sam3.1` are gated on Hugging Face; the local token gets HTTP 403 (access not granted). So no local SAM was run.

## API shape (from dev.meta.ai/docs/sam/segmenting and /reading-segmentation, request shape not verified end to end because of the 402)
`POST https://api.meta.ai/v1/responses`, Bearer key, body `{model:"sam-3.1", input:[{type:"message", role:"user", content:[{type:"input_text", text:"<noun phrase>"},
{type:"input_image", image_url:"data:image/jpeg;base64,..."}]}], stream:false|true}`. Output text lane: `<0f>id<|box;x1;y1;x2;y2;w;h|><|mask;x=0;y=0;data=H,W,<~|!>base85|>`,
records comma-separated, empty when nothing matches. The implementation parses boxes only (the mask payload is base85, not decoded). The output parser is tested only against
the documented grammar, not against a real response.

## What was built (all OFF by default)
- `src/floodproof/sam.py`: calls SAM for 5 prompts per frame (caption text, logo, watermark, text banner, news ticker), converts boxes to a SIFT keep-mask (4 px padding,
  ignored if it would cover more than 60% of the frame), caches per frame by content hash, API failure = no mask (matching continues unmasked).
- `src/floodproof/matcher.py`: `ClipFeatures.from_frames(..., masker=)` / `from_path(..., masker=)` pass the mask to `detectAndCompute` (flipped features use the flipped mask).
  `from_path` enables it only when `FLOODPROOF_SAM=1`. Stored frames and proof images are never altered. Decision rule unchanged.
- `tests/test_sam.py`: 6 mocked tests (parser, mask + cache, failure fallback, over-coverage guard, no keypoints inside the mask and frames unchanged, flag off by default).
- Not done: report note field and before/after image (only required if shipped ON). `report.py` was not touched.

## Expected cost and latency (estimate, unmeasured)
5 prompts per keyframe x up to 48 keyframes per clip = up to 240 API calls per clip; at about 1.5 s each that is minutes per clip unless parallelised or prompts are merged. This is a reason to
reduce prompts or sample fewer frames before any ON decision. Pricing is not published in what was reachable.

## Honest expectation
The unmasked matcher already proves all 30 synthetic variants (`matcher_rule_RESULTS.md`), so the overlay/combo pairs are saturated and G03 is the only pair that could move; the "+2" bar
is unlikely to be met on this set. To complete the study: fix billing (or get HF access to `facebook/sam3.1`), run the 21 recall pairs and the 56 hard negatives with `FLOODPROOF_SAM=1`, and apply the rule as written.
