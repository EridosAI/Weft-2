# DINOv2 Cross-Instance Stability — Report

- Generated: 2026-05-11T22:04:56Z
- Frames source: `/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/data/seed7_dinov2_stability_frames` (jittered single-loop collection)
- Annotations: `/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/data/seed7_dinov2_stability_annotations.jsonl`
- Encoder: `facebook/dinov2-large` (DINOv2 ViT-L/14, CLS token), frozen, eval mode, fp16
- Input: 256×256 RGB → center-crop 224×224 → ImageNet mean/std normalisation. Output L2-normalised.
- Sampling seed: `7`
- Pairs sampled per item: `50`
- Model identity: `params=304368640 first8=['0.007526', '-0.000794', '-0.006355', '-0.001850', '0.005047', '0.004688', '0.007008', '-0.003323']`

## 1. Setup

**Jitter parameters (set at collection time):**

- `jitter_position_m = 0.2` — uniform on each horizontal axis
- `jitter_heading_deg = 10.0` — uniform on yaw
- `jitter_seed = 7` — deterministic per-frame jitter sequence
- Fallback ladder (per explorer): 100% → 50% → 25% → unjittered, on NavMesh-unreachable poses

**Dwell frames retained per viewing position and jitter scales applied:**

| viewing_position_id | object type | n dwell | jitter scale 1.0 | 0.5 | 0.25 | 0.0 (fallback) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `Bed` | 30 | 30 | 0 | 0 | 0 |
| 2 | `DiningTable` | 30 | 30 | 0 | 0 | 0 |
| 3 | `Dresser` | 30 | 30 | 0 | 0 | 0 |
| 4 | `Sofa` | 30 | 30 | 0 | 0 | 0 |
| 5 | `Television` | 30 | 30 | 0 | 0 | 0 |

## 2. Per-viewing-position stability

Cosines computed between all sampled within-instance pairs at each viewing position. Pairs sampled uniformly without replacement from the full C(n, 2) set per item; deterministic given the seed.

| viewing_position_id | object type | n pairs | mean | std | min | max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `Bed` | 50 | 0.9467 | 0.0289 | 0.8475 | 0.9889 |
| 2 | `DiningTable` | 50 | 0.9447 | 0.0189 | 0.9037 | 0.9755 |
| 3 | `Dresser` | 50 | 0.9317 | 0.0453 | 0.7834 | 0.9847 |
| 4 | `Sofa` | 50 | 0.8524 | 0.0969 | 0.6682 | 0.9749 |
| 5 | `Television` | 50 | 0.9547 | 0.0223 | 0.9034 | 0.9846 |

## 3. Aggregate

- Across all 250 sampled pairs (5 items × 50 pairs each): mean **`0.9260`**, std `0.0635`, min `0.6682`, max `0.9889`.
- Spec §5.1 threshold: aggregated mean cosine > `0.75` (PASS), `[0.65, 0.75]` (BORDERLINE), `< 0.65` (FAIL).
- Result: **PASS**.

## 4. Verdict

**PASS**

DINOv2 ViT-L/14 CLS produces stable embeddings under per-frame position+heading jitter on the seed-7 furniture items. Combined with the prior DINOv2 PASS on Check 2 (cross-element distinguishability) and Check 3 (combined gap), the §5 protocol is met on a non-degenerate substrate.

## 5. Comparison to prior degenerate Check 1

The prior DINOv2 verification on the rerender's bit-identical frames returned aggregate Check 1 mean `1.0000` with std `0.0000` — a tautology measuring rendering+encoder determinism, not encoder stability under instance variation. This batch's frames carry deliberate per-frame variation; the difference between aggregate `0.9260` here and the prior `1.0000` is the magnitude of encoder response to the configured jitter.

## 6. Honest interpretation

Verdict only — this batch does not recommend an architectural path. The reviewer decides whether to proceed with DINOv2 as the v0 encoder, run additional variation tests at different jitter magnitudes, or move to SIGReg.

**Caveat on the jitter magnitude.** `0.2 m` and `10°` are SCAFFOLDING values per batch §3 — chosen to produce visible framing variation without dramatically changing what's in view. The verdict is conditional on this jitter magnitude; a non-trivially different magnitude could produce a different aggregate. This is not a weakness specific to DINOv2 — it is inherent to any §5.1 test that doesn't first principle-derive the jitter range from a model of natural agent-instance variation. Flagged for review.
