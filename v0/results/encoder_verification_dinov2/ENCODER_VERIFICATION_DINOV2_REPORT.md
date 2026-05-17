# Encoder Substrate Verification (DINOv2) — Report

- Generated: 2026-05-11T21:00:11Z
- Frames source: `/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/data/seed7_furniture_frames` (re-rendered seed-7 furniture run)
- Annotations: `/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/main/frame_annotations.jsonl`
- Encoder: `facebook/dinov2-large` (DINOv2 ViT-L/14, CLS token), frozen, eval mode, fp16
- Input: 256×256 RGB → center-crop 224×224 → ImageNet mean/std normalisation. Output L2-normalised.
- Sampling seed: `7` (Check 1) / `8` (Check 2) — matches the V-JEPA 2 verification.
- Pairs per set: `50`
- Model identity: `params=304368640 first8=['0.007526', '-0.000794', '-0.006355', '-0.001850', '0.005047', '0.004688', '0.007008', '-0.003323']`

## 1. Setup

- Bank shape: `(100000, 1024)`, dtype `float32`. 
- Dwell-frame norms (first 1000 sampled): mean `1.000000`, std `0.000000`, min `1.000000`, max `1.000000`.
- Annotations records: `100000` (matches bank length).
- Dwell frames retained per viewing-position (transit excluded):

| viewing_position_id | object type | n dwell frames | n loops with ≥1 frame | mean / loop | min / loop | max / loop |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `Bed` | 6570 | 219 | 30.0 | 30 | 30 |
| 2 | `DiningTable` | 6570 | 219 | 30.0 | 30 | 30 |
| 3 | `Dresser` | 6540 | 218 | 30.0 | 30 | 30 |
| 4 | `Sofa` | 6540 | 218 | 30.0 | 30 | 30 |
| 5 | `Television` | 6540 | 218 | 30.0 | 30 | 30 |

## 2. Check 1 — Cross-instance stability (§5.1)

- Aggregate (across all 5 viewing positions, 250 pairs): mean **`1.0000`**, std `0.0000`, min `1.0000`, max `1.0000`.
- Starting threshold: mean cosine > 0.75. Result: **PASS**.

**Per-viewing-position breakdown:**

| viewing_position_id | object type | n pairs | mean | std | min | max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `Bed` | 50 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 2 | `DiningTable` | 50 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 3 | `Dresser` | 50 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 4 | `Sofa` | 50 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 5 | `Television` | 50 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

**Degeneracy note (carries over from the V-JEPA 2 verification).** The rerender's dwell mechanism teleports the agent to the exact same pose every dwell frame in every loop (per [`results/frame_rerender/RERENDER_REPORT.md`](../frame_rerender/RERENDER_REPORT.md)), so AI2-THOR produces bit-identical pixels per viewing position, and DINOv2 (deterministic forward, frozen, eval) maps identical pixels to identical embeddings. Within-instance cosines therefore reflect rendering + encoder determinism, not encoder stability under natural instance variation. Reported honestly per the batch §2.4; not engineered around.

## 3. Check 2 — Cross-element distinguishability (§5.2)

- Aggregate (across all 20 ordered pairs, 1000 pairs): mean **`0.4422`**, std `0.1324`, min `0.2547`, max `0.6709`.
- Starting threshold: mean cosine < 0.60. Result: **PASS**.

**Per-(probe, retrieve) ordered-pair mean cosine matrix:**

| probe \ retrieve | 1 (Bed) | 2 (DiningTable) | 3 (Dresser) | 4 (Sofa) | 5 (Television) |
|---|---:|---:|---:|---:|---:|
| 1 (Bed) | — | 0.4366 | 0.4467 | 0.5420 | 0.6168 |
| 2 (DiningTable) | 0.4366 | — | 0.3647 | 0.6709 | 0.2547 |
| 3 (Dresser) | 0.4467 | 0.3647 | — | 0.3502 | 0.4757 |
| 4 (Sofa) | 0.5420 | 0.6709 | 0.3502 | — | 0.2640 |
| 5 (Television) | 0.6168 | 0.2547 | 0.4757 | 0.2640 | — |

**Per-pair n / mean / std (full breakdown):**

| pair (i→j) | i type | j type | n | mean | std | min | max |
|---|---|---|---:|---:|---:|---:|---:|
| 1→2 | `Bed` | `DiningTable` | 50 | 0.4366 | 0.0000 | 0.4366 | 0.4366 |
| 1→3 | `Bed` | `Dresser` | 50 | 0.4467 | 0.0000 | 0.4467 | 0.4467 |
| 1→4 | `Bed` | `Sofa` | 50 | 0.5420 | 0.0000 | 0.5420 | 0.5420 |
| 1→5 | `Bed` | `Television` | 50 | 0.6168 | 0.0000 | 0.6168 | 0.6168 |
| 2→1 | `DiningTable` | `Bed` | 50 | 0.4366 | 0.0000 | 0.4366 | 0.4366 |
| 2→3 | `DiningTable` | `Dresser` | 50 | 0.3647 | 0.0000 | 0.3647 | 0.3647 |
| 2→4 | `DiningTable` | `Sofa` | 50 | 0.6709 | 0.0000 | 0.6709 | 0.6709 |
| 2→5 | `DiningTable` | `Television` | 50 | 0.2547 | 0.0000 | 0.2547 | 0.2547 |
| 3→1 | `Dresser` | `Bed` | 50 | 0.4467 | 0.0000 | 0.4467 | 0.4467 |
| 3→2 | `Dresser` | `DiningTable` | 50 | 0.3647 | 0.0000 | 0.3647 | 0.3647 |
| 3→4 | `Dresser` | `Sofa` | 50 | 0.3502 | 0.0000 | 0.3502 | 0.3502 |
| 3→5 | `Dresser` | `Television` | 50 | 0.4757 | 0.0000 | 0.4757 | 0.4757 |
| 4→1 | `Sofa` | `Bed` | 50 | 0.5420 | 0.0000 | 0.5420 | 0.5420 |
| 4→2 | `Sofa` | `DiningTable` | 50 | 0.6709 | 0.0000 | 0.6709 | 0.6709 |
| 4→3 | `Sofa` | `Dresser` | 50 | 0.3502 | 0.0000 | 0.3502 | 0.3502 |
| 4→5 | `Sofa` | `Television` | 50 | 0.2640 | 0.0000 | 0.2640 | 0.2640 |
| 5→1 | `Television` | `Bed` | 50 | 0.6168 | 0.0000 | 0.6168 | 0.6168 |
| 5→2 | `Television` | `DiningTable` | 50 | 0.2547 | 0.0000 | 0.2547 | 0.2547 |
| 5→3 | `Television` | `Dresser` | 50 | 0.4757 | 0.0000 | 0.4757 | 0.4757 |
| 5→4 | `Television` | `Sofa` | 50 | 0.2640 | 0.0000 | 0.2640 | 0.2640 |

## 4. Check 3 — Combined gap (§5.3)

- gap = Check 1 mean − Check 2 mean = `1.0000` − `0.4422` = **`0.5578`**.
- Starting threshold: gap ≥ 0.15. Result: **PASS**.

## 5. Recalibration decision

No recalibration applied. Empirical values are not within ±0.05 of the starting thresholds; recalibration would not be justified per the batch's §3 discipline.

## 6. Verdict

**PASS**

All three checks pass against the starting thresholds. DINOv2 ViT-L/14 CLS produces embeddings whose within-instance stability, cross-element distinguishability, and combined gap meet the architectural requirements stated in §5 of the spec on the seed-7 furniture-run rerendered frames.

## 7. Direct comparison to V-JEPA 2 mean-pool

Both verifications use the same sampling seed (7 / 8), same viewing-position filter, same dwell-frame index pool, same 50 pairs per set. Differences are encoder-only.

| metric | DINOv2 ViT-L/14 (this batch) | V-JEPA 2 mean-pool (prior) | difference |
|---|---:|---:|---:|
| Check 1 aggregate mean | `1.0000` | `1.0000` | `+0.0000` |
| Check 2 aggregate mean | `0.4422` | `0.8697` | `-0.4275` |
| Check 3 gap | `0.5578` | `0.1303` | `+0.4275` |
| Verdict | **PASS** | **FAIL** | — |

**Per-ordered-pair Check 2 cosines (DINOv2 vs V-JEPA 2):**

| pair (i→j) | DINOv2 mean | V-JEPA 2 mean | difference |
|---|---:|---:|---:|
| 1→2 (Bed→DiningTable) | `0.4366` | `0.9032` | `-0.4666` |
| 1→3 (Bed→Dresser) | `0.4467` | `0.8347` | `-0.3880` |
| 1→4 (Bed→Sofa) | `0.5420` | `0.8875` | `-0.3455` |
| 1→5 (Bed→Television) | `0.6168` | `0.8761` | `-0.2593` |
| 2→1 (DiningTable→Bed) | `0.4366` | `0.9032` | `-0.4666` |
| 2→3 (DiningTable→Dresser) | `0.3647` | `0.8442` | `-0.4795` |
| 2→4 (DiningTable→Sofa) | `0.6709` | `0.9210` | `-0.2501` |
| 2→5 (DiningTable→Television) | `0.2547` | `0.8618` | `-0.6071` |
| 3→1 (Dresser→Bed) | `0.4467` | `0.8347` | `-0.3880` |
| 3→2 (Dresser→DiningTable) | `0.3647` | `0.8442` | `-0.4795` |
| 3→4 (Dresser→Sofa) | `0.3502` | `0.8353` | `-0.4850` |
| 3→5 (Dresser→Television) | `0.4757` | `0.8583` | `-0.3826` |
| 4→1 (Sofa→Bed) | `0.5420` | `0.8875` | `-0.3455` |
| 4→2 (Sofa→DiningTable) | `0.6709` | `0.9210` | `-0.2501` |
| 4→3 (Sofa→Dresser) | `0.3502` | `0.8353` | `-0.4850` |
| 4→5 (Sofa→Television) | `0.2640` | `0.8752` | `-0.6112` |
| 5→1 (Television→Bed) | `0.6168` | `0.8761` | `-0.2593` |
| 5→2 (Television→DiningTable) | `0.2547` | `0.8618` | `-0.6071` |
| 5→3 (Television→Dresser) | `0.4757` | `0.8583` | `-0.3826` |
| 5→4 (Television→Sofa) | `0.2640` | `0.8752` | `-0.6112` |

## 8. Honest interpretation

**What the verdict means.** Verdict only — this batch does not propose architectural next steps. PASS on DINOv2 would mean an off-the-shelf encoder option exists that meets §5; FAIL means the encoder substrate problem is not solvable by swapping V-JEPA 2 mean-pool for DINOv2 on this bank, and the §5.5 path is left to human review.

**Caveat from the re-render (carries from RERENDER_REPORT §note-for-next-batch).** Items 3 (Dresser) and 4 (Sofa) — both in LivingRoom — have constant per-item offsets from the original V-JEPA 2 bank's frames at the cosine `0.0005`–`0.0008` level (V-JEPA 2 reading). DINOv2 is its own encoder and re-encodes the rerender's frames directly, so its protocol numbers are internally consistent. The caveat is recorded in case any downstream analysis surfaces an unexplained discrepancy at that magnitude.

**Check 1 is degenerate on this bank (same reason as V-JEPA 2).** The per-item std is `0.0000` because dwell frames at the same viewing position are bit-identical across loops within the rerender (verified in RERENDER_REPORT). Check 2 is load-bearing.
