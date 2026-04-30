# Encoder Substrate Verification — Report

- Generated: 2026-04-30T20:33:13Z
- Bank source: `/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/main/memory_bank_embeddings.npy`
- Annotations: `/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/main/frame_annotations.jsonl`
- Sampling seed: `7` (Check 1) / `8` (Check 2)
- Pairs per set: `50`

## 1. Setup

- Bank shape: `(100000, 1024)`, dtype `float32`. 
- Embedding norms (first 1000 sampled): mean `1.000000`, std `0.000000`, min `1.000000`, max `1.000000` — consistent with L2-normalised storage; cosine = dot product.
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

## 3. Check 2 — Cross-element distinguishability (§5.2)

- Aggregate (across all 20 ordered pairs, 1000 pairs): mean **`0.8697`**, std `0.0272`, min `0.8347`, max `0.9210`.
- Starting threshold: mean cosine < 0.60. Result: **FAIL**.

**Per-(probe, retrieve) ordered-pair mean cosine matrix:**

| probe \ retrieve | 1 (Bed) | 2 (DiningTable) | 3 (Dresser) | 4 (Sofa) | 5 (Television) |
|---|---:|---:|---:|---:|---:|
| 1 (Bed) | — | 0.9032 | 0.8347 | 0.8875 | 0.8761 |
| 2 (DiningTable) | 0.9032 | — | 0.8442 | 0.9210 | 0.8618 |
| 3 (Dresser) | 0.8347 | 0.8442 | — | 0.8353 | 0.8583 |
| 4 (Sofa) | 0.8875 | 0.9210 | 0.8353 | — | 0.8752 |
| 5 (Television) | 0.8761 | 0.8618 | 0.8583 | 0.8752 | — |

**Per-pair n / mean / std (full breakdown):**

| pair (i→j) | i type | j type | n | mean | std | min | max |
|---|---|---|---:|---:|---:|---:|---:|
| 1→2 | `Bed` | `DiningTable` | 50 | 0.9032 | 0.0000 | 0.9032 | 0.9032 |
| 1→3 | `Bed` | `Dresser` | 50 | 0.8347 | 0.0000 | 0.8347 | 0.8347 |
| 1→4 | `Bed` | `Sofa` | 50 | 0.8875 | 0.0000 | 0.8875 | 0.8875 |
| 1→5 | `Bed` | `Television` | 50 | 0.8761 | 0.0000 | 0.8761 | 0.8761 |
| 2→1 | `DiningTable` | `Bed` | 50 | 0.9032 | 0.0000 | 0.9032 | 0.9032 |
| 2→3 | `DiningTable` | `Dresser` | 50 | 0.8442 | 0.0000 | 0.8442 | 0.8442 |
| 2→4 | `DiningTable` | `Sofa` | 50 | 0.9210 | 0.0000 | 0.9210 | 0.9210 |
| 2→5 | `DiningTable` | `Television` | 50 | 0.8618 | 0.0000 | 0.8618 | 0.8618 |
| 3→1 | `Dresser` | `Bed` | 50 | 0.8347 | 0.0000 | 0.8347 | 0.8347 |
| 3→2 | `Dresser` | `DiningTable` | 50 | 0.8442 | 0.0000 | 0.8442 | 0.8442 |
| 3→4 | `Dresser` | `Sofa` | 50 | 0.8353 | 0.0000 | 0.8353 | 0.8353 |
| 3→5 | `Dresser` | `Television` | 50 | 0.8583 | 0.0000 | 0.8583 | 0.8583 |
| 4→1 | `Sofa` | `Bed` | 50 | 0.8875 | 0.0000 | 0.8875 | 0.8875 |
| 4→2 | `Sofa` | `DiningTable` | 50 | 0.9210 | 0.0000 | 0.9210 | 0.9210 |
| 4→3 | `Sofa` | `Dresser` | 50 | 0.8353 | 0.0000 | 0.8353 | 0.8353 |
| 4→5 | `Sofa` | `Television` | 50 | 0.8752 | 0.0000 | 0.8752 | 0.8752 |
| 5→1 | `Television` | `Bed` | 50 | 0.8761 | 0.0000 | 0.8761 | 0.8761 |
| 5→2 | `Television` | `DiningTable` | 50 | 0.8618 | 0.0000 | 0.8618 | 0.8618 |
| 5→3 | `Television` | `Dresser` | 50 | 0.8583 | 0.0000 | 0.8583 | 0.8583 |
| 5→4 | `Television` | `Sofa` | 50 | 0.8752 | 0.0000 | 0.8752 | 0.8752 |

## 4. Check 3 — Combined gap (§5.3)

- gap = Check 1 mean − Check 2 mean = `1.0000` − `0.8697` = **`0.1303`**.
- Starting threshold: gap ≥ 0.15. Result: **FAIL**.

## 5. Recalibration decision

No recalibration applied. Empirical values are not within ±0.05 of the starting thresholds, so the starting thresholds fall outside the 'borderline' band and a recalibration would not be justified per §3 of the batch instructions.

## 6. Verdict

**FAIL**

Encoder does not meet the protocol on this bank. The empirical values are sufficiently far from the starting thresholds that a single justified recalibration cannot bring them within criteria. Per spec §5.5, this stops v0 implementation; encoder substitution / fine-tuning is a v0 design decision left to the human reviewer.

## 7. Honest interpretation

**What the verdict means for Weft Inner PAM v0:** the verdict is a precondition check, not a recommendation. PASS clears v0 to proceed with frozen V-JEPA 2 mean-pool as the encoder; FAIL or BORDERLINE leaves that decision to the human review per spec §5.5. This batch does not propose an encoder substitution or any other architectural change — that conversation lives separately.

**Reading the per-(item, pair) breakdowns:** items or pairs with means far from the aggregate are the failure modes if any check fails. The per-pair matrix in §3 shows where cross-element confusability is concentrated; the per-item table in §2 shows whether one item's dwell frames are unusually unstable across loops.

**Check 1 is degenerate on this bank — Check 2 is the load-bearing failure.** The Check 1 aggregated mean of `1.0000` with std `0.0000` across all 50 pairs of all 5 items is *not* a measure of encoder stability under natural variation. The seed-7 furniture-run dwell mechanism teleports the agent to the *exact same pose* (same XZ position, same heading, same horizon) every dwell frame in every loop, so the AI2-THOR rendering pipeline produces *bit-identical pixels* across instances of the same viewing position. V-JEPA 2 (deterministic forward, frozen) then maps identical pixels to identical embeddings. The within-instance cosine is therefore a tautology: it is testing only that V-JEPA 2 is deterministic, which is the implementation, not the encoder property the spec wanted to probe. The spec §5.1 phrasing — *"the architecture requires that recurring elements produce stable representations"* — was written assuming instances would carry natural variation (e.g., different camera angles of the same furniture across loops). This bank does not provide that.

The same pattern shows up in §3's per-pair breakdown: the std on every ordered pair is `0.0000`. Sampling 50 cross-element pairs is redundant when there is only one distinct cosine value per ordered pair (because every dwell frame at item *i* is bit-identical to every other dwell frame at item *i*). The 1000-pair aggregate in Check 2 is in effect 10 unique cosine values (one per unordered pair), each duplicated 100×. This does not invalidate the *direction* of the Check 2 result — the cross-element similarity of `0.8697` is the encoder genuinely failing to separate the 5 furniture items — but it does mean the std and confidence numbers in the per-pair table are sampling artifacts, not encoder noise.

**The structural finding is Check 2.** V-JEPA 2 mean-pool produces embeddings whose cross-element cosines on these 5 furniture items (Bed, DiningTable, Dresser, Sofa, Television, all viewed from `~1.75 m` in seed 7's house) range from `0.8347` (Bed ↔ Dresser, the most distinguishable pair) to `0.9210` (DiningTable ↔ Sofa, the least). All 10 distinct cross-pair values are well above the 0.60 starting threshold, and matter independently of Check 1's degeneracy. This is consistent with the prior Stage 0b room-distinctness diagnostics (in the previous repo's `results/stage_0b/`), which found V-JEPA 2 mean-pool produces 0.95+ inter-room cosines and ~0.83-0.92 inter-room cosines under DINOv2 CLS — same encoder family, same scene-context-dominated geometry.

**On Check 3.** Gap = `0.1303`, just below the 0.15 threshold. But Check 1's `1.0000` is degenerate, so the gap measurement inherits the same caveat: it is `1.0000 − 0.8697`, and the `1.0000` is not real instance-to-instance stability. A gap measured against a non-degenerate Check 1 (with real instance variation) would be smaller, not larger — instances introduce variation, lowering Check 1's mean. The Check 3 result therefore corroborates the Check 2 finding rather than adding independent signal.
