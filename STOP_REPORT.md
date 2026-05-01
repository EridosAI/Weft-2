# STOP_REPORT — Seed-7 Furniture Re-Render Determinism Check FAIL (2026-05-01)

## Triggering condition

Per the batch instructions §5 and §8:

> Pass criterion: cosine similarity > 0.9999 between re-rendered embedding and original bank embedding for every sampled frame (effectively bit-identical given floating-point rounding).
> If the determinism check fails (any sampled frame's cosine < 0.9999): This is an unconditional stop.

The determinism check ran 50 samples (10 per viewing position) against the previous repo's seed-7 bank. **30/50 passed**; **20/50 failed** at the 0.9999 threshold. `cos_min = 0.999188`, `cos_max = 1.000000`.

## Evidence

Full per-sample output in `results/frame_rerender/determinism_check.json`. Headline by viewing position:

| viewing_position_id | object type | room | n samples | mean cos | min cos | max cos | < 0.9999 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Bed | Bedroom | 10 | 1.000000 | 1.000000 | 1.000000 | 0/10 |
| 2 | DiningTable | Bedroom | 10 | 1.000000 | 1.000000 | 1.000000 | 0/10 |
| 3 | Dresser | LivingRoom | 10 | 0.999188 | 0.999188 | 0.999188 | 10/10 |
| 4 | Sofa | LivingRoom | 10 | 0.999481 | 0.999481 | 0.999481 | 10/10 |
| 5 | Television | Bedroom | 10 | 1.000000 | 1.000000 | 1.000000 | 0/10 |

Pattern is striking and consistent:

1. **Items 1, 2, 5 (all Bedroom): bit-identical between original and re-render.** Cosine `1.000000` exactly across all 10 samples per item. Bedroom is the agent's spawn room; renders there are reproducible.
2. **Items 3 and 4 (LivingRoom): differ by a small, *exactly-constant* amount.** Every sampled item-3 frame has `cos = 0.999188` against its corresponding bank entry — the same value to 6 decimal places, across 10 samples drawn from different loops. Item 4 has `cos = 0.999481`, also exactly identical across all 10 samples. This means the rendering produces **one deterministic embedding per item per run**, but the original-run embedding for items 3 and 4 differs from the re-render-run embedding by a per-item constant.

Auxiliary file-level evidence:

- `frame_annotations.jsonl` matches **exactly** between original and re-render (identical md5 `6f241260c0059e57bf96585388aa2fc8`). The trajectory, dwell schedule, transit micro-step sequence, and per-frame metadata are deterministic.
- `memory_bank_embeddings.npy` differs (md5 `a1c581e5...` vs `3029a6a8...`), localised — by the verification — to items 3 and 4. Items 1, 2, 5 match bit-for-bit; items 3 and 4 are off by an item-specific constant offset.

## Analysis

**The rerender is deterministic *within a run*** (frames at the same viewing position across loops produce bit-identical embeddings — confirmed by the constant per-item cosines). The non-determinism is **between the original and re-render runs**, localised to **two of the five items**, both in **LivingRoom**, the room the agent enters second.

**Most plausible cause: scene-state-dependent rendering on first entry to LivingRoom.** AI2-THOR's ProcTHOR-10K rendering pipeline can have:

- **Asset / GPU shader compilation:** the first time geometry from a never-visited room is rendered, shader compilation, BVH build, or texture upload may run in a non-deterministic order, producing pixels that differ at the bit level from a previous run's first-render. Once warm, subsequent renders at the same pose are deterministic — explaining why the within-run cosine is 1.0 across loops.
- **Physics settling on first instantiation:** small dynamic objects (cushions on the sofa, items on the dresser) may settle into slightly different rest poses depending on RNG / step ordering when first instantiated. Once settled, they stay put for the rest of the run.
- **GPU non-determinism in V-JEPA 2 itself:** less likely, because items 1/2/5 match bit-for-bit. If V-JEPA 2 forward were non-deterministic in a way that affected this comparison, it would presumably affect all items, not preferentially LivingRoom.

The bit-identical match for the Bedroom items (and the bit-identical match for items across loops in the rerender) strongly suggests V-JEPA 2 is deterministic in eval mode on this stack; the variance is upstream, in AI2-THOR rendering.

## Numerical interpretation (informational, not a recalibration)

The cosines `0.999188` and `0.999481` correspond to L2 distances of `√(2 − 2·cos)` ≈ `0.040` and `0.032` respectively between unit vectors, against an embedding dimension of 1024 with typical inter-frame L2 distances of ~`0.55` for cross-element pairs (Stage 0b distinctness) and ~`0.0` for bit-identical renders. The re-render embeddings are **far closer** to "identical" than to "different content" — they are 14–17× closer to their original-run counterparts than two different furniture items are to each other in V-JEPA 2's embedding space.

This does not change the verdict. The threshold is `0.9999`, and 20 samples are below it. The numerical interpretation is offered as context for the reviewer's decision, not as a proposal to relax the threshold.

## What is unblocked vs blocked

- **The frames are usable for *internally-consistent* downstream encoder verification.** The rerender's bank is self-consistent (deterministic within the run), so a DINOv2 verification could run against the re-render's frames and produce comparable Check 2 / Check 3 numbers — but it would no longer be directly comparable to the original V-JEPA 2 verification, because the underlying frames differ at items 3 and 4.
- **The frames cannot serve as drop-in substitutes for the original.** Per the batch's purpose (re-encode the same frames the V-JEPA 2 verification analysed), the re-render does not reproduce the substrate the V-JEPA 2 verification used.

## Reviewer options

1. **Relax the threshold.** The 20 failing samples have cosine `≥ 0.999188`, well above 0.99 (a more typical "near-identical" floor) and arguably within the noise band one would accept for AI2-THOR + transformer-encoder reproducibility. The reviewer issues a one-time threshold adjustment with rationale (per spec §5.5 recalibration discipline applied to this protocol). Frames declared usable.
2. **Investigate AI2-THOR non-determinism.** Try forcing single-threaded rendering, fixing the RNG state explicitly, warming up LivingRoom before recording, or pinning a specific shader-compilation order. Re-run the determinism check after the change. Cost: 1–3 hr per attempt + diagnostic effort.
3. **Run DINOv2 verification on the rerender as-is.** Document the determinism caveat upfront. The DINOv2 verdict would be *internally* derived from a self-consistent stream and would be informative about DINOv2's geometry; cross-encoder comparison to the V-JEPA 2 verification would carry the documented caveat.
4. **Treat the original V-JEPA 2 result as final** and skip alternative-encoder verification on this bank. Move directly to a different §5.5 path (SIGReg, reframing the recurring unit, or fresh data with deliberate per-loop perturbation that addresses Check 1 degeneracy and rendering non-determinism in one batch).

## State of the working tree

- Modified script `scripts/run_furniture_main.py` in the previous repo committed at `98578d3` (re-render variant adds opt-in flags only). Not pushed.
- Re-render output: `data/seed7_furniture_frames/` (100 000 PNGs, ~5.2 GB) and `data/seed7_furniture_rerender_aux/` (~411 MB). Both gitignored via `data/`.
- Determinism check: `scripts/check_rerender_determinism.py`, `results/frame_rerender/determinism_check.json` (this report's source data).
- Re-render wall-clock: 11 219 s (~3.1 hr); 218 loops; identical to the original run's loop count.

## Required next steps before resumption

Reviewer chooses one of the four options above and issues a follow-up batch. No autonomous progression beyond this stop.

---

*Following the batch instructions §6: HANDOFF will be updated with this stop, this report committed in the new repo, and no further work performed.*
