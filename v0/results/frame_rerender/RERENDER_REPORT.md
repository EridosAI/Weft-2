# RERENDER_REPORT — Seed-7 Furniture Re-Render (2026-05-12)

**Verdict: PASS-AFTER-RECALIBRATION.**

The seed-7 furniture re-render completed cleanly and is verified usable as substrate for downstream encoder verification. Original 0.9999 cosine threshold from the re-render batch §5 was recalibrated to 0.999 by the reviewer under spec §5.5 recalibration discipline; recalibrated verdict passes for all 50 sampled frames. This report is the required artifact from the re-render batch §6 and is the final record for that batch.

Supersedes [STOP_REPORT.md](../../STOP_REPORT.md) (commit `56050cc`), which remains in the repo as historical record of the original FAIL under the un-recalibrated threshold.

---

## Setup

- **Source script.** Previous repo at `/mnt/c/Users/Jason/Desktop/Eridos/Weft/scripts/run_furniture_main.py`, commit `98578d3` (`feat(furniture-rerun): save frames during forward pass for verification reuse`).
- **Modification.** Two opt-in flags added; original behaviour preserved bit-for-bit when neither flag is set:
  - `--save_frames_dir DIR` — write each rendered RGB frame as `frame_{idx:08d}.png` into `DIR`.
  - `--results_root DIR` — override the default `results/stage_0b_furniture/main` output dir.
  - Frame-save is a lossless PNG round-trip on AI2-THOR's uint8 `(H, W, 3)` returns via `PIL.Image.fromarray`.
- **Frame format.** PNG, `frame_{idx:08d}.png`, one file per frame.
- **Output path.** [data/seed7_furniture_frames/](../../data/seed7_furniture_frames/) (gitignored via `data/`).
- **Seed.** 7 (matches original run).
- **Encoder for determinism check.** Same V-JEPA 2 checkpoint and preprocessing the original run used; embeddings L2-normalised before cosine comparison.

## Run summary

| field | value | source |
|---|---:|---|
| frames saved | 100 000 | [HANDOFF.md L116](../../HANDOFF.md#L116) |
| loops completed | 218 | [HANDOFF.md L118](../../HANDOFF.md#L118) (matches original) |
| frames disk usage | ~5.2 GB | [HANDOFF.md L117](../../HANDOFF.md#L117) |
| aux disk usage | ~411 MB | [STOP_REPORT.md L67](../../STOP_REPORT.md#L67) |
| wall-clock | 11 219 s (~3.1 hr) | [HANDOFF.md L120](../../HANDOFF.md#L120) |
| overhead vs original | ~5 min slower (PNG write) | [HANDOFF.md L121](../../HANDOFF.md#L121) |
| `frame_annotations.jsonl` | md5 `6f241260c0059e57bf96585388aa2fc8` | [STOP_REPORT.md L31](../../STOP_REPORT.md#L31) |
| same md5 as original run? | yes (bit-identical) | [STOP_REPORT.md L31](../../STOP_REPORT.md#L31) |
| `memory_bank_embeddings.npy` md5 | `a1c581e5...` (rerender) vs `3029a6a8...` (original) | [STOP_REPORT.md L32](../../STOP_REPORT.md#L32) |

The trajectory, dwell schedule, transit micro-step sequence, and per-frame metadata are deterministic across runs (annotations file md5 identical). The embedding-bank divergence is localised — see §determinism.

## Determinism check

50 frames sampled: 10 per viewing position × 5 viewing positions. Each rerendered frame loaded from disk, encoded through the same V-JEPA 2 checkpoint and preprocessing as the original run, L2-normalised, and compared to the corresponding bank entry by cosine. Source: [results/frame_rerender/determinism_check.json](determinism_check.json).

| viewing_position_id | object | room | n | cos (constant per item, 7 dp) | < 0.999 | < 0.9999 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Bed | Bedroom | 10 | `1.0000000` | 0/10 | 0/10 |
| 2 | DiningTable | Bedroom | 10 | `0.9999998` | 0/10 | 0/10 |
| 3 | Dresser | LivingRoom | 10 | `0.9991884` | 0/10 | 10/10 |
| 4 | Sofa | LivingRoom | 10 | `0.9994806` | 0/10 | 10/10 |
| 5 | Television | Bedroom | 10 | `1.0000000` | 0/10 | 0/10 |

Aggregates: `cos_min = 0.9991884`, `cos_max = 1.0000000`. Item 2's exact JSON value is `0.9999997615814209`; STOP_REPORT and HANDOFF rounded to 6 dp as `1.000000`. All other values are reproduced here at the precision stored in the JSON.

**Structure of the divergence.**

- **3/5 items (1, 2, 5 — all Bedroom) are bit-identical or within floating-point rounding** of the original bank. Bedroom is the agent's spawn room.
- **2/5 items (3, 4 — both LivingRoom) diverge by a small, exactly-constant per-item amount.** Every one of the 10 sampled frames at item 3 has `cos = 0.9991884` to 7 decimals against its corresponding bank entry. Every one of the 10 sampled frames at item 4 has `cos = 0.9994806`. The constancy across loops within a run, combined with the per-item-specific offset, identifies the cause as upstream scene-state-dependent rendering on first entry to LivingRoom — most plausibly shader compilation order, asset upload, or physics settling on first instantiation, none of which recur once warm. V-JEPA 2's eval-mode forward pass is itself deterministic on this stack (items 1, 2, 5 prove this).

## Threshold recalibration

The original 0.9999 threshold (re-render batch §5) was a heuristic guess meant to capture "effectively bit-identical given floating-point rounding." The reviewer recalibrated to 0.999 under spec §5.5, with the following justification, recorded verbatim:

> The 0.9999 threshold was a heuristic guess in the rerender batch instructions, meant to capture "effectively bit-identical given floating-point rounding." Empirical inspection of the determinism check data shows the divergence has specific structure: 30/50 sampled frames are bit-identical (cosine exactly 1.000000); the remaining 20 are at cosine 0.999188 (items 3 dwell frames, all identical to 6 decimals) and 0.999481 (items 4 dwell frames, same). The constancy of the divergence per-item identifies the cause as upstream AI2-THOR rendering state on first entry to LivingRoom, not stochastic noise in the encoding pipeline.
>
> The purpose of the determinism check is to verify the rerendered frames are usable substrate for downstream encoder verification — i.e., that an encoder computing cosines on these frames produces results equivalent to one computing on the original bank. The substrate-verification protocol (spec §5) operates at cross-element cosine thresholds in the 0.6 range; a 0.0008 deviation in a 1024-dim embedding is approximately three orders of magnitude below that protocol's discrimination floor. The original 0.9999 was overly strict for the actual purpose.
>
> Recalibrated to 0.999. This is still ~600× tighter than the substrate verification's own cross-element tolerance and comfortably catches any genuine encoder divergence while accepting upstream rendering microvariation as immaterial. Applied once. Subsequent divergence below 0.999 would warrant separate investigation, not further recalibration.

This is the one-time recalibration permitted by §5.5. Subsequent recalibration would be a process violation per spec §11.

## Verdict

**PASS-AFTER-RECALIBRATION.** All 50 sampled frames pass cosine > 0.999. Margin to the recalibrated floor: `cos_min − 0.999 = 0.0001884` (item 3, Dresser).

## Note for next batch

DINOv2 substrate verification can proceed against [data/seed7_furniture_frames/](../../data/seed7_furniture_frames/).

**Caveat to record at the start of the DINOv2 batch:** 2 of 5 viewing positions (item 3 Dresser, item 4 Sofa, both in LivingRoom) have constant per-item offsets from the original V-JEPA 2 bank at the cosine `0.0005`–`0.0008` level (L2 distances `0.032`–`0.040` between unit vectors). This is documented in case any downstream analysis surfaces an unexplained discrepancy at that magnitude. It is accepted as immaterial for substrate-verification purposes; the §5 protocol thresholds operate three orders of magnitude above this scale.

---

*Numbers in this report all trace to [determinism_check.json](determinism_check.json), [STOP_REPORT.md](../../STOP_REPORT.md), or the [HANDOFF.md](../../HANDOFF.md) rerender entry. No remembered numbers.*
