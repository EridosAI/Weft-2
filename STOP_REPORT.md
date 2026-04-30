# STOP_REPORT — DINOv2 Substrate Verification batch (2026-04-30)

## Triggering condition

Per the batch instructions §2.1 and §6:

> If the source frames were not retained (only embeddings stored), this is a hard stop — DINOv2 cannot be tested without re-encoding.

> Stop and report if: Source RGB frames cannot be located (only embeddings retained).

The seed-7 furniture-run source RGB frames are not retained in the previous repo. Only encoded embeddings, per-frame annotations, and metadata are present.

## Evidence

`ls /mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/main/`:

```
frame_annotations.jsonl              (per-frame metadata)
launch_info.txt
memory_bank_embeddings.npy           (100 000 × 1024 V-JEPA 2 embeddings)
memory_bank_episode_boundary.npy     (boundary flags)
progress_log.txt
training_log.json
```

No `frames/`, `*.png`, or `*.jpg` artifacts. `find /mnt/c/Users/Jason/Desktop/Eridos/Weft/ -name "*.png" -o -name "*.jpg"` matches only unrelated `teleport_smoke_frames/` from a Stage 0b traversal-mechanism smoke test in early Stage 0b — not the furniture run.

Confirmed by reading `scripts/run_furniture_main.py` in the previous repo (commit `a2cf6bb`): the per-frame loop calls `env.next_frame()` → `frame_to_encoder_tensor()` → `encoder.encode_frame()` → `memory_bank.append(emb)`. The intermediate RGB tensor is not persisted. The same pattern holds for `extract_perturbed_probes.py` (perturbed/ directory contains embeddings only).

## Analysis

**Re-rendering is feasible but is not what this batch asks for.** The seed-7 furniture-run env is deterministic:

  - Same ProcTHOR-10K house (seed 7).
  - Same five viewing positions and headings (`results/stage_0b_furniture/route.json`).
  - Same dwell mechanism (Teleport to fixed pose for 30 frames per item per loop).
  - AI2-THOR rendering pipeline is deterministic for fixed input.

Re-running `scripts/run_furniture_main.py` (or a stripped-down save-frames-only variant) would produce **bit-identical pixels for every dwell frame** — confirmed by the V-JEPA 2 verification's Check 1 result of `1.0000` cosine with std `0.0000` across all instance-pairs. The 100 000-frame run took 9 456 s (~2.6 hr) of cuda + AI2-THOR time on the same hardware. Disk cost: at 256×256 uint8 RGB JPEG, ~10–30 KB/frame ⇒ ~1–3 GB; raw uint8 ndarray ⇒ ~20 GB.

The batch instructions §2.1 are explicit that the absence of source frames is a hard stop, not a directive to re-render. §1 (scope lock) lists "Retraining anything" as out of scope but does not enumerate "regenerating source data". The interpretation that re-rendering counts as a hard-stop trigger (rather than a routine inconvenience to work around) is supported by:

  - The batch was authored knowing the V-JEPA 2 verification preceded it; whoever wrote it had the option to budget re-rendering and chose to make it a stop instead.
  - A 2.5 hour AI2-THOR re-run is non-trivial overhead the batch did not authorise.
  - The reviewer may want to consider alternative paths (different recurring-unit framing, an off-the-shelf encoder verified on a smaller deliberately-rendered probe set, etc.) before committing to a 2.5 hr render.

## What is unblocked vs blocked

- **Computable now without re-rendering:** nothing in this batch — DINOv2 needs source frames.
- **Computable with reviewer authorisation to re-render the seed-7 furniture run:** the full DINOv2 verification protocol against the same probes the V-JEPA 2 verification used. Cost ~2.5–3 hr of cuda + ~1–20 GB of disk.
- **Computable on a smaller deliberate probe set:** if the reviewer authorises rendering only the 5 viewing positions × 1 frame each (5 frames, < 1 minute) the cross-element distinguishability check could be run with very small sample size — but this would deviate from the protocol's specified n ≥ 50 cross-element pairs and is not a substitute.

## Reviewer options

1. **Authorise re-rendering of the seed-7 furniture run** with frames saved to disk. Specify the budget (full 100 k frames at ~2.5 hr, or just the dwell frames at ~1.5 hr / ~30 k frames). DINOv2 verification then proceeds against the regenerated source.
2. **Authorise a shrunk-protocol DINOv2 quick-test:** render a small deliberate probe set (e.g., 50 frames per viewing position with deliberate small per-loop perturbations to introduce real instance variation). Document the deviation from the protocol and run the checks at smaller scale. This trades protocol fidelity for time but addresses the V-JEPA 2 verification's Check 1 degeneracy as a side benefit.
3. **Defer DINOv2 verification.** The V-JEPA 2 result (cross-element 0.8697) already establishes the substrate problem. Move to an alternative spec §5.5 path (SIGReg-trained encoder, reframing the recurring unit) without first ruling out DINOv2.
4. **Provide the source frames** if they exist somewhere I missed (e.g., a backup the user retained outside the project tree).

## State of the working tree

- Last committed state: `552165b` (V-JEPA 2 verification commit).
- Working tree clean modulo the addition of this STOP_REPORT and the HANDOFF update.
- No re-rendering attempted, no encoder loaded, no DINOv2 wheels downloaded.

## Required next steps before resumption

If option 1 or 2 is chosen, the human reviewer issues a follow-up batch authorising the re-render with the chosen scope. If option 3 or 4, the follow-up batch sets a different direction.

---

*Following the batch instructions §6: HANDOFF will be updated with this stop, this report committed, and no further work performed.*
