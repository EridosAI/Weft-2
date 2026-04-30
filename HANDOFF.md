# HANDOFF — Weft 2

**Project:** Weft Inner PAM (continuous-trajectory associative memory, post-architectural-rethink)
**Repo:** `/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/`
**Status:** Fresh repo, bootstrapped. No code yet. Awaiting encoder substrate verification.

---

## What this repo is

This is a fresh repository for the Weft project, built around the architecture articulated in `WEFT_INNER_PAM_v0_Spec.md`. The previous repo at `/mnt/c/Users/Jason/Desktop/Eridos/Weft/` contains four iterations of negative results that established the previous architecture (next-frame prediction with cosine retrieval) was building the wrong thing. The new architecture is path-prediction with Gaussian negative-log-likelihood loss, learning trajectory shapes through repetition. See the spec for full claims.

The previous repo stays in place as historical record. This repo does not edit it or share state with it.

---

## What's been done

- Repo bootstrapped per `instructions/` setup batch.
- `CODING_STANDARDS.md`, `research_operations_v1.md`, `WEFT_INNER_PAM_v0_Spec.md` carried forward.
- Encoder substrate verification (per `instructions/ENCODER_SUBSTRATE_VERIFICATION.md`) **complete — verdict FAIL.**

---

## Encoder substrate verification — verdict FAIL (2026-04-30)

Read-only protocol from `WEFT_INNER_PAM_v0_Spec.md` §5 against the
seed-7 furniture-run bank in the previous repo. Headline numbers from
`results/encoder_verification/verification_data.json`; full breakdown
in `results/encoder_verification/ENCODER_VERIFICATION_REPORT.md`.

| check | aggregate | starting threshold | result |
|---|---:|---|---|
| 1. cross-instance stability (mean cosine, n = 250 pairs across 5 items) | `1.0000` | `> 0.75` | PASS (degenerate — see report §7) |
| 2. cross-element distinguishability (mean cosine, n = 1000 pairs across 20 ordered pairs) | `0.8697` | `< 0.60` | **FAIL** (load-bearing) |
| 3. combined gap (Check 1 − Check 2) | `0.1303` | `≥ 0.15` | FAIL |

**Verdict: FAIL.** Encoder does not meet the protocol on this bank.

**Why FAIL is the right call (load-bearing finding):** Check 2 is the
real failure — V-JEPA 2 mean-pool produces cross-element cosines
ranging `0.8347` (Bed ↔ Dresser) to `0.9210` (DiningTable ↔ Sofa) for
the 5 furniture items in seed 7's house. All 10 distinct cross-pair
values are far above the 0.60 starting threshold. This is consistent
with the prior Stage 0b room-distinctness diagnostics: V-JEPA 2 mean-
pool's geometry is dominated by scene context, not the recurring
unit. Check 2's failure does *not* depend on Check 1.

**Caveat — Check 1 is degenerate, not informative.** The seed-7
furniture-run dwell mechanism teleports the agent to the *exact same
pose* every dwell frame, every loop, so AI2-THOR renders bit-identical
pixels and V-JEPA 2 (deterministic, frozen) produces bit-identical
embeddings. The within-instance cosine of `1.0000` with std `0.0000`
across all 50 sampled pairs at all 5 items reflects this — it is
measuring rendering determinism, not encoder stability under natural
instance variation. Spec §5.1 was written assuming instances would
carry natural variation (different angles, lighting, etc.); this bank
does not provide that. Same artifact appears in §3's per-pair std =
`0.0000` for every ordered pair: with bit-identical embeddings within
an item, sampling 50 pairs reduces to one cosine repeated 50 times.

The verdict therefore stands on Check 2 alone. Check 3's gap of
`0.1303` corroborates rather than adds independent signal: it is the
1.0 (degenerate) minus the 0.87 (real). A non-degenerate Check 1 (on
varied instances) would lower its mean and shrink the gap further.

**Per spec §5.5,** v0 implementation does not proceed on this encoder
without substrate work. The decision (alternative frozen encoder,
fine-tuning, redefining the recurring unit) is human review, not
autonomous.

---

## Next immediate action

Human review of `results/encoder_verification/ENCODER_VERIFICATION_REPORT.md`. The verdict is FAIL on the seed-7 bank; the §5.5 options (alternative encoder, fine-tuning, reframing the recurring unit) are decided in review. No autonomous progression.

If the reviewer concludes the seed-7 bank's lack of instance variation is the binding limitation rather than V-JEPA 2 itself, the natural follow-up is a re-run of the protocol on a bank where instances carry natural variation (e.g., the perturbed-loop frames at `/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/perturbed/`, or new data with deliberate per-loop perturbation). That is a *new* batch under §5 of the spec, not a recalibration of this one.

---

## Operational state

- Working tree: clean.
- Push hold: in effect.
- No running jobs.

---

## DINOv2 substrate verification batch — STOP (2026-04-30)

The DINOv2 substrate verification batch was issued to re-run the §5
protocol against DINOv2 ViT-L/14 on the seed-7 furniture-run frames,
for direct comparison to the V-JEPA 2 result.

**Stop trigger.** Per the batch §2.1 and §6, the seed-7 furniture-run
**source RGB frames are not retained** in the previous repo. Only
encoded V-JEPA 2 embeddings, per-frame annotations, and metadata are
present. The original training script encoded each frame in the
forward pass and discarded the pixels. DINOv2 cannot be evaluated
without re-encoding, and re-encoding requires source pixels.

**Re-rendering is feasible but not authorised by this batch.** The
furniture-run env is deterministic (fixed house seed, fixed Teleport
poses, deterministic AI2-THOR rendering pipeline) — re-running the
route would produce bit-identical pixels for every dwell frame.
Estimated cost: ~2.5 hr cuda + AI2-THOR time + ~1–20 GB disk
(format-dependent). The batch instructions chose to make the
absence-of-frames a hard stop rather than authorise the re-render.

**Full evidence + reviewer options** in `STOP_REPORT.md` at the
project root.

**Resumption requires one of the four reviewer options in the STOP
report:** authorise re-rendering at full scope, authorise a smaller
deliberate probe set (which would also address the V-JEPA 2
verification's Check 1 degeneracy), defer DINOv2 verification, or
provide the source frames if they exist outside the project tree.

*STOP commit: pending.*
