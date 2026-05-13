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

**STOP for experiment-chat review.** Session 4 implemented the continuous-motion substrate per the session-3 reviewer directive, ran a 5-loop calibration, and ran DINOv2 motion-continuity diagnostics. Two findings to review before the full Phase 2 collection begins:

1. **Within-loop motion-continuity PASSES.** All 255 consecutive close_up→close_up pairs and all 1,275 consecutive transit→transit pairs are non-bit-identical (cos < 0.9999 throughout). Mean consecutive cosine across the full 1,579-pair stream is 0.92. The 30-frame static dwell pattern is eliminated.

2. **Cross-loop apex comparison FAILS for 4 of 5 items** at the bit-identical level. Apex frames at items 1 (Bed), 2 (DiningTable), 3 (Dresser), and 4 (Sofa) have **cosine = 1.0000** across all 10 pairs (5 loops × choose 2). The reason is structural: the apex pose for each item is the same `viewing_position + heading=viewing_heading`, repeated each loop; AI2-THOR's frame rendering is deterministic on this stack. Item 5 (Television) is the lone exception (mean cosine 0.97, std 0.015, no bit-identical pairs) — likely TV display dynamics rendered by AI2-THOR at some non-deterministic level. The continuous-motion substrate fixes within-loop targets but does not, by itself, break across-loop bit-identicity.

This means the §2.2 "repetition tightens within-cluster representations" claim is *still* untestable on this substrate unless across-loop variation is introduced. **Proposed variation strategy (load-bearing decision the reviewer needs to sign off on before the full Phase 2 collection):** re-introduce per-frame pose jitter at a **smaller magnitude than the prior Stage 0b stability batch** (0.05 m position, 2° heading vs the prior 0.20 m / 10°). The motion now supplies the bulk of variation; jitter exists only to break the across-loop pose-determinism floor. Reviewer alternatives below.

Until reviewer signs off on the variation strategy, the trajectory implementation and calibration are committed but the full Phase 2 collection does not start. After sign-off, run a second calibration (5 loops) with jitter to verify cross-loop apex variation is now non-degenerate (target: mean cosine 0.92-0.98 across loops at the apex), then launch the full 65k-frame Phase 2 collection.

---

### (Historical) Earlier session-3 STOP — superseded by session 4 substrate change

The session-3 STOP listed three open decisions on G1.3 / G1.4 / S4. These are largely **superseded** by the session-4 substrate change:

- The Phase 1 substrate is now declared **substrate-degenerate** (per reviewer directive); its specific G1.3 / G1.4 / S4 results are not inherited as findings about the architecture and the v0 evidence base restarts at Phase 2 on the new substrate.
- The verified-working items from Phase 1 — predictor scaffolding works (no NaN/Inf, loss decreased monotonically, 21.5M params well within tolerance, gradient flow healthy), encoder pipeline works (DINOv2 deterministic, 100k frames all L2-normed), and cross-cluster discriminability rises with training (M3 trajectory 0.008 → 0.325) — remain valid.
- Phase 1 is not being re-run with the new substrate. The v0 evidence base starts at Phase 2.

For audit, the session-3 G1.3 / G1.4 / S4 disaggregations stay in `results/inner_pam_v0/phase1_main/gate_report.json` and the session-3 HANDOFF entry below documents them. They are not subject to a pending verdict any longer.

---

### Open decisions / proposals for the reviewer (session 4)

1. **Variation strategy.** Three candidate forms; recommend (a):
   - **(a) Per-frame independent jitter at 0.05 m / 2°** drawn fresh each frame. Simplest. Breaks across-loop bit-identicity directly: apex poses across loops would differ by ~0.05 m position and ~2° heading. ContinuousMotionExplorer needs a jitter parameter added; ~30 lines of code.
   - **(b) Per-loop pose offset.** One small `(dx, dz, dh)` offset drawn per loop, applied as a constant shift to all frames in that loop. More principled (whole loop traverses a slightly different path) but requires per-loop state in the explorer.
   - **(c) Hybrid (per-loop base + per-frame micro-noise).** Both signals.
2. **Acceptable jitter magnitude.** 0.05 m / 2° is a starting proposal. The prior stability batch (Stage 0b) used 0.20 m / 10°, but that was the only variation source. Now motion provides most of the variation; jitter only breaks the floor. The reviewer may want it even smaller (0.02 m / 1°) or larger (0.10 m / 5°).
3. **Television-item anomaly.** Item 5 shows non-zero across-loop variation (mean 0.97, std 0.015) without any explicit pose jitter — likely the AI2-THOR Television's rendered display has internal variation. Reviewer may want this investigated separately, or accept it as an item-specific quirk and treat it as another known property of the substrate.
4. **`transit → close_up` boundary frame duplication.** 8 of 24 transit→close_up consecutive pairs have cosine > 0.9999. This is because the final corner-rotation step in transit lands at exactly close_up_start with heading exactly viewing_heading, which is also the first close_up frame's pose — a 1-frame duplication at each phase boundary. Cosmetic; affects 0.5% of consecutive pairs. Fix candidates: drop the last transit-rotation step, or start the close-up one densification step further along. Reviewer call on whether to fix in session 4 or accept for now.
5. **Phase 2 frame budget.** 65k stays valid for the 316-frame loop: 65k/316 ≈ 205 collected loops, 195 trained loops (minus 10 held-out), comfortably into the 100+ rep bin with 95 reps inside. **Item 1 in the held-out region.** The last loop may end partial (depending on where 65k lands), giving (1,2) ~10 cue probes and the other four transitions ~9 each (same partial-loop pattern as Phase 1, plus or minus 1 depending on truncation point); resolves under the same not-a-bug logic as session 2.
6. **Checkpoint cadence for §4.6 recomputed against the 316-frame loop length** (table in the session-4 outcomes entry below). All five rep bins covered, 100+ bin gets 3 checkpoints inside.

Reviewer-action gate: when the variation strategy and magnitude are decided, I'll implement, re-run the 5-loop calibration with jitter, verify cross-loop apex variation is non-degenerate, then launch the full 65k Phase 2 collection.

1. **G1.4 verdict at the gated horizon (k=8).** Main wins at k=1 (mean_diff +0.039, p=1.7e-05) and at k=16 by rank (Wilcoxon p=0.004 even though mean_diff is −0.020). **Main loses at k=8** (mean_diff −0.132, Wilcoxon p=1.0). The mid-horizon failure is real, not an artefact of v1's wrong-shuffle. Diagnosis below points to **rank-512 limited predictor architecture** as a candidate: cosine at the cluster boundary is bounded by the output projection's column space, while squared-error (which the loss actually optimises) shows main beating shuffle decisively. Reviewer call: declare FAIL @ k=8 and pause to investigate, accept the rank-limited reading and recalibrate the gate to squared-error or a different cosine threshold, or proceed with a documented caveat.

2. **G1.3 verdict.** FAIL against the +0.3 absolute scaffolding threshold (separation cue−steady = −0.14, i.e., the wrong direction). But **main has structure shuffle doesn't**: shuffle's separation is 0.0002 (≈ zero, as expected from a temporally-destroyed control), while main's is −0.14 in a stable consistent direction. Main is also ~0.8 log_var more confident than shuffle on both probe types. Substrate-artefact diagnostic is below. Reviewer call: FAIL @ absolute scaffolding stands; PASS-at-relative-baseline depending on how the reviewer weighs "wrong-direction separation but real structure" vs "absolute spec direction".

3. **S4 quantitative thresholds need reviewer-authorised recalibration.** Shuffle did NOT collapse to the form S4 anticipated (`||μ|| < 0.15` AND `log σ² > 0.4`) but DID collapse to a different specific form: `||μ|| ≈ 0.75` AND `log σ² ≈ −7.48` — i.e., predicting the marginal-mean direction with low (calibrated-to-residual) variance. The empirical shuffle distribution is now visible; per instr §6.5 the thresholds are SCAFFOLDING explicitly subject to "recalibrate after observing the empirical shuffle distribution at end of Phase 1." Recommended new thresholds: `||μ|| > 0.6` OR `M3 sharpness < 0.05` (either captures the collapse-to-marginal-mean signature observed). Reviewer chooses the recalibration.

**Mechanical gates remain passed.** G1.1 PASS, G1.2 PASS. **G1.5 trajectory PASS @ scaffolding** (unchanged from session 2; main predictor's M3 sharpness 0.008 → 0.325, 8 of 9 transitions non-decreasing, floor 0.10 cleared).

**Phase 2 entry remains blocked** until the reviewer assigns G1.3 / G1.4 / S4 verdicts.

---

## Operational state (end of session 4)

- Working tree: clean modulo this HANDOFF entry + the session-4 commits (continuous-motion explorer, env wrapper, calibration script, analysis script, calibration data + report, doc updates). 25 commits expected on `main` after this session lands.
- Push hold: in effect.
- No running jobs.
- Phase 1 artefacts: unchanged from session 3 (substrate-degenerate baseline; not being re-run).
- Phase 2 substrate: new continuous-motion explorer + env wrapper at `src/env/continuous_motion_*.py`. 5-loop calibration data at `data/phase2_calibration/` (frames gitignored; annotations + embeddings gitignored per .gitignore rules; report committed). Full Phase 2 collection has NOT begun and will not begin until reviewer signs off on the variation strategy.

---

## Session 4 outcomes — 2026-05-13

**Goal.** Implement the continuous-motion substrate per the session-3 reviewer directive ("the agent moves through each loop as one continuous trajectory with no zero-velocity frames"), run a 5-loop calibration, verify motion-continuity with DINOv2 diagnostics, propose a variation strategy informed by the empirical findings, update the spec and instructions to lock the new substrate in, and STOP for review before any full Phase 2 collection.

### Trajectory design — `ContinuousMotionExplorer`

New explorer at `src/env/continuous_motion_explorer.py` replaces `FurnitureRouteExplorer`. State machine has two phases per item: `close_up` (continuous motion through the item) and `transit` (continuous motion between items).

**Close-up segment** (per item N):
- Direction: perpendicular to `viewing_heading_N` (CCW from forward in the top-down screen sense, consistent across all 5 items so the item visually slides in a consistent direction).
- Endpoints: `viewing_position_N ± (close_up_length_m / 2) * perpendicular_unit_ccw`. With `close_up_length_m = 2.0 m` (SCAFFOLDING), endpoints are at ±1.0 m from the viewing position.
- Heading: locked at `viewing_heading_N` throughout the close-up (so the item enters from one side of the frame, centres at the apex, slides out the other side).
- Densification: 0.20 m steps (SCAFFOLDING), yielding ~10-12 frames per close-up.
- Apex frame: the densified step closest to `viewing_position_N` is tagged `close_up_apex_flag = True`.

**Transit segment** (item N → N+1):
- NavMesh-planned path from `close_up_end_N` to `close_up_start_{N+1}` (different from the old explorer's viewing_position → viewing_position transit).
- Densified at 0.20 m steps + corner rotations at 5° step. Same mechanism as the prior FurnitureRouteExplorer.
- Heading: along walking direction within each NavMesh segment; rotates at corners; final rotation aligns to next item's viewing_heading.

**No static dwell at any pose.** Every consecutive frame pair has a non-zero pose delta.

### 5-loop calibration

`scripts/run_phase2_calibration_collect.py` ran 5 loops with the new explorer; no perturbation, no jitter.

| metric | value |
|---|---:|
| frames collected | **1,580** |
| loops | **5** |
| **frames per loop** | **316** |
| wall-clock | 89.5 s (~17.7 fps) |
| close-up frames per item per loop | 11, 12, 11, 11, 11 (avg ~11) |
| transit frames per loop | 260 |
| transitions planned (5 loops × 5 transitions) | 25 |
| transitions using NavMesh fallback | 10 of 25 (40 %) |
| teleport failures | 0 |

**316 frames/loop** is higher than the 200-250 target the reviewer flagged as a starting point, but the rep-bin coverage arithmetic (below) still works at the 65k Phase 2 budget with comfortable margin. Tunable in the v0 SCAFFOLDING inventory; the close-up length (2 m) or density (0.20 m) can be reduced if the reviewer wants the loop shorter.

### DINOv2 motion-continuity diagnostic

`scripts/run_phase2_calibration_analyse.py` encoded all 1,580 frames via the verified DINOv2 protocol and computed consecutive-frame cosines, disaggregated by phase, plus same-item cross-loop apex comparisons.

**Embedding sanity:** all 1,580 frames have L2 norms in [1−1e-5, 1+1e-5]. ✓

**Consecutive-frame cosines (1,579 pairs):**

| phase pair | n | mean | std | min | max | bit-identical (>0.9999) |
|---|---:|---:|---:|---:|---:|---:|
| close_up → close_up | 255 | **0.9304** | 0.126 | 0.315 | 0.991 | **0** ✓ |
| transit → transit | 1,275 | **0.9202** | 0.106 | 0.107 | 0.992 | **0** ✓ |
| close_up → transit | 25 | 0.8034 | 0.151 | 0.616 | 0.985 | 0 |
| transit → close_up | 24 | 0.9232 | 0.075 | 0.802 | 1.000 | **8** ⚠ |
| **aggregate** | 1,579 | 0.9201 | 0.111 | 0.107 | 1.000 | **8** |

The 8 bit-identical pairs are all `transit → close_up`, at the boundary between transit and close-up: the final corner-rotation step of transit lands at exactly `close_up_start` with heading exactly `viewing_heading`, which is also the first close-up frame's pose. **Cosmetic 1-frame duplication at the boundary; 0.5 % of all pairs.** Open decision (4) above proposes a fix.

**Within-loop continuity verdict: PASS.** Zero bit-identical pairs in close_up→close_up or transit→transit (the two "during motion" categories). The 30-frame static-dwell pattern is eliminated.

**Cross-loop apex comparison (5 apex frames per item × 10 pairs = 10 per item):**

| item | object | n pairs | mean cosine | std | bit-identical (>0.9999) |
|---:|---|---:|---:|---:|---:|
| 1 | Bed | 10 | **1.0000** | 0.000 | **10/10** ✗ |
| 2 | DiningTable | 10 | **1.0000** | 0.000 | **10/10** ✗ |
| 3 | Dresser | 10 | **1.0000** | 0.000 | **10/10** ✗ |
| 4 | Sofa | 10 | **1.0000** | 0.000 | **10/10** ✗ |
| 5 | Television | 10 | 0.9695 | 0.016 | **0/10** ✓ |

**Cross-loop apex verdict: FAIL on items 1-4.** Apex poses across loops are bit-identical, because each loop visits the same `viewing_position + viewing_heading` and AI2-THOR renders deterministically on this stack (confirmed by the substrate-verification batch's session-1 consistency check). Item 5 is the lone exception with non-zero across-loop variation, likely because the Television's rendered display has internal dynamics that AI2-THOR doesn't make deterministic — an item-specific quirk, not a designed feature.

**Implication.** The continuous-motion substrate fixes within-loop static dwell (the original session-3 finding) but does NOT, by itself, break across-loop pose-determinism. Any M3 cluster-sharpness measurement that compares predictor outputs across loops at the same pose (the way Phase 1's M3 worked) will still be substrate-floored at cosine = 1.0 within-cluster.

This is a partial substrate fix. Full resolution requires re-introducing across-loop variation, per the variation-strategy proposal above.

### Proposed variation strategy

**Recommendation: per-frame independent jitter at 0.05 m / 2°** (option (a) above). Rationale:

- **0.05 m position jitter** is much smaller than the 0.20 m densification step, so it doesn't dominate consecutive-frame motion; it just adds enough offset to break the across-loop pose-determinism floor.
- **2° heading jitter** is small enough not to swing the item out of frame at the apex (where the item is ~1.75 m from the agent).
- **Per-frame independent draws** (each frame's jitter is fresh, seeded RNG) means every frame has a unique offset. Cross-loop apex frames at item N would differ by ~0.05 m / 2° drawn from independent distributions, producing non-bit-identical embeddings.

Implementation cost: ~30 lines in `ContinuousMotionExplorer` (analogous to the prior `_apply_jittered_teleport` in `FurnitureRouteExplorer`, but no fallback ladder needed at the smaller magnitude — NavMesh tolerance should accept ±0.05 m at most poses).

Alternative options were considered (per-loop offset, hybrid); recorded in the "Open decisions" list above. The reviewer chooses.

### Recomputed checkpoint cadence — §4.6 update for 316-frame loop

The prior §4.6 cadence was derived from a 458-frame loop. The new substrate has 316-frame loops, so the phase-relative-step → rep-count map shifts:

| bin (perturbed-shape rep count) | first frame into phase | last frame into phase |
|---|---:|---:|
| 1–5 | 316 | 1,580 |
| 6–19 | 1,896 | 6,004 |
| 20–50 | 6,320 | 15,800 |
| 51–99 | 16,116 | 31,284 |
| 100+ | 31,600 | (end at ~61,840) |

**Proposed new checkpoint schedule** (phase-relative steps): **1,000 / 2,000 / 4,000 / 6,500 / 10,000 / 15,000 / 20,000 / 30,000 / 40,000 / 55,000 / end** — 10 checkpoints plus end-of-phase. Coverage:

| step | loops elapsed | bin |
|---:|---:|---|
| 1,000 | 3.2 | 1-5 ✓ |
| 2,000 | 6.3 | 6-19 ✓ |
| 4,000 | 12.7 | 6-19 |
| 6,500 | 20.6 | 20-50 ✓ |
| 10,000 | 31.6 | 20-50 |
| 15,000 | 47.5 | 20-50 |
| 20,000 | 63.3 | 51-99 ✓ |
| 30,000 | 94.9 | 51-99 |
| 40,000 | 126.6 | 100+ ✓ |
| 55,000 | 174.1 | 100+ |
| end | 195.7 | 100+ |

All five bins covered; 100+ bin gets three checkpoints. Cadence updated in instr §4.6 in the same commit as the substrate change.

### Frame budget — §8.3 / §9.3 update

Phase 2/3 budgets stay at 65k each. At 316 frames/loop:

| budget | collected loops | trained loops (−10 held-out) | last bin reached |
|---|---:|---:|---|
| 65k | ~205.7 | ~195 | 100+ with 95 reps inside |

Comfortable margin. Updated §8.3 / §9.3 derivation tables in the same commit.

### Phase 1 disposition

Per reviewer directive: Phase 1's substrate is declared substrate-degenerate. Its findings — predictor scaffolding works, encoder pipeline works, parameter counts within tolerance, gradient flow healthy, M3 cross-cluster discriminability rises with training — are kept as substrate-pipeline-validation evidence. Its substrate-degenerate results (G1.3 inversion, G1.4 k=8 dip, S1-S4 sanity check verdicts, variance-saturation patterns) are not inherited as findings about the architecture and are not used as priors for Phase 2 interpretation. **Phase 1 is not being re-run with the new substrate.** The v0 evidence base starts at Phase 2.

### Drift-detection note added to research_operations_v1.md §15

The 30-frame static dwell survived three drafts of the v0 experiment instructions, an adversarial review pass, and session-1's CC pre-flight before the session-3 reviewer asked "what is dwell?" and the substrate-architecture mismatch surfaced. Added as a universal drift check:

> Re-read foundational substrate assumptions whenever the architectural framing shifts. Inherited collection parameters from prior projects are SCAFFOLDING by default and require explicit re-justification against the current architecture's claims.

### Working-tree contents committed this session

| commit | scope | files |
|---|---|---|
| `feat(env): continuous-motion explorer + env wrapper` | new substrate | `src/env/continuous_motion_explorer.py`, `src/env/continuous_motion_env.py`, `src/env/procthor_house.py` (copy from previous repo) |
| `feat(scripts): phase2 calibration collect + analyse` | calibration tooling | `scripts/run_phase2_calibration_collect.py`, `scripts/run_phase2_calibration_analyse.py` |
| `exp(calibration): 5-loop continuous-motion calibration + DINOv2 continuity report` | calibration data | `results/phase2_calibration/calibration_summary.json`, `results/phase2_calibration/continuity_report.json` |
| `docs(spec/instructions): continuous-motion substrate change` | doc updates | spec §1.3, §2.3; instructions §0/§1.3, §1.5, §4.6, §8.3, §9.3; research_operations_v1.md §15 |
| `docs(handoff): session 4 outcomes — STOP for review` | this entry | `HANDOFF.md` |

---

## Session 3 outcomes — 2026-05-13

**Goal.** Resolve the session-2 STOPs on the reviewer's protocol: (1) re-implement shuffle per spec §10.1's "temporal structure destroyed" rationale, (2) re-train shuffle and re-compute G1.4 + S1-S4, (3) augment G1.3 with main-vs-shuffle log_var comparison, (4) document substrate-artefact diagnostic for G1.3, M3 cross-cluster framing, and cue-probe count confirmation. Pause again for review before any Phase 2 step.

### Cue-probe count: confirmed 46 is the natural maximum, not a bug

After the session-2 cue-probe fix (commit `ea84df1`), 46 cue probes are constructed from the held-out region. The expected ceiling is 10 held-out loops × 5 transitions = 50; the actual ceiling is **(1→2): 10, (2→3): 9, (3→4): 9, (4→5): 9, (5→1): 9 = 46**.

Reason: the seed-7 stream has 100,000 frames over 219 loops (loop_idx 0..218), but loop 218 is **partial** — it completes the dwell at item 1 (Bed) and starts the transit toward item 2 (DiningTable), but doesn't finish the loop. So in the held-out region (loops 209–218), transition (1, 2) has 10 valid windows (one per loop including the partial one) but the other four transitions have only 9. Not a bug.

**Confirmation that all gate metrics used the 46-probe set:** the gate_report.json was first produced (and re-produced this session) AFTER the cue-probe fix commit `ea84df1`. The build_probes call inside the gate report uses the fixed build_cue_probes implementation. No stale 9-probe numbers were used in any reported gate verdict.

### Shuffle re-design — spec §10.1 / §6.3 / §7.5

Replaced the session-2 visit-order-only shuffle with the spec-correct implementation. Single commit `a24d92c`:

- `scripts/run_phase1_shuffle.py`: permutation seed=0 applied via `np.random.default_rng(0).permutation(N_train)` to **`embeddings[:N_train]` AND `annotations[:N_train]` in lockstep** before training begins. Held-out region `[N_train, N)` is preserved unshuffled.
- `src/trainer/online_trainer.py`: `shuffle_seed` parameter removed from `TrainerConfig`. The trainer always traverses its input stream in sequential order. Shuffle is solely a property of the input stream the trainer receives.
- `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md` §7.5: wording tightened from "applies permutation to training indices" to "applies `np.random.default_rng(0).permutation(N_train)` to the training portion of the embedding stream itself" with a paragraph documenting why the earlier wording was ambiguous.
- `.gitignore`: `results/**/ckpt_*/` (bank-state dirs are large binary; not part of audit trail).

First 5 permuted indices applied: `[62740, 36416, 33605, 36404, 55777]` — confirms embeddings at positions 0..4 in the new stream now hold the original frames 62740, 36416, 33605, 36404, 55777 (random unrelated frames). Window construction `embeddings[t-W+1 : t+1]` then yields 16 random unrelated embeddings, destroying temporal structure at source.

### Phase 1 shuffle v2 — re-train

| metric | v1 (visit-order, session 2) | v2 (spec-correct, this session) |
|---|---:|---:|
| wall-clock | 862.7 s | 871.4 s |
| gradient steps | 95,660 | 95,691 |
| final mean_loss_last_1k | **−69,540** | **−53,526** |
| final mean_log_var | **−9.48** | **−7.48** |
| final predictor weight L2 norm | 1,116.51 | 1,237.88 |

The v2 shuffle's loss is **less negative** than main's (−53,526 vs main's −62,607), and its mean_log_var is **higher** than main's (−7.48 vs main's −8.66). Both directions are the spec-correct expectations: shuffle is less optimised than main, and less confident than main. The v1 reversal (shuffle better than main on both) is now resolved.

### Gate verdicts (v2, against the new shuffle baseline)

| gate | criterion | v1 result | v2 result | verdict (vs scaffolding) |
|---|---|---|---|---|
| G1.1 | No NaN/Inf | PASS | PASS (unchanged) | **PASS** |
| G1.2 | Loss decreased | PASS | PASS (unchanged) | **PASS** |
| G1.3 | mean(log_var, steady) + 0.3 < mean(log_var, cue) | FAIL @ scaffolding (sep −0.14) | FAIL @ scaffolding (sep unchanged: −0.14); but main HAS structure (≈ −0.14) that shuffle does not (≈ 0.0002) | **FAIL @ absolute, pending reviewer call on relative** |
| G1.4 | Paired test main > shuffle at k=8 steady (p<0.01) | HELD (S1-S4 unexpected, control invalid) | **FAIL @ k=8** (mean_diff −0.13, Wilcoxon p=1.0); PASS @ k=1 (p=1.7e-5); PASS @ k=16 (Wilcoxon p=0.004 rank-based even though mean diff is −0.02) | **FAIL @ gated horizon** |
| G1.5 | M3 trajectory + floor | PASS (unchanged) | PASS (unchanged) | **PASS** |

#### G1.4 detail (v2)

Paired Wilcoxon on 250 steady-state probes (Shapiro-Wilk rejected normality at p < 1e-15 on every horizon → Wilcoxon used per the spec's fallback rule):

| horizon k | mean_diff (main − shuffle) | shapiro p | Wilcoxon p (one-sided greater) | pass at p < 0.01? |
|---:|---:|---:|---:|---|
| 1 | **+0.0395** | 1.4e-15 | **1.74e-05** | **PASS** |
| 8 (gated) | **−0.1318** | 9.4e-15 | 1.0 | **FAIL** |
| 16 | −0.0200 | 1.2e-19 | 0.0041 | PASS (rank-based) |

Main beats shuffle at the near horizon (k=1), loses at the mid horizon (k=8 gated), and wins at the far horizon (k=16) only by the rank-based test (mean diff is negative but the rank distribution favours main).

**Diagnostic for the k=8 failure (load-bearing for the reviewer's decision):**

Main's per-step M1 (cosine) has a characteristic shape that is below shuffle's flat baseline at mid-K:

| k | main M1 aggregate | shuffle aggregate (≈) |
|---:|---:|---:|
| 1 | 0.69 | 0.65 |
| 2 | 0.62 | 0.66 |
| 3-4 | **0.55** | 0.66 |
| 5-7 | 0.52 | 0.66 |
| 8 | **0.56** | 0.69 |
| 9-12 | 0.58–0.62 | 0.66 |
| 13-16 | 0.62–0.63 | 0.65 |

(Main aggregate over all 296 probes from `eval_95721.json`. Shuffle aggregate ≈ 0.66 is roughly flat because shuffle's predictions are near-constant marginal-mean outputs, see S3 below.)

Two candidate explanations the reviewer should weigh:

**(A) Rank-512 limited predictor architecture.** The output projection is `Linear(512, K*(d+1))`. Each μ_k slot is a 1024×512 sub-matrix mapping the 512-d transformer state to a 1024-d mean. The column space is rank ≤ 512 in the 1024-d output space. Cosine to a 1024-d target is bounded by the alignment of that target with the rank-512 subspace. Main is being penalised on cosine for an architectural reason that the loss (which optimises **squared error scaled by variance**, not cosine) doesn't see. Main's `mean_log_var = −8.66` vs shuffle's `−7.48` means main's squared error is ~half shuffle's (e^−8.66 ≈ 1.7e-4, e^−7.48 ≈ 5.6e-4) — main is winning on the LOSS objective by a large margin. The cosine gate is testing something the loss doesn't optimise.

**(B) Substrate-induced mid-horizon failure mode.** With the un-jittered Phase 1 stream, steady-state probes have bit-identical 16-frame targets; cue probes have a smooth dwell→transit transition. Main may have learned to predict "the first target frame matches the last window frame" (good at k=1, hence the 0.69 peak) and "the eventual stable trajectory matches the average direction" (good at k=16, hence the 0.62 recovery), but the middle of the predicted path drifts. Shuffle, predicting the marginal mean of all training targets, has a flat cosine across k that happens to land above main's mid-K dip.

Under (A), the architecture is fine and the gate is mis-specified; the reviewer would either raise predictor hidden dim (a SCAFFOLDING change per §12) or change the gate metric. Under (B), the substrate's bit-identical-dwell degeneracy plus the rank-limit conspire to produce a real failure of multi-step path prediction that Phase 2/3's jittered substrate may or may not resolve. The diagnostic that separates (A) from (B) is whether main's k=8 cosine improves substantially when the substrate is jittered (Phase 2/3) or whether the same dip persists.

The G1.4 verdict-as-computed stands at **FAIL @ k=8**. I am not declaring this autonomously per the reviewer protocol; the reviewer assigns the verdict.

#### G1.3 detail (v2) — main has structure shuffle doesn't

| stat | main (final ckpt) | shuffle (final ckpt) | main − shuffle |
|---|---:|---:|---:|
| steady mean log_var | −8.26 | −7.48 | **−0.78** |
| cue mean log_var | −8.40 | −7.48 | **−0.92** |
| separation (cue − steady) | **−0.14** | **+0.0002** | — |

Main is `−0.78 / −0.92` log_var **more confident** than shuffle on steady / cue probes respectively. Shuffle's separation across probe types is ≈ 0 (`0.0002`) — exactly what a temporally-destroyed control should show (shuffle has no way to tell steady from cue, so it predicts the same marginal-mean output for both).

So main DOES learn variance structure that the shuffle doesn't:

- Both magnitudes of confidence (main more confident than shuffle on each probe type).
- And differential structure across probe types (main −0.14 vs shuffle 0).

The structure runs in the **opposite direction** from the spec's specific hypothesis (which expected cue more variance than steady). The reviewer decides whether "main learns variance structure but not the specific direction §2.2 hypothesised" counts as a partial pass against the relative baseline, or whether the wrong-direction structure indicates a real architecture/substrate failure mode.

**Substrate-artefact diagnostic note for G1.3 (per session-2 reviewer item 2).** The un-jittered Phase 1 stream produces bit-identical 16-frame target embeddings for steady-state probes (all 16 future frames at the same viewing position are identical to floating-point precision). The optimal predictor for "predict 16 copies of a constant" under Gaussian NLL is "predict the constant with arbitrarily small variance" — log_var → −∞ (clamped to −10). Empirically main lands at log_var ≈ −8.26 for steady, well above the clamp, indicating the predictor hasn't fully saturated even on this trivial sub-problem. Possible reasons: the **rank-512 architecture limit** prevents perfect prediction of arbitrary 1024-d targets (squared error nonzero → variance has to absorb it), or the predictor's loss is dominated by other (transit, cue) targets it can't predict as well, dragging steady log_var up via shared weights. Cue probes are slightly *easier* in this substrate for an unintuitive reason: the dwell→transit smooth motion gives the predictor a directional cue (the last few window frames already moving), and the K=16 transit-frame targets share spatial structure (smooth path), so the predictor's variance fits a wider but consistent region. The inversion (cue more confident than steady) is consistent with this substrate-driven story. The reviewer should consider whether to wait for the Phase 2/3 jittered substrate (which adds genuine within-cluster variance to steady-state probes) before drawing architectural conclusions from G1.3.

#### S1-S4 sanity check (v2)

| check | criterion | observed | direction |
|---|---|---|---|
| S1 | shuffle log_var > main log_var (less confident) | shuffle −7.48 > main −8.66 | **expected ✓** |
| S2 | shuffle aggregate M1 < main aggregate M1 | shuffle 0.656 > main 0.592 | unexpected |
| S3 | shuffle |sharpness| << main |sharpness| | shuffle 0.011 << main 0.325 | **expected ✓** |
| S4 | quantitative collapse-to-mean: `||μ|| < 0.15` AND `log σ² > 0.4` | shuffle `||μ|| = 0.75`, `log σ² = −7.48` | unexpected |

**Aggregate verdict (with the existing S4 thresholds): unexpected (2 of 4 individual checks fail).**

But the qualitative interpretation has shifted from session 2:

- **S3 PASS is load-bearing.** Shuffle's cluster sharpness is **0.011** vs main's **0.325** — shuffle has essentially no item-discriminability, exactly what a temporally-destroyed control should show. Within-cluster cosines are 1.0 (shuffle outputs are deterministic on inputs) and cross-cluster cosines are 0.989 (shuffle outputs are nearly identical regardless of which item the input window is from). This is the **canonical "collapse to a single output direction" pattern** — just not the specific (norm < 0.15, log_var > 0.4) form S4's SCAFFOLDING thresholds were predicting.
- **S2's "unexpected" is the cosine artefact discussed in G1.4 above.** Shuffle's per-k aggregate is flat (predicting a fixed direction across all probes); main's per-k dips at mid-K. The aggregate-over-k comparison conflates the unfair-to-main mid-K dip with the favourable k=1 and k=16 endpoints.
- **S4's failure is a SCAFFOLDING-threshold mis-specification, not a sanity-check failure.** Instr §6.5 explicitly says: *"Starting thresholds (SCAFFOLDING — recalibrate after observing the empirical shuffle distribution at end of Phase 1)"*. The empirical shuffle distribution is now visible (||μ|| ≈ 0.75, log_var ≈ −7.48); the recalibrated thresholds the reviewer can authorise are e.g. `||μ|| > 0.6` (capturing the marginal-mean-direction signature) OR `sharpness < 0.05` (re-using S3's signal). Either captures the observed form of collapse cleanly.

Sanity check report at `results/inner_pam_v0/phase1_shuffle/sanity_check.json`.

### M3 trajectory framing — what spec claim Phase 1 supports

Per the session-2 reviewer note on item 3, the M3 sharpness trajectory 0.008 → 0.325 is real learning but with a specific interpretation given the substrate:

- **Within-cluster cosine ≈ 1.000** by construction: bit-identical pixels at the same viewing position across loops → bit-identical DINOv2 embeddings → bit-identical predictor outputs → within-cluster cosine = 1.0. This is a substrate floor, not a learned property of the predictor.
- **Cross-cluster cosine** is what evolves: at the first checkpoint (10k steps), shuffle-baseline-aligned ≈ 0.992; at the final checkpoint (95,721 steps), 0.675. Sharpness = 1.0 − 0.675 = 0.325.
- The trajectory is therefore measuring **cross-cluster discriminability**: how distinguishably the predictor outputs different vectors at different items. That is real learning, supported by the Phase 1 evidence.
- The **§2.2 "repetition tightens within-cluster representations" claim** is *not* tested in Phase 1 because within-cluster cosine is floored at 1.0 by the substrate. That test moves to Phases 2/3 where jittered collection produces non-identical dwell embeddings, giving within-cluster cosine room below 1.0 to tighten as repetition accumulates.

Architectural-claim status as of Phase 1:

| claim | status after Phase 1 | resolves in |
|---|---|---|
| Predictor learns cross-cluster discriminability | **supported** | (this phase) |
| Within-cluster representations tighten with repetition | **pending** (substrate-floored) | Phase 2/3 (jittered) |
| Predictor learns multi-step trajectory (cosine at mid-K beats shuffle) | **partial / unresolved** | depends on architecture limit diagnosis |
| Predictor learns differential variance structure across probe types | **supported (in direction)**, **wrong (in sign)** vs spec §2.2 | clarifies in Phase 2/3 with jitter |
| Predictor learns the specific cue-more-variance-than-steady pattern | **failed against spec direction** | reviewer call on whether to retain expectation |

### Reviewer-action items before session 4

1. **G1.3 verdict:** FAIL @ absolute scaffolding stands. PASS-at-relative-baseline is the reviewer's call given that main has structure shuffle doesn't, even in the wrong direction. (Item 2 from the session-2 review handed back.)
2. **G1.4 verdict:** FAIL @ k=8 stands as computed. Diagnostic suggests rank-512 architecture limit OR substrate dip; pre-Phase-2 fix candidates include raising PRED_HIDDEN, switching the gate metric to squared error (which main wins decisively), or accepting cosine-at-mid-K as a known weakness with the bit-identical substrate. (Item 1 from session-2.)
3. **S4 threshold recalibration:** empirical shuffle distribution is now known (||μ|| ≈ 0.75, log_var ≈ −7.48). Reviewer authorises new thresholds; recommendation in the §G1.4 section above.
4. **Phase 2 entry depends on items 1–3.** No autonomous progression.

If the reviewer chooses to investigate rather than continue:

- The cheapest single intervention for the cosine-at-mid-K issue is raising PRED_HIDDEN from 512 to ≥ 1024 (rank-unconstrained), which would put the predictor's mu output in the full 1024-d space. Single-variable, ~30 min retrain. Worth doing once before Phase 2 if the verdict is "investigate."
- The G1.3 direction inversion is most cleanly diagnosed by Phase 2/3 evidence; running them and looking at the relative log_var pattern under jittered substrate would settle whether the substrate or the architecture drove the inversion.

---

## Session 2 outcomes — 2026-05-13

**Goal.** Launch Phase 1 main training, sequentially launch shuffle control, run per-checkpoint eval, compute G1.1–G1.5 + S1–S4 disaggregations, STOP for review. Mechanical gates auto-declared; SCAFFOLDING-threshold gates pause for review per reviewer protocol.

### Phase 1 main training

| metric | value |
|---|---|
| wall-clock | 869.9 s (~14.5 min) on RTX 4080 Super, fp16 |
| gradient steps | 95,691 (≈ 110 steps/sec, faster than session-1 smoke's 60 steps/sec) |
| n_train | 95,722 (last 10 of 218 loops held out) |
| held-out region | frames [95,722, 100,000) |
| τ (calibrated at step 10k from steps 5k–10k median) | **8.125** |
| final mean_loss_last_1k | −62,606.72 |
| first mean_loss_last_1k (ckpt 10k) | −59,541.71 |
| final mean_log_var | −8.66 (predictor confidence rising over training, as expected) |
| final predictor weight L2 norm | 1,270.53 |
| final bank size | 95,707 |
| NaN/Inf | none |

Checkpoints at steps 10k, 20k, …, 90k, 95721; predictor + optimizer + bank state at each.

### Phase 1 shuffle control training

| metric | value |
|---|---|
| wall-clock | 862.7 s (~14.4 min) |
| gradient steps | 95,660 |
| shuffle_seed | 0 |
| final mean_loss_last_1k | **−69,540.23** (more negative than main's −62,607; load-bearing finding — see below) |
| final mean_log_var | **−9.48** (predictor MORE confident than main, opposite of S1 expectation) |
| final predictor weight L2 norm | 1,116.51 |

### Cue-probe construction bug found and fixed mid-session

Initial Phase 1 main eval produced only **9 cue probes** (expected ~50). Root cause in `src/eval/probes.py:build_cue_probes`: the destination-item label was identified by scanning only the next K=16 frames after the window, but seed-7 transit segments are typically 60+ frames long, so the next dwell almost never appeared inside the window and almost every cue candidate was skipped.

Fix (commit `ea84df1`): scan forward through the stream until the next dwell frame appears, regardless of distance. The to_item field is metadata only (which transition the probe tags); the K=16 target frames are unchanged.

After fix: 46 cue probes constructed (10 held-out loops × 5 transitions = 50 max, minus 4 that fail other constraints). Probe tests still pass. Main eval re-run after the fix. Shuffle training was launched only after the fix was verified.

### Gate verdicts

The reviewer protocol: G1.1 / G1.2 are MECHANICAL (auto pass/fail); G1.3 / G1.4 / G1.5 carry SCAFFOLDING thresholds and the script does not autonomously declare pass. Verdicts below are reported "against the documented scaffolding threshold" — recalibration is the reviewer's call.

| gate | criterion | result | verdict (vs scaffolding) |
|---|---|---|---|
| G1.1 | No NaN/Inf in predictor weights + final loss finite | no non-finite parameters; loss = −62,606.72 | **PASS** (mechanical) |
| G1.2 | Final mean_loss_last_1k < first mean_loss_last_1k | −62,606.72 < −59,541.71 (decreased by 3,065.0) | **PASS** (mechanical) |
| G1.3 | mean(log_var, steady) + 0.3 < mean(log_var, cue) | steady: −8.256, cue: −8.399, sep: **−0.14** | **FAIL** at scaffolding +0.3 threshold |
| G1.4 | Paired t-test main > shuffle at k=8 steady (p<0.01) | Shapiro-Wilk p=3.3e-8 → Wilcoxon, p_value=1.0, mean_diff=−0.359 | **HELD** (S1–S4 unexpected per §6.5) |
| G1.5 | M3 trajectory rises across Phase 1, floor > 0.10 | sharpness 0.008 → 0.325, 8 of 9 transitions non-decreasing, floor cleared | **PASS** at scaffolding |

#### G1.5 trajectory detail

| step | cluster sharpness | non-dec from prev? |
|---:|---:|---|
| 10,000 | 0.0084 | — |
| 20,000 | 0.0225 | ✓ |
| 30,000 | 0.0252 | ✓ |
| 40,000 | 0.0669 | ✓ |
| 50,000 | 0.0857 | ✓ |
| 60,000 | 0.1841 | ✓ |
| 70,000 | 0.2335 | ✓ |
| 80,000 | 0.3587 | ✓ |
| 90,000 | 0.2966 | ✗ (dip) |
| 95,721 | 0.3249 | ✓ |

Criterion (≥ 7 of last 9 non-decreasing, allowing 2 dips): **8 / 9 satisfied**. Floor 0.10 cleared comfortably at 0.325. Trajectory and floor both pass.

**Determinism caveat that affects M3 interpretation.** Phase 1 trains on the un-jittered rerender stream (per session-1 HANDOFF). Dwell frames at the same viewing position are bit-identical across loops (confirmed by the substrate verification Check 1 degenerate result and by the 1.000000 consistency cosines in the session-1 re-encode). Therefore, the predictor's outputs at steady-state probes within the same viewing position are bit-identical → within-cluster cosine = 1.0 (deterministic). The M3 cluster sharpness then becomes `1.0 − cross_cluster_mean`, i.e., it is functionally measuring **cross-cluster discriminability** (how different the predictor's outputs are at different items) rather than within-cluster tightening over training. The rising trajectory is still a valid signal — it shows the predictor learning to discriminate items over training — but it does not test §2.2's "repetition tightens clusters" claim in the strong form intended. That test will become meaningful in Phases 2/3, where perturbed-shape rep counts start at 0 and accumulate, and where the jittered collection introduces actual within-cluster variance.

#### G1.3 detail

Across the 10 main checkpoints, the steady-vs-cue mean-log-var separation evolved as follows (steady mean − cue mean per checkpoint):

| step | steady mean log_var | cue mean log_var | sep (cue − steady) |
|---:|---:|---:|---:|
| 10,000 | −8.1811 | −8.0978 | +0.0832 |
| 20,000 | −8.1089 | −8.0956 | +0.0132 |
| 30,000 | −8.6761 | −8.8524 | −0.1762 |
| 40,000 | −8.2375 | −8.5147 | −0.2772 |
| 50,000 | −8.5113 | −8.7847 | −0.2734 |
| 60,000 | −7.9887 | −8.2547 | −0.2660 |
| 70,000 | −8.3213 | −8.6742 | −0.3529 |
| 80,000 | −8.2402 | −8.3690 | −0.1287 |
| 90,000 | −8.7949 | −8.9601 | −0.1651 |
| 95,721 | −8.2561 | −8.3987 | −0.1426 |

(Verified against `results/inner_pam_v0/phase1_main/log_var_trajectory.json`.)

The separation evolves in the **wrong direction** for the gate (towards cue being more confident than steady). The expectation in instr §7.7 G1.3 was `cue_log_var − steady_log_var > 0.3` (cue more variance = less confident). The empirical pattern is `cue_log_var − steady_log_var < 0` (cue less variance = more confident). At early checkpoints (10k, 20k) the separation is small and positive (steady marginally noisier than cue); by 30k it has already flipped negative and stays negative through 95,721. Magnitude bounces in the −0.13 to −0.35 range — the final value of −0.14 is not a steady-state value, it's where the trajectory happens to land at the last checkpoint.

Possible reasons (informational only, not for autonomous recalibration):

- *Substrate artefact.* With un-jittered Phase 1 data, steady probes have bit-identical inputs across loops but the predictor's bias terms may produce a non-zero residual error. With "trivial" target, the optimum log_var goes very low — but if there's a constant tiny error, log_var settles where the gradient balances. Cue probes have less trivial targets but smooth predictable transit-frame trajectories; their convergence might be sharper.
- *Architecture / loss artefact.* The loss formulation may permit / reward an unintended local minimum at cue probes.
- *Calibration mis-specification.* The +0.3 threshold may simply not be the right number; reviewer recalibration is on the table per §12 SCAFFOLDING discipline.

Disaggregated per-step k log_var values are in `log_var_trajectory.json`. The numbers are reported; the verdict is the reviewer's.

#### G1.4 detail — shuffle re-design candidate

The S1-S4 shuffle sanity check returns **"unexpected" on all four individual checks**:

| check | expectation | observed (final ckpt) | direction |
|---|---|---|---|
| S1 | shuffle log_var > main log_var (less confident) | shuffle −9.48 < main −8.66 | inverted |
| S2 | shuffle M1 < main M1 (lower centreline accuracy) | shuffle 0.97+ > main 0.97-ε | inverted |
| S3 | shuffle |sharpness| << main |sharpness| | shuffle similar/higher than main | inverted |
| S4 | quantitative collapse-to-mean | shuffle did not collapse | inverted |

Paired Wilcoxon at three horizons (250 steady-state probes, one-sided main > shuffle, Shapiro-Wilk rejected normality at p < 0.05 → Wilcoxon used):

| horizon k | mean_diff (main − shuffle) | shapiro p | test | p_value | pass at p < 0.01? |
|---:|---:|---:|---|---:|---|
| 1 | −0.302 | 6.6e-20 | Wilcoxon | 1.000 | no |
| 8 (gated) | −0.359 | 3.3e-08 | Wilcoxon | 1.000 | **no** |
| 16 | −0.250 | 2.9e-10 | Wilcoxon | 1.000 | no |

Across all three horizons, **shuffle predicts the held-out continuation better than main**, with mean cosine differences of 0.25–0.36. This is a HUGE divergence in the wrong direction, not noise.

**Root-cause diagnosis (load-bearing for the re-design decision).** The current shuffle implementation matches the literal wording of instr §7.5 — "applies `np.random.default_rng(0).permutation(N_train)` to the training indices" — interpreted as permuting the **visit order** of (window, target) pairs during training. Each pair, however, is still a real coherent W=16-frame window followed by a real K=16-frame continuation drawn from consecutive stream positions. Temporal structure within each pair is preserved.

This contradicts the spec rationale at §10.1 ("Should fail because the temporal structure required for shape learning is destroyed") and §6.3 C2 ("temporally-shuffled version of the same stream"). My current implementation is just SGD-with-random-batches, which is the standard ML optimization heuristic — and in fact it *helps* convergence relative to sequential SGD (consecutive sequential batches are highly correlated because of overlapping windows; random ordering decorrelates them). So shuffle ended up being a *better-optimized* version of main, not a worse one.

Per the `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md` preamble — "If this document and the spec disagree, the spec wins and this document is wrong — flag in `HANDOFF.md` and stop" — the spec wins. The shuffle should be re-implemented to actually destroy temporal structure (e.g., permute the embeddings themselves *before* building windows, so each window's contents are random unrelated frames). Cost: ~15 min re-train + ~2 min re-analysis.

If the reviewer agrees, I'll fix the shuffle (preferred path), re-run both shuffle training and the gate analysis, and re-write the §G1.4 section. The G1.4 verdict-as-computed stands at "HELD" pending this re-design.

### S1-S4 sanity check verdict

`results/inner_pam_v0/phase1_shuffle/sanity_check.json`:
- S1 (log_var distribution): shuffle mean −9.48, main mean −8.66. Shuffle MORE confident. **unexpected**.
- S2 (M1 distribution): shuffle aggregate cosine higher than main. **unexpected**.
- S3 (cluster sharpness): shuffle sharpness comparable to main; not zero. **unexpected**.
- S4 (quantitative collapse-to-mean): shuffle ||μ|| not below 0.15 floor; log σ² not above 0.4 floor. **non-collapse**, **unexpected**.

Aggregate verdict: **unexpected**. Per instr §6.5, the gate G1.4 is held; this entry is the documented flag.

### What's in the working tree

All Phase 1 result JSONs are committed in `results/inner_pam_v0/phase1_{main,shuffle}/`:
- `training_summary.json`, `init_report.json`, `tau_calibration.json` (main only).
- `checkpoint_{step}.json` (×10 each in main + shuffle).
- `eval_{step}.json` (×10 in main; shuffle eval was not run separately — the gate-report script computed shuffle predictions directly).
- `m3_trajectory.json`, `log_var_trajectory.json`, `gate_report.json` (main).
- `sanity_check.json` (shuffle).

Predictor checkpoints (`ckpt_*.pt`, ~258 MB each = ~2.5 GB / phase) and bank state dirs (`ckpt_*/`) are gitignored.

### Reviewer-action items before session 3

1. **Resolve the shuffle interpretation** (recommend: re-implement spec-correctly).
2. **Decide G1.3 verdict** (accept the surprising cue-more-confident-than-steady finding for now, or pause to diagnose the inversion). Recalibration of the +0.3 threshold is on the table per §12 SCAFFOLDING discipline.
3. **G1.5 PASS confirmation** at the scaffolding threshold (the trajectory criterion is the primary content; the 0.10 floor is met).
4. **If both shuffle and G1.3 are resolved as PASS / acceptable**: proceed to Phase 2 (LivingRoom RandomizeMaterials wrapper, preflight verification per §8.2). Session 3 setup.

### Host-protection decision recorded

Reviewer empirically confirmed the device stays up indefinitely (10 days of uninterrupted uptime under the current power-plan configuration). Settings verified by my probes:

| item | spec'd | actual | note |
|---|---|---|---|
| AC device sleep | Never | Never | ✓ confirmed by UI screenshot |
| AC screen sleep | (not load-bearing) | 10 min | does not affect WSL2 / training |
| AC hard-disk sleep | Never | **20 min** | not changed; low practical risk given continuous log/checkpoint writes resetting the idle timer |
| WU pause | active | **no pause keys** | low practical risk given active-hours [3, 21) defers reboots |

Recorded as a **deliberate deviation from §0.1** with the reviewer's empirical-evidence rationale. Both items proved out: the ~30-min training runs completed without disk-sleep interruption, and no forced reboot occurred during the session. The §0.1 wording is overspecified for this workload profile; if a future phase ran sub-process-idle (e.g., long pure-Python data loading without disk hits), the disk-sleep item would warrant revisiting.

### DINOv2 determinism observation (carryover note from reviewer)

The 1.000000 consistency cosine on the 50-frame re-encode sample in session 1 is strong evidence that DINOv2 in this environment (fp16 eval mode on RTX 4080 Super, transformers 5.3.0, ImageNet preprocessing pipeline) is genuinely bit-identical-deterministic on identical pixels. If Phase 2/3 substrate sanity-checks ever surface anomalies, re-running the §5 substrate verification (or a smaller spot-check) is a high-confidence first-line diagnostic — encoder behaviour is **not** a run-to-run noise source.

---

## Operational state (end of session 1)

(Historical — preserved for audit.)

- Working tree: clean. Eleven commits added in session 1 (see "Session 1 outcomes" below).
- Push hold: in effect.
- No running jobs.
- Embeddings file at `data/dinov2_embeddings/embeddings.npy`: 100,000 × 1024 fp32, all rows L2-normed (min cosine 1.000000 vs archived dwell-only file).
- Archived dwell-only file: `data/dinov2_embeddings/embeddings_dwell_only_v1.npy`.

---

## Session 1 outcomes — 2026-05-13

**Goal.** Build the v0 code scaffolding ready for session-2 Phase 1 training launch. Not training itself.

**DINOv2 reviewer approval — recorded.** The reviewer approved DINOv2 ViT-L/14 CLS as the v0 encoder on 2026-05-12, citing the substrate-verification + stability batch results: **Check 1 = `0.9260`, Check 2 = `0.4422`, Check 3 gap = `0.4838` — all PASS** against the §5 starting thresholds. The "human review of the DINOv2 stability PASS verdict" gate from the previous next-immediate-action is closed. v0 proceeds on DINOv2 ViT-L/14 CLS.

**STOP caught and resolved at pre-flight: embeddings file was dwell-only.** Pre-flight verification of `data/dinov2_embeddings/embeddings.npy` found that the file had the expected shape `(100000, 1024)` but **only 32,760 of the 100,000 rows were L2-normalised; the remaining 67,240 rows had norm = 0.0** (transit frames). The substrate-verification batch only needed dwell frames; transit frames were never encoded. Phase 1 training requires a contiguous stream (spec §2.3, instr §7.2). Stop reported, full-stream re-encode authorised by reviewer with one tightening (consistency threshold 0.999 → 0.9999).

Re-encode (`scripts/run_dinov2_encode_full_stream.py`, commit `a86c6f0`):
- Protocol: facebook/dinov2-large, frozen, fp16 eval, 256→224 center crop, ImageNet mean/std, CLS token, L2-normalise (same as substrate-verification).
- Wall-clock: 218.6 s on RTX 4080 Super, fp16, batch 64 (~457 frames/s).
- Norm check on all 100,000 rows: PASS (norms in [1−1e-5, 1+1e-5]).
- Consistency check on 50 random dwell frames against `embeddings_dwell_only_v1.npy`: **min cosine = 1.000000** (threshold 0.9999) — DINOv2 forward is bit-identical-deterministic on identical pixels.
- Report: [`data/dinov2_embeddings/encode_full_stream_report.json`](data/dinov2_embeddings/encode_full_stream_report.json) (gitignored).

**Documentation corrections caught at pre-flight.** Two items in `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md` were inconsistent with actual repo state:

| location | original | corrected | how caught |
|---|---|---|---|
| §0 Environment Header | "Python 3.10 (target match to previous repo)" | Python 3.12.3, matching the previous repo's `requirements.txt` header which explicitly says "WSL2, Python 3.12.3, CUDA 12.8 via WSL2 passthrough" | comparing §0 against the old repo's `requirements.txt` comments at pre-flight |
| §0 venv | `Active venv: .venv at repo root` | "none; uses system Python 3.12.3, matching the previous repo's verified-working pattern" | no `.venv` exists; the old repo also used system Python |
| §1.2 / §7.2 (embeddings precondition) | "100,000 frames × 1024 dim, fp32, L2-normalised" — implicitly assumed all 100k rows populated | (now true after the session-1 re-encode; no doc change needed, but the gap was load-bearing and is captured in this entry) | direct inspection of the file's norm distribution |

The §0 corrections are committed in `cc0a6a8`. Both errors were caught **before** any training launch, which is the design intent of the §7.2 / §4.7 norm checks.

**Code scaffolding delivered.** Eleven commits stand up the full Phase 1 pipeline:

| commit | scope | files |
|---|---|---|
| `e640dde` | infra | `requirements.txt`, `.gitignore`, `src/config.py`, all `src/*/__init__.py`, `src/env/explorer_phase{1,2,3}.py` stubs |
| `a86c6f0` | encoding | `scripts/run_dinov2_encode_full_stream.py` |
| `3016f23` | memory | `src/memory/memory_bank.py` (FAISS, hard cap, BankCapExceededError) |
| `a820ce1` | predictor | `src/predictor/inner_pam.py` (4-layer transformer, K*(d+1) head, Gaussian NLL) |
| `11e3f41` | mixing | `src/mixing/recall_mixer.py` (confidence threshold, τ calibration helper) |
| `567799f` | trainer | `src/trainer/online_trainer.py` (single-pass loop + §4.7 init-time checks) |
| `2dd3ae9` | eval | `src/eval/probes.py`, `metrics.py` (M1-M7), `controls.py` (C1 + S1-S4) |
| `4938a50` | encoder | `src/encoder/dinov2_encoder.py` (Phase 2/3 wrapper) |
| `715ba21` | scripts | `scripts/run_phase1_train.py`, `run_phase1_shuffle.py`, `run_eval.py` (with `--developmental` flag wired) |
| `b03062d` | tests | 21 tests across predictor / memory / mixer / probes / embeddings-norm invariant (all pass) |
| `cc0a6a8` | docs | `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md` §0 correction |

**Verification before commit (per instr §4.7 / instr §15-style review-cycle equivalents):**

- **21 unit tests pass** on system Python 3.12.3 / pytest 9.0.3: predictor shapes + param count (21,555,728 trainable params, within 2.6% of the 21M target — well inside the 10% tolerance), Gaussian-NLL closed-form sanity, log_var clamp at [−10, 10], target-detached-from-grad, memory-bank append + retrieve + hard-cap + FIFO + save/load round-trip, mixer routing + tau median calibration, probe construction (held-out boundary, steady-state uniform-dwell, cue dwell-to-transit), and an explicit "no-zero-rows" guard on `embeddings.npy` that would catch the dwell-only failure mode if it ever recurs.
- **§4.7 init-time smoke run on real Phase 1 data** (300-step budget): all four §4.7 checks pass — encoder frozen-equivalent (DINOv2 not loaded at training time; embeddings are precomputed), predictor trainable (21.6M params), forward pass produces correct shapes `(2, K, d)` + `(2, K)`, embedding norm check passes on 1000 sampled rows. 270 gradient steps in 4.3 s, no NaN/Inf, loss trended monotonically downward (first-50 mean ≈ −13,985 → last-50 mean ≈ −30,265 — Gaussian NLL is unbounded below; only the trend is informative). Bank populated correctly. Smoke artefact deleted before commits.

**Estimated session-2 budget.** At ~60 grad steps/sec, full Phase 1 (~95,700 training steps) is ~27 min plus checkpoint I/O. Shuffle control adds another ~27 min sequentially. Eval at 10 checkpoints × ~2 min/ckpt ≈ 20 min. Total session-2 wall-clock ≈ 75-90 min before gate review.

**Operational divergences from the instructions that are now resolved or recorded:**

1. **Python 3.12.3, not 3.10.** Doc corrected (commit `cc0a6a8`). System Python directly; no `.venv`. Matches the substrate-verification batch.
2. **`.env_snapshot.txt` written** (`pip freeze`, 207 packages) and gitignored per CODING_STANDARDS §8.4.
3. **`requirements.txt`** is pinned to the substrate-verification batch's stack plus `scipy==1.17.1` (used by G1.4 / G2.3 / G3.3 t-test + Wilcoxon fallback).
4. **Bug fix during session-1 encoding:** the encode script's temp-file rename relied on `Path.with_suffix(".npy.tmp")` which produced `.npy.tmp`, but `np.save` auto-appends `.npy`, so the file actually landed at `.tmp.npy` and the rename-to-final failed at the end of the encode. The work (encode + checks) had already completed cleanly before the rename; manual rename completed the artefact handover. Script fixed in the same commit so what's in git is what would work clean on a re-run.
5. **GPU has 3.4 GB used by the Windows desktop compositor.** Acceptable (12.6 GB free for training). No compute processes; no other ML jobs.

**Push hold remains in effect.**

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

**Resolution:** The reviewer authorised a full re-render (next entry).

*STOP commit:* `aefa1bc`.

---

## DINOv2 cross-instance stability under per-frame jitter — PASS (2026-05-12)

Fills the Check 1 gap left by the prior DINOv2 verification, whose
aggregate `1.0000` was a tautology (bit-identical pixels → bit-
identical embeddings). New collection: one full loop of the seed-7
furniture route with **per-frame** position+heading jitter applied
inside the explorer's dwell teleport, so every dwell frame has a
genuinely different pose.

**Spec interpretation decision (documented per CODING_STANDARDS §9.2).**
Batch §3 reads "apply per-loop jitter … the agent then dwells at the
jittered pose" (one jitter per visit) but §5 expects "~30 unique
frames per viewing position from one jittered loop" and §9 stops on
"fewer than 15 dwell frames per viewing position". One-jitter-per-
visit on a single loop gives 1 unique pose per item → Check 1 is
degenerate again, exactly the failure mode the batch was built to
fix. Per-frame jitter is the only interpretation consistent with §5's
sample-count expectation, so per-frame is what was implemented. RNG
seeded once with `jitter_seed=7`, drawn sequentially in frame order
for reproducibility. Flagging for review.

**Collection** (previous repo, `scripts/run_furniture_stability_collect.py`):

  - 458 frames total, one loop. Wall-clock 25.0 s.
  - 30 dwell frames at each of items 1..5 (150 total dwell); 308
    transit.
  - Jitter: `position_m=0.2` per horizontal axis, `heading_deg=10.0`,
    fallback ladder 100% → 50% → 25% → unjittered for NavMesh-
    unreachable poses. **Zero fallbacks** — all 150 jittered teleports
    succeeded at full 100% magnitude.
  - frames at [`data/seed7_dinov2_stability_frames/`](data/seed7_dinov2_stability_frames/)
    (PNG, ~12 MB total, gitignored); annotations at
    [`data/seed7_dinov2_stability_annotations.jsonl`](data/seed7_dinov2_stability_annotations.jsonl).
  - Modification in previous repo: `src/env/furniture_route_explorer.py`
    (jitter logic with fallback ladder, opt-in via constructor args),
    `src/env/ai2thor_furniture_env.py` (pass-through), and new
    `scripts/run_furniture_stability_collect.py` (pure data
    extraction, no V-JEPA 2 / predictor / trainer).

**DINOv2 stability test** (new repo, `scripts/run_dinov2_stability_test.py`):

| viewing_position_id | object | n pairs | mean | std | min | max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Bed | 50 | `0.9467` | `0.0289` | `0.8475` | `0.9889` |
| 2 | DiningTable | 50 | `0.9447` | `0.0189` | `0.9037` | `0.9755` |
| 3 | Dresser | 50 | `0.9317` | `0.0453` | `0.7834` | `0.9847` |
| 4 | Sofa | 50 | `0.8524` | `0.0969` | `0.6682` | `0.9749` |
| 5 | Television | 50 | `0.9547` | `0.0223` | `0.9034` | `0.9846` |

**Aggregate**: mean **`0.9260`** (n=250), std `0.0635`, min `0.6682`,
max `0.9889`. Pass criterion (>0.75): **PASS** with margin `0.176`.

**Pattern noted, not a finding.** Sofa is the least stable item (mean
0.8524, std 0.0969, min 0.6682). Sofa also produced the highest
cross-element pair in the prior Check 2 (DiningTable↔Sofa = 0.6709).
Coincidence is plausible; the report flags but does not interpret
the pattern.

**DINOv2 full §5 status (combining Check 1 from this batch with
Checks 2/3 from the prior DINOv2 verification on the same encoder):**

| check | DINOv2 | starting threshold | result |
|---|---:|---|---|
| 1 (cross-instance stability, non-degenerate jitter substrate) | `0.9260` | `> 0.75` | PASS |
| 2 (cross-element distinguishability, prior) | `0.4422` | `< 0.60` | PASS |
| 3 (combined gap, `0.9260 − 0.4422`) | `0.4838` | `≥ 0.15` | PASS |

Full report:
[`results/encoder_verification_dinov2_stability/STABILITY_REPORT.md`](results/encoder_verification_dinov2_stability/STABILITY_REPORT.md);
raw cosines + jitter summary in
[`results/encoder_verification_dinov2_stability/stability_data.json`](results/encoder_verification_dinov2_stability/stability_data.json).

**Caveat (recorded in the report §6).** Jitter magnitudes `0.2 m` /
`10°` are SCAFFOLDING values per the batch's §3 — verdict is
conditional on this magnitude. A non-trivially different magnitude
could produce a different aggregate; the protocol does not first-
principle-derive the jitter range from a model of natural agent-
instance variation. Flagged for reviewer.

*Stability commit: pending in both repos.*

---

## DINOv2 substrate verification on rerendered seed-7 frames — PASS (2026-05-12)

DINOv2 ViT-L/14 CLS, frozen, fp16 eval, encoded over the rerender's
32 760 dwell frames at items 1..5 (224×224 center crop of the 256×256
source, ImageNet mean/std). Same protocol, same seeds (7 / 8), same
pair counts, same sampling procedure as the V-JEPA 2 verification —
encoder is the only variable.

| check | DINOv2 (this batch) | starting threshold | V-JEPA 2 (prior) | DINOv2 result |
|---|---:|---|---:|---|
| 1. cross-instance stability (mean cosine, 250 pairs) | `1.0000` | `> 0.75` | `1.0000` | PASS (degenerate — see below) |
| 2. cross-element distinguishability (mean cosine, 1000 pairs) | **`0.4422`** | `< 0.60` | `0.8697` (FAIL) | **PASS** (load-bearing) |
| 3. combined gap (Check 1 − Check 2) | **`0.5578`** | `≥ 0.15` | `0.1303` (FAIL) | **PASS** |

**Verdict: PASS** (no recalibration applied; empirical values are not
within ±0.05 of the starting thresholds). Per-pair Check 2 means span
`0.2547` (DiningTable ↔ Television) to `0.6709` (DiningTable ↔ Sofa);
DiningTable ↔ Sofa is the only ordered pair above 0.60, and the
aggregate is still 0.16 below the threshold. Full per-pair matrix and
V-JEPA 2 side-by-side in
[`results/encoder_verification_dinov2/ENCODER_VERIFICATION_DINOV2_REPORT.md`](results/encoder_verification_dinov2/ENCODER_VERIFICATION_DINOV2_REPORT.md);
raw cosines in
[`results/encoder_verification_dinov2/verification_data.json`](results/encoder_verification_dinov2/verification_data.json).

**Check 1 carries the same degeneracy caveat as the V-JEPA 2 result.**
Within-position dwell frames are bit-identical across loops within the
rerender, so DINOv2 (deterministic eval-mode forward) produces bit-
identical embeddings — std `0.0000` across all 50 within-instance
pairs at all 5 items. The verdict stands on Check 2, which is genuine
encoder discrimination on bit-identical pixels and is not a sampling
artifact: every per-pair std is `0.0000` for the same reason (one
distinct cosine value per ordered pair), but the *values themselves*
are how DINOv2 separates the 5 items. The 10 distinct cross-pair
values range `0.2547`–`0.6709`, against V-JEPA 2's `0.8347`–`0.9210`
on the same items — a ~0.43 reduction in aggregate cross-element
similarity.

**Caveat (from RERENDER_REPORT).** Items 3 (Dresser) and 4 (Sofa) —
both LivingRoom — have constant per-item offsets from the original
V-JEPA 2 bank at the cosine `0.0005`–`0.0008` level *when read by
V-JEPA 2*. DINOv2 re-encodes the rerender's frames directly, so its
numbers are internally consistent. Recorded in case downstream
analysis surfaces an unexplained discrepancy at that magnitude; not
load-bearing on the verdict.

**Compute:** ~75 s of GPU forward (RTX 4080 Super, fp16, batch 64) +
~15 s for sampling / I/O. 32 760 dwell frames; one embedding per
frame; encoded once and saved to
`data/dinov2_embeddings/embeddings.npy` (391 MB, gitignored).

*Verification commit: pending.*

---

## Seed-7 furniture re-render with frames saved — PASS-AFTER-RECALIBRATION (2026-05-12)

**Final verdict updated 2026-05-12.** Reviewer applied a one-time
threshold recalibration from 0.9999 to 0.999 under spec §5.5; all 50
sampled frames pass the recalibrated threshold (`cos_min = 0.999188`).
Recalibration justification and final report:
[`results/frame_rerender/RERENDER_REPORT.md`](results/frame_rerender/RERENDER_REPORT.md).
Original stop record preserved in
[`STOP_REPORT.md`](STOP_REPORT.md) (commit `56050cc`), now marked
superseded. Frames at [`data/seed7_furniture_frames/`](data/seed7_furniture_frames/)
are usable substrate for downstream encoder verification.

The audit trail below is preserved from the original 2026-05-01 entry.

---

### Original entry (2026-05-01) — superseded by the 2026-05-12 recalibration above

The reviewer authorised re-rendering the seed-7 furniture run with
frames written to disk so DINOv2 (and any future encoder) could be
verified on the same substrate the V-JEPA 2 verification analysed.

**The re-render itself completed cleanly:**
  - 100 000 frames saved as `frame_{idx:08d}.png` to
    `data/seed7_furniture_frames/` (~5.2 GB, gitignored).
  - 218 loops completed — matching the original run's loop count
    exactly.
  - Wall-clock 11 219 s (~3.1 hr); ~5 min slower than the original
    due to PNG-write overhead.
  - `frame_annotations.jsonl` is **bit-identical** to the original
    run's (same md5 `6f241260...`); the explorer's trajectory and
    per-frame metadata are deterministic.
  - Modified script committed in previous repo as `98578d3`
    (`feat(furniture-rerun): save frames during forward pass for
    verification reuse`) — opt-in flags only; original behaviour
    preserved when neither flag is set.

**Determinism check FAILED at the spec'd 0.9999 threshold.** Re-encoded
50 sampled frames (10 per viewing position) through the same V-JEPA 2
checkpoint; compared cosine to original bank entries.

| viewing position | object type | room | n samples | cos (mean = min = max) | < 0.9999 |
|---:|---|---|---:|---:|---:|
| 1 | Bed | Bedroom | 10 | `1.000000` | 0/10 |
| 2 | DiningTable | Bedroom | 10 | `1.000000` | 0/10 |
| 3 | Dresser | LivingRoom | 10 | `0.999188` | **10/10** |
| 4 | Sofa | LivingRoom | 10 | `0.999481` | **10/10** |
| 5 | Television | Bedroom | 10 | `1.000000` | 0/10 |

**Pattern:** Bedroom items render bit-identically across runs (cos =
1.000000 exactly). LivingRoom items 3 and 4 differ from the original
by a small, item-specific, run-constant amount — every sampled
frame at item 3 has cos `0.999188` exactly; every sampled frame at
item 4 has cos `0.999481` exactly. The re-render is deterministic
*within* a run (frames at the same item across loops are bit-
identical, consistent with the V-JEPA 2 verification's degenerate
Check 1) but differs from the original *between* runs at the two
LivingRoom items.

**Most plausible cause:** scene-state-dependent rendering on first
entry to LivingRoom (shader compilation order, asset upload,
physics settling on instantiation). Bedroom is the spawn room and
warms before LivingRoom is ever rendered, so its rendering is stable
across runs. Once LivingRoom is "warm" within a run, it renders
deterministically — explaining the within-run consistency.
Numerically, the cosines correspond to L2 distances of 0.040 / 0.032
between unit vectors, ≈14–17× closer to "identical" than typical
inter-furniture cross-element distances (~0.55) — but the threshold
is 0.9999 and the protocol's stop trigger is "any sample below". Per
spec §5.5 / batch §9, recalibration is reviewer-only; the script does
not recalibrate the threshold autonomously.

**Per the batch §5 and §8, this is an unconditional stop.**

**Full evidence + four reviewer options** in `STOP_REPORT.md` at the
project root. Options range from a one-time threshold relaxation
(items 3 and 4 cluster near `0.999`, well above any plausible
"different content" floor) to investigating AI2-THOR non-determinism,
running DINOv2 on the re-render with the caveat documented, or
treating the V-JEPA 2 result as final and skipping alternative-encoder
verification on this bank.

**Operational state.**
  - Working tree: clean modulo this stop's commits.
  - `data/seed7_furniture_frames/` (5.2 GB), `data/seed7_furniture_rerender_aux/` (411 MB) gitignored.
  - Push hold: in effect.
  - No running jobs.

*STOP commit: pending.*
