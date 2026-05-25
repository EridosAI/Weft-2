# Weft Inner PAM v2 — Recalibration Phase Experiment Instructions

**Status:** First draft. Audience: CC (Claude Code), executing autonomously per `CODING_STANDARDS.md` and `research_operations_v1.md`. Adversarial review pending; CC implementation begins after reviewer sign-off.

**Purpose.** Bridge Phase 0 (partially invalidated) and the suspended Phase 1 (modules survive, measurements invalidated) by re-establishing the predictor's training-step calibration and rebuilding the downstream measurements that depended on it. This is *not* a Phase 1 rewrite — Phase 1 modules (`arm_runner`, `classification`, `parallel_harness`, `sweep_grid`, `controls`, `aggregate`) are correct and inherit unchanged. Only the *measurements* are redone.

**Read order before CC begins:**
1. `WEFT_INNER_PAM_v2_Spec_pass1_sections_1_to_3.md` (architecture and claims)
2. `WEFT_INNER_PAM_v2_Spec_pass2_sections_4_to_9.md` (implementation specification)
3. `WEFT_INNER_PAM_v2_SESSION_BRIEF.md` (design-chat context, including the broken-mechanics episode and four-forks resolution)
4. `CODING_STANDARDS.md` (operational discipline)
5. `research_operations_v1.md` (process discipline)
6. `WEFT_INNER_PAM_v1_CLOSING.md` (institutional memory)
7. `v2/HANDOFF.md` (Phase 0 conclusion, commit `1f5f62a`)
8. `v2/PHASE1_PROGRESS.md` (sub-phase log up to broken-mechanics; reference only — measurements invalidated)
9. This document

**Push hold remains in effect throughout this batch.**

**Origin of this document.** Design-chat session 2026-05-25 reviewed the broken-mechanics episode documented in `WEFT_INNER_PAM_v2_SESSION_BRIEF.md` and committed to the four-forks resolution recorded there. This document operationalises that resolution. Where this document and the session brief disagree, the session brief wins and this document is wrong — flag in `HANDOFF.md` and stop.

---

## 0. Environment Header

| field | value |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| Shell | bash |
| Python | 3.12.3 (invoked as `python3`; bare `python` is not on PATH) |
| PyTorch | 2.10.0+cu128 (CUDA 12.8 via WSL2 passthrough) |
| Host CPU | Intel Core i9-14900K @ 3.20 GHz |
| Host RAM | 64 GB |
| GPU | RTX 4080 Super, 16 GB VRAM (local) — Stage 1 only; later stages may move to vast.ai per §3.8 |
| Host disk | check `df -h` ≥ 100 GB free at batch start |
| Working repo | `/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/` |
| Active venv | none; system Python 3.12.3 |
| Required env var | `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000` |
| Repo state at batch start | v2 at commit `313efa5` (post-confirmatory-training-length test); HANDOFF draft held uncommitted, marked invalidated |

CC verifies the environment matches before any work. Divergence is documented in `HANDOFF.md` and resolved before proceeding.

### 0.1 Host-session protection (local hardware run)

Stage 1 runs locally (~4–6 hours). Per Phase 1 §0.1, the protection layer is:

1. **Use `tmux` for the CC session itself.** `tmux new -s weft` inside WSL2 before launching CC. Detach with `Ctrl-b d`; reattach with `tmux attach -t weft`.
2. **Disable Windows sleep and disk sleep.**
3. **Defer Windows updates.**
4. **Disk budget check before any stage starts.** Estimated for Stage 1: ~5 GB (7 grok-curve training runs at long horizon with checkpoints).
5. **GPU lockout.** Close concurrent GPU-using applications.

If a later stage migrates to vast.ai per §3.8, the protection layer for that stage is replaced by the rental-instance discipline (see §X.5 of the per-stage instructions, populated post-Stage-1 decision).

---

## 1. Scope Lock and Locked Decisions

These are settled by the session brief and the design-chat session that produced this document. They do not get re-litigated in this batch.

### 1.1 What the recalibration phase produces

1. A new `V2_TRAINING_STEPS` value, derived from multi-cell grokking detection with a mean-head-aware lock criterion (replaces the broken loss-plateau calibration).
2. Confirmation via the diagnostic battery (Layer-1 validation) that the recalibrated training produces sensible mechanics across the property space.
3. A characterisation of inter-seed predictor convergence post-grok (the unviable-egg test) — substantive architectural finding regardless of pilot outcome.
4. Fresh PRE-D1a baseline, PRE-E τ_W, and PRE-D2 n-validation at the new training horizon.
5. Phase 1 controls, baseline-variance diagnostic, and pilot at corrected training. Pilot outcome decides whether main effects are warranted.

### 1.2 What is preserved from Phase 0 (not redone)

These are uncorrupted (no predictor training involved). Phase 1 modules and the v2 substrate primitives inherit them unchanged:

- PRE-A construction-sanity sweep (substrate primitives validated; `results/pre_a/`)
- PRE-B worked-example region characterisation (`results/pre_b/worked_example_region.json`)
- PRE-C architectural assertions (`results/pre_c/arch_assertions_v2_substrate.json`)
- PRE-D1c corner-reachability assessment (analytical; `results/pre_d1c/corner_reachability_assessment.md`)
- Sub-phase 1.0 grid validation (`results/phase1/grid_calibration.json`)
- The substrate primitives, property measurement, and protocol modules in `v2/src/`
- All Phase 1 code infrastructure (`v2/src/phase1/`) — the code is correct; it will produce valid results when run against correctly-trained predictors

CC does not modify any of these. They are read-only references for the recalibration cascade.

### 1.3 What is invalidated and requires redo

Every measurement that depended on a trained predictor at the broken `V2_TRAINING_STEPS=10000`:

- PRE-A's `V2_TRAINING_STEPS` lock value (the calibration target was wrong)
- PRE-D1a baseline distribution
- PRE-E τ_W calibration
- PRE-D2 n=10 vs n=20 validation at L_d=2
- Phase 1 sub-phase 1.2 controls (C1/C2)
- Phase 1 sub-phase 1.2.5 baseline-variance diagnostic
- Phase 1 sub-phase 1.3 pilot (L_d=1)
- Phase 1 sub-phase 1.4 capacity extension (L_d=2/4)

These are referenced for shape and pre-registered design but their numerical results do not carry forward.

### 1.4 Locked design-chat decisions (the four forks)

Per the session brief §8 and the design-chat session that produced this document:

| Fork | Decision | Rationale |
|---|---|---|
| 1. Recalibration scope | Option B with 7 cells × n=3 seeds: L_d=1 mid mag=0 anchor, L_d=2 mid mag=0, L_d=4 mid mag=0, D=128 mid, mag=0.9 mid, P=32 mid, cont=0.8 mid. Spans-the-space, not slowest-cell-guess. Per-cell lock is max-over-seeds; global lock is max-over-cells × 1.1. | The broken-mechanics episode came from calibrating on a single cell at a single seed. Multi-cell with deliberate axis spread is the conservative response to the cell-level failure mode; n=3 seeds per cell is the conservative response to the seed-level failure mode. The 1.1× residual margin covers un-probed cells and un-probed seeds; seed variation is now measured, not assumed away. |
| 2. Compute strategy | DEFERRED. Stage 1 runs locally; compute-strategy decision (local 2x vs vast.ai) re-made by experiment chat after Stage 1 produces the lock value. | At 10× per-arm cost, full cascade is impractical locally. But final cost depends on the lock value Stage 1 produces. Decision needs the data. |
| 3. Cascade ordering | Stage 1 → Stage 2 (sanity gate) → Stage 3 (unviable-egg) → Stage 4 (PRE-D1a) → Stage 5 (PRE-E) → Stage 6 (PRE-D2) → Stage 7 (Phase 1 controls) → Stage 8 (baseline-variance) → Stage 9 (pilot L_d=1) → Stage 10 (capacity L_d=2/4). | Sanity battery at Stage 2, not Stage 6 (per session brief). Layer-1 validation must gate Layer-2 measurements — that's the institutional lesson v2 just learned at cost. |
| 4. Closing-trail strategy | Option A. Pilot at corrected training is the deciding artefact. Main effects deferred (possibly to v3+) unless pilot signals. | The methodology product + grokking finding + Layer-1 discipline lesson is already substantive v2 contribution. Pre-committing main effects locks compute against an unknown outcome. |

### 1.5 What NOT to change

- **`V2_TRAINING_STEPS` is determined by Stage 1's lock procedure.** CC does not pick a value before Stage 1 lands. The previous lock (10000) is invalidated and is not the starting point for anything.
- **Mean-head-aware plateau detection replaces NLL-plateau.** Do not re-instate the NLL-flatness criterion anywhere. The variance head saturates first and masks the mean head; loss flatness was the trap.
- **No threshold pre-registration for the unviable-egg test.** Pre-registering a numerical cosine cutoff would commit to interpretation of an architectural finding before the data lands — and we have no validated basis to say whether 0.5, 0.7, or 0.9 is the "high-correlation" regime for trained predictors. CC reports descriptive statistics; frame assignment is experiment-chat work post-hoc with the distributions in hand. See §5.4 for the operational specification.
- **Per-config baselining (Option 1) is the inherited convention.** It may be re-validated at corrected training (§10) but it is not re-litigated as a default.
- **Phase 1 module code is not modified.** Bugs found are flagged in `HANDOFF.md`; fixes go through the experiment chat.
- **v0, v1, and shared trees are frozen** at `58e91d7`. `git diff 58e91d7 HEAD -- v0 v1 shared` must remain empty for the duration of this batch.
- **Push hold preserved.** No `git push` under any circumstances.

### 1.6 What is deferred beyond this phase

- Phase 1 main effects (the 1350-arm-run sweep): only if pilot signals; otherwise deferred to v3+ per Fork 4.
- Closing document drafting: happens after the cascade lands, in a separate session.
- Phase 2 (Inner-PAM under coherent-trajectory-variant perturbations) and any further architectural variants: out of v2 scope per the spec.

---

## 2. Pre-launch gates

Before Stage 1 begins, CC verifies:

### 2.1 Read-only inheritance check

```bash
# From parent repo root
git diff 58e91d7 HEAD -- v0 v1 shared
# Expected: empty output
```

If output is non-empty, STOP and flag in `HANDOFF.md`. The frozen-tree discipline has been violated; the recalibration cannot proceed against a corrupted inheritance.

### 2.2 Canaries

From parent repo root:

```bash
python3 -m pytest v0/tests v1/tests v2/tests
# Expected: 121 PASS (21 v0 + 51 v1 + 49 v2)

python3 v1/scripts/run_pre_d_arch_assertions.py --decoder-n-layers 2
# Expected: 11 PASS
```

If any canary fails, STOP and flag in `HANDOFF.md`.

### 2.3 Working tree clean

```bash
git status
# Expected: clean, on the post-Stage-0.7 commit branch
```

The pre-Stage-1 commit is `313efa5` (the confirmatory training-length test). `git log -1 --oneline` confirms it. If there are uncommitted changes, CC stops and surfaces them — they belong to a prior session and must be reconciled in the experiment chat.

### 2.4 Phase 1 modules still pass tests

```bash
python3 -m pytest v2/tests/phase1
# Expected: 12 PASS (8 sweep_grid + 4 classification)
```

These tests are unaffected by the broken-mechanics finding (they test module behaviour, not predictor output). If they fail, the inheritance is broken and STOP applies.

### 2.5 HANDOFF check

Read `v2/HANDOFF.md`. Confirm it reflects the Phase 0 conclusion (commit `1f5f62a` per session brief §2). If a Phase 1 broken-mechanics HANDOFF has been committed (it should not have been — it was held uncommitted), STOP and surface.

---

## 3. Stage 1 — Multi-cell grokking detection

**Goal:** lock `V2_TRAINING_STEPS` to a value at which all seven cells have reached post-asymptotic learning on `cos(mean, target)` against the trivial last-frame baseline.

**Estimated wall-clock:** 4–6 hours on local 4080 Super at 2x parallelism.

### 3.1 Cell selection

Seven cells, each defined by construction parameters per the locked Phase 1 grid (`v2/src/phase1/sweep_grid.py`):

| ID | L_d | P | D | continuity_center → measured C | magnitude_M | locality_L | Notes |
|---|---|---|---|---|---|---|---|
| C1 | 1 | 256 | 16 | c39 → 0.425 | 0 | 0.5 | Anchor — matches existing confirmatory-test data |
| C2 | 2 | 256 | 16 | c39 → 0.425 | 0 | 0.5 | Capacity probe |
| C3 | 4 | 256 | 16 | c39 → 0.425 | 0 | 0.5 | Capacity probe |
| C4 | 1 | 256 | 128 | (boundary; forced C≈1.0) | 0 | 0.5 | High-D probe; record forced-continuity caveat |
| C5 | 1 | 256 | 16 | c39 → 0.425 | 0.9 | 0.5 | Strong-perturbation probe |
| C6 | 1 | 32 | 4 | c5 → 0.417 | 0 | 0.5 | Low-period probe. D=4 IS the substrate-feasible mid at P=32 (D=16/P=32 is the boundary case with forced continuity, D=128/P=32 is excluded per sub-phase 1.0 grid validation). C6's grok behaviour informs all P=32 grid cells, since they are constrained to D=4 by feasibility. |
| C7 | 1 | 256 | 16 | c59 → 0.869 | 0 | 0.5 | High-continuity probe |

Construction uses the same `sweep_grid.cell_for(...)` helper Phase 1 used; no new substrate code. The locality dial is held at 0.5 across all cells (the variable being probed is grokking onset, not locality dependence).

### 3.2 Procedure: n=3 seeds per cell, each a single training run with progressive checkpoints

For each cell, for each seed in `{0, 1, 2}`:
1. Construct the substrate from the locked grid (per §3.1 parameters); substrate construction is deterministic per the spec and does not vary with the predictor seed.
2. Initialise a Primary predictor with the seed.
3. Train one continuous run to **200000 steps** (the upper bound established by the confirmatory test in PHASE1_PROGRESS §16).
4. At each checkpoint in `{2000, 5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000}`, evaluate `cos(mean, target)` at k=1 and aggregated across K on the trained-on stream.
5. Also compute the **trivial last-frame baseline** (cos between consecutive frames in the substrate) for that cell — this is the threshold to beat.

This is the same pattern as `scripts/run_phase1_training_length_test.py` (commit `313efa5`), generalised to 7 cells × 3 seeds = 21 grok-curve runs. The single-seed version of this experiment is what produced the broken-mechanics episode; n=3 per cell is the measured-not-assumed response to that failure mode.

CC adapts the existing script's runner; the script handles only L_d=1 currently, so the L_d parameter and the seed parameter both need to thread through.

Write per-(cell, seed) results to `v2/results/recalibration/stage1/grok_curve_{cell_id}_seed{N}.json` with schema:
```
{
  "cell_id": "C1",
  "seed": 0,
  "construction": {L_d, P, D, continuity_center, magnitude_M, locality_L},
  "measured_properties": {measured_C, magnitude, locality, period, D_global},
  "trivial_baseline_cos_k1": <float>,
  "curve": [{steps, cos_k1, cos_allK, mean_norm_k1, loss, wall_min}, ...],
  "lock_step_candidate": <int>  // per-seed post-asymptotic step count, computed per §3.3
}
```

After all 21 runs land, aggregate per-cell to `v2/results/recalibration/stage1/grok_curve_{cell_id}_aggregate.json`:
```
{
  "cell_id": "C1",
  "construction": {...},
  "seeds": [0, 1, 2],
  "lock_step_candidate_per_seed": [<s0>, <s1>, <s2>],
  "lock_step_candidate_max": <max over seeds>,  // the conservative per-cell value used in §3.3
  "lock_step_candidate_median": <median over seeds>  // for descriptive reading only
}
```

### 3.3 Lock criterion: post-asymptotic, conservative on measured seed variation

For each (cell, seed), compute `lock_step_candidate` as the smallest checkpoint step at which:
- `cos_k1` ≥ 0.99 × max(`cos_k1` across all checkpoints for that (cell, seed)), AND
- The next checkpoint's `cos_k1` does not exceed it by more than 0.01.

If max `cos_k1` for a (cell, seed) does not exceed the trivial baseline by at least 0.10 at any checkpoint, that (cell, seed) is **not-grokked-within-budget** — record explicitly, surface as a Stage 1 finding, and STOP per §3.5.

Per-cell aggregate: `lock_step_candidate_max` = max over the 3 seeds. This is the per-cell conservative value (slowest-observed-seed); using max rather than median because the broken-mechanics episode was specifically a single-seed extrapolation failure, and the recalibration cascade should not repeat that failure mode at a different granularity.

Across the 7 cells, the locked `V2_TRAINING_STEPS` is:

```
V2_TRAINING_STEPS = round_to_clean(max(lock_step_candidate_max over 7 cells) × 1.1)
```

where `round_to_clean()` rounds up to the nearest of {50000, 75000, 100000, 125000, 150000, 175000, 200000}.

**Why 1.1× and not 1.2×.** The earlier draft used 1.2× as a margin absorbing assumed seed variation. With n=3 seeds per cell, seed variation is now measured rather than assumed, and the conservative reading is the worst-observed seed across the seven cells. The 1.1× residual margin covers un-probed cells and un-probed seeds; it is not absorbing variation that's already in the data.

**Why post-asymptotic, not just post-trivial.** Post-trivial (cos > trivial baseline) confirms the predictor has learned *something*; post-asymptotic confirms it has reached the productive regime. Locking at "barely above trivial" would reproduce the broken-mechanics-style fragility at a different operating point.

### 3.4 Mean-head-aware plateau detection (replaces NLL plateau)

The plateau-detection logic in `v2/src/config.py` (or wherever PRE-A's calibrator lives) is updated as part of this stage. New criterion:

```python
def mean_head_plateau_detected(curve):
    """
    curve: list of {steps, cos_k1, ...} dicts.
    Returns the step at which cos_k1 has plateaued (within 1% of max,
    no improvement > 1% over next checkpoint), or None if not yet plateaued.
    """
    # Implementation per §3.3 lock criterion logic.
```

CC writes this function to `v2/src/preflight/mean_head_plateau.py` and adds unit tests in `v2/tests/preflight/test_mean_head_plateau.py`. The legacy NLL-plateau detector is left in place but no longer invoked for `V2_TRAINING_STEPS` calibration — it is marked deprecated in a comment with a pointer to this section.

### 3.5 STOP conditions

CC stops and surfaces in `HANDOFF.md` if:
- Any (cell, seed) `lock_step_candidate` exceeds 175000 (i.e., the 1.1× margin would push `V2_TRAINING_STEPS` past 200000 — outside the budget characterised by the confirmatory test).
- Any (cell, seed) is not-grokked-within-budget (max `cos_k1` does not clear trivial baseline + 0.10 at any checkpoint).
- Within-cell across-seed spread exceeds 4× (e.g., seed 0 groks at 25k and seed 2 at 100k+) — this surfaces a seed-dependent grokking instability at the cell level that the design chat needs to reckon with before committing the lock value.
- Inter-cell `lock_step_candidate_max` spread exceeds 4× — this surfaces a capacity- or property-coupled grokking finding.

In any STOP case, CC does NOT proceed to Stage 2. The lock value is decided in the experiment chat with Stage 1's curves in hand.

### 3.6 Output and HANDOFF

On Stage 1 completion:
- Per-(cell, seed) curves at `v2/results/recalibration/stage1/grok_curve_{cell_id}_seed{N}.json` (21 files)
- Per-cell aggregates at `v2/results/recalibration/stage1/grok_curve_{cell_id}_aggregate.json` (7 files)
- Write `v2/results/recalibration/stage1/lock_decision.json` with:
  - The 7 cells' `lock_step_candidate_max` values (worst seed per cell)
  - The 7 cells' within-cell across-seed spread
  - The proposed `V2_TRAINING_STEPS` value
  - Whether any STOP condition fired
  - The mean-head-aware plateau detector's per-(cell, seed) verdict
- Update `v2/HANDOFF.md` with Stage 1 outcome and the lock value (if no STOP fired) or the STOP condition (if it did).
- Git commit: `recalibration stage 1: multi-cell n=3 grok detection -> V2_TRAINING_STEPS = <value>` (or `... STOP at <condition>`).

If no STOP fired, CC proceeds to Stage 2 in the same session. If STOP fired, CC halts and the experiment chat re-plans.

### 3.7 Compute envelope

Per the confirmatory test in PHASE1_PROGRESS §16: 38 min to 200k steps at L_d=1 on 4080 Super. Per-step cost scales linearly with L_d. With n=3 seeds per cell:
- L_d=1 cells (C1, C4, C5, C6, C7): 5 cells × 3 seeds × ~38 min = 570 min
- L_d=2 cell (C2): 3 seeds × ~43 min = 129 min
- L_d=4 cell (C3): 3 seeds × ~63 min = 189 min

Sequential total: ~888 min ≈ 15 hours. At 2x parallelism (1.447× overhead per Phase 1 §6): ~10 hours. Allow 16 hours for buffer.

**Why the n=3 cost is justified.** The broken-mechanics episode was a single-point calibration failure. Locking `V2_TRAINING_STEPS` based on one seed per cell would reproduce the same failure pattern at a different granularity. The 10-hour Stage 1 cost is small against the 100+ hour total cascade cost, and it removes a known failure mode from the lock value.

VRAM check: at 2x parallelism the existing Phase 1 modules saturated 9.1/16.4 GB at L_d_main=2. L_d=4 cells consume more — VRAM monitor must run at Stage 1 launch to confirm 2x feasibility for C3. If 2x is infeasible for C3, drop to serial for the three C3 runs only (adds ~90 min wall-clock).

### 3.8 SURFACE: Fork 2 compute strategy decision

On Stage 1 completion, CC writes to `HANDOFF.md` a section titled **"Fork 2 decision input"** containing:

- The locked `V2_TRAINING_STEPS` value
- The per-arm cost ratio (new vs old): `(new_steps / 10000) × baseline_per_arm_seconds`
- Estimated wall-clock for the remaining cascade at local 2x parallelism, computed as:
  - Stage 3: 30 arm-runs × per-arm cost
  - Stage 4: 20 arm-runs × per-arm cost  
  - Stage 6: 200 arm-runs × per-arm cost (L_d=2)
  - Stage 7: 120 arm-runs × per-arm cost
  - Stage 8: 40 arm-runs × per-arm cost
  - Stage 9: 240 arm-runs × per-arm cost
  - Stage 10: 480 arm-runs × per-arm cost
- Estimated wall-clock at vast.ai 8x (assume 8× speedup over local 2x as nominal; experiment chat refines)
- Estimated rental cost at $3.20/hr

CC does NOT make the compute-strategy decision. CC continues to Stage 2 on local hardware regardless. The Fork 2 decision is made by the experiment chat once Stage 3 (the next stage that consumes meaningful compute) is in view.

### 3.9 If Fork 2 lands on vast.ai: cross-hardware smoke required

If the Fork 2 decision (made by the experiment chat) commits to vast.ai for Stage 6 onwards, the rental session opens with a cross-hardware sanity check before launching any cascade work:

1. Re-train one Primary predictor at C1 (the anchor) at the locked `V2_TRAINING_STEPS`, seed=0, on the rental hardware.
2. Re-run the full Stage 2 sanity battery (S1, S2, S3) on the rental-trained predictor.
3. Compare to local Stage 2 results.

Pass criteria:
- S1, S2, S3 all PASS on rental hardware (same gate as Stage 2 §4.2).
- Diff_μ on the training stream within ~20% of the local value (cross-hardware numerical drift is acceptable; order-of-magnitude differences are not).

STOP if cross-hardware smoke fails. The cascade does not proceed on hardware where the mechanics don't replicate. Write to `HANDOFF.md` and halt; experiment chat decides remediation (different rental hardware family, re-validation locally, etc.).

Output: `v2/results/recalibration/stage3_5_vastai_smoke.json` (numbered between Stage 3 and Stage 4 because that is where it temporally falls if Fork 2 is decided then). If Fork 2 is decided later (e.g., at the Stage 6 SURFACE checkpoint), the smoke runs at that time instead.

---

## 4. Stage 2 — Sanity battery gate at locked V2_TRAINING_STEPS

**Goal:** confirm Layer-1 validation passes at the new lock value. This is the gate that the broken-mechanics episode established as institutional discipline (session brief §5 methodology lesson).

**Estimated wall-clock:** 30 minutes (one trained predictor per the anchor cell C1; sanity checks themselves are fast).

### 4.1 Five sanity checks

The diagnostic battery from `scripts/run_phase1_sanity_battery.py` (commit `ee757d8`) is reused with two changes:
1. The predictor is trained at the locked `V2_TRAINING_STEPS` from Stage 1, not 10000.
2. The cell is C1 (the anchor; same as the original sanity battery cell, for direct before/after comparison).

The five checks:
- **S1 (training-stream fit):** `cos(mean, target)` on the predictor's own training stream. Compare to the trivial last-frame baseline (~0.559 at C1).
- **S2 (full trajectory swap):** prediction error on familiar (A) vs novel (B) substrate.
- **S3 (analytical metric cases):** Diff_μ on (a) training-identical, (b) orthogonal-offset, (c) random-noise streams.
- **S4 (loss trajectory inspection):** loss curve shape across capacities. Diagnostic only — not a gate (the broken-mechanics episode established that S4 PASS does not imply mechanics-sound).
- **S5 (inter-seed correlation, single pair):** two seeds at the same cell, cosine between outputs. Diagnostic; the rigorous unviable-egg test is Stage 3.

### 4.2 Gate criteria: sanity 1/2/3 must PASS

Pass criteria (these ARE pre-registered — they are mechanical sanity, not the variance-question pre-registration the design chat declined):

- **S1 PASS:** `cos_k1` on training stream > trivial baseline + 0.10 (i.e., the predictor decisively outperforms the trivial copy-last-frame).
- **S2 PASS:** `err_mean` on novel substrate B exceeds familiar substrate A by at least 0.10 (relative to either's value). The predictor is sensitive to substrate identity.
- **S3 PASS:** Diff_μ ordering is sensible — case (a) near zero (≤ 0.05), case (b) reflects offset magnitude meaningfully (> 0.10), case (c) ≥ case (a) (random noise is not "more satisfying" than the trained-on stream).

### 4.3 STOP condition: any of S1/S2/S3 fail

If any of S1/S2/S3 fail at the locked `V2_TRAINING_STEPS`, CC stops and writes to `HANDOFF.md`:
- Which sanity check failed
- The numbers vs the pass criterion
- Recommendation: experiment chat re-plans (possibilities: lock value still too low; cell C1 has a substrate-specific issue; a deeper defect in the mechanics).

The recalibration cascade does not proceed past Stage 2 with failing Layer-1 validation. This is the gate the broken-mechanics episode established.

### 4.4 Output and HANDOFF

- Write `v2/results/recalibration/stage2/sanity_battery_at_lock.json` with the same schema as `sanity_battery.json` (commit `ee757d8`).
- Update `v2/HANDOFF.md` with Stage 2 verdict (PASS / STOP).
- Git commit: `recalibration stage 2: sanity battery at lock=<value> - PASS` (or `... STOP`).

If PASS, CC proceeds to Stage 3 in the same session.

---

## 5. Stage 3 — Sanity-5 post-grok (the unviable-egg test)

**Goal:** characterise whether different seeds converge to similar functions at the locked `V2_TRAINING_STEPS`, distinguishing "broken mechanics fully explained the variance-limited null" from "broken mechanics + a real architectural variance limit."

**Estimated wall-clock:** depends on lock value. At 100k steps: 30 training runs × ~16 min = ~8 hr sequential, ~5 hr at 2x. At 50k: ~4 hr sequential, ~2.5 hr at 2x.

**Substantive importance:** this is the most architecturally informative test in the cascade. The session brief §5 calls it out explicitly: a low post-grok inter-seed correlation is a real finding about the architecture, independent of whether the pilot signals.

### 5.1 Cells and seed count

Three cells, n=10 seeds each:

| ID | Cell config | Why |
|---|---|---|
| U1 | C1 (L_d=1 mid mag=0 anchor) | Direct comparison to the broken-mechanics 0.124; same cell |
| U2 | C3 (L_d=4 mid mag=0) | Capacity probe — does more capacity stabilise inter-seed convergence? |
| U3 | C5 (L_d=1 mid mag=0.9) | Perturbation probe — does strong perturbation drive seeds toward shared structure or apart? |

Each seed is an integer in `[0..9]`. The seed controls predictor initialisation, dropout RNG, and training-data presentation order (per the existing `OnlineTrainerV1` discipline; no new variation introduced).

### 5.2 Per-cell pairwise cosine measurement

For each cell:
1. Train 10 predictors at the locked `V2_TRAINING_STEPS`, seeds 0..9, all other parameters fixed.
2. On a fixed evaluation stream (the same stream used for S1 in Stage 2; the cell's own training stream), compute each predictor's mean-head output.
3. Compute pairwise cosine between all `10·9/2 = 45` ordered seed pairs.
4. Record the distribution: `{median, IQR, min, max, all 45 values}`.

### 5.3 Cross-cell comparison

Compare the three cells' distributions side-by-side. Report:
- Per-cell median and IQR
- Whether the three cells' distributions overlap or separate (eyeballed; no statistical test pre-registered)

### 5.4 Descriptive output; frame assignment is experiment-chat work

The design-chat decision is to NOT pre-register a numerical cosine threshold for assigning frames. The reasoning: we have one prior data point (0.124 at broken mechanics) and no validated basis to say whether 0.5, 0.7, or 0.9 is the "high-correlation" regime for trained predictors. Picking a cutoff in advance would be making up a number and calling it pre-registration.

What CC does:
- Compute the raw pairwise cosine distribution per cell (all 45 pairs)
- Compute per-cell descriptive statistics: median, IQR, % > 0.7, % > 0.5, % < 0.3
- Write these to the output JSON
- Write a brief narrative summary in `HANDOFF.md` describing the distributions in qualitative terms (e.g., "U1 distribution is bimodal with peaks near 0.2 and 0.85"; "U2 distribution is tightly clustered around 0.6")

What CC does NOT do:
- Assign a frame label (A/B/C/ambiguous) to any cell
- Assign a frame label to the test as a whole
- Compare the distributions against a pre-registered cutoff to declare a verdict

The frame assignment is experiment-chat work, made with the raw distributions in hand. §5.4 (this section) describes the *interpretive frames* that the experiment chat may apply post-hoc — they are not pre-registered thresholds for CC to evaluate:

- **Frame A — convergent:** if the distributions across all three cells suggest seeds learning a shared function, the broken mechanics fully explained the prior 0.124 reading. Variance-limited finding would be an artefact.
- **Frame B — divergent:** if the distributions suggest seeds settling into different functions across cells, the prior 0.124 was *both* broken mechanics *and* a real architectural variance limit. The pilot's reading at corrected training would be more strongly anchored as architectural.
- **Frame C — capacity- or perturbation-coupled:** if the distributions differ markedly across cells (e.g., L_d=4 convergent, L_d=1 with mag=0.9 divergent), the architecture's inter-seed convergence depends on configuration in a way worth characterising.

The frames are tools for the experiment chat to think with; they are not gates CC evaluates.

### 5.5 Output and HANDOFF

- Write `v2/results/recalibration/stage3/unviable_egg_test.json`:
```
{
  "lock_steps": <V2_TRAINING_STEPS>,
  "cells": [
    {
      "id": "U1",
      "config": {...},
      "distribution": {
        "median": <float>,
        "iqr": <float>,
        "pct_above_0_7": <float>,
        "pct_above_0_5": <float>,
        "pct_below_0_3": <float>,
        "pairs": [<45 floats>]
      }
    },
    {"id": "U2", "config": {...}, "distribution": {...}},
    {"id": "U3", "config": {...}, "distribution": {...}}
  ],
  "narrative_summary": "<CC's qualitative description of the per-cell distributions; no frame assignment>"
}
```
- Update `v2/HANDOFF.md` with the per-cell narrative description.
- Git commit: `recalibration stage 3: unviable-egg test post-grok at lock=<value>`.

### 5.6 What this affects downstream

Stage 3 does not gate downstream stages — the pilot runs regardless. But Stage 3's distributions are a **substantive input to the closing interpretation**, not just a recorded observation:

- If Stage 3's distributions suggest convergent seeds (broadly Frame A) AND the pilot lands at robust_null, the null is more readily explained as architectural-variance-was-broken-mechanics. Closing trail reads as "broken mechanics fully accounted for the prior null."
- If Stage 3's distributions suggest divergent seeds (broadly Frame B) AND the pilot lands at robust_null, the null is more strongly anchored as a real architectural variance limit. Closing trail reads as "broken mechanics + a real architectural variance limit, stacked."
- If Stage 3's distributions are mixed across cells (broadly Frame C), the closing interpretation is conditional on which cells groks stably — a richer architectural characterisation.

§13.1 (closing decision) explicitly references Stage 3's distributions as part of the interpretation; CC carries the Stage 3 narrative summary forward into the final HANDOFF.

---

## 6. Stage 4 — PRE-D1a baseline collection at locked V2_TRAINING_STEPS (L_d-specific)

**Goal:** rebuild the bit-identical baseline distributions at the new training horizon, **for each L_d_main ∈ {1, 2, 4}**, replacing the invalidated PRE-D1a (`results/pre_d1a/`).

**Estimated wall-clock:** depends on lock value. At 100k lock, 20 runs per L_d × 3 L_d values = 60 runs:
- L_d=1: 20 × ~16 min = ~5.3 hr
- L_d=2: 20 × ~18 min = ~6.0 hr
- L_d=4: 20 × ~27 min = ~9.0 hr

Sequential total ~20 hr, ~14 hr at 2x.

**Why L_d-specific baselines.** The original Phase 1 sub-phase 1.2.5 found the magnitude=0 baseline is L_d-dependent (C1 medians 0.159 / 0.117 / 0.246 at L_d 1/2/4). Stages 7 and 10 evaluate cells at L_d ∈ {1, 2, 4}; using an L_d=1 baseline to classify L_d=2/4 cells would silently re-introduce the threshold-non-transfer issue that Phase 1 sub-phase 1.2 surfaced. Each L_d_main gets its own baseline distribution.

### 6.1 Configuration

Per the existing PRE-D1a spec (Phase 0.5 §11.7), extended to three L_d values:
- Cell: all-midpoint mag=0 (matches the C1 construction config)
- For each L_d_main ∈ {1, 2, 4}: n=20 bit-identical runs (seeds 0..19)
- Primary architecture only
- Diff_μ and Diff_σ recorded per run

### 6.2 Procedure

CC uses the existing `arm_runner.run_arm(...)` API from `v2/src/phase1/arm_runner.py`. No new code. The only parameter changes are `V2_TRAINING_STEPS` (read from `v2/config.py` which has been updated by Stage 1) and the L_d_main iteration.

For each L_d_main ∈ {1, 2, 4}:
  For each of seeds 0..19:
    - Construct the all-midpoint mag=0 substrate at that L_d_main
    - Train Primary at the locked steps
    - Evaluate Diff_μ and Diff_σ per the existing protocol
    - Record per-run `{seed, L_d_main, diff_mu, diff_sigma, wall_time, final_loss}` to `v2/results/recalibration/stage4/baseline_runs/L_d{N}_seed{NN}.json`

### 6.3 Aggregation and output

Aggregate to `v2/results/recalibration/stage4/baseline_distributions.json`:
```
{
  "lock_steps": <V2_TRAINING_STEPS>,
  "cell_config": {all-midpoint mag=0 config},
  "by_L_d": {
    "1": {"n": 20, "diff_mu": {median, IQR, p5, p95, raw: [...]}, "diff_sigma": {...}},
    "2": {"n": 20, "diff_mu": {...}, "diff_sigma": {...}},
    "4": {"n": 20, "diff_mu": {...}, "diff_sigma": {...}}
  },
  "L_d_dependence": {
    "diff_mu_medians": {"1": <m1>, "2": <m2>, "4": <m4>},
    "diff_mu_spread_ratio": <(max-min)/median over the three L_d medians>
  },
  "comparison_to_broken_pre_d1a": {
    "broken_diff_mu_median_L_d1": 0.0277,
    "new_diff_mu_median_L_d1": <m1>,
    "ratio": <m1 / 0.0277>
  }
}
```

The L_d_dependence block characterises whether the L_d-dependence found at broken mechanics (spread ratio implied by 0.159/0.117/0.246 in the original C1 data) persists at corrected training.

### 6.4 STOP condition

If the new L_d=1 baseline distribution's median or IQR is more than 10× larger than the broken PRE-D1a values, CC stops and surfaces — this would suggest the lock value introduces a new variance regime that needs to be understood before downstream stages proceed.

### 6.5 What Stage 5 consumes

Stage 5 (PRE-E τ_W calibration) operates on the L_d=1 baseline distribution as the primary input (matching the original PRE-E procedure). The L_d=2 and L_d=4 baselines are stored as inputs to Stages 7, 8, 9, and 10 — each stage selects the baseline matching its L_d_main.

### 6.6 HANDOFF

- Update `v2/HANDOFF.md` with Stage 4 outcome (per-L_d baseline statistics + L_d-dependence reading).
- Git commit: `recalibration stage 4: L_d-specific PRE-D1a baselines at lock=<value>`.

---

## 7. Stage 5 — PRE-E τ_W calibration

**Goal:** rebuild the per-head τ_W threshold at the new baseline distribution.

**Estimated wall-clock:** analytical from Stage 4's output. ~30 minutes (no new training).

### 7.1 Procedure

CC reuses the existing PRE-E calibration script (`v2/scripts/run_pre_e_calibration.py` or wherever it lives — verify before invoking). The input changes from the broken `results/pre_d1a/bit_identical_baseline.json` to the new `v2/results/recalibration/stage4/baseline_distribution.json`.

Per the PRE-E procedure (`results/pre_e/scaffolding_calibration.json` schema):
- One-sample Wilcoxon test per head against the new baseline
- p<0.05 calibration for τ_W
- Per-head mean μ and stddev σ computed

### 7.2 Output

Write `v2/results/recalibration/stage5/scaffolding_calibration.json` with the same schema as the original. `v2/config.py`'s `load_calibrated_thresholds()` is updated to point at this new file as the runtime source. The legacy `results/pre_e/scaffolding_calibration.json` is preserved on disk (not deleted) but is no longer referenced.

### 7.3 HANDOFF

- Update `v2/HANDOFF.md` with the new τ_W values.
- Git commit: `recalibration stage 5: PRE-E tau_W at lock=<value>`.

---

## 8. Stage 6 — PRE-D2 n-validation at L_d=2

**Goal:** re-validate whether n=10 vs n=20 changes the classification at the new training horizon.

**Estimated wall-clock:** depends on lock value. At 100k: 200 arm-runs × ~18 min (L_d=2) = ~60 hr sequential, ~42 hr at 2x. **This is the largest single stage.** Fork 2 (compute strategy) is likely re-decided before this stage begins.

### 8.1 Configuration

Per the original PRE-D2 (commits `63eb9c5, b2d0ec7`):
- 10 sweep points (each axis at its 2nd and 4th value, others at midpoint; §11.3 first-10-by-seed subsample)
- n=20 per point at L_d=2 intermediate
- Primary architecture
- Per-head classification: discriminably-working / discriminably-non-working / band-resident (the three-category extension per Phase 0.5 commitment)

### 8.2 Procedure

CC uses the existing PRE-D2 runner. The only changes:
- `V2_TRAINING_STEPS` from Stage 1
- Baseline + τ_W from Stage 5 (the new ones)

CC writes per-run results to `v2/results/recalibration/stage6/d2_runs/` and aggregates to `v2/results/recalibration/stage6/n_validation_report.json` with the same schema as the original PRE-D2 report.

### 8.3 Pre-Stage-6 SURFACE checkpoint (HARD)

Stage 6 launch is a **hard checkpoint** — CC MUST receive explicit experiment-chat confirmation in a fresh session before launching. The combination of Stage 6's wall-clock (potentially 40+ hours at 2x local) and its position as the test that re-validates the n=10 commitment makes "let's see how it goes" not an option here.

Before launching Stage 6, CC writes to `HANDOFF.md` a section titled **"Stage 6 launch checkpoint — HARD"** containing:

- Estimated wall-clock at locked steps on local 2x
- Confirmation that Stages 1–5 completed without STOP
- The new L_d=2 baseline + τ_W values being used (from Stages 4–5)
- **If lock value > 100k:** a Fork 2 recommendation (local 2x vs vast.ai) based on the concrete wall-clock estimate. At lock > 100k, ~50+ hour Stage 6 makes the rental case strong; at lock > 150k, local Stage 6 becomes infeasible and rental is essentially required.
- **If lock value ≤ 100k:** local 2x is feasible; Fork 2 recommendation may still favour rental for total cascade speed but local is a viable option.

CC then halts. The Fork 2 decision is made by the experiment chat at this checkpoint if it wasn't already decided. CC does NOT auto-proceed to Stage 6 under any circumstance. Stage 6 launch resumes in a fresh session with explicit go-ahead in HANDOFF.md or the session brief.

### 8.4 Reading the result

Pre-registered outcome categories (matching the original PRE-D2):
- **Band-residence comparable to broken PRE-D2 (~50–60%):** the n=10 vs n=20 distinction is again variance-limited. n=10 retained per Phase 0.5 commitment (no change).
- **Band-residence materially lower (< 30%) at n=20:** n=20 provides resolution at corrected training that n=10 doesn't. Phase 0.5 sample-size commitment is reviewed.
- **Band-residence materially higher at n=20:** anomalous — STOP and surface to experiment chat.

### 8.5 HANDOFF

- Update `v2/HANDOFF.md` with Stage 6 outcome and the n-validation reading.
- Git commit: `recalibration stage 6: PRE-D2 n-validation at lock=<value>`.

---

## 9. Stage 7 — Phase 1 controls (C1/C2) re-run

**Goal:** re-run the Phase 1 controls at corrected training to confirm the threshold-non-transfer finding (the surface that triggered the broken-mechanics investigation) persists, vanishes, or changes shape.

**Estimated wall-clock:** at 100k: 120 arm-runs × ~16 min (per existing 177s × 10) = ~32 hr sequential, ~22 hr at 2x.

### 9.1 Configuration

Per Phase 1 sub-phase 1.2 (commit `f10873f`):
- C1: magnitude=0 streams at mid Phase-1 config across L_d ∈ {1, 2, 4}, n=20 per L_d
- C2: perturbed streams at locality=0.9 across L_d ∈ {1, 2, 4}, n=20 per L_d
- Same construction grid (locked in `v2/src/phase1/sweep_grid.py`)

### 9.2 Procedure

CC uses the existing controls runner (`v2/src/phase1/controls.py`). No code changes. Inputs are `V2_TRAINING_STEPS`, baseline, and τ_W from Stages 1, 4, 5 respectively.

Output to `v2/results/recalibration/stage7/controls/{c1_report.json, c2_report.json}`.

### 9.3 Reading the result

Per Phase 1 §7 logic:
- **Threshold non-transfer persists at corrected training:** confirms the per-config baselining design (Stage 8 will re-validate Option 1 vs Option 2). Proceed to Stage 8.
- **Threshold non-transfer absent at corrected training:** the broken-mechanics regime was the source. Per-config baselining may not be necessary anymore. Surface to experiment chat; Stage 8 still runs but its interpretation changes.
- **C1 medians cluster around the new baseline:** the per-config caveat from §7 of PHASE1_PROGRESS resolves under corrected training. Record explicitly.

### 9.4 HANDOFF

- Update `v2/HANDOFF.md` with C1 and C2 results.
- Git commit: `recalibration stage 7: Phase 1 controls at lock=<value>`.

---

## 10. Stage 8 — Phase 1 baseline-variance diagnostic re-run

**Goal:** confirm whether per-config baselining (Option 1) remains necessary at corrected training, or whether a simpler scheme (Option 2: midpoint baseline) suffices.

**Estimated wall-clock:** at 100k: 40 arm-runs × ~16 min = ~10.7 hr sequential, ~7.4 hr at 2x.

### 10.1 Configuration

Per Phase 1 sub-phase 1.2.5 (commit `c954ee1`):
- Mid cross, L_d=1, magnitude=0
- D-axis {4, 16, 128} + continuity-axis {0.077, 0.4, 0.8}
- n=10 per point (the midpoint is shared, so 40 new runs covers both axes)

### 10.2 Decision rule (unchanged)

Per the original diagnostic:
- Relative spread `(max − min) / median` over the 3 position-medians on each axis
- `≤ 0.25 → Option 2` (midpoint baseline sufficient)
- `≥ 0.35 → Option 1` (per-config baseline needed)
- `(0.25, 0.35) → surface for design-chat`

### 10.3 Output

Write `v2/results/recalibration/stage8/baseline_variance_diagnostic.json` with the same schema as the original (the D=128/P=256 boundary caveat is preserved verbatim if applicable).

### 10.4 HANDOFF

- Update `v2/HANDOFF.md` with the decision (Option 1 confirmed / Option 2 now suffices / surface).
- Git commit: `recalibration stage 8: baseline-variance diagnostic at lock=<value>`.

---

## 11. Stage 9 — Phase 1 pilot L_d=1 re-run

**Goal:** the central deliverable. The pilot at corrected training is the artefact that determines whether v2 closes on pilot (Fork 4 Option A) or whether main effects are warranted.

**Estimated wall-clock:** at 100k: 240 arm-runs × ~16 min = ~64 hr sequential, ~44 hr at 2x.

### 11.1 Configuration

Per Phase 1 sub-phase 1.3 (commit `8b88c78`):
- 14 cells at L_d=1: magnitude axis {0.1, 0.3, 0.7, 0.9} at mid cross + continuity/dim depth at mid cross
- Per-config baselines at n=20 (5 unique baseline configs after magnitude collapses)
- Cells at n=10
- Per-K instrumentation (preserve per-K Diff_μ in per-cell forensics)
- Paired-baseline analysis (same-seed paired)

### 11.2 Procedure

CC uses the existing pilot harness from `v2/src/phase1/`. No code changes. Inputs: `V2_TRAINING_STEPS` (Stage 1), per-config baselines (computed inline per Stage 10 decision), τ_W (Stage 5).

Output to `v2/results/recalibration/stage9/pilot/` with the same structure as the original.

### 11.3 Pre-registered outcome categories

Per the original pilot's reading:
- **Signal density ≥ 25% (K-aggregated): capacity_signal.** Pilot demonstrates a discriminable working region exists. Main effects justified (Fork 4 Option B becomes viable).
- **Signal density 5–25%: marginal.** Some signal; main effects may or may not be warranted. Surface to experiment chat with cell-level forensics.
- **Signal density < 5%: robust_null.** Pilot replicates the broken-mechanics pilot's reading at corrected training. The variance-limited finding is real. Fork 4 Option A (close on pilot) executes; main effects are NOT run. Closing trail records the architectural finding.

### 11.4 Substantively interesting comparisons

Beyond the pre-registered categories, the pilot's report includes:
- Per-config baselining comparison: does it still produce different conclusions than midpoint baselining at corrected training?
- Paired-baseline re-analysis: does same-seed paired differencing now resolve cells that K-aggregated doesn't?
- K-aggregation refinement: does per-K=15 (or K=8 per the loose thread in session brief §12) reveal signal that K-aggregated swamps?
- Inter-seed pattern: do high-cosine seeds (per Stage 3's Frame A) cluster on the same cell classifications? (Cross-references the unviable-egg findings.)

These are reported descriptively in the per-cell forensics; they are not pre-registered gates.

### 11.5 Pre-Stage-9 SURFACE checkpoint

Before launching Stage 9, CC writes a **"Stage 9 launch checkpoint"** to `HANDOFF.md` mirroring §8.3's structure. Session boundary.

### 11.6 HANDOFF

- Update `v2/HANDOFF.md` with pilot outcome and category reading.
- Git commit: `recalibration stage 9: Phase 1 pilot L_d=1 at lock=<value> - <category>`.

---

## 12. Stage 10 — Phase 1 pilot L_d=2/4 capacity extension re-run

**Goal:** confirm the L_d=1 pilot's reading generalises across decoder capacity, or surfaces capacity-specific signal.

**Estimated wall-clock:** at 100k: 480 arm-runs (240 each at L_d=2 and L_d=4) × ~18 min avg = ~144 hr sequential, ~100 hr at 2x.

### 12.1 Stage 10 launch is always experiment-chat-decided

**CC ALWAYS halts after Stage 9.** Stage 10 is never auto-launched. The launch decision is made by the experiment chat based on Stage 9's outcome, captured in a fresh session.

Decision context for the experiment chat:
- **Stage 9 produced robust_null:** Stage 10 confirms whether the null generalises across decoder capacity, OR is treated as redundant (capacity extension was decided when the original Phase 1 thought the L_d=1 result was real; at corrected training with the unviable-egg result available, Stage 10's marginal information may be lower).
- **Stage 9 produced capacity_signal:** Stage 10 runs to identify which capacity values carry the signal — this becomes part of the substantive finding.
- **Stage 9 produced marginal:** Stage 10 may be the disambiguator, or the experiment chat may decide to close on Stage 9 with a marginal-finding closing trail.

Before halting, CC writes a **"Stage 10 launch checkpoint"** to `HANDOFF.md` containing:
- Stage 9's outcome (category + key numbers)
- Estimated Stage 10 wall-clock at the locked steps
- CC's read of which decision context above applies
- Outstanding Stage 9 forensics that might inform the launch decision

### 12.2 Configuration (if launched)

Per Phase 1 sub-phase 1.4 (commits `f0b6403, d9da253`):
- Same 14-cell + 5-baseline structure, with L_d-specific baselines (per the L_d-dependence confirmed in 1.2.5)
- L_d=2: 240 runs; L_d=4: 240 runs
- Per-K eval preserved

### 12.3 Output and HANDOFF

Write `v2/results/recalibration/stage10/pilot_Ld{2,4}/` per the original structure. Update `HANDOFF.md`. Git commit: `recalibration stage 10: capacity extension at lock=<value>`.

---

## 13. Output and Closing Trail

### 13.1 Recalibration phase concludes when

Stage 9 lands AND the experiment chat has decided on Stage 10. At that point:
- CC writes `v2/RECALIBRATION_HANDOFF.md` summarising the cascade outcome
- The recalibration phase commits to the closing decision per Fork 4. The closing decision is informed by **both** the pilot outcome AND Stage 3's inter-seed distributions:
  - **Pilot capacity_signal + any Stage 3 reading:** main effects warranted; recommend as v3+ scope (per Fork 4 Option A: v2 closes after the recalibration cascade; main effects beyond it are v3+)
  - **Pilot robust_null + Stage 3 distributions suggest convergent seeds:** v2 closes on pilot. Closing trail reads "broken mechanics fully explained the prior variance-limited null; at corrected training, seeds converge and the architecture still doesn't show a discriminable working region — finding stands as architectural."
  - **Pilot robust_null + Stage 3 distributions suggest divergent seeds:** v2 closes on pilot with the *stronger* architectural reading. Closing trail reads "broken mechanics + a real architectural variance limit, stacked; the prior null was both, and the architecture exhibits inter-seed function divergence at corrected training."
  - **Pilot marginal + any Stage 3 reading:** experiment chat decides; Stage 3's distributions are part of the deciding context.
- Stage 3's narrative summary is reproduced verbatim into `RECALIBRATION_HANDOFF.md` to ensure the closing-document drafter has it in hand alongside the pilot's reading.

### 13.2 Closing-document handoff

The closing document itself is drafted in a separate session per CODING_STANDARDS §3.1. This recalibration phase produces inputs for the closing trail, not the closing document.

Inputs prepared for closing:
- Stage 1 grok-detection methodology and the mean-head-aware plateau criterion (new methodology contribution)
- Stage 2 sanity-battery gate (new institutional discipline, formalised)
- Stage 3 unviable-egg test result (Frame A/B/C — substantive architectural finding)
- Stages 4–6 corrected PRE-D1a/E/D2 (replaces invalidated values in the v2 results tree)
- Stages 7–10 pilot at corrected training (the central deliverable)
- The 28 closing items already accumulated in session brief §9 carry forward

### 13.3 What v2 will NOT produce regardless of outcome

Per Fork 4 Option A and the session brief:
- Phase 1 main effects (1350 arm-runs) — deferred to v3+ if pilot signals
- Phase 2 (coherent-trajectory-variant perturbations) — out of v2 scope
- Cross-encoder generalisation (V-JEPA, ViT, etc.) — out of v2 scope
- A standalone PRE-A re-run as a full re-implementation. Per §1.2, PRE-A's construction-sanity sweep is uncorrupted; only `V2_TRAINING_STEPS` calibration changes, and that is exactly what Stage 1 produces with the new mean-head-aware criterion. Any older note referring to "PRE-A redo" is subsumed by Stage 1.

---

## 14. Scaffolding Inventory

Per CODING_STANDARDS §2.5 and research_operations §7.2, every fixed parameter introduced in this phase is labelled.

| Item | ARCHITECTURE / SCAFFOLDING | Removal plan |
|---|---|---|
| `V2_TRAINING_STEPS` value (set in Stage 1) | SCAFFOLDING | Removed when continuous-online deployment subsumes "training horizon" as a concept. Until then, a single fixed value stands in for "the predictor has reached productive learning regime." Phase 1 spec acknowledges this; v3+ continues. |
| Mean-head-aware plateau criterion (Stage 1, §3.4) | ARCHITECTURE (replaces NLL-plateau) | This is the corrected criterion, not a stand-in. Stays. |
| Sanity battery as Layer-1 gate (Stage 2) | ARCHITECTURE (new institutional discipline) | Stays. Formalised in v3+ as a pre-experiment requirement. |
| Three cells × n=10 for unviable-egg test (Stage 3) | SCAFFOLDING | The cell selection is illustrative-not-exhaustive. v3+ may run on a richer cell grid. The n=10 is a sample-size choice that may be re-validated if the distributions are noisy. |
| Stage 1 n=3 seeds per cell | SCAFFOLDING | The n=3 is the minimum needed to characterise per-cell seed variation conservatively (max-over-seeds with n=3 gives a worst-of-3 read). v3+ may sweep n if seed variation turns out to drive the lock value. |
| Stage 1 `max-over-cells × 1.1` global lock margin | SCAFFOLDING | The 1.1× is a residual buffer for un-probed cells and un-probed seeds. Removable once cell variance is better characterised by a denser grid. |
| Pre-committed interpretation frames A/B/C as guidance (§5.4) | ARCHITECTURE (interpretive framework, not gate) | The frames stay as the experiment-chat's thinking tool; CC does not auto-evaluate them. Per-frame numerical cutoffs would be SCAFFOLDING but are deliberately not introduced. |
| Lock value rounded to clean (50k/75k/.../200k) | SCAFFOLDING | Convenience; removable once cell variance is better characterised. |
| L_d-specific baselines at n=20 per L_d (Stage 4) | ARCHITECTURE (corrects for measured L_d-dependence) | Stays — Phase 1 sub-phase 1.2.5 established L_d-dependence; L_d-specific baselining is the corrected design. |

The inventory is reviewed at the end of the phase per research_operations §14.

---

## 15. Operational Discipline

### 15.1 Per CODING_STANDARDS and research_operations_v1

All discipline from those documents applies unchanged. Specifically:
- §8.1 Never kill a running training process
- §8.2 nohup for long-running scripts
- §8.3 Poll logs, do not wait on stdout
- §8.5 Away-mode (no clarifying questions; document decisions in HANDOFF)
- §8.10 STOP at gate failures unless explicitly overridden
- §9.2 Phase boundaries are session boundaries

### 15.2 Push hold

Preserved throughout. No `git push` under any circumstance.

### 15.3 HANDOFF.md updates

Every stage ends with a HANDOFF.md update. The recalibration phase's HANDOFF lives at `v2/HANDOFF.md` (the existing file); this file is updated in-place per stage, not replaced.

Per-stage HANDOFF section includes:
- What was attempted
- Pre-registered gate criteria and whether they passed
- Numbers traced to files (per CODING_STANDARDS §5.5)
- Surfaces / questions for experiment chat
- Next stage's launch decision (auto-proceed or surface checkpoint)

### 15.4 Numbers trace to files

Per CODING_STANDARDS §5.5 and research_operations §4. Every number in any HANDOFF section traces to a specific JSON file under `v2/results/recalibration/`. After context compaction, CC verifies numbers against raw output before quoting them (research_operations §9.6).

### 15.5 Git commits per task

Each stage produces one commit. Commit messages follow `recalibration stage N: <stage description> at lock=<value> - <PASS/STOP/category>`.

### 15.6 Single CC session per stage where feasible

Stages 1, 2, 3 likely fit in one session each. Stages 6, 9, 10 require multi-session execution; the long-running scripts launch via nohup, CC polls logs, and the stage's completion is detected at session boundary by checking for the output JSON.

### 15.7 Sub-agents for bounded tasks

For routine tasks (e.g., re-running an existing PRE-D2 analysis script with updated inputs), CC delegates to sub-agents with the minimum context they need (per CODING_STANDARDS §4.2).

### 15.8 The session brief is institutional context

The session brief (`WEFT_INNER_PAM_v2_SESSION_BRIEF.md`) is read at the start of this phase and at every session boundary that crosses a stage. Its §5 (broken-mechanics episode) is the load-bearing context for why this phase exists; its §6 (grokking finding) is the architectural framing CC carries forward into the unviable-egg test and the pilot interpretation.

---

## 16. STOP gates summary

| Stage | Gate | STOP condition |
|---|---|---|
| 2.1 | Frozen-tree check | `git diff 58e91d7 HEAD -- v0 v1 shared` non-empty |
| 2.2 | Canaries | Any test fails |
| 2.3 | Working tree | Uncommitted changes at batch start |
| 3.5 | Stage 1 lock | Any cell `lock_step_candidate > 175000` OR not-grokked-within-budget OR inter-cell spread > 4× |
| 4.3 | Stage 2 sanity | Any of S1/S2/S3 fail |
| 6.4 | Stage 4 baseline | New baseline > 10× broken baseline |
| 8.3 | Stage 6 pre-launch | Surface checkpoint (not a STOP per se; session boundary requiring experiment-chat confirmation) |
| 8.4 | Stage 6 reading | n=20 band-residence materially higher than n=10 (anomalous) |
| 9.4 | Stage 7 controls | (no automatic STOP; the read is surfaced for interpretation) |
| 11.5 | Stage 9 pre-launch | Surface checkpoint |
| 12.1 | Stage 10 launch | Experiment-chat decision required |

At any STOP, CC writes the STOP condition to HANDOFF.md, commits, and halts. Restart is via the experiment chat in a fresh session.

---

## 17. HANDOFF Protocol

### 17.1 Per-stage HANDOFF

At each stage's completion (or STOP), CC:
1. Writes all numbers to the per-stage JSON files under `v2/results/recalibration/stage{N}/`
2. Updates `v2/HANDOFF.md` with a new section for the stage
3. Verifies the numbers in HANDOFF match the JSON (per CODING_STANDARDS §5.5)
4. Commits with the stage-conventional message
5. Either auto-proceeds to the next stage (if no STOP / SURFACE checkpoint) or halts

### 17.2 End-of-recalibration HANDOFF

When Stage 10 completes (or is declined per §12.1), CC writes `v2/RECALIBRATION_HANDOFF.md` capturing:
- The lock value and Stage 1 evidence (including per-cell seed-variation observations)
- Stage 2 sanity-battery PASS confirmation
- Stage 3 unviable-egg per-cell distributions and CC's narrative summary (verbatim, not auto-assigned frames)
- Stages 4–6 new PRE-* values, with the L_d-specific baseline statistics surfaced
- Stages 7–10 Phase 1 pilot outcome (category, signal density, cell-level forensics)
- The closing interpretation per §13.1 (pilot reading × Stage 3 reading)
- Recommendations for the closing document (per §13.2 inputs list)
- Outstanding loose threads (anything surfaced but not resolved during the cascade)

This document is the input to the closing-document drafting session.

### 17.3 Session boundaries

Per CODING_STANDARDS §9.2 and §12.8, each stage is a session boundary. Long-running stages (6, 9, 10) may span multiple sessions; the boundary is the stage's completion, not arbitrary checkpoints within a stage. Per-session HANDOFF updates capture in-flight progress but the stage's verdict is written when the stage completes.

---

## 18. Drift Detection — Recalibration-Phase-Specific

In addition to the universal drift checks in research_operations §15, the recalibration phase carries these specific checks:

- **Did the lock criterion stay mean-head-aware?** Any reversion to NLL-plateau detection is a drift to the broken-mechanics regime. Flag immediately.
- **Did the sanity-battery gate get bypassed?** Stage 2 PASS must precede Stage 3+ work. Skipping Stage 2 for compute savings is the failure mode the broken-mechanics episode established as the institutional lesson — do not repeat.
- **Did per-config baselining get replaced silently?** Stage 8's diagnostic is the place where Option 1 vs Option 2 is re-evaluated. Any earlier stage that quietly uses midpoint-only baselining is a drift.
- **Did any pre-registered interpretation frame get post-hoc rationalised?** The categories in §9.3 and §11.3 are pre-committed; if the data lands between two categories and CC writes a narrative justifying the "more favourable" reading, that's the kind of drift adversarial review exists to catch. (Stage 3's frames are NOT pre-registered in the same sense — they are experiment-chat interpretive guidance; the drift to watch for there is CC quietly auto-assigning a frame instead of reporting the distribution and stopping.)
- **Did the closing-trail framing get committed prematurely?** Fork 4 Option A (close on pilot) is the lean; it's a *lean*, not a commitment. The experiment chat decides at the end of Stage 9. CC writing "v2 closes here" before the pilot reads is a drift.

---

## 19. Compute and Wall-Clock Estimate (placeholder)

This section is updated by CC at the end of Stage 1 with concrete numbers. Pre-Stage-1 placeholder (assuming 100k lock):

| Stage | Local 2x estimate | Cumulative | Notes |
|---|---|---|---|
| 1 | ~10 hr | 10 hr | n=3 seeds × 7 cells = 21 grok-curve runs |
| 2 | ~0.5 hr | 10.5 hr | Single trained predictor + sanity scripts |
| 3 | ~5 hr | 15.5 hr | 3 cells × 10 seeds = 30 training runs |
| 3.5 (conditional) | ~1 hr | 16.5 hr | Vast.ai cross-hardware smoke if Fork 2 → rental |
| 4 | ~14 hr | 30.5 hr | L_d-specific baselines (20 runs × 3 L_d_main) |
| 5 | ~0.5 hr | 31 hr | Analytical from Stage 4 |
| 6 | ~42 hr | 73 hr | PRE-D2 200 runs at L_d=2 — Fork 2 likely re-decided here |
| 7 | ~22 hr | 95 hr | Phase 1 controls C1/C2 |
| 8 | ~7 hr | 102 hr | Baseline-variance diagnostic |
| 9 | ~44 hr | 146 hr | Phase 1 pilot L_d=1 |
| 10 (conditional) | ~100 hr | 246 hr | Capacity extension L_d=2/4 |

**Local 2x without Stage 10:** ~6 days continuous. **With Stage 10:** ~10 days. The Fork 2 decision is re-made at §3.8 (post-Stage-1) and §8.3 (pre-Stage-6) — at 100k lock, Stage 6 alone is ~42 hours and rental becomes a strong case.

If the lock value lands at 50k instead of 100k, halve all estimates from Stage 3 onwards. If it lands at 150k, multiply by 1.5×.

---

## 20. Review Cycle Before Implementation Begins

Per research_operations §2.2, this document goes through:
1. **Primary reviewer** (Claude oracle chat or equivalent) — alignment with v2 institutional memory and the session brief
2. **Secondary reviewer** (different model — Grok, ChatGPT, or another instance)
3. **Resolution pass** — findings addressed or explicitly overridden

CC does not begin Stage 1 until both reviews are resolved and the experiment chat confirms.

---

*End of Recalibration Phase Experiment Instructions.*

*Source: design-chat session 2026-05-25 (this chat), implementing the four-forks resolution from `WEFT_INNER_PAM_v2_SESSION_BRIEF.md`.*
