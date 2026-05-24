# Weft Inner PAM v2 — Phase 1 PROGRESS / running handoff

**Status (2026-05-22):** Phase 1 first session. The §5 spec gap STOP was resolved
by the design chat; **sub-phase 1.0 (grid validation, added post-Phase-0.5 per the
design-chat-within-Phase-1) is complete and the §5 construction grid is locked in
code.** Now building the remaining sub-phase 1.1 modules. Push hold preserved; no
commit yet (working tree changes are this session's).

This file supersedes the earlier `PHASE1_STOP_HANDOFF.md` (the STOP it documented is
resolved; its content is folded in below).

---

## 1. Pre-launch gates — ALL GREEN (re-state)
Env Py3.12.3/torch2.10.0+cu128/CUDA12.8/GPU; v1 frozen at `58e91d7` (last commit
touching `v1/`); read-only trees clean; canaries v0 21/21, v1 51/51, combined 72/72,
PRE-D 11 arch assertions PASS, v2 Phase-0 37/37; `V2_TRAINING_STEPS`=10000.
Errata candidate: §3's `git -C v1 rev-parse` pin check assumes a nested repo; this is
a monorepo (verified intent the monorepo-correct way).

## 2. The §5 gap (resolved)
Original §5 grid/worked-example coordinate were in **measured-property space** (all
five worked-example values match PRE-B medians: cont 0.077, mag 0.017, loc 0.836,
D 13.75, P 360), but the substrate builds from **construction parameters**, with no
mapping provided (continuity especially: §5 gave measured C∈[0,1] but the knob is the
integer `continuity_center`). Design-chat resolution:
- **Sweep construction params, measure properties post-hoc**; the map keeps
  measured-property axis labels via `grid_mapping.nearest_grid_point` (closing reads
  the map in measurement space; Phase 1 executes in construction space).
- **D grid re-anchored {4,32,256} -> {4,16,128}** (see §4 Flag-1 provenance).
- **continuity_center calibrated and accepted as-is** (sub-phase 1.0).
- **(D=128, P=32) dropped -> not_characterised** (lone P<2D cell).
- magnitude_M {0.1,0.3,0.7} / locality_L {0.3,0.5,0.9} carry over as dials;
  period_P {32,256,2048} / manifold_dim_D {4,16,128} are construction.

## 3. Sub-phase 1.0 — grid validation (complete)
Script `scripts/run_phase1_grid_calibration.py` -> `results/phase1/grid_calibration.json`.
- **Feasibility (D{4,16,128}×P{32,256,2048}):** exactly one infeasible pair (128,32);
  all other 8 build cleanly; measured `D_global` tracks construction D (4→4.6, 16→17.6,
  128→143–150).
- **Stability at out-of-PRE-range extremes (D=128; P∈{32,2048}):** 5/5 short (2000-step)
  Primary runs trained stably (no NaN/Inf, flag=stable). Justifies the range extension.
- **Continuity calibration (continuity_center per period → §5 C-bins {0.077,0.4,0.8}):**
  | period(D) | low | mid | high |
  |---|---|---|---|
  | 32 (4) | c3 → C0.153 | c5 → C0.417 | c7 → C0.729 |
  | 256 (16) | c19 → C0.111 | c39 → C0.425 | c59 → C0.869 |
  | 2048 (128) | c146 → C0.105 | c309 → C0.421 | c472 → C0.877 |
  Caveats (recorded post-hoc per cell): **C=0.077 unreachable at the low cross**
  (P=32,D=4 floor 0.153); **boundary cells D==P//2 (16,32),(128,256) have uncontrollable
  continuity** (whole-band fill, C≈1.0) so a *held* continuity can't be honoured there.

## 4. Locked grid (verified)
`v2/src/phase1/sweep_grid.py` encodes the construction grid + cross-structure + the
continuity_center table + the (128,32) drop. Tests `v2/tests/phase1/test_sweep_grid.py`
**8/8 PASS** (grid values, feasibility, 135 main-effects cells = 45/L_d with 2 dropped =
43 feasible/L_d, anchor recurrence 5× each).

**Flag 1 — D-grid revision provenance** (recorded in `grid_calibration.json` →
`grid_revision_provenance` and `sweep_grid.py` docstring): the {4,32,256}→{4,16,128}
revision is a Phase-0.5 commitment **update via surface-and-confirm, not spec drift**.
Drivers: D=256 infeasible at P∈{32,256} (needs P≥512) — 3 infeasible (D,P) pairs in the
original grid (= 6 cross-structure cell-measurements/L_d) — and D=256 outside PRE's
validated D range. **Fact-check correction:** the drafted phrase "stays within PRE-A's
validated range" is inaccurate — D=128 is *above* PRE's validated D≤64; only D=4/D=16 are
in-range, and D=128 is validated separately by sub-phase 1.0's stability smokes.
Log-spacing: ×4 then ×8 (original was uniform ×8).

## 5. Closing items / process learnings (for the Phase 1 HANDOFF + closing doc)
- **Flag 2 — process improvement:** Phase-0.5 grid commitments were made without a
  construction-feasibility pre-check against the §5 primitive constraints. The
  design-chat-within-Phase-1 added sub-phase 1.0, which caught two issues: the P≥2·D
  constraint yields **3 infeasible (D,P) pairs (6 cross-structure cell-measurements/L_d)**
  in the original {4,32,256} grid, and the D=256 endpoint is outside PRE-A's validated
  range. **Recommendation:** future design-chat grid commitments should include a
  construction-feasibility + range pre-check against substrate primitive constraints
  before locking. (Aligns with §13 budget-reconciliation discipline.)
- Continuity 0.077 unreachable at the low cross (P=32,D=4); realized 0.153 — map records
  measured C, but the low-cross "smoothest" bin is ~2× the worked-example continuity.
- Boundary cells (D==P//2) cannot hold a target continuity (whole-band fill) — single-
  variable-discipline caveat for the dim/period sweeps; recorded post-hoc.
- Carried from §10/spec: L_d=2 baseline characterisation, K-aggregation in eval metric,
  D_global vs D_local discrepancy, BIC-without-effect-size, worked-example outside the
  magnitude grid (5–6× extrapolation), §9.3/§3.3 L_d_main spec inconsistency.

## 6. Sub-phase 1.1 — modules + smoke (COMPLETE, all PASS)
Modules: `arm_runner.py` (wraps PRE-D1a `train_one` + CLI for subprocess concurrency),
`classification.py` (PRE-D2 `classify_head` + three-category + §12.4 conflicting-head),
`parallel_harness.py` (subprocess pool + VRAM monitor + `measure_concurrency`).
Tests `v2/tests/phase1/` 12/12 (8 sweep_grid + 4 classification). Smoke
`scripts/run_phase1_smoke.py` -> `results/phase1/smoke_validation.json`:

- **step 3 reproducibility:** reproduced magnitude@0.3 L_d=2 seed0 Diff_μ =
  0.020366676151752472, **delta = 0.0 (bit-identical)**, tolerance = 0.11291 (n=10 CI
  half-width; the instructions' "≈0.063" is a mis-derivation — corrected). PASS.
- **step 6 eval semantics:** per-stream-point scalars confirmed (not per-(item,ordinal)). PASS.
- **step 5 classification:** all 10 PRE-D2 points × {n10,n20} replicate exactly
  (load_thresholds() == PRE-D2 stored thresholds). PASS.
- **step 4 parallelism:** n1=53.2s, n2=77.0s -> **ratio 1.447×**, VRAM 9119/16376 MB,
  no OOM. Above the 1.2× escalation gate (so 3x not attempted) and below the 1.5× STOP
  -> **parallelism LOCKED = 2x**. Effective throughput ≈ 2/1.447 = 1.38× vs serial.

**Budget-reconciliation note (§13):** the §7 table's ~32 hr-at-2x estimate assumed
≤1.2× overhead; the measured 1.447× means main-effects (1350 runs @ ~130s) ≈ 35 hr and
total Phase 1 ≈ ~45 hr at the locked 2x. To re-confirm at the 100-cell checkpoint (§7.3).

## 7. Sub-phase 1.2 controls — RUN COMPLETE; STOP AT GATE (threshold non-transfer)
120 arm-runs in 255 min at 2x (0 diverged). Observed single-run cost ≈177s (10000
steps, P=256), not the §7 table's 130s — full Phase 1 ≈ ~60 hr at 2x (budget item).

**Finding (fail-fast gate did its job):** the PRE-D1a-calibrated working threshold
(baseline 0.0277 + τ_W → **0.0890**) does **not transfer** to the Phase-1 construction config.
- **C1 (magnitude=0, should sit at baseline 0.0277):** Diff_μ medians **0.159 / 0.117 / 0.246**
  (L_d 1/2/4) — 4–9× baseline, above threshold. At L_d=4 the whole μ-CI [0.170,0.259]
  clears threshold → **μ-head `discriminably_working` on a non-perturbed stream**.
  Holds at L_d=1 too (0.159 vs 0.0277, same L_d as baseline) ⇒ **construction-config
  dependence** (Phase-1 mid P=256/center=39 vs PRE-D1a P=128/center=24), beyond the
  known L_d-dependence caveat (PRE-D2 baseline_caveat §11.7).
- **C2 (perturbed):** Diff_μ 0.017–0.209, comparable-to-LOWER than bit-identical C1, no
  monotonic magnitude dependence ⇒ Diff_μ here is **config-structure-dominated, not
  perturbation signal**.
- **Gate readings:** overall-cell = 0/3 working (script exit 0); **head-level = 1/6 = 16.7%
  > 15%** (trips §7.2 trigger); §7.2 calls C1 unexpected discriminably-working a
  "substrate or τ_W issue" STOP. Building the 1350-cell map vs a non-transferring
  threshold ⇒ confounded map. **STOPPED before main effects** (~7% of Phase-1 compute spent).

Reports: `results/phase1/controls/{c1_report.json,c2_report.json}` (+ 120 `_runs/*.json`).

**Resolution = design-chat decision (per §11/§2.1; τ_W not lowered to admit signal):**
(A) per-config bit-identical baseline (each cell's own magnitude=0 twin) — corrects the
confound, ~doubles arm-runs; (B) a baseline grid keyed by construction config + L_d
(recalibrate threshold per cross/L_d); (C) reconsider the Diff_μ indicator
(config-normalised); (D) accept + document as closing caveat (risks confounded map).

## 8. Sub-phase 1.2.5 — baseline-variance diagnostic -> Option 1 (decisive)
40 new runs (mid cross, L_d=1, magnitude=0; D-axis {4,16,128} + continuity {0.077,0.4,0.8};
midpoint reused from C1-L_d1). `results/phase1/baseline_variance_diagnostic.json`:

| axis | low | mid | high | rel. spread |
|---|---|---|---|---|
| D {4,16,128} | 0.2553 | 0.1590 | 0.0000 (D=128 boundary) | **1.61** |
| continuity {0.077,0.4,0.8} | 0.2840 | 0.1590 | 0.0041 | **1.76** |

max relative spread **1.76 ≫ 0.35 -> Option 1 (per-config bit-identical baseline), decisively.**
The magnitude=0 Diff_μ baseline is monotonically continuity-driven (~70× across the
continuity axis) and collapses to **0** at the D=128/P=256 boundary (continuity forced ≈1.0).
A single global threshold is invalid; each cell must be judged vs its own zero-perturbation twin.

**Option-1 scale (from the locked grid):** 129 feasible main-effects cells; **75 unique
magnitude=0 baseline configs** (magnitude axis collapses; avg 1.72 cells/baseline) -> 750
baseline runs @n=10. Main+baseline @n=10 ≈ **2040 arm-runs**. At the *measured* throughput
(~0.5 runs/min @2x; observed ~177 s/run, not the §7 table's 130 s) ≈ **~60–70 hr (~3 days)**,
not the 12–15 hr the rule's text assumed. Budget-reconciliation item.

**Design-chat must specify before step 5 (baseline collection):**
1. Baseline granularity: 75 configs keyed by (locality, continuity_center, period, dim, L_d),
   magnitude=0 (confirmed shared across the magnitude sweep). Baseline n (10 or 20)?
2. τ_W under per-config baselines: apply the existing per-head τ_W as the margin above each
   cell's own baseline median, or recalibrate τ_W per-config? (PRE-E τ_W was a Wilcoxon margin
   vs the single PRE-D1a baseline distribution.)
3. Working test: discriminably_working ⟺ cell Diff CI-low > (own-baseline median + τ_W)?
4. Boundary/degenerate cells: D=128 (and high-continuity) baselines ≈ 0 with ~0 variance
   (variance collapse) -> the working test may be degenerate/hypersensitive. Special handling
   or `not_characterised`?
5. σ head: per-config baseline for Diff_σ too (1.2.5 measured Diff_μ only)?

## 9. Option-1 PILOT (240 runs, L_d=1) — robust variance-limited NULL
14 cells (n=10) + 5 mag=0 baselines (n=20), per-K, in 390 min at 2x. Per-config baseline
classification (cell Diff CI-low > own-baseline-median + τ_W). `results/phase1/pilot/`.

**Signal density 0/14 discriminably-working — at K-aggregated AND k=15.** Categories: 9
band_resident, 1 discriminably_non_working, 4 baseline_degenerate (high cross:
cont-high/D=128/P=2048 -> σ variance-collapse, Diff_μ≈0).
- **Signal present but variance-limited, not absent:** mag0.9_low median **1.259** (≫ its
  threshold 0.850) but CV 0.42, CI [0.638, 1.504] straddles. 5/10 non-degenerate cells show
  a positive perturbation effect (cell > baseline).
- **R1 (mag=0.9) refuted:** large perturbation raises the low-cross median but not
  resolvability (variance still swamps).
- **R2 (k=15) refuted:** long horizon shows no more signal than K-aggregated.
- **Paired analysis refuted** (`paired_analysis.json`): cell−baseline at the same seed
  (shared init/trajectory/dropout) does NOT cancel the variance — paired CV stays 1.5–12+,
  0/10 resolvable, and the effect sign flips across seeds. ⇒ the limiter is
  training-trajectory divergence (data-driven), not sample size or shared-noise. Confirms
  & strengthens PRE-D2's variance-limited reading.

**Conclusion (scoped):** inner-PAM-in-isolation has **no discriminable working region** at
L_d=1 across the property space, robust to per-config baselines, magnitude (incl. 0.9),
prediction horizon (k=15), and paired variance-cancellation. Closing framing (design-chat):
this characterises inner-PAM-in-isolation; the outer associative memory is out of v2 scope.

**Caveat — capacity dimension untested:** the pilot is **L_d=1 only**. The full grid
includes L_d∈{1,2,4}; PRE-D2 (L_d=2) was variance-limited but had some discriminable
(mostly non-working) points. L_d=2/4 (decoder capacity) is NOT de-risked by this pilot.

## 10. Decision: Path B — L_d=2/4 capacity pilot extension
Design chat chose (B). De-risk the one open dimension (decoder capacity) before any
conclude/full decision, same 14-cell + 5-baseline structure, per-K eval.

Confirmations:
- **Baselines are L_d-SPECIFIC** (cannot reuse L_d=1's — the mag=0 baseline is L_d-dependent,
  cf. 1.2.5 C1 medians 0.159/0.117/0.246 at L_d 1/2/4). Each of L_d=2,4 gets its own 5
  baselines @ n=20. -> 240 runs/L_d, **480 total**.
- **Wall-clock (measured ratios L_d2=1.12×, L_d4=1.66× per step):** L_d=2 arm ≈7.2 hr,
  L_d=4 arm ≈10.1 hr -> **~17 hr total** sequential (not the 9–12 hr first guessed; 2x is the
  parallelism cap). Surfaced so the completion checkpoint expects ~17 hr.
- **Pre-registered outcome reading** (per arm, K-agg working fraction): <5% robust_null /
  5–25% marginal / >25% capacity_signal. Recorded in each `pilot_Ld{N}/pilot_report.json`.

Runner parametrised by `--l-d` (output namespaced `pilot_Ld2/`, `pilot_Ld4/`).

## 11. Capacity extension results — robust null ACROSS CAPACITY
L_d=2 (480 min) + L_d=4 (661 min) = ~19 hr actual (est ~17). All three arms:

| L_d | K-agg working | k=15 working | paired resolvable | categories (K-agg) | outcome |
|---|---|---|---|---|---|
| 1 | 0/14 | 0/14 | 0/10 | 9 band, 1 non-working, 4 degenerate | (matches) |
| 2 | 0/14 | 0/14 | 0/10 | 10 band, 4 degenerate | **robust_null (<5%)** |
| 4 | 0/14 | 0/14 | 0/10 | 8 band, 2 non-working, 4 degenerate | **robust_null (<5%)** |

Pre-registered reading hit: **L_d=2/4 robust null matches L_d=1 -> finding generalises
cleanly across decoder capacity.** The L_d=1 low-cross perturbation *elevation* (mag0.9_low
1.259 vs baseline 0.789) does NOT persist at L_d=2/4 (0.279 vs 0.319; 0.559 vs 0.593) — higher
capacity makes the null cleaner. The 4 baseline_degenerate cells are the high cross
(cont-high/D=128/P=2048, σ variance-collapse) at every L_d.

## 12. PHASE 1 SUBSTANTIVE FINDING (scoped)
**inner-PAM-in-isolation exhibits no discriminable working region** across the tested
property-space coverage (3 crosses spanning all five axes' low/mid/high; magnitude incl.
0.9; mid-cross continuity/dim depth) and **all three decoder capacities (L_d 1/2/4)**, robust
to: per-config bit-identical baselines (Option 1), long-horizon (k=15), and paired
variance-cancellation. The limiter is **training-trajectory variance** (data-driven,
seed-to-seed; not sample size, not config-baseline-offset, not capacity) — i.e.
**variance-limited**, generalising PRE-D2's single-point reading to the property space ×
capacity. Closing framing (design-chat): characterises inner-PAM-in-isolation; the outer
associative memory is out of v2 scope.

**Why the full ~3-day grid is not warranted:** the limitation is at the training-run level,
not a property-space-specific or capacity-specific effect, so the un-swept axes (full
locality/period sweeps) are expected to inherit the same null; C2 (locality=0.9) already
showed no help. Marginal information of the full grid is low against its cost.

## 13. (superseded) — HANDOFF was drafted to conclude with the variance-limited finding
Drafted `v2/PHASE1_HANDOFF.md`, then HELD for a mechanical sanity battery before commit.

## 14. MECHANICAL SANITY BATTERY — BROKEN MECHANICS (finding invalidated)
`scripts/run_phase1_sanity_battery.py` -> `results/phase1/sanity_battery.json` (drift
detection, research_ops §15: do the train+eval mechanics produce meaningful numbers? The five
pilot cross-checks all sat downstream of the same pipe and would converge identically whether
the null is real or the pipe is broken).

**Result: the predictor's mean head never learns to predict the trajectory.**
- S1 (fit own training stream): `cos(mean, target)=−0.011` — orthogonal to truth, on the
  TRAINING stream of a smooth period-256 loop.
- `cos(mean, last_frame)=−0.019` — does not even learn the trivial copy-last-frame; yet
  `cos(last_frame, target)=0.559` -> **the trivial baseline beats the trained predictor**.
- `‖mean‖=0.75` (not zero-collapsed; uninformative direction); σ=0.056; loss=−2917 — the
  loss "convergence" (S4 PASS) is the **variance head fitting the residual of an unlearned
  mean** = plateau-at-chaos, not plateau-at-learned.
- S2 (full-trajectory swap): err_familiar 0.999 ≈ err_novel 1.000 — insensitive because the
  output is uninformative regardless of input.
- S3: pred_error ≈1.0 on (a) training and (c) noise alike (FAIL); Diff_μ does vary with input
  (0.217 vs 0.002) but is measuring the variance of an untrained mean.
- S5: inter-seed output cos 0.124 — seeds settle into different uninformative functions.

**Implication:** the Phase-1 "variance-limited / no-working-region" result is an **ARTIFACT of
an untrained predictor**, not an architectural property. Diff_μ has measured variance-of-an-
untrained-mean throughout. This **implicates PRE-D1a (baseline), PRE-E (τ_W), PRE-D2** — every
v2 measurement built on this train+eval pipe. The HANDOFF draft is HELD/invalidated.

**Likely root cause (pre-registered S1-fail branch):** `V2_TRAINING_STEPS=10000` single-window
(batch=1) steps is far too few to train the ~22M-param v1 transformer; PRE-A's plateau detector
saw the *variance* head converge and called it done (plateau-at-unlearned-mean). Needs a
confirmatory longer-training test (train 1 predictor at e.g. 50k–500k steps and/or with
batching; check whether `cos(mean, target)` rises above the trivial 0.56).

## 15. (superseded by §16) — root cause was to be confirmed by a longer-training test

## 16. ROOT CAUSE CONFIRMED — training-length insufficient (mechanics are SOUND)
`scripts/run_phase1_training_length_test.py` -> `results/phase1/training_length_test.json`.
cos(mean,target) vs steps (mid mag=0 L_d=1, smooth period-256 loop; trivial predict-last=0.559):

| steps | 2k | **10k** | 25k | 50k | 100k | 150k | 200k |
|---|---|---|---|---|---|---|---|
| cos_k1 | +0.003 | **−0.014** | +0.033 | +0.397 | +0.934 | +0.945 | +0.953 |
| ‖mean‖ | 0.27 | 0.19 | 0.19 | 0.87 | 0.90 | 0.94 | 0.94 |

**The predictor learns the trajectory well (cos→0.95, ≫ trivial 0.56) — it just needs ~100k
steps.** Sharp learning onset 25k→50k (grokking-like delayed generalisation); plateau by ~100k.
`V2_TRAINING_STEPS=10000` sits in the pre-transition flat region (cos≈0) — exactly where every
v2 measurement trained. PRE-A's plateau detector fired on the VARIANCE head saturating (loss
−2901→−3013 over 2k→25k, "flat" in relative terms on a large NLL) while the MEAN head's learning
transition (loss −3104→−4236, 50k→100k) had not yet happened. Plateau-at-chaos, confirmed.

**Mechanics are SOUND** — substrate, architecture, eval metric, harness all work; only the step
count was wrong. The entire "variance-limited / no-working-region" result is fully explained as a
pre-learning-transition artifact.

## 17. Next immediate action — STOP, design-chat re-plan (NOT autonomous)
Decisions for the design chat (major implications):
- **Recalibrate `V2_TRAINING_STEPS`** to the learned regime (~100k; diminishing returns after).
  Redo PRE-A's plateau detection with a **mean-head-aware** criterion (cos(mean,target) /
  prediction-error threshold), NOT just NLL-plateau — the variance head saturates first and
  masks the mean head.
- **Re-run everything training-dependent** at the new step count: PRE-A (step calibration),
  PRE-D1a (baseline), PRE-E (τ_W), PRE-D2, and ALL of Phase 1.
- **NOT affected** (no predictor training): PRE-B (worked-example measurement), PRE-C (arch
  assertions), PRE-A's construction-sanity sweep, the substrate/property/protocol modules.
- **Budget:** ~10× compute per run (100k vs 10k steps). Re-plan scale accordingly; per-config
  baselines + pilot-first + the sanity battery remain the template.
Phase 2/3 hard-gate stands. HANDOFF held (invalidated). Push hold preserved.
