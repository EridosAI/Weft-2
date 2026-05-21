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

## 6. Next immediate action
Build remaining sub-phase 1.1 modules: `arm_runner.py` (thin wrapper over the proven
PRE-D1a `train_one` + its per-stream-point `compute_diff_metrics`), `classification.py`
(PRE-D2 `classify_head`/`classify_point` + three-category aggregation + §12.4
conflicting-head rule), `parallel_harness.py`, then the §7.1 smoke validation (reproduce
PRE-D2 `magnitude@0.3` L_d=2 seed0 = 0.020366676151752472 within tolerance; classification
replicates PRE-D2; 2x/3x parallelism). Then the §1.2 controls fail-fast gate before main
effects. Regression canaries green; push hold preserved.
