> # ⚠️ HELD / FRAMING INVALIDATED (2026-05-24)
> The mechanical sanity battery (`results/phase1/sanity_battery.json`,
> `PHASE1_PROGRESS.md` §14) found the **predictor's mean head never learns** (cos(mean,target)
> ≈ 0, worse than the trivial predict-last-frame baseline at 0.56). The "variance-limited /
> no-working-region" finding below is an **artifact of an untrained predictor**, NOT an
> architectural property. **DO NOT COMMIT / SHIP this framing.** This draft is retained only
> as a structural template for a corrected HANDOFF once the training defect is resolved.
> All §-references to a "real architectural finding" are superseded.

---

# Weft Inner PAM v2 — Phase 1 HANDOFF

**Status: Phase 1 CONCLUDED** (2026-05-24). Outcome below is a *pivot-and-conclude*, not the
planned full-grid sweep: the §7.2 controls gate caught a methodology issue (threshold
non-transfer), which — once corrected (per-config baselines) and de-risked (pilot-first +
capacity extension) — produced a robust finding that made the full grid unwarranted.

This HANDOFF is the §10 structured handoff: it surfaces what the v2 closing document must
address and points to artifacts. **It is not the closing document** (that is a separate
writing task). Running record: `v2/PHASE1_PROGRESS.md`. Push hold preserved throughout.

---

## 1. Phase 1 outcome

The planned sub-phases 1.3 (main effects 1350), 1.4 (L_d sweep), 1.5 (corner probe), 1.6
(reallocation), 1.7 (full map) were **not run as specified**. Instead:

- **1.0–1.1** built and validated the harness (grid locked, smoke all-PASS, 2x parallelism).
- **1.2 controls** surfaced a **STOP**: the single PRE-D1a-calibrated working threshold does
  not transfer across construction configs (a magnitude=0 stream sits at 0.12–0.25, not the
  0.028 baseline). Caught at ~7% of planned compute — fail-fast as designed.
- **Pivot to per-config baselines (Option 1)**, chosen by a decisive baseline-variance
  diagnostic (1.2.5: within-cross baseline spread 1.76).
- **Pilot-first de-risking** (L_d=1) + **capacity extension** (L_d=2,4) instead of the full
  collection.

**Result: the architecture–trainer combination exhibits no discriminable working region**
across the tested coverage and all three decoder capacities. The full grid would have
confirmed the same null at ~3 days' cost (see §5). Phase 1 concludes here.

---

## 2. Per-sub-phase trace (commit / arm-runs / artifacts / finding)

| sub-phase | commit | training arm-runs | artifacts | key finding |
|---|---|---|---|---|
| 1.0 grid validation | `7aa20cd` | ~5 stability smokes (+~45 build/measure) | `results/phase1/grid_calibration.json`; `src/phase1/sweep_grid.py` | §5 grid is measured-property space; construction-param resolution; D{4,32,256}→{4,16,128}; (128,32) dropped; continuity_center calibrated |
| 1.1 smoke | `fd1c519` | ~4 | `results/phase1/smoke_validation.json` | reproduces PRE-D2 mag@0.3 **bit-identically**; classification replicates PRE-D2; **2x locked** (1.447×) |
| 1.2 controls C1/C2 | `f10873f` | 120 | `results/phase1/controls/{c1,c2}_report.json` + 120 `_runs/` | **STOP: threshold non-transfer** (config-dependent baseline) |
| 1.2.5 baseline-variance | `c954ee1` | 40 | `results/phase1/baseline_variance_diagnostic.json` | within-cross spread **1.76** → **Option 1 decisive** |
| pilot machinery | `6f7a51e` | (validation) | `src/phase1/eval_perk.py` | per-K eval + `train_one_perk` (bit-identical aggregate) |
| L_d=1 pilot | `8b88c78` | 240 | `results/phase1/pilot/{pilot_report,paired_analysis}.json` + 240 `_runs/` | **0/14 working**; variance-limited (signal present, unresolvable) |
| parametrise | `f0b6403` | — | — | `--l-d` arg; L_d-specific baselines; pre-registered outcome reading |
| L_d=2/4 extension | `d9da253` | 480 | `results/phase1/pilot_Ld{2,4}/...` + 480 `_runs/` | **robust_null across capacity** |

Total training arm-runs this phase ≈ **890** (controls 120 + baseline-var 40 + pilots 720 +
~10 smokes/probes). All eval is per-stream-point (not v1's per-(item,ordinal)).

---

## 3. The finding (the "map" that wasn't built)

No full §3.5 working-region map was produced (it would have been near-uniformly
band_resident / degenerate). The substantive output is the three-L_d pilot:

| L_d | K-agg working | k=15 working | paired resolvable | categories (K-agg, of 14) |
|---|---|---|---|---|
| 1 | 0/14 | 0/14 | 0/10 | 9 band, 1 non-working, 4 degenerate |
| 2 | 0/14 | 0/14 | 0/10 | 10 band, 4 degenerate |
| 4 | 0/14 | 0/14 | 0/10 | 8 band, 2 non-working, 4 degenerate |

Coverage: 3 crosses spanning all five axes' low/mid/high; magnitude {0.1,0.3,0.7,**0.9**};
mid-cross continuity/dim depth; per-config bit-identical baselines; per-K (k=1/8/15); paired
analysis; L_d∈{1,2,4}. The 4 degenerate cells per arm are the high cross
(cont-high/D=128/P=2048, σ variance-collapse). Pointers: the six `pilot*/` report JSONs.

---

## 4. Substantive finding — framing (architecture–trainer, NOT "missing component")

**Supersede the earlier "inner-PAM-in-isolation / outer memory missing" vocabulary.** That
framing wrongly implied an absent component. The correct framing:

> The **architecture–trainer combination**, under **continuous online training** on the §5
> synthetic substrate, produces **seed-trajectory variance that dominates the perturbation
> signal** across the tested capacity range (L_d 1–4) and property regimes. The associative
> memory *is* the trained weights; the finding is about **how reliably those weights converge
> to register input perturbations across seeds** — and they do not, at this configuration.

Evidence it is training-trajectory variance (not sample size, baseline-offset, or capacity):
per-config baselines remove the offset and the null persists; n=20 baselines + PRE-D2's
n=10→20 finding rule out sample size; L_d 1/2/4 rule out capacity; **paired** (same-seed
cell−baseline, shared init/dropout) does **not** cancel the variance (paired CV 1.5–12+, sign
flips) → the variance is the *data-driven divergence of the training trajectory itself*.

**Deployment reading (for closing's central narrative).** Under the continuous-online-training
design vision there is no train/deploy split — Weft is always training. The variance-limited
finding then reads as: *each Weft deployment is one trajectory through learning; the population
of possible trajectories under this trainer reaches meaningfully different prediction-quality
endpoints on the same experience stream.* The "unviable egg" intuition is the right one — some
instances would converge into useful agents, others would not, on identical experience. This
is a substantive **deployment concern** for v3+ to address before fielding, not merely an
academic methodology result. This should be the closing document's spine.

---

## 5. Methodology contribution (working as designed — the secondary contribution)

Pilot-first + fail-fast discipline reached the conclusion with ~**880 diagnostic arm-runs**
(controls 160 + pilots 720) instead of the full per-config collection of ~**2,800** runs
(1,290 cells @ n=10 + 1,500 baselines @ n=20 across 3 L_d; ≈3 days at 2x) — *the full grid
would have reached the same null.* (The earlier ~3,530 figure was a one-baseline-per-cell
upper estimate.) This is methodology working as designed, not luck.

**Five orthogonal cross-checks, all converging on the same null** — a reusable template for
future characterisation work:
1. per-config bit-identical baselines (removes the config-baseline-offset confound),
2. per-K instrumentation (k=1/8/15 — tests horizon-dependence; refuted K-aggregation as a
   signal-swamp),
3. paired analysis (same-seed differencing — isolates the variance mechanism),
4. high-magnitude probe (mag=0.9 beyond grid — tests perturbation-ceiling),
5. capacity extension (L_d 2/4 — tests decoder-capacity-dependence).

---

## 6. v3+ design pointers (enumerated from the data; NOT committed scope)

The finding points at four addressable directions (closing should enumerate, not choose):
1. **Trainer determinism / seed-variance reduction** — weight/stochastic-weight averaging,
   longer `V2_TRAINING_STEPS`, different optimiser. The most directly addressable: the limiter
   is training-trajectory divergence.
2. **Architectural revision** — `PREDICT_K`=16 may be too short for trajectory-recognition to
   engage; `WINDOW_W`; or a non-transformer prediction structure. (K-aggregation per se is
   *not* the issue — k=15 refuted that — but K being too short overall is not ruled out.)
3. **Substrate-perturbation type** — the §5.3 primitive is "noisy departure from base
   trajectory," not "alternative coherent sub-trajectory." Real environments produce both;
   testing the alternative type would tell whether the variance limit is intrinsic or
   substrate-specific.
4. **Encoder dynamic range** — DINOv2's worked example sits at magnitude **0.017**; the
   architecture does not register clearly even at mag **0.9**. The encoder–architecture
   coupling needs work from both sides, not encoder-shopping alone.

---

## 7. Closing items accumulated (for the v2 closing document)

- **Threshold non-transfer** (1.2): single global baseline+τ_W invalid; per-config baselines
  required (the central methodology correction).
- **Degenerate-cell handling**: σ variance-collapse baselines (high cross / D=128 / high
  continuity) → `baseline_degenerate`; the working test is undefined there.
- **Budget reconciliation**: measured ~**177 s/run** (vs §7 table's 130 s); 2x parallelism
  overhead **1.447×** (not ≤1.2×, so 3x not viable); full Phase 1 would have been ~3 days,
  not the table's ~32 hr.
- **§3 monorepo-pin errata**: the `git -C v1 rev-parse HEAD = 58e91d7` check assumes v1 is its
  own repo; it is a subdir — intent verified the monorepo-correct way (last commit touching
  `v1/` is `58e91d7`, tree clean).
- **§7.1 tolerance mis-derivation**: instructions' "≈0.063 = half the CI [0.020,0.246]" is
  inconsistent (half-width is **0.113**; 0.063 ≈ τ_W_mu). Smoke used 0.113.
- **§9.3 / §3.3 L_d_main spec inconsistency** (errata candidate, carried from Phase 0.5).
- **K-aggregation as a methodology dimension** (refuted as a confound source here, but worth
  flagging for future work that varies K).
- **Worked-example magnitude extrapolation**: worked-example mag **0.017** is below the grid
  minimum 0.1 (5–6× gap on the most decision-relevant axis); the map characterises *near*, not
  *at*, the worked example.
- **BIC-without-effect-size** threshold methodology refinement (carried from spec §11.7).
- **D_global vs D_local discrepancy** (carried; D_global tracks construction D, D_local
  underestimates).
- **Substrate (D, P, continuity) coupling at feasibility edges**: `centered_harmonics` needs
  `D ≤ P//2`; at the D==P//2 boundary continuity is uncontrollable (whole-band fill); drove
  the D-grid revision and the (128,32) drop.
- **"Unviable egg" → architecture–trainer characterisation** reframing (§4 above) — the
  closing's spine.

---

## 8. Discipline preserved (institutional evidence)

- **109 inheritance/regression canaries green throughout** (v0 21 + v1 51 + v2 Phase-0 37),
  plus the v1 PRE-D 11 architectural assertions, across multiple sessions and ~real wall-clock
  days. +12 new Phase-1 tests (v2 suite now 49/49).
- **Frozen-tree discipline held**: `v0/`, `v1/`, `shared/`, and `v2/src/{substrate,
  property_measure,protocol,preflight}/` were 0-changed at every sub-phase boundary. The
  **library-import model held under load**.
- **v1 frozen at `58e91d7`** throughout. **Push hold preserved** (local commits only).

---

## 9. Working-tree state & canary status (at HANDOFF)

- Git clean except this file (`v2/PHASE1_HANDOFF.md`, untracked, awaiting review) and
  untracked `logs/`. Commits `7aa20cd … d9da253` on `main`, unpushed.
- Canaries: v2 **49/49**; frozen trees **0-changed**; PRE-D arch assertions PASS.

---

## 10. Next immediate action (Phase 2/3 hard-gate respected)

CC does **not** initiate Phase 2/3 work — there is no Phase 2/3 instructions document in this
session's scope. Next moves are design-chat work, out of scope here:
(a) write the **v2 closing document** (spine per §4; methodology per §5; items per §7),
(b) define **v3 scope**, (c) choose which §6 v3+ direction(s) to pursue.

After this HANDOFF is reviewed, that is the handoff to the design chat.
