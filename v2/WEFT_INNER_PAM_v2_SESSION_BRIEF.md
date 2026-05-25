# Weft Inner PAM v2 — Session Brief for Fresh Design Chat

**Purpose.** This brief captures the design-chat context accumulated across multiple sessions of Phase 0.5 and Phase 1 design-chat work, so a fresh chat can resume design-chat responsibilities without re-reading the prior session arc. It is written for a Claude instance assuming the design-chat role.

**Scope.** This brief covers the design decisions, methodology innovations, broken-mechanics discovery, and substantive findings accumulated through the end of the Phase 1 pilot diagnostic battery. It does **not** cover the v2 spec or Phase 0 instructions themselves — those are in project knowledge and should be read directly.

**Status at handoff.** Phase 1 measurements (controls, baseline-variance diagnostic, L_d=1 pilot, L_d=2/4 capacity extension) are invalidated as broken-mechanics artifacts. PRE-A's V2_TRAINING_STEPS calibration is the root cause defect. All v2 measurements that depend on trained predictors need to be re-run after V2_TRAINING_STEPS is recalibrated. Substantial design-chat decisions are pending.

---

## 1. The architectural design vision (calibration for the design-chat role)

Weft is a continuous-trajectory Predictive Associative Memory system inspired by Vannevar Bush's associative trails concept. Key architectural points the design-chat needs to hold:

**The trained predictor weights ARE the associative memory.** There is no separate "outer memory module" that the predictor queries. Continuous online training on the embedding stream shapes the predictor's weights to recognise repeated trajectories. As trajectories repeat, their corresponding regions in the predictor's learned-map densify, and predictions in those regions become sharper. This is the Bush trails idea operationalised in transformer weights.

**The deployment regime is continuous online training.** "Life is always training time." There is no separate "training phase" and "inference phase" in Weft's design. The same `OnlineTrainerV1` (library-imported from v1, frozen) that runs during v2 experiments is the same regime that would run during deployment — one gradient update per window position, continuously, throughout the agent's existence.

**v2's job is to characterise the architecture-trainer combination, not to test a partial system.** Earlier framings in the design discussion drifted toward "inner PAM in isolation, awaiting outer memory" — this is wrong. v2 tests the whole architecture; synthetic substrates substitute for "real experience" to give the design chat control over what experience the predictor is shaped by, but otherwise the regime is identical to deployment.

**Earlier "outer memory module" framing was incorrect.** If you encounter that vocabulary in the conversation history or in older notes, treat it as drift to be corrected. The Bush trails inspiration is about *trails forming through repeated traversal in a single associative structure*, not separate retrieval from a stored archive.

## 2. The committed Phase 0 result (still valid; survived broken-mechanics episode)

Phase 0 ran seven sub-phases producing four design inputs. Some are still valid; some are invalidated.

**Still valid (no predictor training involved):**
- PRE-A's construction-sanity sweep (substrate primitives validated)
- PRE-B's worked-example region characterisation (DINOv2 on AI2-THOR: mag=0.017, loc=0.836, cont=0.077, P=360, D≈14)
- PRE-C's architectural assertions (11 assertions on v2 substrate; shape-contract checks, no training)
- PRE-D1c's corner reachability assessment (analytical, consumes PRE-B)

**Invalidated (depend on trained predictors at V2_TRAINING_STEPS=10000):**
- PRE-A's V2_TRAINING_STEPS calibration (calibrated at 10000, which is pre-grokking — root cause of the broken-mechanics episode)
- PRE-D1a baseline distribution (collected from untrained predictors)
- PRE-E τ_W calibration (calibrated against the untrained-predictor noise)
- PRE-D2 n-validation at L_d=2 (measured untrained-predictor noise)
- PRE-E's variance-limited reading at L_d=2

**Phase 0 commit hashes (frozen, do not need re-tracking):** 60c8680 (PRE-A), dbe6e34 (PRE-C), 65fecf3 (PRE-B), c1c9d16 (PRE-D1c), ca3972d (PRE-D1a), de80e76 (PRE-E), 63eb9c5 + b2d0ec7 (PRE-D2), 1f5f62a (HANDOFF).

## 3. Phase 0.5 design-chat commitments (decisions stand; measurements need redo)

The design-chat made the following commitments before Phase 1, all of which remain valid as design decisions regardless of the broken-mechanics episode. Phase 1 must re-implement against these commitments at corrected V2_TRAINING_STEPS.

| Commitment | Value | Rationale |
|---|---|---|
| Sample size | n=10 per cell | PRE-D2 validated n=20 buys no resolution over n=10 (though PRE-D2 was on broken mechanics; revisit if recalibration changes this) |
| Map output | three-category: discriminably-working / discriminably-non-working / band-resident | Spec §7.3 binary extension; needed to absorb measurement-fragility region |
| Reallocation contingency | §9.4 rule, capped at 20 cells (graceful degradation above) | Per Grok adversarial review |
| Per-axis density | 3 values per axis (not 5) | Jason's commitment — fine sweep at signal-bearing regions can come later if needed |
| Cross structure | per spec §3.5: 2nd / midpoint / 4th of grid | First-principles axis extremes excluded |
| L_d_main coverage | {1, 2, 4} (intermediate L_d=2 added) | Jason's push back during Phase 0.5 — capacity threshold detection |
| Dim sweep scale | log; D ∈ {4, 16, 128} after sub-phase 1.0 grid revision | Log matches manifold-dim estimation geometry; revised from original {4, 32, 256} per construction-feasibility validation |
| Repetition axis | period P, log-spaced; P ∈ {32, 256, 2048} | Log spacing covers sub-window, single-traversal, multi-traversal regimes |
| F (fidelity) / coverage | reported per cell as outputs, NOT swept | Single-scalar repetition axis is period |
| Corner sampling | §3.5 avoidance retained for main effects; +1 corner-calibration probe | Reliability-over-coverage principle |
| Parallelism | start 2x, escalate to 3x if VRAM allows; lock at 2x measured 1.447× | Empirically locked at 2x (1.447× per-arm overhead, 1.38× effective throughput) |
| Continuity grid | (0.077, 0.4, 0.8) — low anchored to PRE-B worked-example | Asymmetric; documented design choice |
| Magnitude grid | (0.1, 0.3, 0.7) | + magnitude=0.9 probe added during pilot scope expansion |
| Per-K instrumentation | preserve per-K Diff_μ in per-cell forensics (not just K-aggregated) | Tests K-aggregation-swamping hypothesis; refuted by pilot but data preserved |
| Paired-baseline analysis | re-analyze pilot data with paired same-seed differencing | Free re-analysis; methodology contribution |

## 4. Phase 1 work product (invalidated as broken-mechanics; methodology survives)

Phase 1 ran multiple sub-phases discovering increasing concerns culminating in the broken-mechanics diagnosis. The **measurements** are invalidated. The **methodology infrastructure** survives.

### Sub-phase commits (all local, push hold preserved throughout):
- `7aa20cd` — 1.0 grid validation: substrate-feasibility revealed P ≥ 2·D constraint; original D grid {4,32,256} revised to {4,16,128}; lone infeasible (D=128, P=32) cells dropped with `construction-infeasible` label.
- `fd1c519` — 1.1 modules + smoke: arm_runner, classification, parallel_harness modules built; smoke validated bit-identical reproducibility (delta=0.0 against PRE-D2 seed 0), classification replicates PRE-D2, per-K semantics validated, eval-metric synthetic-stream semantics confirmed. 2x parallelism locked (1.447× overhead, 3x failed gate).
- `f10873f` — 1.2 controls (C1/C2, 120 arm-runs): surfaced the **threshold non-transfer finding**. The PRE-D1a baseline (0.0277) and τ_W (0.0613) were calibrated at one config; C1 showed magnitude=0 streams at Phase 1 mid config sit at 0.12-0.25 — the baseline is construction-config-dependent. This was the first signal something was off. STOP triggered for design-chat resolution.
- `c954ee1` — 1.2.5 baseline-variance diagnostic (40 runs): per-axis baseline variation at fixed cross-position at L_d=1. Result: relative spread 1.76 (≫ 0.35 threshold). Option 1 (per-config bit-identical baseline) chosen decisively over Option 2 (baseline grid).
- `6f7a51e` — per-K eval + Option 1 pilot harness built; faithfulness bug caught (per-K train initially diverged 0.034 vs 0.020 because train_one's first-checkpoint body-repr forward was dropped, consuming dropout RNG; restoring it gave bit-identical reproduction).
- `8b88c78` — L_d=1 pilot (240 runs): per-config baselines at n=20, cells at n=10, magnitude axis + magnitude=0.9 probe + continuity/dim depth at mid cross. Result: 0/14 discriminably-working at K-aggregated, 0/14 at k=15, 0/10 paired-resolvable. CV 0.42-1.42 across configs.
- `f0b6403` / `d9da253` — L_d=2 and L_d=4 capacity extension (480 runs): 0/14 at L_d=2, 0/14 at L_d=4. Pre-registered as `robust_null` per the outcome thresholds.
- `ee757d8` — diagnostic battery: caught broken mechanics. cos(mean, target) = -0.011 on the predictor's own training stream; trivial last-frame baseline = 0.56. Mean head never learned to predict anything; loss dropped only because variance head saturated.
- `313efa5` — confirmatory longer-training test: cos(mean,target) at progressively longer steps showed sharp grokking transition between 25k and 50k. Values: 2k → 0.003, 10k → -0.014, 25k → 0.033, 50k → 0.397, 100k → 0.934, 150k → 0.945, 200k → 0.953.

### The salvageable methodology infrastructure (the actual v2 product so far):
- **Substrate primitives** (§5): base manifold trajectory, repetition primitive, perturbation primitive. Validated.
- **Property measurement** (§4): magnitude, locality, continuity, repetition, manifold-dim. Validated.
- **Measurement protocol** (§6): self-similarity matrix, repetition detection, reference-state estimation, perturbation detection, per-axis distributions. Validated.
- **Phase 1 module infrastructure**: arm_runner, classification, parallel_harness, sweep_grid, controls, reallocation, aggregate. Code is correct; will produce valid results when run against trained predictors.
- **Per-config baseline design**: confirmed methodologically necessary (Option 1 over Option 2).
- **Per-K instrumentation**: refuted the K-aggregation-swamping hypothesis on broken mechanics; preserved as a methodology dimension to revisit on corrected mechanics.
- **Paired-baseline analysis**: methodology contribution. Free re-analysis of per-cell forensics, isolates training-trajectory variance from shared noise.
- **Sub-phase 1.0 grid validation**: substrate-feasibility pre-check that caught the (D, P) infeasibility issue. Process learning for future grid commitments.

## 5. The broken-mechanics episode (the load-bearing event)

### What happened

PRE-A calibrated V2_TRAINING_STEPS = 10000 via plateau detection. The plateau detector watched the NLL loss trajectory for flatness. The NLL flatness was real — loss dropped from ~−2081 (initial) to ~−2900 by 10k steps. But this flatness wasn't "the predictor has converged on the trajectory"; it was a combination of (a) the variance head saturating quickly (σ → small to fit residuals) and (b) the loss curve being smooth enough through the pre-grokking region to look like a plateau when it was actually still slowly descending. The mean head, separately, was still in its pre-grokking flat region at 10k steps with cos_k1 ≈ −0.01 (the predictions were essentially random in direction).

Every v2 measurement that involved trained predictors (PRE-D1a, PRE-E, PRE-D2, all Phase 1 measurements) trained at V2_TRAINING_STEPS=10000 — pre-grokking. Diff_μ values for these measurements were reading variance of an unlearned mean output, not signal vs. baseline.

### What the diagnostic battery actually found

Five sanity checks were run on a trained predictor at the broken V2_TRAINING_STEPS=10000 (mid mag=0 cell at L_d=1, P=256, D=16). Three failed independently, one passed (in a way that's actually evidence the loss-as-convergence-signal was misleading), and one revealed the unviable-egg pattern concretely. The full results are in `sanity_battery.json`:

**Sanity 1 — predictor on training stream (FAIL).** Compared the predictor's `cos_k1` on its own training stream to the trivial last-frame baseline (~0.559 measured separately in the confirmatory test). The trained predictor scored **cos_k1 = −0.0093 on the training stream itself**. Held-out stream: cos_k1 = −0.0110. Both negative — worse than a random unit vector in expectation. The predictor not only hadn't generalised; it hadn't even fit the training data. (Diff_μ was 0.217 on training, 0.218 on held-out — almost identical, confirming there's no learning happening, just consistent failure.)

**Sanity 2 — full trajectory swap (FAIL).** Compared prediction error on a familiar trajectory (A, what it was trained on) vs. a completely novel substrate (B, different U, different harmonics, different period). If the predictor learned A's structure, B should produce much higher error. **Delta err_mean = 0.0010** — essentially no difference. The predictor's error is the same whether you feed it the trajectory it was trained on or a wholesale-different one. The predictor isn't sensitive to substrate identity at all.

**Sanity 3 — eval metric on analytical cases (FAIL).** Three known cases:
- (a) Stream identical to training stream → Diff_μ = 0.217 (should be ~0 for a learned predictor; high because predictor didn't learn).
- (b) Same stream + orthogonal offset → Diff_μ = 0.239 (barely different from case a; should reflect offset magnitude meaningfully).
- (c) Random-noise stream → Diff_μ = 0.0020 (essentially zero — *lower* than for the trained-on stream!).

The metric is producing values that don't correlate sensibly with what the inputs represent. Random noise has the lowest Diff_μ — meaning the eval metric is *more "satisfied"* with random noise than with the actual trained substrate. Whatever the metric is computing, it isn't what we thought.

**Sanity 4 — loss trajectory inspection (PASS, but misleadingly so).** Loss decreases monotonically at all three capacities:
- L_d=1: −2081 → −2788 (drop of 707)
- L_d=2: −1401 → −2922 (drop of 1521)  
- L_d=4: −275 → −2811 (drop of 2536)

This passes the "loss is decreasing" check. But final loss is *not* capacity-monotone (L_d=2 has lower final loss than L_d=4, which shouldn't happen for a well-converged training of a higher-capacity model on the same data). And the loss curves are visibly flat from ~2k steps onward at all capacities — the same plateau-at-chaos that PRE-A's detector mistook for convergence.

This check's "PASS" verdict is actually the strongest evidence that loss-flatness alone is an inadequate convergence criterion. The loss "passes," yet sanities 1, 2, 3 all fail. The mean head and the loss are decoupled in the broken-mechanics regime.

**Sanity 5 — inter-seed predictor correlation (low correlation, regime confirmed).** Two predictors trained at the same cell from different seeds, fed the same test windows. Cosine between their outputs: **0.124**. Held-out Diff_μ values: seed 0 = 0.218, seed 1 = 0.067 — a 3× spread between two seeds at the same config.

Regime: "low_corr (<0.3): seeds learn DIFFERENT functions (unviable-egg, concrete)". This is exactly the failure mode Jason intuited weeks earlier as "what if the wrong starting seed prevents the agent from ever growing coherently — like an unviable egg." The empirical confirmation: different seeds at the same broken V2_TRAINING_STEPS produce *fundamentally different functions*, not just different points on a shared learning landscape. This is its own substantive finding about the broken-mechanics regime, and worth keeping separate from the v3+ deployment-implications discussion in §6 — at 10k steps the predictor isn't learning the trajectory at all, so the inter-seed divergence is "different randomness" not "different learned structures." Whether sanity5's finding generalises to corrected mechanics (i.e., do seeds still diverge meaningfully *after* grokking?) is an open question the recalibration cascade should re-test.

### Why the pilot cross-checks didn't catch this

The Phase 1 pilot ran five orthogonal cross-checks: per-config baselines (cleanest threshold), per-K instrumentation (long-horizon test), paired analysis (variance cancellation), magnitude=0.9 probe (signal strength test), capacity extension (L_d=2/4 test). All converged on the same "robust null" finding.

The cross-checks couldn't catch the broken mechanics because they all sat **downstream of the same broken train+eval pipe**. They tested different methodology choices around classification while sharing the same upstream measurement pipeline. A broken pipe produces consistently-broken-looking results across cross-checks; the convergence felt like methodology validation.

In the language of the sanity battery: sanities 1, 2, 3 *all failing simultaneously* is what the pilot's cross-checks couldn't see. Each cross-check operated assuming the predictor had learned something and the metric was measuring something meaningful. Both assumptions were wrong, but the cross-checks couldn't test them.

### What caught it

Jason's interrogation: "0/420 trainings produced any signal at all" looks more like a broken pipe than a substantive variance limit. The Claude design-chat (me) had drifted into accepting the variance-limited reading as architectural finding. Jason flagged that the uniformity of the null across all variables was anomalous in a way the cross-checks didn't address.

The diagnostic battery designed in response (5 sanity checks: training-stream fit, extreme substrate perturbation, eval metric on analytical cases, loss trajectory inspection, inter-seed predictor correlation) caught it decisively. Three independent failures (sanity 1, 2, 3) plus a confirming low-correlation finding (sanity 5) plus a "loss is misleading" observation (sanity 4) — five orthogonal signals that the train+eval pipe wasn't producing meaningful numbers.

### The methodology lesson

Methodology cross-checks at the classification level can't catch failures at the measurement level. The discipline needed for v3+ is **layered validation**:

1. **Mechanics first**: does the predictor actually train? Does the eval metric produce sensible values on analytically-known cases? (The diagnostic battery is this layer.)
2. **Then measurement**: do the measurements vary meaningfully across substrate properties?
3. **Then methodology**: do the classifications-from-measurements behave as expected?
4. **Then findings**: what does the resulting characterisation tell us about the architecture?

v2 jumped from layer 1 (assumed working without explicit validation) to layer 3 (extensive methodology cross-checks). The hole was at layer 1; the cross-checks at layer 3 couldn't see it. Future v3+ work should validate at each layer explicitly before building the next layer on top.

Specifically for v3+: the diagnostic battery (or some refinement of it) becomes a **pre-experiment requirement**, not a post-hoc check. Before any new substrate, encoder, or trainer is exercised in PRE-A scope, the layer-1 validation runs first. Cheap, fast, and prevents the failure mode that just consumed multiple sessions of work.

### Specifically what the recalibration cascade must redo at layer 1

Before any methodology work at corrected V2_TRAINING_STEPS:
- **Re-run sanity1** (training-stream fit): the trained predictor's cos_k1 on the training stream should exceed the trivial baseline (~0.559) decisively.
- **Re-run sanity2** (full trajectory swap): err_mean on novel substrate should be meaningfully higher than on familiar substrate.
- **Re-run sanity3** (analytical cases): Diff_μ should be near zero on the trained-on stream, meaningfully higher on offset/random streams, with sensible relative magnitudes.
- **Re-run sanity5** at corrected V2_TRAINING_STEPS: do seeds converge to *similar* functions post-grokking, or does inter-seed divergence persist? If divergence persists, the variance-limited reading becomes a real architectural finding rather than a broken-mechanics artifact. If correlation rises substantially, the unviable-egg pattern was specific to pre-grokking and resolves with corrected training.

Sanity5's post-grokking re-test is the most substantively interesting of the four. It's the test that distinguishes "broken mechanics fully explains the null" from "broken mechanics + a real architectural variance limit that the cross-checks were measuring on top of broken mechanics." The clean version of this question can only be asked after recalibration.

## 6. The grokking-transition finding (substantive architectural finding, regardless of v2 outcome)

The confirmatory test (`training_length_test.json`, commit `313efa5`) revealed a sharp learning transition between 25k and 50k training steps, with onset somewhere in that interval. The test ran at the mid-mag=0 L_d=1 config (P=256, D=16) with no early stopping, evaluating `cos_k1` (cosine between predicted and target embedding at k=1, the immediate-next-frame prediction) and `cos_allK` (averaged across K) at progressive checkpoints.

**Trivial last-frame baseline: cos_k1 = 0.5586.** (This is the cosine between consecutive frames in the substrate; "predict the last frame seen" achieves this without any learning.)

| Steps | cos_k1 | cos_allK | loss | wall (min) |
|---:|---:|---:|---:|---:|
| 2,000 | 0.003 | 0.001 | −2900.87 | 0.4 |
| 10,000 | −0.014 | 0.002 | −2990.25 | 1.8 |
| 25,000 | 0.033 | 0.005 | −3012.62 | 4.8 |
| 50,000 | 0.397 | 0.422 | −3103.88 | 9.7 |
| 100,000 | 0.934 | 0.932 | −4235.62 | 19.1 |
| 150,000 | 0.945 | 0.943 | −4339.79 | 28.8 |
| 200,000 | 0.953 | 0.951 | −4453.84 | 38.3 |

Key observations from the actual data:

- **Pre-grokking flat region**: at 10k, 25k cos_k1 is negative or near-zero. The mean head's outputs at the perturbed positions have essentially no angular alignment with targets — the predictor isn't producing meaningful directional predictions.
- **Loss kept descending throughout**: from −2081 (initial L_d=1 in sanity4) to −2900 at 2k to −4453 at 200k. The loss curve does not signal the grokking transition — it descends smoothly through it. PRE-A's plateau detector watching loss flatness was watching a signal that doesn't track learning of the mean head at all.
- **Grokking transition between 25k and 50k**: cos_k1 jumps from 0.033 to 0.397, an order of magnitude in one interval. The transition is concentrated in this range.
- **Above trivial baseline by 100k**: cos_k1 = 0.934 > 0.559 (trivial baseline). Predictor now meaningfully outperforms last-frame prediction.
- **Diminishing returns post-transition**: 100k → 200k gains only 0.019 in cos_k1. Most of the gain happens by 100k.
- **Wall-clock per step roughly constant**: ~38 min for 200k steps at the test cell. Translates to ~11.5 ms per step at L_d=1 on the 4080 Super (post-grokking, this scales with capacity per the L_d ratios captured elsewhere).

### Why this matters substantively

**For v3+ deployment vision**: the architecture-trainer combination requires ~100k experience-frames before predictions are useful. At 30 fps video, this is ~55 minutes of continuous experience with stable trajectory structure. A deployed Weft would be "blind" to perturbations during its first ~25k-50k frames of any new trajectory regime. This has implications for:
- Bootstrap phase of fresh Weft deployments
- Robustness to interrupted training
- Cross-trajectory generalisation (does grokking on one trajectory accelerate grokking on similar ones?)
- The "unviable egg" concern made concrete (and confirmed empirically by sanity5 — see §5)

**For v3+ design directions**: can architectural revisions move grokking onset earlier? Variables worth probing:
- Optimizer choice (Adam variants vs SGD with momentum vs weight averaging)
- Batch size (v1 trainer was online batch=1; batched training may grok faster)
- Learning rate schedule (warmup, cosine, adaptive)
- Initialization scheme (smaller/larger initial weights)
- Attention structure variations
- Capacity (L_d=4 may grok at different step count than L_d=1 — confirmatory test was at L_d=1 only)

**For v3+ methodology refinements**: PRE-A's plateau detector needs a mean-head-aware criterion. NLL loss flatness is necessary but not sufficient — the loss continues to descend monotonically through the grokking transition (see the table; loss goes −2990 → −3013 → −3104 → −4236 across the transition window). Future plateau detection should track `cos_k1` (or equivalent prediction-quality metric on the training stream) against the trivial last-frame baseline as the actual convergence criterion. The variance head's saturation and the mean head's grokking are decoupled in the loss signal; either alone can fool a loss-flatness detector.

### Cell-dependent grokking concern

The confirmatory test ran at exactly one cell (mid-mag=0 L_d=1, the simplest case for learning to succeed — single base trajectory, no perturbation, low capacity). The grokking transition might occur at different step counts at different cells:
- **High-perturbation cells** may take longer to grok (substrate is more chaotic for the predictor to learn).
- **High-D cells** may take longer (more manifold dimensions to fit).
- **Low-locality cells** may take longer (perturbations spread across more positions).
- **L_d=2 and L_d=4 cells** are entirely uncharacterised for grokking onset; the parameter counts are 22M and 30.5M vs L_d=1's 17.9M. Could grok earlier (more capacity helps fit faster) or later (more parameters to coordinate). The sanity4 loss trajectory data hints L_d=2 and L_d=4 both reach similar final loss values to L_d=1 at 10k steps but from very different initial losses, suggesting different dynamics that may or may not preserve at the longer training horizons.

This is a load-bearing question for recalibration scope (see §8 Fork 1).

## 7. What v2 has actually delivered (closing-trail framing)

Re-framing v2's contributions given the broken-mechanics episode:

### Primary contributions (what closing should foreground):
1. **Methodology template for characterisation work.** Per-config baselines, per-K instrumentation, paired-baseline analysis, magnitude probe, capacity extension, pilot-first discipline, controls-before-main-effects ordering, multi-layer adversarial review (Claude oracle + secondary model). All transferable to v3+ and future variants.
2. **The layered-validation discipline lesson.** Layer 1 (mechanics) before layer 3 (methodology cross-checks). The diagnostic battery as a pre-experiment requirement. This is the institutional learning from the broken-mechanics episode.
3. **The grokking-transition characterisation.** Architecture-trainer combination has sharp learning onset between 25k and 50k steps at the tested cell. Substantive architectural property with deployment implications.
4. **Substrate-feasibility methodology**: sub-phase 1.0 grid validation as a process-improvement for future grid commitments. P ≥ 2·D constraint and out-of-validated-range checks.
5. **Threshold non-transfer finding**: the baseline is construction-config-dependent. Single global threshold doesn't transfer across configs. Per-config baselining is methodologically necessary (Option 1).

### Secondary contributions (technical infrastructure):
6. Substrate primitives, property measurement, protocol modules — all valid, library-ready for v3+.
7. Phase 1 module infrastructure — code is correct; produces valid results when run against correctly-trained predictors.
8. Empirical compute characterisation: L_d=4 is ~1.66× L_d=1 per-arm cost; 2x parallelism produces 1.447× overhead on the 4080 Super.

### Not delivered (originally scoped for v2):
- The working-region map (Phase 1's central deliverable). Recalibration + re-run needed before this can be produced.
- PRE-A V2_TRAINING_STEPS calibration with mean-head-aware criterion. Needed for recalibration.
- PRE-D1a baseline at corrected training. Needed.
- PRE-E τ_W at corrected baseline. Needed.
- PRE-D2 at corrected training. Needed.
- All Phase 1 measurements at corrected training.

## 8. Decisions pending for the recalibration phase

### Fork 1: V2_TRAINING_STEPS recalibration scope

**The question:** lock V2_TRAINING_STEPS based on what evidence?

**Option A: Single-cell grok detection (cheapest).** Run grok-detection at one cell (e.g., mid-cross all-midpoint at L_d=1, matching the confirmatory test). Lock V2_TRAINING_STEPS to where this cell crosses the grokking transition. Risk: other cells (high-perturbation, low-locality, high-D) might grok later; locked value might be insufficient.

**Option B: Multi-cell grok detection.** Run grok-detection across the 14-cell pilot structure (or a representative subset). Lock V2_TRAINING_STEPS to where the *slowest-to-grok* cell crosses transition, with safety margin. More confidence; ~10-14 hours of compute at the new training cost.

**Option C: Per-cell adaptive training with early-stopping.** Generous upper bound (e.g., 200k steps), but stop each training run early when `cos(mean, target)` plateaus. Most adaptive; per-cell training time variable. Adds complexity to the trainer.

**Recommendation lean from this session:** Option B. Single-cell calibration is what got us into the broken-mechanics episode; multi-cell is the safer methodological response. Per-cell adaptive is elegant but adds harness complexity at a moment when stability matters more than efficiency.

### Fork 2: Compute strategy

**The question:** local 4080S 2x, or rent vast.ai capacity for the re-validation?

At 10× per-arm cost (V2_TRAINING_STEPS=100k vs 10k), local 2x parallelism takes the Phase 1 pilot from ~11 hr to ~5-6 days. Full re-validation (PRE-A redo, PRE-D1a baseline, PRE-E τ_W, PRE-D2, all of Phase 1) at local 2x is weeks of wall-clock.

**Vast.ai cost estimate (from earlier discussion):** 8× RTX 4090 instance at $0.40/hr/GPU ≈ $3.20/hr. Full Phase 1 pilot at 8-way parallelism (if memory allows): ~6-8 hours. Re-validation cascade: 1-2 days of rental. Cost ~$80-150.

**Cross-hardware reproducibility concern:** PRE-D2's reproducibility check passed bit-identical (delta=0.0) against PRE-D1a baseline on the 4080 Super with determinism flags. Cross-hardware (4080S → 4090) may produce numerical drift. This was a more serious concern when bit-identical reproduction was load-bearing; now that the broken-mechanics finding requires re-baselining entirely, the cross-hardware concern is moot for the re-validation. The re-validation collects fresh baselines on whatever hardware is used; the cross-hardware concern only matters if intermediate results on rented hardware need to match local results.

**Recommendation lean from this session:** vast.ai for the re-validation. 10× per-arm cost makes local impractical for v2 completion. The setup overhead (1-2 hours of human attention) is small against the days of wall-clock saved. The re-validation produces a self-contained set of measurements that don't need to match anything from the broken-mechanics era.

If vast.ai isn't acceptable, the alternative is accepting weeks of wall-clock for v2 completion, with Phase 1's central deliverable arriving in roughly a month.

### Fork 3: Re-validation scope and ordering

**The question:** which sub-phases get re-run, in what order, against what acceptance criteria?

**Recommended cascade:**
1. **Recalibrate V2_TRAINING_STEPS** at multi-cell grok-detection (Option B from Fork 1). Lock new value. Update `v2/config.py` and PRE-A's plateau detector with mean-head-aware criterion.
2. **PRE-A re-run** (training-steps sanity sweep at new V2_TRAINING_STEPS). Confirm trained predictors at corrected step count produce sensible `cos(mean, target)` on training stream across the 5-axis sanity grid.
3. **PRE-D1a baseline collection** at new V2_TRAINING_STEPS. n=20 bit-identical baseline at the all-midpoint magnitude=0 configuration, L_d_main=1. Per Phase 0.5 §11.7 commitments.
4. **PRE-E τ_W calibration** against the new baseline distribution. Per-head Wilcoxon p<0.05 calibration.
5. **PRE-D2 re-run** at L_d=2 with new training. n=10 vs n=20 CI validation. May produce different sample-size recommendation given grokked predictor's stochasticity may differ from un-grokked.
6. **Diagnostic battery re-run** to confirm the corrected mechanics produce sensible numbers. Layer-1 validation explicit.
7. **Phase 1 controls (C1/C2) re-run** with corrected training.
8. **Phase 1 baseline-variance diagnostic (1.2.5) re-run** to confirm per-config baselining is still necessary at corrected training (it almost certainly is, but verify).
9. **Phase 1 pilot (L_d=1, 14 cells, with per-config baselines, per-K, magnitude probe, capacity extension at L_d=2/4)** with corrected training. This is the resumed Phase 1 work.
10. **Phase 1 main effects** if pilot shows signal. Per Phase 0.5 D2: 3 values per axis × 3 crosses × 5 axes × 3 L_d_main × n=10.

Steps 1-5 are the "Phase 0 redo" portion. Steps 6-9 are the "Phase 1 redo" portion. Step 10 is "Phase 1 main effects" which has never run.

### Fork 4: Closing-trail strategy

**The question:** how does v2 close given the recalibration cascade?

**Option A: Close v2 after recalibration cascade (Steps 1-9 above).** Phase 1 pilot becomes v2's deliverable. Main effects deferred to v3+.

**Option B: Close v2 after full Phase 1 main effects.** Substantial additional compute; pushes v2 completion further.

**Option C: Close v2 with the current state — methodology template + broken-mechanics lesson + grokking finding — and treat the recalibration cascade as v3+.** Re-frames v2 as a methodology development phase rather than a working-region characterisation.

**Recommendation lean from this session:** Option A. The recalibration cascade is conceptually part of v2 (same scope, same spec, same architecture); deferring it makes v2 unfinished. But also: the pilot-with-corrected-training is sufficient to either confirm the variance-limited reading (now as a real finding) or surface signal. The main-effects sweep is large and may not be necessary if pilot is decisive. Option A produces a meaningful v2 deliverable without committing to the full main-effects compute.

This is a substantive design-chat decision; the new chat should think through it with the recalibration outcomes in hand.

## 9. Items already accumulated for the v2 closing document

These are items that should appear in v2 closing regardless of which Option in Fork 4 is chosen. Each is a substantive finding or methodology observation that needs to be recorded.

### Substantive findings
1. **The grokking-transition characterisation** (§6 of this brief).
2. **The threshold non-transfer finding**: baseline is construction-config-dependent across the 5D property space. Single global threshold doesn't transfer; per-config baselining is methodologically necessary.
3. **The substrate-architecture coupling at feasibility edges**: D=128 at low P produces continuity ≈ 1.0 (forced by construction-primitive harmonics). The 5D property axes are not independent at extreme corners.
4. **Empirical compute characterisation**: per-arm cost ratio L_d=4 : L_d=1 ≈ 1.66×; 2x parallelism overhead 1.447× on RTX 4080 Super (3x failed gate due to higher overhead).
5. **The continuity bimodality observation**: PRE-B continuity measurement showed statistical bimodality (BIC improvement 100.55) at near-zero effect-size (component medians 0.0759 vs 0.0778). Methodology refinement: §6.3 multimodal detection should pair BIC threshold with an effect-size companion threshold.
6. **D_global vs D_local discrepancy**: PRE-B's worked-example manifold-dim measured 13.75 globally vs ~5.27 locally. Different estimators with different sensitivities; both consistent with low-dim manifold structure.
7. **L_d_main = 2 spec inconsistency**: spec §9.3 says "L_d_main = 2" but spec §3.3 commits to L_d_main ∈ {1, 4}. Interpreted as intermediate L_d=2 per §11.6. Spec errata candidate.
8. **Worked-example off-grid on magnitude**: PRE-B measured magnitude 0.017; lowest grid value 0.1 (5-6× extrapolation). Phase 1 L_d sweep includes off-grid arm-runs at exact PRE-B coordinates per adversarial-review Finding 2.

### Methodology contributions
9. **The diagnostic battery** (5 sanity checks) as pre-experiment validation requirement for v3+.
10. **Layered validation discipline**: mechanics → measurement → methodology → findings. Layer 1 before layer 3.
11. **Per-config baselining**: required when baseline is config-dependent. Option 1 over Option 2 decisively (spread 1.76).
12. **Per-K instrumentation**: not aggregated at measurement time; per-cell forensics preserve per-K Diff_μ for closing analysis.
13. **Paired same-seed analysis**: free re-analysis isolates training-trajectory variance from shared noise.
14. **Pilot-first discipline**: small-scope diagnostic before large-scope collection.
15. **Substrate-feasibility pre-check** (sub-phase 1.0): construction-feasibility validation before grid commitment.
16. **Mean-head-aware plateau detection**: NLL loss flatness is insufficient (variance head saturates first); track `cos(mean, target)` against trivial baseline.
17. **Multi-layer adversarial review**: Claude oracle + secondary model (Grok) review before CC launch. Both reviews caught real issues.
18. **Pre-registered outcome thresholds**: write outcome categories into result JSON before results land. Eliminates post-hoc framing pressure.
19. **Bit-identical reproduction smoke test** for harness changes: catches faithfulness bugs (e.g., the per-K instrumentation initially dropped train_one's first-checkpoint body-repr forward, breaking dropout RNG consumption).

### Architectural / design-vision observations
20. **The trained predictor weights ARE the associative memory** (correction of earlier "outer memory module" framing drift).
21. **Continuous online training as deployment regime**, with grokking onset implying ~55 min of stable experience before useful predictions emerge.
22. **K-aggregation as future v3+ design knob**: per-K disaggregation lets future architectures track near-future vs far-future asymmetry, potentially adaptively. v2 confirmed K-aggregation isn't the broken-mechanics confound; future work could explore K=4 / K=32 / etc.
23. **§5.3 perturbation primitive is "noisy departure," not "coherent alternative trajectory"**: substrate doesn't test learnable-perturbation-recognition. v3+ inheritance question.

### Process learnings
24. **Phase 0.5 grid commitments lacked construction-feasibility validation**. Sub-phase 1.0 (added during Phase 1) caught this. Future grid commitments should include a feasibility pre-check.
25. **Two STOP-and-surface moments during Phase 0 (PRE-B segmentation, §2.4 repetition range ambiguity)** worked as designed. CC discipline held.
26. **One bug-catch during Phase 0 (PRE-D1a assess_trajectory misfire on negative NLL)**: CC self-detected before reporting, fixed with regression tests. Discipline working as intended.
27. **The continuity bimodality recantation**: PRE-B initially read as false positive, PRE-E confirmed as genuine (BIC 100.55). Self-correction across sub-phases.
28. **Phase 0.5 process-improvement learnings already in v2 closing trail per discussion**.

## 10. Recommended file inclusions for the fresh chat's context

The fresh chat should have access to (in approximate priority order):

### Essential (must read first):
- **This brief** (`WEFT_INNER_PAM_v2_SESSION_BRIEF.md`)
- **v2 spec passes 1+2** (`WEFT_INNER_PAM_v2_Spec_pass1_sections_1_to_3.md`, `WEFT_INNER_PAM_v2_Spec_pass2_sections_4_to_9.md`)
- **`research_operations_v1.md`** (operational discipline)
- **`WEFT_INNER_PAM_v1_CLOSING.md`** (institutional memory from v1)

### Phase 0 context (for design decisions inheriting from Phase 0):
- **Phase 0 EXPERIMENT_INSTRUCTIONS** (1026-line as-shipped version in Jason's repo; this is the version CC executed against — *not* the working drafts that appeared in earlier conversation outputs)
- **Phase 0 HANDOFF** (`v2/HANDOFF.md` from end of Phase 0, commit `1f5f62a`)

### Phase 1 context (for the recalibration cascade):
- **Phase 1 EXPERIMENT_INSTRUCTIONS** (1021-line as-shipped version in Jason's repo; this is the version CC executed against — same caveat as Phase 0)
- **`v2/PHASE1_PROGRESS.md`** (running record CC maintained through Phase 1 sub-phases)
- **Phase 1 broken-mechanics HANDOFF draft** (held uncommitted; CC marked invalidated at top; reference for the pre-broken-mechanics framing, not the current state)

### Phase 0 result artifacts (for the broken-mechanics-affected vs unaffected distinction):
- `v2/results/pre_a/`: substrate-sanity sweep results (unaffected; valid)
- `v2/results/pre_b/worked_example_region.json`: worked-example characterisation (unaffected; valid)
- `v2/results/pre_c/arch_assertions_v2_substrate.json`: architectural assertions (unaffected; valid)
- `v2/results/pre_d1a/`: PRE-D1a baselines + endpoint stability (**invalidated**; reference only)
- `v2/results/pre_d1c/corner_reachability_assessment.md`: corner reachability (unaffected; valid analytical)
- `v2/results/pre_e/scaffolding_calibration.json`: SCAFFOLDING calibration (**invalidated**; τ_W needs redo; reference only)
- `v2/results/pre_d2/n_validation_report.json`: n-validation (**invalidated**; reference only)

### Phase 1 result artifacts (the broken-mechanics episode):

The diagnostic-battery and confirmatory-test results do **not** live in dedicated subdirectories. They are three JSON files (currently uploaded to this conversation; need to be located in or moved into the v2 results tree for the fresh chat to find them):

- **`sanity_battery.json`** (committed `ee757d8`): the five sanity checks. Critical reading — sanities 1, 2, 3 all FAIL independently; sanity 5 confirms the low-correlation/unviable-egg regime at cos = 0.124. This is the file that decisively diagnosed broken mechanics. Per §5 of this brief.

- **`training_length_test.json`** (committed `313efa5`): the grokking-curve characterisation from 2k to 200k steps at the mid mag=0 L_d=1 cell. Shows the sharp transition between 25k and 50k. Critical reading — this is what confirmed the root cause as V2_TRAINING_STEPS-insufficient rather than a deeper substrate/architecture/eval defect. Per §6 of this brief.

- **`baseline_variance_diagnostic.json`** (committed `c954ee1`): the 40-run diagnostic from sub-phase 1.2.5 that established Option 1 (per-config baselines) over Option 2 (baseline grid). Relative spreads 1.61 (D-axis) and 1.76 (continuity-axis), both decisively above the 0.35 threshold. Notes the D=128 = P//2 boundary caveat (forced continuity ≈ 1.0). This is methodologically still relevant — per-config baselining was the correct call against the broken mechanics; the question is whether it remains the correct call against corrected mechanics, which the recalibration cascade should re-test.

The other Phase 1 broken-mechanics result artifacts (controls C1/C2, pilot L_d=1/2/4, capacity extension) exist in the v2/results/phase1/ tree at the commits noted in §4. They are all **invalidated** but should remain accessible as reference for the framing the cross-checks produced before broken mechanics was diagnosed.

### Code infrastructure (Phase 1 modules; still valid for recalibration):
- `v2/src/phase1/arm_runner.py`, `classification.py`, `parallel_harness.py`, `sweep_grid.py`, `controls.py`, `reallocation.py`, `corner_probe.py`, `aggregate.py`

### Adversarial review documents (for methodology context):
- Phase 0 adversarial reviews (Claude oracle + Grok)
- Phase 1 adversarial reviews (Claude oracle + Grok)
- These are not committed to the repo but were used to refine Phase 0 and Phase 1 instructions; if surfaced to the fresh chat, they provide context for the patches already applied.

### Optional but valuable:
- Conversation logs from this session (specifically the broken-mechanics episode interrogation and the architectural design-vision discussion). The fresh chat should have access to these even if not loading them into context by default, in case design-chat reasoning needs to be reconstructed.

## 11. The fresh chat's immediate first move

The fresh chat picks up at: **recalibration of V2_TRAINING_STEPS, with the multi-cell grok-detection structure (Fork 1, Option B) recommended but not committed.**

Immediate decisions for the fresh chat:
1. **Confirm or revise the four forks** (V2_TRAINING_STEPS recalibration scope, compute strategy, re-validation scope/ordering, closing-trail strategy) based on its own reading of the evidence and Jason's preferences.
2. **Draft a Recalibration Phase CC instructions document** covering the cascade in Fork 3 (whichever ordering is committed). This is the equivalent of "Phase 0 redo + Phase 1 redo" instructions in one document. Length probably similar to the current Phase 0 instructions (~700 lines).
3. **Surface the design-vision implication of the grokking finding** to Jason explicitly. The 100k-step training requirement vs continuous online deployment is a real architectural question that v2 has now characterised; Jason hadn't fully absorbed this when the broken-mechanics episode dominated the discussion.
4. **Re-verify with Jason which methodology infrastructure carries forward** (likely all of it, but the recalibration cascade should explicitly confirm not just inherit).

## 12. Loose threads worth bookmarking

These didn't reach commitment in this session but were surfaced and may need future attention:

- **Per-K=8 reporting in addition to k=1 and k=15.** Spec discussion suggested mid-horizon (k=8) as the "prediction structure starting to engage" position. Pilot harness preserved per-K data but reports only K-aggregated and k=15 explicitly. Worth surfacing per-K=8 in re-run pilot's report.
- **L_d=1 vs L_d=2 vs L_d=4 grokking-onset comparison.** Confirmatory test was at one capacity. Different capacities may grok at different step counts. Worth confirming during recalibration that V2_TRAINING_STEPS works across L_d_main ∈ {1, 2, 4}, not just one.
- **vast.ai cross-hardware reproducibility validation.** If vast.ai is committed, run a small bit-identical-against-local reproducibility check at start of rental session against PRE-D2 seed 0 (or equivalent post-recalibration baseline). Confirm hardware family compatibility before launching full re-validation.
- **The K-aggregation methodology refinement.** Per-K=15 didn't show more signal than K-aggregated in the (broken-mechanics) pilot. On corrected mechanics, this is worth re-testing — the broken-mechanics result was uninformative on whether K-aggregation swamps long-horizon signal.
- **§5.3 perturbation primitive as v3+ design question.** Coherent-trajectory-variant perturbations vs noisy-departure perturbations. Substrate could test both; v2 only tested noisy-departure. v3+ inheritance.
- **Closing-document drafting itself.** Not Phase 1 work; happens after recalibration cascade completes. Draft a closing document structure based on §7 of this brief.

---

## End of brief

**State as of brief creation:**
- v0 frozen at 58e91d7; v1 frozen at 58e91d7; v2 at commit `313efa5`.
- v0 21/21 + v1 51/51 + v2 49/49 = 121/121 pytest PASS.
- Frozen trees (v0, v1, shared) git diff against 58e91d7 empty.
- Push hold preserved throughout. All commits local.
- HANDOFF draft for Phase 1 conclusion held uncommitted, marked invalidated at top.
- `v2/PHASE1_PROGRESS.md` records sub-phases 1.0 through diagnostic battery and confirmatory test.

**This brief should be saved at `v2/SESSION_BRIEF.md` or equivalent, and read by the fresh chat as its first context after the project knowledge base.**

*— End of Weft Inner PAM v2 Session Brief —*
