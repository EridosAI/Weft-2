# Weft Inner PAM v2 — Closing Document

**Status:** Final. v2 closes after Stage 1 of the recalibration phase. The grokking finding plus the architectural-shape reading are sufficient closing evidence. No publication intended; this document is institutional record.

**Last commit on the research line:** `5d21a27` (Stage 1 results) + the HANDOFF closing marker. Push hold lifted; record published to remote.

---

## 1. What v2 was supposed to do

Per the v2 spec, the research line had two declared outputs:

1. **Output 1 (the map):** an architecture-property map characterising where in stream-property space the Inner PAM architecture exhibits a discriminable working region — mean and variance differentiation against the per-config baseline, across axes of perturbation magnitude, locality, continuity, repetition, and manifold dimensionality, at two decoder capacities.

2. **Output 2 (the protocol):** a measurement protocol applicable to arbitrary encoder/environment pairs, with the worked example being DINOv2-on-AI2-THOR.

The plan was a multi-phase cascade: Phase 0 (pre-sweep verification across PRE-A through PRE-E), Phase 1 (pilot at L_d=1 + capacity extension at L_d=2/4), then potentially main effects in a 1350-arm-run sweep.

This document records what happened, what was learned, and where the research line stands as v2 closes.

---

## 2. What v2 actually produced

### 2.1 Phase 0 (complete; uncorrupted)

Phase 0 landed across seven sub-phases (commits `60c8680` through `b2d0ec7`). PRE-A construction primitives, PRE-B worked-example characterisation, PRE-C architectural assertions, PRE-D corner reachability and endpoint stability, and PRE-E SCAFFOLDING threshold calibration all completed without STOP. The substrate primitives, property measurements, and protocol modules were built and tested. Phase 0 produced four documented inputs for the Phase 0.5 design chat, which committed the Phase 1 sweep grid.

### 2.2 Phase 1 (modules complete; measurements invalidated)

Phase 1 progressed through sub-phases 1.0 (grid calibration), 1.1 (parallel harness), 1.2 (controls C1/C2 at three L_d), 1.2.5 (baseline-variance diagnostic), and 1.3 (pilot at L_d=1). The module code is correct and inherits forward; the measurements depended on a broken `V2_TRAINING_STEPS=10000` calibration and are not load-bearing.

The breakage was diagnosed in the broken-mechanics episode (session brief §5), which established two findings that carry forward:

1. **Layer-1 validation must gate Layer-2 measurement.** PRE-A's training-step calibration target was loss-plateau, which the variance head saturates and masks the mean head. A sanity battery directly testing whether the predictor's mean head learns its training stream is the gate; this was not in place and the cost was Phase 0.4 → Phase 0.7 → all of Phase 1.

2. **The mean head was not learning.** At the broken 10000 steps, `cos(mean, target)` on the training stream was ~−0.014 — uncorrelated with target. The trivial last-frame baseline was 0.559. Loss decreased throughout because the variance head was doing the work of fitting the NLL objective.

A confirmatory training-length test (commit `313efa5`) showed the predictor groks at ~50–100k steps for the C1 configuration, reaching cos 0.93+ at 100k. This invalidated the Phase 1 measurements and triggered the recalibration phase plan.

### 2.3 Recalibration phase Stage 1 (the final compute v2 invested)

Stage 1 of the recalibration phase ran 21 grok-curve runs (7 cells × 3 seeds × 200000 steps) on the locked Phase 1 grid. The cells were chosen to span the property space rather than guess at slowest configurations: L_d ∈ {1, 2, 4} at mid, plus D=128, mag=0.9, P=32, and continuity=0.8 probes.

Three findings:

**Mechanics confirmed fixed.** On 6 of 7 cells, the mean head groks to cos 0.95–0.98 with sufficient training. The broken-mechanics episode was real and is closed. The mean head can learn the trajectory.

**Grok onset scales with manifold dimensionality.** D=4 cells grok at ~25–50k steps, D=16 at ~50–100k, D=128 still rising at the 200k budget ceiling (max cos ~0.82–0.85). The L_d-coupling is mild compared to the D-coupling.

**Training dynamics converge across seeds.** Within-cell `lock_step_candidate` spread ≤1.5× — different seeds reach the grokked regime at similar step counts, and the n=3 design intended to characterise per-cell seed variation found minimal variation to characterise. Whether different seeds converged to the *same function* is unmeasured: that comparison would have required Stage 3's pairwise-cosine measurement, which the cascade stopped before reaching.

The §3.5 STOP fired on C4 (D=128) because its lock_step_candidate exceeded the 175000 ceiling for all three seeds. The 200000 × 1.1 = 220000 margin overflowed the 200000 budget the confirmatory test characterised. CC halted before Stage 2 per the spec; no V2_TRAINING_STEPS lock was written.

The cascade stops here.

---

## 3. The architectural-shape finding

This is the v2 closing claim.

The Inner PAM architecture, in its v1/v2 formulation (transformer encoder + per-K Gaussian NLL path-prediction loss), groks to a high-performance regime on stationary periodic synthetic substrate when trained for enough steps, with consistent training dynamics across seeds. That is what Stage 1 showed.

This is the success mode of a well-conditioned function approximator on a fixed signal. Gradient descent finds a function that minimises NLL given enough capacity and enough steps; the seed-level consistency in *when* grokking occurs is what one would expect from a well-conditioned optimisation landscape on this objective and this data. Whether the seeds converged to the *same* minimum is an unmeasured question — but for the closing claim it does not matter, because the architectural-shape argument rests on the form of the architecture and its objective, not on a seed-function-convergence measurement. The grok-onset scaling with dimensionality is a memorisation-capacity result: bigger functions take more steps to fit.

It is not the phenomenology the Weft hypothesis was about. The original research question — whether an embodied agent can learn trajectory-level associative structure through repeated visual exposure, in the spirit of Bush's adaptive trail growth — is closer to a retrieval/memory system with locality structure than to a parametric predictor minimising path-prediction loss. Successful grokking on stationary data tells us the architecture fits its training distribution; it does not tell us that associative trails formed or that the system learned anything trail-shaped at all.

The v0 PAM had more of the right shape (FAISS-indexed associations, retrieval-based recall, no training-step concept). The v1/v2 turn toward parametric predictors with Gaussian NLL loss was a reasonable response to noisy evaluation in the v0 era, but it moved the architecture away from the thing the research question is about. By the time v2 was calibrating mean-head plateau detectors at 200k training steps on synthetic periodic substrate, the experiment was characterising a transformer's behaviour on stationary signal — a real subject of study, but not Weft.

This is not a failure result. It is a useful negative finding: this architecture is not the right shape for the Weft hypothesis, and Stage 1 has produced the evidence to claim that without running another 17 days of cascade compute.

---

## 4. The methodology lessons that carry forward

These are the items worth keeping regardless of where the research line goes next.

### 4.1 Layer-1 validation gates Layer-2 measurement

The broken-mechanics episode established this as institutional discipline. Before measuring whether a predictor's outputs satisfy some downstream property (signal density, classification, statistical comparison), verify directly that the predictor learns its training stream. A sanity battery at the cell of interest — `cos(prediction, target)` against the trivial baseline, sensitivity to full-trajectory swap, sensible behaviour on analytically-distinct cases — costs minutes and forecloses weeks of wasted measurement.

The cost of skipping this gate in v2 was Phase 0.4 onwards plus Phase 1. The cost of running it would have been a single training run plus the diagnostic script.

This generalises: any experiment with a downstream property metric needs a direct check that the upstream mechanism is producing the intermediate signal the metric is built on top of.

### 4.2 Calibration targets must be functionally aligned

PRE-A locked `V2_TRAINING_STEPS` against a loss-plateau criterion. The loss was the NLL across both heads of the predictor. The variance head saturates first and dominates the loss surface; the mean head can be uncorrelated with target while loss is plateaued. The plateau-detection logic was technically correct (loss had stopped decreasing) but semantically wrong (the thing the experiment cared about — mean differentiation — had not started yet).

The lesson is not "use better plateau detection." It is: the calibration criterion must directly reflect the property the downstream experiment depends on. A cos-against-target criterion would have caught this immediately. The mean-head-aware plateau detector built in Stage 1 (`v2/src/preflight/mean_head_plateau.py`) is the corrected form, but the more portable lesson is the alignment principle.

### 4.3 Single-point calibration is brittle in high-variance domains

The original PRE-A calibration ran on one cell at one seed. The broken-mechanics episode was, in part, a single-cell extrapolation failure: the chosen calibration cell happened to have a loss plateau that masked the mean-head behaviour, and there was no signal from other cells to flag the issue. Stage 1's seven-cell, three-seed design was the corrected form, and it surfaced the dimensionality coupling that single-cell calibration would have missed entirely.

The portable lesson: when calibrating a parameter that gates downstream measurement, sweep enough cells to characterise the parameter's variation across the property space the experiment will operate on. Single-point calibration is acceptable only when the parameter has been shown to be constant across the space.

### 4.4 Push hold discipline preserves experimental integrity under uncertainty

v2 ran under a push hold from start to finish. Every commit accumulated locally; nothing landed on remote until the closing decision. This meant that when the broken-mechanics episode invalidated Phase 1's measurements, no public-facing record claimed those measurements as findings. The closing record is published in one batch, after the closing decision, with a clean narrative.

This worked. It cost nothing (commits accumulated normally; the only constraint was no `git push`) and preserved the ability to make corrections without retraction. For long experiments with downstream interpretation risk, push hold is worth the trivial overhead.

### 4.5 Frozen-tree inheritance discipline holds across complex inheritance

v2 inherited v1 and v0 via library imports, not file copies. The frozen-tree gate (`git diff 58e91d7 HEAD -- v0 v1 shared` empty) was checked at every session boundary and held throughout. The two failures of the gate during execution were both stochastic test artefacts (the v1 PRE-D canary rewriting a results JSON with log_var floats); both were caught and reset before any commit propagated the corruption.

The discipline cost is small (one diff check per session) and the protection is real: a v2 mistake cannot silently corrupt v0 or v1's record.

### 4.6 Pre-registered interpretation frames without pre-registered thresholds

The recalibration phase instructions, written in the closing design chat, declined to pre-register numerical cosine thresholds for the unviable-egg test while still pre-registering the interpretive frames (convergent / divergent / mixed). This was a deliberate choice against threshold theatre: with one prior data point at 0.124, picking 0.5 or 0.7 as "the high-correlation cutoff" would have been making up a number.

The cleaner form: CC reports the raw distribution and descriptive statistics; frame assignment is experiment-chat work post-hoc with the data in hand. This preserves the discipline of "decide in advance what the outcome means" without committing to numerical cutoffs that have no validated basis.

Worth keeping as a pattern for any pre-registered analysis where the underlying scale is not yet calibrated.

---

## 5. The empirical findings that carry forward

Smaller than the methodology items, but worth noting.

### 5.1 Transformer + cosine-objective groks periodic synthetic substrate with onset scaling in manifold dimensionality

The Stage 1 data is a clean characterisation of grok onset for a 22M-parameter transformer trained with Gaussian NLL on a per-K path-prediction objective, against synthetic periodic substrate constructed per the v2 property primitives. D=4 cells grok at ~25–50k steps, D=16 at ~50–100k, D=128 at 175k+. L_d-coupling is mild relative to D-coupling.

This is not a Weft contribution — it's a property of the transformer + the objective + the substrate construction — but the curves are recorded in `v2/results/recalibration/stage1/` and are useful if anyone trains similar architectures on similar data.

### 5.2 DINOv2 inter-room cosine similarities are very high

From the v1 closing trail and v0 inheritance: DINOv2 produces inter-room cosine similarities in the 0.95+ range, which made the cosine filter impossible to satisfy by construction in the original Weft Stage 0b cross-boundary evaluation. This is encoder-specific and worth knowing for any future work that builds retrieval on top of DINOv2 features.

### 5.3 Manifold dimensionality matters more than expected for trajectory prediction

Across both the v1 closing observations and the Stage 1 D-coupling, manifold dimensionality of the input signal has more leverage on training dynamics than the spec authors anticipated. For predictive architectures, the dim sweep is worth doing early.

---

## 6. Where the research line stands

The Weft hypothesis is not falsified by v2. v2 falsified one specific architectural formulation of it (transformer + per-K Gaussian NLL path prediction). The hypothesis itself — that an embodied agent can learn trajectory-level associative structure through sustained visual exposure — is still open.

If anyone returns to it, here is what the trail suggests:

**The architectural shape probably wants to be retrieval/memory, not parametric prediction.** The v0 PAM direction (FAISS-indexed associations, retrieval-based recall, no training-step concept) is closer to the phenomenology the hypothesis is about. Bush's framing of trails growing through traversal does not map naturally onto gradient descent on a fixed objective; it maps more naturally onto incremental updates to a retrieval structure that gets queried during ongoing experience.

**The evaluation question needs to come before the architecture, not after.** v0 and v1 both struggled with evaluation; v2 responded by abstracting evaluation into a methodology product, which moved the architecture away from the research question. A cleaner approach: state the evaluation question in terms the hypothesis can fail at, then design the architecture to be testable against that question. "Does an embodied agent's repeated traversal of a route produce predictive associations that did not exist at the start?" is the kind of question that can be answered with a direct retrieval-quality metric over time, without needing a property-axis sweep or a per-config baseline framework.

**DINOv2 may not be the right encoder for trajectory-level work.** The high inter-frame similarity that broke v0 Stage 0b's cross-boundary metric is a signal about what DINOv2 represents. Object-level work (the furniture-sequence redesign mentioned in the v1 closing memory) might be a more natural fit; for scene-level trajectory work, an encoder with stronger inter-scene differentiation is probably warranted.

**The methodology discipline from v2 is portable to any successor.** Layer-1 validation gates, calibration-target alignment, multi-cell single-point avoidance, push hold under uncertainty, frozen-tree inheritance — none of these depend on the transformer-PAM architecture. They carry forward unchanged.

---

## 7. What v2 explicitly did not produce

To close clean on scope:

- No Phase 1 main effects (1350 arm-runs) — Fork 4 Option A from the closing design chat.
- No characterisation of inter-seed convergence at the worked-example point (Stage 3 of the recalibration phase) — the cascade stopped at Stage 1.
- No corrected PRE-D1a baseline, PRE-E τ_W, or PRE-D2 n-validation at a working V2_TRAINING_STEPS — Stages 4–6 of the cascade.
- No Phase 1 controls or pilot re-run at corrected training — Stages 7–10.
- No Output 1 (the map) or Output 2 (the protocol applied to DINOv2-on-AI2-THOR) — the v2 spec's two declared outputs. These were not produced because the architectural-shape finding made the cascade-completing question moot for the Weft research line.
- No Phase 2 (coherent-trajectory-variant perturbations), no cross-encoder generalisation, no closing on the Bush-framework alignment — all out of v2 scope from the start.

The v2 record contains: Phase 0 (uncorrupted), Phase 1 (modules complete, measurements invalidated), recalibration phase Stage 1 (grok detection), this closing document. Stages 2–10 of the recalibration phase are documented as planned but not executed (`v2/WEFT_INNER_PAM_v2_RECALIBRATION_EXPERIMENT_INSTRUCTIONS.md`).

---

## 8. Acknowledgement of drift

This section is here because the discipline documents call for it (research_operations §3.6 on lessons-learned + §12.3 on session-level notes), and because closing a research line cleanly requires saying out loud what happened.

Weft v0 was Bush-flavoured retrieval. Weft v1 was a transformer predictor with three architectural arms and an evaluation framework that produced a verdict pattern. Weft v2 was a methodology product (the map + the protocol) that characterised the architecture's behaviour across a property space. The intellectual coherence between v0 and v2 became progressively harder to defend across the iterations.

This was not a single bad decision. Each step was a reasonable response to the previous step's difficulties: v0's noisy results motivated a more rigorous predictor in v1; v1's evaluation challenges motivated a more systematic measurement framework in v2. The accumulated drift only becomes visible when you look at the start and the end together.

The portable lesson: research lines that span multiple major versions need an explicit re-statement of the research question at each version boundary, with explicit alignment between the question and the architecture being tested. If the question and the architecture have drifted apart, that is itself a finding — close the line, document the drift, and start the next line from the question rather than from the previous architecture.

v2 closes here. The architecture is wrong shape for the question; the question is still open; the methodology discipline carries forward; the institutional memory is on remote.

---

## 9. Repository state at closing

- Branch: `main`
- HEAD: `5d21a27` (Stage 1 results) + HANDOFF closing marker + this document
- Remote: `origin/main` matches HEAD (push hold lifted in closing batch)
- Frozen-tree: `git diff 58e91d7 HEAD -- v0 v1 shared` empty
- Canaries: 131 pytest PASS (121 inherited + 10 Stage 1 detector tests), 11 PRE-D PASS
- Working tree: clean
- v2 results: complete through Stage 1 of the recalibration phase; Stages 2–10 not executed
- Documentation: Phase 0 docs, Phase 1 docs, recalibration phase instructions, this closing document

The record is durable. v2 is closed.

---

*Drafted in the v2 closing experiment chat, 2026-05-27. Source documents: session brief, recalibration instructions, Stage 1 HANDOFF, v0 and v1 closing documents, Phase 0 and Phase 1 progress logs.*
