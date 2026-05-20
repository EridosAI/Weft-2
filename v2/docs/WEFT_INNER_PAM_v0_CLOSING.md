# Weft Inner PAM v0 — Closing Document

**Purpose.** Records the v0 verdict, the institutional memory v0 produced, and the disambiguating questions v1 inherits. Companion to the spec, instructions, and HANDOFF; written to be the strategic input for v1 design without prejudging v1 direction.

**Status.** Verdict recorded after the per-ordinal cross-loop input variation diagnostic decisively supported reading (i). v1 scoping is open.

---

## 1. The verdict

**V2 — Shape-learning falsified, with coupling-mechanism caveat.**

The path-prediction mechanism (spec §4.1) fit and learned. Loss decreased; predicted means tracked centrelines; per-item cluster structure formed on TV/Dresser/Sofa (per-pose differentiation in predicted log-variance, observable in the variance-by-ordinal disaggregation). The variance head learned. The failure is specifically at the level of how the per-K-step isotropic scalar variance head's gradient propagates across the predictor's state: variance updates appear to propagate uniformly across positions rather than tracking the inputs that produced the surprise.

**Decisive evidence.** Bed close-up ordinals 9 and 10 are pixel-MD5 identical across all four sampled Stage B loops (30, 50, 75, 100); the DINOv2 cosine between loop_30 and loop_100 frames at those ordinals is 1.000000 to encoder precision. The predictor's variance on those frames still drifted by ~0.42 nat over 70 Stage B loops, indistinguishable from drifts at the 9 other Bed ordinals whose inputs did vary across loops. Across all 11 Bed ordinals, Pearson r between per-ordinal cross-loop input variation and per-ordinal variance drift is −0.0128 (p = 0.97). Variance change without input change can only come from gradient updates produced by training on other frames.

The spec §2.2 prediction — that variance responds to per-item surprise — does not survive on this substrate with this loss formulation.

---

## 2. What v0 demonstrated

The architectural-test machinery worked.

- The encoder substrate verification protocol (spec §5) caught real substrate problems before they became architectural-claim confusion: the inherited 30-frame static dwell (§2.3 violation), the camera-elevation bug, the floor-y oscillation, the DiningTable view-through, and (during execution) the cross-room rendering coupling. These were found by the protocol and resolved or characterised before the architectural claim's verdict.
- The gate machinery (§§8.4, 8.7a) fired correctly on both encoder-level and predictor-level structure. §8.4's ratio + Wilcoxon formulation passed cleanly. §8.7a's three-part criterion at loop 100 surfaced the coupling result.
- Disaggregation discipline (per-(item, ordinal), per-loop, per-K-step) revealed structure the aggregates hid. The aggregate G2.T2 result was "Dresser/Sofa widened slightly, controls drifted, ratio failed." The per-ordinal disaggregation showed TV/Dresser/Sofa had per-pose differentiated responses (consistent with the variance mechanism operating at the input level) while Bed had pose-uniform drift (the coupling finding). Without disaggregation the verdict would have been ambiguous; with it, the verdict became precise.
- The disambiguating diagnostic discipline (cheapest test that can close the path) resolved the reading-(i)-vs-(ii) ambiguity in one minute of compute on existing data.

The operational discipline established in CODING_STANDARDS and research_operations carried the batch. Nine STOPs, none of them retries-with-different-numbers; each STOP a real disambiguation that improved the next decision.

---

## 3. What v0 did not demonstrate

The architectural claim (§2.2: shape representations form through repetition and produce per-input variance responses) was tested and the per-input-variance prediction failed. The other parts of §2.2 — shape representation forming through repetition, density structure emerging in the predictor's weights, repetition driving cluster formation — were not cleanly tested by v0 because Phase 1's substrate-degenerate baseline was discarded and Phase 2 alone doesn't provide the repetition gradient that M4 (repetition-stratified accuracy) was designed to measure.

What v0 does not say:

- That the path-prediction loss formulation (§4.1) is wrong. The mechanism learned. The failure is in how variance gradients propagate, which is a specific subcomponent of the formulation.
- That DINOv2-frozen is the wrong encoder substrate. DINOv2 produced the encoder-level signal §8.4 verified. The coupling result is downstream of the encoder.
- That AI2-THOR + ProcTHOR is the wrong environment. Substrate issues found were addressable; the environment supported the experiment well enough to produce a clean verdict.
- That Inner PAM as an architectural concept is invalidated. The §2.2 prediction failed in a specific way; this constrains v1 design rather than refutes the concept.

---

## 4. The eight substrate findings (institutional memory)

v0 surfaced eight substrate properties not anticipated at design time. They constrain v1 substrate design and should be checked early in any successor experiment. Seven are substrate issues requiring correction; one (finding 4) is distinct in character — an interpretive failure mode rather than a substrate property requiring change.

1. **Python 3.10 → 3.12.3.** Instructions document had the wrong version; the prior repo's verified-working environment was 3.12.3.

2. **`embeddings.npy` partial population.** Substrate verification encoded only dwell frames; 67,240 of 100,000 rows were zero. Caught because of the §5 verification check on embedding norms.

3. **30-frame static dwell.** Inherited from Stage 0b. Violated §2.3 (no zero-velocity frames). Required substrate redesign to continuous-motion. The pattern survived multiple draft reviews before being caught — silent inheritance of substrate parameters across project pivots is a recurring failure mode.

4. **Across-loop apex pose-determinism interpreted as failure.** Initial calibration found apex frames bit-identical across loops; mistakenly flagged as needing per-frame jitter. Resolved by the curriculum-as-variation framing: Stage A bit-identicity is the baseline state, not a failure. **This was a reasoning correction, not a substrate fix.** The substrate property (Stage A bit-identicity) was correct as designed; the initial interpretation as "something needing jitter" was the error. Distinct in character from the other seven findings: substrate-as-feature interpreted as substrate-as-bug, not a substrate issue requiring correction. v1 designers should expect both kinds of findings (real substrate issues and substrate-as-feature interpretive errors) and apply different remediation patterns to each.

5. **Camera elevation bug from `forceAction=True`.** Placed agent at non-floor y values; produced bird's-eye-view renders alternating with eye-height renders. Same root cause as finding 6.

6. **Floor-y derivation.** Required explicit modal-y across `GetReachablePositions` results; cannot inherit from arbitrary route waypoints. Now in spec §5.7.

7. **DiningTable view-through at original pose.** Original viewing pose caught the LivingRoom doorway in FOV, producing 0.045 cosine units of out-of-scope perturbation response. Resolved by pose adjustment (h117.57°), reducing to ~0.014.

8. **Cross-room visual leakage from `RandomizeMaterials(inRoomTypes=[X])`.** API correctly scopes the *material change* to room X, but rendering of unmodified items against a changed lighting environment produces small but nonzero pixel-level variation in out-of-scope rooms. The substrate's locality and the API's locality are two different properties. Now in spec §5.8 as a general substrate-property check.

---

## 5. The five SCAFFOLDING-threshold lessons (§15 institutional pattern)

Five absolute-magnitude thresholds did not survive empirical contact. The §15 principle ("absolute-magnitude SCAFFOLDING gates anchored before empirical data are structurally vulnerable to scale mismatch") was promoted to default discipline during v0 on the strength of this pattern.

1. **Pixel-cosine preflight threshold.** Replaced with relative ratio + DINOv2 differential. The 300×300-pixel-RGB metric compressed into a dynamic range too tight to discriminate locality.
2. **S1 0.02 contrast floor.** Tripped on every run after substrate-adjustment cycle established empirical contrast in the 0.006–0.012 range. Dropped; replaced with the §8.4 perturbed/control ratio + statistical-distinguishability.
3. **Loop-length 10% threshold.** Evaluated against actual downstream rep-bin arithmetic rather than the round-number threshold; accepted +13.9% as worth the locality fix.
4. **§8.4 0.05 absolute floor.** Anchored to spec §5.1/5.2 pair-cosine pattern (~0.5 magnitude), wrong scale for the ~0.01-magnitude within-item perturbation responses `RandomizeMaterials` produced. Replaced with the Wilcoxon signed-rank Reading C formulation.
5. **§8.7a G2.T2 0.5 absolute floor.** Anchored to "what a 30–40% cross-stage cosine drop would produce in log_var" — but the empirical perturbation is 1–2% cross-stage cosine drop. The 0.5 threshold was anchored to the wrong perturbation scale. Restructured to trajectory-direction + shape + differential at loop 100.

The pattern: anchoring a threshold to a scale that has not been empirically observed creates a gate that either trips trivially or never trips. The discipline replacement: prefer statistical-distinguishability tests, prefer ratio comparisons against measured controls, prefer trajectory criteria over single-point thresholds when the architectural claim is about evolution rather than a static property.

**Sixth pattern instance, methodological.** The pattern also applies to thresholds introduced mid-experiment for verdict-disambiguation, not just to architectural-strength gates from the original instructions document. The Pearson 0.7/0.3 thresholds initially proposed for the per-ordinal cross-loop diagnostic were SCAFFOLDING in the same way the original gates were; the discipline of preferring binary-decidable observations (ord 9-10 pixel-MD5 identicity → reading-(i) discriminator) over correlation thresholds applies equally. The §15 discipline applies whenever a threshold is introduced, regardless of whether it appears in the spec at design time or in a CC instruction mid-batch.

---

## 6. The localised-widening question (Sofa ord-1)

The variance-by-ordinal disaggregation surfaced one observation v0 did not resolve: Sofa ord-1 drift loop 30 → 100 is **+0.056** — the only positive (widening) drift across all (item, ordinal) pairs in the disaggregated data. Bed/TV/Dresser have monotonic-narrowing trajectories at every ordinal; Sofa narrows at ords 0 and 2–10 but widens at ord 1.

Two readings:

- **(a) Localised architectural signal.** The variance mechanism may operate at finer (item, ordinal) granularity than the aggregate G2.T2 measured, and aggregate "narrowing" trajectories may conceal pockets where the mechanism behaves as designed. Disaggregation discipline revealed structure that, if reproducible, would refine the V2 verdict toward "the mechanism works in places."
- **(b) Noise at n=1.** Sofa ord-1 is one (item, ordinal) pair out of 44 sampled. Below the §3.6 small-sample threshold for drawing conclusions. Could be a stochastic excursion that disappears under repetition.

v0 cannot disambiguate. The Bed coupling result establishes V2 independent of Sofa ord-1's status, but the question is load-bearing for v1 design: an evaluation framework that ignores per-(item, ordinal) localisation could miss a real architectural signal v0 inadvertently surfaced.

**The cheapest v1 disambiguation requires two prerequisites, both load-bearing.** (1) **Methodology requirement:** v1 evaluation operates at per-(item, ordinal) granularity from design time, not aggregate-then-disaggregate — aggregate gates will continue to average over locally-specific signals. (2) **Substrate requirement:** v1's perturbation regime is strong enough that the architectural prediction produces visible effect size at per-(item, ordinal) sample counts — at v0's perturbation magnitude, even per-(item, ordinal) evaluation leaves n=1 below the §3.6 noise floor, so a stronger perturbation regime is needed for the question to be decidable. Both prerequisites are necessary; meeting only one leaves the question undecidable.

With both prerequisites met: run a comparable Stage A → Stage B substrate with a stronger perturbation regime and check whether (a) Sofa ord-1 widens again at the same (item, ordinal) cell, (b) widening appears at random (item, ordinal) pairs, or (c) no widening appears anywhere. (a) supports the localised-architectural-signal reading; (b) supports the noise reading; (c) is informative either way (perturbation magnitude matters or the mechanism is absent).

---

## 7. Disambiguating-evidence framework for v1

The v0 verdict is V2 — but V2 in a specific form ("variance updates propagate as a coupled global shift") that constrains v1 in different ways depending on which underlying cause produced the coupling. Three plausible underlying causes; v1 design should disambiguate which.

### 7.1 Cause: pooled-readout topology collapses per-input information before any output head sees it

The variance head produces a single scalar log-variance per predicted step (spec §3.3, §4.1). The original §7.1 framing argued that K scalars per step gave gradient backprop no architectural pathway to route variance updates per position. BCDD evidence (2026-05-16) showed this framing was incomplete: under v0's predictor architecture, the body collapses all W input positions to a single pooled vector (`last_token = x[:, -1, :]`, src/predictor/inner_pam.py:83) BEFORE any K-step output is generated. The output projection then fans the single pooled vector to K*(d+1) values via one dense linear layer (src/predictor/inner_pam.py:64, 84). Both the mean head and the variance head sit downstream of this collapse.

**The mechanism, made precise by BCDD.** Between loop-30 and loop-100 checkpoints, the body's pooled `last_token` cosine on bit-identical Bed ord 9/10 inputs is 0.754 — a ~24.6% representational drift on the same input across training. The drift magnitude is essentially input-agnostic: across all 22 invariant input windows tested (11 ordinals × 2 loop windows), `last_token` cosine std is 0.0004. Body parameters have accumulated training updates that shift the pooled representation roughly uniformly regardless of input.

Both downstream heads inherit this drift. The mean head, despite having ~1024× more parameters per output step than the variance head, exhibits the same uniform-drift signature on the same invariant inputs: mean drift at ords 9 and 10 falls inside the 2σ band built from all 11 ordinals' mean drifts (band [0.358, 0.413], ord 9 = 0.377, ord 10 = 0.369). Variance drift exhibits the same pattern (the v0 finding). Both heads produce input-blind drift because they both read from a pooled vector that has drifted in an input-agnostic way.

**This rules out the original §7.1 mechanism as dominant.** If the variance head's K-scalar parameter count were the dominant coupling source, the mean head — with K·d parameters per step — should respond differently to input variation across ords. It does not. Both heads pattern-match v0's uniform-drift signature.

**Disambiguating evidence (revised).** A v1 that retains v0's pooled-readout topology and addresses only the variance head's parameter count (per-dimension variance, mixture-density variance) should reproduce the coupling result on invariant inputs, because the upstream pooled vector still drifts uniformly. A v1 with a readout topology that preserves K-positional structure through to the output stage (K learnable output queries with cross-attention into the encoder output; K positional read-out tokens; per-K output heads) should produce per-input differentiation in mean drift at invariant ordinals if this is the cause. The variance representation change is downstream of the readout change and well-defined only once K position-distinguished hidden vectors exist.

**Per-K coefficient of variation at invariant ordinals (BCDD test C).** Mean drift CoV across K at ord 9 is 0.189; at ord 10, 0.183. The per-K profile shows a coherent horizon-dependent pattern (largest drift at small k, smallest at k=13) rather than random per-row scatter. This is consistent with a single `last_token` shift mapped through output-projection rows that have different alignments with the drift direction — not independent per-row drift in the output projection itself. Most of the coupling signal originates upstream of the output projection.

**Evidence source.** `results/v1_design/bcdd_results.json` (canonical); `results/v1_design/bcdd_results_failed_preflight.json` (forensic record of the protocol-mismatch trace that surfaced the K-back window construction and mean-over-K statistic before sanity-check pass). BCDD script at `scripts/bcdd.py`.

### 7.1.PRESERVED — Original framing (superseded by BCDD evidence 2026-05-16)

**Note (2026-05-16):** The original §7.1 framing identified the variance head's K-scalar parameter count as the architectural pathway gap. BCDD diagnostic on v0 checkpoints (loop-30 vs loop-100) demonstrated that the mean head — with K·d parameters per output step — exhibits the same uniform-drift signature on bit-identical inputs as the variance head. This rules out variance-head topology as the dominant coupling source. The revised §7.1 above identifies the body's pooled-readout collapse as the upstream cause; the original framing is preserved here as institutional memory of how the diagnosis evolved.

*Original section heading: 7.1 Cause: per-K-step scalar isotropic variance is wrong*

The variance head produces a single scalar log-variance per predicted step (spec §3.3, §4.1). One scalar per step is one parameter per step that must absorb everything the per-K-step gradient delivers, regardless of which input position drove the surprise. When all variance updates flow through one scalar per step, the gradient produces a single direction of change in the predictor's variance representation — which propagates as a uniform shift across positions.

**The mechanism, made precise.** The variance head has K scalar parameters per predicted step, no per-position parameters. Gradient backprop through the path-prediction loss has no architectural pathway to route variance updates to specific positions; whatever update gradient delivers, it applies uniformly. A variance head with per-position parameters would create the architectural pathway gradient currently lacks. The coupling result is therefore not just an observation about training dynamics — it is what the architecture's parameter topology *must* produce, because the variance representation lacks the parameters needed for non-uniform updates.

**Disambiguating evidence.** A v1 with anisotropic (per-dimension) variance, or a v1 with per-position variance, or a v1 with mixture-density variance (mixture of Gaussians per step) should produce per-position variance evolution if this is the cause. A v1 that retains the scalar isotropic variance head but addresses a different cause should reproduce v0's coupling.

---

### 7.2 Cause: DINOv2-frozen substrate doesn't isolate per-item information

DINOv2 was trained on internet-scale image data; its CLS token is not optimised for the architectural claim being tested. The per-item information may be represented in DINOv2's embeddings in a way that is entangled with global scene information at a level the predictor cannot extract via gradient descent on the path-prediction loss.

**Disambiguating evidence.** A v1 with an encoder trained for the kind of recurring-trajectory recognition Inner PAM needs (SIGReg-from-scratch is the candidate v0 named, but not the only one) should produce per-position variance evolution if this is the cause. The SIGReg path needs operational scoping: pretrain regime, evaluation criteria, integration timeline. A v1 with DINOv2 but a different perturbation regime should reproduce v0's coupling if this is the cause.

### 7.3 Cause: the perturbation regime was too weak

`RandomizeMaterials` produces ~0.01-magnitude cross-stage cosine drops at perturbed items. The architectural mechanism may operate as designed at larger perturbation magnitudes but produce subthreshold responses at the v0 magnitude. The coupling result on Bed could then be specifically a small-perturbation-regime artefact: when the input-driven variance signal is small enough, gradient coupling from other items dominates.

**Disambiguating evidence.** A v1 with a stronger perturbation regime (per-object material replacement, asset replacement at fixed coordinates, hand-built texture swaps producing 0.05–0.1-magnitude cross-stage cosine drops) should produce per-position variance evolution if this is the cause. A v1 with the same perturbation magnitude but a different encoder or different variance representation should reproduce coupling if this is the cause.

### 7.4 Combinations

The three causes are not mutually exclusive. v1 design likely needs to address more than one simultaneously, but should be structured so each can be evaluated independently — single-variable discipline applied to architectural-direction selection, not just to hyperparameter tuning. A v1 that bundles "new encoder + new perturbation + new variance head" and produces clean per-position variance evolution does not tell us which intervention earned the result.

---

## 8. v1 candidate directions (questions, not commitments)

Four candidate directions surfaced during v0. Each maps to a primary disambiguating question; none is committed.

### 8.1 SIGReg-from-scratch encoder

**Primary question.** Does an encoder pretrained for trajectory-shape sensitivity produce different variance behaviour than DINOv2-frozen on the same architectural test?

**Operational scoping required.**
- What "from-scratch" means concretely: pretrain a new encoder before v1 begins; co-train encoder + predictor; replace DINOv2 in place with a SIGReg-trained head; something else.
- Pretrain regime: what data, what loss, what evaluation criteria, what compute envelope.
- Integration: does Inner PAM's path-prediction loss provide enough signal to co-train the encoder, or is pretraining required?
- Substrate-verification protocol updates: §5's thresholds were calibrated against DINOv2; a new encoder needs its own calibration.

### 8.2 Per-position or anisotropic variance representation

**Primary question.** Does replacing per-K-step scalar isotropic variance with a richer variance representation (per-dimension, per-position, mixture-density) eliminate the coupling result?

**Operational scoping.** Smaller architectural change than 8.1; same encoder and environment. The loss formulation changes; the predictor architecture changes in the variance head. Variance calibration metrics (M2) need rework for richer variance representations. Likely the cheapest v1 path if the cause is §7.1.

### 8.3 Cleaner perturbation mechanism

**Primary question.** Does a perturbation regime with verified rendered-frame locality and larger magnitude produce per-position variance evolution?

**Operational scoping.** Per-object material setting where the AI2-THOR API supports it; asset replacement at fixed coordinates with rendered-frame verification (the v0 Phase 3 preferred mechanism, which preflight may or may not pass); hand-built texture swaps with offline rendering. The §5.8 cross-scope locality check becomes a load-bearing gate for any of these. Smallest architectural change; most direct test of §7.3.

### 8.4 Synthetic-stream alternative

**Primary question.** Does the architectural mechanism produce per-position variance evolution on a substrate where every property of the input stream is controlled?

**Operational scoping.** Bypass AI2-THOR. Construct procedural trajectories in a synthetic embedding space (e.g. low-dim manifolds with known recurring-pattern structure) and run Inner PAM directly. Loses ecological validity; gains complete substrate control. Useful as a sanity check on the other v1 paths, or as a stand-alone v1 if the architectural claim is the priority over environment-grounding.

---

## 9. What carries forward, what does not

**Carries forward to v1.**
- The operational discipline (CODING_STANDARDS, research_operations).
- The eight substrate findings as default checks (including the substrate-as-feature-vs-bug distinction from finding 4).
- The five SCAFFOLDING-threshold lessons + the sixth methodological corollary as default discipline.
- Spec §§1–3 (the architectural commitments, except §2.2's per-item-variance-response prediction which v1 will need to re-state).
- Spec §5 (substrate verification), including the new §5.8.
- The disaggregation discipline (per-(item, ordinal), per-loop, per-K-step) — without it, v0's verdict would have been ambiguous.
- The disambiguating-diagnostic discipline — the per-ordinal cross-loop input variation check was a one-minute resolution of an otherwise multi-page ambiguity.

**Does not carry forward.**
- Phase 1's 100k substrate-degenerate baseline embeddings. Not load-bearing for v0 verdict; not load-bearing for v1.
- The specific v0 frame budgets, checkpoint cadences, and gate thresholds. v1 will derive its own from its substrate.
- The DINOv2-frozen-as-the-only-substrate assumption. v1 candidate 8.1 explicitly questions this.
- The per-K-step scalar isotropic variance commitment. v1 candidate 8.2 explicitly questions this.

**Open questions for v1 spec discipline (separate chat).**
- Which v1 candidate(s) to pursue and in what order.
- How to structure single-variable discipline across architectural-direction selection.
- Whether v1 retains the AI2-THOR-ProcTHOR substrate or pivots.
- Whether v1's perturbation regime can be designed to disambiguate §§7.1–7.3 in a single experiment.
- Sofa ord-1: what evaluation framework keeps per-(item, ordinal) localisation visible?

---

## 10. Push hold

Push hold remains in effect on both repos through v1 design. v0 artifacts stay local until v1 scope is settled and a decision is made about which v0 outputs (substrate findings, SCAFFOLDING-threshold lessons, §5.8 addition) belong in any external-facing artefact.

---

*End of v0-closing document. Companion to `WEFT_INNER_PAM_v0_Spec.md`, `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md`, and `HANDOFF.md`. v1 design begins in a separate chat with its own spec discipline.*
