# Weft Inner PAM v1 — Specification (Pass 1: §§1–6)

**Status.** Architecture-and-claims layer (§§1–6) for review. Implementation specification (§§7–11) follows in pass 2 as a CC-handoff document.

**Companion documents.** `WEFT_INNER_PAM_v0_CLOSING.md` (v0 institutional memory, including the BCDD-revised §7.1 and the disambiguating-evidence framework); `WEFT_INNER_PAM_v2_DESIGN_INTAKE.md` (the body-family question v1 explicitly defers to v2). v1 design intake brief is upstream input; this document is the architectural specification that intake produced.

**Scope conditioning.** v1 is a transformer-family verdict on the Inner PAM architectural claim, not an Inner PAM architectural verdict per se. The body family and W=16 sliding window pattern are retained from v0 unchanged; the body-family question is v2's primary architectural question. v1's verdict, whether positive or negative, conditions on these inherited commitments.

---

## 1. Verdict template and architectural claim

### 1.1 What v1 tests

v1 tests whether the Inner PAM architectural claim (§2 below) produces its predicted behaviour — co-primary mean and variance differentiation per input across (item, ordinal) — under three architectural interventions that v0's evidence supports:

1. **Readout topology change.** Replace v0's pooled single-vector readout (`last_token = x[:, -1, :]`) with K learnable output queries that cross-attend into the encoder body's W-positional output. Each K-th query produces its own position-distinguished hidden vector. (BCDD evidence: v0's pooled readout was the dominant coupling locus.)
2. **Variance representation change.** Replace v0's per-K-step scalar isotropic variance with per-K-position parameters matching the readout. Each K-th query's output projects to (mean, log-variance) pair where the variance representation has its own per-K parameters. (Closing-doc §7.1 revised: per-position variance is well-defined only once K position-distinguished hidden vectors exist.)
3. **Perturbation strength increase.** Replace v0's `RandomizeMaterials` mechanism (producing ~0.01-magnitude cross-stage cosine drops) with a regime producing 0.05–0.1-magnitude drops. (Closing-doc §6 prerequisite for per-(item, ordinal) decidability; §7.3 disambiguating-evidence framework.)

These three interventions are tested as a bundle (the v1 primary arm) plus two ablation arms that attribute the result to specific interventions.

### 1.2 Verdict template

v1's verdict has the form:

> **V[n] — [verdict label], with [transformer body family and W=16 sliding window pattern retained from v0; body-family question deferred to v2].**

Six verdict categories, each producing different v2 design implications (see `WEFT_INNER_PAM_v2_DESIGN_INTAKE.md` §6):

- **V1A — Architectural claim supported under proper configuration.** Primary arm produces co-primary mean and variance differentiation; ablations attribute cleanly (Ablation 1 isolates variance representation, Ablation 2 isolates readout topology). v2's body-family test becomes a refinement question against this baseline.
- **V1B — Architectural claim supported; variance representation not load-bearing.** Primary succeeds; Ablation 1 also produces variance differentiation despite scalar variance; Ablation 2 reproduces v0 coupling. The readout topology change is load-bearing but variance representation richness is not. v2 inherits the question of why per-position variance wasn't load-bearing as a secondary question.
- **V1C — Architectural claim falsified under proper configuration.** Primary fails despite BCDD-indicated corrections; Ablation 2 cleanly reproduces v0 coupling; Ablation 1 also fails. v2's body-family change becomes the primary architectural question with stronger evidence than thesis-tension alone provides.
- **V1D — Mixed / partial result.** Specific D-class sub-outcomes (mean works without variance; variance works without mean; per-(item, ordinal) heterogeneity at scale). v2 design proceeds case-by-case based on the specific D-class outcome.
- **V1E — Perturbation-as-sufficient.** Primary succeeds; *both* Ablation 1 (scalar variance + new readout) and Ablation 2 (v0 readout + scalar variance) also succeed under strong perturbation. The architectural corrections (readout topology, variance representation) were not load-bearing; perturbation strength alone was sufficient to produce per-(item, ordinal) differentiation. This is fundamentally distinct from V1B: V1B isolates one architectural correction as non-load-bearing; V1E isolates *both* as non-load-bearing, meaning the entire architectural-corrections framing v1 inherited from BCDD was misweighted relative to substrate strength. v2 inherits "was the architectural-corrections framing wrong" as a primary question, larger in scope than V1B's secondary question.
- **V1P — Protocol-failure outcome.** v1 architectural verdict deferred pending substrate correction; substrate findings recorded for v2 inheritance. Distinct from V1A–V1E in that the architectural claim is not tested by v1's run.

**SCAFFOLDING note on verdict-category thresholds.** The specific thresholds for "what fraction of (item, ordinal) pairs constitutes which verdict" — e.g., what fraction of pairs showing co-primary differentiation constitutes V1A vs V1D — are SCAFFOLDING parameters set empirically at the instructions-document level, not absolute thresholds at the spec level. v0's §15 lesson (absolute-magnitude SCAFFOLDING gates anchored before empirical data are structurally vulnerable to scale mismatch) applies here: verdict thresholds are calibrated against measured per-(item, ordinal) distributions produced by v1's runs, not declared a priori. The instructions document specifies the calibration procedure; this spec specifies only the verdict-category structure.

### 1.2.1 Verdict resolution timing

v1's verdict is assigned at a specific point in the experimental workflow, not iteratively:

1. **All three arms complete training.** Primary, Ablation 1, Ablation 2 reach the planned end-of-Stage-B checkpoint (or stop at a unconditional-failure trigger).
2. **Per-(item, ordinal) evaluation runs across all arms.** All metrics specified in §5 are computed at per-(item, ordinal) granularity for each arm.
3. **One round of disaggregating diagnostics runs if aggregate evaluation is ambiguous.** If the per-(item, ordinal) pattern doesn't cleanly map to a verdict category (heterogeneous outcomes; signal at edge of significance; structured outliers requiring characterisation), one round of disaggregating diagnostics is run to clarify. This round is bounded: it characterises what the existing evaluation produced, but does not collect new data, retrain, or extend training.
4. **v1 chat produces verdict-assignment recommendation.** The recommendation maps the per-(item, ordinal) evaluation to one of V1A–V1P with reasoning. The recommendation is presented to the reviewer chat (separate context) for adversarial review per research_operations §2.2.
5. **Reviewer chat issues verdict.** The verdict is recorded in v1's closing document and triggers v2 design intake refinement.

Further diagnostics after verdict assignment are v2 intake material, not v1 verdict revision. This is a deliberate constraint: v0's experience suggested that without explicit verdict-resolution timing, the disaggregation-and-clarification process can extend indefinitely. v1 commits to a single round of disambiguating diagnostics post-evaluation, and post-verdict diagnostics flow to v2.

### 1.3 What v1's verdict cannot establish

v1's verdict establishes results for the *transformer body family with W=16 sliding window pattern*. v1 does not test, and v1's verdict cannot speak to:

- Whether body-family change (recurrent, state space, convolutional) produces different results. This is v2's primary architectural question.
- Whether the W=16 sliding window pattern is the right temporal-context mechanism for this task. This is tied to the body-family question and inherited by v2 alongside it.
- Whether DINOv2 is the right encoder substrate. v0 verified DINOv2 against the §5 substrate verification protocol; v1 inherits that verification. Encoder substitution is a deferred question.
- Whether AI2-THOR + ProcTHOR is the right environment. The environment supported v0's experiment well enough to produce a clean verdict; v1 inherits the substrate decision.

These inherited commitments are listed explicitly in §2 below under "inherited from v0, not tested by v1."

### 1.4 Co-primary discipline

v1 tests mean differentiation and variance differentiation as *co-primary* claims. Both must hold for the architectural claim to be supported; failure of either falsifies v1's architecture for this task.

**This is a deliberate framing choice, not a procedural sequencing.** Mean differentiation and variance differentiation are co-falls and co-rise predictions of the same architectural claim (shape representations form through repetition and produce per-input responses in both predicted means and predicted variances). Treating one as primary risks v1 declaring victory on the mean test while missing variance failures, or vice versa.

The co-primary commitment also implicitly binds v1 to per-position variance representation. Mean-primary framing would allow scalar variance as a v1 choice; co-primary framing requires the variance representation to have per-K-position parameters matching the readout, because variance differentiation requires the parameter pathway to express it.

---

## 2. Architectural commitments

v1 has three categories of architectural commitments: those tested by v1's interventions, those inherited from v0 unchanged, and those explicitly deferred to v2. The category distinctions are load-bearing — v1's verdict scope depends on which architectural commitments are tested vs inherited vs deferred.

### 2.1 Architectural commitments tested by v1

1. **Readout topology produces K position-distinguished hidden vectors.** K learnable output queries cross-attend into the encoder body's W-positional output (§7 implementation specification). Each query produces its own (B, hidden) output token; per-K parameters live in the queries and the cross-attention. This commitment replaces v0's pooled-`last_token` readout.

2. **Variance representation has per-K-position parameters matching the readout.** Each K-th query's output projects to (mean, log-variance) where the variance representation has per-K-position parameters (specific form to be specified in §3.3). This commitment replaces v0's per-K-step scalar isotropic variance.

3. **Perturbation regime produces 0.05–0.1-magnitude cross-stage cosine drops at perturbed items.** Specific mechanism (asset replacement, hand-built texture swaps, or alternate-scene approach) to be determined by §7 preflight; the architectural commitment is to the *magnitude* of perturbation, not to a specific mechanism. This commitment replaces v0's `RandomizeMaterials` regime.

4. **Co-primary mean and variance differentiation.** Both predicted means and predicted variances must show per-(item, ordinal) differentiation for the architectural claim to be supported. Variance differentiation cannot be deferred or treated as secondary.

5. **Per-(item, ordinal) evaluation granularity from design time.** All evaluation metrics, gates, arm-comparison criteria, and verdict thresholds operate at per-(item, ordinal) granularity by default. Aggregation operations require explicit justification (§5 below).

### 2.2 Architectural commitments inherited from v0, not tested by v1

1. **Shape as the unit of representation** (v0 spec §2.1). Carries forward unchanged.

2. **Repetition as the learning signal** (v0 spec §2.2 first clause, retaining "shape representations form through repetition" but not the per-input-variance-response prediction). The variance-response prediction is reformulated as the v1 hypothesis in §3 below.

3. **Continuous-time, single-pass processing** (v0 spec §2.3). Carries forward unchanged.

4. **Always-on architecture** (v0 spec §2.4). Carries forward unchanged.

5. **Modality-agnostic output structure** (v0 spec §2.5 first clause, retaining the architectural commitment to (mean, variance) pair output while replacing v0's specific scalar-variance form). The new variance form is specified in §3.3.

6. **Shapes live in weights, instances live in bank** (v0 spec §2.6). Carries forward unchanged.

7. **Shape recall and instance recall same function** (v0 spec §2.7). Carries forward unchanged.

8. **Transformer body family.** Encoder is `nn.TransformerEncoder` with the same architectural family as v0 (multi-head self-attention, GELU activation, norm-first, batch-first). Specific hyperparameters in §7.

9. **W=16 sliding window pattern.** Temporal context for prediction is a sliding window of W=16 input embeddings. The window pattern is tied to the body-family choice (transformer-natural) and inherited from v0 alongside it.

10. **Additive positional embeddings** (`nn.Embedding(window_w, hidden)` added to input projection). Position information is encoded as additive metadata on input embeddings, in the transformer pattern v0 used.

11. **DINOv2-Large frozen encoder.** Encoder substrate verified by v0 against the §5 protocol; carries forward unchanged.

12. **AI2-THOR + ProcTHOR environment.** Substrate environment carries forward, with the substrate findings v0 produced as default checks.

### 2.3 Architectural commitments deferred to v2

1. **Body family.** Whether the transformer body family is the right architectural commitment for this task. v2's primary architectural question.

2. **Temporal-context mechanism.** Whether the W=16 sliding window pattern is the right temporal-context mechanism. Tied to the body-family question and inherited by v2 alongside it.

3. **Positional information mechanism.** Whether additive positional embeddings (transformer convention) or recurrent dynamics (recurrent convention) is the right way to encode sequential order. Tied to the body-family question.

4. **Bounded memory mechanism.** Whether hard-window memory (transformer convention) or state-compressed memory (recurrent/state-space convention) is the right way to handle bounded memory. Tied to the body-family question.

### 2.4 Discipline note

Category 2.2 commitments are not architectural endorsements of v0's choices. They are *deliberate deferrals* — v1's evidence base supports changing categories of architectural commitments where v0's evidence directly implicated them (readout, variance, perturbation), and defers categories of commitments where v0's evidence did not directly implicate them (body, windowing, encoder, environment).

This is the single-variable discipline of v0 applied to architectural-direction selection: change what evidence supports changing, defer what evidence doesn't directly implicate, never bundle architectural changes that the evidence doesn't disentangle.

---

## 3. v1 hypothesis statement

### 3.1 The hypothesis

**Inner PAM v1 hypothesis (with transformer body family and W=16 sliding window pattern retained from v0):**

> Shape representations form through repetition and produce per-input responses in both predicted means and predicted variances when (a) the readout topology preserves K-positional structure through to per-K output heads, and (b) the variance representation has per-K-position parameters matching the readout. The architectural claim requires both differentiations to hold; failure of either falsifies the v1 architecture for this task, with the disaggregated falsification (which differentiation failed at which (item, ordinal)) informing v2 design.

### 3.2 What the hypothesis predicts

For an (item, ordinal) pair where the substrate provides per-input variation (the item is perturbed across Stage A → Stage B; the ordinal samples a viewing position that captures perturbation-induced visual change), v1's predictor should produce:

- **Per-input mean response.** Predicted means at this (item, ordinal) should differ in Stage B relative to Stage A in a way that reflects the perturbation-induced input change. The mean prediction should track the centreline of the perturbed visual trajectory.
- **Per-input variance response.** Predicted variances at this (item, ordinal) should differ in Stage B relative to Stage A in a way that reflects the perturbation-induced uncertainty change. The variance prediction should reflect trajectory thickness around the perturbed centreline.

For an (item, ordinal) pair where the substrate is bit-identical across Stage A → Stage B (the item is unperturbed or the ordinal samples a viewing position that doesn't capture perturbation-induced change), v1's predictor should produce:

- **Mean stability across loops.** Predicted means should remain stable (cross-loop mean drift indistinguishable from numerical noise) on bit-identical inputs.
- **Variance stability across loops.** Predicted variances should remain stable (cross-loop variance drift indistinguishable from numerical noise) on bit-identical inputs.

The combination — differentiation where input varies, stability where input is bit-identical — is what v0's pooled-readout architecture *failed* to produce (BCDD evidence: variance drift on bit-identical Bed ords 9/10 was indistinguishable from variance drift on input-varying ords). v1's architectural interventions are designed to enable this combination.

### 3.3 Per-position variance representation specification

The variance representation has per-K-position parameters. Three candidate forms, in increasing expressivity:

**Form 1 — Per-K scalar (isotropic).** Each of K output queries projects to (μ_k, log_σ²_k) where μ_k ∈ R^d is the mean and log_σ²_k ∈ R is a scalar log-variance. K scalars total; per-K parameters in the variance head; isotropic variance per output step. This is the minimum architectural change from v0's variance head that respects the co-primary commitment — v0 had K scalars from a single pooled vector; this has K scalars from K position-distinguished vectors.

**Form 2 — Per-K per-dimension (anisotropic).** Each of K output queries projects to (μ_k, log_σ²_k) where log_σ²_k ∈ R^d is a d-dimensional log-variance vector. K · d variance parameters total; per-K and per-dimension parameters in the variance head; anisotropic variance per output step.

**Form 3 — Per-K mixture density.** Each of K output queries projects to (M Gaussian components per K) with per-component mean, per-component log-variance, and mixing weights. K · M (d + d + 1) parameters total. Richer variance representation; addresses the spec §4.2 commitment about Gaussian assumption at divergence points.

**v1 commits to Form 1 (per-K scalar) as the variance representation.** Form 1 is the minimum architectural change that respects co-primary discipline: v0's variance head had K scalars from a pooled vector; v1's Form 1 has K scalars from K position-distinguished vectors. The architectural difference is the upstream readout, not the variance head's capacity.

**Forms 2 and 3 are explicitly held in reserve for v2, on the following discipline reasoning.** If v1's Form 1 succeeds (V1A or V1B verdict), Forms 2 and 3 become v2 *refinement* questions — does richer variance representation improve on the Form 1 baseline, and is that improvement worth its operational cost? If v1's Form 1 fails on variance differentiation specifically (V1D sub-outcome where mean works without variance), Forms 2 and 3 become v2 *diagnostic* questions — does the failure reflect insufficient variance representation capacity, or something deeper about how variance learns on this substrate? Either way, deferring Forms 2 and 3 keeps v1's verdict clean about the role of the readout topology change separate from the role of variance representation richness. Bundling them into v1 would reintroduce the architectural-bundle confound v1's 3-arm structure is designed to avoid.

---

## 4. Mechanism walkthrough

### 4.1 The path-prediction loss

v1 retains v0's path-prediction loss formulation with the variance representation updated for Form 1:

For window W = (w_0, ..., w_{W-1}) and K-step prediction targets Y = (y_0, ..., y_{K-1}):

```
L_k = 0.5 · (||y_k - μ_k||² / σ²_k + d · log σ²_k)
L = Σ_k L_k / K
```

where μ_k ∈ R^d and log_σ²_k ∈ R are produced by the predictor's K-th output query's output projection. This is the per-K-step Gaussian NLL averaged across K.

For Form 2 (anisotropic), the loss becomes:

```
L_k = 0.5 · (Σ_i (y_k,i - μ_k,i)² / σ²_k,i + Σ_i log σ²_k,i)
```

For Form 3 (mixture density), the loss becomes the negative log-likelihood under a mixture-of-Gaussians per K-step.

v1 primary uses Form 1; the loss formulation above applies.

### 4.2 The architectural pathway for per-input differentiation

v1's architectural pathway is *designed* to support per-input differentiation through the following mechanism; whether the mechanism produces the predicted behaviour is the v1 hypothesis tested by training and evaluation. Specifying the pathway is not asserting the pathway works — the pathway is what v1 is testing.

Under v1's readout topology (K learnable output queries with cross-attention into the encoder body's W-positional output), per-input differentiation is designed to operate through the following sequence:

1. **Input encoding.** The encoder body processes the W-positional input window, producing W position-distinguished hidden vectors. Per-input information is preserved in the hidden vectors (v0 verified this — TV/Dresser/Sofa per-pose differentiation showed the body produces input-distinguished representations).

2. **Cross-attention readout.** Each of K learnable output queries cross-attends into the W hidden vectors. Each K-th query has its own learnable parameters; gradient updates to one query do not directly update other queries (per-K parameter isolation). The K-th query produces its own (B, hidden) output token by attending differently than other queries.

3. **Per-K output projection.** Each K-th query's output token projects to (μ_k, log_σ²_k) via a shared output projection. Per-K differentiation is established upstream of this projection; the projection is a position-agnostic mapping from hidden vector to (mean, variance).

4. **Loss-driven gradient signal.** The per-K-step loss provides per-K gradient signal. Each K-th query receives gradient updates specifically for its K-step prediction error, with no architectural pathway for gradient updates to other queries to interfere with this query's parameters.

This is the architectural pathway BCDD's evidence identified as missing in v0. v0's pooled-`last_token` readout collapsed all input-position information to a single vector before any K-step output was generated; v1's K output queries preserve K-positional structure through to the output stage.

### 4.3 Why this pathway addresses the BCDD-identified pathology

BCDD's evidence had three components:

- **Test A:** `last_token` cosine across checkpoints on bit-identical inputs was 0.754 (input-agnostic body drift across training).
- **Test B:** Mean drift at bit-identical ords was indistinguishable from mean drift at input-varying ords (input-blind drift signature).
- **Test C:** Per-K coefficient of variation at bit-identical ords was 0.189 / 0.183 (a single `last_token` shift mapped through different output-projection rows with different alignments).

Under v1's K output queries readout, the corresponding architectural responses are:

- **Test A pathology — design response.** The body's pooled vector is no longer the single source of per-K predictions. K output queries each have their own parameters; the architectural prediction is that even if the body drifts uniformly across training, the K queries can develop differentially. Whether this prediction holds is what v1 tests.
- **Test B pathology — design response.** Per-K differentiation upstream of the output projection is designed to let the K-th query for an input-varying ord develop differently from the K-th query for a bit-identical ord (or, equivalently, the same query attending to different input contexts to produce differentiated outputs). Whether this prediction holds is what v1 tests.
- **Test C pathology — design response.** Per-K parameter isolation is designed to make each K-th query's drift across training independent of other queries' drift, breaking the "single shift mapped through K projection rows" pattern. Whether this prediction holds is what v1 tests.

### 4.4 Why this pathway requires per-position variance to be co-primary

The architectural pathway above gives K an upstream position dimension where parameters live separately per K. The mean head reads from K position-distinguished hidden vectors via per-K output projections; per-K mean differentiation has the parameter pathway to learn.

For variance differentiation to have the same parameter pathway, the variance representation must also have per-K-position parameters. Under Form 1 (per-K scalar), this is satisfied: each K-th query's output projects to a scalar log-variance with K · 1 variance parameters total, but the K parameters are per-K-position (one scalar per query). Under v0's scalar variance from a pooled vector, the K scalars all read from the same source vector and inherit any drift in that source vector — which is the M3 pathology BCDD identified.

If v1 used scalar variance from K position-distinguished hidden vectors but did *not* have per-K variance parameters (e.g., the same single variance parameter shared across all K queries), the variance head would re-introduce the BCDD pathology at a different layer. Per-K variance parameters are necessary for the architectural pathway to be coherent across mean and variance heads.

This is the architectural reason co-primary discipline implicitly commits v1 to per-position variance representation, as flagged in §1.4.

### 4.5 What v1's mechanism doesn't address

Three things v1's architectural mechanism does *not* address, surfaced explicitly:

1. **The body's input-blind training-time drift itself is not removed.** BCDD's Test A finding (last_token cosine 0.754 across checkpoints on bit-identical inputs) reflects the body's parameter accumulation across training. v1's readout change makes the K-th outputs robust to this drift (because per-K parameters compensate), but the body itself still drifts uniformly. v1 succeeds if the per-K parameter pathway can compensate for body drift; v1 fails if it cannot.

2. **The W=16 sliding window pattern is preserved.** Each frame is still re-processed in up to 16 different contextual positions across the trajectory. The architectural cost of this is inherited from v0 unchanged.

3. **Additive positional embeddings are preserved.** Position information is still encoded as additive metadata on input embeddings rather than through architectural dynamics. v1 does not test whether this is the right way to encode sequential order.

Items 2 and 3 are tied to the body-family question and deferred to v2.

---

## 5. Means-doing-work discipline

### 5.1 The discipline statement

Every aggregation operation in v1 — gates, arm-comparison metrics, verdict thresholds, summary statistics — has one of three justifications:

1. **The aggregation is the right summary at that point.** Justified by the operational question the aggregation answers. Example: total loss across a training epoch is an aggregation across (item, ordinal, K-step), but its operational purpose is "is the training making progress overall," which is the right operational question for that aggregation.

2. **The aggregation is paired with a per-(item, ordinal) disaggregated form.** Justified by the disaggregation discipline preserving locality information. Example: a gate that fires on aggregate mean drift across all (item, ordinal) pairs *and* surfaces the per-(item, ordinal) drift values for inspection, so structured outliers don't get hidden.

3. **The aggregation is explicitly contraindicated and replaced with disaggregated form.** Justified by v0's lesson that aggregate-only gates hid the localised Sofa-ord-1 signal. Default for any operation where structured outliers could be load-bearing for v1's verdict.

The default disposition is option 3 for any operation where v1's verdict could plausibly be sensitive to structured outliers (which is most of them).

### 5.2 Specific applications

**Arm-comparison metrics.** Each arm's behaviour is summarised at per-(item, ordinal) granularity for comparison across arms. Arm-aggregate metrics (e.g., "mean variance differentiation across all (item, ordinal) for Arm A") are paired with per-(item, ordinal) disaggregated forms. The arm-comparison verdict structure (V1A through V1D) is evaluated at the disaggregated level, not the aggregate level.

**Gates and thresholds.** No SCAFFOLDING absolute-magnitude thresholds without empirical calibration (v0's §15 lesson, promoted to default discipline). All gates evaluated against measured controls or empirical distributions. All gate decisions surface per-(item, ordinal) values alongside aggregate values for human review.

**2σ-band discriminators.** v0's BCDD diagnostic used 2σ-band-around-the-mean discriminators (Test B's uniformity check). These are *across-ordinal* distribution properties and hide structured outliers by construction (an outlier within the band is invisible; an outlier outside the band signals non-uniformity but doesn't characterise *how* the distribution departs from uniformity). v1's verdict-level gates do not use 2σ-band-only discriminators — they pair the band check with per-(item, ordinal) inspection, and the verdict assignment looks at both.

**Verdict assignment.** The V1A/V1B/V1C/V1D categorisation is determined at per-(item, ordinal) granularity, not aggregate. V1A requires per-(item, ordinal) differentiation; V1C requires per-(item, ordinal) coupling. D-class is the category for verdicts where per-(item, ordinal) heterogeneity is itself the result, and D-class assignment surfaces the heterogeneity pattern (which items, which ordinals) as part of the verdict.

### 5.3 The Sofa-ord-1 inheritance

v0 produced one localised signal that aggregate gates would have hidden: Sofa-ord-1 variance widened (+0.056) while every other (item, ordinal) pair narrowed. Aggregate gates classified this as "controls drifted, ratio failed"; disaggregation revealed the localised widening as a candidate architectural signal (the localised-widening question, v0 closing §6).

v1's verdict framework treats this kind of localised signal as a first-class result. If v1 produces a verdict where most (item, ordinal) pairs show coupling but a few show clean differentiation, that pattern is itself the result — not a noisy V1C verdict to be cleaned up by averaging. The per-(item, ordinal) granularity discipline ensures the pattern is visible; the verdict-assignment discipline ensures the pattern is recorded.

### 5.4 Disaggregation cost

Per-(item, ordinal) evaluation produces more numbers than aggregate evaluation, with corresponding cost in review burden and result-document length. The cost is accepted as the price of v1's evaluation integrity. v0's closing document established that disaggregation discipline turned an ambiguous aggregate verdict into a precise verdict; v1 commits to the same discipline by default.

---

## 6. Substrate verification and deferrals

### 6.1 §5 substrate verification protocol

v0's §5 substrate verification protocol carries forward to v1 unchanged. The protocol tests:

- **§5.1 Cross-instance stability.** Mean cosine of encoder embeddings across instances of the same item at the same viewing position across loops. Threshold > 0.75 (recalibratable per §5 discipline).
- **§5.2 Cross-element distinguishability.** Mean cosine of encoder embeddings across different items. Threshold < 0.60 (recalibratable).
- **§5.3 Combined gap.** §5.1 − §5.2 ≥ 0.15.
- **§5.4 Substrate verification on perturbed items.** Verify that perturbation produces the predicted magnitude of cross-stage cosine drop (0.05–0.1 per §1.2 commitment 3). This is a new check for v1; v0's §5.4 was magnitude-agnostic.
- **§5.7 Floor-y derivation.** Modal-y across `GetReachablePositions` results. Inherited from v0.
- **§5.8 Cross-scope perturbation locality.** v0's check on substrate vs API locality differences. Inherited from v0 with extension: v1's stronger perturbation mechanism may produce different locality patterns; §5.8 needs re-running on v1's perturbation regime.

DINOv2 has passed §5.1–§5.3 against v0's substrate. v1 inherits this verification. The §5.4 magnitude verification and §5.8 locality re-check are new v1 substrate verification work.

### 6.2 The eight v0 substrate findings as default checks

v0's substrate findings (closing-doc §4) carry forward as default substrate-verification checks for v1:

1. Python 3.12.3 environment.
2. Embeddings.npy full-population check (all rows non-zero).
3. No 30-frame static dwell (continuous-motion substrate).
4. Substrate-as-feature vs substrate-as-bug interpretive discipline.
5. Camera elevation check (`forceAction=True` failure mode).
6. Floor-y derivation via modal-y.
7. View-through check at viewing poses (orientation adjustment if needed).
8. Cross-room visual leakage check on perturbation mechanism.

v1 substrate verification produces a checklist that includes all eight checks plus the v1-specific perturbation magnitude verification.

### 6.3 Deferrals to v2

Architectural deferrals (§2.3) summarised here for completeness:

1. **Body family.** v2 primary architectural question.
2. **Temporal-context mechanism.** Tied to body family.
3. **Positional information mechanism.** Tied to body family.
4. **Bounded memory mechanism.** Tied to body family.

Non-architectural deferrals (research-direction questions surfaced by v0/v1 that v2 will engage):

5. **Encoder substitution.** DINOv2-frozen inherited from v0. SIGReg-from-scratch or other encoder substrates remain candidate v2/v3 questions.
6. **Multi-modal extensions.** v0 spec §6 deferral list; carries forward.
7. **Action / reward integration.** v0 spec §6 deferral list; carries forward.
8. **Episodic-to-semantic consolidation.** v0 spec §6 deferral list; carries forward.

### 6.4 The synthetic-stream companion question

The synthetic-stream alternative (v0 closing §8.4) was considered as a candidate companion arm for v1. Under the M3 verdict from BCDD, synthetic substrate has appeal as a cleanly-controlled test of whether the new readout produces per-position outputs.

**v1 defers synthetic-stream to v2.** Two reasons. First, synthetic-stream introduces a substrate that needs its own verification — a v1 synthetic arm would require §5-equivalent verification on the synthetic substrate before the architectural claim could be tested on it, which adds substrate verification work without addressing v1's primary verdict. Second, synthetic-stream answers a different question than v1's primary verdict: v1 tests whether the transformer-family architecture works on the real (AI2-THOR-rendered) substrate under proper configuration; synthetic-stream tests whether the architecture works on a controlled substrate. These are complementary but distinct questions.

v2 design intake already lists synthetic-stream as a candidate (`WEFT_INNER_PAM_v2_DESIGN_INTAKE.md` §3 / §8 checklist). The deferral is clean.

---

*End of v1 spec §§1–6 (architecture-and-claims layer). Companion to `WEFT_INNER_PAM_v0_CLOSING.md` and `WEFT_INNER_PAM_v2_DESIGN_INTAKE.md`. §§7–11 (implementation specification) follow as a CC-handoff document.*
