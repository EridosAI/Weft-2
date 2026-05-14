# Weft Inner PAM — v0 Architecture Spec (Revised)

**Status:** Third draft, post-Grok-review-2. Grok's remaining four pressure points (confidence aggregation, recalibration traceability, divergence-point monitoring framing, output projection reshape) folded in. Supersedes prior drafts. Supersedes `PAM_Tiered_v0_Spec.md` and obsoletes `pam_tier_a_grok_instructions.md` in their current forms. Both prior project specs stay as historical record; this document does not edit them.

**Purpose:** Capture the architectural claims of Inner PAM, separate architectural commitments from implementation choices, and specify the v0 implementation concretely enough that it can be built and falsified. This document has two layers: architecture and claims (§§1–6), and implementation specification for v0 (§§7–11). Section 12 records design choices considered and set aside, with rationale.

**Scope:** Inner PAM specifically. The wider Weft system (multiple cortices, developmental scale shifting, multi-modal integration, sleep-phase consolidation, bi-hemispheric retrieval, decision-making mechanisms beyond outer-JEPA mediation, action and reward) is acknowledged but not specified here.

---

## 1. What Inner PAM is

Inner PAM is a learned predictor over an embedding stream produced by an outer encoder. Its job is to recognise *trajectory shapes* — recurring patterns of embedding evolution across time — in the agent's continuous experience, and to surface the continuation of a recognised shape when given a partial cue.

The unit of learning is the *shape*, not the frame. A shape is a multi-frame pattern of embedding evolution that recurs across the agent's experience. "Make tea" is a shape. "Walk down the hallway" is a shape. Multiple instances of a shape are similar in the embedded space — they form a denser region with a central area where instances overlap most strongly and looser boundaries where they diverge. Shapes are not pre-defined or labelled; they emerge as a consequence of repetition in continuous experience producing density structure in the embedding space.

Inner PAM is *always on*. Every tick of the agent's experience, Inner PAM is querying the embedding bank for shape recall and surfacing trajectory continuations. It is not gated by a separate trigger mechanism. The agent's continuous perceptual stream is itself the trigger.

Inner PAM's output is *modality-agnostic*. It produces a trajectory shape signal in the shared embedding space. Downstream systems (motor cortex, visual cortex, auditory cortex, in their eventual full forms) co-develop with Inner PAM and learn to interpret this signal in their own modality-specific terms. Inner PAM does not produce motor commands or sensory predictions; it produces shape patterns that other systems read.

The collection trajectory that produces this stream is **continuous motion throughout**. The agent moves through each loop as one connected trajectory with no held-pose segments — items in the environment enter view, fill view, and slide out of view as the agent approaches, passes through, and departs. The agent never holds a fixed pose. This is a load-bearing commitment of the architecture: a path-prediction system requires path-shaped targets, and any zero-velocity segment turns the K-step prediction problem into an identity-prediction problem repeated K times. (The session-3 reviewer surfaced an inherited 30-frame static-dwell pattern from a prior Stage-0b experiment that survived multiple draft reviews; corrected in session 4. The substrate-vs-architecture mismatch is recorded in `HANDOFF.md`.)

---

## 2. Core architectural commitments

These are the load-bearing claims. Implementation choices in later sections must be consistent with these; alternatives that would violate them are out of scope for v0.

**2.1 Shape-as-unit.** The predictor learns trajectory shapes as the primary unit of representation. Frames are inputs to learning, not the unit being learned. A predictor that successfully learns frame-to-frame transitions but does not represent trajectory shapes as recognisable patterns is not Inner PAM, regardless of its retrieval performance.

**2.2 Repetition is the learning signal.** Shape formation requires repeated exposure to trajectories with the same underlying pattern but variation in surface details. A trajectory experienced once is an episode; a trajectory pattern experienced many times is a shape. The architecture must support the developmental gradient from episodic to shape-level representation as repetition accumulates.

**2.3 Continuous time, single-pass.** Inner PAM trains on the agent's experience as it unfolds, in temporal order, single-pass. There is no epoch-based training, no frame-pair sampling, no contrastive-batch construction from arbitrary positive/negative selections. The temporal structure of experience itself is the learning signal.

The agent's experience stream consists of genuine motion: no zero-velocity segments, no held-pose dwell, no frames that are bit-identical to their immediate predecessors at the rendering layer. Every consecutive frame pair has a non-trivial pose delta. A 30-frame static dwell, or any segment in which the agent observes from a fixed pose, violates §2.2 (turning the K-step prediction problem into an identity-prediction problem repeated K times) and therefore violates the architecture's distinctive claim — the predictor cannot learn trajectory shapes from non-trajectory targets. This is a load-bearing substrate commitment, not a collection-side parameter choice.

The continuous-time commitment extends across the experimental phase progression: in v0's curriculum (Stage A baseline → Stage B perturbed-LivingRoom → Stage C compounding perturbation), the same predictor and the same memory bank carry forward across phase boundaries without reset. A phase change alters the *input distribution* (perturbation introduced or escalated) but not the *trainer state*. Resetting predictor weights or clearing the bank at a phase boundary would re-introduce epoch-style discontinuities and break the spec's online single-pass property.

**2.4 Always on.** Inner PAM runs every tick. It does not require an external trigger to activate. Its output is consumed (or ignored) by downstream systems based on prediction confidence, error signals, or other modulators that are themselves part of the larger system, not part of Inner PAM's internal logic.

**2.5 Shape regions with internal density structure.** A learned shape is not a single path through embedding space. It is a region with a dense central area (corresponding to the most-repeated traversals) and looser boundaries (admitting variation across instances). The shape representation must preserve this internal structure: queries should be able to surface central-density traversals (default, fluent recall) or outer-region traversals (novelty, creative recombination — addressed in later versions). The v0 predictor architecture must be capable of representing density structure even if v0 only uses central-density retrieval.

**2.6 Modality-agnostic output.** Inner PAM's output is a trajectory shape signal in the shared embedding space. It is not specialised to motor, visual, auditory, or any other modality. Downstream cortices co-develop and specialise their interpretation of the shape signal.

**2.7 Shapes live in weights; instances live in the bank.** The predictor's parameters encode learned shapes. The memory bank holds depth-modulated instances of past experience. These are separate stores with separate decay characteristics. Bank decay (instances fading from retrievability) does not destroy shapes (which persist in the predictor's weights as long as the architecture is live and the shape is reinforced often enough).

**2.8 Shape recall and instance recall are the same function at different depths.** The system does not have separate mechanisms for "remember the shape" and "remember a specific time." Recall is a single graded operation: high-confidence shape recall produces shape continuation without instance involvement; low-confidence shape recall extends to instance retrieval to find anchoring information. Depth-elevated instances (those marked by surprise, prediction error, or other modulators) can surface even when shape confidence is high.

---

## 3. The architecture in motion

A walk through what happens at every tick of the agent's experience.

**3.1 Perception.** The outer encoder produces an embedding for the current frame from raw sensory input. This embedding is added to the bank, with a depth-modulation value derived from current signals (in v0: recency only; later: surprise, prediction error, etc.).

**3.2 Window construction.** A sliding window of the most recent W embeddings is maintained. This window is the input to Inner PAM's forward pass.

**3.3 Shape recall — v0 output specification.** Inner PAM takes the current window and produces a trajectory shape signal: a centreline path of K predicted embeddings and a per-step scalar variance. Concretely, the output tensor is `(K, embedding_dim + 1)` where the first `embedding_dim` values per step are the predicted mean (centreline) and the final value is the predicted log-variance for that step. The variance is interpreted as isotropic (the same in every direction in embedding space). The variance directly provides the confidence measure used by the recall mixing function (§3.4): low variance = high confidence; high variance = low confidence. Per-step covariance and anisotropic variance are deferred to v1, when bi-hemispheric retrieval becomes active.

**3.4 Instance retrieval (conditional).** If shape confidence is below a threshold, or if depth-elevated instances exist whose embeddings are similar to the current state, the bank is queried for instances. Retrieved instances are folded into the surfaced signal. The mixing function — how shape and instance contributions combine — is a v0 implementation choice (§9).

**3.5 Surfacing.** The combined signal (shape continuation, optionally with instances) is made available to whatever downstream system uses it. In v0 with no full cortices, "surfacing" means writing the signal to evaluation logs; in eventual full system, this is consumed by motor cortex, decision-making mechanisms, etc.

**3.6 Learning.** The predictor's weights update based on the loss signal derived from the difference between predicted and actual continuation. The specific loss function is specified in §4.

**3.7 Bank update.** Older bank entries decay according to the decay schedule (§3.8). Decay is independent of whether the entries' shapes are still present in the predictor's weights — bank entries can decay while shapes persist.

**3.8 Decay.** Two kinds of decay operate in the architecture:

- *Bank decay.* Depth-modulated instances in the bank decay over time. v0 implementation: hard cap on bank size with recency-based eviction; depth-modulation deferred. Specific schedule is a SCAFFOLDING choice in the experiment instructions document.

- *Shape decay (in predictor weights).* Shapes loosen through *non-use* — gradient pressure from new experiences gradually modifies the predictor's function in regions that are no longer being reinforced. This is an emergent consequence of continued online training, not an explicit decay mechanism. The predictor may use a small explicit weight decay for regularisation (separately from this), but it is intentionally much slower than bank decay so that shapes outlast the bank instances that taught them.

The architectural commitment is the asymmetry: bank decay is faster than shape decay. The specific rates are SCAFFOLDING.

---

## 4. The training objective

### 4.1 Primary v0 arm — Path prediction with Gaussian negative log-likelihood

The predictor is trained to produce the v0 output specified in §3.3: a centreline path of K predicted embeddings plus per-step scalar variance. The training signal is online: at each training step, the model predicts the next K embeddings from the current window, then K steps later the actual observed embeddings are compared to the prediction.

**Loss function.** For a window producing predicted means `μ_1, ..., μ_K` and predicted log-variances `log σ²_1, ..., log σ²_K`, with actual observed embeddings `y_1, ..., y_K`:

```
L = sum over k=1..K of:  
    0.5 * ( ||y_k - μ_k||² / σ²_k  +  d * log σ²_k )
```

where `d` is the embedding dimension (1024 for V-JEPA 2 ViT-L). This is the negative log-likelihood of `y_k` under an isotropic Gaussian with mean `μ_k` and variance `σ²_k`, summed across the K predicted steps, with constant terms dropped.

**Weighting across K steps.** Uniform. Each step contributes equally to the loss. The gradient pressure from getting later predictions wrong is allowed to dominate naturally without an explicit weighting schedule.

**Stop-gradient on targets.** `y_k` for each k must be detached from the encoder's computation graph. The loss gradient flows only through the predictor.

**v0 K value.** K = 16. SCAFFOLDING. Matches the prior window size for symmetry; variable-K and learned-K are deferred.

**Why this formulation:**

- The predicted mean is the shape centreline. Its accuracy reflects whether the predictor has learned the dense-region path through embedding space.
- The predicted variance is the shape thickness. Repetition produces consistent observations across many training steps, driving the variance down for well-grooved regions. Divergence points (where multiple continuations are plausible) drive the variance up because the actual observations vary across instances.
- The loss naturally rewards both correct centreline prediction and well-calibrated uncertainty. A model that predicts low variance and is wrong is heavily penalised; a model that predicts high variance everywhere pays a `log σ²` penalty for over-cautious predictions.
- No explicit shape-segmentation, no contrastive negative-pair construction, no historical co-occurrence statistics. The shape emerges from density structure that the loss naturally creates through repetition.

### 4.2 Known limitations of the primary arm

These are not reasons to defer the primary arm; they are flagged as potential failure modes to monitor.

**Fixed K horizon.** Real shapes have variable length. Some recurring patterns in the agent's experience may be shorter than 16 frames; some may be longer. Fixed K means the predictor always predicts the next 16 frames regardless. v0 accepts this limitation; variable-horizon prediction is deferred.

**Gaussian assumption.** Embeddings produced by the encoder are not guaranteed to be well-modelled by independent per-step isotropic Gaussians. At true divergence points, the distribution is multi-modal (e.g., bedroom OR sink after tea), not Gaussian. The learned variance will likely be mis-calibrated at such points. v0 will explicitly monitor calibration at known divergence points; persistent mis-calibration there is an expected signature of the isotropic Gaussian assumption and will be reported as such rather than treated as model failure. Richer distributional models (mixtures, normalising flows) are v1 candidates if mis-calibration is severe enough to obscure the architectural claim.

**Independence assumption across K steps.** The loss treats each predicted step independently. Real future observations are correlated. v0 accepts this for simplicity; correlated predictions are a v1 candidate.

### 4.3 Single primary arm, not parallel arms

Prior versions of this document carried Options A (contrastive over windows) and C (historical co-occurrence distribution) as parallel arms alongside the path-prediction approach. They have been set aside for v0 with rationale recorded in §12. They remain v1 candidates *if and only if* the primary arm fails in ways that point specifically to missing contrastive or historical-co-occurrence signal.

---

## 5. Encoder substrate verification protocol

The architecture's claims rest on the encoder producing embeddings with certain properties. Before any v0 training run, the existing memory bank from prior experiments is analysed against these properties. Failure means the architecture cannot test its claims on this encoder; encoder substitution or fine-tuning becomes a v0 design decision.

**5.1 Cross-instance stability check.** For recurring elements (e.g., the same furniture viewing position visited across many loops in the prior experiments), measure the mean cosine similarity of embeddings *across instances*. The architecture requires that recurring elements produce stable representations.

- *Pass criterion:* mean cosine > 0.75 across at least 50 instance pairs.
- *Starting target — may need calibration.* If V-JEPA 2 mean-pool produces a different baseline range, the threshold gets recalibrated based on the empirical distribution. Recalibration is allowed; widening the criterion until everything passes is not.

**5.2 Cross-element distinguishability check.** For pairs of *different* recurring elements (e.g., sofa-frames vs. fridge-frames in the prior furniture experiment), measure the mean cosine similarity. The architecture requires that the encoder distinguishes the relevant unit.

- *Pass criterion:* mean cosine < 0.60 across at least 50 cross-element pairs.
- *Starting target — may need calibration.* Same recalibration discipline applies.

**5.3 Combined criterion.** The within-instance similarity must exceed the cross-element similarity by at least 0.15 (so the architecture has signal-to-noise to learn from). If this gap is below 0.15, the encoder is producing representations dominated by something other than the relevant unit (likely scene context). v0 cannot proceed on this encoder without substrate work.

**5.4 What the protocol does not check.** Tempo invariance. Multi-modal extension. Sufficient granularity to support eventually-bi-hemispheric retrieval. These are deferred.

**5.5 What happens if the protocol fails.** If the existing encoder fails the protocol, v0 stops and the failure mode is reported in detail. Options at that point: (a) try a different frozen encoder (DINOv2, SigLIP, etc.); (b) introduce a SIGReg-trained or otherwise fine-tuned encoder (which moves the project into Tier B territory and is a deliberate decision); (c) reframe what counts as the recurring unit. The decision is made with human review, not autonomously.

Any threshold recalibration (per §5.1, §5.2) must be reported in the experiment log with explicit justification — for example, "threshold adjusted from 0.75 to 0.68 after inspecting the empirical distribution of within-instance cosines, which clustered between 0.65 and 0.78; gap criterion held at 0.15." Recalibration without recorded justification is a process violation per §11.

**5.6 Per-item stability under out-of-scope perturbation.** When the experimental phase structure introduces a scoped perturbation (e.g., Phase 2's `RandomizeMaterials(inRoomTypes=["LivingRoom"])`), the substrate is also verified for *per-item embedding stability under that perturbation* — items the perturbation is scoped *away from* must not show meaningful embedding shift under it, or the locality claim the curriculum depends on is broken at the encoder level. The verification is procedurally equivalent to §5.1's cross-instance stability check, but the variation source is the scoped perturbation rather than natural agent-pose variation. The pass criterion is per-item: each control item's mean embedding cosine across N independent draws of the perturbation must exceed a threshold (in practice, > 0.98 for DINOv2 CLS cosines on a non-perturbed item under a small-scope material swap; recalibratable from the empirical distribution per the discipline above). Failure indicates either the perturbation is bleeding cross-scope (e.g., a viewing pose that catches out-of-scope perturbed content in its FOV) or the encoder is more globally sensitive than the locality claim assumes. In the seed-7 ProcTHOR house, the DiningTable viewing pose was found to capture the LivingRoom doorway in its FOV, producing DINOv2 movement of ~0.045 cosine units under LivingRoom retexturing (vs ~0.01 for Bed and Television). The substrate-specific viewing poses are SCAFFOLDING; the DiningTable pose was adjusted 2026-05-14 to remove the doorway view. Concrete values live in instructions §1.3 and `data/route_phase2.json`.

---

## 6. What Inner PAM is not, and what is deferred

Statements that bound the scope of v0 and prevent drift.

**6.1 Inner PAM is not a memory bank.** The bank is a separate component. Inner PAM queries it. The shape representations live in Inner PAM's weights; the bank holds depth-modulated instances.

**6.2 Inner PAM is not a decision-maker.** It surfaces trajectory shape signals; it does not decide what to do with them. The mediation between Inner PAM's output and outer JEPA's prediction (or eventual decision-making mechanisms) is deferred.

**6.3 Inner PAM is not multi-scale in v0.** Architectural commitment (§§2.1–2.8) is consistent with eventual multi-scale operation, but v0 fixes a single temporal scale (window size W, prediction horizon K). Developmental scale shifting is deferred.

**6.4 Inner PAM does not include depth-modulation logic in v0.** The architectural commitment is that depth modulation exists. v0 implementation defaults to recency-only or uniform depth. Surprise, prediction-error, emotional-weight modulation come later.

**6.5 Inner PAM does not include sleep-phase consolidation in v0.** Bank entries that decay below retrievability are simply discarded in v0. Consolidation, dreaming, and offline replay are deferred.

**6.6 Inner PAM does not include bi-hemispheric retrieval in v0.** The architectural commitment is that shape representation must preserve enough internal structure to *eventually* support central-vs-outer retrieval. The v0 output (centreline + per-step scalar variance) supports this in principle (the variance defines the cloud width); v0 retrieval defaults to central-density-only.

**6.7 Inner PAM does not include the outer-JEPA-mediation mechanism in v0.** This is explicitly deferred and named as load-bearing for the eventual system. v0 evaluation can use simple criteria (was the shape recall correct? did it match the actual trajectory?) without requiring a mediation mechanism between Inner PAM and outer JEPA.

**6.8 Inner PAM does not include action or reward in v0.** The agent is a passive observer. The architectural commitment of Weft is that associative memory from continuous observation is the foundational mechanism. Active agency and reward-driven learning are out of scope.

**6.9 v0 uses consistent-tempo trajectories only.** Trajectories in the v0 experience stream maintain roughly consistent timing across instances. Variable tempo (the same shape unfolding over different numbers of frames in different instances) is a known architectural challenge that is deferred to v1. The waypoint / sub-trajectory nesting picture sketched during architecture development is also v1+.

---

# Implementation Specification (v0)

The following sections specify v0 implementation choices concretely. Choices labelled SCAFFOLDING are tractable defaults that may be revised; choices that follow directly from §§1–6 are not flagged separately.

---

## 7. v0 component inventory

**7.1 Encoder.** A frozen pre-trained visual encoder, single-modality, producing per-frame embeddings. SCAFFOLDING — specific choice deferred to v0 experiment design pending the encoder substrate verification protocol (§5). Frozen V-JEPA 2 ViT-L is the existing default but its suitability is explicitly tested by §5 before any v0 training run.

**7.2 Memory bank.** Append-only embedding store with depth-modulation values. Backed by a nearest-neighbour index (FAISS `IndexFlatIP` or similar). Exposes: append, retrieve-by-similarity, retrieve-by-depth, decay. The depth-modulation field is ARCHITECTURE; the index implementation and bank size are SCAFFOLDING.

**7.3 Inner PAM predictor.** Neural network that takes a window of W embeddings and produces the v0 output specified in §3.3 (centreline + per-step variance over K=16 steps). Recommended starting architecture: 4-layer transformer encoder, 8 heads, hidden dim 512, MLP dim 2048, GELU, pre-LayerNorm. Output projection produces `K * (d + 1)` values, then reshaped to `(K, d + 1)` before returning the centreline and log-variance tensors. SCAFFOLDING — depth/width are tunable; the requirement is that the architecture be capable of representing density structure, which the variance head provides.

**7.4 Trainer.** Online, single-pass training loop. Consumes the embedding stream, maintains the window of W recent embeddings, runs Inner PAM forward on each step, computes the loss specified in §4.1 using the actual K-step future as it becomes available, updates predictor weights. ARCHITECTURE for the online single-pass property and the loss formulation; SCAFFOLDING for specific optimizer choice and learning rate (recommended starting points: AdamW, lr=3e-4, weight_decay=0.01 for predictor regularisation only — not a shape-decay mechanism).

**7.5 Recall mixing function.** Combines shape signal and instance-retrieval signal into the surfaced output. v0 implementation: confidence-thresholded — when the confidence score is above threshold (high confidence), surface shape only; when below threshold (low confidence), additionally surface top-K bank instances similar to the current window. The confidence score is the mean of predicted variance over the first M steps of the predicted path. v0 default M = 3 (a single step gives location but no direction; aggregating across the first few predicted steps disambiguates which trajectory the current state belongs to). M is SCAFFOLDING; adaptive M is a v1 candidate. The confidence threshold τ and the instance-mixing weights are also SCAFFOLDING; ARCHITECTURE for the principle of confidence-graded mixing.

**7.6 Evaluation harness.** Constructs probes from the experience stream, runs Inner PAM on each probe, records surfaced output, compares to ground-truth continuation. ARCHITECTURE for the evaluation principle (faithful recall of repeated shapes from partial cues); specific probe construction is detailed in §10.

---

## 8. v0 environment requirements

The environment is not specified architecturally. The architecture's claims do not depend on a specific environment. v0 experiments should choose an environment that:

- Produces continuous experience streams (not pre-segmented episodes).
- Contains genuinely recurring trajectory shapes — patterns that repeat with variation across many traversals.
- Allows controlled construction of the recurrence (so we can verify shapes exist in the data before training on it).
- Maintains roughly consistent tempo across instances of the same shape (per §6.9).
- Supports the encoder substrate verification protocol (§5).

The existing AI2-THOR + densified-Teleport infrastructure satisfies these requirements but is not the only option. Environment choice is a v0 design decision in the experiment instructions document.

---

## 9. v0 implementation choices remaining open

The primary v0 arm specification in §4.1 settles most of what was previously open. The following choices remain and are decided in the experiment instructions document, not here:

**9.1 Predictor depth/width specifics.** Starting recommendation in §7.3; tunable.

**9.2 Optimizer specifics.** Starting recommendations in §7.4; tunable.

**9.3 Bank size and eviction policy.** SCAFFOLDING. Hard cap with recency-based eviction is the v0 default; specific cap depends on expected experience stream length.

**9.4 Confidence threshold τ for the mixing function.** SCAFFOLDING. Initial value calibrated against the variance distribution observed in early training.

**9.5 Window size W and prediction horizon K.** v0 defaults: W = 16, K = 16. Both SCAFFOLDING.

---

## 10. v0 evaluation principle and required analyses

Inner PAM v0 is evaluated on whether it learns trajectory shapes from repeated exposure and surfaces correct continuations from partial cues. The minimum tractable evaluation is:

- Construct an experience stream containing repeated trajectories with controlled variation.
- Train Inner PAM on the stream in continuous time, single-pass, with the loss specified in §4.1.
- Test by presenting partial trajectory cues and measuring whether the surfaced continuation matches the actual trajectory (across instances of the shape).

**10.1 Required controls.**

- *Pure cosine baseline.* Probe's own embedding, no learned predictor. Should fail when shapes require non-similar associations. Required.
- *Shuffle control.* Predictor trained on temporally-shuffled version of the same stream. Should fail because the temporal structure required for shape learning is destroyed. Required.

**10.2 Required disaggregations.**

- *Per-shape breakdown.* Performance reported per recurring shape, not just aggregated. Aggregate metrics hide failure modes; the prior work was bitten by this repeatedly.
- *Per-position-within-shape breakdown.* Performance at the start vs. middle vs. end of a shape may differ significantly.
- *Repetition-stratified analysis.* Performance on shapes that appeared 1-5x vs. 20-50x vs. 100+x within the same training run. This directly tests §2.2 (repetition is the learning signal) — if the model does not show meaningfully better recall on more-repeated shapes, the central architectural claim is unsupported.
- *Bank-vs-weights recall fraction.* For each successful recall, what fraction of the surfaced signal comes from predictor-weight-mediated shape recall vs. bank-instance retrieval? Measured at multiple training checkpoints. This directly tests §2.7 and §2.8 — the architecture predicts that early in training, recall depends heavily on bank instances; as repetition accumulates, recall increasingly comes from predictor weights even as bank instances decay. If this transition does not occur, the episodic-to-semantic story is unsupported.

**10.3 Verdict structure.** Each evaluation produces a named verdict in plain language: shape-learning supported, shape-learning falsified, mixed/ambiguous. Honest summaries, no both-sides write-ups. The discipline established in prior experiments carries forward.

---

## 11. Operational discipline

The operational discipline established in `CODING_STANDARDS.md` and `research_operations_v1.md` carries forward unchanged. The architectural rethink does not relax any operational principle. In particular:

- Spec-driven, single-variable changes, simple baselines, controls designed up front.
- Adversarial review by primary and secondary reviewers before implementation.
- Scaffolding labelled and inventoried.
- Numbers trace to files. No remembered numbers.
- Phase gates and HANDOFF.md updates.
- Push hold remains in effect.

The new architecture introduces specific failure modes that should be added to operational vigilance:

- **Drift from architectural commitment to familiar shape.** The Tier A specs under-articulated the architecture and the implementation drifted into next-frame prediction. v0 implementation must be reviewed specifically for whether each component aligns with §2's commitments, not whether it matches a familiar pattern.
- **Encoder-substrate failures masquerading as architecture failures.** §5 establishes what the encoder must support. Failures attributable to encoder limitations are encoder failures, not Inner PAM failures.
- **Aggregate metrics hiding shape-vs-frame distinctions.** Inner PAM is evaluated on shape-level recall. Shape-level metrics must be reported before any aggregate metric. Any aggregate number that looks good while per-shape recall is broken must be flagged as invalid.

---

## 12. Options considered and set aside

These design choices were considered during architecture development and explicitly set aside for v0, with rationale recorded here so future reviewers and revisions can see the reasoning rather than having to re-derive it.

**12.1 Option A — Contrastive learning over trajectory windows.**

*The idea:* Train the predictor such that windows occupying the same role in a recurring shape produce similar internal representations, with negative examples being windows from different shapes or different roles. Continuation prediction is a separate readout.

*Why set aside:* The formulation pre-supposes the things the architecture is supposed to learn — what counts as the same shape across instances, what "same relative position" means when shapes have variable tempo, where shapes start and stop. Asking the model to learn from those assumptions imports the segmentation problem we want the architecture to dissolve. More fundamentally: the shape vs background distinction emerges naturally from density structure produced by repetition under the path-prediction loss — adding contrastive machinery is bolting on a mechanism for what the data structure already provides.

*v1 candidate when:* the primary arm fails specifically at shape recognition (the model produces accurate predictions but cannot discriminate which shape a partial cue belongs to).

**12.2 Option C — Predict the historical co-occurrence distribution.**

*The idea:* Given the current window, predict the distribution over states that have temporally co-occurred with similar windows across the agent's history.

*Why set aside:* Once the encoder is doing similarity, this collapses into the primary arm. The encoder represents similar perceptual states with similar embeddings; the predictor trained on path-prediction is naturally learning what tends to follow similar states. The two framings are descriptions of the same mechanism at different levels. Implementing C as a separate objective would require maintaining historical co-occurrence statistics or a dedicated contrastive sampler, neither of which adds beyond what density structure under the primary loss provides.

*v1 candidate when:* the primary arm fails specifically at retrieving cross-segment associations that share little visual similarity (the architecture's distinctive claim) — at which point an explicit historical-co-occurrence training signal becomes worth the implementation cost.

**12.3 B-2 (fixed-length trajectory code with decoder).**

*The idea:* The predictor outputs a fixed-length code (e.g., 256-dim) representing the entire predicted trajectory; a separate decoder reconstructs the K-step path from this code.

*Why set aside:* The formulation assumes trajectories have well-defined edges. In continuous experience, trajectories don't have edges — they have divergence points where the future spreads across multiple continuations. A fixed code forces the model to commit to one continuation when the actual data structure is multi-modal. The Gaussian NLL formulation handles divergence points naturally through learned per-step variance.

*v1 candidate when:* shape recognition (rather than shape continuation) becomes the primary task and a clean shape identifier is needed.

**12.4 B-3 (DTW-style alignment loss).**

*The idea:* Predicted and actual paths compared via dynamic time warping, allowing temporal misalignment.

*Why set aside:* Variable tempo is a v1 problem (per §6.9). DTW is the right tool for it when it becomes load-bearing, but adds complexity v0 doesn't need with consistent-tempo trajectories.

*v1 candidate when:* variable-tempo trajectories enter the training stream and the primary arm shows tempo-sensitivity failures.

**12.5 Later-step weighting in the primary loss.**

*The idea:* Weight later predictions more heavily in the loss to force the model to learn shape-level structure.

*Why set aside:* Imports a hyperparameter without strong justification. Uniform weighting is simpler and lets gradient pressure from getting later steps wrong dominate naturally. If the model collapses to local prediction with uniform weighting, weighted variants become a candidate intervention.

---

*End of v0 spec. Ready for adversarial review on the revised version.*
