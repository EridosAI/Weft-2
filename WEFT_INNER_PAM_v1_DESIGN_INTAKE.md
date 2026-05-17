# Weft Inner PAM v1 — Design Intake Brief

**Purpose.** Strategic state of play before v1 design begins. Captures forward-looking context the v0-closing document deliberately doesn't have: my current leanings on the four candidate directions, the constraints v1 inherits, the open decisions a v1 design chat needs to work through with me, and what "v1 success" should look like.

**Status.** Pre-spec. v0 closed at commit fef8a52; this brief is the strategic-handoff document to the v1 design chat. Companion to (not replacement for) `WEFT_INNER_PAM_v0_CLOSING.md`, which is the canonical retrospective input.

**Rule for using this document.** This brief carries my current leanings. They are not commitments. A v1 design chat that adversarially pressure-tests these leanings — including telling me where my hunches don't match the evidence in the closing doc — is doing the job correctly. Where my leaning is wrong, I want to find out before the spec drafts, not after CC has burned compute on the wrong architecture.

---

## 1. State of play

v0 closed cleanly with verdict **V2 — Shape-learning falsified, with coupling-mechanism caveat**. The path-prediction mechanism fit and learned; the variance head learned; but per-K-step isotropic scalar variance gradients propagate uniformly across positions regardless of which inputs produced the surprise, so the spec §2.2 per-item-variance-response prediction does not survive on this substrate with this loss formulation.

The verdict identified a specific architectural lever (variance-head gradient topology) rather than just falsifying §2.2 in the abstract. That is a more useful negative result than a clean failure would have been — v1 knows what to change, not just that change is needed.

v0 also produced institutional memory v1 inherits as default discipline:
- Eight substrate findings (one of them — finding 4 — the substrate-as-feature-vs-bug interpretive distinction).
- Five SCAFFOLDING-threshold lessons plus the sixth methodological corollary (mid-experiment disambiguation thresholds also need §15 discipline).
- The disaggregation discipline (per-(item, ordinal), per-loop, per-K-step) that turned an ambiguous aggregate result into a precise verdict.
- The disambiguating-diagnostic discipline (cheapest test that can close the path) that resolved the reading-(i)-vs-(ii) ambiguity in one minute of compute on existing data.

The closing doc §7 surfaces three plausible underlying causes of the coupling result (per-K-step scalar isotropic variance, DINOv2-frozen substrate, perturbation magnitude too weak), explicitly non-mutually-exclusive. §8 surfaces four v1 candidate directions framed as questions. Both sections are load-bearing input to v1 design.

Push hold remains in effect. v0 artifacts stay local until v1 scope is settled and an external-facing decision is made about which v0 outputs belong outside the local repos.

---

## 2. What "v1 success" should look like

This is a load-bearing strategic question the v1 design chat needs to engage with me on early. The framing matters because it shapes spec discipline level and pivot allowance.

**My current frame: test-discipline with pivot allowance.** v1 is structured to produce a verdict where one is supportable — same disaggregation rigour, same gate discipline, same single-variable attribution where the experimental structure supports it. But v1 is *not* held to rigid sequencing if mid-experiment evidence reveals the experiment is testing the wrong thing. Pivots are first-class events, not failures of discipline. We are designing something new; the architectural concept is still being discovered.

What this means concretely:

- **Single-variable attribution is preserved through planned ablations**, not through forced single-direction sequencing. A compound intervention (e.g., variance-head fix *plus* stronger perturbation) is acceptable provided the design includes an ablation arm that isolates each component's contribution. The §7.4 warning against bundling interventions applies to bundling *without an attribution plan*, not bundling per se.
- **Disaggregation discipline carries forward as default**. Per-(item, ordinal) granularity from design time, not aggregate-then-disaggregate. This is non-negotiable per closing doc §6.
- **Pivots are documented, not suppressed**. If v1 begins testing variance-head topology and mid-experiment evidence reveals the bottleneck is somewhere else, the right response is to pivot the experiment and document the pivot — not push through and force a verdict on the original framing. The discipline is operational and epistemic, not bureaucratic.
- **A verdict is produced if the experiment supports one**. If the architectural intervention works clearly or fails clearly, v1 records the verdict (V1, V2, V3 in v0's nomenclature, or a new framing if the evidence demands it). If the experiment pivots before producing a clean verdict, v1 records what was learned at the pivot and scopes v2 from there.

This frame is sharper than pure "v1-as-test" (which over-emphasises rigidity) and sharper than "v1-as-exploration" (which under-emphasises attribution discipline). It is what v0 actually was, retrospectively: nine STOPs, multiple substrate findings, no rigid commitment to the original v0 design beyond what the evidence supported. v1 inherits the same posture explicitly.

---

## 3. The four candidate directions, as I hold them now

The closing doc §8 presents the four directions as equally-weighted questions. I do not hold them as equally weighted. My current leanings, with reasoning, concerns, and what would shift my view on each:

### 3.1 Direction 8.1 — SIGReg-from-scratch encoder

**My current leaning: deferred as a full direction; pre-spec diagnostic worth running.**

Reasoning. The closing doc §7.2 says a DINOv2 substrate that doesn't isolate per-item information could plausibly contribute to the coupling result, but the §7.1 mechanism explanation (per-K-step scalar variance head has no per-position parameters) is a tighter argument that does not depend on the encoder. If the variance-head topology is the dominant cause, fixing the encoder doesn't help. Full SIGReg-from-scratch is operationally expensive (closing doc §8.1 lists pretrain regime, compute envelope, integration approach, substrate-verification protocol updates) — too expensive for v1.

But there is a cheaper version of the encoder question worth running before v1 spec locks. Two pre-spec diagnostics:

- **Lightweight-SIGReg-fine-tune-on-existing-data** (per Grok's review): run a small SIGReg-style fine-tune of an existing backbone on v0's collected stream and compare the per-position variance behaviour against DINOv2-frozen on the same predictor. Hours of compute, not days. If results differ materially, encoder choice is a contributing cause and v1 scope adjusts. If results match, the deferral is confirmed.

- **V-JEPA 2 retry on current substrate** (per SS7.2 below): re-run substrate verification with V-JEPA 2 (ViT-Tiny SIGReg) on the *current* seed-7 continuous-motion stream and see whether it passes §5 protocol acceptably. The prior project's V-JEPA 2 inter-room cosine collapse was on a different substrate; whether it still fails on the corrected current substrate is an open question. If V-JEPA 2 passes, it becomes a much cheaper alternative to from-scratch SIGReg for any v2 encoder work.

What would shift my view (toward elevating 8.1 to a full v1 direction): either diagnostic showing encoder choice materially affects coupling behaviour, or a research-strategic argument that v1 should resolve the encoder question definitively even at high operational cost.

### 3.2 Direction 8.2 — Per-position or anisotropic variance representation

**My current leaning: high interest, lowest operational cost, most architecturally precise response to the coupling finding.**

Reasoning. The closing doc §7.1 mechanism is unusually clean: per-K-step scalar variance head has no per-position parameters, so gradient backprop has no architectural pathway to route variance updates to specific positions, so the coupling result is what the architecture's parameter topology *must* produce. Changing the variance head to have per-position parameters creates the pathway gradient currently lacks. This is the direct architectural response to the diagnosis.

Operational cost is low. Same encoder, same environment, same data pipeline. The change is in the variance head's parameter layout, the loss formulation's variance-evaluation expression, and the M2 variance-calibration metrics. v0 spec §3.3 explicitly identified per-step covariance and anisotropic variance as deferred-to-v1 items; this is the deferred path.

Concern. The §7.1 mechanism is theoretically clean but it's still a *hypothesis* about why coupling happens. See §7.1 below — the v1 design chat's first substantive technical task is establishing the diagnosis from the math and code, not from empirical inference alone.

What would shift my view. (a) The §7.1 math/code walkthrough reveals coupling sources beyond the variance head (e.g., shared transformer-body parameters producing topologically-equivalent uniform updates). In that case, 8.2's intervention is necessary but insufficient, and v1 needs to address the additional sources too. (b) Pre-spec diagnostics (§3.1) reveal encoder substrate is dominant; 8.2 becomes secondary.

### 3.3 Direction 8.3 — Cleaner perturbation mechanism, reframed as v1 substrate baseline

**My current leaning: not a separate architectural direction; v1 substrate baseline that runs regardless of which architectural direction is primary.**

Reasoning. The closing doc §7.3 makes the case that v0's perturbation regime was too weak (~0.01-magnitude cross-stage cosine drops). v0's coupling result holds at v0's perturbation magnitude. Whether it would hold at stronger magnitudes is an open question independent of the architectural direction. A v1 that runs with stronger perturbation has a tighter test of whatever architectural intervention it makes, *even if* perturbation strength isn't itself the architectural direction being tested.

**Concrete substrate mechanisms to consider** (decision deferred to v1 design chat with feasibility-cost estimates):

- **Agent movement variation across loops.** Vary the agent's exact viewpoint at each ordinal between loops while keeping the per-(item, ordinal) trajectory length identical. Same furniture sequence, same ordinal count per item, different exact pose at each ordinal. Materially different from `RandomizeMaterials` — adds genuine within-experiment input variation that DINOv2 is built to be sensitive to (view variation is what DINOv2 is trained on), naturally local (no rendering-engine coupling), and tests the architectural mechanism against the kind of perturbation the system would experience in the real world. Requires care to ensure trajectory lengths stay matched across loops; doable.

- **DINO sensitisation.** DINOv2's outputs may be compressing genuine signal below the predictor's effective resolution. Options: (a) multiplier or contrast amplification in the embedding pipeline, (b) a small projection/sensitisation layer between DINOv2 output and predictor input, (c) lightweight fine-tune of DINOv2's head on the v0 stream. This is substrate-side intervention orthogonal to the architectural direction and could resolve the §7.2 encoder-substrate question more cheaply than full SIGReg work.

- **Per-object material setting via AI2-THOR API.** If supported, this gives clean per-item perturbation locality without the cross-room rendering leakage v0 found. Requires API-feasibility preflight.

- **Asset replacement at fixed coordinates.** v0 Phase 3 preferred mechanism; never reached preflight. Requires preflight on locality and magnitude.

- **Hand-built texture swaps with offline rendering.** Bypasses AI2-THOR API entirely. New substrate-construction pipeline; highest operational cost.

- **Retain `RandomizeMaterials` and accept the magnitude limitation.** Default fallback if other mechanisms don't pan out.

Agent movement variation and DINO sensitisation are my current preferences because they directly target the two limitations v0 identified (perturbation magnitude too weak, encoder substrate possibly compressing signal) without depending on rendering-engine internals. The §5.8 cross-scope locality check becomes a load-bearing pre-experiment gate for whichever mechanism is chosen.

What would shift my view. v1 design chat surfacing an option I haven't considered, or a feasibility assessment showing one of the above is materially cheaper or more controllable than I'm estimating.

### 3.4 Direction 8.4 — Synthetic-stream alternative

**Status.** Not previously discussed in detail; needs concrete explanation before I can hold a leaning.

What synthetic-stream means concretely: bypass AI2-THOR entirely. Construct procedural trajectories in a synthetic embedding space rather than rendered visual frames. For example, define K furniture-item embeddings as fixed points in a low-dimensional manifold (say, 64 or 128 dimensions), define trajectories as paths through that manifold (with the same per-(item, ordinal) ordinal structure as the real substrate), and feed those embeddings directly to Inner PAM as if they had come out of DINOv2.

Perturbations are then controlled by construction:
- Locality is exact (perturbing item A's embedding has zero effect on item B's embedding by definition).
- Magnitude is tunable (perturb embedding A by ε in a chosen direction; ε is a hyperparameter).
- No rendering engine, no cross-room leakage, no encoder-substrate confound.

**What we could learn from a synthetic-stream arm:**

- **Pure architecture test.** If the architectural intervention (per-position variance head) produces per-position variance evolution on synthetic and fails on AI2-THOR, the AI2-THOR substrate is the bottleneck, not the architecture. If the intervention fails on synthetic, the architecture is the bottleneck regardless of substrate. This is unusually clean attribution.

- **Encoder-substrate isolation.** Synthetic streams skip the encoder entirely. If coupling reproduces on synthetic, encoder substrate is not the cause; the issue is downstream. This addresses the §7.2 / §3.1 encoder question cheaply.

- **Perturbation magnitude sweep.** Cheap to run at multiple ε values, which tells us where the architectural mechanism's signal-to-noise threshold actually sits — answering the §7.3 perturbation-too-weak question directly.

**Trade-offs:**

- Loss of ecological validity. Weft's research thesis is associative memory from continuous visual experience; a synthetic-manifold substrate is not continuous visual experience. A v1 verdict from synthetic alone tells us the architecture works in a domain we don't ultimately care about.
- Gain of complete substrate control. The same loss-of-ecological-validity that makes synthetic insufficient as a sole verdict-bearer makes it ideal as a *companion arm* that sharpens whatever the real-substrate arm produces.

**My current leaning, given this fuller picture.** Synthetic-stream as **companion arm** to a real-substrate primary, run in parallel, used to isolate architecture-from-substrate attribution. Operational cost is low (no AI2-THOR, no rendering, runs in hours not days). Verdict-sharpening value is high.

**What I want the v1 design chat to test.** Is there a case for synthetic as *primary* rather than companion — e.g., is the architectural concept far enough from working that a clean test of the architecture in isolation is more valuable than another mixed real-substrate result? If yes, synthetic-as-primary changes the v1 structure substantially. If no, companion arm is the right framing.

### 3.5 Synthesis of my current leanings

In rough order of v1 priority:

1. **8.2 (variance-head architecture)** as primary architectural direction, contingent on §7.1 math/code walkthrough confirming the diagnosis.
2. **8.3 (cleaner perturbation)** as v1 substrate baseline rather than separate direction. Agent-movement variation and DINO sensitisation as preferred mechanisms; decided in v1 design chat with feasibility estimates.
3. **8.4 (synthetic-stream)** as companion arm to sharpen attribution. Default companion, candidate-primary if v1 design chat surfaces a strong argument.
4. **8.1 (SIGReg-from-scratch)** deferred. Two pre-spec diagnostics (lightweight-SIGReg fine-tune, V-JEPA 2 retry) run before v1 spec locks; their results may shift this.

This is my current frame. It is *not* a commitment, and I want the v1 design chat to engage with it adversarially.

---

## 4. Constraints v1 inherits

These constrain the design space whether v1 likes them or not.

### 4.1 Hardware and wall-clock

- **GPU:** RTX 4080 Super, 16 GB VRAM, local. v0 ran tight on this for predictor + frozen DINOv2 + FAISS + activations; v1 needs to fit similar or less.
- **Host:** Intel Core i9-14900K, 64 GB RAM, ~590 GB free disk at v0 close.
- **Wall-clock cadence.** v0 ran approximately 8 hours of compute total across two days, broken into quick runs per phase, nothing over an hour per run. This is *much* less restrictive than a multi-day-session regime. v1 spec discipline does not need to anticipate long sessions or session-protection complications; the actual constraint is fitting individual runs into ~1-hour blocks and managing the cadence across multiple short runs.
- **Single GPU:** no parallelism across runs. Main and shuffle controls run sequentially per v0 §7.5.

### 4.2 Substrate

- **AI2-THOR + ProcTHOR** is the v0 substrate. v0 found cross-room rendering coupling (substrate finding 8) but did not invalidate AI2-THOR as an environment. v1 may retain it, swap it, or run a synthetic companion arm; all three are open.
- **Seed-7 furniture house** is the v0 scene. Five items, two-room layout (Bedroom: Bed, DiningTable, TV; LivingRoom: Dresser, Sofa). v1 may retain the scene or pick a different one; retention has the advantage of preserving substrate verification work (§5.1–5.8 already calibrated against this scene).
- **DiningTable view-through residual** (~0.014 cosine units of cross-room bleed at the h117.57° pose). Lower than original but nonzero; v1 should not assume it's resolved.

### 4.3 Repo and process

- **Working repo:** `/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/`. v0 close left it clean at fef8a52.
- **Reference repo:** `/mnt/c/Users/Jason/Desktop/Eridos/Weft/` (read-only; explorer scripts, frame collection, seed-7 reproducibility).
- **Push hold remains in effect** on both repos through v1 design. No external-facing decisions until v1 scope is settled.
- **Operational discipline:** CODING_STANDARDS, research_operations_v1.md, and the §15-promoted absolute-magnitude-threshold discipline all carry forward as defaults. v1 does not get to relax operational rules.
- **Single CC session per working tree** (CODING_STANDARDS §4.3). v1 inherits this.
- **Adversarial review required** before CC implementation (research_operations §2.2): primary review (this chat or the v1 review chat), secondary review (Grok or another model), resolution pass. Same protocol as v0.

### 4.4 Spec discipline level

v1 is held to the same spec discipline as v0: spec before code (§2.1), single-variable changes via planned ablations (§5.1), controls designed up front (§3.2), simple baselines from day one (§3.3), every fixed parameter labelled ARCHITECTURE or SCAFFOLDING with removal plans (§2.3), pre-experiment gates (§6), numbers trace to files (§4), HANDOFF.md at every session boundary (§8.7).

The test-discipline-with-pivot-allowance frame from §2 does not relax spec discipline; it relaxes the *consequence* of mid-experiment evidence demanding a pivot. Spec discipline still binds the experiment that's been authorised. Pivots are explicit, documented, and trigger spec revision — they are not silent drift.

---

## 5. Open decisions the v1 design chat needs to work through with me

Each of these is a decision the chat shepherds me to, not one the chat makes autonomously.

### 5.1 Compound intervention with planned ablation, or single-variable sequencing

Framed in §2 above. The chat tests whether the planned-ablation structure (variance-head fix *plus* stronger perturbation, with an ablation arm isolating variance-head contribution) preserves attribution adequately, or whether forced single-variable sequencing (variance-head fix alone first, stronger perturbation only if needed) is the safer structure. My current frame is the former; the chat may argue the latter.

### 5.2 Which of the four candidate directions to run, and in what relationship

My §3.5 synthesis is the starting frame. The chat tests it. Specific sub-questions:
- Is 8.2 (variance-head architecture) the right primary direction, or has the chat surfaced an argument for a different primary?
- Is 8.3 (cleaner perturbation) appropriately reframed as v1 substrate baseline rather than separate architectural direction? Which concrete mechanism (agent movement variation, DINO sensitisation, per-object material setting, asset replacement, hand-built textures, or retaining `RandomizeMaterials`)?
- Is 8.4 (synthetic-stream) appropriately framed as a companion arm, or should it be primary?
- Is 8.1 (SIGReg-from-scratch) appropriately deferred to v2, or do the two pre-spec diagnostics (lightweight-SIGReg fine-tune, V-JEPA 2 retry) belong inside v1?

### 5.3 Substrate decision

If 8.3 is reframed as v1 substrate baseline (per §3.3), what's the concrete perturbation mechanism? Decision matrix the chat surfaces:

- API-feasibility preflight on per-object material setting and asset replacement.
- Engineering cost estimate for agent movement variation (trajectory-length matching across varied poses).
- Engineering cost estimate for DINO sensitisation (which approach: multiplier, projection layer, head fine-tune).
- Engineering cost estimate for hand-built texture swaps as fallback.

Decision authorised by me after the chat surfaces the matrix.

### 5.4 Spec scope and length

**Fresh from scratch.** Confirmed. v1 spec mirrors v0's structure (§1–§16) but populated with v1's choices. Keep density comparable to v0, not longer. Delta-document approach explicitly rejected.

### 5.5 Evaluation framework

Per closing doc §6, v1 evaluation needs per-(item, ordinal) granularity from design time to keep the Sofa-ord-1-style localised-signal question decidable. Concrete form open for chat to draft:

- What's the n at which per-(item, ordinal) measurements become meaningful? (v0's §3.6 small-sample threshold is n≥20 in high-variance domains; per-(item, ordinal) needs to clear this.)
- Does v1 evaluation aggregate within-stage *and* report per-(item, ordinal), or only per-(item, ordinal)?
- What's the gate criterion at per-(item, ordinal) granularity? (v0's gates were on aggregates; v1 needs to define what success looks like at the cell level.)

Grok's draft criterion ("≥70% of perturbed-item ordinals show statistically distinguishable variance widening vs controls at loop 100, with no coupling on unperturbed items") is a starting point worth pressure-testing.

### 5.6 Pre-experiment gates

v0 had §8.4 (encoder-level differential) and §8.7a (predictor-level architectural-strength) as the two pre-experiment gate clusters. v1's gates depend on which architectural direction is chosen and what its expected signal looks like. The chat drafts v1 gates; I sign off.

Two specific gates v1 inherits from §3 above:
- §5.8 cross-scope locality check as load-bearing gate for whichever perturbation mechanism is chosen.
- Variance-calibration metrics reworked for the richer variance representation (M2 becomes per-position or per-dimension separation rather than scalar).

### 5.7 Compute budget and phase structure

v0 ran three phases (1, 2, 3) and discarded Phase 1 as substrate-degenerate. v1's phase structure depends on whether it inherits the Stage A → Stage B → Stage C curriculum framing or moves to a different structure. My current frame: v1 likely has just one substantive training phase (no curriculum, no escalation) because the architectural test is more focused. But the chat can argue against this.

### 5.8 Loop-100 safety check status

v0's loop-100 G2.T2 evaluation was the load-bearing predictor-level gate; v1 inherits the methodology but the specific value (loop 100, 70 Stage B loops post-onset) was tied to v0's curriculum. v1 needs to derive its own equivalent from its own substrate and rep-bin coverage. The chat does this derivation; I review.

### 5.9 Pre-spec diagnostics — what runs before v1 spec locks

Three pre-spec diagnostics surfaced during intake-brief drafting:

- **§7.1 math/code walkthrough** of the variance-head + shared-parameter gradient pathway. Establishes (or refutes) the §7.1 coupling diagnosis analytically before the empirical intervention is designed. **First substantive technical task in the v1 design chat.**
- **Lightweight-SIGReg fine-tune on existing data.** Resolves §7.2 / §3.1 encoder question cheaply.
- **V-JEPA 2 retry on current substrate.** Resolves whether V-JEPA 2 fails on the corrected substrate or just on the prior project's substrate.

The chat decides with me which to run before spec lock vs which to defer or skip. My current leaning: walkthrough definitely before lock; V-JEPA 2 retry probably before lock; lightweight-SIGReg fine-tune optional depending on walkthrough result.

---

## 6. Out-of-scope for v1 (deferred to v2 or later)

Capturing these here so they don't bleed into v1 scoping conversations.

- **Outer-JEPA mediation logic.** v0 spec §6.7 deferred this; v1 retains the deferral.
- **Action and reward.** v0 spec §6.8 deferred; v1 retains. Agent is a passive observer.
- **Sleep-phase consolidation / Bush adaptive trail growth.** v0 spec §6.5 deferred; v1 retains. We are nowhere near needing a sleep phase yet.
- **Bi-hemispheric retrieval.** v0 spec §6.6 deferred (architecturally supported via the variance representation but not exercised in retrieval); v1 retains the deferral by default, *unless* the chosen architectural direction (8.2) specifically benefits from exercising central-vs-outer retrieval as a test of whether the new variance representation supports it. Open question for the v1 design chat.
- **Depth modulation beyond recency.** v0 spec §6.4 deferred; v1 retains.
- **Multi-scale operation (variable W or K).** v0 spec §6.3 deferred; v1 retains.
- **Variable tempo trajectories.** v0 spec §6.9 deferred; v1 retains. Consistent-tempo trajectories only.
- **Multi-modal extension.** v0 spec §5.4 noted as not checked; v1 single-modality only.
- **Episodic-to-semantic consolidation.** Mentioned in user memories as "post-scope considerations"; remains post-scope through v1.
- **Multi-hop retrieval / spreading-activation.** Mentioned in user memories as a citable comparison baseline; out of scope for v1 implementation, may inform evaluation framing.
- **Outward-facing publication / external comparison work.** Push hold remains in effect. v1 is internal.

---

## 7. Questions I'm holding that didn't make it into the closing doc

These are the live-but-unresolved threads I want the v1 design chat to know I'm carrying.

### 7.1 Is the coupling-mechanism diagnosis actually right? (Math/code walkthrough is the first substantive task)

The closing doc §7.1 makes a clean architectural argument: per-K-step scalar variance head has no per-position parameters, gradient backprop has no pathway to route updates to specific positions, therefore coupling is what the topology must produce. This is a theoretically clean diagnosis, but it's still a hypothesis. The empirical evidence is the per-ordinal cross-loop input variation diagnostic — Bed ord 9/10 input invariance with drift indistinguishable from other ordinals.

**Nothing about Inner PAM is a black box.** CC wrote every line. The variance head, the loss function, the gradient pathway are all readable. We can trace analytically which parameters update on which inputs and identify exactly where position-uniform updates originate.

**The §7.1 diagnosis is a mathematical question first, an empirical question second.** The v1 design chat's first substantive technical task is a walkthrough of the variance-head and shared-parameter gradient flow, establishing the diagnosis from the code and math directly. If the walkthrough confirms the §7.1 diagnosis cleanly, we have mathematical certainty before designing the empirical intervention. If the walkthrough reveals additional coupling sources (e.g., shared transformer-body parameters producing topologically-equivalent uniform updates regardless of the variance head's parameter layout), we know v1's intervention scope needs to be larger than just the variance head.

The walkthrough is cheap (hours, not days). It is the highest-leverage pre-spec diagnostic v1 can run. Empirical confirmation (e.g., freezing the variance head and observing residual coupling) is supplementary, not primary.

### 7.2 V-JEPA 2 retry on the current substrate

User memories carry V-JEPA 2 (SIGReg ViT-Tiny) as the visual backbone for the Weft research line. V-JEPA 2 was tried on the *prior project's* substrate and found insufficient — inter-room cosines collapsing to ~0.95+ — which is why DINOv2-large was substituted for v0.

The current substrate is different from the prior project's: corrected curriculum, continuous-motion, view variation across loops where present, the eight v0 substrate findings resolved or characterised. Whether V-JEPA 2 still fails on the current substrate is an open empirical question.

A V-JEPA 2 retry run is a concrete pre-spec diagnostic. Run V-JEPA 2 substrate verification on the current seed-7 stream and see whether it passes §5 protocol at acceptable cross-element distinguishability and jitter stability. If it passes, V-JEPA 2 becomes a much cheaper alternative to SIGReg-from-scratch for any v2 encoder work, and may also be relevant for v1's encoder choice. If it fails, the deferral to v2 stands.

The v1 design chat schedules this against the math/code walkthrough and the optional lightweight-SIGReg diagnostic.

---

## 8. Operational logistics for the v1 design chat

### 8.1 Reading order

The v1 design chat should read in this order:
1. This document (intake brief) — strategic state of play.
2. `WEFT_INNER_PAM_v0_CLOSING.md` — v0 verdict and disambiguating-evidence framework.
3. `research_operations_v1.md` — process discipline (unchanged).
4. `CODING_STANDARDS.md` — operational discipline (unchanged).
5. `WEFT_INNER_PAM_v0_Spec.md` — architectural spec at v0 close, including §5.8 and §11 V0 verdict block.
6. `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md` — reference for what a full-discipline instructions doc looks like; not directly inherited.
7. `HANDOFF.md` — current repo state.

Optional supporting data files (raw JSON for verifying specific numbers if needed):
- `variance_by_ordinal.json`
- `within_loop_invariance.json`
- `per_ordinal_cross_loop_input.json`

### 8.2 What the v1 design chat produces

In rough order:
1. A reading plan + initial read of the strategic state of play. Identifies where my §3 leanings hold up against the closing doc evidence and where they don't.
2. **The §7.1 math/code walkthrough of the variance-head and shared-parameter gradient pathway.** Highest-leverage first task.
3. A pass through the open-decision agenda in §5, working through each decision with me explicitly.
4. The pre-spec diagnostics agreed in §5.9 (V-JEPA 2 retry; lightweight-SIGReg fine-tune if walkthrough leaves the encoder question open).
5. The v1 spec (architecture document) — fresh from scratch per §5.4.
6. The v1 instructions doc (execution plan).
7. Adversarial review with secondary reviewer (Grok or another model) per research_operations §2.2.
8. Resolution pass on review findings.
9. CC instruction documents for v1 implementation (later, after the spec and instructions are locked).

### 8.3 What the v1 design chat does NOT do

- Does not commit anything to the repo without my authorisation (same pattern as v0 reviewer chats).
- Does not start CC implementation without spec + instructions + adversarial review complete.
- Does not push to remote (push hold remains in effect).
- Does not relitigate v0's verdict. The closing doc is canonical.
- Does not extend scope beyond what §5 decides. "While we're at it" expansions go to v2 design.

---

## 9. Where this brief might be wrong

A short list of places I'm explicitly uncertain about my own leanings, so the v1 design chat knows where to apply pressure.

- **§2's test-discipline-with-pivot-allowance framing.** This is sharper than pure test or pure exploration, but it's a frame I'm articulating for the first time. The chat may surface ways the frame is internally inconsistent or fails to bind in practice.
- **§3.3's reframing of 8.3 as substrate baseline.** This reframing was an in-the-moment realisation while drafting. It might be wrong. The chat tests it.
- **§3.4's framing of 8.4 as companion not primary.** I'm dismissing 8.4-as-primary on ecological-validity grounds, but if v1 is structurally compute-constrained or if the architectural concept is far enough from working that a clean architecture-only test is more valuable, synthetic-as-primary may be the right choice.
- **§5.4's confirmed fresh-from-scratch spec.** Could go either way. A delta document might be the right choice if v1's architectural direction is tightly scoped (e.g., 8.2 alone with v0's everything-else preserved). I've locked fresh; chat can argue.
- **§7.1's variance-head-vs-shared-parameter question.** This is what the math/code walkthrough resolves. Genuinely uncertain on the answer until the walkthrough runs.

---

*End of v1 design intake brief. Companion to `WEFT_INNER_PAM_v0_CLOSING.md`. v1 design chat opens with this document plus the closing doc as primary input.*
