# WEFT Inner PAM v2 — Design Intake (Pre-v1-Verdict Draft)

**Status.** Pre-v1-verdict draft. The body-architecture analysis below captures the reasoning state at v1 design lock-in (2026-05-17). v2 design intake will be refined after v1's verdict lands; the body-architecture analysis itself is what v0's evidence supports *now*, and is preserved here so that v2 design begins from a substantive starting point rather than reconstruction.

**Purpose.** Records the body-family architectural question that v1 explicitly defers to v2, the analysis supporting GRU as v2's primary body candidate, and the readout topology reframing that follows from a recurrent body. Companion to `WEFT_INNER_PAM_v0_CLOSING.md` (v0 institutional memory) and the forthcoming v1 spec / instructions / closing documents. Written to be v2's design starting point regardless of v1's verdict outcome.

**Scope discipline.** This document captures architectural analysis. It does *not* commit to v2 design decisions. v2 design begins with v1's verdict in hand; this document is input to that design, not specification of it. Sections marked "v2 design question" are open questions, not commitments.

---

## 1. The deferred question

v1 explicitly retained v0's transformer body and W=16 sliding window pattern, on the discipline argument that v1 should test the BCDD-implicated interventions (readout topology + variance representation + perturbation strength) without confounding them with body-family change. v2's primary architectural question is whether the body family itself supports the Inner PAM architectural claim, or whether the family — and the sliding-window pattern that comes with it — should be replaced.

The question has three dimensions, not one:

1. **Body family.** Transformer, recurrent (RNN/LSTM/GRU), 1D convolutional (CNN/TCN), state space model (Mamba/S4), or something else.
2. **Temporal-context mechanism.** Sliding window (v0's choice; transformer-natural), recurrent state (RNN-natural), causal convolution (CNN-natural), selective state (Mamba-natural).
3. **Readout pattern.** Pooled-vector readout (v0's choice; problematic per BCDD), per-output-step readout (transformer with output queries, v1's choice), per-time-step rollout (recurrent-natural), per-position output (CNN-natural).

These three dimensions are not independent. Body family typically determines temporal-context mechanism, which constrains readout patterns. v2 design needs to engage all three jointly rather than picking one and inheriting the others by default.

---

## 2. Why the body family became a v2 question

The thesis tension — Weft's research thesis commits to associative memory emerging from continuous visual experience analogous to biological mechanisms, and v0's transformer architecture violates this commitment at three load-bearing points — was identified during v1 design but explicitly deferred to v2. The deferral was a discipline choice (v1 should test what v0's evidence supports changing), not an architectural endorsement of the transformer family.

### 2.1 The three thesis-mismatched architectural commitments in v0/v1

**Single-pass per input.** Biological sensory processing handles each frame once as it arrives; v0's transformer with W=16 sliding window re-processes each frame in up to 16 different contextual positions across the trajectory. The frame at time t appears as window position W-1 at step t, position W-2 at step t+1, ..., position 0 at step t+W-1. This is computational overhead with no biological analogue and no architectural justification beyond "transformers are how the field handles sequential input."

**State carried through recurrent dynamics.** Biological sequential processing preserves order through recurrent dynamics — the state at time t is a function of state at t-1 and input at t. v0's transformer encodes sequential order through additive positional embeddings (learned `nn.Embedding(window_w, hidden)` added to input projections, predictor source line 81). Position is metadata added to inputs, not a property of the architectural dynamics. The architectural commitment to "position as additive tag on inputs" is a transformer-pattern convention, not a thesis-derived choice.

**Bounded memory through windowing rather than state compression.** Biological memory persists or fades through dynamics — old experiences gradually become less retrievable rather than being architecturally cut off at a hard boundary. v0's W=16 window means frame t-17 is architecturally inaccessible regardless of its relevance to the prediction at t. The hard boundary is a transformer-pattern convention (transformers have quadratic-in-context-length attention cost, which makes long windows expensive); it is not what the research thesis calls for.

### 2.2 What BCDD revealed that is body-architecture-specific

The body-coupling pathology BCDD identified at v0 closing is partly transformer-pattern-specific, not architecture-family-neutral.

The `last_token = x[:, -1, :]` readout pattern is a *transformer* convention — the choice to pool a transformer's W-positional output to a single "summary" vector before downstream heads. The architectural pathology BCDD diagnosed (single pooled vector that all K predictions read from, body drift across training that affects this pooled vector uniformly regardless of input) is partly a consequence of transformer outputs being W-positional sequences that require an explicit pooling-or-readout decision.

Non-transformer bodies do not have this exact problem in this exact form:

**RNN/LSTM/GRU bodies** produce per-step hidden states by the recurrence equation; each step's hidden state is the natural per-step output. K-step prediction can be implemented as K-step recurrent rollout (closed-loop), making each K-th prediction architecturally a forward-rolled version of the body's state. The "single pooled vector that all K predictions read from" pattern does not arise — each K-th prediction reads from its own forward-rolled hidden state.

**1D CNN/TCN bodies** produce per-position outputs by construction (causal convolutions over the temporal dimension). K-step prediction can read from per-time-step outputs directly. No pooling required.

**State space models (Mamba/S4)** maintain a continuous state that's natively per-position; their architectural commitment to selective state update is arguably the closest architectural analogue to biological associative-trail maintenance (the state itself learns what to retain from the experience stream).

BCDD's evidence localised v0's failure to the pooled-readout topology specifically, and v1 addresses that with K learnable output queries (which give K an upstream position dimension within the transformer family). But the deeper architectural question — *why was a pooled readout natural in v0's architecture in the first place?* — has an answer: because transformers produce W-positional output sequences that require explicit pooling-or-readout decisions, and the simplest choice (the last token) was the wrong one. An architecture that doesn't require this choice doesn't have the pathology to fix.

### 2.3 What v0's evidence did not support changing in v1

V0's verdict was V2 (shape-learning falsified, with coupling-mechanism caveat). The evidence supported changes to the readout topology, the variance representation, and the perturbation regime. The evidence did *not* directly diagnose the body family as the failing component — the body produced input-distinguished `last_token` representations for input-varying ords (Test A: last_token cosines varied with input variation), demonstrating that the transformer body was doing the per-input-distinction work the architecture intended at inference. The body's training-time drift was uniform across inputs, but this is a drift-across-training pathology on a pooled readout, not a per-input-representation pathology of the body itself.

The discipline argument for v1 was: change what evidence supports changing (readout, variance, perturbation), defer what evidence doesn't directly implicate (body, windowing). This argument was correct for v1's scope. It does not extend to v2: v2's question is not "what does v0's evidence support changing" but "given v1's verdict, what does the combined v0+v1 evidence support changing for v2."

---

## 3. The candidate set for v2's body

Four candidate body families, with v1-verdict-independent analysis. v2's actual choice depends on v1's verdict (see §6 below).

### 3.1 GRU (gated recurrent unit)

**Why.** Native sequential dynamics; no positional embeddings; single-pass processing; state carried through gate-controlled updates. Architecturally well-matched to the research thesis. Simpler than LSTM (one gate fewer, fewer parameters), often performs comparably on sequential prediction tasks. Well-understood, well-tooled, battle-tested across two decades of sequential modelling work. Debuggability is mature; reference implementations are abundant; failure modes are well-characterised.

**Practical considerations.**
- Vanishing/exploding gradients limit effective sequence length in theory, though gated mechanisms (GRU, LSTM) substantially mitigate this. For trajectories of ~36k frames (v0's scale), the practical sequence length is likely a learning-dynamics consideration, not a hard architectural limit.
- Training is sequential per-step, removing the GPU-parallelisation advantage transformers have. On 4080 Super at v0's scale (~8 hours wall-clock per arm), this is an inconvenience rather than a blocker.
- Hidden state dimension is the primary capacity hyperparameter. For matched capacity against v0's transformer (hidden=512, n_layers=4, mlp_dim=2048 ≈ 8.4M body parameters), a GRU with hidden=512 has ~1.6M parameters — significantly smaller, which is either an advantage (less overfitting risk on continuous-trajectory data) or a constraint (less capacity to learn complex prediction-tuned representations), depending on what the task actually demands.

**Architectural fit to thesis.** Strong on dimensions 1 and 2 (single-pass, recurrent state). Less strong on dimension 3 (state still has effectively-bounded memory through gate forgetting, but the bound is learned rather than architecturally fixed — closer to "memory persists or fades through dynamics" than transformer's hard window).

### 3.2 LSTM (long short-term memory)

**Why.** Same architectural family as GRU, with separate cell state for stronger long-range memory in principle. More parameters per unit (three gates plus cell state vs GRU's two gates), more capacity, more hyperparameters.

**Practical considerations.** Architecturally heavier than GRU without clear evidence that the additional machinery earns its keep on this task. LSTM's long-range memory advantage matters most when the architectural limit on effective sequence length is being pushed; for v1/v2 scale, this may be premature optimisation.

**Architectural fit to thesis.** Equivalent to GRU on the three dimensions; the choice between GRU and LSTM is a hyperparameter-and-tooling question more than an architectural-commitment question.

### 3.3 State space model (Mamba / S4 family)

**Why.** Selective state update mechanism that learns what to keep/forget from the input stream. Native sequential processing. Linear-time inference. The selective-update mechanism is arguably the closest architectural analogue to biological associative-trail maintenance: the model learns which features of the experience stream to preserve in state, rather than (a) attending to a window of recent inputs as transformers do or (b) applying fixed gate dynamics as RNNs do.

**Architectural fit to thesis.** Strongest of the four candidates. Mamba's selective state mechanism corresponds operationally to "the model learns its own trail-maintenance policy" — which is exactly what the Weft thesis asks for. State persistence is learned rather than architecturally bounded.

**Practical considerations.** Newer architecture (Mamba 2023). Less established tooling. Fewer reference implementations to validate against. Debugging Mamba-specific issues alongside Inner PAM-specific issues compounds the diagnostic load — for a v2 whose verdict needs to be defensible, this is non-trivial. The hardware-aware implementation details (scan operations, CUDA kernels) introduce engineering surface area that GRU/LSTM do not have.

**v2 design question.** Whether Mamba's thesis-fit advantage justifies the operational cost. The answer depends on v2's risk appetite: if v2 is the "test thesis-fit at maximum architectural alignment" experiment, Mamba is the right candidate; if v2 is the "produce a defensible body-family verdict with clean interpretation," GRU is the right candidate. These are different research goals and v2 design needs to pick one.

### 3.4 1D CNN / TCN (temporal convolutional network)

**Why.** Causal convolutions produce per-position outputs by construction; dilated TCNs can have very long effective receptive fields. Translation equivariance is a natural inductive bias for some sequential tasks.

**Architectural fit to thesis.** Weak. Translation equivariance is the wrong inductive bias for associative memory — associative trails are *not* translation-equivariant; the meaning of a frame depends on its position in the trajectory, not on its relative position to other frames. Fixed receptive field (set by depth × dilation rather than learned) also conflicts with the thesis: associative-trail maintenance should be a learned policy, not an architectural constant.

**v2 design question.** Whether TCN's per-position-output advantage is worth its inductive-bias mismatch. The provisional answer is no — TCN is not recommended for v2 unless v1's verdict produces evidence that specifically motivates it.

---

## 4. Readout topology under a recurrent body

If v2 changes the body family to recurrent (GRU/LSTM), the readout topology question gets reframed. v1's K learnable output queries pattern is transformer-natural; recurrent bodies have their own readout patterns.

Three candidate readout patterns for a recurrent body:

### 4.1 Per-step prediction from current hidden state

At each input time step, produce K-step-ahead predictions from `h_t` directly. Each prediction step has its own learnable output projection (K separate small MLPs reading from `h_t`). Per-K parameters live in the heads; the body produces a single per-time-step `h_t` and the K outputs are fan-outs from it.

**Architectural property.** This is structurally analogous to v0's pooled-readout pattern — all K predictions read from the same hidden vector, just with the vector being `h_t` (per-time-step) rather than `last_token` (pooled-from-window). The pathology BCDD identified could in principle re-emerge in this pattern: if `h_t` drifts uniformly across training, all K predictions inherit the drift. This is the *not-recommended* readout for v2 if avoiding v0's pathology is the goal.

### 4.2 K-step-ahead recurrent rollout

From `h_t`, predict `h_{t+1}` via the recurrence equation (closed-loop forward roll), then `h_{t+2}` from `h_{t+1}`, ..., `h_{t+K}` from `h_{t+K-1}`. Each `h_{t+k}` produces its own prediction via a shared head. The body itself produces K position-distinguished hidden states; the head is single.

**Architectural property.** K-positional structure is built into the architecture by the rollout itself, not added by per-K head parameters. The K=3 prediction is architecturally a *rolled-forward version of the body's state to step t+3*, which is exactly what "predicting forward along a trajectory" means in the associative-memory framing. This is the most thesis-aligned readout pattern: the rollout is the trail-traversal mechanism.

**v2 design question.** Whether closed-loop rollout's well-known instability properties (errors compound across rollout steps; early-stage training can produce unstable rollouts that prevent learning) are managed adequately for v2's prediction horizon (K=16 in v0/v1). If rollout instability becomes a v2 substrate issue, the per-step pattern (4.1) or an encoder-decoder pattern (4.3) becomes the fallback. Worth surfacing as a v2 substrate verification check before architectural commitment.

### 4.3 Encoder-decoder pattern

GRU encoder processes the input sequence; GRU decoder produces K outputs. Per-K position emerges from decoder dynamics. Higher parameter count; closer to seq2seq translation patterns; arguably overkill for v1/v2 task structure but worth listing as the standard-architecture option.

**Architectural property.** Decoupling encoder from decoder gives the encoder room to compress the input sequence into a "context vector" and the decoder room to produce K outputs from that context. This is closer to v0's pooled-readout pattern than to a thesis-aligned trail-traversal — the encoder still compresses to a context vector, which is the pathology v1 is trying to avoid.

### 4.4 Provisional recommendation

If v2 commits to GRU body, the K-step-ahead recurrent rollout (4.2) is the most thesis-aligned readout pattern. The per-step pattern (4.1) is the fallback if rollout instability becomes a substrate issue. The encoder-decoder pattern (4.3) is not recommended unless v2 design produces specific reasons to prefer it.

This is a v2 design question, not a commitment. The decision depends on v1's verdict and on v2's substrate verification of rollout stability at the chosen prediction horizon.

---

## 5. The framing v2 inherits from v1

v1's framing was deliberately constrained to be a transformer-family verdict, not an Inner PAM architectural verdict. The conditioning prefix v1 attaches to its verdict ("with the transformer body family and W=16 sliding window pattern retained from v0") establishes what v1's verdict can and cannot establish.

v2 inherits this framing as a strength, not a limitation. v1's verdict, whether positive or negative, becomes a *baseline against which v2's body-family change is tested*. v2 is not running a body-family change in a vacuum — it is running a body-family change against a known v1 transformer-family result. This is the cleaner v2 the v1-as-transformer-completion decision was designed to produce.

### 5.1 What v1's verdict means for v2's design

The body-family question's framing depends on v1's verdict outcome (see §6 below for verdict-conditional scope). The skeleton across all outcomes:

- **v2 tests body-family change against the v1 transformer-completion baseline.** The comparison is "does GRU body + appropriate readout + the architectural choices v1 made + matched perturbation regime + matched variance representation + matched evaluation discipline produce different results than the v1 transformer-completion baseline?"
- **v2's evaluation framework inherits v1's disaggregation discipline.** Per-(item, ordinal) granularity, the means-doing-work discipline, all carry forward unchanged.
- **v2's substrate inherits v1's perturbation regime.** Strong perturbation is now a substrate commitment, not an architectural variable.
- **v2's encoder inherits v0/v1's DINOv2 substrate.** Encoder is not the v2 architectural question.

### 5.2 The W=16 sliding window question

The W=16 sliding window is tied to the body-family question because recurrent bodies don't require windowing in the same way. v2's choice of body family implicitly determines the temporal-context mechanism:

- **GRU/LSTM body:** Recurrent state replaces sliding window. The "window" becomes effectively unbounded (state-compressed-by-gates) or set to the full trajectory length (typically with truncated backprop through time for training).
- **Mamba body:** Selective state replaces sliding window. Similar to recurrent in that the architectural "window" is effectively unbounded.
- **Hypothetically retaining transformer with non-windowed input:** Possible but expensive (quadratic attention cost over full trajectory length).

v2 design needs to surface the temporal-context-mechanism choice as a deliberate architectural commitment, not inherit it by default from the body-family choice.

---

## 6. v2 scope under each v1 verdict outcome

v2's design depends on v1's verdict in a way that goes deeper than "more or less ambitious." Different verdict outcomes produce different v2 questions.

### 6.1 v1 verdict: Outcome A (clean success)

v1 produces co-primary mean and variance differentiation; ablations attribute cleanly to readout topology (Ablation 2 reproduces v0 coupling) and variance representation (Ablation 1 shows mean differentiation without variance).

**v2 question becomes a refinement question.** "Does body-family change improve on the transformer-family baseline?" Specifically: does a GRU body with K-step recurrent rollout produce *cleaner* per-position differentiation, *better-calibrated* variance, or *broader-substrate-generalisation* than the v1 transformer-family architecture? The comparison is no longer existential ("does the architecture work at all?") but quantitative ("does body-family change earn its operational cost?").

v2 scope under Outcome A is the *smallest*: a single GRU-body arm tested against v1's transformer-family baseline. The 3-arm structure v1 designed to attribute v0's failure may not be needed if the v1 verdict is clean — v2 can focus on the body comparison directly.

### 6.2 v1 verdict: Outcome B (primary succeeds, variance representation didn't independently help)

v1's primary succeeds, but Ablation 1 (scalar variance + new readout) also produces variance differentiation, indicating that per-position variance representation wasn't load-bearing for the regime.

**v2 question: refinement plus a deferred-from-v1 question.** Body-family change as in Outcome A, plus the question of why per-position variance wasn't load-bearing — was it loss-shape coupling that disappears under the readout fix (M2), or some other dynamic? v2 may want to include a variance-representation diagnostic arm to characterise this.

### 6.3 v1 verdict: Outcome C (primary fails, transformer architecture inadequate)

v1's primary fails despite BCDD-indicated corrections; Ablation 2 cleanly reproduces v0 coupling; Ablation 1 also fails (readout fix alone doesn't help).

**v2 becomes the architectural rescue.** Body-family change is no longer a refinement question — it becomes the primary architectural question, with much stronger evidence than the thesis-tension argument alone would provide. v2's scope expands: not just "GRU vs transformer baseline" but "is recurrent architecture the right family for this task at all" with multiple body candidates worth testing (GRU primary, possibly Mamba as a second candidate if v2 has compute envelope and design time for it).

v2 scope under Outcome C is the *largest*: potentially multiple body-family arms, possibly with their own ablation structure to attribute differences. This is the outcome that motivates the most aggressive v2 design.

### 6.4 v1 verdict: Outcome D (partial / mixed)

v1 produces mixed outcomes — mean differentiation works, variance doesn't; or per-(item, ordinal) outcomes are heterogeneous (Sofa-ord-1 pattern at scale).

**v2 question: depends on the specific D-class outcome.** D1 (mean works, variance doesn't) suggests v2 should focus on variance-representation refinement, possibly with body-family change as a secondary variable. D3 (heterogeneous per-(item, ordinal)) suggests v2 should focus on what substrate properties earn architectural success — body-family change becomes one of several variables.

D-class outcomes are the hardest to plan for in advance. v2 design under Outcome D will require its own intake document, building on what v1's specific D-class outcome reveals.

### 6.5 v1 verdict: Protocol-failure outcome

v1 doesn't produce an architectural verdict because of substrate issues (analogous to v0's substrate findings). The architectural verdict is "deferred pending substrate correction."

**v2 question deferred until v1's substrate issues are resolved.** This may involve a v1.1 re-run with corrected substrate before v2 begins, or v2's design may incorporate the substrate findings directly. Depends on the specific substrate issue and whether it can be addressed within v1's frame.

---

## 7. What this document is and isn't

**This document is:** the body-architecture analysis that v0's evidence supports as v2's primary architectural question, captured at the moment when the analysis is freshest. It records the thesis-tension framing, the BCDD reinterpretation under a recurrent-body lens, the candidate set with reasoning, the readout topology reframing, and the verdict-conditional scope structure.

**This document is not:** a v2 specification. v2 design begins with v1's verdict in hand; this document is one input to that design, not the design itself. The body-family choice, the readout pattern choice, the temporal-context mechanism choice, the scope under each v1 verdict outcome — all of these are *analytic recommendations* to be revisited when v1 verdict lands, not commitments.

**This document will be refined.** After v1's verdict lands, this document gets a closing-doc-equivalent revision that integrates v1's evidence into the v2 design starting point. The body-architecture analysis itself will likely survive that revision substantially intact, because it doesn't depend on v1's verdict outcome — but the verdict-conditional scope sections (§6) will be reduced to whichever scope actually applies.

---

## 8. v2 design checklist (preliminary, to be refined post-v1-verdict)

When v2 design begins, the following decisions need to be made in order:

1. **Read v1's verdict.** Classify into Outcome A / B / C / D / Protocol-failure per §6.
2. **Determine v2's existential vs refinement framing.** Outcome A is refinement; Outcome C is existential; Outcomes B/D are case-by-case.
3. **Lock body family.** GRU as default primary candidate per §3.1; Mamba as alternative if v2's framing supports thesis-fit-at-maximum-alignment as the research goal; LSTM as alternative if GRU shows specific limitations during v2 substrate verification.
4. **Lock readout pattern.** K-step recurrent rollout (§4.2) as default for recurrent body; per-step pattern (§4.1) as fallback if rollout instability is verified at v2 substrate.
5. **Lock temporal-context mechanism.** Recurrent state for GRU/LSTM/Mamba bodies; this decision is made implicitly by body-family choice but should be surfaced explicitly in v2 spec.
6. **Lock arm structure.** Single body-comparison arm under Outcome A; multi-arm structure under Outcomes C / D depending on specific outcome.
7. **Inherit v1's evaluation discipline.** Per-(item, ordinal) granularity, means-doing-work discipline, disaggregation patterns. These carry forward unchanged.
8. **Inherit v1's substrate.** Strong perturbation, DINOv2 encoder, AI2-THOR-ProcTHOR environment, the substrate findings v0 and v1 collectively produced.
9. **Verify v2 substrate.** §5-equivalent encoder verification on whatever v2's specific substrate becomes; rollout-stability verification for recurrent-body readouts; any v2-specific substrate checks the body-family change introduces.
10. **Produce v2 spec, instructions, intake brief.** Per the v0/v1 pattern.

---

## 9. Push hold

Push hold remains in effect through v1 and v2 design. This document is institutional memory, not external-facing artefact.

---

*End of v2 design intake (pre-v1-verdict draft). Companion to `WEFT_INNER_PAM_v0_CLOSING.md`, the forthcoming `WEFT_INNER_PAM_v1_Spec.md`, and the v1 closing document that will be produced after v1's verdict lands. This document is revisited and refined after v1's verdict; the body-architecture analysis itself is preserved as v2's design starting point regardless of v1 outcome.*
