# Weft Inner PAM v2 — Specification (Pass 1: §§1–3)

**Status.** Methodology-and-scope layer plus substrate-and-sweep design, for adversarial review alongside the revised pass 2 (§§4–9). Design-chat corrections applied: continuity/coupling axis swap; project-arc paragraph deferred to v2 closing; W4 outcome added; property-first ordering acknowledged; five-crosses commitment; single L_d_main commitment.

**Companion documents.** `WEFT_INNER_PAM_v2_Spec_pass2_sections_4_to_9.md` (review-ready; depends on this pass 1 for axis selection and sweep structure); `WEFT_INNER_PAM_v1_CLOSING.md` (primary input from v1; especially §§10–12); `WEFT_INNER_PAM_v0_CLOSING.md` (institutional memory; BCDD evidence at §7.1); `research_operations_v1.md` (operational discipline).

**Scope conditioning.** v2 is a methodology-design experiment, not an architectural-verdict experiment. v2's deliverables condition on the v1 scaffold (three predictor arms, online trainer, evaluation framework) and v0/v1 institutional memory (BCDD readout-topology attribution; substrate-prerequisite finding; eight v0 + v1 finding 9 substrate properties on the v0/v1 naturalistic build). v2's substrate is synthetic; v2 produces a methodology for tuning environment-encoder-architecture fit, plus a single worked example applying the methodology to the v0/v1-lineage (env, enc) pair.

---

## 1. Deliverables, scope, and worked example

### 1.1 What v2 produces

v2 produces a **methodology for diagnosing where any (environment, encoder, architecture) triple lands relative to the architecture's working region, and what that landing implies for the system's behaviour**. The methodology articulates a tuning process for environment-encoder-architecture fit; it does not pick a privileged answer for any specific (env, enc) pair.

The framing rests on a three-variable view of coherent agents: environment, encoder, and architecture are co-shaped, not independently choosable. v0/v1 implicitly held environment constant (AI2-THOR) without naming it as a variable. v2 names it as the held-constant variable explicitly and produces a methodology whose application to other environment-encoder pairs is the v3+ portability question.

v2 does **not** produce: an architectural verdict on the v1 claim (deferred to whatever the methodology says about the v0/v1-lineage worked example, which produces empirical input rather than verdict-shaped resolution); a privileged "right" stream-property region for the architecture (the right region is environment-encoder-relative, not universal); a recommendation on encoder choice independent of environment (the recommendation lives in the methodology's application step, not as a v2 deliverable).

The methodology framing's portability beyond v2 — articulation as project-level structure for v3+ — is closing-time material, not spec-time. v2 closing refines the framing once execution surfaces what generalises and what doesn't.

### 1.2 The two outputs

The methodology has two distinct outputs, both load-bearing, articulated separately so spec discipline tracks them independently.

**Output 1: Characterisation map.** The architecture's response surface across stream-property space. Built once via the synthetic sweep (§§2–3). Reusable across any (env, enc) pair whose landing region falls inside the characterised area. Answers, for any point in property space the map covers: does the architecture work here, and how does it fail when it fails?

**Output 2: Measurement protocol.** A procedure for measuring where any given (env, enc) pair lands in stream-property space. Applied per (env, enc) pair, much cheaper than building the map. Inputs: an environment with a trajectory-producing policy, an encoder, and a finite trajectory collection. Outputs: empirical distributions on each of the property axes the map characterises, projected to a region within the map's coordinates.

The conditional claim "for (E, F), the system lands at R; R is a working region if it has properties X, Y, Z" requires both outputs. v2 produces both; v2's spec discipline applies to both. Underspecifying either invalidates the methodology.

**SCAFFOLDING note on map-protocol coupling.** The mapping between protocol-measured distributions and map-coordinate points is itself a methodology design choice. Labelled SCAFFOLDING; resolution in pass 2 §6.3.

### 1.3 The worked example: DINOv2-on-AI2-THOR

v2 produces one worked example: applying the methodology to the v0/v1-lineage (env, enc) pair (AI2-THOR with the v0/v1 substrate; DINOv2 encoder). The worked example is committed to because **whether this pair lands in a working region is itself a diagnostic question the map exists to answer**. v0/v1's troubles were attributed to readout topology (BCDD evidence) and perturbation-mechanism limitations (PRE-B). Neither attribution actually checks whether DINOv2-on-AI2-THOR lands in a region where the architecture works at all. The worked example checks.

The worked example is framed **diagnostically, not demonstratively**. Default language ("demonstrate the methodology on DINOv2-on-AI2-THOR") understates what the worked example does. The committed framing: *apply the methodology to DINOv2-on-AI2-THOR to determine where the v0/v1-lineage pair lands in property space; the result is empirical input to v3 design, regardless of which region the pair lands in*.

### 1.4 Pre-committed outcome space for the worked example

Per v1 closing §9.3's discipline (substrate-prerequisite findings are first-class outcomes, not protocol-failure subcategories), v2 pre-commits at design time to the outcome space the worked example produces. This discharges the temptation to retrofit interpretation around an "unexpected" outcome.

**Outcome W1 — Lands in working region.** The architecture's behaviour at DINOv2-on-AI2-THOR's measured property region matches the working-region pattern the map identifies. Implication: v0/v1's coupling result is attributable to architecture-and-experimental-design (BCDD's readout topology, v1's verdict-and-substrate-prerequisite framing) rather than encoder-environment misfit. The v1 architectural fix is supported as the right intervention for this (env, enc) pair; v3 inherits the methodology and the fix together.

**Outcome W2 — Lands in non-working region.** The architecture's behaviour at DINOv2-on-AI2-THOR's measured region matches a non-working pattern. Implication: v0/v1's troubles include encoder-environment misfit, not only architecture-and-experimental-design. v3 has to engineer around the fit (encoder change, architecture revision for that region, or environment change), not just the architecture in isolation. The "DINOv2 might be the wrong encoder for AI2-THOR entirely" hypothesis is empirically grounded.

**Outcome W3 — Lands on a boundary or in ambiguity.** The pair is marginal — at the edge of a working region, or in a region where the architecture's behaviour is mixed or under-characterised. Implication: v0/v1's results are partly explained by being near a working-region edge; v3 inherits both the methodology and a sharpened question (what does it take to move the system into the interior of a working region; is that an encoder change, an environment change, or an architecture change).

**Outcome W4 — Falls outside the map's coverage.** The measured region for DINOv2-on-AI2-THOR falls outside the per-axis ranges the map characterises. Implication: the first-principles sweep ranges (§2.4) under-spanned what real (env, enc) pairs produce. The methodology learning is that range derivation was insufficient — itself a v3+ inheritance about how to scope sweeps. W4 is pre-committed at design time per the §9.3 discipline rather than being a post-hoc surprise; the methodology can be re-applied to DINOv2-on-AI2-THOR at v3 timing after sweep ranges are adjusted.

All four outcomes are substantive. The worked example cannot fail.

### 1.5 Scope: held-constant variables and deferrals

v2 holds environment constant (AI2-THOR for the worked example; no environment is involved in the map's synthetic substrate, but the methodology's environmental-variability claim is not tested by v2). v2's worked example uses one encoder (DINOv2); the methodology's encoder-variability claim is not tested by v2 either.

**Deferred to v3+, as a single coherent question:** does the methodology survive variation in both environment and encoder? v3+ applies the methodology to at least one non-AI2-THOR environment and at least one non-DINOv2 encoder, ideally as separate worked examples that, together with v2's, populate the portability claim across both axes.

Rationale for the single-worked-example commitment over multiple-worked-examples-at-fixed-environment: multiple encoders at fixed environment sits awkwardly between v2's "methodology applied once" and v3+'s "methodology applied across environments." The middle category invites the portability question v2 has deferred. Single demonstration at v2 produces a cleaner v3+ inheritance: map + protocol + one demonstrated application + the methodology articulation, with encoder-variation and environment-variation portability handled together by v3+ as a single coherent question.

### 1.6 Inherited commitments from v1

v2 inherits, without re-testing:

- The v1 scaffold's three predictor classes (Primary, Ablation 1, Ablation 2) and the architectural property assertions (PRE-D, all PASS) per v1 spec §§7.2.4 / 7.3.4 / 7.4.4.
- The online trainer's skip-until-W contract, per-(item, ordinal) callback discipline (reframed as per-stream-point callback discipline for synthetic substrate), AdamW + grad-clip, resume-from-checkpoint.
- The evaluation framework's per-stream-point granularity (renamed from per-(item, ordinal); same mechanism, different stream-construction context).
- The corrected parameter-count envelope for L_d ∈ {1, 2, 3, 4}: Primary 17.9M → 30.5M, Ablation 1 identical to Primary minus 512 params per L_d, Ablation 2 21.6M at L_d=2.
- v0/v1 institutional memory: substrate findings 1–8 (re-verification not applicable to synthetic substrate), finding 9 (ProcTHOR-renderer-binding; relevant only to the worked example's measurement step), BCDD attribution of v0 coupling to pooled-readout topology.

v2 does **not** inherit: the V1A/B/C/D/E/P verdict schema (mismatched to characterisation outcomes); the perturbation magnitude band [0.05, 0.10] (substrate-relative; in synthetic substrate, magnitude is a sweep dimension); the §1.2 commitments quantifying substrate properties (perturbation magnitudes, locality thresholds, reproducibility tolerances become inputs to construction, not constraints to verify); AI2-THOR-as-substrate for v2's primary characterisation work (used only for the worked example's measurement step).

### 1.7 Evaluation questions

Per `research_operations_v1.md` §3.1, the evaluation question is stated explicitly before metric design. v2 has three:

1. **Map evaluation question.** For each point in the sampled stream-property space (at the architecture configurations swept), does the architecture produce co-primary per-stream-point differentiation in mean and variance? The "co-primary" framing inherits from v1 spec §1.4 — both must hold for the architecture to be "working" at that property point.

2. **Protocol evaluation question.** For the DINOv2-on-AI2-THOR (env, enc) pair, where in stream-property space does the pair's empirical distribution land? Distributions, not points; the measurement protocol's output is itself a region, not a coordinate.

3. **Worked-example evaluation question.** Does the protocol's measured region for DINOv2-on-AI2-THOR overlap a working region in the map? Four outcomes per §1.4.

Note that no v2 evaluation question takes the form "does the architecture work" in absolute terms. The map characterises *where* it works; the protocol measures *where the pair lands*; the worked example combines them.

---

## 2. Substrate construction

### 2.1 Option β commitment

v2's substrate is **Option β: synthetic embeddings constructed directly in d=1024 space, with no encoder in the loop**. The architecture under test reads d=1024 input streams that are constructed programmatically; the encoder layer of the stack is not present.

The Option β / γ / something-else decision was raised in v1 closing §§10, 12 as an elevated-weight question for v2 design intake. Resolution rationale, recorded here per `research_operations_v1.md` §2.3:

**Why Option β, not Option γ (synthetic images + designed encoder).** BCDD's mechanistic evidence (v0 closing §7.1 revised) demonstrates that v0's coupling pathology is readout-topology-mediated and downstream of the encoder: uniform `last_token` drift on bit-identical input, with the mean head exhibiting the same uniform-drift signature as the K-scalar variance head despite K·d more parameters. The pathology is a readout-topology + path-prediction-loss interaction, not an encoder-output structure interaction. If the v1 architectural fix (K positional output queries; per-K-position variance) sits at the readout downstream of the encoder, characterisation of the fixed architecture against stream property space does not require the encoder to be present in the loop. Option γ's standalone cost (designing a synthetic image generator; designing an encoder; verifying both in PRE-style protocols; integrating into the sweep) is not justified by the residual encoder-coupling concern that BCDD's evidence already addresses.

**Why Option β does not collapse the stack-fitting reframe (v1 closing §10).** Stack-fitting argues that environment, encoder, architecture, and recall co-shape. Option β does not deny this; it isolates the architecture-level characterisation by removing the encoder confound, on the empirical basis that the architecture's pathology being characterised is readout-topology-mediated. The encoder's role under the methodology is to map (environment, policy) data into d=1024 space; the measurement protocol (§1.2 Output 2) characterises where that mapping lands. The encoder is not absent from the methodology; it is absent from the map's construction.

**Residual concern Option β must address: stream construction calibration.** β constructs streams in d=1024 space; nothing guarantees these streams span regions actual encoders produce on actual environments. A characterisation map covering a property region no encoder ever produces is precise but irrelevant. §2.4 addresses this concern via first-principles property ranges rather than anchoring to a single (env, enc) measurement.

### 2.2 Stream characterisation in d=1024

Streams are produced as sequences of d=1024 vectors, each step a single vector, indexed by stream position. A stream is characterised by its property values along each of the §3.1 axes (perturbation magnitude, locality, continuity, repetition structure, manifold dimensionality), measured as defined in pass 2 §4. Sweeping property axes means producing streams with controlled values on these properties.

The synthetic substrate is deliberately simple — its advantage is interpretability and control, not realism. SCAFFOLDING items in §2.6 list the substrate-level commitments calibrated empirically rather than declared a priori; construction-primitive specifics live in pass 2 §5.

### 2.3 Property-first ordering: where pass 1 commits and where pass 2 derives

Pass 1 commits to **which** stream properties the methodology characterises (§3.1) and the coarse-grained sweep structure that traverses them (§3.3). It does not define the properties themselves or specify how synthetic streams instantiate them. The order is **property-first**: each axis is defined in pass 2 §4 in terms of measurable observables on a finite trajectory in d=1024 space; construction primitives that produce streams with controlled values on those observables are derived in pass 2 §5 from the §4 definitions.

The ordering matters for methodology portability. The measurement protocol (Output 2) measures the §4 observables on an empirical (env, enc) pair's trajectory. If property axes were defined in terms of construction-primitive parameters first, the axis values would be substrate-construction-specific and the protocol's mapping back to map coordinates would be ill-defined for empirical (env, enc) pairs. Property-first ordering preserves the methodology's claim that any (env, enc) pair can be located on the map. Same shape as the Option β cross-check distinction at one level down: separate the architecture's response surface from any specific instantiation.

This pass 1 commits to: which axes the map characterises (§3.1); at what sweep structure (§3.3 — five parallel crosses + L_d sweep at the worked-example point + interaction probes); against what controls (§3.4). Pass 2 commits to: the measurable definitions of each axis (§4); the construction primitives that produce streams with controlled axis values (§5); the measurement protocol that applies §4 definitions to arbitrary (env, enc) pairs (§6).

### 2.4 First-principles property ranges

Sweep ranges for each §3.1 axis are derived from first-principles plausibility, not anchored to DINOv2-on-AI2-THOR statistics alone. The principle: sweeps must span what plausible (encoder × environment) combinations could produce, not what one specific combination produces.

Each axis range commits to three reference points:

- **Lower bound.** The smallest value at which any plausible (env, enc) pair could land. For magnitude: zero (bit-identical streams). For dimensionality: 1 (degenerate manifold). For repetition: no repetition (each stream position unique).
- **Upper bound.** The largest value at which any plausible (env, enc) pair could land. For magnitude: full-stream replacement (cosine drop of ~1.0 magnitude). For dimensionality: d=1024 (full ambient space). For repetition: full pattern saturation.
- **DINOv2-on-AI2-THOR measured value.** Where the v0/v1-lineage (env, enc) pair lands on this axis. Empirically grounded; used to ensure the sweep includes the worked-example region and to verify the protocol's measurement returns a value the map covers. Not privileged as a sweep-density anchor — see §3.5.

This addresses the §2.1 residual concern: by spanning first-principles plausible ranges with the DINOv2-on-AI2-THOR measurement included as one point, the map covers both the worked-example region and the broader space relevant to v3+ portability. The map is not built around the worked example; the worked example is one point in a map built around the property axes themselves.

### 2.5 The worked example's empirical measurement step

The DINOv2-on-AI2-THOR worked example requires a measurement step distinct from the map's construction: forward-pass DINOv2 over an AI2-THOR trajectory collection (the v0/v1 substrate, inherited; v1's continuous-motion explorer producing the same trajectory class v0/v1 trained on), then compute the per-axis property distributions per the measurement protocol (Output 2 of the methodology).

This measurement step is the **first instantiation of the measurement protocol**. Designing the protocol and applying it to DINOv2-on-AI2-THOR happen together; the protocol's design is constrained by what it has to be able to measure on the v0/v1-lineage pair, and the protocol's first-application validation is the worked example's measurement step. This is intentional bundling — the protocol must be applicable to at least one real (env, enc) pair to be a methodology output rather than a hypothetical procedure.

The v0/v1 substrate inheritance for this step uses the v1 build (post-PRE-A `_assign_close_up_ordinals` fix; v1 finding 9 noted but irrelevant to encoder forward-pass since no mutation API is exercised; ProcTHOR-renderer-binding doesn't affect DINOv2's forward pass over pre-rendered frames).

### 2.6 SCAFFOLDING parameters

Per `research_operations_v1.md` §§2.3 / 7.1–7.3, every fixed parameter at this stage is labelled.

**SCAFFOLDING — calibrated against measurement before sweep:**
- The specific manifold dimensionalities sampled (lower, intermediate, upper).
- The specific magnitude values at each sweep step.
- The specific repetition periods at each sweep step.
- The specific continuity values at each sweep step.
- The specific locality values at each sweep step.
- The number of stream-positions per stream (analogous to v0's frames-per-loop).
- Per-sweep-point sample size (committed in pass 2 §9.3 at n=10).

**ARCHITECTURE — load-bearing v2 commitments:**
- Option β substrate (no encoder in loop).
- d=1024 embedding dimension (inherited from v0/v1 scaffold; matches DINOv2 output; not swept).
- The three predictor arms (Primary, Ablation 1, Ablation 2) from the v1 scaffold.
- W=16 sliding window (inherited from v1 scaffold; not swept).
- Path-prediction loss formulation (inherited from v0/v1; not swept).

Items currently SCAFFOLDING are promoted to ARCHITECTURE only with explicit rationale at spec-finalisation review.

---

## 3. Stream-property axes and sweep design

### 3.1 Property axes — decision-anchored selection

v1 closing §12 listed nine candidate axes. v2 selects a subset anchored to the four decisions v2's deliverables enable:

- **D1 — v3 substrate choice** (continue synthetic / return naturalistic / hybrid). Map informs whether the working region overlaps the worked-example region.
- **D2 — v3+ encoder design** (when v3 attempts environment-substitution portability). Map identifies axes the architecture is sensitive to; those become encoder design targets.
- **D3 — Architectural revision triggers.** If the architecture fails across most of the map, that's evidence the v1 fix is insufficient and the next move is architecture revision, not substrate engineering.
- **D4 — Multimodality triggers (v3+).** If failure modes pattern-match "vision-only signal insufficient for resolution at this property region," that's a multimodality signal for v3+ to act on.

Five axes selected, with the decision each anchors:

1. **Perturbation magnitude.** D1, D3. The v1 middle-band finding (PRE-B) sits on this axis; v2 sweeps it as a primary axis to determine whether the architecture's working region is magnitude-bounded.
2. **Perturbation locality.** D2, D3. v0 finding 8 (cross-room visual leakage) and v1's locality criterion identify locality as a measured property. Sweeping it characterises architecture sensitivity to localisation in stream changes.
3. **Continuity / smoothness.** D1, D2, D3. v0's continuous-motion substrate work (v0 substrate findings 3, 4) sat at a specific continuity value implicitly; v2 makes the inheritance explicit by characterising the architecture's behaviour across the continuity range. Continuity is plausibly variable across (env, enc) pairs — different policies and different encoders produce streams with different consecutive-step similarities — so sweeping it tests methodology portability against an axis that v3+ environment-substitution will vary substantially.
4. **Repetition structure.** D1, D3. The Bush associative-trail thesis treats repetition as the mechanism by which associations form; v0's §2.2 (shape learning through repetition) made this load-bearing for the architectural claim. Sweeping repetition structure tests whether the architecture's working region requires specific repetition patterns. Also subject to the §4.2 scope restriction (locality definition depends on repetition presence); the trajectory's repetition value floors the methodology's protocol applicability.
5. **Manifold dimensionality.** D2, D3. A property axis where (env, enc) variation is plausibly large — different encoders project to different effective dimensionalities. Sweeping it characterises whether the architecture's working region is dimensionality-bounded.

**Inter-item coupling not in main effects.** v1 design intake treated coupling as central. v2 demotes it for two reasons: (i) coupling's operationalisation is tightly bound to v0/v1's item structure (Bed/TV/Dresser/Sofa as discrete repeating "items"), which v2 is supposed to characterise more generally — environment-substitution at v3+ may not preserve item structure at all; (ii) coupling fits naturally as an interaction-probe target ("does the architecture's behaviour at high-magnitude perturbation change when items are highly coupled?") without needing main-effects coverage. Treated as an interaction-probe candidate in §3.3; construction primitive specified in pass 2 §5.4.

Three candidate axes from v1 closing §12 not selected at this stage: persistence, noise floor, categorical-vs-continuous variation. Rationale: each is plausibly relevant but does not anchor a v2-decision in the decision-to-axes table. They may be reintroduced as interaction-probe axes (§3.3) if main effects surface coupling. Explicit deferral, not exclusion.

### 3.2 L_d as characterisation axis

L_d (decoder layers in the v1 scaffold's three predictor classes) is treated as an architecture-property axis swept at the worked-example point, not as a hyperparameter selected for the main sweep.

Rationale, resolving v1 closing §12's elevated-weight question: under v2's characterisation framing, capacity is exactly the variable you characterise across rather than select for. The v0-baseline comparison that anchored v1's "best-differentiation-within-stable" rule weakens under Option β substrate (v0 was DINOv2-coupled training; v2 is synthetic; the architectures aren't in comparable substrates). Sweeping L_d as an axis at the worked-example point sidesteps the selection problem and produces capacity-dependence information at the most decision-relevant stream-property location.

L_d sweep range: L_d ∈ {1, 2, 3, 4} per v1's corrected envelope (Primary 17.9M → 30.5M).

### 3.3 Three-crosses sweep design with two L_d_main values

A full factorial across L_d × five stream-property axes × multiple levels × three predictor arms is computationally explosive and would produce most of its sample points in regions where the architecture's behaviour is either uniformly successful or uniformly failing — wasted signal. A single cross through 5D space (sweep one axis at a time, hold others at midpoint) characterises main effects but provides no information about interactions — the surface's curvature and any axis × axis coupling is invisible. v2 uses an intermediate: **three corner-avoiding crosses through 5D property space, replicated at two L_d_main values**, characterising main effects with interaction-detection built in across both stream-property axes and capacity.

**Main effects via three corner-avoiding crosses at L_d_main.** When sweeping axis k, the other four axes are held constant at one of three configurations — the 2nd value, midpoint, and 4th value of the per-axis grid. The per-axis extremes are deliberately excluded from held-axis positions to avoid sampling 5D corners where §2.4 first-principles plausibility is weakest and reachability by any (env, enc) pair is doubtful (extremes appear only when they are the swept axis's value, so the map still characterises per-axis extreme response under one held-axis context). At each sweep point in each cross, all three predictor arms are trained and evaluated.

The three-crosses design gives axis × axis interaction information from main effects: if the three crosses along axis k agree (same response shape), axis k's main effect is independent of other-axis position — real main effect, no interactions. If they disagree, the disagreement is itself the interaction signal, surfaced from main-effects rather than discovered post-hoc and chased with probes. Interaction detection becomes the main-effects sweep's natural output, not a separate phase.

**Two L_d_main values at baseline.** Main effects run at L_d ∈ {1, 4} — endpoints of the corrected envelope (Primary 17.9M and 30.5M parameters; matching capacity range with the L_d sweep at the worked-example point). The two-L_d_main commitment gives axis × L_d interaction information from main effects via the same disagreement principle that gives axis × axis interactions from three crosses: if the architecture's working-region pattern at L_d=1 disagrees with the pattern at L_d=4 along some axis, the disagreement is the L_d × stream-property interaction signal. Single-L_d_main would leave the §8.1 M5 "L_d_main-specific structure" instance unfalsifiable from main effects; two-L_d_main makes it detectable.

**L_d sweep at the worked-example stream-property point.** L_d ∈ {1, 2, 3, 4} (full range) is swept with stream properties held at the DINOv2-on-AI2-THOR measured values (or the nearest sweep-grid point; §3.5 covers the measurement-to-grid mapping). All three predictor arms at each L_d. Output: capacity-dependence at the worked-example region, supplementing the two-L_d_main main-effects characterisation with finer-grained capacity sampling at the most decision-relevant stream-property location.

The worked-example point's privilege is "we have empirical measurements for this combination, so we can demonstrate methodology application here," not "this is the decision-relevant point." The full L_d sweep happens here because the worked example must characterise capacity-dependence as part of methodology application, and the worked-example point is the only point in v2 where we have empirical (env, enc) grounding.

**Interaction probes at coupling points identified post-main-effects.** If the main-effects sweep (via three-cross disagreement, L_d_main disagreement, sharp transitions in working-region status, or asymmetric ablation behaviour) surfaces structured behaviour at specific stream-property regions, additional sample points run the L_d range at those regions. These are diagnostic-pair probes (`research_operations_v1.md` §1.4 cheapest-diagnostic discipline); selection and budget per pass 2 §9.4 (4 probe locations baseline; slack reallocates to higher n at existing probe points per the §9.8 reliability-over-coverage principle).

**Inter-item coupling as interaction-probe candidate.** Per the §3.1 demotion, inter-item coupling is not a main-effects axis. A natural probe pairing is coupling × magnitude — does the architecture's behaviour at high-magnitude perturbation change when items share trajectory structure? Pass 2 §5.4 specifies a coupling construction primitive for this use; pass 2 §9.4 decides whether the probe budget covers it.

**Wall-clock vs density trade-off resolved post-PRE.** The arm-run envelope under three crosses × two L_d_main × n=10 is ~5460 arm-runs total (including PRE-D2's empirical n=10-vs-n=20 CI validation per pass 2 §9.3 override discipline). Per-arm-run cost is unknown until V2-PRE-D measures it. Pass 2 §9.5 commits to a **mandatory post-PRE checkpoint** between Phase 0 and Phase 1: design chat receives PRE-D's measured per-arm-run cost, the n=10 vs n=20 CI estimate against the §7.3 τ_W threshold, and corner reachability characterisation; extrapolates the full envelope; commits to wall-clock vs density trade-off with empirical evidence. Three crosses can extend to five (or four) crosses if wall-clock allows; alternatively, density can reduce if extrapolated wall-clock is intolerable, subject to the §9.1 hard floor (three crosses, three values per axis; below the floor, design chat re-scopes axes or defers the sweep to v3+). This is a scheduled checkpoint with empirical input, not a contingency that may invoke.

### 3.4 Negative controls

Per `research_operations_v1.md` §3.2, controls are part of the design at day one, not added post-hoc. v2 designs three negative controls into the sweep:

**Control C1 — Streams that violate the architectural claim's assumptions.** Specifically: streams with no repetition structure (every stream position unique; the architectural claim about associative-trail formation cannot apply). Expected behaviour: the architecture produces no per-stream-point differentiation; failure mode is *across-the-board indistinguishability*, not coupling. If the architecture *does* produce differentiation here, the architectural claim's framing is wrong — it's not what we thought it was learning.

**Control C2 — Shuffled-temporal-order streams.** Stream positions from a repeating stream are presented in random order, breaking the temporal structure but preserving the per-position content distribution. Expected behaviour: lower differentiation than ordered streams at otherwise-matched property values. If shuffled and ordered match, temporal structure isn't load-bearing for the architecture's working region — different failure mode than expected.

**Control C3 — Bit-identical baseline.** Streams with no perturbation at all (magnitude axis lower bound). Inherits v1 spec §1.4's co-primary baseline. Architecture should produce no differentiation; signal differentiation greater than this baseline is what the map's "working region" determinations are anchored to (per pass 2 §7.3).

Controls C1 and C2 run at both L_d_main values (L_d=1 and L_d=4), matching the main-effects two-L_d_main commitment. C3 runs at every sweep point (it's the architecture's noise floor at that point, not a separate experiment).

### 3.5 Sample-point density and the worked-example-to-grid mapping

Per `research_operations_v1.md` §3.6, n<20 in high-variance domains is unreliable. Per-sweep-point sample sizes are sized to the statistical-distinguishability requirement, committed in pass 2 §9.3 at n=10.

Per-axis sample-point count: 5 points per axis (lower bound, intermediate-low, midpoint, intermediate-high, upper bound). Three corner-avoiding crosses through five axes produces 3 × 5 × 5 = 75 unique sample-point configurations per L_d_main; two L_d_main values → 150 configurations; with three arms, 450 base configurations before per-sweep-point sample size is multiplied in. At n=10: 4500 main-effects arm-runs. Plus 4 L_d points × 3 arms × n=10 = 120 at the worked-example region. Plus controls at both L_d_main values (2 × 3 arms × 2 controls × n=10 = 120). Plus interaction probes (~480 baseline). Plus PRE (~240, including PRE-D2 empirical n-validation per pass 2 §9.2). Pass 2 §9.1 compiles the full envelope at ~5460 arm-runs.

**Held-axis positioning across the three crosses (committed at spec time).** Each cross uses one of three held-axis configurations: the 2nd value, the midpoint, and the 4th value of the per-axis grid. Per-axis extremes are deliberately excluded from held-axis positions — sampling 5D corners (e.g., the all-min or all-max corners) would characterise regions where §2.4 first-principles plausibility is weakest and reachability by any (env, enc) pair is doubtful. Map's non-working-region demarcations at unreachable corners would feed v3+ inheritance as if substantive when they are methodology-internal noise. Corner-avoidance is a design-chat decision committed at spec time, not SCAFFOLDING; corner-sampled crosses can be added as Phase 3 probes if PRE-D demonstrates plausible reachability of the corner regions.

Per-axis extremes still appear in the map: they appear only when they are the swept axis's value (not the held-axis context). Each axis is therefore characterised across its full per-axis range (min through max) under three held-axis contexts (intermediate-low, midpoint, intermediate-high).

**Worked-example-to-grid mapping.** DINOv2-on-AI2-THOR's empirical property distributions don't generally land exactly on sweep grid points. Pass 2 §6.3 commits to nearest-grid-point lookup per axis, with interpolation distance reported as part of the worked-example outcome, and multi-modal detection (GMM + BIC) before summarisation. If the IQR or multi-modal analysis surfaces a pair whose distribution falls between grid points with non-trivial interpolation distance, closing records this as a methodology learning rather than as a worked-example failure.

---

**End of pass 1 §§1–3 (revised, review-ready).** Design-chat corrections applied; consistent with pass 2 §§4–9. Ready for adversarial review per `research_operations_v1.md` §2.2.

Items flagged for reviewer attention:

- **§1.4 outcome exhaustiveness.** Four outcomes (W1–W4) are pre-committed. Reviewer should challenge: are these exhaustive? Specifically, can the methodology produce a fifth outcome — e.g., the protocol returns a multi-modal distribution that lands in two different working-region statuses simultaneously? Pass 2 §6.3's multi-modal handling addresses this partially; reviewer should confirm the W1–W4 commitments cover the protocol's actual output space, or whether multi-modal pairs warrant a fifth outcome explicitly.
- **§2.4 first-principles plausibility framing.** "Plausible (env, enc) combinations" is operationalised in pass 2 §4 / §5 via measurable observables on a finite trajectory. Reviewer should check whether the operationalisation in pass 2 is principled rather than arbitrary — specifically, whether the lower- and upper-bound commitments in §2.4 survive the property-first definitions in pass 2 §4.
- **§3.1 axis selection.** Five axes after the swap (magnitude, locality, continuity, repetition, dimensionality); inter-item coupling demoted. Reviewer should challenge: are these five right? Specifically, should persistence (perturbation duration) be promoted given its relevance to v3+ environments where disturbances may have different time-scales than v0/v1's loop-based exploration?
- **§3.3 three-crosses-with-corner-avoidance design.** Three crosses (held axes at 2nd, midpoint, 4th of per-axis grid; per-axis extremes appear only as swept-axis values) commits at spec time rather than as SCAFFOLDING. Reviewer should challenge whether three crosses give sufficient axis × axis interaction resolution, or whether the conservative three-crosses choice trades away interaction signal that the abandoned five-crosses design would have surfaced. Post-PRE checkpoint (§3.3) provides an empirical-evidence path to extending to four or five crosses if wall-clock allows.
- **§3.3 two-L_d_main commitment at endpoints (L_d ∈ {1, 4}).** Two L_d_main at envelope endpoints gives axis × L_d interaction information from main effects, making the §8.1 M5 "L_d_main-specific structure" instance detectable. Reviewer should challenge whether endpoints are the right two values (alternative: L_d=2 and L_d=3 around the median, or L_d=1 and L_d=3 if L_d=4 turns out to be unstable at sweep boundaries; PRE-D's stability characterisation informs this).
- **§3.5 corner-avoidance commitment at spec time.** Corner-avoidance is committed at spec time, not SCAFFOLDING. Reviewer should challenge whether corner-sampled crosses belong in Phase 3 probes by default (in case PRE-D shows plausible corner reachability), or whether the corner-avoidance commitment is sufficient.
