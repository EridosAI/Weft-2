# Weft Inner PAM v2 — Specification (Pass 2: §§4–9)

**Status.** Property-definition layer (§4), construction-primitive layer (§5), measurement-protocol layer (§6), evaluation-axis layer (§7), characterisation-outcome framework (§8), and compute-envelope / execution plan (§9). For adversarial review per `research_operations_v1.md` §2.2 alongside the revised pass 1 (§§1–3 with corrections from the design chat applied).

**Companion documents.** `WEFT_INNER_PAM_v2_Spec_pass1_sections_1_to_3.md` (revised); `WEFT_INNER_PAM_v1_CLOSING.md` (primary input from v1; especially §§10–12); `WEFT_INNER_PAM_v0_CLOSING.md` (institutional memory; BCDD evidence at §7.1); `research_operations_v1.md` (operational discipline).

**Pass 2 design principle.** Property-first ordering: every property axis is defined in terms of measurable observables on a finite trajectory in d=1024 space *before* any construction primitive is named. The methodology-portability test for each definition: can this be measured on a stream produced by an arbitrary (env, enc) pair, without reference to v2's construction primitives? Definitions that fail this test are substrate-construction-specific and disqualified.

---

## 4. Measurable property definitions

Each §3.1 axis is defined here as a function of observables on a finite trajectory $\{x_1, x_2, \ldots, x_T\} \subset \mathbb{R}^{1024}$. The definitions make no reference to how the trajectory was produced. The measurement protocol §6 applies these definitions to empirical trajectories from any (env, enc) pair; the construction primitives §5 produce synthetic trajectories with controlled values on these properties.

### 4.1 Perturbation magnitude

**Definition.** For a trajectory $\{x_t\}$ with a designated reference subsequence $S_{\text{ref}}$ (the trajectory's baseline state, e.g., the dominant repeating pattern) and a designated perturbed subsequence $S_{\text{pert}}$ (positions where the trajectory deviates from baseline), perturbation magnitude is

$$
M = 1 - \text{median}_{i \in S_{\text{pert}}, j \in S_{\text{ref}}}\left[\cos(x_i, x_j)\right]
$$

where the median is taken over pairs $(i, j)$ with $i$ in the perturbed subsequence and $j$ in the nearest reference-subsequence neighbourhood by trajectory position. $M = 0$ when perturbed and reference are indistinguishable in cosine; $M = 1$ when they are orthogonal.

**Measurability on arbitrary (env, enc) pair.** Requires identifying $S_{\text{ref}}$ and $S_{\text{pert}}$ on an empirical trajectory. The measurement protocol §6.2 specifies a generic procedure (repetition detection via self-similarity matrix; deviation detection via residual from the dominant repeating pattern). Both detection steps operate on cosine structure alone; no construction-side information required.

**SCAFFOLDING — calibrated.** The "nearest reference-subsequence neighbourhood" window size; the threshold for declaring a position perturbed vs reference. Pass 2 §6.2 specifies the calibration procedure.

### 4.2 Perturbation locality

**Definition.** Given the perturbed subsequence $S_{\text{pert}}$ from §4.1, locality is the fraction of trajectory positions that are not perturbed but whose cosine similarity to their reference neighbourhood is shifted by the perturbation:

$$
L = 1 - \frac{|\{i \notin S_{\text{pert}} : |\cos(x_i, x_{i'}) - \cos(x_i^{\text{ref}}, x_{i'}^{\text{ref}})| > \tau_L\}|}{|\{i : i \notin S_{\text{pert}}\}|}
$$

where $x_i^{\text{ref}}$ is the reference-state counterpart at position $i$ (the trajectory value at $i$ if the perturbation had not occurred), $x_{i'}$ is the position $i$'s reference neighbour, and $\tau_L$ is the shift threshold. $L = 1$ when the perturbation affects only positions in $S_{\text{pert}}$ (perfect locality); $L = 0$ when the perturbation shifts every other position's relations (no locality).

**Measurability on arbitrary (env, enc) pair.** Requires a reference-state counterpart $x_i^{\text{ref}}$, which doesn't exist for a single empirical trajectory. The measurement protocol §6.2 estimates $x_i^{\text{ref}}$ from the trajectory's dominant repeating pattern (the trajectory value at position $i$ in repetitions where perturbation does not occur). Operational only when the trajectory contains both perturbed and unperturbed traversals of the same trajectory region — a property the protocol verifies before computing $L$.

**Structural limitation.** Locality is undefined on trajectories whose repetition coverage falls below the calibrated threshold. The measurement protocol §6.2 returns "locality undefined" for such trajectories. This is a structural scope restriction of v2's methodology, not a methodology limitation to be engineered around in v2: the protocol applies only to (env, enc) pairs producing trajectories with repetition structure. The DINOv2-on-AI2-THOR worked-example trajectory satisfies the repetition-coverage threshold by construction (v0/v1 loop-based exploration policy); v3+ environment-substitution work inherits the restriction as a constraint on environment-policy pairs the methodology applies to, or as a v3+ protocol-extension task. v2 closing records this as methodology scope.

**SCAFFOLDING — calibrated.** $\tau_L$; the neighbourhood definition for $i'$.

### 4.3 Continuity / smoothness

**Definition.** The expected cosine distance between consecutive trajectory positions:

$$
C = 1 - \mathbb{E}_t\left[\cos(x_t, x_{t+1})\right]
$$

with the expectation taken over all consecutive pairs in the trajectory. $C = 0$ corresponds to bit-identical consecutive positions (zero motion); $C$ near 1 corresponds to consecutive positions being orthogonal (every step jumps to an unrelated location). Lower $C$ means smoother trajectory.

**Variant — local-curvature smoothness.** A secondary measurable that captures whether continuity is uniform along the trajectory or punctuated by jumps:

$$
C_{\text{curv}} = \text{std}_t\left[1 - \cos(x_t, x_{t+1})\right]
$$

High $C_{\text{curv}}$ at moderate $C$ indicates the trajectory is mostly smooth with occasional discontinuities; low $C_{\text{curv}}$ indicates uniform smoothness.

**Measurability on arbitrary (env, enc) pair.** Trivial. Both observables are direct computations on consecutive trajectory positions; no reference state or perturbation detection required.

**Note on inheritance.** v0's continuous-motion substrate work (substrate findings 3, 4) was designed against an implicit continuity assumption. Treating continuity as a measurable axis rather than a substrate property makes the inheritance explicit: v0's substrate sat at a specific $C$ value; v2 characterises the architecture's behaviour across the $C$ range, including v0's region.

### 4.4 Repetition structure

**Definition.** For a trajectory of length $T$, construct the self-similarity matrix $S_{ij} = \cos(x_i, x_j)$. Identify repeating subsequences as off-diagonal stripes in $S$ above a similarity threshold $\tau_R$. Repetition structure is a pair $(P, F)$:

- **Period $P$.** The dominant repetition period — the off-diagonal offset $k$ at which $\text{mean}_i[S_{i, i+k}]$ is maximised, over the range $k \in [1, T/2]$. $P = \infty$ (or undefined) if no off-diagonal mean exceeds a noise floor.
- **Fidelity $F$.** The mean cosine similarity at the dominant period, $\text{mean}_i[S_{i, i+P}]$. $F$ near 1 indicates near-bit-identical repetition; lower $F$ indicates approximate repetition.

**Variant — repetition coverage.** The fraction of trajectory positions $i$ for which $S_{i, i+P} > \tau_R$ — the fraction of the trajectory that participates in the dominant pattern.

**Measurability on arbitrary (env, enc) pair.** Standard signal-processing operation. The self-similarity matrix is $O(T^2)$ in storage and computation; for long trajectories ($T > 10^5$), block-wise estimation suffices. No construction-side information required.

**SCAFFOLDING — calibrated.** $\tau_R$; the noise floor for declaring "no repetition"; the block size for long-trajectory estimation.

### 4.5 Manifold dimensionality

**Definition.** The effective dimensionality of the trajectory's local neighbourhood structure. Estimated via the participation ratio of local PCA eigenvalues:

$$
D = \mathbb{E}_t\left[\frac{(\sum_k \lambda_k^{(t)})^2}{\sum_k (\lambda_k^{(t)})^2}\right]
$$

where $\lambda_k^{(t)}$ are the eigenvalues of the covariance matrix of trajectory positions in a local window around position $t$. $D = 1$ for a one-dimensional manifold (trajectory in a line); $D = d = 1024$ for full ambient-space coverage; intermediate values for intermediate manifolds.

**Variant — global dimensionality.** Apply the participation ratio to the full trajectory's covariance rather than local windows. Distinguishes "trajectory lives on a low-dim manifold globally" from "trajectory is locally low-dim but globally high-dim" (e.g., a winding curve in high-dim space).

**Measurability on arbitrary (env, enc) pair.** Standard manifold-learning operation. Local PCA at every position is expensive but well-defined; subsampling reduces cost without affecting the population estimate. No construction-side information required.

**SCAFFOLDING — calibrated.** Local-window size; subsampling rate.

### 4.6 Property-definition portability checklist

Each definition above is verified against the methodology-portability test (§4 preamble):

| Property | Definition refers to ... | Construction-side info needed? |
|---|---|---|
| Magnitude | reference vs perturbed subsequences, identifiable via repetition + residual | no |
| Locality | reference state counterpart, estimable from repetition | no, but requires repetition presence |
| Continuity | consecutive trajectory positions | no |
| Repetition | self-similarity matrix structure | no |
| Manifold dim | local covariance eigenvalues | no |

Locality's footnote ("requires repetition presence") is a structural limitation: locality is undefined on trajectories with no repetition. The measurement protocol §6.2 returns "locality undefined" for trajectories below a repetition-coverage threshold; the worked-example trajectory must pass this threshold for the locality axis to be measured at all. v0/v1's continuous-motion substrate satisfies this trivially.

---

## 5. Stream construction primitives

The primitives produce synthetic trajectories with controlled values on the §4 properties. Each primitive's parameters map directly to one or more §4 axes. Construction is deliberately simple — synthetic substrate's advantage is interpretability and control, not realism.

### 5.1 Base manifold trajectory

A parameterised low-dimensional manifold $\mathcal{M}_D \subset \mathbb{R}^{1024}$ of dimensionality $D$ (matching §4.5), embedded via a fixed random linear projection. The trajectory is generated by sampling a smooth path on $\mathcal{M}_D$:

$$
x_t = U \cdot \phi(\theta_t) + \epsilon_t
$$

where $U \in \mathbb{R}^{1024 \times D}$ is the (fixed, random orthogonal) embedding matrix, $\phi: \mathbb{R}^D \to \mathbb{R}^D$ is a smooth parameterisation (e.g., a sinusoidal lift), $\theta_t \in \mathbb{R}^D$ is a low-dim trajectory parameter evolved via a smooth ODE with controllable step size (matching §4.3 $C$), and $\epsilon_t$ is per-position noise (calibrated to substrate noise floor; not a swept axis in v2).

**Parameter-to-axis mapping:** $D$ controls §4.5 manifold dimensionality; ODE step size controls §4.3 continuity; deterministic-periodic ODEs produce §4.4 repetition with controlled $P$ and $F$.

### 5.2 Repetition primitive

A trajectory composed of a base segment $\{y_1, \ldots, y_P\}$ repeated $R$ times with controlled fidelity:

$$
x_{kP + i} = y_i + \nu_{k,i}
$$

for $k \in [0, R)$ and $i \in [1, P]$, where $\nu_{k,i}$ is a per-repetition noise term whose magnitude controls §4.4 fidelity $F$. The base segment $y$ itself is drawn from the §5.1 manifold trajectory primitive at the desired manifold dimensionality and continuity.

**Parameter-to-axis mapping:** $P$ → §4.4 period; $R \cdot P$ → trajectory length; $\nu$ magnitude → §4.4 fidelity; segment $y$'s properties → §4.3 continuity and §4.5 manifold dim within the repeating segment.

### 5.3 Perturbation primitive

Applied on top of a §5.2 trajectory. At designated repetitions $k^* \in K_{\text{pert}}$ and position $i^* \in I_{\text{pert}}$ within each perturbed repetition, the trajectory value is shifted:

$$
x_{kP + i} = y_i + \nu_{k,i} + \delta_{k, i} \cdot \mathbb{1}[k \in K_{\text{pert}} \wedge i \in I_{\text{pert}}]
$$

where $\delta_{k, i}$ is a shift vector controlling §4.1 magnitude. Locality (§4.2) is controlled by the spread of $I_{\text{pert}}$ relative to the period $P$ — a narrow $I_{\text{pert}}$ produces high locality, a wide $I_{\text{pert}}$ produces low locality. Choice of $\delta_{k,i}$ direction (random direction in d=1024 vs targeted) is a SCAFFOLDING parameter; pass-2 default is random isotropic.

**Parameter-to-axis mapping:** $\|\delta\|$ → §4.1 magnitude; $|I_{\text{pert}}| / P$ → inverse of §4.2 locality.

**Direction handling.** $\delta_{k, i}$'s direction is random isotropic in $\mathbb{R}^{1024}$. Per-repetition seed assignment: each of the $n$ training repetitions per sweep point uses a different shift-direction seed; the SET of seeds is fixed across the spec for reproducibility. Per-sweep-point evaluation aggregates over direction-seeds, characterising the architecture's direction-agnostic behaviour by construction.

Rationale: fixed single direction across the sweep would characterise "robustness to this specific direction" rather than "robustness to perturbations" — the map's interpretation would be direction-conditional, and the n=1 direction at a structural level would replicate v0 closing §6's noise problem one level up. Re-sampling per repetition with reproducible seed set gives direction-agnostic characterisation at no compute cost (the per-sweep-point repetitions are already running for the per-arm sample-size discipline).

Direction-specific characterisation — "what does the architecture do under perturbation aligned with a specific stream direction?" — is an interaction-probe option available post-main-effects, not a main-effects default.

**Contingency for reduced n.** Direction-seeds scale linearly with n (n=10 → 10 seeds; n=5 → 5 seeds), so direction-agnostic characterisation degrades proportionally rather than via phase transition. Below n=5, the methodology's direction-agnostic claim no longer holds at full strength. Design chat at that point chooses among three named options: (i) accept weakened claim and document in closing; (ii) hold n=5 and reduce sweep density further to preserve direction-agnostic claim; (iii) hold direction fixed and reallocate sample-size budget to a separate direction-probe. Naming the three options at design time prevents in-flight mode confusion when the contingency invokes.

### 5.4 Inter-item coupling primitive (interaction-probe only)

Per the design-chat decision to demote coupling to interaction probes, this primitive is not part of the main-effects sweep. Specified here for the probe-design section of §9.

Two §5.2 trajectories with base segments $y^{(A)}$ and $y^{(B)}$ are interleaved across repetitions. Coupling is parameterised by $\rho \in [0, 1]$: $\rho = 0$ produces independent base segments; $\rho = 1$ produces $y^{(B)} = y^{(A)}$ (perfectly coupled). Intermediate $\rho$ produces base segments that share a controllable fraction of their dimension structure.

**Parameter-to-axis mapping:** $\rho$ → inter-item coupling strength (not a §4 main-effects axis; characterised via probe).

### 5.5 SCAFFOLDING items

Per `research_operations_v1.md` §7.1–7.3, every construction-side fixed parameter is labelled. Calibrated against pre-sweep verification at §9.2; no construction parameter is locked at spec-time without explicit calibration plan.

**SCAFFOLDING — calibrated at §9.2 pre-sweep:**
- The random orthogonal embedding $U$ (fixed across all v2 runs for reproducibility; not a swept dimension).
- Per-position noise magnitude $\|\epsilon_t\|$.
- The smooth parameterisation $\phi$.
- The base segment $y$'s sampling protocol for repetition primitives.
- The shift vector $\delta$ direction choice (random isotropic default).

**ARCHITECTURE — v2 commitments:**
- Option β substrate (no encoder in loop). Inherited from §2.1.
- d=1024 embedding dimension. Inherited from v0/v1 scaffold.
- The §5.1–5.3 primitive structure. The primitives are the v2 substrate; they are not swept and not ablated within v2.

---

## 6. Measurement protocol (Output 2)

The protocol applies the §4 definitions to an arbitrary (env, enc) pair's empirical trajectory collection, producing distributions on each §4 axis. The protocol is the methodology's second output; the worked example's application to DINOv2-on-AI2-THOR is its first instantiation.

### 6.1 Protocol inputs

- An environment $E$ with a trajectory-producing policy $\pi$.
- An encoder $F: \text{Env-observation} \to \mathbb{R}^{1024}$.
- A finite trajectory collection $\mathcal{T} = \{\tau_1, \tau_2, \ldots\}$, each $\tau_k = \{F(o_1^{(k)}), F(o_2^{(k)}), \ldots\} \subset \mathbb{R}^{1024}$, produced by running $\pi$ in $E$ and encoding observations via $F$.

For DINOv2-on-AI2-THOR: $E$ is the v0/v1 substrate (ProcTHOR seed-7 house, v1 build with `_assign_close_up_ordinals` fix); $\pi$ is v0/v1's continuous-motion explorer; $F$ is DINOv2 (frozen, ViT, CLS-token output); $\mathcal{T}$ is the v0/v1 trajectory collection at the v1-finalised settings. v1 finding 9 (ProcTHOR-renderer-binding) does not affect this step — no mutation API is exercised; DINOv2 forward-passes pre-rendered frames.

### 6.2 Protocol procedure

For each trajectory $\tau_k$ in $\mathcal{T}$:

1. **Self-similarity matrix.** Compute $S_{ij} = \cos(\tau_{k,i}, \tau_{k,j})$ for the trajectory. Block-wise estimation for $|\tau_k| > 10^5$.
2. **Repetition detection.** Identify the dominant period $P_k$ via the off-diagonal mean-similarity maximum; compute fidelity $F_k$ and coverage. Record §4.4 values.
3. **Reference state estimation.** For each position $i$, define $x_i^{\text{ref}}$ as the median over repetitions $\{\tau_{k, i \bmod P_k}, \tau_{k, i \bmod P_k + P_k}, \ldots\}$. If repetition coverage is below threshold, mark $x_i^{\text{ref}}$ undefined and proceed to step 6 with the locality axis marked "undefined."
4. **Perturbation detection.** Identify perturbed positions as $\{i : \|\tau_{k,i} - x_i^{\text{ref}}\| > \tau_{\text{pert}}\}$. Record this as $S_{\text{pert}}$ for §4.1 and §4.2 computation.
5. **Magnitude and locality.** Compute §4.1 $M_k$ and §4.2 $L_k$ from the perturbed and reference subsequences.
6. **Continuity.** Compute §4.3 $C_k$ and $C_{\text{curv}, k}$.
7. **Manifold dimensionality.** Compute §4.5 $D_k$ (local) and $D_{\text{global}, k}$ via local PCA at sampled positions and global PCA on the full trajectory.

Aggregate across $\mathcal{T}$: per axis, produce the empirical distribution of per-trajectory values. The protocol's output is a five-dimensional region in property space, characterised as a distribution rather than a point.

### 6.3 Protocol-to-map mapping

The map (§§3, 8) is built on five property axes at discrete grid points (5 values per axis, per the design-chat commitment in pass 1 §3.5). The protocol's output is a distribution over each axis; mapping to grid points requires summarisation.

**Multi-modal detection before summarisation.** Multi-modal distributions are plausible on some axes — e.g., perturbation magnitude on trajectories containing both weak and strong disturbances; manifold dimensionality on trajectories crossing structurally distinct regions. Median + IQR summarisation collapses these into a single point with spread that doesn't capture the bimodality. Per axis, before summarisation: fit a two-component Gaussian mixture model to the per-trajectory values. If the BIC improvement of the two-component fit over a single-component fit exceeds a calibrated threshold (SCAFFOLDING; calibrated at V2-PRE-E against synthetic single-mode and multi-mode references), the axis is flagged "multi-modal" and the two component medians are reported separately. The worked-example outcome reading (§8.3) accommodates multi-modal axes: the pair's region is characterised by both component medians, and the worked-example outcome may be different at each.

**Default summarisation.** For unimodal axes: median + IQR of the empirical distribution. The median is the grid-point lookup; the IQR characterises the (env, enc) pair's spread on that axis. A pair whose IQR is small relative to grid spacing maps cleanly to one grid point; a pair whose IQR spans multiple grid points has its lookup distributed across them, and the worked-example outcome reflects the architecture's behaviour at each.

**Worked-example-to-grid mapping refinement.** Pass 1 §3.5 noted two options: (i) nearest-grid-point lookup, (ii) off-grid stream construction matching the measured distributions exactly. Pass 2 commits to (i) — nearest grid point per axis — with the interpolation distance per axis reported as part of the worked-example outcome. Rationale: (i) uses points the main-effects sweep already covers; (ii) requires off-grid stream construction at substantial added design cost for marginal gain. If the IQR or multi-modal analysis surfaces a pair whose distribution falls between grid points with non-trivial interpolation distance, the closing document records this as a methodology learning rather than as a worked-example failure.

### 6.4 SCAFFOLDING items

- $\tau_{\text{pert}}$ threshold in step 4.
- Repetition-coverage threshold in step 3 for declaring locality undefined.
- Local-PCA window size in step 7.
- BIC-improvement threshold for declaring an axis multi-modal in §6.3.
- Per-axis summarisation choice (median + IQR default for unimodal axes; two-component medians for multi-modal axes; revisit if worked-example measurement surfaces tri-modal or higher).

All calibrated at §9.2 pre-sweep verification.

---

## 7. Measurement axes for the architecture's behaviour

Per the v1 spec §10.3 inheritance (per-stream-point granularity, co-primary mean and variance differentiation), v2 evaluates the architecture at each sweep point on the following metrics. Each metric is computed per-stream-point (the v2 analogue of v1's per-(item, ordinal)).

### 7.1 Per-stream-point mean differentiation

For each stream point $i$ in the held-out evaluation stream, compute the predicted mean $\hat{\mu}_i$ from the trained predictor at end-of-training. Differentiation is the variance of $\{\hat{\mu}_i\}$ across stream points, normalised by the variance of the ground-truth targets $\{x_{i+1, \ldots, i+K}\}$:

$$
\text{Diff}_\mu = \frac{\text{Var}_i[\hat{\mu}_i]}{\text{Var}_i[x_{i+1:i+K}]}
$$

$\text{Diff}_\mu$ near 1 means the predictor's mean tracks the target's per-stream-point variation; $\text{Diff}_\mu$ near 0 means the predictor produces stream-point-invariant means.

### 7.2 Per-stream-point variance differentiation

For each stream point $i$, compute the predicted scalar log-variance $\hat{\sigma}^2_{i, k}$ (or per-K-position log-variance under Primary / Ablation 1 architectures) at end-of-training. Differentiation is the variance of $\{\hat{\sigma}^2_{i, k}\}$ across stream points at each K-position $k$, averaged over $k$:

$$
\text{Diff}_\sigma = \mathbb{E}_k\left[\text{Var}_i[\hat{\sigma}^2_{i, k}]\right]
$$

$\text{Diff}_\sigma > 0$ means the predictor's variance tracks per-stream-point variation; $\text{Diff}_\sigma$ near 0 means variance is stream-point-invariant (the v0 coupling pathology's signature).

### 7.3 Co-primary working-region indicator

A sweep point is in the architecture's **working region** if both $\text{Diff}_\mu$ and $\text{Diff}_\sigma$ exceed the bit-identical-baseline threshold (control C3 at the same sweep point). The "co-primary" framing inherits from v1 spec §1.4: both must hold for the architecture to be characterised as working at that point.

The bit-identical baseline serves as the per-sweep-point noise floor: $\text{Diff}_\mu > \text{Diff}_\mu^{(C3)} + \tau_W$ and $\text{Diff}_\sigma > \text{Diff}_\sigma^{(C3)} + \tau_W$, with $\tau_W$ a statistical-distinguishability threshold calibrated against per-sweep-point sample variance (SCAFFOLDING; calibrated against pre-sweep arm-run distributions at §9.2).

**Anchoring to v1's co-primary discipline.** $\tau_W$ is chosen such that both $\text{Diff}_\mu$ and $\text{Diff}_\sigma$ must exceed their respective bit-identical baselines by an amount considered meaningful under v1 spec §1.4's co-primary discipline — not a marginal one-head-succeeds-while-the-other-is-just-above-noise outcome. The calibration target at V2-PRE-E commits to a threshold that excludes "near-baseline on one head while above-baseline on the other" patterns from the working region. Per the discipline at §9.2, the response to per-sweep-point variance large enough to obscure signal is to increase n, not to lower $\tau_W$.

**Operationalisation.** $\tau_W$ per head is calibrated such that the per-head Wilcoxon signed-rank test (v0 inherited default) yields $p < 0.05$ against the bit-identical baseline distribution. Joint working-region determination requires both heads pass at $p < 0.05$; joint $\alpha \approx 0.0025$ if heads were statistically independent (they aren't on this architecture, but the empirical dependence is characterisable from cross-arm contrasts and reported in closing). V2-PRE-E calibrates $\tau_W$ per head against the empirical baseline distribution from PRE-D's stability runs until the per-head significance condition is met. This commitment replaces the earlier SCAFFOLDING framing for $\tau_W$ at spec-time — calibration is operationalisable now; PRE-E executes the operationalisation against empirical baseline distributions.

### 7.4 Cross-arm contrast metrics

Per the three-arm structure inherited from v1 scaffold:

- **Primary − Ablation 1.** Isolates the variance representation change (per-K-position vs scalar). At a working sweep point, this contrast characterises whether variance representation is load-bearing.
- **Primary − Ablation 2.** Isolates the full v1 architectural fix vs v0 architecture. Characterises whether the fix has any effect at this sweep point.
- **Ablation 1 − Ablation 2.** Isolates the readout topology change. At a working sweep point, this contrast characterises whether readout topology is the load-bearing intervention.

Cross-arm contrasts are reported per-sweep-point as part of the map. Aggregations across sweep regions (e.g., "readout topology is load-bearing across the high-magnitude region") are part of the §8 characterisation-outcome reading, not v2's per-sweep-point output.

### 7.5 Training-time stability indicators

Per `research_operations_v1.md` §3.4 regression-canary discipline:

- **Loss trajectory.** Monotonic non-increasing (with stochastic-batch tolerance); no divergence; no NaN/Inf in-flight.
- **Per-stream-point variance trajectory.** No collapse to single value (variance-head death); no explosion.
- **Body-representation cosine across checkpoints.** v0's BCDD finding of pooled-readout drift on bit-identical input — applied to v2's K-positional readout, the analogue is per-K-position drift on bit-identical streams. Reported per arm per sweep point as a v0-coupling reproduction indicator.

These are not part of the map's working-region determination but are reported alongside it. Stability failures at a sweep point change the interpretation of that point's working-region status.

### 7.6 SCAFFOLDING items

- $\tau_W$ — co-primary working-region threshold.
- Statistical-distinguishability test choice (Wilcoxon vs t-test vs bootstrap CI; inheriting v0's Wilcoxon as tentative default).
- Held-out evaluation stream length per sweep point.
- Number of training repetitions per sweep point (per `research_operations_v1.md` §3.6, n<20 in high-variance domains is unreliable; pass 2 §9.3 sizes against this).

---

## 8. Characterisation-outcome framework

v2's outcomes are not pass/fail verdicts. The map's "result" is the surface itself — the set of working-region sweep points, the failure modes at non-working points, the cross-arm contrasts across the surface. §8 specifies how that surface is read and what readings are pre-committed as substantive.

### 8.1 Map outcomes (Output 1)

The main-effects sweep at L_d_main (five crosses × five axes × five values × three arms per pass 1 §3.5 commitment, plus controls) produces a per-sweep-point evaluation. The map's outcome is characterised by:

1. **Working-region structure.** The set of sweep points where the co-primary indicator (§7.3) is satisfied. Per pass 1 §1.4-style discipline, five structural categories are pre-committed:

   - **M1 — Connected working region with identifiable boundaries.** Working points form a contiguous region in property space; boundaries are characterisable. Methodology output: any (env, enc) pair whose measured region falls inside the working region is supported; pairs outside are not. v3+ inheritance: the boundaries become encoder/environment design targets.
   - **M2 — Fragmented working region.** Working points form disconnected islands. Methodology output: working-region lookup requires fine-grained measurement; encoder/environment design targeting depends on which island a pair lands in. v3+ inheritance: the fragmentation itself is a methodology finding — the architecture's working conditions are not interval-shaped on these axes.
   - **M3 — No working region in sweep range.** No sweep points satisfy the co-primary indicator. Methodology output: either the architecture does not work on the property axes characterised (which would falsify the v1 architectural fix as a general intervention), or the sweep range under-spans relevant property regions. Distinguishable by inspecting the worked-example outcome (§8.3) and by inspecting whether *Ablation 2* — the v0 architecture — produces working points anywhere.
   - **M4 — Saturated working region (architecture works everywhere swept).** All sweep points satisfy the co-primary indicator. Methodology output: the architecture is robust across the characterised property region; sweep ranges may under-cover the regions where it fails. v3+ inheritance: extending sweep ranges (revisiting §2.4 first-principles bounds) is the v3 starting point.
   - **M5 — Structured-but-non-M1-M2-M3-M4.** The working-region pattern has structure that doesn't fit M1–M4. The catch-all preserves coverage without enumerating exhaustively; closing characterises any M5 instance in prose. Worked examples of what M5 captures:
     - **L_d × stream-property interaction structure.** Under the two-L_d_main main-effects commitment (L_d ∈ {1, 4}), working-region status differs between L_d=1 and L_d=4 along some axis — capacity interacts with the working-region boundary in a structured way detectable from main effects. The L_d sweep at the worked-example point (L_d ∈ {1, 2, 3, 4}) refines the characterisation at one stream-property location; main-effects two-L_d_main disagreement signals whether the interaction is local to the worked-example region or distributed.
     - **Directional working regions.** The working region is anisotropic in the swept property space — sweeping axis k away from the worked-example point preserves working-region status, while sweeping axis k' away from the same point exits the working region.
     - Other structured patterns the main-effects sweep may surface; the M5 reading is "the working region has structure worth characterising in prose, not slotting into M1–M4."

     **M5 declaration discipline.** Any M5 instance must declare in prose: (a) what structural pattern is present in the working-region behaviour; (b) why the pattern doesn't fit M1–M4; (c) whether the structure is consequential for methodology outputs (working-region lookup is changed by the structure) or methodology-internal observation (interesting characterisation, doesn't change the methodology's outputs). If (c) is consequential, the named sub-pattern becomes part of v3+ inheritance (working-region grammar); if internal, it is recorded in closing as methodology observation only. This is interpretive discipline, not sub-classification — preventing M5 from becoming a holding pen for "interesting things we noticed."

2. **Failure-mode classification at non-working sweep points.** Per `research_operations_v1.md` §1.3 (honest negative results contain the prescription for the fix), each non-working sweep point's failure mode is classified:

   - **F-mean.** $\text{Diff}_\mu$ fails but $\text{Diff}_\sigma$ satisfies. Variance learns; mean does not.
   - **F-var.** $\text{Diff}_\sigma$ fails but $\text{Diff}_\mu$ satisfies. Mean learns; variance does not (v0-style coupling pathology).
   - **F-both.** Both fail. Architecture produces no per-stream-point differentiation.
   - **F-stability.** Training instability at this sweep point (NaN/Inf, divergence). The co-primary indicator is undefined; failure is upstream of evaluation.

3. **Cross-arm contrast patterns across the surface.** §7.4 contrasts aggregated regionally:
   - Where is Primary load-bearing vs Ablation 1? (Variance representation matters in region X.)
   - Where is Primary load-bearing vs Ablation 2? (Full fix matters in region Y.)
   - Where is Ablation 1 load-bearing vs Ablation 2? (Readout topology matters in region Z.)

   The "load-bearing" determination at a sweep region is the cross-arm contrast exceeding a region-aggregated significance threshold (SCAFFOLDING; calibrated against per-sweep-point contrast distributions).

### 8.2 Capacity-coupling outcomes (L_d sweep at worked-example point + two-L_d_main main effects)

Capacity-coupling characterisation has two complementary sources under v2's design:

**Main-effects two-L_d_main disagreement.** Phase 1 runs main effects at L_d_main ∈ {1, 4}. Disagreement between L_d=1 and L_d=4 working-region patterns along any axis is a first-class L_d × stream-property interaction signal, surfaced across the full stream-property sweep range. Per the §3.3 framing, this makes the §8.1 M5 "L_d × stream-property interaction structure" detectable from main effects rather than inferred only from the worked-example L_d sweep.

**L_d sweep at the worked-example point (Phase 2).** Refines the two-L_d_main main-effects characterisation at one stream-property location with the full L_d ∈ {1, 2, 3, 4} range. Outcomes per the L_d sweep at the worked-example point:

- **L-flat.** Working-region status is L_d-independent at the worked-example point. Capacity is not the constraint at this point. Consistent main-effects reading: L_d=1 and L_d=4 working-region patterns agree.
- **L-monotonic.** Working-region status improves with L_d (or degrades — both possible). Capacity is the constraint; v3+ inheritance is "scale L_d for this region." Consistent main-effects reading: L_d=1 and L_d=4 patterns differ in a monotonic direction.
- **L-non-monotonic.** Working-region status varies non-monotonically with L_d (e.g., works at L_d=2 and L_d=4 but not L_d=3). Methodology finding: capacity interacts with optimisation in a structured way; further characterisation deferred. Main-effects reading is constrained — two-L_d_main at envelope endpoints can't distinguish non-monotonic from monotonic patterns at intermediate L_d.

**Cross-reading.** If two-L_d_main main effects disagree at the worked-example point but the L_d sweep produces L-flat at the worked-example point, the disagreement signal localises elsewhere in the stream-property space; Phase 3 probes target where it localises. If main effects agree but L_d sweep shows L-monotonic, capacity-coupling is local to the worked-example point and the main-effects map may under-characterise capacity-coupling at other stream-property regions — closing flags this as a methodology scope item for v3+.

Phase 3 capacity-coupling probes (§9.4) extend the L_d sweep to stream-property regions where main-effects two-L_d_main disagreement is largest, or where cross-reading suggests capacity-coupling localisation Phase 1 / Phase 2 didn't fully characterise.

### 8.3 Worked-example outcomes (Output 2 applied to DINOv2-on-AI2-THOR)

Per pass 1 §1.4 commitment, four outcomes are pre-committed:

- **W1 — Lands in working region.** DINOv2-on-AI2-THOR's measured region overlaps a working region in the map. Implication: v0/v1's troubles are attributable to architecture-and-experimental-design (the BCDD readout topology pathology; v1's substrate-prerequisite issue), not encoder-environment misfit. The v1 architectural fix is supported as the right intervention; v3 inherits the methodology and the fix together.
- **W2 — Lands in non-working region.** Implication: v0/v1's troubles include encoder-environment misfit. v3 has to engineer around the fit. The "DINOv2 might be the wrong encoder for AI2-THOR entirely" hypothesis is empirically grounded.
- **W3 — Lands on a boundary or in ambiguity.** Implication: v0/v1's results are partly explained by being near a working-region edge. v3 inherits the methodology and the sharpened question — what moves the system into the interior of a working region.
- **W4 — Falls outside the map's coverage.** Implication: the sweep ranges under-spanned the (env, enc) pair's actual location. The first-principles ranges in §2.4 were insufficient. Methodology learning is recorded; v3+ inherits the sweep-design refinement.

All four outcomes are substantive per the pass 1 §9.3 discipline. The worked example cannot fail.

### 8.4 The map × worked-example crosswalk

The map outcome (M1–M5) and the worked-example outcome (W1–W4) interact. Some crosswalks have specific methodology readings:

- **M3 + W2.** Architecture has no working region in the sweep range; worked example lands in non-working region. Reading: the architecture as fixed by v1 may not work in any practically-relevant region. Both outputs corroborate. Strongest single signal v2 can produce for "v1 fix was insufficient."
- **M4 + W1.** Architecture works everywhere swept; worked example lands in working region. Reading: either sweep ranges too narrow (likely, per §2.4 first-principles upper bounds), or architecture is genuinely robust across all plausible regions. Distinguishable by whether the controls (§3.4) and Ablation 2 also work everywhere.
- **M1 + W2.** Architecture has a working region; worked example is outside it. Reading: the v1 fix works in principle but not for the v0/v1-lineage pair; v3+ encoder-environment work moves the pair into the working region.
- **M2 + W3.** Fragmented map; worked example on a boundary. Reading: the worked example sits at a structurally interesting location; understanding its boundary may shed light on fragmentation. Worth interaction-probe budget if available.

Other crosswalks have more diffuse readings; pass 2 does not pre-commit interpretations for the full 4×4 matrix. The pre-committed cases are the ones most likely to drive v3 design decisions.

### 8.5 What v2 does not produce

Explicit non-deliverables, recorded to prevent post-hoc scope creep:

- **No verdict on the v1 architectural claim.** Whatever the worked example produces is empirical input to v3 design, not a closing-of-the-loop on v1's hypothesis. The hypothesis remains in the "characterise rather than test" frame (v1 closing §10).
- **No recommendation on v3 encoder choice.** Encoder choice depends on the (env, enc) pair targeting a specific working region; that's a v3 design question against v2's map.
- **No recommendation on v3 substrate choice.** Same reasoning.
- **No portability claim across environments.** v2 holds environment constant explicitly. The portability claim is what v3+ tests.
- **No claim about (env, enc) pairs lacking repetition structure.** Per §4.2's structural limitation, v2's methodology applies only to (env, enc) pairs producing trajectories with repetition structure. v2 closing records this as methodology scope; v3+ inherits the restriction as a constraint on environment-policy pairs the methodology applies to, or extends the protocol to handle single-pass exploration as a v3+ deliverable.

---

## 9. Compute envelope and execution plan

### 9.1 Total arm-run envelope

Compiled from pass 1 §§3.3–3.5 (three corner-avoiding crosses, two L_d_main values at envelope endpoints {1, 4}) and the n=10 per-sweep-point sample-size commitment in §9.3:

| component | sample-size n | arm-runs |
|---|---|---|
| Main effects (3 crosses × 2 L_d_main × 5 axes × 5 values × 3 arms) | 10 | 4500 |
| L_d sweep at worked-example point (4 L_d × 3 arms) | 10 | 120 |
| Negative controls (C1, C2; 2 L_d_main values × 3 arms × 2 controls) | 10 | 120 |
| Negative control C3 (bit-identical baseline) | n/a | absorbed into main effects |
| Capacity-coupling probes (4 probes × 4 L_d × 3 arms; see §9.4 for slack rule) | 10 | 480 |
| **Subtotal — sweep and probes** | | **5220** |
| Pre-sweep verification runs (per §9.2; includes ~200 for V2-PRE-D2 n-validation) | varies | ~240 |
| Worked-example measurement protocol (forward-pass DINOv2 over v0/v1 trajectory collection; not an arm-run; one-off) | n/a | n/a |
| **Total — arm-runs** | | **~5460** |

Per the design-chat compute frame ("throw compute at this, characterise rather than conserve"; "compute won't be a problem"), this envelope is taken as the v2 baseline rather than as a stretch goal. Per-arm-run cost is determined by per-sweep-point training time, which depends on stream length and other §7.6 SCAFFOLDING parameters. Pass 2 commits the arm-run envelope; per-run cost is empirical at §9.2 and the total wall-clock is computed once per-run cost is in hand.

**Compute envelope honesty.** At typical per-arm-run costs (~5–15 minutes per arm-run on local compute), the n=10 envelope of ~5460 arm-runs translates to ~19–57 days wall-clock serial — substantially beyond the "day or two" frame the design chat anchored to informally during pass 1. This is the actual envelope at typical per-run cost, not a hoped-for-favourable estimate. Surfaced in spec rather than waiting for V2-PRE-D to discover. The post-PRE checkpoint (§9.5) provides the design-chat venue to commit to wall-clock vs density trade-off with empirical evidence; v2 does not assume per-run cost will resolve the gap.

**Mandatory post-PRE checkpoint.** Per-arm-run cost is unknown until V2-PRE-D measures it. Between Phase 0 (pre-sweep verification) and Phase 1 (main effects), a **mandatory design-chat checkpoint** runs: design chat receives PRE-D's measured per-arm-run cost, extrapolates the full envelope, and commits to wall-clock vs density trade-off with empirical evidence. This is a scheduled checkpoint with empirical input, not a contingency that may invoke. The checkpoint can extend density (e.g., three crosses → four or five crosses if wall-clock allows) or reduce density (fewer crosses, fewer per-axis values, or reduced axis count back through §3.1's decision-anchoring), per the §9.8 reliability-over-coverage principle. The reduce-density-not-n discipline applies if density reduction is the decision.

**Hard density floor.** Density reduction stops at a hard floor of three crosses and three values per axis; if that floor would still exceed the wall-clock ceiling, the Phase 0.5 design chat re-scopes the number of axes (back through §3.1's decision-anchoring) or defers the sweep to v3+ rather than produce an under-powered map. This floor protects the M1-vs-M2 distinction and the working-region boundary characterisations the methodology depends on; below the floor, the methodology is no longer producing characterisation outcomes the §8.1 outcome categories were designed against.

### 9.2 Pre-sweep verification (PRE-style)

Per v1's preflight discipline (PRE-A through PRE-D), v2 runs pre-sweep verification before main effects commits to the full envelope. Modules:

- **V2-PRE-A. Construction-primitive sanity.** Verify that each §5 primitive produces streams whose measured §4 properties match the intended values across the sweep range. Run the §6 protocol on synthetic streams; verify median property values match the construction parameters within tolerance.
- **V2-PRE-B. Worked-example measurement.** Apply the §6 protocol to DINOv2-on-AI2-THOR's v0/v1 trajectory collection. Output: the (env, enc) pair's measured region in property space. Identifies the worked-example sweep point per §3.3's L_d sweep, and surfaces W4 (outside-coverage) before main effects commits.
- **V2-PRE-C. Architectural property assertions on v2 substrate.** Re-verify v1 PRE-D's 11 architectural assertions on streams from the §5 construction primitives. The PRE-D assertions are architecture-level (per-K-position parameterisation, cross-attention shapes, etc.) and inherit unchanged; V2-PRE-C confirms they hold under v2 substrate.
- **V2-PRE-D. Training-stability calibration at sweep boundaries + empirical n-validation + corner reachability.** Three sub-modules:
  - **D1a — Endpoint stability at sweep boundaries.** Each stream-property axis at its minimum and maximum value, with other axes held at midpoint, at both L_d_main values (L_d=1 and L_d=4). Five axes × two endpoints × two L_d_main × one arm-run = 20 arm-runs. Stability failures (NaN/Inf, divergence) identify sweep-range adjustments before main effects commits. Note: arm selection at D1a uses Primary only — D1a's deliverable is stability of the training procedure at sweep boundaries, not cross-arm comparison; arm-comparison stability is checked in PRE-C.
  - **D1b — Per-arm-run cost measurement.** Wall-clock per arm-run is recorded from D1a's 20 runs; no additional arm-runs needed. Output feeds Phase 0.5's wall-clock vs density trade-off decision.
  - **D1c — Corner reachability characterisation (analytical, no arm-runs).** Inspect V2-PRE-B's empirical worked-example measurement plus the §4 property-definition first-principles bounds. Output: assessment of whether 5D corner regions of property space (held axes at extreme values simultaneously) are physically plausible for any (env, enc) pair. If PRE-B's empirical measurement sits centrally with reasonable spread, corners are not reachable and §3.5 corner-avoidance commitment holds; if PRE-B's measurement falls near corner regions or extreme values on multiple axes simultaneously, corners may be reachable and Phase 0.5 considers corner-sampled crosses as Phase 3 probes. Zero arm-runs; analytical only.
  - **D2 — Empirical n-validation per §9.3 override discipline.** On ~10 sweep-point pairs (e.g., midpoint of each axis at L_d_main = 2), run n=20 per arm and compute the empirical confidence interval at n=10 vs n=20. Output: whether the n=10 commitment is sufficient to discriminate working-region status at the §7.3 τ_W threshold. Feeds the Phase 0.5 design-chat checkpoint with empirical evidence on the §3.6 override.
- **V2-PRE-E. SCAFFOLDING calibration.** Calibrate §4 SCAFFOLDING items ($\tau_L$, $\tau_R$, $\tau_W$, repetition-coverage threshold, etc.) against the V2-PRE-A / V2-PRE-B distributions.

Each module produces a stop-or-proceed signal per v1's preflight discipline. STOP on any unconditional-failure trigger; report and resolve in the design chat before main effects commits.

### 9.3 Per-sweep-point sample size

Per `research_operations_v1.md` §3.6, n<20 in high-variance domains is unreliable. v2's "high-variance domain" status at each sweep point is empirical and depends on training-time stochasticity per arm.

Pass 2 commitment: **n=10 training repetitions per arm per sweep point**, with the per-sweep-point evaluation aggregated across repetitions. The commitment is anchored to the design-chat compute frame ("throw compute at this, characterise rather than conserve") and to the statistical-distinguishability requirement of M1-vs-M2 distinctions in the §8.1 framework. n=3 produces per-sweep-point determinations whose confidence intervals span roughly half the point estimate, making working-region boundary determinations statistically fragile regardless of how many sweep points the map covers; n=10 brings the CI to a working-region-determinative range.

**Override discipline against `research_operations_v1.md` §3.6.** `research_operations` §3.6 states n<20 in high-variance domains is unreliable. v2's commitment to n=10 is below that threshold. The override is explicit: v2 commits to n=10 rather than n=20 because the doubling cost (sweep-and-probes subtotal ~5220 → ~10440 arm-runs at n=20; ~36–109 days wall-clock at typical per-run cost) is structurally infeasible at the spec's compute frame, and because v2's per-sweep-point determinations are aggregated across direction-seeds (§5.3) and cross-arm comparisons (n=30 effective at the arm-contrast level), partially offsetting the per-arm sample-size shortfall.

**Empirical override validation at V2-PRE-D.** Per the override discipline, V2-PRE-D produces an empirical CI estimate at n=10 vs n=20 on a small subset of sweep points (e.g., midpoint of each axis at L_d_main = 2; ~10 sweep-point pairs × n=20 = 200 arm-runs added to the PRE budget). If the n=10 CI is sufficient to discriminate working-region status at the §7.3 τ_W threshold, the n=10 commitment holds for Phase 1. If not, the Phase 0.5 design chat commits to either (a) raising n to 20 with corresponding density reduction per §9.1's hard floor, or (b) accepting weakened per-sweep-point determinations with corresponding methodology-scope narrowing in closing. This is the operationalisation of the `research_operations` change-rule preamble for principle deviations: change stated explicitly with rationale, empirical validation, and accepted alternatives.

Cross-arm comparisons at the same sweep point use 30 runs (10 per arm × 3 arms), which is well above v0's Wilcoxon discriminability thresholds and supports the §7.4 cross-arm contrast pattern detection at the regional aggregation level.

**Discipline commitment.** If V2-PRE-D / V2-PRE-E surfaces per-sweep-point variance large enough to obscure signal at n=10, the response is to increase n at the affected sweep points, not to lower $\tau_W$. Lowering $\tau_W$ admits noise into the working-region determination; increasing n reduces the noise being admitted. This commitment inherits v0 closing §5 SCAFFOLDING-threshold discipline at the v2 sample-size axis.

**Direction-seed allocation under n=10.** Per §5.3, each of the 10 repetitions per sweep point uses a different shift-direction seed; the seed set is reproducible. Direction-agnostic characterisation is achieved by construction; no additional repetitions needed for direction-averaging.

### 9.4 Capacity-coupling probe sizing

Probes are post-main-effects. The probe budget commits to an upper bound rather than being sized post-hoc: **4 probe locations baseline**, each running L_d ∈ {1, 2, 3, 4} × 3 arms × n=10 = 120 arm-runs per probe location, totalling 480 arm-runs baseline.

Probe location selection rule:

1. Regions where the working-region indicator transitions sharply across one axis (boundary characterisation).
2. Regions where cross-arm contrast (§7.4) shows structured behaviour (load-bearing arm differs from main-effects expectation).
3. Regions where parallel-cross disagreement is largest (interaction signal from main effects).
4. **Regions of stream-property space where the Phase 2 worked-example L_d sweep produces L-non-monotonic outcome (§8.2).** Two-L_d_main main effects at {1, 4} cannot distinguish monotonic from non-monotonic L_d patterns at intermediate L_d values; if Phase 2 shows L-non-monotonic at the worked-example point, a Phase 3 probe at the worked-example region (or at the nearest property-space region exhibiting cross-arm or sweep-axis structure) runs L_d ∈ {1, 2, 3, 4} to characterise the non-monotonic pattern away from the worked-example point. Addresses the endpoint-only L_d coverage blind spot identified in the §8.2 cross-reading.

**Slack reallocation rule.** If fewer than 4 probe-worthy locations emerge from main effects, the remaining probe arm-runs reallocate to **n=20 at the locations that exist**, not back to general pool. This addresses v0 closing §11's pattern (post-hoc-sized budgets weaken commitment). At n=20, the probe-location characterisation reaches v0's per-(item, ordinal) sample-size threshold, providing the strongest possible characterisation at the most decision-relevant locations the main effects surface.

If more than 4 probe-worthy locations emerge, prioritise by which probe most directly informs v3 design decisions per the §1 decision-to-axes table; record un-probed locations in closing as v3+ inheritance.

### 9.5 Execution phases and STOP conditions

Per `research_operations_v1.md` §8.12, long specs are broken into phases with explicit stop points. v2's phases:

- **Phase 0 — Pre-sweep verification.** V2-PRE-A through V2-PRE-E. STOP at each module's unconditional-failure trigger. Resolve in the design chat. Approximate budget: ~240 arm-runs (PRE-D2 n-validation included) + measurement-protocol runs.
- **Phase 0.5 — Post-PRE design-chat checkpoint (mandatory).** Between Phase 0 and Phase 1. Design chat receives three empirical inputs from Phase 0: (i) PRE-D1's measured per-arm-run cost; (ii) PRE-D2's n=10 vs n=20 CI estimate against the §7.3 τ_W threshold; (iii) PRE-D1 reachability characterisation of property-space corners. Design chat extrapolates the full Phase 1 envelope, commits to wall-clock vs density trade-off with empirical evidence, and resolves the n=10/n=20 override per §9.3. Output: density commitment (three crosses retained, extended to four/five, or reduced; per-axis values retained at 5 or reduced; axis count retained at 5 or reduced per §3.1 decision-anchoring); sample-size commitment (n=10 retained, or n=20 with corresponding density reduction per §9.1 hard floor); corner-sampling commitment (corner-avoidance retained, or corner-sampled crosses added as Phase 3 probes per PRE-D1 reachability). Two L_d_main commitment at {1, 4} is structurally protected (the §8.1 M5 detection argument applies regardless of stream-density choices); the checkpoint reduces stream-property density before reducing L_d_main count. STOP for this checkpoint is built into the phase boundary — Phase 1 cannot proceed without checkpoint output.
- **Phase 1 — Main effects.** Three corner-avoiding crosses (or revised per Phase 0.5) × two L_d_main values × five axes × five values × three arms × n=10; controls absorbed. 4500 + 120 = 4620 arm-runs at baseline density. STOP at completion for results review; do not proceed to Phase 2 without design-chat sign-off on main-effects map.
- **Phase 2 — L_d sweep at worked-example point.** 120 arm-runs at n=10 (L_d ∈ {1, 2, 3, 4} × 3 arms). STOP at completion for cross-reading against Phase 1 main effects.
- **Phase 3 — Capacity-coupling probes.** 4 probes baseline per §9.4; 480 arm-runs (or reduced count at n=20 under slack rule, or extended with corner-sampled crosses per Phase 0.5 reachability finding). STOP at completion for closing-document drafting.
- **Phase 4 — Closing.** Map outcome read (M1–M5); worked-example outcome read (W1–W4); crosswalk read (§8.4); methodology refinements articulated for v3+.

Phase 1 is the largest. The Phase 0.5 checkpoint handles wall-clock decisions with empirical input rather than spec-time guess; the sweep-range adjustment for stability issues at sweep boundaries (V2-PRE-D) also resolves at Phase 0.5, not in-flight.

### 9.6 Per-phase HANDOFF discipline

Per `research_operations_v1.md` §8.7 (HANDOFF at every session boundary) and §9.2 (phase boundaries are session boundaries), each phase ends with a HANDOFF.md update recording:

- Phase outputs (which arm-runs completed; what their map-relevant findings are).
- Pre-committed outcome categories the phase populated (e.g., Phase 1 → preliminary M1–M5 indication; Phase 2 → preliminary L-flat/monotonic/non-monotonic indication).
- Working-tree state.
- Next phase's immediate action.

The closing document drafts only after Phase 4 — no in-flight verdict assignment per pass 1 §1.4-style discipline.

### 9.7 Push hold

Per pass 1 §1.5 and v1 closing §13, push hold remains in effect throughout v2 design and execution. v2 artifacts stay local until v2 closing is drafted and a decision is made about which v2 outputs (the map, the protocol, the worked-example outcome) belong in any external-facing artefact.

### 9.8 Per-sweep-point reliability over sweep-point coverage

The §9.1 contingency rule ("reduce sweep density, not n") and the §9.4 slack reallocation rule ("reallocate to n=20 at existing probe points if fewer than 4 probe-worthy locations emerge") both instantiate one project-level principle: **per-sweep-point statistical reliability is prioritised over sweep-point coverage**. The working-region characterisation depends on per-sweep-point determinations being statistically decisive — M1-vs-M2 distinctions, working-region boundary identification, cross-arm contrast patterns at the regional level all require that each sweep point's working-region status be a high-confidence determination, not a marginal one. A map of many sweep points whose individual statuses are uncertain is less informative than a map of fewer sweep points whose statuses are clear.

This principle inherits as project-level discipline for v3+. Future contingency or slack situations across the project apply the principle without case-by-case design-chat re-derivation: when faced with a tradeoff between sample reliability per point and number of points characterised, choose reliability. Coverage-over-reliability is a deliberate exception requiring explicit design-chat rationale, not a default operating mode.

---

**End of pass 2 §§4–9 (final, post-adversarial-review).** Reviewer feedback from Claude, Grok, and two adversarial review chats (Claude review + Grok review) integrated. Locked pending v3+ inheritance.

Items still flagged for v3+ attention (not blocking v2 execution):

- **§4.2 / §8.5 repetition-presence scope restriction.** Accepted as v2 methodology scope; v3+ inherits the restriction and engages it (either targeting environments satisfying the prerequisite, or extending the methodology to handle single-pass exploration as a v3+ deliverable). Same shape as v1 closing §9.2's "perturbation-mechanism PRE-B belongs in spec design" inheritance pattern.
- **§7.3 / §9.2 joint α empirical robustness.** $\tau_W$ per head is operationalised at PRE-E via Wilcoxon $p < 0.05$ against bit-identical baseline; the joint $\alpha$ depends on empirical head-dependence which is reported in closing rather than pre-committed. Adversarial review confirmed empirical-rather-than-Bonferroni framing as the right call.
- **§9.5 Phase 0.5 design-chat discipline.** Phase 0.5 design chat must be held to the same spec-discipline standard as the current chat. Adversarial review flagged this as the residual risk after density-floor commitment (§9.1) addressed the substantive concern; v2 closing reviews Phase 0.5 discipline retrospectively.

Resolved by pass-2 revisions, design-chat feedback, and adversarial review integration:

- ~~n=3 per arm per sweep point.~~ Committed to n=10 with explicit `research_operations_v1.md` §3.6 override discipline (§9.3) and PRE-D2 empirical n=10-vs-n=20 CI validation feeding Phase 0.5.
- ~~Multi-modal detection in §6.3.~~ GMM + BIC step added.
- ~~τ_W anchoring to v1 co-primary discipline.~~ Anchoring sentence + Wilcoxon $p<0.05$ operationalisation added.
- ~~M5 missing from M-categories.~~ Added as catch-all with declaration discipline (a)/(b)/(c) + consequential/internal v3+ inheritance tie-back.
- ~~Probe budget post-hoc sizing.~~ Upper-bound commit with slack reallocation rule to n=20.
- ~~Shift-vector direction under-specified.~~ Per-repetition direction-seed allocation specified, with three-options contingency for reduced n.
- ~~Locality repetition-dependence buried.~~ Promoted to structural-limitation paragraph in §4.2 (with worked-example-by-construction clarification) and methodology scope in §8.5.
- ~~Single L_d_main blind to L_d × stream-property interactions.~~ Two L_d_main at envelope endpoints {1, 4} committed at baseline; M5 L_d-interaction case detectable from main effects.
- ~~Corner-sampling at 5D extremes risks methodology-internal noise.~~ Corner-avoidance committed at spec time (held axes at 2nd, midpoint, 4th); per-axis extremes appear only as swept-axis values; corner-sampled crosses available as Phase 3 probes if PRE-D1 shows reachability. Adversarial review proposed demotion to SCAFFOLDING; declined per spec-time commitment with PRE-D1-empirical-evidence path preserved.
- ~~Contingency trigger as may-invoke.~~ Replaced with mandatory Phase 0.5 design-chat checkpoint between Phase 0 and Phase 1, receiving three empirical inputs (per-run cost, n=10/20 CI, corner reachability).
- ~~"Reliability-over-coverage" principle implicit.~~ Named explicitly at §9.8 with project-level inheritance.
- ~~Density reduction without floor.~~ Hard density floor added at §9.1 (three crosses, three values per axis); below floor, design chat re-scopes axes or defers sweep to v3+.
- ~~Phase 3 probe selection blind to Phase 2 L-non-monotonic.~~ Fourth selection criterion added at §9.4 covering the endpoint-only L_d coverage gap.
- ~~Endpoint-only L_d_main main-effects coverage.~~ Addressed via §9.4 probe selection rule (criterion 4) rather than additional L_d_main values.
