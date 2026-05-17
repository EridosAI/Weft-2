# Weft Inner PAM v1 — Specification (Pass 2: §§7–11, Implementation)

**Status.** Implementation specification (§§7–11), CC-handoff document. Pass 1 (§§1–6, architecture-and-claims) committed at v1 design lock-in.

**Companion documents.** `WEFT_INNER_PAM_v1_Spec_pass1_sections_1_to_6.md` (architecture and claims, the design rationale for everything specified here); `WEFT_INNER_PAM_v0_CLOSING.md` (institutional memory carrying forward); `WEFT_INNER_PAM_v2_DESIGN_INTAKE.md` (deferred questions). v1 *instructions* document (operational procedures, frame budgets, preflight protocols, calibration procedures) is downstream of this spec and will be produced separately for CC.

**Audience.** This document is CC's implementation reference. Spec-level architectural commitments are stated at code-level precision; operational details (calibration procedures, preflight protocols, frame budgets, checkpoint discovery) are deferred to the instructions document. The division mirrors v0's spec/instructions split.

---

## 7. Predictor architecture specification

v1 specifies three predictor architectures, one per arm. All three share the same encoder body family (transformer) and the same W=16 sliding window pattern (inherited from v0). They differ in readout topology and variance representation. This section specifies all three at code-level precision so each arm can be implemented independently without architectural ambiguity.

### 7.1 Shared scaffolding across all three arms

All three arms inherit the following from v0:

- **Embedding dimension:** d = 1024 (DINOv2-Large output dimension).
- **Window length:** W = 16 input embeddings.
- **Prediction horizon:** K = 16 output steps.
- **Encoder body:** `nn.TransformerEncoder` with norm_first=True, batch_first=True, GELU activation. Hyperparameters: hidden=512, n_heads=8, n_layers=4, mlp_dim=2048. (v0 spec §7.3 SCAFFOLDING defaults; carries forward unchanged.)
- **Input projection:** `nn.Linear(d, hidden)` mapping (B, W, d) → (B, W, hidden).
- **Positional embedding:** `nn.Embedding(W, hidden)` added to input projection outputs. (Note: this is a v0-inherited transformer convention deferred to v2 per §2.3.)
- **Log-variance clamp:** Log-variance outputs clamped to LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX (v0 values inherited from CODING_STANDARDS).
- **Initialisation:** All `nn.Linear` and `nn.Embedding` parameters initialised with PyTorch defaults. Output queries (Primary, Ablation 1) initialised with `nn.init.normal_(std=0.02)` to match transformer-decoder query initialisation convention.

### 7.2 Primary arm specification

The Primary arm implements v1's full architectural commitment: K learnable output queries with cross-attention into the encoder body's output, and per-K-position scalar variance.

#### 7.2.1 Constructor signature

```python
class InnerPAM_v1_Primary(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        window_w: int = 16,
        predict_k: int = 16,
        hidden: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        mlp_dim: int = 2048,
        decoder_n_layers: int = None,   # SCAFFOLDING — calibrated in instructions
    ):
```

**SCAFFOLDING note on `decoder_n_layers`.** Decoder depth is the architectural component v1 adds beyond v0; its capacity affects whether per-K differentiation develops during training (more layers → more cross-attention capacity for queries to learn position-distinguished attention patterns; too few → queries may not develop differentiation; too many → unnecessary parameters, slower training, and potentially harder optimisation). There is no a-priori principled value; this parameter is calibrated empirically per the v0 §15 absolute-magnitude-threshold discipline. The instructions document specifies the calibration procedure (likely: short calibration runs at decoder_n_layers ∈ {1, 2, 3, 4} on a subset of Stage A data, with the value producing stable training dynamics and non-degenerate per-K query differentiation selected; the calibrated value is then locked across all three arms for v1's main runs). No silent default; CC raises an explicit error if `decoder_n_layers=None` reaches the constructor.

#### 7.2.2 Module composition

```python
        # Shared scaffolding (per §7.1)
        if decoder_n_layers is None:
            raise ValueError(
                "decoder_n_layers must be specified explicitly (SCAFFOLDING; "
                "calibrated per instructions document, not silently defaulted)."
            )
        self.embed_dim = embed_dim
        self.window_w = window_w
        self.predict_k = predict_k

        self.input_proj = nn.Linear(embed_dim, hidden)
        self.pos_emb = nn.Embedding(window_w, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # K learnable output queries (v1 Primary-specific)
        self.output_queries = nn.Parameter(torch.empty(predict_k, hidden))
        nn.init.normal_(self.output_queries, std=0.02)

        # Cross-attention decoder block (v1 Primary-specific)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_n_layers)

        # Per-K output projection (mean + scalar log-variance)
        # Each K-th query's hidden vector projects to (d + 1) values: d for mean, 1 for log-variance
        self.output_proj_mean = nn.Linear(hidden, embed_dim)
        self.output_proj_log_var = nn.Linear(hidden, 1)
```

#### 7.2.3 Forward pass

```python
    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # window: (B, W, d)
        assert window.ndim == 3
        b, w, d = window.shape
        assert w == self.window_w
        assert d == self.embed_dim

        # Input projection + positional embedding
        x = self.input_proj(window)                                # (B, W, hidden)
        positions = torch.arange(self.window_w, device=window.device)
        x = x + self.pos_emb(positions).unsqueeze(0)               # (B, W, hidden)

        # Encoder body produces W position-distinguished hidden vectors
        memory = self.encoder(x)                                   # (B, W, hidden)

        # K learnable output queries expanded to batch
        queries = self.output_queries.unsqueeze(0).expand(b, -1, -1)  # (B, K, hidden)

        # Cross-attention decoder: K queries attend into W memory positions
        decoded = self.decoder(queries, memory)                    # (B, K, hidden)

        # Per-K output projection (shared across K positions, parameters in queries+decoder give per-K differentiation)
        mean = self.output_proj_mean(decoded)                      # (B, K, d)
        log_var = self.output_proj_log_var(decoded).squeeze(-1)    # (B, K)
        log_var = log_var.clamp(LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX)

        assert mean.shape == (b, self.predict_k, self.embed_dim)
        assert log_var.shape == (b, self.predict_k)
        return mean, log_var
```

#### 7.2.4 Architectural property assertions

The Primary architecture has the following load-bearing properties; implementation must verify each:

1. **K output queries are per-K parameters.** Each row of `self.output_queries` has its own parameters; gradient updates to one row do not directly update other rows. Verification: parameter count of `self.output_queries` equals K × hidden.
2. **Cross-attention preserves K-positional structure.** The decoder's cross-attention produces K position-distinguished hidden vectors (one per query). Verification: `decoded.shape == (B, K, hidden)`.
3. **Per-K variance reads from K position-distinguished hidden vectors.** The log-variance output for K-th position is produced from `decoded[:, k, :]`, which is the K-th query's cross-attention output. Verification: log_var[:, k] gradient w.r.t. `decoded[:, j, :]` for j ≠ k is zero (per-K parameter isolation in the head).
4. **No pooled `last_token` readout.** The architecture must not contain any line equivalent to `last_token = x[:, -1, :]` or similar pooling-to-single-vector operations between encoder output and per-K projections.

#### 7.2.5 Parameter count

Approximate parameter count for Primary (assuming hidden=512, n_layers=4, K=16, d=1024; decoder layer count `L_d` is SCAFFOLDING per §7.2.1):

- Encoder body: ~8.4M (matches v0 transformer body)
- Output queries: K × hidden = 16 × 512 = 8.2K
- Decoder body: L_d × ~3.2M per layer (each `nn.TransformerDecoderLayer` is ~3.2M params at hidden=512, mlp_dim=2048 — self-attention block ~1.05M, cross-attention block ~1.05M, FFN ~2.1M, layer norms small). For L_d=2, ~6.3M; for L_d=4, ~12.6M.
- Output projection mean: hidden × d = 512 × 1024 = 524K
- Output projection log_var: hidden × 1 = 512
- **Total at L_d=2: ~15.3M parameters; at L_d=4: ~21.6M parameters**

For comparison, v0's predictor was ~16.8M parameters (8.4M encoder + 8.4M output projection from `last_token` to K*(d+1)). v1 Primary is in the same order of magnitude at L_d=2, larger at L_d=4. Per-K differentiation comes from the queries and decoder cross-attention, not from per-K head parameters (the mean and variance projections are shared across K positions).

### 7.3 Ablation 1 specification (variance-head ablation)

Ablation 1 implements the readout topology change but reverts the variance representation to a single shared parameter. This isolates whether per-K-position variance parameters were load-bearing for v1's verdict.

#### 7.3.1 Constructor signature

```python
class InnerPAM_v1_Ablation1(nn.Module):
    def __init__(self, **kwargs):
        # Same as Primary
        ...
```

#### 7.3.2 Module composition

Identical to Primary §7.2.2 *except*:

```python
        # Single shared log-variance parameter (Ablation 1-specific)
        # Replaces self.output_proj_log_var = nn.Linear(hidden, 1)
        self.shared_log_var = nn.Parameter(torch.zeros(1))
```

#### 7.3.3 Forward pass

Identical to Primary §7.2.3 *except*:

```python
        # Variance is a single shared parameter, broadcast to (B, K)
        log_var = self.shared_log_var.expand(b, self.predict_k).clamp(
            LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX
        )
```

#### 7.3.4 Architectural property assertions

1. **K output queries preserved.** Same as Primary 7.2.4 property 1.
2. **Cross-attention preserves K-positional structure.** Same as Primary 7.2.4 property 2.
3. **Variance is parameter-shared across K.** `self.shared_log_var.numel() == 1`. All K log-variance outputs read from the same parameter.
4. **No pooled `last_token` readout.** Same as Primary 7.2.4 property 4.

#### 7.3.5 What Ablation 1 isolates

Ablation 1 replaces Primary's `nn.Linear(hidden, 1)` log-variance head with a single shared `nn.Parameter`. This architectural change removes two things at once: (a) the per-K parameter pathway in the variance head, and (b) the variance head's ability to read from K position-distinguished hidden vectors via a learned projection. Ablation 1 tests *whether variance differentiation requires any learned readout from per-K hidden vectors* — a stronger architectural claim than "per-K-position variance parameters were load-bearing." A learned linear projection from `decoded[:, k, :]` to a scalar is the minimum machinery that could let variance differentiate per K; Ablation 1 removes it entirely.

If Primary succeeds and Ablation 1 *fails on variance differentiation* (mean differentiation works, variance does not), the variance head's learned readout from per-K hidden vectors was load-bearing — variance differentiation requires the variance head to be able to read per-K context, not just the parameter capacity itself.

If Primary succeeds and Ablation 1 *also produces variance differentiation* despite the shared parameter (which would be architecturally surprising — a single scalar parameter cannot encode per-K variance differences directly), variance differentiation must be emerging from some other mechanism such as the loss-driven indirect coupling between mean predictions and the shared variance estimate. This outcome is V1B per §1.2 (if Ablation 2 still reproduces v0 coupling) or V1E per §1.2 (if Ablation 2 also succeeds), but with a stronger qualifier in the verdict-assignment recommendation: the variance differentiation under Ablation 1 is mechanistically suspicious and warrants disaggregating diagnostic characterisation under §10.5 step 3 before verdict assignment.

**Note on what Ablation 1 does *not* test.** A weaker ablation — mean-pool `decoded` over K, then project to a single scalar via a learned linear layer — would test "per-K parameter pathway" specifically while preserving "variance reads from decoder output." v1 does not include this weaker ablation; the stronger ablation (shared scalar parameter) was chosen because it tests a more fundamental architectural commitment with cleaner attribution. The weaker variant is a v2 refinement question if v1 produces V1A or V1B verdict.

### 7.4 Ablation 2 specification (readout-topology ablation)

Ablation 2 reverts both the readout topology and the variance representation to v0's pattern. It tests whether stronger perturbation alone (the §1's commitment 3 intervention, not part of the architectural change) is sufficient to produce per-(item, ordinal) differentiation. If it is, the architectural corrections were not load-bearing — V1E per §1.2.

#### 7.4.1 Constructor signature

```python
class InnerPAM_v1_Ablation2(nn.Module):
    def __init__(self, **kwargs):
        # Same as Primary
        ...
```

#### 7.4.2 Module composition

Identical to v0's `InnerPAM`:

```python
        self.input_proj = nn.Linear(embed_dim, hidden)
        self.pos_emb = nn.Embedding(window_w, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden, predict_k * (embed_dim + 1))
```

No decoder block; no output queries; no per-K projection. This is v0's predictor architecture inherited verbatim.

#### 7.4.3 Forward pass

Identical to v0's forward pass (per `PREDICTOR_FORWARD_EXCERPT.md` lines 74–93):

```python
    def forward(self, window):
        b, w, d = window.shape
        x = self.input_proj(window)
        positions = torch.arange(self.window_w, device=window.device)
        x = x + self.pos_emb(positions).unsqueeze(0)
        x = self.encoder(x)
        last_token = x[:, -1, :]                                   # (B, hidden)
        flat = self.output_proj(last_token)                        # (B, K*(d+1))
        flat = flat.view(b, self.predict_k, self.embed_dim + 1)
        mean = flat[..., : self.embed_dim]
        log_var = flat[..., self.embed_dim].clamp(LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX)
        return mean, log_var
```

#### 7.4.4 Architectural property assertions

1. **Architecture matches v0 verbatim.** `InnerPAM_v1_Ablation2` should be code-equivalent to v0's `InnerPAM` (modulo class name). Verification: `isinstance(model.output_proj, nn.Linear)` and `model.output_proj.out_features == K * (d + 1)`.
2. **Pooled `last_token` readout retained.** The architecture must contain the line `last_token = x[:, -1, :]` in the forward pass. This is the v0 architecture being reproduced as a baseline.
3. **No output queries, no decoder.** Verification: model state_dict contains no parameters with names starting with "output_queries" or "decoder".

#### 7.4.5 What Ablation 2 isolates

Ablation 2 is the "v0 architecture under v1's substrate" control. It tests whether v1's substrate change (stronger perturbation) alone reproduces v0's coupling result.

If Ablation 2 reproduces v0 coupling under v1's stronger perturbation (variance drift on bit-identical (item, ordinal) pairs uniform across input-varying pairs, mirroring v0's Bed result), the architectural corrections are confirmed as necessary — v0's failure was architectural, not substrate-induced. This is the expected outcome under V1A, V1B, V1C.

If Ablation 2 *does not* reproduce v0 coupling under v1's stronger perturbation (per-(item, ordinal) differentiation emerges from substrate strength alone, without the architectural corrections), the architectural corrections were not load-bearing — V1E. This is the fundamentally distinct verdict V1E captures.

### 7.5 Implementation discipline note for CC

The three arms are architecturally distinct PyTorch modules. CC must implement them as three separate classes (`InnerPAM_v1_Primary`, `InnerPAM_v1_Ablation1`, `InnerPAM_v1_Ablation2`) rather than as a single configurable class with arm-switching flags. The separation is deliberate: arm-switching flags in a single class create implementation paths that can silently drift apart, and the architectural verdict v1 produces must be unambiguous about which architecture each arm ran.

Each class lives in its own file: `src/predictor/inner_pam_v1_primary.py`, `src/predictor/inner_pam_v1_ablation1.py`, `src/predictor/inner_pam_v1_ablation2.py`. Each file is independently testable. Shared utilities (LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX, etc.) live in a shared module that all three import.

---

## 8. Substrate and perturbation mechanism

### 8.1 Substrate inheritance and v1-specific changes

v1's substrate inherits from v0 with one architectural change:

- **Environment:** AI2-THOR + ProcTHOR. Inherited from v0.
- **Encoder:** DINOv2-Large frozen. Inherited from v0; substrate verification (v0 §5) carries forward.
- **Stage structure:** Stage A (baseline) → Stage B (perturbed). Inherited from v0.
- **Item bank:** Five furniture items (Bed, Dresser, Sofa, TV, DiningTable), each with multiple viewing positions in the ProcTHOR house. Inherited from v0 with substrate findings 5–8 applied (camera elevation, floor-y derivation, view-through pose adjustment, cross-room locality).
- **Perturbation mechanism:** *Changed from v0.* See §8.2.
- **Continuous-motion trajectory:** Inherited from v0 (substrate finding 3 — no static dwell).

### 8.2 Perturbation mechanism specification

v1's architectural commitment (§1's intervention 3) is to a perturbation regime producing 0.05–0.1-magnitude cross-stage cosine drops at perturbed items. The specific mechanism is not specified at the spec level; the instructions document specifies the preflight procedure for mechanism selection.

#### 8.2.1 Verification criterion

A candidate perturbation mechanism is acceptable for v1 if it satisfies all of:

1. **Magnitude.** Cross-stage DINOv2 cosine drop at perturbed items, measured at viewing position 1 of each perturbed item, falls in [0.05, 0.10]. Tighter than v0's `RandomizeMaterials` regime (~0.01); looser than would alter the architectural identity of the perturbed item.
2. **Locality.** Cross-stage DINOv2 cosine drop at *unperturbed* items (different items, or unperturbed viewing positions of perturbed items) falls below 0.015. The v0 §5.8 cross-scope perturbation locality check applies; v1's mechanism is acceptable only if it passes §5.8 at the v1 magnitude.
3. **Reproducibility.** Cross-stage cosine drops are stable across multiple AI2-THOR sessions when the same seed is used. Run-to-run variation falls below 0.005.
4. **No substrate corruption.** The mechanism does not produce camera-elevation, floor-y, view-through, or cross-room-leakage substrate issues (v0 substrate findings 5–8). The substrate verification protocol §5 is run on each candidate mechanism before acceptance.

#### 8.2.2 Candidate mechanisms

Four candidate mechanisms in order of investigation priority, per the v0 closing-doc §8.3 hierarchy:

1. **Per-object material setting** (`SetObjectMaterials` if API supports per-object granularity; or per-object `RandomizeMaterials` if subscoped).
2. **Asset replacement at fixed coordinates** (replace the perturbed item's mesh and texture asset at the same world coordinates, producing a different item-instance visually while preserving pose).
3. **Hand-built texture swaps** (offline-rendered texture replacements applied via custom shader path).
4. **Alternate ProcTHOR scene** (perturbed Stage B uses a structurally-equivalent ProcTHOR scene with different items at the same viewing positions).

The instructions document specifies the preflight protocol: CC tests each mechanism in order, running the §5 substrate verification + §8.2.1 magnitude/locality/reproducibility verification at each. The first mechanism passing all criteria is selected. If none pass, CC escalates.

#### 8.2.3 Per-arm substrate consistency

All three arms use the *same* perturbation mechanism. This is load-bearing for v1's arm-comparison structure: differences between arms must be attributable to architectural differences, not to substrate differences. The instructions document specifies a single CC preflight that selects the mechanism, then all three arms train against the same selected mechanism.

### 8.3 Substrate verification protocol (extending v0 §5)

#### 8.3.1 §5.1–§5.3 inherited

The DINOv2 encoder has passed v0's §5.1–§5.3 substrate verification against the seed-7 furniture bank. This carries forward unchanged. CC re-runs §5.1–§5.3 on the v1 substrate to verify the substrate has not regressed, but the protocol itself is unchanged from v0.

#### 8.3.2 §5.4 magnitude verification (new for v1)

The perturbation mechanism selected per §8.2 must pass §5.4: cross-stage DINOv2 cosine drop at perturbed items falls in [0.05, 0.10]. Measured at viewing position 1 of each of the five perturbed items in the item bank. v0's §5.4 was magnitude-agnostic; v1's §5.4 is magnitude-specific.

#### 8.3.3 §5.7 floor-y derivation (inherited)

Modal-y across `GetReachablePositions` results. Inherited from v0.

#### 8.3.4 §5.8 cross-scope perturbation locality (re-run for v1)

v0 §5.8 measured cross-scope locality of `RandomizeMaterials`. v1's perturbation mechanism is different; §5.8 needs re-running on the selected v1 mechanism. The protocol is unchanged: measure cross-stage DINOv2 cosine drop at unperturbed items, verify it falls below 0.015.

#### 8.3.5 §5.9 reproducibility check (new for v1)

A new substrate verification check for v1: cross-stage cosine drop reproducibility across AI2-THOR sessions. CC runs the perturbation mechanism twice with the same seed, verifies cross-stage cosines are within 0.005 of each other at all perturbed items.

### 8.4 Substrate failure modes

The v1 substrate has five substrate findings inherited from v0 as default checks (substrate findings 5–8 of v0 closing §4, plus the v0-finding-4 substrate-as-feature-vs-bug discipline). CC's substrate verification includes:

1. Python 3.12.3 environment check.
2. Embeddings.npy full-population check.
3. Continuous-motion substrate check (no 30-frame static dwell).
4. Substrate-as-feature-vs-bug interpretive discipline at the v1 calibration boundary (see §10 for instances where v1's calibration may surface substrate-as-feature outcomes).
5. Camera elevation check.
6. Floor-y derivation via modal-y.
7. View-through pose check.
8. Cross-room visual leakage check (now §5.8 on v1's perturbation mechanism).

Substrate findings produced by v1 (issues not anticipated at design time) are recorded in v1's HANDOFF.md and inherited by v2 as additional default checks. This is the v0 institutional-memory pattern carrying forward.

---

## 9. Training pipeline

### 9.1 Per-arm training structure

Each of the three arms (Primary, Ablation 1, Ablation 2) is trained independently with the same training pipeline. The training pipeline differs from v0's only in arm-specific architecture and arm-specific logging.

### 9.2 Stage structure

v1 retains v0's two-stage structure:

- **Stage A (baseline):** 100k frames of the agent traversing the unperturbed furniture-sequence trajectory in the ProcTHOR house. Continuous-motion substrate; no perturbation; same trajectory across loops at this stage.
- **Stage B (perturbed):** 200+ loops of the agent traversing the same trajectory with the §8.2 perturbation applied. Number of loops determined by §10's evaluation requirements (the loop-100-equivalent for v1 is empirically calibrated against per-(item, ordinal) signal stability — see instructions document).

Stage A → Stage B transition is the architectural-claim test: under the path-prediction loss, the predictor's representations should accommodate the Stage B perturbation by producing per-input differentiation at perturbed (item, ordinal) pairs while maintaining stability at bit-identical (item, ordinal) pairs.

### 9.3 Training loop specification

Each arm trains via the following loop, executed independently:

```python
for stage in ["A", "B"]:
    for loop_idx in range(n_loops[stage]):
        for frame_idx in range(frames_per_loop):
            # Get current window of W=16 most recent frames
            window = embeddings[max(0, frame_idx - W + 1):frame_idx + 1]
            # Pad if early in stream (first W-1 frames)
            if len(window) < W:
                window = pad_start(window, target_len=W)

            # Get K-step target embeddings
            target = embeddings[frame_idx + 1:frame_idx + 1 + K]
            if len(target) < K:
                continue  # Skip if not enough future frames

            # Forward pass
            mean, log_var = model(window.unsqueeze(0))

            # Compute path-prediction loss (per §4.1 of pass 1)
            loss = path_prediction_loss(mean, log_var, target.unsqueeze(0))

            # Backward pass + optimiser step
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Per-(item, ordinal) logging at every checkpoint frame
            if frame_idx in checkpoint_frames:
                log_per_item_ordinal_metrics(...)
```

The exact frame budgets, checkpoint cadence, optimiser hyperparameters, and the `pad_start` operation specification (zero-padding vs replicate-first-frame vs skip-until-W-frames-available, and the implications for very-early-trajectory training signal) are specified in the instructions document (downstream of this spec). The spec specifies the *structure* of the training loop; the instructions specify the *numerics* and the early-trajectory handling.

### 9.4 Optimiser

v1 retains v0's optimiser configuration: AdamW with weight_decay=0.01, learning rate 3e-4, no learning rate schedule. Inherited from v0 unchanged.

If v1 produces V1C, V1D, or V1P verdicts, the optimiser configuration becomes a candidate v2 variable for re-examination. v1 inherits unchanged because v0's evidence did not implicate the optimiser as a failure source.

### 9.5 Checkpoint cadence

Checkpoints saved every 10k training steps, matching v0. Specific checkpoint numbers (e.g., which checkpoints are used for per-(item, ordinal) evaluation) are specified in the instructions document.

The means-doing-work discipline §5 of pass 1 applies to checkpoint cadence: if early-stage training shows large transitions before settling (the soft v1 implication flagged in BCDD analysis), CC may increase checkpoint cadence early in training. The decision is made at calibration time based on observed dynamics, not pre-committed.

### 9.6 Logging requirements

Per-(item, ordinal) metrics are logged at every checkpoint, per the means-doing-work discipline. Specifically:

1. **Per-(item, ordinal) predicted mean.** The K=K-1 (final-step) predicted mean for each (item, ordinal) pair, evaluated on the canonical window for that pair (defined in §10).
2. **Per-(item, ordinal) predicted log-variance.** The K=K-1 predicted log-variance for each (item, ordinal) pair (Primary, Ablation 1) or the shared scalar log-variance broadcast (Ablation 2).
3. **Per-(item, ordinal) loss.** The path-prediction loss evaluated on the canonical window for each (item, ordinal) pair.
4. **Per-K disaggregation.** All three metrics also logged at each K step (not just K=K-1), so per-K profiles are inspectable.
5. **Aggregate metrics for comparison.** Total loss across the training corpus, mean cosine similarity of predicted-vs-actual targets, per-arm parameter counts. These are aggregate metrics paired with disaggregated metrics, per §5.

All metrics logged in structured JSON format (per the instructions document's logging schema).

### 9.7 What does not carry forward from v0's training pipeline

- **Single-arm structure.** v0 was single-arm; v1 is 3-arm. All three arms train independently, with full per-arm state isolation.
- **Phase 1 substrate-degenerate baseline.** v0's Phase 1 (100k substrate-degenerate baseline) is not load-bearing for v1's verdict and is not carried forward.

---

## 10. Evaluation framework

### 10.1 Per-(item, ordinal) evaluation discipline

v1's evaluation operates at per-(item, ordinal) granularity by default. The discipline §5 of pass 1 binds every evaluation operation:

- All metrics computed per-(item, ordinal) pair.
- All gates evaluated against per-(item, ordinal) distributions.
- All arm-comparison metrics summarised at per-(item, ordinal) granularity.
- All verdict assignments made at per-(item, ordinal) granularity per §1.2.1.

Aggregate metrics are paired with disaggregated forms for inspection. Aggregate-only metrics are forbidden at verdict-load-bearing decisions.

### 10.2 Canonical windows and target frames

For each (item, ordinal) pair, the canonical window and canonical target frame are defined as follows:

- **Canonical viewing position:** Viewing position 1 of each item (the established v0 convention). Each item has 5 viewing positions; ordinal 1 of position 1 is the canonical evaluation point.
- **Canonical window:** The W=16 embeddings ending K frames before the canonical target frame. (v0's K-back window convention, established via the BCDD protocol-mismatch trace and confirmed via the v0-style variance drift sanity check.)
- **Canonical target frame:** The frame at the canonical viewing position 1, ordinal 1, for each item.

Per-(item, ordinal) metrics are computed using these canonical windows and targets. The instructions document specifies the loop-100-equivalent for v1's Stage B duration (when "end of Stage B" is reached and evaluation runs).

### 10.3 The seven primary evaluation metrics

For each (item, ordinal) pair and each arm, compute:

1. **Mean drift at canonical target.** Cosine distance between predicted mean (averaged over K) at end-of-Stage-A checkpoint vs end-of-Stage-B checkpoint. Per v0's BCDD-confirmed protocol.
2. **Variance drift at canonical target.** log_σ² difference between end-of-Stage-A and end-of-Stage-B checkpoints, averaged over K. Same convention as v0's published variance drift values.
3. **Per-K mean drift profile.** Mean drift values per K step (not averaged). Surfaces the per-K patterns the BCDD analysis identified.
4. **Per-K variance drift profile.** Variance drift values per K step. Surfaces per-K variance dynamics.
5. **Mean stability at bit-identical (item, ordinal) pairs.** Cosine distance between end-of-Stage-A and end-of-Stage-B predicted means at (item, ordinal) pairs that are pixel-MD5-identical across stages. Expected near zero under v1's architectural claim.
6. **Variance stability at bit-identical (item, ordinal) pairs.** log_σ² difference at pixel-MD5-identical (item, ordinal) pairs across stages. Expected near zero.
7. **Body representation cosine across stages (BCDD Test A continuation).** Encoder body's pre-readout representation cosine on bit-identical input windows across end-of-Stage-A and end-of-Stage-B checkpoints. Diagnostic for whether body-coupling persists across training (which v0/BCDD evidence established as the dominant coupling pathology). The readout point differs by architecture:
   - **Primary and Ablation 1.** Encoder output at the W-th (last) position: `memory[:, -1, :]` where `memory = self.encoder(x)` per §7.2.3. This is the encoder body's final-window-position output before cross-attention. Chosen for direct comparability with v0/BCDD's `last_token` measurement (same encoder, same window position, same architecture for the body itself).
   - **Ablation 2.** `last_token = x[:, -1, :]` per §7.4.3, identical to v0's BCDD-measured representation by construction (Ablation 2 reproduces v0 architecture).

   The metric is computed identically across all three arms (cosine between two (B, hidden) vectors), but the underlying representation source differs. For Primary and Ablation 1, the metric characterises body drift independent of the readout topology change; for Ablation 2, the metric reproduces v0's BCDD measurement. Cross-arm comparison is meaningful because the body itself is architecturally identical across arms (same encoder, same window, same input projection) — only the downstream readout differs.

All seven metrics are computed for each arm independently. Per-arm summaries are produced for arm-comparison (see §10.4); per-(item, ordinal) values are preserved for verdict assignment (see §10.5).

### 10.4 Arm-comparison structure

Arm comparison uses per-(item, ordinal) values, not arm aggregates. The structure:

#### 10.4.1 Per-(item, ordinal) per-arm matrix

For each metric in §10.3, produce a (item, ordinal) × arm matrix. Each cell is the metric value for that (item, ordinal) pair in that arm. The matrix is the primary arm-comparison artifact.

#### 10.4.2 Verdict-class-specific patterns

The matrix supports per-verdict-class pattern recognition:

- **V1A pattern.** Primary column shows clean differentiation (large drift at perturbed (item, ordinal), small drift at bit-identical (item, ordinal)). Ablation 1 column shows mean differentiation similar to Primary but variance differentiation weakened or absent. Ablation 2 column shows v0-style coupling (uniform drift across input-varying and bit-identical (item, ordinal)).
- **V1B pattern.** Primary and Ablation 1 columns both show co-primary differentiation. Ablation 2 reproduces v0 coupling.
- **V1C pattern.** All three columns show some form of v0-style coupling.
- **V1D pattern.** Three concrete sub-patterns, each corresponding to a §1.2 D-class sub-outcome:
  - **V1D-mean-only.** Primary column shows mean differentiation matching V1A's mean pattern, but variance differentiation is absent or weakened in Primary's variance metrics. Ablations 1 and 2 show patterns consistent with V1A's expectations for their respective architectural variants. The verdict isolates a variance-specific failure on Primary despite per-K-position parameters being present.
  - **V1D-variance-only.** Primary column shows variance differentiation matching V1A's variance pattern, but mean differentiation is absent or weakened. Ablations show patterns consistent with V1A's expectations. The verdict isolates a mean-specific failure, which is architecturally surprising (the mean head has more parameters and stronger gradient signal than the variance head); verdict-assignment recommendation flags this as warranting mechanistic characterisation.
  - **V1D-heterogeneous.** Primary column shows co-primary differentiation at some (item, ordinal) pairs but not others (the Sofa-ord-1 pattern at scale — localised signal). Specific items, specific ordinals, or specific (item, ordinal) combinations succeed while others fail. The verdict-assignment recommendation includes the pattern of which (item, ordinal) succeed and which fail; this is itself the result, not a noisy V1A or V1C verdict to be cleaned up by averaging.
- **V1E pattern.** All three columns show co-primary differentiation. Architectural corrections were not load-bearing.
- **V1P pattern.** One or more substrate issues invalidate the per-(item, ordinal) measurements before verdict-class assignment is meaningful.

#### 10.4.3 Verdict-pattern threshold calibration

The specific thresholds defining "differentiation" vs "coupling" vs "stability" at the per-(item, ordinal) level are SCAFFOLDING parameters per §1.2 — calibrated against the empirical distribution of values produced by v1's runs, not declared a priori.

The instructions document specifies the calibration procedure. The high-level pattern: after all three arms complete training, the empirical distribution of each metric across (item, ordinal) × arm is examined; thresholds are anchored to this distribution (e.g., "differentiation" defined as values in the upper 25th percentile, "stability" defined as values below the 25th percentile of bit-identical (item, ordinal) pairs); verdict patterns are assigned based on these empirically-anchored thresholds.

### 10.5 The verdict-assignment workflow

Per §1.2.1, v1's verdict is assigned via a 5-step workflow:

1. **All three arms complete training.** CC verifies all arms reached end-of-Stage-B without unconditional-failure stop triggers.
2. **Per-(item, ordinal) evaluation runs across all arms.** CC produces the per-(item, ordinal) × arm matrix for all seven metrics in §10.3.
3. **One round of disaggregating diagnostics runs if ambiguous.** If the matrix pattern doesn't cleanly map to a verdict class (heterogeneous outcomes; signal at the edge of significance; structured outliers requiring characterisation), CC runs one round of disaggregating diagnostics. Bounded: characterise the existing data, don't collect new data, don't retrain, don't extend training. The instructions document specifies what disaggregating diagnostics are available.
4. **v1 chat produces verdict-assignment recommendation.** v1 chat (this design chat, post-evaluation) reviews the matrix and the disaggregating diagnostics, classifies the pattern into V1A–V1P, and produces a verdict-assignment recommendation with reasoning. The recommendation is a structured document specifying: the assigned verdict class, the per-(item, ordinal) pattern supporting the assignment, the disaggregating diagnostics consulted, and confidence level.
5. **Reviewer chat issues verdict.** Separate-context reviewer chat (per research_operations §2.2) reviews v1 chat's recommendation adversarially. Reviewer chat issues the final verdict, which is recorded in v1's closing document. Reviewer chat may request additional disaggregating diagnostics if the recommendation's evidence is insufficient; the request goes back to CC, results return to reviewer chat, and the workflow completes.

Post-verdict diagnostics are v2 intake material, not v1 verdict revision.

### 10.6 What v1's evaluation framework does not do

Three things explicitly out-of-scope for v1 evaluation:

1. **Repetition-stratified accuracy.** v0 spec §M4 metric measuring per-repetition-frequency prediction accuracy. v0 did not cleanly test this; v1 does not attempt to test it. v2 may revisit.
2. **Action / reward integration.** v0 deferred; v1 inherits the deferral.
3. **Episodic-to-semantic consolidation.** v0 deferred; v1 inherits the deferral.

---

## 11. Operational discipline

### 11.1 Chat handoffs

v1 design and execution involves four distinct chat contexts. Each has specific input/output requirements:

#### 11.1.1 v1 design chat (this chat)

- **Input:** v0 closing document, v1 design intake brief, BCDD diagnostic results.
- **Output:** v1 spec pass 1 (§§1–6), v1 spec pass 2 (§§7–11), v2 design intake document, v1 instructions document.
- **Status at v1 design lock-in:** complete; produces no further design artifacts until v1 verdict workflow §10.5 step 4.

#### 11.1.2 v1 reviewer chat

- **Input:** v1 spec pass 1, v1 spec pass 2, v1 instructions document.
- **Output:** Adversarial review of v1 design; sign-off for CC implementation or surface-of-concerns flagging.
- **Status:** runs after v1 design chat produces complete v1 spec + instructions; before CC begins implementation.

#### 11.1.3 v1 CC implementation chat (CC working sessions)

- **Input:** v1 spec pass 2, v1 instructions document, post-reviewer-sign-off.
- **Output:** v1 implementation (three predictor classes, three training runs, per-(item, ordinal) evaluation matrix, HANDOFF.md updates per session).
- **Status:** runs after reviewer sign-off; ends when all three arms complete training and per-(item, ordinal) evaluation is produced.

#### 11.1.4 v1 verdict-assignment chat (post-evaluation)

- **Input:** CC's per-(item, ordinal) × arm matrix, disaggregating diagnostics if needed; v1 spec passes 1 and 2; v1 instructions document; v0 closing document.
- **Output:** Verdict-assignment recommendation per §10.5 step 4.
- **Status:** runs after CC produces evaluation matrix.
- **Practical note on chat continuity.** §10.5 step 4 names "v1 chat" as the producer of verdict-assignment recommendation. In practice, the v1 design chat context will likely be at or near token capacity by the time evaluation completes (after pass 1, pass 2, v2 intake, instructions, plus any mid-design pressure-tests). The verdict-assignment chat may therefore be a *fresh chat reading v1 design artifacts cold* rather than a literal continuation of the v1 design chat. The artifacts (passes 1 and 2, instructions, v0 closing, evaluation matrix, HANDOFF.md) are the canonical handoff record; a fresh chat working from them produces a verdict-assignment recommendation with the same disciplinary basis as a continuation chat would. The reviewer chat in §11.1.5 reviews the recommendation on its merits, not on chat-continuity grounds.

#### 11.1.5 v1 verdict-issuance chat (reviewer chat continuation)

- **Input:** v1 design chat's verdict-assignment recommendation, CC's evaluation matrix.
- **Output:** Final verdict per §10.5 step 5.
- **Status:** runs after v1 design chat's recommendation; ends when verdict is issued.

### 11.2 Stop conditions

The following are unconditional stop conditions for v1 CC implementation. CC stops and reports per CODING_STANDARDS §9.4:

1. **Substrate verification failure.** Any of §8.3.1–§8.3.5 fails during preflight or in-flight. CC halts arm execution and reports.
2. **NaN/Inf in training.** Loss, gradients, or model outputs produce NaN or Inf at any training step. CC halts the affected arm and reports.
3. **Architectural property assertion failure.** Any of §7.2.4, §7.3.4, §7.4.4 fails at construction or at forward pass. CC halts arm execution and reports.
4. **Per-(item, ordinal) logging gap.** If checkpoint-time logging fails for any (item, ordinal) pair, CC halts the affected arm and reports.
5. **Reproducibility failure.** If §8.3.5 reproducibility check fails mid-experiment (cross-stage cosines drift more than 0.005 between sessions on the same seed), CC halts and reports.
6. **In-flight perturbation magnitude drift.** At every checkpoint during Stage B, CC re-measures cross-stage DINOv2 cosine drop at perturbed items (the §8.2.1 magnitude verification). If the measured magnitude departs from preflight's measured value by more than 0.01, or falls outside the [0.05, 0.10] band, CC halts and reports. Rationale: v0's cross-room visual leakage finding emerged from training-time behaviour rather than preflight (substrate finding 8 in v0 closing §4); the precedent supports in-flight monitoring of the perturbation magnitude as a first-class stop condition rather than trusting preflight to characterise full-experiment behaviour.

Stop conditions trigger CC ⇆ v1 design chat handoff. v1 design chat reviews the stop condition, recommends resolution, and either authorises resumption or escalates to reviewer chat.

### 11.3 HANDOFF.md discipline

v1 inherits v0's HANDOFF.md discipline. Each CC session ends with a HANDOFF.md entry recording:

- Session goals and what was attempted.
- Code commits in this session.
- Substrate verification status (which §8.3 checks ran, what results).
- Stop conditions encountered (if any) and their resolution.
- Outstanding questions for v1 design chat or reviewer chat.

HANDOFF.md is the canonical institutional-memory artifact across CC sessions. Verdict-load-bearing decisions and diagnostic findings flow through HANDOFF.md to v1's closing document.

### 11.4 Push hold maintenance

Push hold remains in effect through v1 execution. v1's repository commits land locally; nothing pushes to remote until v1 verdict is issued and the v1 closing document is produced.

Push hold lift is a deliberate decision made jointly by v1 design chat, reviewer chat, and the human researcher, after v1 verdict is issued.

### 11.5 The single-variable-discipline reminder

v1 is structured to test specific architectural interventions with attribution clean across arms. Any mid-experiment scope expansion (adding a fourth arm; changing perturbation mechanism after preflight selects one; adding architectural variants) violates the single-variable discipline and is a stop condition triggering v1 design chat handoff.

This is institutional discipline from v0 carrying forward: v0's verdict was clean because v0 did not bundle changes. v1's verdict will be clean only if v1 does not bundle changes either.

---

## 12. Document status and next steps

### 12.1 Document status

Pass 2 (§§7–11, implementation specification) is the CC-handoff document. After review and lock-in, v1's design phase produces one more artifact:

- **v1 instructions document.** Operational procedures, frame budgets, optimiser hyperparameters, checkpoint cadence specifics, preflight protocols, calibration procedures. Downstream of this spec; the spec specifies *what* CC implements, the instructions specify *how* CC operates the implementation.

### 12.2 Next steps

1. **v1 design chat reviews this document** for completeness, internal consistency, and code-level precision.
2. **v1 design chat produces v1 instructions document** as the final design artifact.
3. **v1 reviewer chat reviews pass 1 + pass 2 + instructions** per §11.1.2 adversarially.
4. **CC begins v1 implementation** per §11.1.3 after reviewer sign-off.

Push hold remains in effect through all of the above.

---

*End of v1 spec §§7–11 (implementation specification). Companion to `WEFT_INNER_PAM_v1_Spec_pass1_sections_1_to_6.md`, `WEFT_INNER_PAM_v0_CLOSING.md`, and `WEFT_INNER_PAM_v2_DESIGN_INTAKE.md`. v1 instructions document follows as the next design artifact.*
