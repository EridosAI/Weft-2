# Weft Inner PAM v1 — Experiment Instructions

**Status:** First draft. Audience: CC (Claude Code), executing autonomously per `CODING_STANDARDS.md` and `research_operations_v1.md`. Adversarial review pending; CC implementation begins after reviewer sign-off per spec §11.1.2.

**Purpose:** Specify *how* CC operates the implementation laid out in `WEFT_INNER_PAM_v1_Spec_pass1_sections_1_to_6.md` and `WEFT_INNER_PAM_v1_Spec_pass2_sections_7_to_11.md`. The spec specifies *what* the architecture is and *why*; this document specifies operational procedures, frame budgets, preflight protocols, and calibration procedures. If this document and the spec disagree, the spec wins and this document is wrong — flag in `HANDOFF.md` and stop.

**Read order:**
1. `WEFT_INNER_PAM_v1_Spec_pass1_sections_1_to_6.md` (architecture and claims)
2. `WEFT_INNER_PAM_v1_Spec_pass2_sections_7_to_11.md` (implementation specification)
3. `CODING_STANDARDS.md` (operational discipline)
4. `research_operations_v1.md` (process discipline)
5. `WEFT_INNER_PAM_v0_CLOSING.md` (institutional memory, including BCDD-revised §7.1)
6. `HANDOFF.md` (current state)
7. This document

**Push hold remains in effect throughout this batch.**

**Structural differences from v0 instructions.** v1 is *three architecturally distinct arms in parallel structure*, not three sequential phases. There is no recall mixer (v1 does not test confidence-graded mixing; that question is downstream of the architectural-claim verdict v1 produces). There is no shape-clustering evaluation framework (v1's evaluation is per-(item, ordinal) on the seven metrics specified in spec §10.3, calibrated against empirical distributions per §10.4.3). The v0 instructions document is the discipline reference for what a full-discipline instructions document looks like; v1 inherits the structure and adapts the content.

---

## 0. Environment Header

| field | value |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| Shell | bash |
| Python | 3.12.3 (matches v0; substrate finding 1 carries forward) |
| PyTorch | 2.10.0+cu128 (CUDA 12.8 via WSL2 passthrough), pinned in `requirements.txt` |
| Encoder model | `facebook/dinov2-large` (loaded via `transformers`); inherits v0's substrate verification |
| Host CPU | Intel Core i9-14900K @ 3.20 GHz |
| Host RAM | 64 GB |
| GPU | RTX 4080 Super, 16 GB VRAM (local) |
| Host disk | check `df -h` ≥ 100 GB free at batch start (see §0.1 disk budget) |
| Working repo | `/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/` |
| Reference repo | `/mnt/c/Users/Jason/Desktop/Eridos/Weft/` (read-only; route metadata only) |
| Active venv | none; system Python 3.12.3 (v0 pattern) |
| Required env var | `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000` |
| AI2-THOR version | `ai2thor==5.0.0` (v0 pin; carries forward) |

CC verifies the environment matches before any work. Divergence is documented in `HANDOFF.md` and resolved before proceeding.

### 0.1 Host-session protection (local hardware run)

This batch runs on local hardware over an estimated ~40–60 hours of wall-clock across multiple sessions. The host-session protection from v0 §0.1 carries forward verbatim:

1. **Use `tmux` for the CC session itself.** `tmux new -s weft` inside WSL2 before launching CC. Detach with `Ctrl-b d`; reattach with `tmux attach -t weft`. `sudo apt install tmux` if not present.

2. **Disable Windows sleep and disk sleep** for the batch duration. Settings → System → Power → Screen and sleep → "When plugged in, put my device to sleep" = **Never**. Advanced power settings → Hard disk → "Turn off hard disk after" = **0 (Never)**.

3. **Defer Windows updates** for the batch window. Settings → Windows Update → Pause updates for at least the planned batch duration.

4. **Disk budget check before any arm starts.** Estimated disk usage across the full batch:
   - Per-arm collected frames (Stage A 100k + Stage B ~72–144k at 360 frames/loop): ~9–12 GB × 3 arms (the perturbation mechanism is shared across arms but each arm collects its own frames against the same perturbation seed, since training is sequential and the seed-controlled scene state is restored).
   - Embeddings (~172–244k × 1024 × fp32 × 3 arms): ~2–3 GB total.
   - Checkpoints (~15–22M params per arm × ~15 checkpoints × 3 arms): ~3–5 GB.
   - Calibration runs (decoder layer calibration, perturbation mechanism characterisation): ~5 GB.
   - Logs, TensorBoard runs, per-(item, ordinal) JSON: ~2–3 GB.
   - **Total: ~25–35 GB.** CC verifies `df -h` returns ≥50 GB free at every arm boundary and stops if it doesn't.

5. **GPU lockout.** Same as v0 — close concurrent GPU-using applications before starting. Experiment chat's responsibility before CC is launched.

CC's responsibility within this protection layer: launch all long-running scripts with `nohup`, poll logs, honour CODING_STANDARDS §5.2 / §5.3. The host-side protections above are verified at the start of each new session alongside reading HANDOFF.md per CODING_STANDARDS §1.2.

---

## 1. Scope Lock and Locked Decisions

These are settled by the v1 spec. They do not get re-litigated in this batch.

### 1.1 Architectural commitments (from spec §§2, 7)

- **Three predictor architectures, three separate classes, separate files** (spec §7.5). `InnerPAM_v1_Primary`, `InnerPAM_v1_Ablation1`, `InnerPAM_v1_Ablation2` live in `src/predictor/inner_pam_v1_primary.py`, `src/predictor/inner_pam_v1_ablation1.py`, `src/predictor/inner_pam_v1_ablation2.py`. No arm-switching flags in a single class.
- **Shared encoder body** across all three arms: `nn.TransformerEncoder`, hidden=512, n_heads=8, n_layers=4, mlp_dim=2048, GELU, norm_first, batch_first (spec §7.1).
- **Path-prediction loss** with Gaussian NLL (spec §4.1); Form 1 variance representation (per-K scalar) for Primary, single shared scalar parameter for Ablation 1, v0-style scalar-from-pooled-vector for Ablation 2 (spec §3.3, §7.2.3, §7.3.3, §7.4.3).
- **Three architectural property assertions per arm** (spec §7.2.4, §7.3.4, §7.4.4). Verified at PRE-D preflight (§6.4 below).
- **Co-primary mean and variance differentiation** (spec §1.4). Both must hold for V1A; failure of either falsifies v1 architecture for this task.

### 1.2 Encoder choice (spec §6.1)

DINOv2-Large frozen, fp16 eval, ImageNet mean/std normalisation, L2-normalised output. v0 verified against the §5 protocol; v1 inherits and re-runs §5.1–§5.3 to confirm no substrate regression (§6.1 below).

### 1.3 Environment, trajectory, and the v0 substrate inheritance (spec §8.1)

- AI2-THOR seed-7 furniture house, driven by `src/env/continuous_motion_explorer.py` from v0.
- **Five-item route:** `Bed → DiningTable → Dresser → Sofa → Television → Bed → ...` (inherited from v0).
- **Route file:** `data/route_phase2.json` (v0's session-5 substrate revision with adjusted DiningTable pose; carries forward unchanged unless preflight identifies a substrate regression).
- **Continuous motion throughout.** No held-pose dwell. v0's session-5 substrate locks this in (§2.3 architectural commitment).
- **Camera elevation** set by modal-y derivation from `GetReachablePositions` (spec §5.7 / v0 finding 6). Inherited.
- **Cross-loop variation** comes from Stage A → Stage B perturbation regime, not from pose jitter (v0's session-5 decision; carries forward).
- **Room composition.** LivingRoom: Dresser, Sofa. Bedroom: Bed, DiningTable, Television. Inherited.
- **Empirical loop length:** v0 measured 360 frames/loop on the v2 substrate. v1 re-calibrates loop length on the *selected v1 perturbation mechanism* (PRE-B, §6.2 below) before deriving Stage B frame budget — different mechanisms may produce different loop lengths if asset replacement or alternate-scene paths alter NavMesh transit distances.

### 1.4 Three-arm structure

v1 is structured as three independent training runs against the *same* substrate and *same* perturbation mechanism (spec §8.2.3). All three arms train against a **single shared frame collection** — Stage A collected once, Stage B collected once, both shared across arms. Each arm differs only in its predictor architecture (per spec §11.5 single-variable discipline; the variables tested across arms are architectural, the substrate is held constant).

The perturbation mechanism is locked after PRE-B; no re-selection occurs per arm.

Each arm:

1. Initialises a freshly-constructed predictor of its arm-specific class.
2. Trains on the shared Stage A embeddings (unperturbed; 100k frames per spec §9.2).
3. Trains on the shared Stage B embeddings (signal-stability-calibrated duration; §8.1 below).
4. Logs per-(item, ordinal) metrics at every checkpoint (spec §9.6).
5. Produces final checkpoint files at end-of-Stage-A and end-of-Stage-B for downstream evaluation.

The three arms run **sequentially** (VRAM contention on the 16 GB 4080 Super prevents parallel runs). Per-arm training wall-clock is ~1.5 hours (predictor-only, no encoding at training time, since embeddings are precomputed); full three-arm training sequence is ~5 hours plus collection (~6 hours one-time shared), encoding (~75 minutes one-time shared), preflight (~5 hours), and evaluation (~1.5 hours).

**Why shared collection rather than per-arm collection.** The architectural variables tested across arms are predictor-internal; the substrate is identical across arms by spec §8.2.3. Sharing frame collection across arms makes substrate-equivalence true by construction rather than by verification. The reproducibility check that would otherwise re-verify substrate-equivalence per arm is satisfied once at PRE-B (spec §8.3.5). This matches v0's Phase 2 pattern (collected once, trained by both main and shuffle predictors) and is the architecturally cleaner choice.

Arm-comparison is post-hoc: after all three arms complete, the per-(item, ordinal) × arm matrix is constructed and verdict thresholds are calibrated against the empirical distribution per spec §10.4.3 and §9.4 below.

### 1.5 What NOT to change

- **The architecture specs.** If a problem encountered during implementation suggests the spec is wrong, stop and report — do not edit the spec.
- **The encoder** (DINOv2-Large CLS, frozen) — v0's verification carries forward unchanged.
- **The shared scaffolding** in spec §7.1 (d=1024, W=16, K=16, hidden=512, n_heads=8, n_layers=4, mlp_dim=2048).
- **The Form 1 variance representation** for Primary (per-K scalar; spec §3.3). Forms 2 and 3 are explicitly held in reserve for v2.
- **The three-arm structure.** Adding a fourth arm, removing one, or changing one mid-experiment is a stop condition per spec §11.5 (single-variable discipline).
- **The perturbation mechanism after PRE-B selects one.** All three arms use the same mechanism (spec §8.2.3); switching mid-experiment violates the arm-comparison structure.
- **No pose jitter, no per-frame perturbation, no held-pose dwell.** v0's session-4 / session-5 substrate decisions carry forward verbatim.
- **No tightening of perturbation magnitude beyond [0.05, 0.10] without explicit authorisation.** This range is spec §1.2 commitment 3; tighter substrate produces unintended outcomes per §11.2 condition 6 (in-flight perturbation magnitude drift).
- **No bundling of architectural changes across arms.** Each arm differs from Primary along exactly one architectural axis (Ablation 1: variance head; Ablation 2: readout topology + variance head reverting to v0).
- **Numbers trace to files.** No remembered numbers. No mental arithmetic on metrics.

### 1.6 What is deferred to v2

Per spec §2.3 and §6.3:

1. Body family (transformer family inherited from v0).
2. Temporal-context mechanism (W=16 sliding window inherited).
3. Positional information mechanism (additive positional embeddings inherited).
4. Bounded memory mechanism (hard-window memory inherited).
5. Encoder substitution.
6. Form 2 / Form 3 variance representations.
7. Synthetic-stream companion arm.
8. Repetition-stratified accuracy, action/reward integration, episodic-to-semantic consolidation (spec §10.6).

If implementation surfaces a "we'd want this" instinct that touches any of these, write a note in `HANDOFF.md` for the v2 design chat and continue with v1.

---

## 2. Repository Layout

CC builds out this structure. Establish before file count grows (CODING_STANDARDS §2.1).

```
Weft 2/
├── WEFT_INNER_PAM_v1_Spec_pass1_sections_1_to_6.md      (exists)
├── WEFT_INNER_PAM_v1_Spec_pass2_sections_7_to_11.md     (exists)
├── WEFT_INNER_PAM_v1_EXPERIMENT_INSTRUCTIONS.md         (this doc)
├── CODING_STANDARDS.md                                  (exists)
├── research_operations_v1.md                            (exists)
├── WEFT_INNER_PAM_v0_CLOSING.md                         (exists; v0 institutional memory)
├── WEFT_INNER_PAM_v2_DESIGN_INTAKE.md                   (exists; v2 deferral record)
├── HANDOFF.md                                           (exists; updated per session)
├── requirements.txt                                     (pinned per CODING_STANDARDS §8)
├── src/
│   ├── encoder/
│   │   └── dinov2_encoder.py        # v0; carries forward unchanged
│   ├── predictor/
│   │   ├── inner_pam.py             # v0 InnerPAM; preserved for Ablation 2 reference
│   │   ├── inner_pam_v1_shared.py   # LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX, shared helpers
│   │   ├── inner_pam_v1_primary.py  # Primary arm class
│   │   ├── inner_pam_v1_ablation1.py
│   │   └── inner_pam_v1_ablation2.py
│   ├── trainer/
│   │   └── online_trainer_v1.py     # Stage A → Stage B per-arm loop; arm-agnostic
│   ├── eval/
│   │   ├── per_item_ordinal_metrics.py  # the seven §10.3 metrics
│   │   ├── arm_comparison_matrix.py     # per-(item, ordinal) × arm matrix construction
│   │   └── threshold_calibration.py     # §9.4 below
│   ├── env/
│   │   ├── continuous_motion_explorer.py  # v0 explorer; carries forward
│   │   └── explorer_v1.py                  # selected-perturbation-mechanism wrapper
│   ├── preflight/
│   │   ├── pre_a_substrate_verification.py
│   │   ├── pre_b_perturbation_mechanism.py
│   │   ├── pre_c_decoder_layer_calibration.py
│   │   └── pre_d_arch_property_assertions.py
│   └── config.py                    # all hyperparameters, with ARCHITECTURE/SCAFFOLDING labels
├── scripts/
│   ├── run_pre_a_substrate.py
│   ├── run_pre_b_perturbation.py
│   ├── run_pre_c_decoder_calibration.py
│   ├── run_pre_d_arch_assertions.py
│   ├── run_collect_stage_a.py           # one-time, shared across arms
│   ├── run_encode_stage_a.py
│   ├── run_collect_stage_b.py           # one-time, shared across arms
│   ├── run_encode_stage_b.py
│   ├── run_arm_primary_train.py
│   ├── run_arm_ablation1_train.py
│   ├── run_arm_ablation2_train.py
│   ├── run_per_item_ordinal_eval.py
│   ├── run_arm_comparison_matrix.py
│   └── run_threshold_calibration.py
├── data/
│   ├── route_phase2.json                                (exists; v0 inheritance)
│   ├── v1_perturbation_mechanism_metadata.json          (PRE-B output)
│   ├── v1_shared/{frames_stage_a/, frames_stage_b/, embeddings_stage_a.npy, embeddings_stage_b.npy, annotations_stage_a.jsonl, annotations_stage_b.jsonl, bit_identical_item_ordinal.json}
│   └── pre_c_calibration/{frames/, embeddings.npy, annotations.jsonl}  (PRE-C subset; 30k Stage A frames)
├── results/
│   ├── inner_pam_v1/
│   │   ├── pre_a_substrate/
│   │   ├── pre_b_perturbation/
│   │   ├── pre_c_decoder_calibration/
│   │   ├── pre_d_arch_assertions/
│   │   ├── arm_primary/
│   │   ├── arm_ablation1/
│   │   ├── arm_ablation2/
│   │   ├── arm_comparison_matrix/
│   │   ├── threshold_calibration/
│   │   └── verdict_recommendation/
│   └── v1_design/                                       (exists; bcdd_results.json etc.)
└── logs/
```

**Note on shared frame collection.** Stage A and Stage B frames are collected once per stage, then shared across all three arms (per §1.4 above). Per-arm directories live under `results/inner_pam_v1/arm_{name}/` for arm-specific outputs (predictor checkpoints, per-(item, ordinal) metrics, training logs), while the input data is shared at `data/v1_shared/`. The bit-identical (item, ordinal) set, computed from canonical-target-frame pixel-MD5 comparison across the shared Stage A and Stage B frame sets, is also at `data/v1_shared/bit_identical_item_ordinal.json` and used by all arms' per-(item, ordinal) evaluation.

---

## 3. Predictor Implementations

Three classes per spec §7. Each in its own file. Shared utilities (clamp constants, helper functions) in `inner_pam_v1_shared.py`.

### 3.1 Primary (spec §7.2)

Implementation per spec §7.2.1–§7.2.3 verbatim. Constructor signature, module composition, and forward pass match the spec. CC implements without deviation; if any implementation choice is ambiguous, flag in HANDOFF and consult v1 design chat per §13 / spec §11.1.4.

**The `decoder_n_layers` parameter is SCAFFOLDING per spec §7.2.1; CC raises `ValueError` if the constructor is called with `decoder_n_layers=None`.** Calibrated by PRE-C (§6.3 below); the calibrated value is written to `src/config.py` as `V1_DECODER_N_LAYERS` and imported by training scripts. The single calibrated value is used by both Primary and Ablation 1.

### 3.2 Ablation 1 (spec §7.3)

Identical to Primary except: `self.shared_log_var = nn.Parameter(torch.zeros(1))` replaces the per-K log-variance head; forward pass broadcasts the shared scalar to `(B, K)` before clamping. Spec §7.3.4 architectural property assertions verified at PRE-D.

Note on parameter initialisation: the spec specifies `torch.zeros(1)`, which means Ablation 1's initial log-variance is exactly 0 (variance = 1). This sits well inside the LOG_VAR_CLAMP range and is consistent with v0's per-K scalar initialisation pattern.

**Framing (per spec §7.3.5).** Ablation 1 tests a stronger claim than "per-K-position variance parameters are load-bearing." It tests whether variance differentiation requires the variance head to read from per-K position-distinguished hidden vectors via a learned projection at all. If Ablation 1 fails on variance differentiation while Primary succeeds, the conclusion is that variance must read from the decoder output (not merely that it needs per-K parameters).

### 3.3 Ablation 2 (spec §7.4)

This is v0's `InnerPAM` class with a v1-specific class name. CC implements by *importing v0's `InnerPAM` and subclassing it* with a class-name override only — no architectural changes:

```python
# inner_pam_v1_ablation2.py
from src.predictor.inner_pam import InnerPAM

class InnerPAM_v1_Ablation2(InnerPAM):
    """v0 InnerPAM architecture inherited verbatim for the readout-topology ablation."""
    pass
```

Rationale: spec §7.4.2 states "Ablation 2 should be code-equivalent to v0's `InnerPAM` (modulo class name)." Subclassing without modification is the cleanest way to enforce this — any change to v0's `InnerPAM` propagates automatically (and PRE-D's property assertions catch any unintended divergence).

### 3.4 Architectural property assertions (spec §7.2.4 / §7.3.4 / §7.4.4)

CC implements assertions as standalone test functions in `src/preflight/pre_d_arch_property_assertions.py`. The four assertions per arm are verified at PRE-D (§6.4 below) on each arm's constructed module before training launches. Failure of any assertion is a stop condition (spec §11.2 condition 3).

### 3.5 Parameter count check

At PRE-D, CC logs `sum(p.numel() for p in model.parameters() if p.requires_grad)` for each arm and verifies against the following targets:

- Primary at L_d=1: ~17.9M; L_d=2: ~22.1M; L_d=3: ~26.3M; L_d=4: ~30.5M. Tolerance ±10%.
- Ablation 1: Primary's count − 512 params (the per-K log-var head `nn.Linear(hidden, 1)` replaced by the shared scalar `nn.Parameter(torch.zeros(1))`). At each L_d, Ablation 1 sits 512 below Primary; within Primary's ±10% tolerance band by construction.
- Ablation 2: ~21.6M (v0 InnerPAM verbatim). Tolerance ±10%.

CC writes the counts to `results/inner_pam_v1/pre_d_arch_assertions/parameter_counts.json` and stops if any is outside tolerance. The per-module breakdown (encoder body per-layer, decoder body per-layer, heads / projections) is written alongside; the L_d capacity envelope across {1, 2, 3, 4} is written to `parameter_counts_l_d_envelope.json` so the realistic capacity range is visible alongside PRE-D rather than after PRE-C completes.

**Discrepancy with spec §7.2.5 (recorded for reviewer chat).** Spec §7.2.5 listed Primary L_d=2: ~15.3M; L_d=4: ~21.6M; v0 / Ablation 2: ~16.8M. The empirical counts measured at PRE-D (2026-05-19) are ~22.1M / ~30.5M / ~21.6M respectively. Source of error: the spec's per-encoder-layer cost of "~2.1M" omitted the ~1.05M self-attention QKVO block (each `nn.TransformerEncoderLayer` is FFN ~2.1M + self-attention ~1.05M + norms = ~3.15M, ×4 = ~12.6M, not ~8.4M as the spec claims); and the per-decoder-layer "~3.2M" likewise undercounts (each `nn.TransformerDecoderLayer` is self-attention 1.05M + cross-attention 1.05M + FFN 2.1M + norms = ~4.2M, not 3.2M). The implementation is faithful to spec §7.2.2 module composition; the spec arithmetic in §7.2.5 is the error. Per instr §1.5 ("only the design chat edits spec"), the spec is not edited from CC's side; this targets-update is staged for reviewer-chat sign-off as part of the v1 review cycle (§14).

---

## 4. Online Trainer

`src/trainer/online_trainer_v1.py`. Single-pass, continuous-time, one gradient step per resolved prediction, per spec §9.3. Arm-agnostic: the trainer accepts a constructed predictor of any of the three classes and runs the same loop.

### 4.1 Stream contract

The trainer consumes a precomputed stream of L2-normalised DINOv2 embeddings: `embeddings: np.ndarray` of shape `(N, 1024)` and corresponding annotations `(N,)` (matching the v0 JSONL schema, with the `perturbation_active: bool` field per arm).

Per-arm streams: `data/arm_{primary,ablation1,ablation2}/embeddings.npy` and `data/arm_{primary,ablation1,ablation2}/annotations.jsonl`. CC verifies at training-start that shapes are consistent, L2 norms are 1.0 ± 1e-5 on a 1000-frame sample, and annotation count matches embeddings rows.

### 4.2 Training loop (per spec §9.3)

At each step `t`:
1. Build window `[t - W + 1, ..., t]` of W=16 most recent embeddings.
2. If window is shorter than W (early-trajectory frames), apply `pad_start` per §4.3.
3. Build K-step target `[t + 1, ..., t + K]`. If fewer than K future frames remain, skip the step (no prediction enqueued).
4. Forward pass: `mean, log_var = model(window.unsqueeze(0))`.
5. Compute path-prediction loss per spec §4.1.
6. Backward pass; optimiser step; zero gradients.
7. At checkpoint frames (§4.5), pause training and run per-(item, ordinal) logging (§4.6).

The trainer contract is **one prediction enqueued per step and one gradient step per resolved prediction**. Implementation detail is left to CC; the contract is what matters.

### 4.3 `pad_start` early-trajectory handling

**v1 commits to skip-until-W-frames-available.** The trainer does *not* feed early-trajectory windows of length < W to the predictor — the predictor's first training step is at frame `t = W - 1 = 15`. The 15 frames before this are observed but not used for training.

Rationale. Three options were considered (spec §9.3 lists them implicitly via "the exact `pad_start` operation specification"):

- **Zero-padding** feeds non-physical content to the encoder body. The transformer's attention sees a window where the first positions are zero-vectors; the body's attention patterns are shaped by these zero-tokens, polluting the training signal in ways that don't correspond to anything the substrate produces during the main trajectory.
- **Replicate-first-frame** creates a phantom 15-frame static dwell at trajectory start, re-introducing the §2.3 architectural violation v0 fought through three substrate revisions (substrate finding 3) to remove. v1 inherits the no-zero-velocity-frames commitment (spec §8.1 substrate inheritance); replication contradicts it.
- **Skip-until-W** loses 15 training frames per arm against a 100k-frame Stage A budget — 0.015% of training signal. Trivial, and produces a clean early-trajectory contract: training begins when a physically-meaningful window of length W exists.

The 15 unused frames are still encoded and saved to `embeddings.npy` for downstream analysis; only the training loop skips them.

### 4.4 Optimiser (spec §9.4)

- AdamW
- `lr = 3e-4` (v0 SCAFFOLDING; carries forward)
- `weight_decay = 0.01` (v0 SCAFFOLDING)
- `betas = (0.9, 0.999)`
- No LR schedule
- Gradient clipping: `clip_grad_norm_(predictor.parameters(), max_norm=1.0)` (v0 SCAFFOLDING)

If v1's verdict is V1C, V1D, or V1P, the optimiser configuration is a candidate v2 re-examination variable per spec §9.4 closing paragraph. v1 inherits unchanged.

### 4.5 Checkpoint cadence

Two regimes, combined:

**Dense early-Stage-A cadence.** Steps 1k / 2k / 4k / 6.5k / 10k. Captures the initial training transition the spec §9.5 means-doing-work paragraph flags. Five early checkpoints per arm.

**Standard cadence past 10k steps.** Every 10k training steps thereafter, plus end-of-Stage-A and end-of-Stage-B as canonical checkpoints. End-of-Stage-A lands at frame 100,000 (~step 99,985 after `pad_start` skip); end-of-Stage-B is signal-stability-calibrated per §8.1.

For a 100k-frame Stage A + 72k-frame Stage B (minimum 200 loops at 360 frames/loop), the checkpoint schedule produces:

- Stage A: 1k / 2k / 4k / 6.5k / 10k / 20k / 30k / 40k / 50k / 60k / 70k / 80k / 90k / end (14 checkpoints).
- Stage B: 110k / 120k / 130k / 140k / 150k / 160k / end (variable count depending on calibration).

End-of-Stage-A and end-of-Stage-B are load-bearing canonical checkpoints for the per-(item, ordinal) evaluation (spec §10.3). Other checkpoints are diagnostic — used for in-flight monitoring (§8.2) and for any post-verdict disaggregating diagnostics (spec §10.5 step 3).

**Checkpoint files** to `results/inner_pam_v1/arm_{name}/ckpt_{step}.pt`:
- Predictor state dict
- Optimiser state dict
- RNG state (numpy + torch)
- Git commit hash at save time
- Training step

Per CODING_STANDARDS §5.4. A checkpoint that can't be reproduced from its commit hash is a checkpoint we don't have.

### 4.6 Logging requirements (spec §9.6)

**Per-step (TensorBoard):** loss, mean log-variance (across K), gradient norm, current step index.

**Per-checkpoint (JSON):** at every checkpoint, write to `results/inner_pam_v1/arm_{name}/checkpoint_{step}.json`:
- training step
- mean loss over last interval (10k steps for standard cadence, 1k for dense)
- mean log-variance per step k (vector of K values) — aggregate
- predictor weight L2 norm
- timestamp
- git commit hash

**Per-(item, ordinal) (JSON):** at every checkpoint, write to `results/inner_pam_v1/arm_{name}/per_item_ordinal_{step}.json` per spec §9.6:
1. Per-(item, ordinal) predicted mean at the K=K-1 (final step) head, evaluated on the canonical window per spec §10.2.
2. Per-(item, ordinal) predicted log-variance at K=K-1.
3. Per-(item, ordinal) path-prediction loss on the canonical window.
4. Per-K disaggregation of (1), (2), (3) (vectors of K values).
5. Per-(item, ordinal) body representation cosine across stages (the §10.3 metric 7; requires comparison against end-of-Stage-A checkpoint, computed lazily once both checkpoints exist).

**Aggregate metrics (paired with disaggregated forms) per spec §5 / §9.6 item 5:** total loss across the training corpus, mean cosine similarity of predicted-vs-actual targets, per-arm parameter count.

**Per-session (HANDOFF.md):** at session boundaries — what was attempted, what completed, what's running (PID + log path), next immediate action.

All metrics in structured JSON. Schema specified in `src/eval/per_item_ordinal_metrics.py` module-level docstring (CODING_STANDARDS §6.2).

### 4.7 Shape assertions and init-time checks (per CODING_STANDARDS §7)

Before training starts on each arm:
1. Assert encoder is frozen: `encoder.eval()` and `for p in encoder.parameters(): assert not p.requires_grad`.
2. Assert predictor is trainable: `sum(p.numel() for p in predictor.parameters() if p.requires_grad) > 0`.
3. Run one forward pass with synthetic input; verify output shapes match the spec §7 forward-pass contracts.
4. Verify embedding norms in the stream: sample 1000 random indices, check norms in `[1.0 - 1e-5, 1.0 + 1e-5]`.
5. Verify the four §7.2.4 / §7.3.4 / §7.4.4 architectural property assertions for the arm's class.

Failure of any check is a stop condition (spec §11.2).

---

## 5. Evaluation Framework — Per-(item, ordinal) Discipline

The evaluation framework is specified in detail in spec §10. This section specifies the *operational* dimensions: the canonical-window construction, the per-(item, ordinal) metric procedure, and the threshold calibration that maps the per-(item, ordinal) × arm matrix to a verdict pattern.

### 5.1 Canonical windows and target frames (spec §10.2)

Per spec §10.2:
- **Canonical viewing position:** viewing position 1 of each item (the v0 convention).
- **Canonical window:** the W=16 embeddings ending K=16 frames *before* the canonical target frame. The K-th predicted step lands ON the canonical target frame. (This is v0's K-back window convention, established by the BCDD protocol-mismatch trace; the per-ordinal evaluation depends on it.)
- **Canonical target frame:** the frame at the canonical viewing position 1 for each item, at each ordinal of the close-up segment.

CC implements canonical-window construction in `src/eval/per_item_ordinal_metrics.py`. The annotations file specifies which frame indices are canonical viewing-position-1 frames per item per close-up ordinal; CC builds the (item, ordinal) → (canonical_window_indices, canonical_target_index) map at module init and re-uses it for every checkpoint's per-(item, ordinal) evaluation.

**Bit-identical (item, ordinal) pair identification.** A pair is bit-identical across Stage A → Stage B if the canonical target frame's pixel-MD5 hash is identical across Stage A and Stage B (the v0 substrate finding that surfaced in HANDOFF — the cross-room visual leakage check). CC computes the pixel-MD5 set across Stage A canonical target frames and Stage B canonical target frames at PRE-A (substrate verification) and at the end of frame collection; the bit-identical set is written to `data/arm_{name}/bit_identical_item_ordinal.json` and used by the threshold-calibration procedure (§9.4).

### 5.2 The seven metrics (spec §10.3)

For each (item, ordinal) pair and each arm, CC computes:

1. **Mean drift at canonical target.** Cosine distance (= 1 − cosine) between predicted mean (averaged over K) at end-of-Stage-A checkpoint vs end-of-Stage-B checkpoint, evaluated on the canonical window.
2. **Variance drift at canonical target.** log_σ² difference (B − A; mean-over-K) between end-of-Stage-A and end-of-Stage-B checkpoints. Same convention as v0's `variance_drift_loop30_to_loop100` (the BCDD-confirmed sign convention).
3. **Per-K mean drift profile.** Mean drift per K step (not averaged); vector of K values.
4. **Per-K variance drift profile.** Variance drift per K step; vector of K values.
5. **Mean stability at bit-identical pairs.** Cosine distance between Stage A and Stage B predicted means at canonical pairs that are pixel-MD5-identical across stages. Expected near zero under v1's architectural claim.
6. **Variance stability at bit-identical pairs.** log_σ² difference at pixel-MD5-identical pairs.
7. **Body representation cosine across stages (BCDD Test A continuation).** Per spec §10.3 metric 7: encoder body's pre-readout representation cosine on bit-identical input windows across end-of-Stage-A and end-of-Stage-B checkpoints. Readout point per arm:
   - Primary, Ablation 1: `memory[:, -1, :]` where `memory = self.encoder(x)` (the encoder body's final-window-position output before cross-attention into the K queries).
   - Ablation 2: `last_token = x[:, -1, :]` per spec §7.4.3 (identical to v0's BCDD measurement).

   The metric is computed identically across all three arms (cosine between two `(B, hidden)` vectors), but the underlying representation source differs per arm. This is intentional: cross-arm comparison is meaningful because the body is architecturally identical across arms (only the readout differs).

Metrics 1–6 are computed per-(item, ordinal); metric 7 is per-canonical-window (one value per arm at each canonical (item, ordinal) pair that meets the bit-identical criterion).

### 5.3 Per-(item, ordinal) × arm matrix construction

After all three arms complete training, CC constructs the matrix via `scripts/run_arm_comparison_matrix.py`:

- Rows: (item, ordinal) pairs (~55 rows for 5 items × ~11 close-up ordinals per item).
- Columns: arm × metric (3 arms × 7 metrics = 21 columns).
- Cell: the metric value for that (item, ordinal) in that arm.

The matrix is written to `results/inner_pam_v1/arm_comparison_matrix/matrix.json` and `matrix.csv`. The CSV is for human inspection (reviewer chat, verdict-assignment chat); the JSON is for downstream programmatic threshold calibration.

### 5.4 Threshold calibration procedure (spec §10.4.3)

Per spec §10.4.3, "differentiation" / "coupling" / "stability" thresholds are SCAFFOLDING calibrated against the empirical distribution of values produced by v1's runs. The procedure:

**Step 1. Build the stability reference distribution.** For each metric, collect the metric values at all bit-identical (item, ordinal) pairs across all three arms (per the §5.1 bit-identical set). Under v1's architectural claim these should cluster near zero (no drift on bit-identical inputs); the empirical distribution characterises what "near zero" actually looks like under v1's training dynamics.

**Step 2. Build the differentiation reference distribution.** For each metric, collect the metric values at *input-varying* (item, ordinal) pairs (the complement of the bit-identical set) across all three arms. Under v1's architectural claim these should be larger in magnitude (substrate input variation produces drift); the empirical distribution characterises the response magnitude.

**Step 3. Anchor thresholds to percentiles.**
- **Stability threshold** = 75th percentile of the bit-identical distribution. A pair's metric value above this threshold is classified as "non-stable" for that metric.
- **Differentiation threshold** = 25th percentile of the input-varying distribution. A pair's metric value below this threshold is classified as "non-differentiated."
- **Coupling pattern** = values that cluster near the stability threshold *for input-varying pairs* (the v0 BCDD pattern: variance drift on bit-identical Bed ords 9/10 was indistinguishable from drift on input-varying ords).

The 75 / 25 percentile choices are SCAFFOLDING. They reflect the asymmetric assumption that under v1's architectural claim, ~75% of bit-identical pairs cluster near zero and ~75% of input-varying pairs show meaningful response. After the first arm completes, the empirical distributions are inspected; if the percentile choices produce a degenerate verdict assignment (e.g., 100% of pairs cross the differentiation threshold including the bit-identical ones — meaning the threshold is too lenient — or 0% — meaning too strict), CC flags in HANDOFF and the verdict-assignment chat (spec §11.1.4) re-calibrates.

**Step 4. Map per-(item, ordinal) cells to verdict patterns** per spec §10.4.2:
- V1A pattern: Primary column clean differentiation (large drift at input-varying, near-zero at bit-identical); Ablation 1 column mean-only differentiation; Ablation 2 reproducing v0 coupling.
- V1B pattern: Primary + Ablation 1 both co-primary differentiation; Ablation 2 v0 coupling.
- V1C pattern: All three columns v0-style coupling.
- V1D sub-patterns (mean-only / variance-only / heterogeneous): see spec §10.4.2.
- V1E pattern: All three columns co-primary differentiation.
- V1P pattern: substrate failure prevents per-(item, ordinal) measurement.

Threshold calibration outputs to `results/inner_pam_v1/threshold_calibration/{thresholds.json, distributions.{json,png}}`. The PNG renders the stability and differentiation distributions per metric for human inspection.

### 5.5 Means-doing-work discipline (spec §5)

Per spec §5: every aggregation operation has one of three justifications. Default disposition is option 3 (paired with disaggregated form). The trainer logs aggregate metrics (mean loss across training corpus, mean cosine similarity) for arm-level comparison but pairs each with the per-(item, ordinal) disaggregated form.

Specific applications inherited from spec §5.2:

- **Arm-comparison metrics.** Per-(item, ordinal) granularity by default. Arm-aggregate metrics (e.g., "mean variance differentiation across all (item, ordinal) for Primary") are paired with the per-(item, ordinal) disaggregated form.
- **Gates and thresholds.** No absolute-magnitude thresholds without empirical calibration (spec §1.2 SCAFFOLDING note). All gate decisions surface per-(item, ordinal) values alongside aggregate values.
- **2σ-band discriminators.** Spec §5.2 forbids 2σ-band-only discriminators at verdict-load-bearing decisions. The threshold calibration above uses percentile-based thresholds on the full empirical distributions, not 2σ-band tests; CC's verdict-assignment output preserves both the percentile thresholds and the per-(item, ordinal) values supporting the assignment.

### 5.6 What evaluation does *not* do (spec §10.6)

Out of scope:
1. Repetition-stratified accuracy (v0 spec §M4). Not tested by v1.
2. Action / reward integration. Deferred to v2.
3. Episodic-to-semantic consolidation. Deferred to v2.

If implementation surfaces an instinct to add evaluation along these axes, write a note in HANDOFF for v2 and continue with v1's scope.

---

## 6. Preflight Protocols

Four preflights run sequentially before any arm training begins. Each produces a verification report; failure of any preflight is a stop condition. Preflights are one-off (run once for the v1 batch; their outputs are then locked across all three arms).

### 6.1 PRE-A: Substrate verification (spec §8.3)

Runs the v0 §5 substrate verification protocol on the v1 substrate, plus v1-specific extensions.

#### 6.1.1 §5.1–§5.3 (inherited from v0)

CC re-runs the v0 protocol on the v1 substrate to confirm no regression:

- **§5.1 Cross-instance stability.** Mean cosine of DINOv2 embeddings across multiple instances of the same item at the same viewing position across loops. Threshold > 0.75.
- **§5.2 Cross-element distinguishability.** Mean cosine across different items. Threshold < 0.60.
- **§5.3 Combined gap.** §5.1 − §5.2 ≥ 0.15.

CC runs this on a short calibration collection (50 frames per item, 10 frames per viewing position) before the main per-arm collection.

#### 6.1.2 §5.7 floor-y derivation (inherited)

Modal-y across `GetReachablePositions` results, per spec §5.7 and v0 finding 6. CC verifies the explorer sets `self._agent_floor_y` to the modal-y value at controller-init time and that all subsequent Teleport calls use that fixed y.

#### 6.1.3 Eight substrate findings checklist (spec §6.2 / §8.4)

CC runs each as a discrete check:
1. Python 3.12.3 environment.
2. Embeddings full-population check (post-collection, all rows non-zero, L2 norms in tolerance).
3. Continuous-motion substrate check (no 30-frame static dwell; verified at collection time by checking that all consecutive frame pairs have DINOv2 cosine < `FINDING_3_COS_MAX` = 0.999). Threshold recalibrated 0.9999 → 0.999 on 2026-05-19 per design-chat determination: the strict 0.9999 catches the 25 transit→close_up structural-handoff duplicates (5 items × 5 loops × 1 pair = 25) that are not 30-frame static dwell. v0 STOP_REPORT (seed-7 rerender, 2026-05-12) set the precedent for this recalibration class. v2 candidate: run-length-aware check that separates dwell from segment-handoff without threshold recalibration (see `WEFT_INNER_PAM_v2_DESIGN_INTAKE.md`).
4. Substrate-as-feature-vs-bug interpretive discipline: CC documents any observed substrate property in HANDOFF and flags ambiguous cases for v1 design chat review before classifying as a bug requiring fix.
5. Camera elevation check.
6. Floor-y derivation (per §6.1.2 above).
7. View-through pose check (DiningTable adjusted pose verified per `data/route_phase2.json`).
8. Cross-room visual leakage check (per §8.3.4 below on v1's perturbation mechanism).

Output: `results/inner_pam_v1/pre_a_substrate/pre_a_report.{md,json}`. PASS verdict iff all eight checks pass.

### 6.2 PRE-B: Perturbation mechanism selection (spec §8.2)

Selects the perturbation mechanism producing 0.05–0.10-magnitude cross-stage cosine drops at perturbed items, with locality and reproducibility per spec §8.2.1.

#### 6.2.1 Candidate mechanisms (spec §8.2.2)

Four candidates in priority order:
1. **Per-object material setting** (`SetObjectMaterials` if AI2-THOR API supports per-object granularity; else per-object `RandomizeMaterials` if subscoped).
2. **Asset replacement at fixed coordinates** (replace the perturbed item's mesh and texture asset at the same world coordinates via `RemoveFromScene` + `PlaceObjectAtPoint` or equivalent).
3. **Hand-built texture swaps** (offline-rendered texture replacements applied via custom shader path).
4. **Alternate ProcTHOR scene** (perturbed Stage B uses a structurally-equivalent ProcTHOR scene with different items at the same viewing positions).

#### 6.2.2 Test all four (framing-disposition decision)

**v1 PRE-B runs every candidate through the verification criteria, not stop-at-first-pass.** Reasoning: the spec's "first mechanism passing all criteria is selected" rule (spec §8.2.2) governs *selection for the main run*, but characterising what each mechanism produces is cheap and valuable independent of selection — v2's body-family experiment will need to make its own perturbation-mechanism choice, and full PRE-B characterisation across all four candidates means v2 inherits empirical data on every candidate rather than just the one v1 selected. Compute cost: each candidate requires ~50 frames collected + DINOv2 encoded + verification metrics computed; per-candidate ~30 minutes of pure work, but AI2-THOR scene reset overhead between candidate mechanisms is non-trivial (each candidate requires a clean scene init plus the mechanism's specific setup — `RandomizeMaterials` state, asset placement, alternate-scene generation), pushing the realistic total to ~3–4 hours for all four. Negligible against the ~50-hour batch.

#### 6.2.3 Per-candidate verification (spec §8.2.1)

For each candidate, CC executes the following sequence:

1. **API verification.** Run the candidate's AI2-THOR API call(s); verify `lastActionSuccess = True`. Capture metadata (e.g., material identifiers from `RandomizeMaterials`).
2. **Frame capture.** Capture canonical-viewing-position-1 frames for each of the 5 items, both pre-perturbation (Stage A state) and post-perturbation (Stage B state).
3. **DINOv2 encoding.** Encode the 10 frames (5 items × {pre, post}) through frozen DINOv2-Large.
4. **§8.2.1 verification.**
   - **Magnitude (§8.2.1.1).** Compute cross-stage cosine drop = 1 − cos(pre_embedding, post_embedding) at each item's viewing position 1. Verify drop ∈ [0.05, 0.10] at perturbed items (Dresser, Sofa for LivingRoom-scoped mechanisms; configurable per mechanism).
   - **Locality (§8.2.1.2).** Verify cross-stage drop < 0.015 at unperturbed items.
   - **Reproducibility (§8.2.1.3).** Re-run the same mechanism with the same seed; verify cross-stage cosines stable within 0.005 across the two runs (spec §8.3.5).
   - **No substrate corruption (§8.2.1.4).** Run the eight-finding checklist (§6.1.3) on the perturbed substrate; verify no new substrate issues.

5. **Loop-length calibration.** For the candidate, run a short 5-loop trajectory under the perturbed substrate and measure frames-per-loop. v0 measured 360 frames/loop on the v2 substrate; v1's mechanism may differ (especially asset replacement, which may force NavMesh re-planning). Record the empirical loop length.

#### 6.2.4 Selection and characterisation outputs

For each candidate, write `results/inner_pam_v1/pre_b_perturbation/{candidate_name}/pre_b_report.{md,json}` recording all five verification criteria and the loop-length calibration. Across candidates, write a summary at `results/inner_pam_v1/pre_b_perturbation/summary.{md,json}` ranking the candidates by which criteria they pass.

**Selection rule (spec §8.2.2):** The mechanism for the main run is the first candidate in §6.2.1 priority order that passes *all* five verification criteria. If multiple candidates pass, the higher-priority one is selected per spec §8.2.2. If none pass, CC escalates per spec §11.2 condition 1 (substrate verification failure).

Selected mechanism is written to `src/config.py` as `V1_PERTURBATION_MECHANISM` and the corresponding loop length to `V1_FRAMES_PER_LOOP`. Both are imported by all subsequent collection and training scripts.

### 6.3 PRE-C: Decoder layer count calibration (spec §7.2.1 SCAFFOLDING)

The `decoder_n_layers` parameter is SCAFFOLDING per spec §7.2.1. There is no a-priori principled value; calibration is empirical.

#### 6.3.1 Calibration protocol

CC runs **all four** of L_d ∈ {1, 2, 3, 4} (framing-disposition decision: compute is not the constraint; characterising the response across L_d values produces v2 inheritance and informs the selection rule's robustness):

For each L_d value:
1. Construct an `InnerPAM_v1_Primary` instance with `decoder_n_layers=L_d`.
2. Train on a Stage A subset: first 30k frames of the to-be-collected Stage A stream for the Primary arm. (Stage A frames are unperturbed for all arms; one collection serves all four calibration runs.)
3. Use the same optimiser configuration as the main run (§4.4).
4. Save checkpoints at steps 1k / 5k / 10k / 20k / 30k.
5. Compute calibration metrics at end-of-calibration (step 30k):
   - **Loss curve smoothness.** Standard deviation of step-to-step loss difference over the final 5k steps; ratio against the global loss mean. Lower is more stable; degenerate values flag training instability.
   - **Per-K query differentiation.** On a sample batch of 64 canonical windows (drawn from the calibration Stage A subset), compute the K=16 output queries' hidden vectors post-decoder (`decoded` per spec §7.2.3). Compute pairwise cosine similarity across the K query outputs, averaged over the batch. Lower mean pairwise cosine = more differentiated queries; degenerate = high (e.g., > 0.95) cosine indicating queries are not learning position-distinguishing attention patterns.
   - **Final-checkpoint loss.** End-of-calibration mean loss over the final 5k steps.
6. Write per-L_d report to `results/inner_pam_v1/pre_c_decoder_calibration/L_d_{value}/calibration_report.{md,json}`.

#### 6.3.2 Selection rule

Under the framing disposition (compute is not the constraint), the selection rule is best-differentiation-within-stable, not minimal-sufficient. Among L_d values satisfying:

(a) **Stable training dynamics.** Loss curve smoothness ratio < 0.5 (the loss noise is less than half the typical loss magnitude — measured-noise-band guard rail; the 0.5 is SCAFFOLDING and recalibrated if measured noise on the first L_d's run suggests a different anchor). No NaN/Inf in checkpoints or output.

…select the L_d producing the **lowest mean pairwise output-query cosine** on the sample batch (most differentiated K queries). Tie-breaking toward smaller L_d if differentiation values are within 0.02. If no L_d satisfies (a), CC escalates to v1 design chat per §13.

**Rationale.** Narrowly missing a v1 verdict outcome because of conservative capacity selection is the failure mode the framing disposition is designed to prevent. The full L_d ∈ {1, 2, 3, 4} sweep is preserved as v2 inheritance regardless of which L_d is selected for the main run (per §6.3.1 protocol); the calibration data does not lose value when a larger L_d wins.

Cross-L_d comparison summary at `results/inner_pam_v1/pre_c_decoder_calibration/summary.{md,json}`.

#### 6.3.3 Compute envelope

Each L_d calibration is ~30k training steps on ~30k frames. At v0's ~5 minutes/10k-step pace (predictor-only training, no encoding at training time), one calibration is ~15 minutes. Four calibrations: ~1 hour. Plus the Stage A subset collection (~30k frames is ~83 loops at 360 frames/loop = ~1 hour with AI2-THOR scene init overhead). Total PRE-C ~2 hours.

#### 6.3.4 Selected value

Selected `decoder_n_layers` is written to `src/config.py` as `V1_DECODER_N_LAYERS` and imported by `inner_pam_v1_primary.py` and `inner_pam_v1_ablation1.py` at constructor time. Ablation 2 has no decoder.

### 6.4 PRE-D: Architectural property assertion verification (spec §7.2.4 / §7.3.4 / §7.4.4)

After PRE-C selects `V1_DECODER_N_LAYERS`, CC constructs an instance of each of the three predictor classes and verifies the architectural property assertions specified in spec §7.

#### 6.4.1 Primary assertions (spec §7.2.4)

1. **K output queries are per-K parameters.** `model.output_queries.numel() == K * hidden`.
2. **Cross-attention preserves K-positional structure.** On a sample batch, the decoder output `decoded.shape == (B, K, hidden)`.
3. **Per-K variance reads from K position-distinguished hidden vectors.** Compute `log_var[:, k]` gradient w.r.t. `decoded[:, j, :]` for `j ≠ k`; assert zero (per-K parameter isolation in the head).
4. **No pooled `last_token` readout.** Source-code inspection: assert no line equivalent to `last_token = x[:, -1, :]` between encoder output and per-K projection in `inner_pam_v1_primary.py`. CC inspects the file's source text for any pattern matching `\[:,\s*-1,\s*:\]` and stops if found in the predicted-step-generation path.

#### 6.4.2 Ablation 1 assertions (spec §7.3.4)

1. **K output queries preserved.** Same as Primary 1.
2. **Cross-attention preserves K-positional structure.** Same as Primary 2.
3. **Variance is parameter-shared across K.** `model.shared_log_var.numel() == 1`. All K log-variance outputs read from the same parameter.
4. **No pooled `last_token` readout.** Same as Primary 4.

#### 6.4.3 Ablation 2 assertions (spec §7.4.4)

1. **Architecture matches v0 verbatim.** `isinstance(model.output_proj, nn.Linear)` and `model.output_proj.out_features == K * (d + 1)`.
2. **Pooled `last_token` readout retained.** Source-code inspection: assert `last_token = x[:, -1, :]` exists in the forward pass.
3. **No output queries, no decoder.** `model.state_dict()` contains no parameters with names starting with "output_queries" or "decoder".

#### 6.4.4 Forward-pass smoke test

For each arm, construct a synthetic input window `(B=2, W=16, d=1024)` of random L2-normalised embeddings; run forward pass; verify:
- Output shapes match spec §7's forward-pass contracts.
- No NaN/Inf in outputs.
- log_var values are within `[LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX]`.

#### 6.4.5 Output

Write `results/inner_pam_v1/pre_d_arch_assertions/pre_d_report.{md,json}` recording per-arm assertion results and forward-pass smoke-test results. PASS verdict iff all assertions pass for all three arms. Failure stops the batch (spec §11.2 condition 3).

---

## 7. Stage A — Baseline Training per Arm

Stage A is 100k frames of unperturbed continuous-motion trajectory through the seed-7 furniture house, per spec §9.2. The same Stage A substrate is collected once per arm (re-collection at the same seed is a reproducibility verification per §11.2 condition 5).

### 7.1 Frame collection

`scripts/run_arm_{name}_collect.py` invokes the v0 `ContinuousMotionExplorer` against `data/route_phase2.json` with the PRE-B-selected mechanism *disabled* for Stage A loops. The mechanism is fired only at the Stage A → Stage B boundary; Stage A is unperturbed by construction.

**Frame budget.** 100k frames (spec §9.2). At v0's measured 360 frames/loop on the v2 substrate, this is ~278 loops. v1's PRE-B may produce a different loop length; the budget is fixed at 100k frames regardless of loop count (the spec's commitment is to frames, not loops, for Stage A).

**Annotations.** Per-frame annotations include the v0 fields (`frame_index`, `loop_index`, `phase`, `viewing_position_id`) plus the v1 `perturbation_active: bool` field — `False` for all Stage A frames.

### 7.2 DINOv2 encoding

`scripts/run_arm_{name}_encode.py` encodes the Stage A frames through frozen DINOv2-Large (fp16, ImageNet normalisation, L2-normalised output). Output: `data/arm_{name}/embeddings_stage_a.npy` (100,000 × 1024 fp32) and `data/arm_{name}/annotations_stage_a.jsonl`.

CC verifies post-encoding: shape matches collection, L2 norms in tolerance, no zero rows (v0 finding 2 default check).

### 7.3 Training

`scripts/run_arm_{name}_train.py` invokes the trainer (§4) on the Stage A embeddings with arm-specific predictor class. The trainer:
- Initialises a freshly-constructed predictor (no pretrain transfer between arms).
- Runs the §4.2 training loop on Stage A frames.
- Saves checkpoints at the §4.5 dense early cadence + standard cadence.
- Logs per-(item, ordinal) metrics at every checkpoint per §4.6 / §5.

**End-of-Stage-A checkpoint** lands at training step ~99,985 (after 15-frame `pad_start` skip per §4.3). Saved as `ckpt_end_stage_a.pt` *and* with its numeric step name (for audit-trail clarity).

### 7.4 In-flight stop conditions (spec §11.2)

CC monitors during Stage A training:
- **NaN/Inf in training** (condition 2). Loss tensor, gradients, model outputs at any step. Stop and report.
- **Architectural property assertion failure** (condition 3). Re-checked at each checkpoint by running the §6.4 assertions on the current checkpoint. (PRE-D verifies at construction; checkpoint-time re-checks guard against silent drift via gradient updates.)
- **Per-(item, ordinal) logging gap** (condition 4). If logging fails for any (item, ordinal) pair at any checkpoint, stop and report.

Per CODING_STANDARDS §9.4, additional stop conditions:
- Encoder forward produces non-L2-normalised output (configuration drift).
- Disk fills above 90% on the working volume.
- 5 failed tool calls in sequence.

### 7.5 Compute envelope per arm

Stage A: ~100k frames, ~100k training steps (one prediction per step contract). At v0's ~5 minutes per 10k steps for the ~21M-parameter v0 predictor on the 4080 Super, Stage A is ~50 minutes pure training. Including AI2-THOR collection (~3 hours), DINOv2 encoding (~30 minutes), and checkpoint I/O, end-to-end Stage A per arm is ~4–5 hours.

---

## 8. Stage B — Perturbed Training per Arm

Stage B applies the PRE-B-selected perturbation mechanism. Duration is signal-stability-calibrated per §8.1 below.

### 8.1 Stage B duration calibration (the loop-100-equivalent)

Spec §9.2 specifies "200+ loops" for Stage B and §10.2 specifies that "end of Stage B" is the canonical evaluation point, calibrated against per-(item, ordinal) signal stability.

**v1's signal-stability calibration procedure:**

1. **Minimum duration.** Stage B runs *at least* 200 loops before any termination consideration. At v0's 360 frames/loop substrate, this is ~72k Stage B frames. v1's PRE-B-measured loop length is used to convert "200 loops" → frames; CC writes `V1_STAGE_B_MIN_FRAMES` to `src/config.py` post-PRE-B.

2. **Convergence check (post-minimum).** At every checkpoint past the minimum (every 10k frames under standard cadence), CC computes per-(item, ordinal) variance drift values *relative to the end-of-Stage-A checkpoint* for the **input-varying canonical (item, ordinal) pairs only** (i.e., pairs whose canonical-window inputs differ across the Stage A → Stage B transition). Each value is the metric-2 "variance drift" defined in §5.2. Bit-identical pairs are excluded from the convergence check by design (see closing note below).

3. **Stability criterion.** Across the last 5 Stage B checkpoints, for each input-varying canonical (item, ordinal) pair, compute the max − min of the variance drift values. Signal is stable if the max − min < ε for *all input-varying* (item, ordinal) pairs simultaneously.

4. **Stability threshold ε.** Initial value **ε = 0.05 nat** (SCAFFOLDING). Anchored to: v0's variance drift values were typically 0.3–0.5 nat across the loop-30 → loop-100 window; 0.05 nat is ~12% of that magnitude, a reasonable "settled" criterion. Recalibrated against the empirical drift-trajectory shape after Primary arm completes — if Primary's drift values are systematically smaller than v0's, ε is scaled down proportionally.

5. **Maximum duration.** Hard cap at **400 loops** (~144k frames at 360 frames/loop). If signal is not stable by the maximum, CC ends Stage B at the maximum, flags in HANDOFF, and notes that the stability criterion was not met. Verdict-assignment chat (spec §11.1.4) accounts for non-converged Stage B in the verdict-pattern interpretation.

6. **Practical effect.** Per the framing-disposition allowance, generous Stage B durations are preferred. CC runs Stage B until stability is satisfied or the maximum is hit; the typical expected duration is ~250–300 loops (~90–110k frames).

**Why input-varying pairs only drive the convergence trigger.** Bit-identical (item, ordinal) pairs' variance drift trajectory *is the architectural claim being tested* — Primary and Ablation 1 predict near-zero drift on bit-identical pairs throughout training; Ablation 2 predicts v0-style coupling drift. Using bit-identical pair stability as the convergence trigger means Ablation 2's Stage B duration would be determined by the v0 pathology it is designed to reproduce, which produces unprincipled stopping (a more strongly coupled Ablation 2 would converge later, by definition). The convergence trigger must use the channel whose stability is *not* the thing being measured at end-of-Stage-B. Bit-identical pairs' values are evaluated separately as metric 5 / metric 6 outcomes at end-of-Stage-B; their value *is* the value at end-of-Stage-B, not their stability trajectory across training.

### 8.2 In-flight perturbation magnitude monitoring (spec §11.2 condition 6)

Per spec §11.2 condition 6: at every Stage B checkpoint, CC re-measures cross-stage DINOv2 cosine drop at perturbed items (re-running §6.2.3 step 4 magnitude verification). If the measured magnitude:
- Departs from PRE-B's measured value by more than **0.01**, OR
- Falls outside the [0.05, 0.10] band,

CC halts the arm and reports per spec §11.2.

**Rationale (from spec §11.2 condition 6).** v0's cross-room visual leakage finding emerged from training-time behaviour rather than preflight (substrate finding 8 in v0 closing §4); the precedent supports in-flight monitoring of perturbation magnitude as a first-class stop condition rather than trusting preflight to characterise full-experiment behaviour.

The re-measurement is cheap: 10 frames (5 items × {Stage A, Stage B canonical positions}), one DINOv2 forward pass batch, < 30 seconds at each checkpoint.

### 8.3 In-flight stop conditions (full list per spec §11.2)

All six spec §11.2 conditions apply during Stage B:

1. **Substrate verification failure** (§6.1 / §6.2 re-checks).
2. **NaN/Inf in training.**
3. **Architectural property assertion failure** (re-checked at every checkpoint).
4. **Per-(item, ordinal) logging gap.**
5. **Reproducibility failure** (cross-stage cosines drift more than 0.005 between sessions on the same seed).
6. **In-flight perturbation magnitude drift** (per §8.2 above).

Plus CODING_STANDARDS §9.4 conditions: encoder non-normalisation, disk filling, 5 failed tool calls in sequence.

Any stop triggers CC → v1 design chat handoff per spec §11.2.

### 8.4 Frame collection

Per-arm: re-launch the explorer with the PRE-B-selected mechanism enabled. CC verifies at collection start that:
- The mechanism's API call succeeds (per PRE-B's verification).
- Cross-stage cosine drop at viewing position 1 of perturbed items is in [0.05, 0.10] (the PRE-B magnitude criterion, re-confirmed at the actual collection seed).
- Annotations are written with `perturbation_active: True` for all Stage B frames.

Output: `data/arm_{name}/embeddings_stage_b.npy` (~72–144k × 1024) and `data/arm_{name}/annotations_stage_b.jsonl`.

### 8.5 Training

The trainer resumes from the end-of-Stage-A checkpoint for the same arm. Predictor weights, optimiser state, and RNG state carry across the Stage A → Stage B boundary (spec §9.2 inherits v0's no-reset-at-stage-boundary commitment).

Trainer runs through Stage B frames at the standard cadence (every 10k frames), evaluating §8.1's convergence check at each checkpoint past the minimum.

**End-of-Stage-B checkpoint** is the checkpoint at which §8.1's stability criterion is met (or at the maximum if not met). Saved as `ckpt_end_stage_b.pt` *and* with its numeric step name.

### 8.6 Compute envelope per arm

Stage B: ~72–144k frames, ~72–144k training steps. At v0's pace (~5 min per 10k steps), Stage B training is ~36–72 minutes pure compute. Including collection (~3–5 hours) and encoding (~30–45 minutes), end-to-end Stage B per arm is ~4–7 hours.

**Combined Stage A + Stage B per arm: ~8–12 hours.** Across three arms (sequential): ~24–36 hours wall-clock. This is the spec §11.5 compute estimate envelope; the framing-disposition allows the upper end as the working budget rather than the lower.

---

## 9. Per-(item, ordinal) Evaluation

After all three arms complete training, evaluation runs per spec §10.5 workflow.

### 9.1 Per-(item, ordinal) × arm matrix construction

`scripts/run_per_item_ordinal_eval.py` reads the per-arm end-of-Stage-A and end-of-Stage-B checkpoints, plus the per-checkpoint JSON logs, and constructs the per-(item, ordinal) × arm matrix per §5.3. The matrix is the primary arm-comparison artifact.

CC verifies at matrix construction:
- All three arms have valid end-of-Stage-A and end-of-Stage-B checkpoints.
- Bit-identical (item, ordinal) set is computed from pixel-MD5 comparison of canonical target frames across Stage A and Stage B per arm.
- All seven metrics are computed for every (item, ordinal) × arm cell.

### 9.2 Threshold calibration

`scripts/run_threshold_calibration.py` applies the §5.4 procedure to the matrix:
1. Build stability distribution from bit-identical (item, ordinal) values across arms.
2. Build differentiation distribution from input-varying values across arms.
3. Compute the 75th-percentile-of-stability and 25th-percentile-of-differentiation thresholds.
4. Map each (item, ordinal) × arm cell to one of: stable, non-stable, differentiated, coupling, indeterminate.
5. Write thresholds + distributions to `results/inner_pam_v1/threshold_calibration/`.

### 9.3 Verdict-pattern recognition (spec §10.4.2)

`scripts/run_arm_comparison_matrix.py` produces a verdict-pattern recommendation by matching the calibrated matrix to the spec §10.4.2 patterns:

- **V1A pattern check.** Primary column: high fraction of input-varying pairs classified as "differentiated" in both mean and variance metrics; low fraction of bit-identical pairs classified as "non-stable." Ablation 1 column: mean differentiation similar to Primary, variance differentiation weakened or absent. Ablation 2 column: high fraction of pairs (both bit-identical and input-varying) showing v0-style coupling.
- **V1B / V1C / V1D / V1E / V1P patterns** per spec §10.4.2 / §1.2 verdict categories.

The output is a *recommendation*, not a verdict. The verdict is assigned by the v1 verdict-assignment chat per spec §11.1.4 from the same matrix data.

CC writes the recommendation to `results/inner_pam_v1/verdict_recommendation/recommendation.{md,json}` with:
- The matrix data
- The calibrated thresholds
- The empirical distributions per metric
- The pattern-matching rationale (which v1A–V1P pattern best fits the matrix)
- Confidence level (high if pattern matches cleanly; low if matrix is ambiguous or heterogeneous)
- Disaggregating diagnostics potentially useful for clarification (per spec §10.5 step 3)

### 9.4 Disaggregating diagnostics (spec §10.5 step 3)

If the verdict-assignment chat or reviewer chat requests disaggregating diagnostics, CC runs one bounded round per spec §10.5 step 3. Available diagnostics (instructions-level specification):

1. **Per-K profile inspection.** For specific (item, ordinal) pairs flagged as ambiguous, render the per-K mean drift and per-K variance drift profiles (metrics 3 and 4) and compare against the v0 BCDD per-K patterns (Test C in `bcdd_results.json`). Output: per-(item, ordinal) per-K PNG.
2. **Body representation drift inspection.** For specific (item, ordinal) pairs, compute the body representation cosine (metric 7) at intermediate checkpoints (not just end-of-Stage-A vs end-of-Stage-B), tracking how the body's representation drifts across training. Output: trajectory plot per (item, ordinal).
3. **Loss-curve inspection per arm.** Loss-trajectory comparison across the three arms, surfacing arm-specific training dynamics that may explain matrix patterns.
4. **Per-loop variance trajectory per arm.** Per-(item, ordinal) variance value trajectory across Stage B loops (similar to v0's `variance_by_ordinal.json`), surfacing the path each (item, ordinal) pair takes through training.

All diagnostics characterise existing data (no retraining, no extension of training, no new data collection). The bound per spec §10.5 is explicit: this is one round, post-evaluation, pre-verdict.

### 9.5 The verdict-assignment workflow (spec §10.5)

CC's contribution to the workflow is steps 1–3 of spec §10.5 plus the recommendation in step 4 (subject to v1 design chat / verdict-assignment chat continuation per spec §11.1.4):

1. **All three arms complete training.** CC verifies via checkpoint existence and HANDOFF-recorded session entries.
2. **Per-(item, ordinal) evaluation runs across all arms.** CC produces the matrix per §9.1.
3. **One round of disaggregating diagnostics runs if ambiguous.** CC runs the §9.4 diagnostics as requested by the verdict-assignment chat.
4. **v1 chat produces verdict-assignment recommendation.** CC's role is to produce `recommendation.{md,json}` per §9.3; the verdict-assignment chat reviews it.
5. **Reviewer chat issues verdict.** Outside CC's role; spec §11.1.5.

CC does *not* issue the verdict. CC produces the matrix, the threshold calibration, the recommendation, and any requested disaggregating diagnostics, and presents these to the verdict-assignment chat.

---

## 10. Verdict Structure (reference; spec §1.2)

The v1 verdict is one of six categories per spec §1.2. CC's recommendation maps the per-(item, ordinal) × arm matrix to one of these; the verdict-issuance chat (spec §11.1.5) issues the final verdict.

### 10.1 The six verdict categories (reproduced from spec §1.2)

- **V1A — Architectural claim supported under proper configuration.** Primary co-primary differentiation; Ablation 1 isolates variance-representation contribution; Ablation 2 reproduces v0 coupling.
- **V1B — Variance representation not load-bearing.** Primary succeeds; Ablation 1 also succeeds despite scalar variance; Ablation 2 reproduces v0 coupling.
- **V1C — Architectural claim falsified under proper configuration.** Primary fails despite BCDD-indicated corrections; both ablations confirm.
- **V1D — Mixed / partial result.** Three concrete sub-patterns (per spec §10.4.2), each with a distinct matrix signature:
  - **V1D-mean-only.** Primary column shows mean differentiation matching V1A's mean pattern, but variance differentiation is absent or weakened in Primary's variance metrics. Ablations 1 and 2 show patterns consistent with V1A's expectations for their respective architectural variants. The verdict isolates a variance-specific failure on Primary despite per-K-position parameters being present.
  - **V1D-variance-only.** Primary column shows variance differentiation matching V1A's variance pattern, but mean differentiation is absent or weakened. Ablations show patterns consistent with V1A's expectations. The verdict isolates a mean-specific failure, which is architecturally surprising (the mean head has more parameters and stronger gradient signal than the variance head); verdict-assignment recommendation flags this as warranting mechanistic characterisation.
  - **V1D-heterogeneous.** Primary column shows co-primary differentiation at some (item, ordinal) pairs but not others (the Sofa-ord-1 pattern at scale — localised signal). Specific items, specific ordinals, or specific (item, ordinal) combinations succeed while others fail. The verdict-assignment recommendation includes the pattern of which (item, ordinal) succeed and which fail; this is itself the result, not a noisy V1A or V1C verdict to be cleaned up by averaging.
- **V1E — Perturbation-as-sufficient.** All three arms produce co-primary differentiation; architectural corrections were not load-bearing.
- **V1P — Protocol-failure outcome.** Substrate failure prevents architectural verdict.

### 10.2 v2 design implications per verdict

Per spec §1.2 and the v2 design intake (`WEFT_INNER_PAM_v2_DESIGN_INTAKE.md` §6), each verdict produces different v2 design implications. CC does not engage with v2 design implications during v1 execution; the verdict-assignment chat and v2 design chat handle that downstream.

### 10.3 Resolution timing (spec §1.2.1)

Verdict is assigned at a specific workflow point, not iteratively. Spec §10.5 specifies the 5-step workflow. Post-verdict diagnostics are v2 intake material, not v1 verdict revision (spec §10.5 closing paragraph).

---

## 11. Scaffolding Inventory

| label | item | location | removal plan |
|---|---|---|---|
| ARCHITECTURE | Three-arm structure (Primary, Ablation 1, Ablation 2) | spec §7, instr §1.4 | inherited to v2 verdict-assignment as v1's experimental design |
| ARCHITECTURE | Co-primary mean and variance differentiation | spec §1.4 | inherited |
| ARCHITECTURE | Per-(item, ordinal) evaluation granularity | spec §10 | inherited |
| SCAFFOLDING | `K = 16`, `W = 16` | spec §7.1, instr §3 | v2 may revisit alongside body-family change |
| SCAFFOLDING | Predictor hidden=512, n_layers=4, n_heads=8, mlp_dim=2048 | spec §7.1 | v2 may revisit |
| SCAFFOLDING | `V1_DECODER_N_LAYERS` (PRE-C calibrated) | spec §7.2.1, instr §6.3 | re-calibrate if substrate or capacity changes; v2 may revisit |
| SCAFFOLDING | `lr=3e-4`, `weight_decay=0.01`, AdamW | spec §9.4, instr §4.4 | v2 if V1C/V1D/V1P |
| SCAFFOLDING | grad clip max_norm=1.0 | instr §4.4 | revisit if needed |
| SCAFFOLDING | log_var clamp `[LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX]` | spec §7.1, instr §3 | inherited from v0 (revisit if calibration poor) |
| SCAFFOLDING | Stage A frame budget = 100k | spec §9.2, instr §7.1 | v2 may revisit |
| SCAFFOLDING | Stage B min = 200 loops, max = 400 loops | instr §8.1 | recalibrate after first arm if stability behavior different than expected |
| SCAFFOLDING | Stage B stability threshold ε = 0.05 nat | instr §8.1 | recalibrate after Primary arm's drift-trajectory inspection |
| SCAFFOLDING | Verdict threshold: 75th-percentile of bit-identical / 25th-percentile of input-varying | spec §10.4.3, instr §5.4 | recalibrate post-first-run if distribution produces degenerate verdict |
| SCAFFOLDING | PRE-C selection: among L_d passing loss-smoothness < 0.5, pick lowest mean pairwise query-cosine (tie-break smaller L_d within 0.02) | instr §6.3.2 | recalibrate against empirical PRE-C distribution |
| SCAFFOLDING | Perturbation magnitude band [0.05, 0.10] | spec §1.2, §8.2.1 | inherited; v2 may revisit alongside body-family change |
| SCAFFOLDING | In-flight perturbation magnitude drift tolerance 0.01 | spec §11.2 cond 6 | recalibrate against empirical PRE-B distribution |
| SCAFFOLDING | Reproducibility tolerance 0.005 | spec §8.3.5, §11.2 cond 5 | inherited from v0 (revisit if AI2-THOR behavior changes) |
| SCAFFOLDING | Locality tolerance 0.015 | spec §8.2.1 | inherited from v0 §5.8; revisit if v1 mechanism produces different locality regime |
| SCAFFOLDING | Substrate verification §5.1 threshold 0.75 | spec §6.1, instr §6.1 | inherited from v0 |
| SCAFFOLDING | Substrate verification §5.2 threshold 0.60 | spec §6.1, instr §6.1 | inherited from v0 |
| SCAFFOLDING | Checkpoint cadence (dense early then every 10k) | instr §4.5 | inherited from v0; revisit if dynamics warrant |
| SCAFFOLDING | Finding 3 cosine threshold = 0.999 (v0-inherited, structural-handoff-tolerant) | instr §6.1.3, `FINDING_3_COS_MAX` in `v1/src/config.py` | v2 run-length-aware check candidate (`WEFT_INNER_PAM_v2_DESIGN_INTAKE.md`); recalibrate if substrate changes the handoff pattern |

The scaffolding inventory is reviewed at start and end of each session per research_operations §7.2.

---

## 12. Operational Discipline

### 12.1 Per CODING_STANDARDS and research_operations

CC inherits all CODING_STANDARDS rules (§1–§10) and research_operations rules (§§1–16) by default. v1-specific additions below; otherwise see those documents.

### 12.2 Push hold

Push hold remains in effect through v1 execution. Repository commits land locally; nothing pushes to remote until v1 verdict is issued and v1 closing document is produced. Push hold lift is a deliberate decision made jointly by v1 design chat, reviewer chat, and the human researcher, after the verdict (spec §11.4).

### 12.3 HANDOFF.md updates

Every CC session ends with a HANDOFF.md entry per spec §11.3:
- Session goals and what was attempted.
- Code commits in this session.
- Substrate verification status (which PRE-A/B/C/D checks ran, what results).
- Stop conditions encountered (if any) and resolution.
- Outstanding questions for v1 design chat or reviewer chat.

HANDOFF.md is the canonical institutional-memory artifact across CC sessions. Verdict-load-bearing decisions and diagnostic findings flow through HANDOFF.md to v1's closing document.

### 12.4 Numbers trace to files

Every number in HANDOFF or any summary traces to its source JSON / log file before being quoted. Per CODING_STANDARDS §5.5 and research_operations §4.1. Post-compaction summaries are re-verified.

### 12.5 Git commits per task

Suggested v1-specific commits in chronological order:

- `infra(v1): bootstrap src/predictor/inner_pam_v1_*.py, src/preflight/, src/eval/`
- `feat(predictor-v1): InnerPAM_v1_Primary with K output queries and per-K scalar variance`
- `feat(predictor-v1): InnerPAM_v1_Ablation1 with shared scalar log-variance`
- `feat(predictor-v1): InnerPAM_v1_Ablation2 inheriting v0 InnerPAM`
- `feat(trainer-v1): online_trainer_v1.py with stage-A→stage-B per-arm loop`
- `feat(eval-v1): per_item_ordinal_metrics.py for the seven §10.3 metrics`
- `feat(preflight): pre_a_substrate, pre_b_perturbation, pre_c_decoder_calibration, pre_d_arch_assertions`
- `exp(pre_a): substrate verification on v1, all 8 checks PASS, report at <hash>`
- `exp(pre_b): perturbation mechanism preflight, all 4 candidates characterised, selected <mechanism>, report at <hash>`
- `exp(pre_c): decoder layer calibration L_d ∈ {1,2,3,4}, selected L_d=<value>, report at <hash>`
- `exp(pre_d): architectural property assertions all PASS for 3 arms, report at <hash>`
- `exp(arm_primary): stage A 100k frames + stage B <N> loops, ckpt at <hash>`
- `exp(arm_ablation1): stage A + stage B, ckpt at <hash>`
- `exp(arm_ablation2): stage A + stage B, ckpt at <hash>`
- `exp(eval): per-(item, ordinal) × arm matrix, threshold calibration, verdict recommendation at <hash>`

Per CODING_STANDARDS §2.5–2.6.

### 12.6 Single CC session per working tree

Per CODING_STANDARDS §4.3 and research_operations §9.5.

### 12.7 Sub-agents

Bounded tasks may be delegated to a sub-agent with minimal context per CODING_STANDARDS §4.1–4.2. Candidate sub-agent tasks for v1: parameter-count verification at PRE-D; matrix construction at evaluation; per-K profile diagnostic rendering.

### 12.8 Arm boundaries are session boundaries

Each arm's end (Stage A complete; Stage B complete; arm transition) is a session boundary. Per research_operations §9.2. End-of-arm produces a HANDOFF entry, a git commit, and a clean session start for the next arm.

### 12.9 Single-variable discipline reminder (spec §11.5)

v1 is structured to test specific architectural interventions with attribution clean across arms. Mid-experiment scope expansion (adding a fourth arm; changing perturbation mechanism after PRE-B; adding architectural variants) violates single-variable discipline and is a stop condition triggering v1 design chat handoff.

If implementation surfaces a "we'd want this" instinct that touches scope expansion, write a note in HANDOFF for the v2 design chat and continue with v1's locked scope.

---

## 13. Handoff Protocol

At the end of each session, before ending the CC chat:

1. `git status` clean (or pending changes committed).
2. `requirements.txt` current; `pip freeze > .env_snapshot.txt` committed if changed.
3. HANDOFF.md updated per §12.3.
4. Any running jobs: PID, log path, expected completion time recorded.
5. Scaffolding inventory reviewed (start and end of session per research_operations §7.2).
6. Regression check: did any prior verification value degrade? If yes, flag.
7. Numbers in HANDOFF verified against source files.

Per research_operations §14 and CODING_STANDARDS §10. Sessions ending without these steps have to be reconstructed from git history and guesswork — do not end sessions that way.

---

## 14. Review Cycle Before Implementation Begins

Per spec §11.1.2 and §12.2:

1. v1 design chat (current chat) signs off on this document.
2. Document is submitted to v1 reviewer chat (separate context) per research_operations §2.2 adversarial review.
3. Findings from review resolved or explicitly overridden in writing.
4. Revised version committed.
5. CC implementation begins (PRE-A → PRE-B → PRE-C → PRE-D → Primary → Ablation 1 → Ablation 2 → evaluation) only after step 4.

---

## 15. Drift Detection — Project-Specific

Beyond the universal checks in research_operations §15 and the v0 inherited checks (drift back to next-frame prediction; cosine retrieval becoming the headline mechanism; aggregate-only reporting; threshold recalibration without justification; "while we're at it" scope expansion; encoder-substrate failure misread as architecture failure), v1 watches for:

- **Drift back to v0's pooled-readout pattern.** If a refactor or simplification of Primary's predictor inadvertently re-introduces pooled-vector readout (e.g., a "simplification" that mean-pools the decoder output before per-K projection), that is drift back to the BCDD-identified pathology v1 was designed to fix. Stop and report.

- **Aggregate-only verdict reporting.** Per spec §5 / §10, all verdict-load-bearing decisions are made at per-(item, ordinal) granularity. Any HANDOFF or recommendation that quotes only aggregate metrics without the per-(item, ordinal) disaggregated forms is a process violation per CODING_STANDARDS §6.3 and research_operations §3.5.

- **Threshold recalibration without justification.** SCAFFOLDING thresholds (PRE-C selection criteria, Stage B stability ε, verdict percentile choices) may be recalibrated after observing empirical distributions, but with explicit reasoning per research_operations §16. Silent recalibration is a process violation.

- **Mid-experiment arm modification.** Adding architectural variants, changing perturbation mechanism, modifying loss formulation mid-experiment — all violate spec §11.5 single-variable discipline. Stop and report; do not silently adapt.

- **Skipping the bit-identical pair identification step.** The §5.1 pixel-MD5 check is load-bearing for §5.4 threshold calibration (the stability reference distribution requires the bit-identical set). If the check is skipped or shortcut, the threshold calibration is malformed. CC verifies the bit-identical set was computed before running threshold calibration.

- **Treating Ablation 2 as an architectural failure mode.** Ablation 2 is designed to reproduce v0's coupling pattern under v1's substrate. If Ablation 2 *does* reproduce v0 coupling, that is the expected V1A / V1B / V1C / V1D outcome — not a bug. If Ablation 2 does *not* reproduce v0 coupling, that is the V1E pattern's signal, which is a verdict outcome — not an indication that Ablation 2 was implemented incorrectly. The PRE-D architectural property assertions verify Ablation 2 matches v0 verbatim; trust those.

- **Substrate-as-feature-vs-bug interpretive errors** (v0 finding 4). v0's session-4 jitter proposal was an interpretive error: across-loop pose-determinism in Stage A was substrate-as-feature (the curriculum's baseline state), not substrate-as-bug requiring jitter. v1 inherits the discipline: when a substrate property is observed during preflight or training, classify as substrate-as-feature by default and require positive evidence of bug-character before remediating.

---

## 16. Open Decisions for Execution-Time

These are deliberately unresolved here; CC decides with documentation per CODING_STANDARDS §9.2:

- **Specific PRE-B-selected perturbation mechanism.** Selected by §6.2.4 selection rule from the four candidates.
- **Specific PRE-C-calibrated `decoder_n_layers` value.** Selected by §6.3.2 selection rule from {1, 2, 3, 4}.
- **Specific Stage B duration per arm.** Determined by §8.1 signal-stability convergence; written to HANDOFF as the per-arm end-of-Stage-B step.
- **Whether disaggregating diagnostics are needed.** Decided at the verdict-assignment chat per spec §10.5 step 3.
- **Specific replacement asset for mechanism 2 (asset replacement) if selected** (the v0 Phase 3 preferred mechanism's question carries forward).
- **Specific frame indices for the canonical (item, ordinal) targets per arm.** Computed from annotations at evaluation time.
- **Whether to re-run a single arm if its end-of-Stage-B reports unconverged.** Decided at the verdict-assignment chat after inspecting the empirical drift-trajectory shape.

Each documented in HANDOFF when chosen.

---

## 17. Compute and Wall-Clock Estimate

Total wall-clock estimate, broken out:

| step | wall-clock |
|---|---|
| PRE-A substrate verification | ~30 minutes |
| PRE-B perturbation mechanism preflight (4 candidates) | ~3–4 hours |
| PRE-C decoder layer calibration (4 L_d values + Stage A subset collection) | ~2 hours |
| PRE-D architectural property assertions | ~15 minutes |
| Primary arm: Stage A (100k frames) + Stage B (~250 loops) | ~10 hours |
| Ablation 1 arm: Stage A + Stage B | ~10 hours |
| Ablation 2 arm: Stage A + Stage B | ~10 hours |
| Per-(item, ordinal) evaluation + matrix construction + threshold calibration | ~30 minutes |
| Disaggregating diagnostics (if needed) | ~1 hour |
| **Total** | **~37–38 hours** |

This sits well inside the framing-disposition allowance (compute is not the constraint). Generous calibration runs are preferred to minimal ones; the upper end of the per-arm estimate (~14 hours) is the working budget.

Per CODING_STANDARDS §5.2, all training and collection scripts are launched with `nohup`. CC polls logs per §5.3 and does not block on stdout.

---

*End of v1 experiment instructions, first draft. Adversarial review per spec §11.1.2 and research_operations §2.2 follows. CC implementation begins after reviewer sign-off.*

*Source: `WEFT_INNER_PAM_v1_Spec_pass1_sections_1_to_6.md` (architecture and claims), `WEFT_INNER_PAM_v1_Spec_pass2_sections_7_to_11.md` (implementation specification), v0 institutional memory (`WEFT_INNER_PAM_v0_CLOSING.md`, `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md` as discipline reference), operational discipline from `CODING_STANDARDS.md` and `research_operations_v1.md`, BCDD evidence (`bcdd_results.json`), repo state (`HANDOFF.md`).*
