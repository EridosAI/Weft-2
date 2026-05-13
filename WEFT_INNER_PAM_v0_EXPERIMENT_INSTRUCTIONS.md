# Weft Inner PAM v0 — Experiment Instructions

**Status:** Fourth draft. Audience: CC (Claude Code), executing autonomously per `CODING_STANDARDS.md` and `research_operations_v1.md`. Adversarial review complete; this draft incorporates the resulting fixes.

**Purpose:** Specify the v0 experiment that tests the architectural claims in `WEFT_INNER_PAM_v0_Spec.md`. This document tells CC *what to build* and *what to run*; the spec tells CC *what the architecture is* and *why*. If this document and the spec disagree, the spec wins and this document is wrong — flag in `HANDOFF.md` and stop.

**Read order:**
1. `WEFT_INNER_PAM_v0_Spec.md` (architecture)
2. `CODING_STANDARDS.md` (operational discipline)
3. `research_operations_v1.md` (process discipline)
4. `HANDOFF.md` (current state)
5. This document

**Push hold remains in effect for both repos throughout this batch.**

**Changes from first draft (applied in second draft).** Five fixes applied before adversarial review:
- G1.4 gate replaced with statistical significance test (no magic threshold).
- Evaluation commits to running at checkpoint boundaries with training paused; running parallel to training is not relied upon.
- Shuffle control retained, with explicit sanity-check diagnostic added.
- Phase 2/3 frame budgets derived from repetition-bin coverage arithmetic.
- Phase 2/3 perturbation mechanism revised to use only verified-supported AI2-THOR API, with explicit preflight verification and fallback hierarchy.

**Changes from second draft (applied in third draft).** One correction following experiment-chat review:
- Phase 2 and Phase 3 frame budgets raised from 50k to 65k each. The 50k budget left 99 trained loops after the 10-loop held-out, which sits at the upper edge of the 51–99 repetition bin and never enters 100+. The 100+ bin would have been empty at every Phase 2 / Phase 3 eval checkpoint — invalidating the strongest test of spec §2.2 (repetition-as-learning-signal). 65k yields 132 trained loops, ~32 reps into the 100+ bin. Bank cap raised from 200k to 250k to preserve the no-eviction-in-v0 property given the higher total stream (230k). Checkpoint schedule extended with one additional checkpoint at phase-relative step 55k to capture the new 100+ region.

**Changes from third draft (applied in this fourth draft).** Eight items following adversarial review plus experiment-chat catch:
- Phase 2 two-item perturbation framing added to G2.2 / G2.3 and M6 (§6.2, §8.7) — explicit acknowledgement that accommodation is measured against a joint Dresser+Sofa perturbation, not a single-item axis.
- Phase 2 frame-budget robustness parenthetical added (§8.3) — CC re-computes the budget from collected annotations if held-out is widened or actual loop length exceeds 470 frames.
- Wilcoxon signed-rank fallback added to G1.4, G2.3, G3.3 (§7.7, §8.7, §9.7) — used if Shapiro-Wilk on the per-probe difference distribution rejects normality at p < 0.05. Same one-sided structure, same p < 0.01 threshold.
- Three-point τ sensitivity diagnostic added to M5 and the developmental arc (§6.2, §10) — predictor-only fraction reported at the nominal τ (5k–10k median), τ at 20k, and τ at the final Phase 1 checkpoint.
- Quantitative S4 collapse-to-mean thresholds added (§6.5) — labelled SCAFFOLDING with explicit recalibration commitment from the empirical shuffle distribution.
- Bank cap kept at 250k; eviction-disabled flag with hard stop added (§13.1, §13.4) — catches any overrun loudly rather than allowing silent eviction to pollute M5.
- G1.5 converted from absolute-threshold gate to trajectory criterion (§7.7) — M3 must rise across Phase 1 *and* clear the 0.10 floor. This tests §2.2 directly rather than betting on the floor value alone.
- **G1.4 horizon corrected from k=1 to k=8** (§7.7) — k=1 is reachable by a next-frame predictor and does not test the path-prediction claim §2.2 actually makes. k=8 is the principled mid-horizon (clearly multi-step, inside the trained window, highest signal-to-noise). k=1 and k=16 retained as diagnostic context but not gated.
- **Eval probe-count arithmetic corrected in §6.4** — was 50 (off by 10×); actual is 500 per checkpoint (250 steady-state + 250 cue). Total eval overhead estimate updated: ~34 checkpoints × ~2 min ≈ ~1 hour across the experiment. Stale Phase 2 checkpoint schedule in §8.5 also updated to match §4.6 (was missing the 55k step added in the third draft).

---

## 0. Environment Header

| field | value |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| Shell | bash |
| Python | 3.12.3 (matches the substrate-verification batch's verified-working environment in the previous repo; earlier drafts incorrectly listed 3.10 — corrected 2026-05-13 per HANDOFF session 1) |
| PyTorch | 2.10.0+cu128 (CUDA 12.8 via WSL2 passthrough), pinned in `requirements.txt` |
| Encoder model | `facebook/dinov2-large` (loaded via `transformers`) |
| Host CPU | Intel Core i9-14900K @ 3.20 GHz |
| Host RAM | 64 GB |
| GPU | RTX 4080 Super, 16 GB VRAM (local) |
| Host disk | 1.23 / 1.82 TB used (≈ 590 GB free at batch start; see §0.1 disk budget) |
| Working repo | `/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/` |
| Reference repo | `/mnt/c/Users/Jason/Desktop/Eridos/Weft/` (read-only; reused for AI2-THOR explorer, frame collection scripts, seed-7 reproducibility) |
| Active venv | none; uses system Python 3.12.3 directly, matching the previous repo's verified-working pattern. A `.venv` may be added later if isolation is needed (see HANDOFF session 1 for the deferred decision) |
| Required env var | `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000` |
| AI2-THOR version | `ai2thor==5.0.0` (matches previous repo; loaded only for Phase 2/3 collection) |

CC verifies the environment matches before any training. Divergence is documented in `HANDOFF.md` and resolved before proceeding.

### 0.1 Host-session protection (local hardware run)

This batch runs on local hardware over an estimated ~20–40 hours of wall-clock across multiple sessions. The host shell is a single point of failure: if the terminal closes, WSL2 stops, the laptop sleeps, or Windows applies an update, **every running collection and training job dies with the shell** — even though individual scripts are launched with `nohup` per CODING_STANDARDS §5.2. The `nohup` pattern protects against shell hangup signals; it does not protect against the parent shell being terminated.

Host-session protection is therefore part of the run setup, not optional:

1. **Use `tmux` for the CC session itself.** Before launching CC, start a named tmux session inside WSL2:
   ```bash
   tmux new -s weft
   ```
   Launch CC inside this tmux session. Detach with `Ctrl-b d` (CC keeps running); reattach with `tmux attach -t weft` to watch live output. tmux is `sudo apt install tmux` if not already present.

2. **Disable Windows sleep and disk sleep** for the duration of the batch. Settings → System → Power → Screen and sleep → set "When plugged in, put my device to sleep after" to **Never**. Also under "Additional power settings" → "Change plan settings" → "Change advanced power settings" → "Hard disk" → set "Turn off hard disk after" to **0 (Never)**. WSL2 freezes when the Windows host sleeps; a frozen WSL2 stops all CC activity even though the session is technically still alive.

3. **Defer Windows updates** for the batch window. Settings → Windows Update → Pause updates for at least the planned batch duration. An auto-applied update with a forced reboot will kill the entire run mid-phase with no graceful exit.

4. **Disk budget check before Phase 1 starts.** Estimated disk usage across the full batch:
   - Phase 1 frames: 100k PNG × ~50 KB ≈ 5 GB
   - Phase 2 frames: 65k × ~50 KB ≈ 3.3 GB
   - Phase 3 frames: 65k × ~50 KB ≈ 3.3 GB
   - Embeddings (230k × 1024 × fp32): ≈ 940 MB
   - Checkpoints (predictor weights ~84 MB × ~34 checkpoints + bank state): ≈ 8–15 GB
   - Logs, TensorBoard runs, results JSON: ≈ 1–3 GB
   - **Total: ~25 GB.** Well within the ~590 GB free, but CC verifies `df -h` returns ≥50 GB free at every phase boundary and stops if it doesn't.

5. **GPU lockout.** The RTX 4080 Super is the only GPU. Concurrent GPU-using applications (games, video editing, other ML jobs) will cause CUDA OOM during training and silent fp16 contention during encoding. Close them before starting. CC does not need to verify this — the experiment chat does, before launching.

CC's responsibility within this protection layer: launch all long-running scripts with `nohup`, poll logs, and continue to honour the §5.2 / §5.3 patterns. The host-side protections above are the experiment chat's responsibility before CC is launched, and are re-verified at the start of each new session (alongside reading HANDOFF.md per CODING_STANDARDS §1.2).

---

## 1. Scope Lock and Locked Decisions

These are settled. They do not get re-litigated in this batch.

### 1.1 Architectural commitments (from spec §§1–6)
- Shape-as-unit learning via path prediction with Gaussian NLL (spec §4.1).
- Continuous time, single-pass training (spec §2.3).
- Always-on Inner PAM (spec §2.4).
- Shapes live in predictor weights; instances live in the bank (spec §2.7).
- Confidence-graded mixing of shape recall and instance retrieval (spec §2.8).
- Modality-agnostic predictor output: centreline `(K, d)` + per-step scalar log-variance `(K, 1)` (spec §3.3).

### 1.2 Encoder choice
- DINOv2 ViT-L/14 CLS token, frozen, fp16 eval, ImageNet mean/std normalisation, L2-normalised output. Verified per spec §5 (Check 2 = 0.4422, Check 1 jittered = 0.9260, gap = 0.4838 — all PASS).
- The encoder is not re-trained, re-tuned, or replaced in this batch.

### 1.3 Environment and trajectory

**Substrate revised in session 4** (per `HANDOFF.md` session-4 entry). The original entry inherited a 30-frame static-dwell pattern from the prior Stage-0b experiment; that pattern violates spec §2.3's "no zero-velocity frames" commitment (and the architectural claim that the predictor learns path-shaped targets, not identity targets repeated K times). The session-3 reviewer surfaced the substrate-architecture mismatch and authorised the redesign. Current entry reflects the new substrate:

- AI2-THOR seed-7 furniture house, driven by `src/env/continuous_motion_explorer.py`.
- Five-item route: `Bed → DiningTable → Dresser → Sofa → Television → Bed → ...` (item identities unchanged from the prior Stage-0b run).
- **Trajectory: continuous motion throughout.** No held-pose dwell. Each item gets a "close-up" segment: a straight 2 m densified path (0.20 m step → ~10-12 frames) passing through the item's viewing position perpendicular to the item-facing heading, with heading locked at the item-facing bearing so the item enters the frame from one side, centres at the apex, slides out the other side. Transit between items is NavMesh-densified at 0.20 m with corner rotations at 5° (mechanism unchanged from prior explorer).
- **Loop length:** ~316 frames per loop empirically (5-loop calibration, 2026-05-13). Verified non-bit-identical at the consecutive-frame level inside motion phases (close_up→close_up and transit→transit both have 0 / >1500 bit-identical pairs).
- **Tempo:** consistent across loops (spec §6.9). The motion is deterministic given seeds; cross-loop pose variation is supplied by the variation mechanism specified below.
- **Cross-loop variation.** Continuous motion within a loop is necessary but not sufficient on its own — apex poses across loops at the same item are bit-identical absent some perturbation (AI2-THOR renders deterministically at a fixed pose). **Variation comes from the phase structure itself, not from pose jitter.** Phase 1's Stage A loops are intentionally identical (the baseline state of the curriculum); Stage B onward introduces per-loop variation via `RandomizeMaterials` on the LivingRoom items (Dresser + Sofa); Stage C adds the Phase 3 perturbation on top. Across-loop apex bit-identicity at items 1–5 in Stage A is the curriculum working correctly, not a substrate degeneracy. The Phase 1 within-loop static dwell that needed fixing (a §2.3 violation) is fixed by continuous motion; across-loop pose-determinism is not a separate violation and does not need agent-pose, furniture-position, or per-frame jitter to break it. (Decision recorded 2026-05-14 by experiment-chat review of the session-4 calibration findings, superseding the session-4 jitter proposal. See HANDOFF session-5 entry.)
- **Room composition** (verified from the seed-7 annotations): **LivingRoom** contains Dresser and Sofa; **Bedroom** contains Bed, DiningTable, and Television.

### 1.4 Phase structure (the four locked decisions, with the curriculum framing recorded 2026-05-14)

The v0 experiment is a **Stage A → Stage B → Stage C curriculum on a single continuously-trained predictor**. The architectural claim is that the predictor learns each item's trajectory as a structure in embedding space (Stage A: line → Stage B: tube around the line → Stage C: manifold around the line) and that recall remains coherent as the structure widens. The phase structure already supplies the variation gradient; no additional jitter or per-frame perturbation is needed.

- **Continuous training**: training runs without pausing across all three phases at the level of the *training loop*. Evaluation pauses training at checkpoint boundaries (§6.4 below); it does not run in parallel. Predictor weights and memory bank carry across phase boundaries without reset (spec §2.3). Phase 1 is discarded as the substrate-degenerate baseline (session-4 disposition), so Phase 2 starts from a *freshly-initialised* predictor; Phase 3 then resumes from Phase 2's end checkpoint.

- **Phase 2 internal structure — Stage A baseline before Stage B perturbation begins.** The first **30 loops** of Phase 2 run with `RandomizeMaterials` **disabled** (Stage A within Phase 2 — pure continuous-motion trajectory through an unperturbed environment; this is the architectural baseline for "identical loops produce identical trajectories", and the bit-identicity observed in session-4 calibration is Stage A working correctly). From **loop 31 onward**, `RandomizeMaterials(inRoomTypes=["LivingRoom"], useTrainMaterials=True)` is applied at the start of every loop, supplying fresh LivingRoom textures per loop (Stage B). Two items are perturbed (Dresser + Sofa together) per the LivingRoom scope; Bed, DiningTable, and Television remain unperturbed in-phase as within-experiment control items. The single-item axis originally intended for Phase 2 is weaker than planned (§8.7 records the joint-perturbation framing).

- **Phase 3 internal structure — no separate Stage A baseline.** Phase 3 starts immediately with the Phase 3 perturbation active from loop 1. The Phase 2 → Phase 3 transition is itself the Stage B → Stage C step in the curriculum; an internal Stage A within Phase 3 would dilute the Stage C signal without diagnostic value. Phase 2's LivingRoom perturbation remains active throughout Phase 3 (in the preferred asset-replacement mechanism) or is overwritten by full-house retexturisation (in the fallback).

- **Phase 3 perturbation — additive over Phase 2.** Preferred mechanism: Television asset replacement-in-place at the same teleport coordinates (e.g., replace Television with ArmChair via `RemoveFromScene` + `PlaceObjectAtPoint` or equivalent procedural placement). Fallback (if asset replacement does not pass preflight verification in §9.2): full-house `RandomizeMaterials()` so that all 5 items are re-textured. The asset-replacement path is preferred because it tests "is the shape the route or the visuals" at a qualitatively different level than texture variation; the fallback preserves the load progression (more positions perturbed) but loses the qualitative distinction.

- **Frame reuse**: Phase 1 trains on the existing 100k DINOv2 embeddings at `data/dinov2_embeddings/embeddings.npy` (substrate-degenerate baseline; kept for audit; not re-run). Phase 2 and Phase 3 require fresh collection and encoding on the continuous-motion substrate.

The original ordering rationale (weak-then-strong) survives the revision: Phase 2 is the weaker visual perturbation (texture change on 2 items), Phase 3 is the stronger (texture change on 2 items + asset change on a third, or texture change on all 5 in the fallback case).

### 1.5 What NOT to change
- The architecture spec. If a problem encountered during implementation suggests the spec is wrong, stop and report — do not edit the spec.
- The encoder (DINOv2 ViT-L/14 CLS, frozen).
- The loss formulation (Gaussian NLL with isotropic per-step scalar variance, uniform weighting across K steps, stop-gradient on targets).
- `K = 16`, `W = 16`. SCAFFOLDING per spec §9.5 but locked for this batch.
- Bank as append-only with recency-based FIFO eviction.
- Confidence-thresholded mixing (single τ, M=3 steps for confidence aggregation).
- The cosine and shuffle controls (both required, both run per phase; shuffle has an explicit sanity-check diagnostic in §6.5).
- Per-shape / per-position / repetition-stratified disaggregations before aggregate metrics.
- Numbers trace to files. No remembered numbers. No mental arithmetic on metrics.
- **Dwell as pause is not part of the architecture. The agent moves continuously.** Any session that re-introduces a held-pose dwell, a "stand still and look at X" segment, or any other zero-velocity frame is reverting the session-4 substrate change and re-creating the substrate-degenerate baseline. (Added 2026-05-13 after the inherited 30-frame static dwell was caught in session 3. See `HANDOFF.md` session-4 entry and the drift-detection note in `research_operations_v1.md` §15.)
- **No agent-pose jitter, no furniture-position jitter, no per-frame perturbation of any kind.** Variation comes from the phase structure itself (Stage A → Stage B → Stage C curriculum, §1.4). The bit-identical across-loop apex frames observed in session-4 calibration are Stage A working correctly, not a substrate degeneracy. A jitter parameter, a noise term in the explorer's pose computation, or a stochastic offset added inside the env wrapper would re-introduce a variation source the curriculum is explicitly designed not to need. (Added 2026-05-14 after the session-4 jitter proposal was withdrawn by the experiment chat in favour of the phase-structure-as-variation framing. See `HANDOFF.md` session-5 entry.)
- **No Stage A baseline inside Phase 3.** Phase 3 starts immediately with the Phase 3 perturbation active from loop 1; the Phase 2 → Phase 3 transition itself is the Stage B → Stage C step in the curriculum. Inserting an unperturbed segment at the start of Phase 3 would dilute the Stage C signal. (Added 2026-05-14 with the curriculum framing.)
- **Predictor and bank carry across phase boundaries without reset** (spec §2.3). Phase 3 resumes from Phase 2's final checkpoint and final bank state; weights and bank are not re-initialised at the boundary. (Phase 1 → Phase 2 is the exception: Phase 1 is discarded as substrate-degenerate, so Phase 2 starts from a freshly-initialised predictor with an empty bank.)

### 1.6 What is deferred
All items in spec §6 (variable tempo, multi-scale operation, depth modulation beyond recency, sleep-phase consolidation, bi-hemispheric retrieval, action/reward, outer-JEPA mediation logic, etc.). If implementing v0 surfaces a "we'd want this" instinct that touches any of those, write a note in `HANDOFF.md` for the v1 conversation and continue with v0.

---

## 2. Repository Layout

CC builds out this structure. Establish before file count grows (per `CODING_STANDARDS.md` §2.1).

```
Weft 2/
├── WEFT_INNER_PAM_v0_Spec.md                      (exists)
├── WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md   (this doc)
├── CODING_STANDARDS.md                            (exists)
├── research_operations_v1.md                      (exists)
├── HANDOFF.md                                     (exists, updated per session)
├── requirements.txt                               (pinned per §8 of CODING_STANDARDS)
├── .gitignore
├── src/
│   ├── encoder/
│   │   └── dinov2_encoder.py     # frozen DINOv2 forward, fp16, L2-norm
│   ├── memory/
│   │   └── memory_bank.py        # append-only bank, FAISS IndexFlatIP
│   ├── predictor/
│   │   └── inner_pam.py          # the path predictor (§3 below)
│   ├── trainer/
│   │   └── online_trainer.py     # online single-pass loop (§4 below)
│   ├── mixing/
│   │   └── recall_mixer.py       # confidence-thresholded mixing (§5 below)
│   ├── eval/
│   │   ├── probes.py             # probe construction (§6 below)
│   │   ├── metrics.py            # M1..M7 metrics
│   │   └── controls.py           # cosine and shuffle baselines + shuffle sanity check
│   ├── env/
│   │   ├── explorer_phase1.py    # imports existing seed-7 explorer (Phase 1 uses cached frames; not needed at training time)
│   │   ├── explorer_phase2.py    # LivingRoom RandomizeMaterials wrapper
│   │   └── explorer_phase3.py    # adds Phase 3 perturbation on top of explorer_phase2
│   └── config.py                  # all hyperparameters, with ARCHITECTURE/SCAFFOLDING labels
├── scripts/
│   ├── run_phase1_train.py
│   ├── run_phase1_shuffle.py
│   ├── run_phase2_preflight.py   # API verification before collection (§8.2)
│   ├── run_phase2_collect.py
│   ├── run_phase2_encode.py
│   ├── run_phase2_train.py
│   ├── run_phase3_preflight.py   # API verification before collection (§9.2)
│   ├── run_phase3_collect.py
│   ├── run_phase3_encode.py
│   ├── run_phase3_train.py
│   └── run_eval.py
├── data/
│   ├── seed7_furniture_frames/                    (exists, gitignored, ~5.2 GB)
│   ├── seed7_dinov2_stability_frames/             (exists, gitignored)
│   ├── seed7_dinov2_stability_annotations.jsonl   (exists)
│   ├── dinov2_embeddings/embeddings.npy           (exists, 391 MB, gitignored)
│   ├── phase2_frames/                             (created in Phase 2, gitignored)
│   ├── phase2_embeddings/                         (created in Phase 2, gitignored)
│   ├── phase3_frames/                             (created in Phase 3, gitignored)
│   └── phase3_embeddings/                         (created in Phase 3, gitignored)
├── results/
│   ├── encoder_verification/                      (exists)
│   ├── encoder_verification_dinov2/               (exists)
│   ├── encoder_verification_dinov2_stability/     (exists)
│   ├── frame_rerender/                            (exists)
│   └── inner_pam_v0/
│       ├── phase1_main/
│       ├── phase1_shuffle/
│       ├── phase2_preflight/
│       ├── phase2_main/
│       ├── phase3_preflight/
│       ├── phase3_main/
│       └── developmental_arc/
├── logs/                                          (gitignored except .gitkeep)
└── archive/                                       (for superseded code)
```

The previous repo provides:
- The AI2-THOR explorer (`Weft/src/env/furniture_route_explorer.py`) with densified-Teleport mechanism — superseded for Phase 2/3 by `src/env/continuous_motion_explorer.py` (session 4). The previous repo's explorer also carries jitter logic from the Stage 0b stability batch; that logic is *not* ported forward — v0 uses no per-frame jitter (see §1.5).
- The frame annotation format (`Weft/results/stage_0b_furniture/main/frame_annotations.jsonl`) — Phase 2/3 collections produce JSONL in the same schema, with additional `phase` and `perturbation` fields.

---

## 3. Predictor Implementation

`src/predictor/inner_pam.py`

### 3.1 Architecture

- Input projection: `nn.Linear(1024, 512)` (DINOv2 d=1024 → hidden=512).
- Positional encoding: learned, additive, applied to the W=16 input embeddings. `nn.Embedding(W, 512)`.
- Encoder: 4-layer `nn.TransformerEncoder` with `nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, activation='gelu', norm_first=True, batch_first=True)`. Pre-LayerNorm.
- Pooling: take the last token's hidden state (the position corresponding to the most recent frame).
- Output projection: `nn.Linear(512, K * (d + 1))` where K=16, d=1024 → `K*1025 = 16400` units.
- Reshape output to `(K, d + 1)`; split last axis into:
  - `mean`: `(K, d)` — the predicted centreline.
  - `log_var`: `(K, 1)` — predicted log-variance per step (isotropic scalar).

**Initialisation:** PyTorch defaults; verify at training start that no parameters are accidentally frozen (per `CODING_STANDARDS.md` §7.2). Log `sum(p.numel() for p in predictor.parameters() if p.requires_grad)` and write to results.

### 3.2 Forward signature

```python
def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        window: (B, W, d) tensor of W=16 most recent embeddings.

    Returns:
        mean: (B, K, d) predicted centreline.
        log_var: (B, K) predicted log-variance per step (last dim squeezed).
    """
```

Shape assertions at entry and exit per `CODING_STANDARDS.md` §7.1.

### 3.3 Loss

For each predicted step k=1..K with predicted mean `μ_k`, predicted log-variance `ℓ_k = log σ²_k`, actual observation `y_k`:

```python
# y_k is L2-normalised (DINOv2 output property); μ_k is not constrained to unit norm.
# y_k is detached from the encoder's graph (encoder is frozen anyway; explicit .detach() for clarity).

squared_error = (y_k - mu_k).pow(2).sum(dim=-1)         # (B,), Euclidean squared error
loss_k = 0.5 * (squared_error / log_var_k.exp()  +  d * log_var_k)

# Total loss: sum over K, mean over batch.
loss = loss_k_all.sum(dim=1).mean(dim=0)                # scalar
```

**Numerical hygiene:**
- Clamp `log_var` to `[-10, 10]` before exponentiating.
- Assert no NaN/Inf at loss-computation boundary per `CODING_STANDARDS.md` §7.5.

### 3.4 Parameter count check
At init, CC logs the parameter count and verifies it is approximately 21M (input projection ~525k, transformer encoder ~12M, output projection ~8.4M). Tolerance: ±10%. If outside, stop and report.

---

## 4. Online Trainer

`src/trainer/online_trainer.py` — single-pass, continuous-time, one gradient step per resolved prediction.

### 4.1 Stream contract

The trainer consumes a precomputed stream of L2-normalised embeddings: `embeddings: np.ndarray` of shape `(N, 1024)` and corresponding annotations `(N,)` (matching the JSONL schema).

Phase 1 uses `data/dinov2_embeddings/embeddings.npy` directly. Phases 2 and 3 use freshly collected, freshly encoded streams.

### 4.2 Loop (sketch)

At each step `t`:
1. Resolve any pending prediction whose target step has arrived (compute loss; one optimizer step).
2. Append `embeddings[t]` to the bank with recency depth = `t`.
3. Update rolling window of W=16 most recent embeddings.
4. If window is full, run predictor forward; enqueue prediction with `target_step = t + K`.

The contract is **one prediction enqueued per step and one gradient step per resolved prediction**. Implementation detail (queue / circular buffer / tensor indexing) is left to CC; the contract is what matters.

### 4.3 Optimiser

- AdamW
- `lr = 3e-4`  (SCAFFOLDING; spec §7.4 default)
- `weight_decay = 0.01`  (regularisation, not shape-decay; SCAFFOLDING)
- `betas = (0.9, 0.999)`
- No LR schedule for v0 (SCAFFOLDING).
- Gradient clipping: `clip_grad_norm_(predictor.parameters(), max_norm=1.0)`. SCAFFOLDING.

### 4.4 Logging during training

Three levels per `CODING_STANDARDS.md` §6.1:

**Per-step (TensorBoard):** loss, mean log-variance (across K), gradient norm, current step index.

**Per-checkpoint (JSON):** at every checkpoint (see §4.6), write to `results/inner_pam_v0/{phase}_main/checkpoint_{step}.json`:
- training step
- mean loss over last interval
- mean log-variance per step k (vector of K values)
- predictor weight L2 norm
- bank size
- timestamp
- git commit hash

**Per-session (HANDOFF.md):** at session boundaries, what was attempted, what completed, what's running (PID + log path), next immediate action.

### 4.5 Checkpoint files

Save to `results/inner_pam_v0/{phase}_main/ckpt_{step}.pt`:
- predictor state dict
- optimizer state dict
- bank index file (FAISS serialisation) + bank metadata (JSON: depths, frame indices)
- RNG state (numpy + torch)
- git commit hash at save time

Per `CODING_STANDARDS.md` §5.4. A checkpoint that can't be reproduced from its commit hash is a checkpoint we don't have.

### 4.6 Checkpoint cadence (derived from repetition-bin coverage)

The repetition-stratified metric (spec §10.2, this doc §6.2 M4) requires evaluation at training points where the perturbed shape has been observed at counts spanning {1–5, 6–19, 20–50, 51–99, 100+}.

**Loop length: ~316 frames** for the continuous-motion substrate (5-loop calibration, 2026-05-13: 1,580 frames / 5 = 316 frames per loop, comprising ~57 close-up + ~260 transit + corner-rotation frames). The prior 458-frame estimate was inherited from the 30-frame-static-dwell Stage-0b substrate and is no longer applicable; see §1.3 and the session-4 HANDOFF entry. For perturbed shapes in Phase 2 and Phase 3, the bins map to training-step ranges as follows:

| bin | loops within phase | frames into phase |
|---|---:|---|
| 1–5 | 1–5 | 316–1,580 |
| 6–19 | 6–19 | 1,896–6,004 |
| 20–50 | 20–50 | 6,320–15,800 |
| 51–99 | 51–99 | 16,116–31,284 |
| 100+ | 100+ | 31,600+ |

Checkpoint cadence is therefore:

- **Phase 1 (existing 100k embeddings, ~218 loops on the *prior* substrate):** every 10,000 steps. Substrate-degenerate baseline per session-4 disposition; not re-run. Cadence preserved for the audit trail.

- **Phase 2 and Phase 3 (perturbed shapes start at 0 reps, new 316-frame substrate):** denser checkpointing in early phase to cover the 1–5 and 6–19 bins. Phase-relative steps: **1,000 / 2,000 / 4,000 / 6,500 / 10,000 / 15,000 / 20,000 / 30,000 / 40,000 / 55,000 / end** — 10 checkpoints plus end-of-phase. Coverage:

  | step | loops elapsed | bin |
  |---:|---:|---|
  | 1,000 | 3.2 | 1-5 ✓ |
  | 2,000 | 6.3 | 6-19 ✓ |
  | 4,000 | 12.7 | 6-19 |
  | 6,500 | 20.6 | 20-50 ✓ |
  | 10,000 | 31.6 | 20-50 |
  | 15,000 | 47.5 | 20-50 |
  | 20,000 | 63.3 | 51-99 ✓ |
  | 30,000 | 94.9 | 51-99 |
  | 40,000 | 126.6 | 100+ ✓ |
  | 55,000 | 174.1 | 100+ |
  | end (~61.8k) | ~195.7 | 100+ |

  All five bins covered; 100+ bin gets three checkpoints (40k, 55k, end). Old cadence (1k / 2.5k / 5k / 9k / 12k / 16k / 23k / 34k / 46k / 55k / end) preserved in `src/config.py` `PHASE_2_3_CKPT_STEPS` for the audit trail; the new schedule replaces it for v0 work post-session-4.

CC verifies this schedule produces non-empty bins by computing the rep count from the annotations at each checkpoint.

### 4.7 Shape assertions and freezing checks (init-time)

Before training starts:
1. Assert encoder is frozen: `encoder.eval()` and `for p in encoder.parameters(): assert not p.requires_grad`.
2. Assert predictor is trainable: `sum(p.numel() for p in predictor.parameters() if p.requires_grad) > 0`.
3. Run one forward pass with synthetic input, verify output shapes match the contract.
4. Verify embedding norms in the stream: sample 1000 random indices, check norms are 1.0 ± 1e-5.

---

## 5. Recall Mixing Function

`src/mixing/recall_mixer.py`

### 5.1 Confidence score

Per spec §7.5: confidence is the *negative* mean of predicted log-variance over the first M=3 steps. Lower variance → higher confidence.

```python
def confidence(log_var: torch.Tensor, M: int = 3) -> torch.Tensor:
    # log_var: (B, K). Returns (B,).
    return -log_var[:, :M].mean(dim=1)
```

### 5.2 Threshold τ calibration

τ is the median confidence value observed across steps 5,000–10,000 of Phase 1 training. Computed at step 10,000 and frozen thereafter. Stored in `results/inner_pam_v0/phase1_main/tau_calibration.json`. SCAFFOLDING.

Reasoning: by step 5k the predictor has moved away from initialisation noise; by step 10k it has seen ~22 full loops of the route. Median of the post-warmup distribution gives a roughly 50/50 split at calibration time, leaving room to observe the predictor-only fraction *rising* over the rest of training (spec §10.2 bank-vs-weights claim).

### 5.3 Mixing

```python
def mix(probe_window, predictor, bank, tau, M=3, top_k_instances=10):
    mean, log_var = predictor(probe_window)
    conf = confidence(log_var, M)

    if conf > tau:
        return {"mode": "predictor_only", "mean": mean, "log_var": log_var}
    else:
        window_mean = probe_window.mean(dim=1)
        window_mean = F.normalize(window_mean, dim=-1)
        instances = bank.retrieve_by_cosine(window_mean, k=top_k_instances)
        return {"mode": "predictor_plus_bank", "mean": mean, "log_var": log_var, "instances": instances}
```

`mode` is logged per probe and used in M5 (§6.2).

---

## 6. Evaluation Harness

`src/eval/probes.py`, `src/eval/metrics.py`, `src/eval/controls.py`

### 6.1 Probe construction

**Held-out frames:** last 10 loops of each phase's stream are reserved from training. The trainer reads `embeddings[:N_train]` where `N_train = N_total - 10 * loop_length`. The evaluator reads `embeddings[N_train:]` for the held-out region. The exact `N_train` per phase is computed from the annotations (Phase 1: ~95,400 from the seed-7 run; Phase 2 and 3 depend on collected stream length).

**Steady-state probes:** for each viewing position in {Bed, DiningTable, Dresser, Sofa, Television}, sample 50 windows where all W=16 frames are dwell frames at that position. Targets are the next K=16 actual frames.

**Cue probes:** for each transition (item A → item B), sample 50 windows where the W=16 frames straddle the dwell-to-transit boundary (last 8 frames dwelling at A, first 8 frames transiting toward B). Targets are the next K=16 actual frames.

Each probe is labelled with `(probe_type, from_item, to_item if cue else None, loop_index, phase)`.

### 6.2 Metrics

**M1 — Centreline accuracy per step.** Cosine similarity between predicted `μ_k` (L2-normalised) and actual `y_k`, per step k=1..16, per probe. Reported as: mean and std across probes, separately for each probe-type and from-item combination.

**M2 — Variance calibration.** For each step k, compute the empirical squared error `||y_k - μ_k||²` and the predicted variance `σ²_k`. Ratio `(squared_error / d) / σ²_k` should be ~1.0 if calibrated. Reported per step k, disaggregated by probe-type and by distance-from-transition bin.

**M3 — Shape clustering (developmental observation, spec §10.4).** At each evaluation checkpoint:
- Within-shape predicted-mean cosine: pairwise within (probe_type × from_item).
- Cross-shape predicted-mean cosine: pairwise across different (probe_type × from_item) combinations.
- Cluster sharpness: within − cross.
- Predictor variance at recurring states: mean `σ²` at steady-state probes.

A developmental time series across the full training run.

**M4 — Repetition-stratified accuracy.** For each shape cluster present in evaluation, count how many times its frames appeared in the training stream up to the current evaluation point. Bin into {1–5, 6–19, 20–50, 51–99, 100+}. Report M1 disaggregated by bin. Phase 1 populates only the 100+ bin (for un-perturbed shapes); Phase 2 and 3 populate the lower bins for perturbed shapes via the dense early-phase checkpointing in §4.6.

**M5 — Bank-vs-weights recall fraction.** For each probe, run the mixing function and record `mode`. Report fraction of probes where `mode == "predictor_only"` across training checkpoints.

*τ sensitivity (three-point):* M5 is also reported at two alternative τ values computed from later windows of Phase 1: (a) τ = nominal (median over steps 5k–10k, the value used by the mixer in production); (b) τ' = median over a 5k-step window ending at step 20k; (c) τ'' = median over the final 5k steps of Phase 1. The three M5 trajectories are overlaid in the developmental-arc output (§10). Purpose: detect whether the nominal τ placement materially affects the bank-vs-weights conclusion. If the three trajectories disagree qualitatively (e.g., one shows rising predictor-only fraction and another shows flat), the τ choice is doing more work than spec §2.7 / §2.8 should require, and the result is flagged ambiguous rather than supporting V1.

**M6 — Cluster accommodation (Phase 2/3 only).** For perturbed-shape probes:
- Within-original cluster cosine (Phase 1 held-out at same item).
- Within-perturbed cluster cosine (Phase 2 or 3 held-out at same item).
- Cross-cluster cosine (Phase 1 vs Phase 2/3 at same item).
- Tracked across Phase 2/3 checkpoints to observe accommodation evolution.

Note: Phase 2's perturbation is joint (Dresser + Sofa together); M6 reports aggregate over both items and also per-item. The aggregate is the gate-relevant value (§8.7); per-item is for diagnosis. The single-item isolation originally intended for Phase 2 is weaker than planned (§1.4); the architectural claim under joint perturbation is still testable.

**M7 — Compounding accommodation (Phase 3 only).** Cluster sharpness on both Phase 2 perturbed shapes and Phase 3 newly-perturbed shapes, plus cross-cluster cosine between Phase 1 / Phase 2 / Phase 3 predicted means at the same items.

### 6.3 Controls (run per phase)

**C1 — Pure cosine baseline.** No predictor. For each probe window: compute the mean window embedding (L2-normalised); retrieve top-K=10 nearest bank embeddings by cosine. Compare the next-frame retrieval to `y_1`. The cosine baseline cannot produce a centreline path — it produces the bank entry most similar to the current window. The reportable comparison is "given the current window, is the most-similar bank entry close to `y_1`?"

**C2 — Shuffle control.** A second predictor with identical architecture, trained on a *temporally shuffled* version of the same embedding stream (seed=0 permutation of training indices). Held-out evaluation uses the *unshuffled* held-out region. Same metric suite as the main predictor.

### 6.4 Evaluation runs at checkpoint boundaries with training paused

Evaluation does not run in parallel with training. Each evaluation runs after a checkpoint write, against the just-written checkpoint, with the training process paused until evaluation completes. Rationale:
- The 16 GB GPU has tight VRAM with predictor + frozen DINOv2 + FAISS index + activations in flight; parallel eval risks OOM and silent slowdowns.
- Checkpoint boundaries are already discrete moments — evaluation between checkpoints would interpolate rather than reveal new structure.
- Operational discipline: paused-for-eval is simpler to reason about, log, and reproduce.

Eval cost per checkpoint is ~500 probes × per-probe inference (250 steady-state + 250 cue, per §6.1). For a 21M-param predictor on 16-frame windows, 500 forward passes complete in seconds rather than minutes; the dominant cost is the FAISS k-NN sweep plus M3/M4/M5/M6 metric computation. Empirical expected: under 2 minutes per checkpoint. Phase 1 has 10 checkpoints; Phase 2 and Phase 3 each have 12 (phase start + 11 from the schedule); total ≈ 34 checkpoints × ~2 minutes ≈ ~1 hour total across the experiment. Acceptable.

If empirical eval time exceeds 10 minutes per checkpoint, CC reports in HANDOFF.md and the experiment chat decides whether to reduce the checkpoint count.

### 6.5 Shuffle control sanity check

The shuffle control's behaviour under Gaussian NLL with learned variance is empirically unknown. Under MSE / cosine retrieval (the previous architecture's regime), shuffle predictors collapsed to mean-output. Under the new loss, the optimum may differ: shuffle could learn "predict mean embedding with high variance everywhere" (calibrated-but-uninformative), or it could learn position-correlated patterns from the marginal distribution that produce non-trivial centreline accuracy.

If shuffle's behaviour lands accidentally close to PAM's, the PAM-vs-shuffle gap (G1.4 in §7.7) misfires not because the architecture failed but because the control was wrong.

CC runs an explicit shuffle sanity check at the end of each phase:

**S1.** Distribution of shuffle's predicted log-variance across held-out probes. Expected: high mean (lower confidence than main predictor), possibly broader spread.

**S2.** Distribution of shuffle's M1 centreline cosine across held-out probes. Expected: lower mean than main predictor, possibly with much higher variance across probes.

**S3.** Shuffle's M3 cluster sharpness. Expected: near zero (no temporal structure → no shape clusters).

**S4.** Visual inspection of shuffle's predicted means for 5 randomly sampled probes. Are they qualitatively similar across very different probe types? (If yes, the "collapse to mean" pattern holds.) Are they near zero magnitude with high uniform variance?

*Quantitative collapse-to-mean check (added to S4):* compute (a) the L2 norm of shuffle's predicted mean averaged across all held-out probes, and (b) the mean of shuffle's predicted log-variance across all held-out probes. **Starting thresholds (SCAFFOLDING — recalibrate after observing the empirical shuffle distribution at end of Phase 1):** if `mean ||μ_shuffle||₂ < 0.15` *and* `mean log σ²_shuffle > 0.4`, the quantitative check reads "collapse-to-mean (expected)." Otherwise, the quantitative check reads "non-collapse" and the qualitative S4 inspection plus S1–S3 determine the verdict, flagged for experiment-chat review if any of {S1, S2, S3, quantitative S4} disagree.

The starting thresholds derive from: y_k is unit-norm (DINOv2 L2-normalised), so if μ predicts near the mean of unit vectors (which has L2 < 1 by Jensen), an "collapsed" predictor produces low-norm outputs; log σ² > 0.4 means σ² > 1.49, which is "high variance everywhere" for unit-norm targets. Both values are first-cut guesses; recalibrate from the actual end-of-Phase-1 shuffle distribution.

The sanity check produces a one-page report at `results/inner_pam_v0/{phase}_shuffle/sanity_check.md` with values for S1–S4 and a qualitative verdict: "expected" (shuffle behaves as a degraded predictor) or "unexpected" (shuffle is doing something the PAM-vs-shuffle gap was not designed to compare against).

The PAM-vs-shuffle gates in §7.7 / §8.7 / §9.7 are only interpretable if the sanity check returns "expected." If "unexpected," the gate is flagged for re-design in HANDOFF.md and the experiment chat consults Grok before proceeding.

---

## 7. Phase 1 — Fixed Environment Training

### 7.1 Goal
Train Inner PAM on the existing 100k DINOv2 embeddings of the seed-7 furniture route. Establish baseline shape-learning behaviour with no perturbations.

### 7.2 Inputs
- `data/dinov2_embeddings/embeddings.npy` (exists; 100,000 frames × 1024 dim, fp32, L2-normalised).
- `Weft/results/stage_0b_furniture/main/frame_annotations.jsonl` (exists; 100,000 records).

CC verifies at session start:
- `embeddings.shape == (100000, 1024)`.
- Per-frame norms in `[1.0 - 1e-5, 1.0 + 1e-5]`.
- Annotation count = 100,000.
- Annotation schema matches expected fields (`frame_index`, `loop_index`, `phase` ∈ {dwell, transit}, `viewing_position_id` for dwell frames).

If any verification fails, stop and report.

### 7.3 Held-out split
Last 10 loops of the 218-loop stream. Compute the exact frame index boundary from the annotations.

### 7.4 Training run

Single `nohup` invocation:
```
nohup python -u scripts/run_phase1_train.py \
    --config src/config.py \
    --output_dir results/inner_pam_v0/phase1_main \
    > logs/phase1_main_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/phase1_main.pid
```

Expected wall-clock: 30–90 minutes (predictor is ~21M params, single-sample SGD, no encoder forward at training time since embeddings are precomputed).

### 7.5 Shuffle control

After Phase 1 main training completes (sequential, not parallel — VRAM contention):
```
nohup python -u scripts/run_phase1_shuffle.py \
    --config src/config.py \
    --output_dir results/inner_pam_v0/phase1_shuffle \
    --shuffle_seed 0 \
    > logs/phase1_shuffle_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

The shuffle script reads `embeddings.npy`, applies `np.random.default_rng(0).permutation(N_train)` **to the training portion of the embedding stream itself** (rows `[0, N_train)`), and trains an otherwise-identical predictor on the permuted stream in sequential order. After permutation, when the trainer builds window `[t-W+1, ..., t]`, those W=16 contiguous positions in the permuted array hold W=16 *random unrelated* embeddings from the original stream — temporal structure within and across the (window, target) pair is destroyed at the source, as spec §10.1 / §6.3 require. Annotations are permuted in lockstep so bank-entry metadata reflects each embedding's original frame. Held-out evaluation uses the *unshuffled* held-out region.

(Earlier drafts of this section permitted a literal reading where "permutation of training indices" meant the visit *order* of (window, target) pairs, leaving each pair's contents temporally coherent — that is standard SGD batch shuffling and does NOT test the spec's "temporal structure destroyed" claim. The wording above was tightened on 2026-05-13 after session 2's gate analysis found shuffle outperforming main under the visit-order interpretation; see HANDOFF session-2 entry.)

### 7.6 Evaluation
At each Phase 1 checkpoint (paused training):
```
python scripts/run_eval.py \
    --checkpoint results/inner_pam_v0/phase1_main/ckpt_{step}.pt \
    --probes phase1 \
    --output results/inner_pam_v0/phase1_main/eval_{step}.json
```

Shuffle sanity check (§6.5) runs once at end of Phase 1 against the final shuffle checkpoint.

### 7.7 Phase 1 gates

CC checks these at end of Phase 1 main training; failure of any gate stops the batch.

**G1.1 — Training completed without NaN/Inf.** Loss tensor has no NaN/Inf at final step; predictor weights have no NaN/Inf at final checkpoint.

**G1.2 — Loss decreased.** Mean loss over final 10k steps < mean loss over first 10k steps (excluding warmup steps 0–500).

**G1.3 — Variance learned structure.** Final mean `log σ²` over steady-state probes is lower than final mean `log σ²` over cue probes. Specifically: `mean(log_var, steady-state probes) + 0.3 < mean(log_var, cue probes)`. (SCAFFOLDING threshold; recalibrate after observing the empirical distribution.)

**G1.4 — Multi-step centreline accuracy beats shuffle, statistically.** Gates on **k=8** (mid-horizon — far enough that a next-frame predictor or 1-step Markov model cannot shortcut the test, but inside the regime where the predictor has meaningful signal). For paired probes (same held-out probe evaluated under main predictor and shuffle predictor), the per-probe difference `M1_main(probe, k=8) − M1_shuffle(probe, k=8)` is significantly positive across the 250 steady-state probes. Computed as a one-sided paired t-test (`scipy.stats.ttest_rel`, `alternative='greater'`); gate passes if `p < 0.01`. **Normality fallback:** if Shapiro-Wilk on the per-probe difference distribution rejects normality at `p < 0.05` (likely with n=250 cosine-bounded data), use `scipy.stats.wilcoxon` with `alternative='greater'` instead; same `p < 0.01` threshold. **Conditional on the shuffle sanity check (§6.5) returning "expected".** If the sanity check returns "unexpected," G1.4 is held pending re-design rather than passed or failed.

*Diagnostic context (reported but not gated):* the same paired test is also computed at k=1 (near-horizon, where a next-frame shortcut could matter) and k=16 (far-horizon, predictor at its noisiest). All three p-values are logged. Cleanest story: main beats shuffle at k=8 *and* k=16 *and* k=1. If main beats at k=8 but fails at k=1, that's diagnostic — shuffle is doing well at next-frame via marginals, which is interpretable but should be flagged. If main beats at k=1 but fails at k=8, the architecture is operating as a next-frame predictor rather than a path predictor and §2.2 is not supported in the expected form — flag and stop.

Rationale for k=8 as the gate horizon: k=1 is reachable by a next-frame predictor and doesn't test the path-prediction claim. k=16 is the longest horizon and the noisiest — choosing it would make the gate harder to pass for reasons unrelated to whether the architecture works. k=8 is the principled midpoint: clearly multi-step (rules out 1-step shortcuts), inside the trained window, and the predictor's signal-to-noise is highest in the middle of W=16.

**G1.5 — Shape clustering present and forming.** Compound criterion testing spec §2.2 directly (clustering tightens with repetition):

- *Trajectory:* M3 cluster sharpness at the final Phase 1 checkpoint > M3 at the first checkpoint after warmup (step 1k). The trend across all logged checkpoints in between is monotonically non-decreasing on at least 7 of the final 9 checkpoint-to-checkpoint transitions (allowing 2 stochastic dips). The trajectory criterion is the primary content of the gate — it is what §2.2 actually claims.
- *Floor:* M3 cluster sharpness at the final Phase 1 checkpoint > 0.10 (SCAFFOLDING — guards against a trajectory that rises from −0.05 to 0.00 and passes the trajectory criterion while being substantively meaningless).
- *Calibration report:* At end of Phase 1, log the per-cluster within-shape cosine and the M3 value at every checkpoint to `results/inner_pam_v0/phase1_main/m3_trajectory.json`. This file is reviewed by the experiment chat before declaring G1.5 pass/fail; if the 0.10 floor turns out to be unreasonable (either trivially clearable or unreachable) given the empirical distribution observed, the floor is recalibrated with explicit justification per §16 drift-detection.

Rationale for trajectory-over-threshold: the original G1.5 (`M3 > 0.10` at final checkpoint only) gates on an absolute value that we have no prior data to calibrate. The trajectory form tests the architectural claim (repetition produces tightening clusters) rather than betting on the threshold value alone; the floor still catches "rose from nothing to nothing" trajectories.

**Gate failure protocol:** stop, write to HANDOFF.md, do not proceed to Phase 2. Override requires explicit justification in the experiment chat per `research_operations_v1.md` §8.10.

---

## 8. Phase 2 — Stage A Baseline + LivingRoom Material Re-texturisation

### 8.1 Goal
Run a Stage A baseline (loops 1–30, unperturbed) followed by a Stage B perturbation segment (loops 31+, `RandomizeMaterials` fired per loop on the LivingRoom). The Stage A → Stage B transition within Phase 2 is observed by the in-flight transition diagnostic (§8.7a). The architectural prediction is that Dresser and Sofa trajectories widen from sharp lines (Stage A) into tubes (Stage B) around their Stage A means, while Bed, DiningTable, and Television trajectories remain on their Stage A lines as within-experiment controls.

### 8.2 Phase 2 preflight (run before collection)

`scripts/run_phase2_preflight.py` verifies the perturbation mechanism behaves as specified before any data collection. This is mandatory and gates the rest of Phase 2.

**Preflight tasks:**

1. **Verify `RandomizeMaterials` action exists and accepts `inRoomTypes`.**
   - Launch a controller against the seed-7 scene.
   - Call `controller.step(action="RandomizeMaterials", inRoomTypes=["LivingRoom"], useTrainMaterials=True)`.
   - Verify the action returns success (event.metadata["lastActionSuccess"] is True).
   - If the action fails or `inRoomTypes` is unrecognised, stop and report.

2. **Verify per-loop re-application produces fresh textures.**
   - After step 1, capture an RGB frame at the Dresser viewing position (item 3, position from seed-7 metadata).
   - Call `RandomizeMaterials` again with the same parameters (the per-loop pattern Phase 2 uses from loop 31 onward).
   - Capture another RGB frame at the Dresser viewing position.
   - Verify the two frames differ measurably (cosine of flattened RGB < 0.95). If identical, `RandomizeMaterials` is not actually re-randomising per call — stop and report.
   - Also call `RandomizeMaterials` a third time and verify the second→third capture also differs (rules out a two-state cycle).

3. **Verify perturbation locality (does not affect Bedroom items).**
   - Capture frames at Bed and Television viewing positions before and after the LivingRoom RandomizeMaterials call.
   - Verify pre/post frames at Bedroom items are pixel-identical (cosine > 0.999).
   - If Bedroom items also changed, `inRoomTypes` scoping is not working — stop and report.

4. **Verify perturbation is visually detectable at LivingRoom items.**
   - Capture frames at Dresser and Sofa viewing positions before and after.
   - Verify pre/post frames differ measurably (cosine < 0.9 in flattened RGB).
   - If frames are visually identical despite RandomizeMaterials succeeding, the perturbation is a null perturbation — stop and report.

5. **Verify determinism across controller sessions.**
   - Launch a fresh controller, apply `RandomizeMaterials` with the same parameters.
   - Compare the resulting Dresser frame to step 1's Dresser frame.
   - If identical: determinism confirmed; proceed.
   - If different: AI2-THOR's RandomizeMaterials is non-deterministic across sessions. CC documents the specific materials applied (read from `event.metadata` after the call) and treats them as a fixed-per-run perturbation. Reproducibility is per-run rather than global. Flagged in HANDOFF.md.

Preflight output: `results/inner_pam_v0/phase2_preflight/preflight_report.md` with all five verifications and pass/fail per item.

**Preflight gates:** all five must pass (with documented partial pass on item 5 if determinism is per-run). If any of 1–4 fails unconditionally, stop and report; do not proceed.

### 8.3 Frame collection

`src/env/explorer_phase2.py` wraps the **continuous-motion** explorer (`src/env/continuous_motion_explorer.py`, session 4):
- The collection script accepts a `--perturbation_start_loop` argument (default **31**). Loops with `loop_index < perturbation_start_loop` run as the Stage A baseline: pure continuous-motion route with **no `RandomizeMaterials` call**. From the start of every loop with `loop_index >= perturbation_start_loop`, the wrapper calls `controller.step(action="RandomizeMaterials", inRoomTypes=["LivingRoom"], useTrainMaterials=True)` *once at the start of that loop*, supplying fresh LivingRoom textures for that loop. The call returns success metadata plus material identifiers; these are appended per loop to `data/phase2_collection_metadata.json` (`{loop_index: [material_id, ...], ...}`) for reproducibility.
- Per-frame annotations carry a `perturbation_active: bool` field — `False` for all frames in Stage A loops (loops 1 to `perturbation_start_loop - 1`), `True` for all frames in Stage B loops onward. Downstream analysis disaggregates Stage A vs Stage B frames via this field.
- No pose jitter of any kind. The cross-loop pose-determinism observed in session-4 calibration on Bed, DiningTable, Sofa, and Dresser apex frames is preserved in Stage A — that is the curriculum's baseline state. The variation in Stage B comes from `RandomizeMaterials` re-texturing the LivingRoom items (Dresser, Sofa) per loop; Bed, DiningTable, and Television remain visually constant across loops as within-experiment controls.

**Frame budget:** **65,000 frames ≈ 205 loops** (at ~316 frames/loop, calibration 2026-05-13). Minus 10 held-out loops, the model trains on **~195 loops** total: **~30 Stage A baseline loops** plus **~165 Stage B perturbed loops** (Dresser + Sofa with fresh LivingRoom textures per loop). Stage B's perturbed-shape rep count at the final eval checkpoint is ~165 — comfortably into the 100+ rep bin with 65+ reps of margin.

Derivation: M4 stratifies perturbed-shape recall by repetition count into bins {1–5, 6–19, 20–50, 51–99, 100+}. With the Stage A baseline occupying loops 1–30 (perturbation_active = False), the *perturbed-shape* rep count for Dresser and Sofa accumulates only from loop 31 onward. At the final eval checkpoint, the perturbed-shape rep count equals `trained_loops − (perturbation_start_loop − 1)`. With 195 trained loops and Stage A spanning loops 1–30, Stage B contributes ~165 perturbed-shape reps — comfortably into the 100+ bin with ~65 reps of margin.

| budget choice | substrate | collected loops | Stage A loops | Stage B trained loops | bin reached at final eval |
|---|---|---:|---:|---:|---|
| 65k | new (316 frames/loop, continuous motion) | ~205 | 30 | ~165 (after 10 held-out) | **~65 reps into 100+** ✓ |

The held-out region is the last 10 *collected* loops, all Stage B. Stage A frames are not held out — the Stage A baseline is fully trained on.

The remaining bins are populated within Stage B as before: 1–5 at phase-relative loops 31–35 (~316–1,580 Stage-B frames after the 30-loop Stage A offset of ~9,500 frames), 6–19 at loops 36–49, 20–50 at loops 50–80, 51–99 at loops 81–129, and 100+ from loop 130 onward through the final trained Stage B loop at ~195.

*Robustness:* CC re-computes the budget from the collected `phase2_annotations.jsonl` before Phase 2 training begins. If actual loop length exceeds 340 frames (≥7.5% above the 316-frame estimate), or if held-out is ever widened beyond 10 loops, the trained-loop count is re-checked against the 100+ boundary; budget is increased to preserve ≥20 reps of margin past 100, with the new value documented in HANDOFF.md.

Output: `data/phase2_frames/` (PNG, gitignored) + `data/phase2_annotations.jsonl`. Annotations carry `phase: "phase2"`, `perturbation: "livingroom_retexture"`, and `perturbation_active: bool` (False for Stage A loops, True for Stage B). Per-loop applied materials in `data/phase2_collection_metadata.json`.

Script: `scripts/run_phase2_collect.py`. nohup, log to `logs/phase2_collect_*.log`.

### 8.4 Encoding

`scripts/run_phase2_encode.py` runs frozen DINOv2 over Phase 2 frames, writes `data/phase2_embeddings/embeddings.npy`. Same encoder configuration as the substrate verification (fp16 eval, ImageNet mean/std normalisation, L2-normalised output).

CC verifies post-encoding:
- Shape: `(65000, 1024)`.
- Norms in `[1.0 - 1e-5, 1.0 + 1e-5]`.
- **Perturbation effect check (Stage B vs Stage A within Phase 2).** Mean cosine between 50 randomly sampled Phase 2 *Stage-B* Dresser-apex embeddings (`perturbation_active = True`, viewing_position_id = 3) and 50 randomly sampled Phase 2 *Stage-A* Dresser-apex embeddings (`perturbation_active = False`, same item) *vs* mean cosine within each set. If the cross-set mean is similar to the within-set means (< 0.05 separation), the per-loop `RandomizeMaterials` did not produce a measurable encoder-level perturbation — flag and stop. Repeat for Sofa. (Phase 1 embeddings are *not* used as the unperturbed reference: Phase 1 is on the substrate-degenerate baseline, the encoder-level comparison would conflate substrate effects with perturbation effects. The Stage A loops *inside* Phase 2 are the right reference: same substrate, same continuous-motion trajectory, only the LivingRoom textures differ.)

### 8.5 Continued training

**Phase 2 starts from a freshly-initialised predictor** (Phase 1 is discarded as substrate-degenerate; session-4 disposition). Bank is also fresh (empty at the start of Phase 2). Phase 3 will resume from Phase 2's final checkpoint and final bank state (spec §2.3); Phase 2 → Phase 3 is the only resumption boundary in v0.

Held-out: last 10 loops of Phase 2 reserved (all Stage B, since Stage A is the early phase).

Same training loop, denser checkpoint cadence per §4.6 (phase-relative steps **1k / 2k / 4k / 6.5k / 10k / 15k / 20k / 30k / 40k / 55k / end**, 316-frame-loop substrate). Output: `results/inner_pam_v0/phase2_main/`.

### 8.6 Phase 2 evaluation
Phase 1's metric suite plus M6 (cluster accommodation). Evaluated at each Phase 2 checkpoint.

### 8.7 Phase 2 gates

**G2.1 — Training continued without NaN/Inf.** Same as G1.1 at Phase 2 endpoint.

**G2.2 — Stage A cluster preserved through Stage B.** With Phase 1 discarded as substrate-degenerate, the within-experiment cluster-preservation reference is the *Stage A* segment of Phase 2 itself. M3 cluster sharpness on Bed / DiningTable / Television (the unperturbed control items) measured at end of Phase 2 (Stage B trained) ≥ 70% of cluster sharpness measured at end of Stage A (loop 30, before the perturbation begins). The unperturbed-item M3 trajectory is logged at every Phase 2 checkpoint to support this evaluation. (SCAFFOLDING threshold; recalibrate after empirical distribution is observed.)

**G2.3 — Variance responded to perturbation.** Mean predicted variance on perturbed-LivingRoom (Dresser, Sofa) probes is higher at the *start of Stage B* (loops 31–35) than at the *end of Stage A* (loops 26–30) within Phase 2, and decreases over the remainder of Stage B training. Statistical form: paired t-test on per-probe variance, end-of-Stage-A vs start-of-Stage-B; gate passes if start-of-Stage-B is significantly higher (p < 0.01). **Normality fallback:** if Shapiro-Wilk on the per-probe difference distribution rejects normality at `p < 0.05`, use `scipy.stats.wilcoxon` with `alternative='greater'`; same `p < 0.01` threshold. The same comparison run on Bed / DiningTable / Television probes must *not* show a significant variance change across the transition — those are the within-experiment controls (§8.7a in-flight diagnostic carries the same logic at per-loop granularity).

**Joint-perturbation framing (applies to G2.2, G2.3, and M6).** The Phase 2 perturbation affects two items (Dresser + Sofa) simultaneously via the LivingRoom `RandomizeMaterials` scope. Accommodation is therefore measured against a *joint* perturbation, not the single-item axis originally intended. This is a deliberate concession to AI2-THOR API constraints (§1.4). The architectural claim — shape representations widen but survive under visual variation — remains testable; the single-axis isolation is weaker than originally planned. When reporting G2.3 and M6, the perturbed-shape result aggregates over Dresser and Sofa unless reported per-item; per-item disaggregation is logged but the gate is computed on the joint set.

### 8.7a In-flight transition diagnostic (Stage A → Stage B)

The Stage A → Stage B transition inside Phase 2 (loop 30 → loop 31) is observed in flight during training, not just at the end-of-phase checkpoint. The trainer maintains per-loop running aggregates of training-time loss and predicted log-variance, disaggregated by `viewing_position_id`, and writes them to `results/inner_pam_v0/phase2_main/transition_diagnostic.json` after each loop completes. Coverage is at minimum loops 25–40 (the transition window); the implementation logs every loop for completeness.

**Three SCAFFOLDING gates** evaluated at the end of loop 35 (the first point where the post-transition window is fully populated). Thresholds recalibratable after observing the empirical loop-level distribution; record empirical values in HANDOFF whether or not the gates trigger.

**G2.T1 — Loss spike check (training stability across the perturbation onset).** Compute `max_loss = max(mean_loss[loop=31..35])` and `baseline_loss = mean(mean_loss[loop=25..30])`. Gate trips if `max_loss > 3.0 * baseline_loss`. A trip indicates the perturbation has produced a training instability the architecture cannot absorb within 5 loops; flag in HANDOFF and pause for experiment-chat review before letting Phase 2 training continue.

**G2.T2 — Perturbed-item variance widening (Dresser + Sofa).** Compute `log_var_loop30 = mean(log_var[loop=30, viewing_position_id ∈ {3, 4}])` and `log_var_loop35 = mean(log_var[loop=35, viewing_position_id ∈ {3, 4}])`. Gate trips if `log_var_loop35 - log_var_loop30 < 0.5` (in natural-log units). A trip indicates the predictor's variance is not absorbing the new perturbation within 5 loops; flag and pause.

**G2.T3 — Control-item variance stability (Bed + DiningTable + Television).** Compute per-item `delta_log_var = log_var[loop=35] - log_var[loop=30]` for each of viewing_position_id ∈ {1, 2, 5}. Gate trips if any item's `|delta_log_var| > 0.3`. A trip indicates cross-item interference — the perturbation on Dresser/Sofa is leaking into the predictor's representation of items the perturbation does not visually affect, which the architecture's per-item representations should not produce. Flag and pause.

**Trip behaviour.** If any of G2.T1 / G2.T2 / G2.T3 trips, the trainer:
1. Writes the trip verdict and the values to `transition_diagnostic.json` with `gate_tripped: true` and the specific gate name.
2. Writes a marker file `results/inner_pam_v0/phase2_main/transition_diagnostic_TRIPPED.txt` for easy detection by the launching shell.
3. Exits non-zero (status 3).
4. Does *not* continue training past loop 35.

The launching session then updates HANDOFF and pauses for experiment-chat review per `research_operations_v1.md` §8.10.

**Pass behaviour.** All three gates pass (loss within 3×, perturbed-item widening ≥ 0.5, control-item drift ≤ 0.3). Training continues to end of phase; the diagnostic JSON is updated per-loop through the rest of training as record-only.

Thresholds (3.0× loss, 0.5 log_var widening, 0.3 control drift) are SCAFFOLDING and recalibratable from the empirical distribution observed in the first run — recalibration requires explicit justification in HANDOFF per §16.

Gate failures stop the batch; do not proceed to Phase 3.

---

## 9. Phase 3 — Additive: LivingRoom Retexturisation + Stronger Perturbation

### 9.1 Goal
Add a second variation on top of Phase 2's perturbation. Test whether shape representations accommodate compounding perturbation. Preferred mechanism: a *qualitatively different* perturbation type (asset replacement) at a different position. Fallback: a *quantitatively larger* perturbation of the same type (full-house retexturisation).

**No internal Stage A baseline.** Phase 3 starts immediately with the Phase 3 perturbation active from loop 1; the Phase 2 → Phase 3 transition is itself the Stage B → Stage C step in the curriculum. By the time Phase 3 starts the predictor has 195 trained loops of Stage A + B experience and a memory bank populated with the same; an internal Stage A within Phase 3 would dilute the Stage C signal without diagnostic value (§1.5). Phase 3 resumes from Phase 2's final predictor checkpoint and final bank state (spec §2.3); weights and bank are *not* re-initialised at the boundary.

### 9.2 Phase 3 preflight (run before collection)

`scripts/run_phase3_preflight.py` verifies the preferred mechanism (asset replacement) and selects fallback if needed.

**Preferred mechanism: Television asset replacement-in-place.**

1. **Verify per-object removal action.** Call `controller.step(action="RemoveFromScene", objectId=<Television_object_id>)`. Verify success. Verify the Television is no longer present (`event.metadata["objects"]` does not include it).

2. **Verify asset placement at a specific pose.** Identify a replacement asset (suggested: any `ArmChair` from the seed-7 scene's asset library, or a generic ArmChair from AI2-THOR's asset database). Attempt to place it at the Television's original pose using `PlaceObjectAtPoint` or equivalent. Verify success and that the replacement is visible at the Television viewing position.

3. **Verify navigation route unchanged.** Run the explorer's teleport-to-Television-pose, capture a frame, verify the agent reaches the same XZ position and the replacement asset is visible in the frame.

4. **Verify perturbation locality and persistence.** Same as §8.2.3 and §8.2.4 adapted: capture frames at non-Television positions before and after; verify they are unchanged. Capture frames at Television position before and after; verify they differ.

5. **Verify the Phase 2 perturbation is preserved.** Apply Phase 2's `RandomizeMaterials(inRoomTypes=["LivingRoom"], ...)` *before* the asset replacement. Verify Dresser and Sofa frames still reflect the Phase 2 textures *and* Television frame reflects the replacement.

If all five preferred-mechanism preflights pass: Phase 3 uses asset replacement. Materials applied are documented in `data/phase3_collection_metadata.json`.

**Fallback mechanism: full-house `RandomizeMaterials()`.**

If any of the preferred-mechanism preflights fails (commonly: PlaceObjectAtPoint not supported for the chosen asset in this AI2-THOR version, or asset library doesn't include a suitable ArmChair):

1. Apply `controller.step(action="RandomizeMaterials", useTrainMaterials=True)` (no `inRoomTypes` → full house).
2. Verify all 5 items' frames differ from both Phase 1 originals and Phase 2 textures (Phase 2's LivingRoom textures are overwritten in this case).
3. Verify perturbation persistence across loops (as in §8.2.2).

If fallback also fails, stop and report — Phase 3 cannot proceed without consulting the experiment chat.

Preflight output: `results/inner_pam_v0/phase3_preflight/preflight_report.md` documenting which mechanism was selected and why.

### 9.3 Frame collection

`src/env/explorer_phase3.py` wraps the **continuous-motion** explorer (`src/env/continuous_motion_explorer.py`, session 4):
- At scene initialisation: apply Phase 2's RandomizeMaterials (LivingRoom), then apply Phase 3's perturbation (asset replacement or full-house retexture, per preflight).
- Standard continuous-motion route with the variation mechanism selected per §1.3.

**Frame budget:** **65,000 frames ≈ 205 loops** (same derivation as §8.3 — ~195 trained loops puts the final eval ~95 reps into the 100+ bin). Output: `data/phase3_frames/` + `data/phase3_annotations.jsonl`. Annotations carry `phase: "phase3"` and `perturbation: <selected_mechanism>` fields.

### 9.4 Encoding
Same as Phase 2. Cross-check: 50 sampled Phase 3 Television-position embeddings vs 50 Phase 1 Television-position embeddings — cross-mean cosine should be substantially lower than within-mean cosine. If similar, the perturbation was visually too close to the original; flag and report.

### 9.5 Continued training
Resume from Phase 2 final checkpoint. Same loop, same dense early-phase checkpointing. Output: `results/inner_pam_v0/phase3_main/`.

### 9.6 Phase 3 evaluation
Phase 2's metric suite plus M7 (compounding accommodation). Evaluated at each Phase 3 checkpoint.

### 9.7 Phase 3 gates

**G3.1 — Training continued without NaN/Inf.**

**G3.2 — Phase 1 + Phase 2 clusters preserved.** Cluster sharpness on un-perturbed shapes (Phase 1 held-out) and Phase 2 perturbed shapes does not collapse during Phase 3. Same 70% threshold per cluster.

**G3.3 — New perturbation produces high initial variance.** Mean variance on Phase-3-newly-perturbed probes (Television position) at start of Phase 3 is significantly higher than end-of-Phase-2 variance on Television probes. Paired t-test, p < 0.01. **Normality fallback:** if Shapiro-Wilk on the per-probe difference distribution rejects normality at `p < 0.05`, use `scipy.stats.wilcoxon` with `alternative='greater'`; same `p < 0.01` threshold.

---

## 10. Aggregate Evaluation — Developmental Arc

After all three phases complete, `scripts/run_eval.py --developmental` produces:

- Time series of M3 cluster sharpness across Phase 1, 2, 3 — separately for each shape.
- Time series of M5 bank-vs-weights recall fraction across the full training run, overlaid for three τ values (nominal 5k–10k median, τ at 20k, τ at end of Phase 1) per §6.2 M5.
- M4 repetition-stratified accuracy with each bin populated.
- M2 variance trajectories per shape.
- M1 centreline accuracy at endpoint, disaggregated per shape and per step k=1..16.
- Shuffle sanity-check verdicts per phase (S1–S4 summary, including quantitative collapse-to-mean check).

Output: `results/inner_pam_v0/developmental_arc/`.

---

## 11. Verdict Structure

The v0 experiment produces one of three named verdicts at endpoint. CC reports the evidence; the experiment chat (with Grok review) assigns the verdict.

### V1 — Shape-learning supported
- G1.5 passed: M3 cluster sharpness trajectory rises across Phase 1 (monotonically non-decreasing on ≥7 of last 9 transitions) and the end-of-Phase-1 floor (>0.10, or recalibrated) clears.
- G2.2 passed (clusters preserved across Phase 2).
- G3.2 passed (clusters preserved across Phase 3).
- M5 shows a *rising* predictor-only fraction across training under the nominal τ, and the τ-sensitivity overlay (§6.2) shows the rising trend is robust across all three τ values.
- M4 shows *increasing* centreline accuracy with repetition count.
- Shuffle sanity check returned "expected" each phase (S1–S3 quantitative checks pass, S4 quantitative collapse-to-mean check passes or is overridden with documented S4 qualitative inspection).

### V2 — Shape-learning falsified
- G1.5 fails, OR
- Multiple subsequent gates fail in ways consistent with "the predictor never represented shapes as recurring patterns."
- M5 stays flat or falls.
- M4 shows no repetition effect.

### V3 — Mixed / ambiguous
- Some gates pass, some fail.
- Specific failure modes documented; root cause not unambiguously attributable to the architecture.
- The pattern matches one of: encoder-substrate issue (despite §5 PASS), insufficient training duration, perturbation mechanism not producing expected signal, shuffle-sanity-check "unexpected" verdict invalidating the control, or others. Report in detail; do not assign V1 or V2.

`research_operations_v1.md` §1.3: every verdict ships with root cause reasoning.

---

## 12. Scaffolding Inventory

| label | item | location | removal plan |
|---|---|---|---|
| SCAFFOLDING | `K = 16`, `W = 16` | spec §9.5, this doc §3 | v1: variable-K, learned-K |
| SCAFFOLDING | predictor hidden=512, depth=4, heads=8 | this doc §3.1 | tune in v1 if capacity is limiting |
| SCAFFOLDING | `lr=3e-4`, `weight_decay=0.01`, AdamW | this doc §4.3 | v1 if loss plateaus |
| SCAFFOLDING | grad clip max_norm=1.0 | this doc §4.3 | revisit if needed |
| SCAFFOLDING | log_var clamp `[-10, 10]` | this doc §3.3 | revisit if calibration is poor |
| SCAFFOLDING | bank cap = 250,000, FIFO eviction | this doc §13.1 | depth-modulated eviction is v1 |
| ARCHITECTURE-for-v0 | `BANK_ALLOW_EVICTION = False` (hard stop on cap exceeded) | this doc §13.1 | v1 work flips this to exercise eviction dynamics |
| SCAFFOLDING | τ = median confidence over steps 5k–10k | this doc §5.2 | adaptive τ is v1 |
| SCAFFOLDING | `M = 3` (confidence aggregation window) | this doc §5.1 | adaptive M is v1 |
| SCAFFOLDING | `top_k_instances = 10` | this doc §5.3 | tune per phase |
| SCAFFOLDING | Phase 2/3 frame budget = 65k each (derived for rep-bin coverage incl. 100+) | this doc §8.3, §9.3 | revisit if bins under-populated or if held-out widened |
| SCAFFOLDING | Held-out = last 10 loops per phase | this doc §6.1, §7.3 | revisit if held-out is too small for n≥20 per shape |
| SCAFFOLDING | G1.3 threshold (`+0.3` log_var separation between cue and steady) | this doc §7.7 | recalibrate after empirical distribution observed |
| SCAFFOLDING | G1.5 floor (cluster sharpness > 0.10 at final Phase 1 checkpoint) | this doc §7.7 | recalibrate after empirical m3_trajectory.json reviewed; trajectory criterion is primary, floor is guard |
| SCAFFOLDING | G2.2 / G3.2 threshold (cluster sharpness ≥ 70% of Phase 1 end) | this doc §8.7, §9.7 | recalibrate after Phase 1 distribution observed |
| SCAFFOLDING | G1.4 / G2.3 / G3.3 use paired t-test at p < 0.01, Wilcoxon fallback if normality rejected | this doc §7.7, §8.7, §9.7 | re-examine if shuffle sanity check returns "unexpected" |
| SCAFFOLDING | S4 quantitative thresholds: `||μ_shuffle||₂ < 0.15` AND `mean log σ²_shuffle > 0.4` for "collapse-to-mean (expected)" | this doc §6.5 | recalibrate from empirical shuffle distribution at end of Phase 1 |
| SCAFFOLDING | Phase 2 perturbation = LivingRoom RandomizeMaterials | this doc §1.4, §8 | per-object material setting in v1 if API supports |
| SCAFFOLDING | Phase 3 preferred = Television asset replacement; fallback = full-house RandomizeMaterials | this doc §1.4, §9 | preflight selects mechanism per run |
| SCAFFOLDING | Phase 2 Stage A baseline length: `perturbation_start_loop = 31` (30 unperturbed loops before Stage B begins) | this doc §1.4, §8.3 | recalibrate if 30 loops produces inadequate Stage A baseline density; v1 may explore curriculum-length sensitivity |
| SCAFFOLDING | In-flight transition gate thresholds: G2.T1 `3.0×` loss spike, G2.T2 `≥ 0.5` log_var widening on perturbed items, G2.T3 `≤ 0.3` log_var drift on control items | this doc §8.7a | recalibrate from empirical per-loop distribution observed during the first run; v1: derive thresholds from loop-level variance of each metric in Stage A |
| SCAFFOLDING | Dense early-phase checkpointing schedule (§4.6) | this doc §4.6 | revisit if checkpoint cost > 10 min each |

Inventory reviewed at start and end of each phase per `research_operations_v1.md` §7.2.

---

## 13. Operational Rules

### 13.1 Memory bank specifics
- Append-only structure backed by `faiss.IndexFlatIP` (inner product = cosine on L2-normalised vectors).
- Hard cap: **250,000 entries** (SCAFFOLDING). Total stream across all three phases ≈ 230k (100k Phase 1 + 65k Phase 2 + 65k Phase 3); cap chosen with ~20k margin to hold the full stream without eviction in normal operation. Eviction in v0 is a SCAFFOLDING property of the bank, not an architectural feature being tested — the cap is set to keep it from firing so M5 (bank-vs-weights recall fraction) reflects predictor learning dynamics rather than bank-attrition dynamics.
- **Eviction-disabled flag:** in v0, FIFO eviction is *disabled at the code level*. If `bank.size() + 1 > cap` is ever encountered, the bank raises `BankCapExceededError` and the calling script writes the error to HANDOFF.md and exits non-zero. This is intentional: a silent FIFO eviction during v0 would invalidate M5; a loud hard stop forces the experiment chat to diagnose the overrun (likely a bug in collection or budget arithmetic) before continuing. The flag is `BANK_ALLOW_EVICTION = False` in `src/config.py`; v1 work that exercises eviction dynamics will flip it.
- Each entry carries: embedding vector, frame index, loop index, viewing_position_id, phase tag, perturbation tag.
- Bank state serialised at every checkpoint.

### 13.2 nohup discipline
Every training and collection script launched with `nohup` and a timestamped log file, PID captured to `logs/{stage}.pid`. CC polls log files for progress, never blocks on stdout. Per `CODING_STANDARDS.md` §5.2.

### 13.3 Never kill training
Per `CODING_STANDARDS.md` §5.1. If a job is running, let it complete. If termination is genuinely required, decision is made in the experiment chat and recorded in HANDOFF.md with rationale; checkpoint state captured before kill.

### 13.4 Stop conditions (unconditional)

Beyond gate failures, stop and report when:
- NaN/Inf appears in loss or predictor weights.
- Memory bank fails to serialise / deserialise cleanly at a checkpoint boundary.
- `BankCapExceededError` raised (eviction-disabled hard stop per §13.1; indicates a stream-length overrun that would otherwise pollute M5).
- Encoder verification re-check (sanity sampling on collected frames) fails.
- A perturbation produces a "null perturbation" finding (Phase 2 §8.4 or Phase 3 §9.4 cross-cosine sanity check).
- AI2-THOR perturbation preflight (§8.2, §9.2) fails on a non-recoverable item.
- Encoder forward produces non-L2-normalised output for any reason (configuration drift).
- Shuffle sanity check returns "unexpected" and the experiment chat has not signed off on continuing.
- 5 failed tool calls in sequence (`CODING_STANDARDS.md` §9.3).
- Disk fills above 90% on the working volume.

### 13.5 HANDOFF.md updates
Per `CODING_STANDARDS.md` §10. Updated at end of every session with:
- What was attempted.
- What worked (file paths to outputs).
- What failed (failure mode + evidence).
- What is in progress (running PIDs, log paths, expected completion).
- Next immediate action.

### 13.6 Numbers trace to files
Every number that appears in HANDOFF.md or any summary is verified against its source JSON / log file before quoting. Per `CODING_STANDARDS.md` §5.5. Post-compaction summaries are re-verified.

### 13.7 Git commits per task
Each completed task gets its own commit per `CODING_STANDARDS.md` §2.5–2.6. Suggested commits:
- `infra(repo): bootstrap src/, scripts/, results/ layout`
- `feat(predictor): inner_pam path predictor with Gaussian NLL loss`
- `feat(trainer): online single-pass trainer`
- `feat(memory): append-only bank with FAISS IndexFlatIP`
- `feat(mixing): confidence-thresholded recall mixer`
- `feat(eval): probe construction + M1..M7 metrics + shuffle sanity check`
- `feat(env): phase2 explorer wrapper with LivingRoom retexturisation`
- `feat(env): phase3 explorer wrapper with selected perturbation mechanism`
- `exp(phase1): main training run, 100k frames, ckpt at <hash>`
- `exp(phase1): shuffle control run, 100k frames, ckpt at <hash>`
- `exp(phase2): preflight verification + main + shuffle, ckpt at <hash>`
- `exp(phase3): preflight verification + main + shuffle, ckpt at <hash>`

### 13.8 Single CC session per working tree
Per `CODING_STANDARDS.md` §4.3 and `research_operations_v1.md` §9.5.

### 13.9 Sub-agents
Bounded tasks may be delegated to a sub-agent with minimal context per `CODING_STANDARDS.md` §4.1–4.2.

### 13.10 Phase boundaries are session boundaries
Per `research_operations_v1.md` §9.2. Each phase's end produces a HANDOFF.md update, a git commit, and a clean session start for the next phase.

---

## 14. Handoff Protocol

At the end of each session, before ending the CC chat:

1. `git status` clean (or pending changes committed).
2. `requirements.txt` current; `pip freeze > .env_snapshot.txt` committed if changed.
3. HANDOFF.md updated with the four required fields (attempted / worked / failed / next).
4. Any running jobs: PID, log path, expected completion time recorded.
5. Scaffolding inventory reviewed (start and end of phase).
6. Regression check: did any prior gate value degrade? If yes, flag.
7. Numbers in HANDOFF.md verified against source files.

Per `research_operations_v1.md` §14 (end-of-phase checklist) and `CODING_STANDARDS.md` §10.

---

## 15. Review Cycle Before Implementation Begins

Before CC writes any code for this batch:

1. Experiment chat (primary review) signs off on this document.
2. Document is submitted to Grok (secondary review) per `research_operations_v1.md` §2.2.
3. Findings from both reviews resolved or explicitly overridden in writing.
4. Revised version committed.
5. Phase 1 setup tasks (repo layout, predictor/trainer/bank/mixer/eval code) begin only after step 4.

---

## 16. Drift Detection — Project-Specific

Beyond the universal checks in `research_operations_v1.md` §15, this batch watches for:

- **Drift back to next-frame prediction.** If a draft, refactor, or "simplification" of the trainer would have the predictor optimise frame-by-frame next-step prediction rather than Gaussian NLL over K steps, that's drift back to the previous architecture's failure. Stop and report.

- **Cosine retrieval becoming the headline mechanism.** If implementation makes the bank the primary store and the predictor a query-generator, that's drift. Spec §3.3 inverts this — predictor forward pass *is* shape recall.

- **Aggregate-only reporting.** Any HANDOFF entry that quotes only aggregate metrics without the required disaggregations (per-shape, per-position, repetition-stratified, per-step k) is a process violation. Re-run with disaggregations.

- **Threshold recalibration without justification.** Any SCAFFOLDING gate threshold (G1.3 `+0.3`, G1.5 `>0.10`, G2.2 / G3.2 `≥70%`) may be recalibrated after observing the empirical distribution, but the recalibration must be reported with explicit reasoning. Recalibration without reasoning is a process violation per the encoder verification batch precedent.

- **"While we're at it" scope expansion.** Other questions get other batches.

- **Encoder-substrate failure misread as architecture failure.** Substrate verification has been done. If the predictor seems to behave as if it can't separate items during training, the first hypothesis is *something broke in the embedding pipeline*, not *the architecture failed*. Re-run substrate sanity checks on the actual embeddings being consumed before concluding the architecture is broken.

- **Skipping the shuffle sanity check.** The PAM-vs-shuffle gates rely on shuffle behaving as expected. Running the gates without the sanity check is a process violation; document the sanity check verdict before interpreting the gate.

---

## 17. Open Decisions for Execution-Time

These are deliberately unresolved here; CC decides with documentation:

- Replacement asset choice for Phase 3 preferred mechanism (one ArmChair from the asset library, if asset replacement preflight passes).
- Whether Phase 3 fallback (full-house retexture) is invoked, based on preflight outcomes.
- Specific materials selected by `RandomizeMaterials` per run (recorded from event metadata).
- Specific frame indices for the held-out boundary in each phase (computed from annotations).
- Eval batch size (depends on VRAM available; eval runs sequentially per §6.4 so contention with training is not an issue).
- Sequential vs parallel main-and-shuffle training (sequential is the default per §7.5; only run in parallel if Phase 1 wall-clock proves much shorter than expected and VRAM permits).

Each documented when chosen.

---

*End of v0 experiment instructions, fourth draft. Adversarial review (Grok) complete; review findings resolved in this draft. Ready for CC implementation.*

*Source: `WEFT_INNER_PAM_v0_Spec.md` (architecture), four locked decisions from preceding experiment chat, fixes from drafting-session self-review and experiment-chat review and Grok adversarial review, AI2-THOR API verification against public documentation, operational discipline from `CODING_STANDARDS.md` and `research_operations_v1.md`.*
