"""Hyperparameters and labels for v0.

Every fixed value below is tagged ARCHITECTURE or SCAFFOLDING per
research_operations_v1.md §7.1. ARCHITECTURE = removing it would break a
load-bearing claim in WEFT_INNER_PAM_v0_Spec.md §§1-6. SCAFFOLDING = a
tractable default with a removal plan in the Scaffolding Inventory of
WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md §12.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")


# ---------- ARCHITECTURE ---------------------------------------------------

# Output specification: centreline + per-step scalar log-variance (spec §3.3).
EMBED_DIM: int = 1024
# Bank exists as a separate store from predictor weights (spec §2.7).
BANK_ALLOW_EVICTION: bool = False  # v0 hard-stop on cap exceeded (instr §13.1)
# Confidence-graded mixing (spec §2.8).
# (No constant; the mode-selection function lives in src/mixing/recall_mixer.py.)


# ---------- SCAFFOLDING ----------------------------------------------------

# Window / prediction horizon (instr §1.5, spec §9.5). v1: variable-K, learned-K.
WINDOW_W: int = 16
PREDICT_K: int = 16

# Predictor architecture (instr §3.1). v1: tune if capacity limits.
PRED_HIDDEN: int = 512
PRED_LAYERS: int = 4
PRED_HEADS: int = 8
PRED_MLP_DIM: int = 2048
PRED_ACTIVATION: str = "gelu"

# Loss numerical hygiene (instr §3.3).
LOG_VAR_CLAMP_MIN: float = -10.0
LOG_VAR_CLAMP_MAX: float = 10.0

# Optimizer (instr §4.3). v1: revisit if loss plateaus.
LR: float = 3e-4
WEIGHT_DECAY: float = 0.01
ADAM_BETAS: tuple[float, float] = (0.9, 0.999)
GRAD_CLIP_MAX_NORM: float = 1.0

# Memory bank (instr §13.1). v1 flips BANK_ALLOW_EVICTION above for eviction dynamics.
BANK_CAP: int = 250_000

# Recall mixing (instr §5.1-5.3). v1: adaptive M, adaptive τ.
CONFIDENCE_M: int = 3  # number of leading prediction steps aggregated into confidence
TOP_K_INSTANCES: int = 10  # bank retrieval count when mode == predictor_plus_bank

# τ calibration window — median predictor-confidence over Phase 1 steps
# [TAU_CALIB_START, TAU_CALIB_END]. Computed at TAU_CALIB_END and frozen.
TAU_CALIB_START_STEP: int = 5_000
TAU_CALIB_END_STEP: int = 10_000

# Phase 1 checkpoint cadence (every 10k steps).
PHASE1_CKPT_EVERY: int = 10_000

# Phase 2 / Phase 3 dense early-phase checkpoints (instr §4.6). Phase-relative steps.
# Recomputed in session 4 for the new continuous-motion substrate (316 frames/loop
# vs prior 458). All five rep bins covered; 100+ bin gets 3 checkpoints inside.
PHASE_2_3_CKPT_STEPS: tuple[int, ...] = (
    0, 1_000, 2_000, 4_000, 6_500, 10_000, 15_000, 20_000, 30_000, 40_000, 55_000,
)
# Plus one final checkpoint at end-of-phase (computed at runtime from stream length).
# Prior schedule (458-frame loop): (0, 1_000, 2_500, 5_000, 9_000, 12_000, 16_000,
# 23_000, 34_000, 46_000, 55_000). Preserved here as a comment for the audit trail.

# Held-out reservation (instr §6.1, §7.3).
HELD_OUT_LOOPS: int = 10

# Probe construction (instr §6.1).
PROBES_PER_VIEWING_POSITION: int = 50  # steady-state
PROBES_PER_TRANSITION: int = 50         # cue
# Each phase: 5 steady-state items × 50 + 5 transitions × 50 = 500 probes / checkpoint.

# Variance-learned-structure gate threshold (instr §7.7 G1.3).
G1_3_LOG_VAR_SEPARATION: float = 0.3

# Cluster-sharpness floor (instr §7.7 G1.5).
G1_5_M3_FLOOR: float = 0.10

# Cluster-preservation thresholds (instr §8.7 G2.2, §9.7 G3.2).
G2_2_PRESERVATION_FRAC: float = 0.70
G3_2_PRESERVATION_FRAC: float = 0.70

# Statistical thresholds for the t-test / Wilcoxon gates (instr §7.7, §8.7, §9.7).
GATE_ALPHA: float = 0.01
NORMALITY_ALPHA: float = 0.05  # Shapiro-Wilk threshold for falling back to Wilcoxon
G1_4_GATE_HORIZON_K: int = 8   # mid-horizon principled gate step
G1_4_DIAGNOSTIC_HORIZONS: tuple[int, ...] = (1, 16)  # reported, not gated

# Shuffle S4 quantitative collapse-to-mean thresholds (instr §6.5).
S4_MEAN_NORM_MAX: float = 0.15
S4_MEAN_LOG_VAR_MIN: float = 0.4

# Frame budgets (instr §1.4, §8.3, §9.3).
PHASE_2_FRAME_BUDGET: int = 65_000
PHASE_3_FRAME_BUDGET: int = 65_000

# Phase 2 Stage A → Stage B curriculum split (instr §1.4, §8.3).
# Loops with loop_index < PHASE_2_PERTURBATION_START_LOOP run unperturbed (Stage A).
# Loops with loop_index >= PHASE_2_PERTURBATION_START_LOOP fire
# RandomizeMaterials(inRoomTypes=["LivingRoom"]) at loop start (Stage B).
PHASE_2_PERTURBATION_START_LOOP: int = 31

# In-flight transition diagnostic thresholds (instr §8.7a). SCAFFOLDING:
# recalibrate from empirical per-loop distribution observed during the first run.
TRANSITION_LOSS_SPIKE_RATIO: float = 3.0      # G2.T1
TRANSITION_LOG_VAR_WIDENING_MIN: float = 0.5  # G2.T2 (perturbed items)
TRANSITION_CONTROL_DRIFT_MAX: float = 0.3     # G2.T3 (control items)

# Item-set identifiers for the transition diagnostic.
TRANSITION_PERTURBED_ITEMS: tuple[int, ...] = (3, 4)        # Dresser, Sofa
TRANSITION_CONTROL_ITEMS: tuple[int, ...] = (1, 2, 5)       # Bed, DiningTable, Television

# Transition diagnostic evaluation window (instr §8.7a).
# Gates evaluate at end of loop 35 over loops 25-30 (baseline) and loops 31-35 (post-onset).
TRANSITION_BASELINE_LOOPS: tuple[int, int] = (25, 30)   # inclusive
TRANSITION_POST_ONSET_LOOPS: tuple[int, int] = (31, 35) # inclusive


# ---------- File-system paths (configurable; defaults at REPO_ROOT) --------

@dataclass(frozen=True)
class Paths:
    repo: Path = REPO_ROOT
    embeddings: Path = REPO_ROOT / "data/dinov2_embeddings/embeddings.npy"
    embeddings_dwell_archive: Path = (
        REPO_ROOT / "data/dinov2_embeddings/embeddings_dwell_only_v1.npy"
    )
    annotations_phase1: Path = Path(
        "/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/main/"
        "frame_annotations.jsonl"
    )
    phase2_frames: Path = REPO_ROOT / "data/phase2_frames"
    phase2_annotations: Path = REPO_ROOT / "data/phase2_annotations.jsonl"
    phase2_embeddings: Path = REPO_ROOT / "data/phase2_embeddings/embeddings.npy"
    phase3_frames: Path = REPO_ROOT / "data/phase3_frames"
    phase3_annotations: Path = REPO_ROOT / "data/phase3_annotations.jsonl"
    phase3_embeddings: Path = REPO_ROOT / "data/phase3_embeddings/embeddings.npy"
    results_phase1_main: Path = REPO_ROOT / "results/inner_pam_v0/phase1_main"
    results_phase1_shuffle: Path = REPO_ROOT / "results/inner_pam_v0/phase1_shuffle"
    results_phase2_main: Path = REPO_ROOT / "results/inner_pam_v0/phase2_main"
    results_phase2_shuffle: Path = REPO_ROOT / "results/inner_pam_v0/phase2_shuffle"
    results_phase2_preflight: Path = REPO_ROOT / "results/inner_pam_v0/phase2_preflight"
    results_phase3_main: Path = REPO_ROOT / "results/inner_pam_v0/phase3_main"
    results_phase3_shuffle: Path = REPO_ROOT / "results/inner_pam_v0/phase3_shuffle"
    results_phase3_preflight: Path = REPO_ROOT / "results/inner_pam_v0/phase3_preflight"
    results_developmental_arc: Path = (
        REPO_ROOT / "results/inner_pam_v0/developmental_arc"
    )


PATHS = Paths()


# ---------- Per-phase configuration ----------------------------------------

@dataclass(frozen=True)
class PhaseConfig:
    name: str
    embeddings: Path
    annotations: Path
    results_main: Path
    results_shuffle: Path
    ckpt_phase_relative_steps: tuple[int, ...]
    ckpt_every: Optional[int] = None  # used by Phase 1 only
    perturbation_tag: str = "none"
    loaded_from_phase: Optional[str] = None  # name of phase to resume from
    expected_loops: Optional[int] = None
    loop_length_estimate: int = 316    # continuous-motion substrate (session-4 calibration)


PHASE1 = PhaseConfig(
    name="phase1",
    embeddings=PATHS.embeddings,
    annotations=PATHS.annotations_phase1,
    results_main=PATHS.results_phase1_main,
    results_shuffle=PATHS.results_phase1_shuffle,
    ckpt_phase_relative_steps=tuple(),  # Phase 1 uses ckpt_every instead
    ckpt_every=PHASE1_CKPT_EVERY,
    perturbation_tag="none",
    loaded_from_phase=None,
    expected_loops=218,
    loop_length_estimate=458,  # Phase 1 substrate (substrate-degenerate baseline)
)

# Phase 2 starts from a freshly-initialised predictor (Phase 1 discarded as
# substrate-degenerate; session-4 disposition). Phase 3 resumes from Phase 2.
PHASE2 = PhaseConfig(
    name="phase2",
    embeddings=PATHS.phase2_embeddings,
    annotations=PATHS.phase2_annotations,
    results_main=PATHS.results_phase2_main,
    results_shuffle=PATHS.results_phase2_shuffle,
    ckpt_phase_relative_steps=PHASE_2_3_CKPT_STEPS,
    ckpt_every=None,
    perturbation_tag="livingroom_retexture",
    loaded_from_phase=None,
    expected_loops=205,
)

PHASE3 = PhaseConfig(
    name="phase3",
    embeddings=PATHS.phase3_embeddings,
    annotations=PATHS.phase3_annotations,
    results_main=PATHS.results_phase3_main,
    results_shuffle=PATHS.results_phase3_shuffle,
    ckpt_phase_relative_steps=PHASE_2_3_CKPT_STEPS,
    ckpt_every=None,
    perturbation_tag="phase3_preflight_selected",  # asset_replacement | full_house_retexture
    loaded_from_phase="phase2",
    expected_loops=205,
)


# ---------- Five-item route metadata --------------------------------------

# Order is deterministic (Bed → DiningTable → Dresser → Sofa → Television).
# viewing_position_id mapping matches the seed-7 furniture-run annotations.
VIEWING_POSITION_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)
VIEWING_POSITION_LABELS: dict[int, str] = {
    1: "Bed",
    2: "DiningTable",
    3: "Dresser",
    4: "Sofa",
    5: "Television",
}
LIVINGROOM_VIEWING_POSITIONS: tuple[int, ...] = (3, 4)        # Dresser, Sofa
BEDROOM_VIEWING_POSITIONS: tuple[int, ...] = (1, 2, 5)        # Bed, DiningTable, Television

ROUTE_TRANSITIONS: tuple[tuple[int, int], ...] = (
    (1, 2),  # Bed → DiningTable
    (2, 3),  # DiningTable → Dresser
    (3, 4),  # Dresser → Sofa
    (4, 5),  # Sofa → Television
    (5, 1),  # Television → Bed (route loops)
)


# ---------- Random seeds ---------------------------------------------------

# Each component carries its own seed so single-variable changes don't reseed others.
SEED_PREDICTOR_INIT: int = 0
SEED_SHUFFLE_PERMUTATION: int = 0
SEED_PROBE_SAMPLING: int = 13
SEED_ENCODE_CONSISTENCY: int = 42
