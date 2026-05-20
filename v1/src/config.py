"""Hyperparameters and labels for Weft Inner PAM v1.

Every fixed value is tagged ARCHITECTURE or SCAFFOLDING per
`research_operations_v1.md` §7.1 and the §11 scaffolding inventory of
`WEFT_INNER_PAM_v1_EXPERIMENT_INSTRUCTIONS.md`.

ARCHITECTURE = removing it would break a load-bearing claim in
`WEFT_INNER_PAM_v1_Spec_pass1_sections_1_to_6.md` or
`WEFT_INNER_PAM_v1_Spec_pass2_sections_7_to_11.md` §§1-6 / §7-§10.

SCAFFOLDING = a tractable default with a removal plan in the §11
scaffolding inventory.

PRE-B / PRE-C constants (V1_PERTURBATION_MECHANISM, V1_FRAMES_PER_LOOP,
V1_DECODER_N_LAYERS) are written here by preflight scripts and imported
by training scripts. Until preflight runs, they raise on access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
V1_ROOT = REPO_ROOT / "v1"


# ---------- ARCHITECTURE ---------------------------------------------------
# Spec §§2, 7.1. Inherited from v0 unchanged.

EMBED_DIM: int = 1024              # DINOv2-Large CLS output dimension
WINDOW_W: int = 16                 # Sliding window length (spec §2.2 commitment 9)
PREDICT_K: int = 16                # K-step prediction horizon

# ProcTHOR-10K revision pin for v1.
#
# 2026-05-19 design-chat build-configuration investigation, Reading C
# resolution: the AI2-THOR 5.0.0 build prints a version-mismatch warning
# at every load_house call recommending either an ai2thor upgrade
# (5.0.0 IS the latest on PyPI; no upgrade exists) or a procthor-10k
# downgrade to `ab3cacd0fc17754d4c080a3fd50b18395fae8647`. CC tested the
# downgrade empirically and found that the ab3cacd revision is
# INCOMPATIBLE with ai2thor==5.0.0's scene loader: controller init
# silently produces a broken state (agent at y=-38.86 in scene void,
# 0 objects in metadata, GetReachablePositions returns 0 positions).
# The downgrade target is the OLDER procthor-10k revision; AI2-THOR
# 5.0.0 only loads the newer (currently-installed) 439193522... revision
# properly. The warning's downgrade recommendation is misleading.
#
# Conclusion: the current build (procthor-10k 4391935... + ai2thor 5.0.0)
# is the only viable configuration. V1_PROCTHOR_REVISION is set to None
# so load_house uses the latest cached revision (=4391935...). The
# DisableObject / HideObject render-NO-OP behavior observed in the
# asset-replacement smoke test is therefore an API-limitation property
# of ai2thor==5.0.0 itself, not a build misconfiguration.
V1_PROCTHOR_REVISION: str | None = None


# ---------- SCAFFOLDING (architecture hyperparameters) ---------------------
# Spec §7.1; §11 scaffolding inventory rows for hidden/n_layers/n_heads/mlp_dim.

PRED_HIDDEN: int = 512
PRED_LAYERS: int = 4
PRED_HEADS: int = 8
PRED_MLP_DIM: int = 2048
PRED_ACTIVATION: str = "gelu"


# ---------- SCAFFOLDING (loss numerical hygiene) ---------------------------
# Spec §7.1, instr §3. Inherited from v0.

LOG_VAR_CLAMP_MIN: float = -10.0
LOG_VAR_CLAMP_MAX: float = 10.0


# ---------- SCAFFOLDING (optimiser) ----------------------------------------
# Spec §9.4, instr §4.4. Inherited from v0; v2 may revisit if V1C/V1D/V1P.

LR: float = 3e-4
WEIGHT_DECAY: float = 0.01
ADAM_BETAS: tuple[float, float] = (0.9, 0.999)
GRAD_CLIP_MAX_NORM: float = 1.0


# ---------- SCAFFOLDING (stage frame budgets) ------------------------------
# Spec §9.2; instr §7.1, §8.1.

STAGE_A_FRAME_BUDGET: int = 100_000          # spec §9.2
STAGE_B_MIN_LOOPS: int = 200                  # instr §8.1
STAGE_B_MAX_LOOPS: int = 400                  # instr §8.1
STAGE_B_STABILITY_EPSILON_NAT: float = 0.05   # instr §8.1; recalibrated post-Primary


# ---------- SCAFFOLDING (checkpoint cadence) -------------------------------
# Instr §4.5.

STAGE_A_DENSE_CHECKPOINTS: tuple[int, ...] = (1_000, 2_000, 4_000, 6_500, 10_000)
STAGE_A_STANDARD_STEP: int = 10_000   # every 10k after dense regime
STAGE_B_STANDARD_STEP: int = 10_000


# ---------- SCAFFOLDING (substrate / perturbation thresholds) -------------
# Spec §6.1, §8.2.1; instr §6.1, §6.2.

SUBSTRATE_CROSS_INSTANCE_STABILITY_MIN: float = 0.75   # §5.1
SUBSTRATE_CROSS_ELEMENT_DISTINGUISH_MAX: float = 0.60  # §5.2
SUBSTRATE_COMBINED_GAP_MIN: float = 0.15               # §5.3

# Finding 3 (continuous-motion check): pair-wise cosine threshold.
# v0-inherited; recalibrated 0.9999 → 0.999 on 2026-05-19 per design-chat
# determination. Reason: the strict 0.9999 catches the 25 structural-by-design
# transit→close_up handoff duplicates (5 items × 5 loops × 1 pair = 25),
# which are not 30-frame static dwell. v0 STOP_REPORT (seed-7 rerender,
# 2026-05-12) set the precedent for the 0.9999 → 0.999 recalibration on a
# closely-related determinism check. v2 candidate: run-length-aware check
# that separates dwell from segment-handoff without threshold recalibration
# (recorded in WEFT_INNER_PAM_v2_DESIGN_INTAKE.md).
FINDING_3_COS_MAX: float = 0.999

PERTURBATION_MAGNITUDE_MIN: float = 0.05               # spec §1.2 commitment 3
PERTURBATION_MAGNITUDE_MAX: float = 0.10
PERTURBATION_LOCALITY_MAX: float = 0.015               # spec §8.2.1.2
PERTURBATION_REPRODUCIBILITY_TOL: float = 0.005        # spec §8.3.5
PERTURBATION_IN_FLIGHT_DRIFT_TOL: float = 0.01         # spec §11.2 cond 6


# ---------- SCAFFOLDING (PRE-C decoder-layer selection) -------------------
# Instr §6.3.2.

PRE_C_L_D_CANDIDATES: tuple[int, ...] = (1, 2, 3, 4)
PRE_C_CALIBRATION_FRAMES: int = 30_000
PRE_C_CALIBRATION_CHECKPOINTS: tuple[int, ...] = (1_000, 5_000, 10_000, 20_000, 30_000)
PRE_C_LOSS_SMOOTHNESS_RATIO_MAX: float = 0.5
PRE_C_TIEBREAK_QUERY_COSINE_BAND: float = 0.02


# ---------- SCAFFOLDING (verdict percentile thresholds) -------------------
# Spec §10.4.3; instr §5.4.

VERDICT_STABILITY_PERCENTILE: int = 75          # of bit-identical distribution
VERDICT_DIFFERENTIATION_PERCENTILE: int = 25    # of input-varying distribution


# ---------- SCAFFOLDING (decoder-query initialisation) --------------------
# Spec §7.1.

OUTPUT_QUERY_INIT_STD: float = 0.02


# ---------- Random seeds ---------------------------------------------------
# Each component carries its own seed so single-variable changes don't reseed
# the others (inherited from v0 discipline).

SEED_PREDICTOR_INIT: int = 0
SEED_PROBE_SAMPLING: int = 13
SEED_ENCODE_CONSISTENCY: int = 42


# ---------- Five-item route metadata --------------------------------------
# Inherited from v0 unchanged. Order: Bed → DiningTable → Dresser → Sofa →
# Television (instr §1.3). viewing_position_id mapping matches the seed-7
# furniture-run annotations.

VIEWING_POSITION_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)
VIEWING_POSITION_LABELS: dict[int, str] = {
    1: "Bed",
    2: "DiningTable",
    3: "Dresser",
    4: "Sofa",
    5: "Television",
}
LIVINGROOM_VIEWING_POSITIONS: tuple[int, ...] = (3, 4)
BEDROOM_VIEWING_POSITIONS: tuple[int, ...] = (1, 2, 5)


# ---------- File-system paths ----------------------------------------------

@dataclass(frozen=True)
class Paths:
    repo: Path = REPO_ROOT
    v1_root: Path = V1_ROOT

    # Route (inherited from v0)
    route_phase2: Path = REPO_ROOT / "v0" / "data" / "route_phase2.json"

    # v1-specific data
    data_shared: Path = V1_ROOT / "data" / "v1_shared"
    embeddings_stage_a: Path = V1_ROOT / "data" / "v1_shared" / "embeddings_stage_a.npy"
    embeddings_stage_b: Path = V1_ROOT / "data" / "v1_shared" / "embeddings_stage_b.npy"
    annotations_stage_a: Path = (
        V1_ROOT / "data" / "v1_shared" / "annotations_stage_a.jsonl"
    )
    annotations_stage_b: Path = (
        V1_ROOT / "data" / "v1_shared" / "annotations_stage_b.jsonl"
    )
    frames_stage_a: Path = V1_ROOT / "data" / "v1_shared" / "frames_stage_a"
    frames_stage_b: Path = V1_ROOT / "data" / "v1_shared" / "frames_stage_b"
    bit_identical_item_ordinal: Path = (
        V1_ROOT / "data" / "v1_shared" / "bit_identical_item_ordinal.json"
    )

    # PRE-C calibration subset
    pre_c_data: Path = V1_ROOT / "data" / "pre_c_calibration"

    # Results
    results_root: Path = V1_ROOT / "results" / "inner_pam_v1"
    results_pre_a: Path = V1_ROOT / "results" / "inner_pam_v1" / "pre_a_substrate"
    results_pre_b: Path = V1_ROOT / "results" / "inner_pam_v1" / "pre_b_perturbation"
    results_pre_c: Path = (
        V1_ROOT / "results" / "inner_pam_v1" / "pre_c_decoder_calibration"
    )
    results_pre_d: Path = (
        V1_ROOT / "results" / "inner_pam_v1" / "pre_d_arch_assertions"
    )
    results_arm_primary: Path = V1_ROOT / "results" / "inner_pam_v1" / "arm_primary"
    results_arm_ablation1: Path = (
        V1_ROOT / "results" / "inner_pam_v1" / "arm_ablation1"
    )
    results_arm_ablation2: Path = (
        V1_ROOT / "results" / "inner_pam_v1" / "arm_ablation2"
    )
    results_matrix: Path = (
        V1_ROOT / "results" / "inner_pam_v1" / "arm_comparison_matrix"
    )
    results_thresholds: Path = (
        V1_ROOT / "results" / "inner_pam_v1" / "threshold_calibration"
    )
    results_verdict: Path = (
        V1_ROOT / "results" / "inner_pam_v1" / "verdict_recommendation"
    )


PATHS = Paths()


# ---------- PRE-B / PRE-C runtime-locked constants -------------------------
# Spec §7.2.1, §8.2.3, instr §6.2.4, §6.3.4.
#
# These are written to disk by the preflight scripts and read here at import
# time. If preflight has not run, accessing them raises a descriptive error
# (no silent default — spec §7.2.1's "no silent default" rule).
#
# The on-disk locations:
#   PRE-B: results_pre_b / "selected.json"  with keys: mechanism, frames_per_loop
#   PRE-C: results_pre_c / "selected.json"  with keys: decoder_n_layers


_PRE_B_LOCK = PATHS.results_pre_b / "selected.json"
_PRE_C_LOCK = PATHS.results_pre_c / "selected.json"


def _load_pre_b() -> dict:
    if not _PRE_B_LOCK.exists():
        raise FileNotFoundError(
            f"PRE-B has not selected a perturbation mechanism yet. Run "
            f"scripts/run_pre_b_perturbation.py first; the selection is "
            f"written to {_PRE_B_LOCK}."
        )
    return json.loads(_PRE_B_LOCK.read_text())


def _load_pre_c() -> dict:
    if not _PRE_C_LOCK.exists():
        raise FileNotFoundError(
            f"PRE-C has not selected a decoder_n_layers value yet. Run "
            f"scripts/run_pre_c_decoder_calibration.py first; the selection "
            f"is written to {_PRE_C_LOCK}."
        )
    return json.loads(_PRE_C_LOCK.read_text())


def get_v1_perturbation_mechanism() -> str:
    """Returns the PRE-B-selected perturbation mechanism identifier.

    One of: 'per_object_material_setting', 'asset_replacement',
    'hand_built_texture_swaps', 'alternate_procthor_scene'.
    """
    return _load_pre_b()["mechanism"]


def get_v1_frames_per_loop() -> int:
    """Returns the PRE-B-measured loop length under the selected mechanism."""
    return int(_load_pre_b()["frames_per_loop"])


def get_v1_stage_b_min_frames() -> int:
    """STAGE_B_MIN_LOOPS × frames_per_loop, per instr §8.1 step 1."""
    return STAGE_B_MIN_LOOPS * get_v1_frames_per_loop()


def get_v1_stage_b_max_frames() -> int:
    """STAGE_B_MAX_LOOPS × frames_per_loop, per instr §8.1 step 5."""
    return STAGE_B_MAX_LOOPS * get_v1_frames_per_loop()


def get_v1_decoder_n_layers() -> int:
    """Returns the PRE-C-selected decoder_n_layers value."""
    return int(_load_pre_c()["decoder_n_layers"])


def stage_a_checkpoint_steps() -> tuple[int, ...]:
    """Stage A checkpoint steps: dense early + every 10k thereafter through end.

    Returns step indices (relative to Stage A start). End-of-Stage-A is
    appended as the final step.
    """
    end = STAGE_A_FRAME_BUDGET - WINDOW_W + 1  # spec §4.3 skip-until-W
    standard = tuple(
        s for s in range(STAGE_A_STANDARD_STEP * 2, end, STAGE_A_STANDARD_STEP)
    )
    return STAGE_A_DENSE_CHECKPOINTS + standard + (end,)
