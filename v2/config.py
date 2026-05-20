"""Weft Inner PAM v2 — configuration constants and SCAFFOLDING gating.

Every fixed value is tagged ARCHITECTURE or SCAFFOLDING per
`research_operations_v1.md` §§7.1-7.3, mirroring v1's `v1/src/config.py`.

ARCHITECTURE = removing it would break a load-bearing v2 commitment
(spec pass 1 §2.6 ARCHITECTURE list; pass 2 §5.5 ARCHITECTURE list).

SCAFFOLDING = a tractable default with a calibration plan. SCAFFOLDING
values that V2-PRE-E calibrates carry placeholder defaults usable by
PRE-A's sanity sweep; PRE-E overwrites them and records the source in
`v2/results/pre_e/scaffolding_calibration.json`.

V2_TRAINING_STEPS is the exception: it is calibrated at V2-PRE-A (smoke
run) and lock-file-gated (raises on access until written), because
PRE-D1a / PRE-D2 must not train on a guessed step count (instr §7.1).

v2 inherits the architecture scalars (EMBED_DIM, WINDOW_W, PREDICT_K)
from the frozen v1 scaffold via the library-import model (spec §1.6).
"""

from __future__ import annotations

import json
from pathlib import Path

from v1.src.config import EMBED_DIM, PREDICT_K, WINDOW_W  # ARCHITECTURE, inherited

# --------------------------------------------------------------------------
# Paths (derived from this file's location; repo path contains a space)
# --------------------------------------------------------------------------

V2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V2_ROOT.parent

RESULTS_ROOT = V2_ROOT / "results"
DATA_ROOT = V2_ROOT / "data"
LOGS_ROOT = V2_ROOT / "logs"

RESULTS_PRE_A = RESULTS_ROOT / "pre_a"
RESULTS_PRE_B = RESULTS_ROOT / "pre_b"
RESULTS_PRE_C = RESULTS_ROOT / "pre_c"
RESULTS_PRE_D1A = RESULTS_ROOT / "pre_d1a"
RESULTS_PRE_D1C = RESULTS_ROOT / "pre_d1c"
RESULTS_PRE_D2 = RESULTS_ROOT / "pre_d2"
RESULTS_PRE_E = RESULTS_ROOT / "pre_e"


# --------------------------------------------------------------------------
# ARCHITECTURE — v2 load-bearing commitments (spec pass 1 §2.6 / pass 2 §5.5)
# --------------------------------------------------------------------------

D_AMBIENT: int = EMBED_DIM        # 1024; Option β synthetic streams in d=1024 (spec §2.1)
# WINDOW_W = 16, PREDICT_K = 16 inherited above (spec §2.6 ARCHITECTURE).

# The §5.1-5.3 primitive structure is the v2 substrate; not swept, not ablated.


# --------------------------------------------------------------------------
# SCAFFOLDING — substrate construction (spec pass 2 §5.5; calibrated at PRE-A/E)
# --------------------------------------------------------------------------

# Random orthogonal embedding U ∈ R^{1024×1024}; generated once, persisted,
# reloaded across all v2 runs (spec §5.5). For a manifold of dimensionality D
# the construction uses the first D columns U[:, :D] (orthonormal R^D→R^1024).
EMBEDDING_U_SEED: int = 0                       # SCAFFOLDING — fixed for reproducibility
EMBEDDING_U_PATH: Path = DATA_ROOT / "embedding_U.npy"

# Per-position noise ε_t magnitude (spec §5.1; not a swept axis). Small default
# keeps PRE-A property control clean; PRE-E calibrates against substrate floor.
EPS_NOISE_STD: float = 1e-3                     # SCAFFOLDING — calibrated at PRE-E

# Shift-vector δ direction (spec §5.3): random isotropic in R^1024 (pass-2 default).
DELTA_DIRECTION: str = "random_isotropic"       # SCAFFOLDING — spec §5.3 default

# Base per-coordinate harmonic seed (phases ψ_k and harmonic assignment).
SUBSTRATE_SEED: int = 7                          # SCAFFOLDING — fixed for reproducibility


# --------------------------------------------------------------------------
# SCAFFOLDING — measurement thresholds (spec §§4/6/7; calibrated at PRE-E)
#
# Placeholder defaults: usable by PRE-A's sanity sweep and PRE-B. PRE-E
# overwrites these (instr §7.6) and records provenance in
# results/pre_e/scaffolding_calibration.json.
# --------------------------------------------------------------------------

TAU_R: float = 0.6              # SCAFFOLDING — calibrated at PRE-E — repetition similarity threshold (§4.4)
REPETITION_NOISE_FLOOR: float = 0.3   # SCAFFOLDING — calibrated at PRE-E — off-diagonal "no repetition" floor (§4.4)
TAU_L: float = 0.05             # SCAFFOLDING — calibrated at PRE-E — locality shift threshold (§4.2)
TAU_PERT: float = 0.25          # SCAFFOLDING — calibrated at PRE-E — perturbation-detection residual threshold (§6.2 step 4); placeholder set above the fidelity-noise residual floor (~sqrt(2(1-F))) and below the perturbation residual (~sqrt(2M))
REPETITION_COVERAGE_THRESHOLD: float = 0.5   # SCAFFOLDING — calibrated at PRE-E — locality-undefined floor (§6.2 step 3)
BIC_IMPROVEMENT_THRESHOLD: float = 10.0      # SCAFFOLDING — calibrated at PRE-E — multimodal GMM/BIC (§6.3)
LOCAL_PCA_WINDOW: int = 64      # SCAFFOLDING — calibrated at PRE-E — local-PCA window for manifold dim (§4.5)
MANIFOLD_SUBSAMPLE_RATE: int = 4   # SCAFFOLDING — calibrated at PRE-E — local-PCA position subsampling (§4.5)
REF_NEIGHBOURHOOD_WINDOW: int = 3  # SCAFFOLDING — calibrated at PRE-E — reference-neighbour radius for §4.1/§4.2
BLOCK_SIZE_LONG: int = 20000    # SCAFFOLDING — calibrated at PRE-E — block size for self-sim on |τ|>1e5 (§6.2 step 1)

# τ_W (co-primary working-region threshold, §7.3) — calibrated PER-HEAD at PRE-E
# via Wilcoxon p<0.05 vs the bit-identical baseline from PRE-D1a. No usable
# placeholder: PRE-A does not consume τ_W. Read via results/pre_e output.
# (Listed here for the SCAFFOLDING inventory; not a numeric default.)


# --------------------------------------------------------------------------
# SCAFFOLDING — PRE-A construction-vs-measurement tolerance (instr §7.1 step 6)
# --------------------------------------------------------------------------

CONSTRUCTION_MEASUREMENT_TOLERANCE: float = 0.10   # SCAFFOLDING — placeholder 10% relative; PRE-E refines


# --------------------------------------------------------------------------
# SCAFFOLDING — stream length (spec §2.6; analogous to v0 frames-per-loop)
# --------------------------------------------------------------------------

# Default number of repetitions tiled by the §5.2 repetition primitive. Stream
# length = REPETITIONS × P. Enough repetitions for reference-state estimation
# (§6.2 step 3 needs multiple traversals of each period position).
DEFAULT_N_REPETITIONS: int = 12       # SCAFFOLDING — calibrated at PRE-E


# --------------------------------------------------------------------------
# SCAFFOLDING — smoke-run training calibration (instr §7.1 PRE-A smoke test)
# --------------------------------------------------------------------------

# Max steps the PRE-A smoke run trains while observing the loss trajectory to
# locate a plateau. Not the production step count — that is V2_TRAINING_STEPS,
# derived from this run and lock-file-gated below.
V2_SMOKE_MAX_STEPS: int = 40_000      # SCAFFOLDING — PRE-A smoke-run ceiling
V2_SMOKE_CHECKPOINT_EVERY: int = 2_000  # SCAFFOLDING — smoke-run loss-trajectory cadence
# Plateau: first checkpoint whose relative interval-loss improvement over the
# previous interval falls below this and stays below for the rest of the run.
V2_PLATEAU_REL_IMPROVEMENT: float = 0.02   # SCAFFOLDING — plateau-detection threshold


# --------------------------------------------------------------------------
# V2_TRAINING_STEPS — lock-file-gated (calibrated at PRE-A smoke run)
# --------------------------------------------------------------------------

_V2_TRAINING_STEPS_LOCK = RESULTS_PRE_A / "v2_training_steps.json"


def get_v2_training_steps() -> int:
    """Return the PRE-A-calibrated production training-step count.

    Raises until the PRE-A smoke run writes the lock file — PRE-D1a / PRE-D2
    must not train on a guessed step count (instr §7.1; mirrors v1 PRE-C).
    """
    if not _V2_TRAINING_STEPS_LOCK.exists():
        raise FileNotFoundError(
            "V2_TRAINING_STEPS has not been calibrated yet. Run the PRE-A "
            "smoke test (scripts/run_pre_a.py) first; the calibrated value is "
            f"written to {_V2_TRAINING_STEPS_LOCK}."
        )
    return int(json.loads(_V2_TRAINING_STEPS_LOCK.read_text())["v2_training_steps"])


def write_v2_training_steps(value: int, rationale: dict) -> None:
    """Write the lock file (called once by the PRE-A smoke run)."""
    _V2_TRAINING_STEPS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    payload = {"v2_training_steps": int(value), **rationale}
    _V2_TRAINING_STEPS_LOCK.write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------
# PRE-E calibrated-threshold loader (placeholder-aware)
# --------------------------------------------------------------------------

_PRE_E_LOCK = RESULTS_PRE_E / "scaffolding_calibration.json"


def pre_e_calibrated() -> bool:
    """True once PRE-E has written calibrated thresholds."""
    return _PRE_E_LOCK.exists()


def load_calibrated_thresholds() -> dict:
    """Return PRE-E-calibrated thresholds if available, else module placeholders.

    PRE-A and PRE-B run before PRE-E and use the placeholder defaults above;
    Phase 1 / PRE-D2 consume the calibrated values once PRE-E has run.
    """
    placeholders = {
        "TAU_R": TAU_R,
        "REPETITION_NOISE_FLOOR": REPETITION_NOISE_FLOOR,
        "TAU_L": TAU_L,
        "TAU_PERT": TAU_PERT,
        "REPETITION_COVERAGE_THRESHOLD": REPETITION_COVERAGE_THRESHOLD,
        "BIC_IMPROVEMENT_THRESHOLD": BIC_IMPROVEMENT_THRESHOLD,
        "LOCAL_PCA_WINDOW": LOCAL_PCA_WINDOW,
        "MANIFOLD_SUBSAMPLE_RATE": MANIFOLD_SUBSAMPLE_RATE,
        "REF_NEIGHBOURHOOD_WINDOW": REF_NEIGHBOURHOOD_WINDOW,
        "BLOCK_SIZE_LONG": BLOCK_SIZE_LONG,
    }
    if not _PRE_E_LOCK.exists():
        return placeholders
    calibrated = json.loads(_PRE_E_LOCK.read_text()).get("thresholds", {})
    placeholders.update({k: v for k, v in calibrated.items() if k in placeholders})
    return placeholders
