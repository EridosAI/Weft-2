"""V2 Phase 1 — locked construction sweep grid (the §5 rewrite).

Phase 1 sweeps CONSTRUCTION parameters and measures the §4 properties post-hoc;
the working-region map keeps measured-property axis labels (assigned via
`v2.src.protocol.grid_mapping.nearest_grid_point`). This module is the single
source of truth for the per-cell `StreamParams`.

PROVENANCE (surface-and-confirm, not spec drift):
  * Axis density (3 values/axis), cross structure (low/mid/high), and the
    measured-property bins are Phase-0.5 design-chat commitments.
  * `manifold_dim_D` grid was revised {4,32,256} -> {4,16,128} during a
    design-chat-within-Phase-1 (2026-05-22): the original D=256 endpoint is both
    construction-infeasible across most periods (needs P>=512) and outside the
    PRE-A-validated D range. See `results/phase1/grid_calibration.json`
    -> `grid_revision_provenance` for the full revision note + fact-check.
  * `CONTINUITY_CENTER` values were calibrated by
    `scripts/run_phase1_grid_calibration.py` and accepted as-is by the design
    chat; realised measured-C is recorded per cell post-hoc.
  * (D=128, P=32) is dropped -> `not_characterised` (the lone P<2D cell).

Every fixed value below is SCAFFOLDING tagged with its derivation.
"""

from __future__ import annotations

from typing import Optional

from v2.src.substrate.stream_builder import StreamParams

# The five §4 axes, keyed by their construction handle in this module.
AXES = ("mag", "loc", "cont", "period", "dim")
LEVELS = ("low", "mid", "high")           # the 3 per-axis values == the 3 cross positions

# SCAFFOLDING — construction-parameter grids (Phase-0.5; D re-anchored 2026-05-22).
MAGNITUDE_M = {"low": 0.1, "mid": 0.3, "high": 0.7}    # magnitude_M dial (spec §4.1)
LOCALITY_L = {"low": 0.3, "mid": 0.5, "high": 0.9}     # locality_L dial (spec §4.2)
PERIOD_P = {"low": 32, "mid": 256, "high": 2048}       # period_P (spec §4.4), log
MANIFOLD_D = {"low": 4, "mid": 16, "high": 128}        # manifold_dim_D (spec §4.5), log

# SCAFFOLDING — fixed (not swept): fidelity per cell (Phase-0.5 D4; PRE-D1a/D2 MID).
FIDELITY_F = 0.97

# SCAFFOLDING — continuity_center per (period_P, level), calibrated to land nearest
# the §5 measured-C bins. Source: results/phase1/grid_calibration.json (section B,
# accepted as-is). Realised measured-C (magnitude=0 reference) in trailing comments.
CONTINUITY_CENTER = {
    32:   {"low": 3,   "mid": 5,   "high": 7},     # C ~ 0.153 / 0.417 / 0.729 (0.077 unreachable at P=32,D=4; floor 0.153)
    256:  {"low": 19,  "mid": 39,  "high": 59},    # C ~ 0.111 / 0.425 / 0.869
    2048: {"low": 146, "mid": 309, "high": 472},   # C ~ 0.105 / 0.421 / 0.877
}
# Measured-C bin labels the map's continuity axis carries (the §5 "values").
CONTINUITY_TARGET_C = {"low": 0.077, "mid": 0.4, "high": 0.8}

L_D_MAIN = (1, 2, 4)                      # Phase-0.5 (intermediate L_d=2 added; Jason push-back)


def is_feasible(D: int, P: int) -> bool:
    """centered_harmonics needs D distinct harmonics in [1, P//2] => P >= 2D."""
    return D <= P // 2


def cell_params(mag: str, loc: str, cont: str, period: str, dim: str,
                fidelity_F: float = FIDELITY_F) -> Optional[StreamParams]:
    """StreamParams for one cell given per-axis LEVELS, or None if infeasible (dropped).

    continuity_center is looked up by (period_P, continuity level): the calibrated
    table is ~D-independent except at the D==P//2 boundary (whole-band fill), where
    the held continuity cannot be honoured — the post-hoc measurement records the
    actual C there.
    """
    P, D = PERIOD_P[period], MANIFOLD_D[dim]
    if not is_feasible(D, P):
        return None
    return StreamParams(
        magnitude_M=MAGNITUDE_M[mag],
        locality_L=LOCALITY_L[loc],
        period_P=P,
        manifold_dim_D=D,
        continuity_center=CONTINUITY_CENTER[P][cont],
        fidelity_F=fidelity_F,
    )


def _cell_id(L_d_main: int, swept_axis: str, cross: str, levels: dict) -> str:
    lv = "_".join(f"{a}{levels[a][0]}" for a in AXES)   # e.g. magl_locm_...
    return f"Ld{L_d_main}__sweep-{swept_axis}__cross-{cross}__{lv}"


def main_effects_cells(l_d_values=L_D_MAIN) -> list[dict]:
    """The §3.5 cross-structure: per L_d, per cross (low/mid/high), sweep each axis
    at its 3 values with the other 4 held at the cross position.

    Per L_d: 3 cross x 5 axes x 3 values = 45 cell-measurements (anchors collapse:
    the all-low/all-mid/all-high corners recur 5x each across the axis sweeps).
    Infeasible (D=128,P=32) cells are flagged feasible=False (dropped downstream).
    """
    cells = []
    for L_d in l_d_values:
        for cross in LEVELS:
            base = {a: cross for a in AXES}
            for axis in AXES:
                for val in LEVELS:
                    levels = dict(base)
                    levels[axis] = val
                    params = cell_params(**levels)
                    cells.append({
                        "cell_id": _cell_id(L_d, axis, cross, levels),
                        "L_d_main": L_d,
                        "swept_axis": axis,
                        "cross": cross,
                        "levels": levels,
                        "feasible": params is not None,
                        "is_anchor": all(levels[a] == cross for a in AXES),
                        "params": params,
                    })
    return cells
