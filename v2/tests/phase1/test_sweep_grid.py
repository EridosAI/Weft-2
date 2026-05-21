"""Unit tests for the locked Phase 1 construction sweep grid (the §5 rewrite)."""

from __future__ import annotations

from v2.src.phase1 import sweep_grid as sg
from v2.src.substrate.base_manifold_trajectory import centered_harmonics
from v2.src.substrate.stream_builder import StreamParams


def test_grids_match_locked_values():
    assert sg.MANIFOLD_D == {"low": 4, "mid": 16, "high": 128}        # revised from {4,32,256}
    assert sg.PERIOD_P == {"low": 32, "mid": 256, "high": 2048}
    assert sg.MAGNITUDE_M == {"low": 0.1, "mid": 0.3, "high": 0.7}
    assert sg.LOCALITY_L == {"low": 0.3, "mid": 0.5, "high": 0.9}
    assert sg.CONTINUITY_CENTER == {
        32: {"low": 3, "mid": 5, "high": 7},
        256: {"low": 19, "mid": 39, "high": 59},
        2048: {"low": 146, "mid": 309, "high": 472},
    }


def test_feasibility_rule():
    assert sg.is_feasible(128, 256) and sg.is_feasible(128, 2048)
    assert sg.is_feasible(16, 32) and sg.is_feasible(4, 32)
    assert not sg.is_feasible(128, 32)            # the lone dropped cell
    assert not sg.is_feasible(256, 256)           # (would-be) original-grid infeasible


def test_only_128_32_is_infeasible_in_grid():
    infeasible = {(sg.MANIFOLD_D[d], sg.PERIOD_P[p])
                  for d in sg.LEVELS for p in sg.LEVELS
                  if not sg.is_feasible(sg.MANIFOLD_D[d], sg.PERIOD_P[p])}
    assert infeasible == {(128, 32)}


def test_cell_params_drops_infeasible():
    # dim=high (D=128) + period=low (P=32) -> dropped
    assert sg.cell_params("mid", "mid", "mid", "low", "high") is None
    p = sg.cell_params("mid", "mid", "mid", "mid", "mid")
    assert isinstance(p, StreamParams)
    assert p.period_P == 256 and p.manifold_dim_D == 16
    assert p.continuity_center == 39 and p.magnitude_M == 0.3 and p.locality_L == 0.5
    assert p.fidelity_F == sg.FIDELITY_F


def test_continuity_center_lookup_is_period_keyed():
    p_lo = sg.cell_params("low", "low", "high", "low", "low")    # P=32, cont high -> center 7
    p_hi = sg.cell_params("low", "low", "high", "high", "high")  # P=2048, cont high -> center 472
    assert p_lo.continuity_center == 7
    assert p_hi.continuity_center == 472


def test_every_locked_continuity_center_is_constructable():
    # Each (period, center) at the cross-D for that period must satisfy the harmonic band.
    cross_D = {32: 4, 256: 16, 2048: 128}
    for P, by_level in sg.CONTINUITY_CENTER.items():
        for center in by_level.values():
            h = centered_harmonics(cross_D[P], P, center)        # raises if infeasible
            assert h.min() >= 1 and h.max() <= P // 2


def test_main_effects_cell_counts():
    cells = sg.main_effects_cells()
    assert len(cells) == 135                      # 3 L_d x 3 cross x 5 axes x 3 values
    per_ld = [c for c in cells if c["L_d_main"] == 2]
    assert len(per_ld) == 45
    feasible = [c for c in per_ld if c["feasible"]]
    dropped = [c for c in per_ld if not c["feasible"]]
    assert len(dropped) == 2                      # (128,32) via dim-sweep@low & period-sweep@high
    assert len(feasible) == 43
    # the two drops are exactly the (D=128,P=32) combinations
    for c in dropped:
        assert sg.MANIFOLD_D[c["levels"]["dim"]] == 128
        assert sg.PERIOD_P[c["levels"]["period"]] == 32


def test_anchor_cells_recur_five_times_each():
    per_ld = [c for c in sg.main_effects_cells() if c["L_d_main"] == 2]
    anchors = [c for c in per_ld if c["is_anchor"]]
    # 3 corners (all-low/all-mid/all-high) x 5 axis-sweeps = 15 anchor measurements
    assert len(anchors) == 15
    by_cross = {}
    for c in anchors:
        by_cross.setdefault(c["cross"], 0)
        by_cross[c["cross"]] += 1
    assert by_cross == {"low": 5, "mid": 5, "high": 5}
