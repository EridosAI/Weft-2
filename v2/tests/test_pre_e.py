"""V2-PRE-E unit tests — τ_W calibration helper (instr §7.6, spec §7.3)."""

from __future__ import annotations

import numpy as np

from v2.src.preflight.pre_e_scaffolding_calibration import tau_w_for_head


def test_tau_w_reachable_on_spread_baseline():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0.03, 0.02, 20).tolist()
    out = tau_w_for_head(baseline)
    assert out["tau_W"] is not None
    assert out["p_value"] < 0.05
    assert out["tau_W"] > 0
    # The margin is on the order of the baseline spread.
    assert out["tau_W"] <= 10 * max(out["baseline_iqr"], 1e-9)


def test_tau_w_degenerate_baseline_gives_small_margin():
    # A zero-spread baseline is trivially distinguishable: any margin > 0 puts the
    # whole (constant) baseline below median+τ_W -> signed-rank significant -> the
    # minimal grid step is locked. (So this baseline-anchored calibration is
    # essentially always reachable; the §7.6 unreachable-STOP is a defensive guard.)
    out = tau_w_for_head([0.01] * 20)
    assert out["tau_W"] is not None
    assert out["tau_W"] < 0.01    # minimal grid step on a tiny scale


def test_tau_w_scales_with_baseline():
    rng = np.random.default_rng(1)
    tight = tau_w_for_head(rng.normal(0.03, 0.005, 20).tolist())["tau_W"]
    wide = tau_w_for_head(rng.normal(0.03, 0.05, 20).tolist())["tau_W"]
    assert tight is not None and wide is not None
    assert wide > tight   # wider baseline spread -> larger required margin
