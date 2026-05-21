"""V2-PRE-D2 unit tests — sweep points + CI classification (instr §7.7)."""

from __future__ import annotations

import numpy as np

from v2.src.preflight.pre_d2_n_validation import (
    bootstrap_ci_median, classify_head, classify_point, sweep_points,
)


def test_sweep_points_are_ten_at_2nd_and_4th_values():
    pts = sweep_points()
    assert len(pts) == 10                       # 5 axes x 2 (2nd, 4th)
    axes = [p[1] for p in pts]
    assert sorted(set(axes)) == ["continuity", "dim", "locality", "magnitude", "period"]
    assert all(axes.count(a) == 2 for a in set(axes))


def test_classify_head_three_categories():
    rng = np.random.default_rng(0)
    thr = 0.5
    working = classify_head(rng.normal(0.9, 0.02, 20), thr, n=20)
    nonworking = classify_head(rng.normal(0.1, 0.02, 20), thr, n=20)
    band = classify_head(rng.normal(0.5, 0.05, 20), thr, n=20)
    assert working["category"] == "discriminably_working"
    assert nonworking["category"] == "discriminably_non_working"
    assert band["category"] == "band_resident"


def test_classify_point_band_on_either_head_is_band_resident():
    rng = np.random.default_rng(1)
    th = {"mu": 0.5, "sigma": 0.5}
    mu_working = rng.normal(0.9, 0.02, 20)       # mu clearly working
    sigma_band = rng.normal(0.5, 0.05, 20)        # sigma straddles threshold
    out = classify_point(mu_working, sigma_band, th, n=20)
    assert out["band_resident"] is True
    assert out["discriminable"] is False


def test_bootstrap_ci_orders():
    lo, hi = bootstrap_ci_median(np.linspace(0, 1, 20))
    assert lo <= hi
