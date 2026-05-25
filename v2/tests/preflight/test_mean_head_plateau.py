"""Unit tests for the mean-head-aware plateau detector (recalibration §3.4)."""

from __future__ import annotations

from v2.src.preflight.mean_head_plateau import (
    mean_head_plateau_detected,
    mean_head_plateau_step,
    not_grokked_within_budget,
)


def _curve(pairs):
    """Build a minimal curve [{steps, cos_k1}, ...] from (steps, cos_k1) pairs."""
    return [{"steps": s, "cos_k1": c} for s, c in pairs]


def test_empty_curve_returns_none():
    assert mean_head_plateau_step([]) is None


def test_clean_plateau_returns_onset_step():
    # Rises to ~0.80 then flattens; onset is the first within-1%-of-max point
    # whose forward gain into the next checkpoint is <= 0.01.
    curve = _curve([
        (2000, 0.20), (5000, 0.45), (10000, 0.65), (25000, 0.78),
        (50000, 0.795), (75000, 0.80), (100000, 0.801), (150000, 0.802),
    ])
    # max = 0.802; threshold = 0.99*0.802 = 0.79398.
    # 50000 (0.795) >= threshold; next (0.80) - 0.795 = 0.005 <= 0.01 -> qualifies.
    assert mean_head_plateau_step(curve) == 50000


def test_points_below_within_max_threshold_are_skipped():
    # Everything below 0.99*max is skipped; the first point reaching the band is
    # the onset (its tiny forward gain trivially satisfies condition 2 for cos<=1).
    curve = _curve([
        (2000, 0.10), (10000, 0.50),
        (50000, 0.792),   # 0.792 < 0.99*0.805 = 0.79695 -> skipped
        (100000, 0.80),   # >= threshold, gain to next = 0.005 -> onset
        (150000, 0.805),  # global max
    ])
    assert mean_head_plateau_step(curve) == 100000


def test_condition2_forward_gain_disqualifies_under_low_within_max_frac():
    # With within_max_frac=0.5, early points clear condition 1 but their large
    # forward gains (>0.01) disqualify them via condition 2; only the asymptote
    # (no successor) qualifies. This exercises condition 2 in isolation.
    curve = _curve([(2000, 0.40), (10000, 0.55), (50000, 0.80)])
    assert mean_head_plateau_step(curve, within_max_frac=0.5, next_gain_abs=0.01) == 50000


def test_monotonic_to_end_returns_last_step():
    # Still climbing > 0.01 at every interval: only the final checkpoint (no
    # successor) qualifies. Caller applies the §3.5 >175000 STOP separately.
    curve = _curve([
        (2000, 0.10), (50000, 0.30), (100000, 0.55),
        (150000, 0.75), (200000, 0.95),
    ])
    assert mean_head_plateau_step(curve) == 200000


def test_unsorted_input_is_handled():
    curve = _curve([
        (100000, 0.801), (2000, 0.20), (50000, 0.795),
        (75000, 0.80), (10000, 0.65), (25000, 0.78),
    ])
    assert mean_head_plateau_step(curve) == 50000


def test_alias_matches_primary():
    curve = _curve([(2000, 0.5), (10000, 0.79), (50000, 0.80)])
    assert mean_head_plateau_detected(curve) == mean_head_plateau_step(curve)


def test_not_grokked_when_below_trivial_plus_clearance():
    curve = _curve([(2000, 0.50), (50000, 0.60), (200000, 0.64)])
    # trivial = 0.559; max - trivial = 0.081 < 0.10 -> not grokked.
    assert not_grokked_within_budget(curve, trivial_baseline_cos_k1=0.559) is True


def test_grokked_when_clears_trivial_plus_clearance():
    curve = _curve([(2000, 0.50), (50000, 0.70), (200000, 0.75)])
    # max - trivial = 0.75 - 0.559 = 0.191 >= 0.10 -> grokked.
    assert not_grokked_within_budget(curve, trivial_baseline_cos_k1=0.559) is False


def test_threshold_at_exactly_max_single_point():
    curve = _curve([(50000, 0.42)])
    assert mean_head_plateau_step(curve) == 50000
