"""Unit tests for the Phase 1 three-category cell classifier (§7.3 + §12.4)."""

from __future__ import annotations

from v2.src.phase1.classification import classify_cell

# thresholds: μ working-region threshold 0.10, σ threshold 1e-7 (toy values).
TH = {"mu": 0.10, "sigma": 1e-7}


def _const(v, n=10):
    return [v] * n


def test_both_heads_working_is_discriminably_working():
    # μ well above 0.10 and σ well above 1e-7 -> both working -> cell working.
    r = classify_cell(_const(0.5), _const(5e-7), TH, 10)
    assert r["head_mu"]["category"] == "discriminably_working"
    assert r["head_sigma"]["category"] == "discriminably_working"
    assert r["overall"] == "discriminably_working"
    assert not r["conflicting_heads"] and r["discriminable"]


def test_both_heads_non_working_is_discriminably_non_working():
    r = classify_cell(_const(0.001), _const(1e-9), TH, 10)
    assert r["overall"] == "discriminably_non_working"
    assert not r["conflicting_heads"] and r["discriminable"]


def test_conflicting_heads_aggregate_to_band_resident():
    # μ discriminably_working, σ discriminably_non_working -> direction conflict (§12.4).
    r = classify_cell(_const(0.5), _const(1e-9), TH, 10)
    assert r["head_mu"]["category"] == "discriminably_working"
    assert r["head_sigma"]["category"] == "discriminably_non_working"
    assert r["overall"] == "band_resident"
    assert r["conflicting_heads"] and not r["discriminable"]


def test_straddling_head_makes_cell_band_resident():
    # μ values straddling the threshold -> band_resident regardless of σ.
    r = classify_cell([0.05, 0.2, 0.04, 0.3, 0.08, 0.25, 0.06, 0.18, 0.02, 0.4],
                      _const(5e-7), TH, 10)
    assert r["head_mu"]["category"] == "band_resident"
    assert r["overall"] == "band_resident"
    assert not r["conflicting_heads"]   # band, not a direction conflict
