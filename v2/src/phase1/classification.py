"""V2 Phase 1 — per-cell three-category classification (spec §7.3; instr §12.4).

Inherits PRE-D2's per-head classifier verbatim (bootstrap CI of the median vs the
working-region threshold `baseline + τ_W`); adds the cell-level three-category
aggregation and the §12.4 conflicting-head rule:

  * each head (Diff_μ, Diff_σ) -> discriminably_working / discriminably_non_working /
    band_resident (PRE-D2 `classify_head`).
  * cell overall:
      - either head band_resident                -> band_resident
      - both heads same non-band category        -> that category
      - heads disagree on direction (one working,
        one non-working)                          -> band_resident, conflicting_heads=True (§12.4)

Thresholds come from `load_thresholds()` (PRE-D1a bit-identical baseline median +
PRE-E per-head τ_W) — the same runtime source PRE-D2 used.
"""

from __future__ import annotations

from v2.src.preflight.pre_d2_n_validation import (  # frozen Phase-0 inheritance
    bootstrap_ci_median,
    classify_head,
    load_thresholds,
)

CATEGORIES = ("discriminably_working", "discriminably_non_working", "band_resident")

__all__ = ["classify_cell", "classify_head", "load_thresholds",
           "bootstrap_ci_median", "CATEGORIES"]


def classify_cell(diff_mu_vals, diff_sigma_vals, thresholds: dict, n: int) -> dict:
    """Three-category per-cell classification with the §12.4 conflicting-head rule.

    Args:
        diff_mu_vals, diff_sigma_vals: per-rep Diff_μ / Diff_σ (>= n values).
        thresholds: dict with 'mu' and 'sigma' = baseline + τ_W per head.
        n: number of reps to use (first-n-by-seed, matching PRE-D2 §11.3).
    """
    h_mu = classify_head(diff_mu_vals, thresholds["mu"], n)
    h_sigma = classify_head(diff_sigma_vals, thresholds["sigma"], n)
    c_mu, c_sigma = h_mu["category"], h_sigma["category"]

    conflicting = False
    if c_mu == "band_resident" or c_sigma == "band_resident":
        overall = "band_resident"
    elif c_mu == c_sigma:
        overall = c_mu                       # both working or both non-working
    else:
        overall = "band_resident"            # direction conflict (§12.4)
        conflicting = True

    return {
        "head_mu": h_mu,
        "head_sigma": h_sigma,
        "overall": overall,
        "conflicting_heads": conflicting,
        "band_resident": overall == "band_resident",
        "discriminable": overall in ("discriminably_working", "discriminably_non_working"),
        "n": n,
    }
