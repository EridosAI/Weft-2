"""V2-PRE-E — SCAFFOLDING calibration (instr §7.6; analytical, 0 arm-runs).

Calibrates the §4/§6/§7 SCAFFOLDING thresholds against empirical distributions
from PRE-A (construction primitives — regenerated here, no training), PRE-B
(worked-example repetition coverage), and PRE-D1a (bit-identical baseline Diff
distribution). Locks values into scaffolding_calibration.json; the runner then
updates v2/config.py.

Thresholds (spec ref → rule):
  tau_pert  (§6.2.4) — above the magnitude=0 reference-vs-frame residual floor.
  tau_L     (§4.2)   — above the magnitude=0 relation-shift (Δcos) noise floor.
  tau_R + noise_floor (§4.4) — above the off-period self-similarity noise.
  repetition_coverage_threshold (§6.2.3) — a value PRE-A construction satisfies
                       and PRE-B's worked example (coverage 1.0) must clear.
  bic_improvement_threshold (§6.3) — above single-mode reference BIC improvements
                       (PRE-B flagged placeholder 10 too permissive).
  local_pca_window (§4.5) — window where D_local is stable across subsampling.
  tau_W per head (§7.3) — smallest margin where a one-sample Wilcoxon signed-rank
                       test finds the PRE-D1a baseline significantly below
                       (baseline_median + tau_W) at p<0.05. STOP if unreachable.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

from v2.config import (
    DEFAULT_N_REPETITIONS, RESULTS_PRE_B, RESULTS_PRE_D1A, RESULTS_PRE_E,
)
from v2.src.protocol.grid_mapping import detect_multimodal
from v2.src.protocol.protocol import estimate_reference_state
from v2.src.property_measure.manifold_dim import measure_manifold_dim
from v2.src.property_measure.repetition import autocorrelation_profile
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams, build_stream

MID = dict(period_P=128, manifold_dim_D=16, continuity_center=24, fidelity_F=0.97,
           magnitude_M=0.5, locality_L=0.5)


def _norm(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def calibrate_residual_thresholds(U) -> dict:
    """tau_pert + tau_L from magnitude=0 (fidelity-noise-only) streams."""
    residuals, dcos_noise = [], []
    for F in (0.97, 0.99, 0.999):
        bs = build_stream(StreamParams(**{**MID, "fidelity_F": F, "magnitude_M": 0.0,
                                          "n_repetitions": DEFAULT_N_REPETITIONS}), U)
        x = _norm(bs.stream); P = bs.params.period_P
        ref = estimate_reference_state(x, P)
        residuals.append(np.linalg.norm(x - ref, axis=1))
        # Δcos relation-shift noise (actual vs reference) over radius-1 neighbours.
        a = np.sum(x[:-1] * x[1:], axis=1)
        r = np.sum(ref[:-1] * ref[1:], axis=1)
        dcos_noise.append(np.abs(a - r))
    res = np.concatenate(residuals); dc = np.concatenate(dcos_noise)
    tau_pert = float(np.percentile(res, 99.5))
    tau_L = float(np.percentile(dc, 99.0))
    return {"TAU_PERT": tau_pert, "TAU_L": tau_L,
            "_residual_m0_p99_5": tau_pert, "_dcos_noise_p99": tau_L,
            "_residual_m0_max": float(res.max())}


def calibrate_repetition_thresholds(U) -> dict:
    """tau_R + noise floor from off-period vs at-period self-similarity."""
    bs = build_stream(StreamParams(**{**MID, "magnitude_M": 0.0,
                                       "n_repetitions": DEFAULT_N_REPETITIONS}), U)
    x = _norm(bs.stream); P = bs.params.period_P
    m = autocorrelation_profile(x, max_lag=min(x.shape[0] // 2, 3 * P))
    lags = np.arange(1, m.size)
    off = m[1:][(lags % P != 0)]                 # off-period lags = noise
    at = m[1:][(lags % P == 0)]                  # at-period lags = repetition
    noise_floor = float(np.percentile(off, 95))
    at_min = float(at.min()) if at.size else 1.0
    tau_R = float((noise_floor + at_min) / 2.0)  # midway between noise and repetition
    return {"TAU_R": tau_R, "REPETITION_NOISE_FLOOR": float(max(noise_floor, np.percentile(off, 99))),
            "_offperiod_p95": noise_floor, "_atperiod_min": at_min}


def calibrate_coverage_threshold(U) -> dict:
    """Coverage floor for locality-defined: below PRE-A construction & PRE-B (1.0)."""
    covs = []
    from v2.src.property_measure.repetition import measure_repetition
    for F in (0.95, 0.97, 0.999):
        bs = build_stream(StreamParams(**{**MID, "fidelity_F": F, "magnitude_M": 0.0,
                                          "n_repetitions": DEFAULT_N_REPETITIONS}), U)
        r = measure_repetition(_norm(bs.stream), tau_R=0.6, noise_floor=0.3)
        covs.append(r["coverage"])
    min_cov = float(min(covs))
    thr = float(round(min(0.5, min_cov * 0.8), 3))   # construction satisfies; PRE-B 1.0 clears
    return {"REPETITION_COVERAGE_THRESHOLD": thr, "_pre_a_min_coverage": min_cov}


def calibrate_bic_threshold(n_ref: int = 60, sample_n: int = 150) -> dict:
    """BIC-improvement threshold above single-mode reference distributions (§6.3)."""
    rng = np.random.default_rng(0)
    single_bic, multi_bic = [], []
    for _ in range(n_ref):
        s = rng.normal(0.05, 0.01, sample_n)
        single_bic.append(detect_multimodal(s, bic_improvement_threshold=0.0)["bic_improvement"])
        mu = np.concatenate([rng.normal(0.02, 0.005, sample_n // 2),
                             rng.normal(0.10, 0.005, sample_n // 2)])
        multi_bic.append(detect_multimodal(mu, bic_improvement_threshold=0.0)["bic_improvement"])
    thr = float(np.percentile(single_bic, 99))
    return {"BIC_IMPROVEMENT_THRESHOLD": float(round(max(thr, 0.0), 2)),
            "_single_mode_bic_p99": thr, "_single_mode_bic_max": float(np.max(single_bic)),
            "_multi_mode_bic_median": float(np.median(multi_bic))}


def calibrate_local_pca_window(U) -> dict:
    """Window where D_local is stable across subsampling and recovers known D."""
    D = 16
    bs = build_stream(StreamParams(**{**MID, "manifold_dim_D": D, "magnitude_M": 0.0,
                                      "continuity_center": 24, "n_repetitions": 6}), U)
    x = _norm(bs.stream)
    best = None
    for w in (32, 64, 128):
        ests = [measure_manifold_dim(x, window=w, subsample_rate=sr)["D_local"]
                for sr in (2, 4, 8)]
        cv = float(np.std(ests) / (np.mean(ests) + 1e-12))
        cand = {"window": w, "cv": cv, "mean_D_local": float(np.mean(ests))}
        if best is None or cv < best["cv"]:
            best = cand
    return {"LOCAL_PCA_WINDOW": int(best["window"]), "MANIFOLD_SUBSAMPLE_RATE": 4,
            "_window_selection": best}


def tau_w_for_head(values, n_grid: int = 400) -> dict:
    """Smallest margin tau_W where one-sample Wilcoxon signed-rank finds `values`
    significantly below (median + tau_W) at p<0.05. Returns tau_W=None if unreachable."""
    B = np.array(values, dtype=np.float64)
    med = float(np.median(B))
    iqr = float(np.subtract(*np.percentile(B, [75, 25])))
    scale = max(iqr, abs(med) * 0.1, 1e-12)
    tau_locked, p_locked = None, None
    for i in range(1, n_grid + 1):
        tau = scale * i / 20.0
        d = B - (med + tau)
        if np.allclose(d, 0):
            continue
        try:
            _, p = wilcoxon(d, alternative="less")
        except ValueError:
            continue
        if p < 0.05:
            tau_locked, p_locked = float(tau), float(p)
            break
    return {"tau_W": tau_locked, "p_value": p_locked, "baseline_median": med,
            "baseline_iqr": iqr, "baseline_n": int(B.size)}


def calibrate_tau_w() -> dict:
    """tau_W per head via one-sample Wilcoxon signed-rank vs the PRE-D1a baseline."""
    base = json.loads((RESULTS_PRE_D1A / "bit_identical_baseline.json").read_text())
    out, stop = {}, None
    for head, key in (("mu", "diff_mu_distribution"), ("sigma", "diff_sigma_distribution")):
        out[head] = tau_w_for_head(base[key]["values_sorted"])
        if out[head]["tau_W"] is None:
            stop = (f"tau_W ({head}) unreachable: no margin yields signed-rank p<0.05 "
                    f"against the n={out[head]['baseline_n']} baseline (do NOT lower the standard).")
    return {"per_head": out, "stop": stop,
            "joint_alpha_note": "joint working-region requires both heads at p<0.05; "
            "joint alpha ~0.0025 if independent (heads are not independent on this "
            "architecture; empirical dependence reported in v2 closing per §7.3)."}


def post_coverage_check() -> dict:
    """PRE-B worked-example coverage must clear the calibrated coverage threshold."""
    region = json.loads((RESULTS_PRE_B / "worked_example_region.json").read_text())
    cov = region["per_axis_distributions"]["repetition"]["whole_stream"]["coverage"]
    return {"pre_b_worked_example_coverage": cov}
