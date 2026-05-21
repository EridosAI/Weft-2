"""V2-PRE-D2 — n=10 vs n=20 CI validation (instr §7.7; spec §9.3).

10 sweep points (each axis at its 2nd & 4th of-five grid value, others at
midpoint; avoids the all-midpoint coordinate), Primary arm only, at L_d=2
(intermediate capacity per §11.6 — CI extrapolation to L_d_main {1,4} is
approximate). n=20 training repetitions per point (different seeds).

Per head (Diff_μ, Diff_σ), per point, at n=10 (first-10-by-seed per §11.3) and
n=20, classify the bootstrap CI of the point's Diff vs the working-region
threshold `baseline + τ_W` (PRE-D1a bit-identical baseline median + PRE-E τ_W):
  discriminably-working      — CI lower bound  > threshold
  discriminably-non-working  — CI upper bound  < threshold
  band-resident              — CI straddles threshold (genuine non-working per
                               §7.3, but unresolvable from noise at n=k)
A point is "discriminable at n=k" iff both heads are non-band-resident.

Reuses PRE-D1a's train_one (v1 Primary + path_prediction_loss); the deliverable
is the n=10→n=20 resolution gain feeding Phase 0.5 (Input 2).
"""

from __future__ import annotations

import json

import numpy as np

from v2.config import RESULTS_PRE_D1A, RESULTS_PRE_E
from v2.src.preflight.pre_d1a_endpoint_stability import MID, train_one
from v2.src.substrate.stream_builder import StreamParams

# Five-value per-axis grids (midpoint = MID = the 3rd value). PRE-D2 uses the
# 2nd and 4th values (two of the three §3.5 held-axis positions).
GRID5 = {
    "magnitude":  [0.1, 0.3, 0.5, 0.7, 0.9],   # magnitude_M
    "locality":   [0.1, 0.3, 0.5, 0.7, 0.9],   # locality_L
    "continuity": [8, 16, 24, 40, 60],          # continuity_center
    "period":     [64, 96, 128, 192, 256],      # period_P
    "dim":        [4, 8, 16, 32, 64],            # manifold_dim_D
}
L_D_INTERMEDIATE = 2   # §11.6: spec §9.3 "L_d=2" read as intermediate L_d, not L_d_main


def params_for(axis: str, value) -> StreamParams:
    kw = dict(period_P=MID["period_P"], manifold_dim_D=MID["manifold_dim_D"],
              continuity_center=MID["continuity_center"], fidelity_F=MID["fidelity_F"],
              magnitude_M=MID["magnitude_M"], locality_L=MID["locality_L"])
    if axis == "magnitude":
        kw["magnitude_M"] = value
    elif axis == "locality":
        kw["locality_L"] = value
    elif axis == "continuity":
        kw["continuity_center"] = value
    elif axis == "dim":
        kw["manifold_dim_D"] = value
    elif axis == "period":
        kw["period_P"] = value
        kw["continuity_center"] = max(1, round(MID["continuity_center"] * value / MID["period_P"]))
    return StreamParams(**kw)


def sweep_points() -> list[tuple]:
    """10 points: (label, axis, value) at the 2nd and 4th grid value per axis."""
    pts = []
    for axis, grid in GRID5.items():
        for idx in (1, 3):                       # 2nd and 4th values
            pts.append((f"{axis}@{grid[idx]}", axis, grid[idx]))
    return pts


def load_thresholds() -> dict:
    base = json.loads((RESULTS_PRE_D1A / "bit_identical_baseline.json").read_text())
    cal = json.loads((RESULTS_PRE_E / "scaffolding_calibration.json").read_text())
    tw = cal["tau_W"]["per_head"]
    return {
        "mu": base["diff_mu_distribution"]["median"] + tw["mu"]["tau_W"],
        "sigma": base["diff_sigma_distribution"]["median"] + tw["sigma"]["tau_W"],
        "baseline_mu_median": base["diff_mu_distribution"]["median"],
        "baseline_sigma_median": base["diff_sigma_distribution"]["median"],
        "tau_W_mu": tw["mu"]["tau_W"], "tau_W_sigma": tw["sigma"]["tau_W"],
    }


def bootstrap_ci_median(vals, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=np.float64)
    meds = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def classify_head(vals, threshold: float, n: int) -> dict:
    sub = list(vals)[:n]
    lo, hi = bootstrap_ci_median(sub)
    if lo > threshold:
        cat = "discriminably_working"
    elif hi < threshold:
        cat = "discriminably_non_working"
    else:
        cat = "band_resident"
    return {"ci_low": lo, "ci_high": hi, "median": float(np.median(sub)),
            "threshold": threshold, "category": cat}


def recommend_framing(band_n10: int, band_n20: int, total: int) -> tuple[float, str]:
    """Phase 0.5 Input-2 framing from band-residence at n=10 vs n=20 (§7.7 step 6).

    Adds a fourth reading beyond the spec's three (sufficient / marginal /
    n=20-needed): when n=20 gives no resolution gain AND band-residence is high
    at both n, the limit is per-rep VARIANCE, not sample size — n=20 is not
    justified over n=10, but per-point reliability is a §9.8 / closing concern.
    """
    gain = (band_n10 - band_n20) / max(total, 1)   # positive = n=20 reduces band-residence
    if band_n10 <= max(1, int(0.2 * total)):
        f = "n=10 sufficient (low band-residence at n=10)"
    elif gain >= 0.1:
        f = "n=10 marginal (n=20 materially reduces band-residence)"
    elif gain <= 0.0 and band_n10 >= 0.3 * total:
        f = ("variance-limited: n=20 gives NO resolution gain over n=10 "
             "(band-residence not reduced) and band-residence is high at both n -> "
             "working-region determination is variance-limited, not sample-limited at "
             "n<=20. n=20 NOT justified over n=10; per-point reliability is a §9.8 / "
             "closing concern (the architecture's Diff_mu is high-variance at L_d=2).")
    elif band_n10 > 0.5 * total:
        f = "n=20 needed (high band-residence at n=10 with resolution gain at n=20)"
    else:
        f = "mixed: n=20 marginally changes resolution; design-chat judgement"
    return round(gain, 3), f


def classify_point(diff_mu_vals, diff_sigma_vals, th: dict, n: int) -> dict:
    h_mu = classify_head(diff_mu_vals, th["mu"], n)
    h_sigma = classify_head(diff_sigma_vals, th["sigma"], n)
    band = (h_mu["category"] == "band_resident") or (h_sigma["category"] == "band_resident")
    discriminable = not band
    return {"head_mu": h_mu, "head_sigma": h_sigma, "discriminable": discriminable,
            "band_resident": band}
