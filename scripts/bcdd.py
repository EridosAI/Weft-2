#!/usr/bin/env python3
"""
BCDD — Body-Coupling Disambiguating Diagnostic
==============================================

Measures whether parameter drift between v0 checkpoints (loop-30 vs loop-100)
produces drift in PREDICTED MEANS and the body's pooled hidden state — not
just in variance — at Bed ordinals.

Background
----------
The v0 verdict observed variance drift at Bed ord 9/10 despite pixel-identical
input. Under architecture (a) (single last_token pooled → linear fan-out to
K*(d+1)), three candidate mechanisms remain plausible:

  M1: Capacity-limited variance head. K scalars total; 1024× fewer parameters
      per step than the mean head. Insufficient capacity for per-input
      variance differentiation even when last_token IS input-distinguished.
  M2: Loss-shape coupling. The Gaussian NLL variance gradient is a scalar
      depending only on mean-prediction-error magnitude. Weakly
      input-distinguished, so the variance scalar's accumulated updates drift
      coupledly even when the body produces input-distinguished last_token.
  M3: Body or output_proj parameter drift. Shared upstream parameters
      accumulate updates that affect last_token (or its projection) for ALL
      inputs uniformly, independent of which input drove the surprise.

BCDD distinguishes M3 from M1/M2 by measuring drift signatures on the SAME
input across checkpoints. Same-input → no input variation → any drift is
pure parameter effect.

Tests, in order of decisiveness
-------------------------------
  Test A — last_token cosine. cos(last_token(model_30, win),
           last_token(model_100, win)) is the most direct M3 test. If cos ≈ 1,
           the body produces identical representations across checkpoints
           and any coupling is downstream of the body. If cos < 1, the body
           itself has drifted.

  Test B — aggregate mean drift per ordinal. 1 − cos(predicted mean) at each
           Bed ordinal, aggregated over K. Mirrors the v0 variance-drift
           uniformity analysis. If mean drift is uniform across all 11
           ordinals (matching variance-drift uniformity), M3 supported.

  Test C — per-k variation at invariant ords (9, 10). Coefficient of variation
           of per-k drift across the K predicted steps. Under architecture (a):
             low CoV  → drift driven by last_token shift (body or pooled-head)
             high CoV → drift driven by output_proj rows independently

The v0-style variance drift (different ckpt + different input window, what
v0 actually reported) is computed for direct cross-reference against
per_ordinal_cross_loop_input.json's published values (-0.4356 at ord 9,
-0.4059 at ord 10).

Usage
-----
Run from the Weft 2 repo root (so default paths resolve):

  python bcdd.py \\
      --ckpt-30 <path/to/checkpoint_near_loop_30> \\
      --ckpt-100 <path/to/checkpoint_near_loop_100> \\
      [--fresh-stage-a-ckpt <path/to/end_of_stage_a_checkpoint>] \\
      [--embeddings data/phase2_embeddings/embeddings.npy] \\
      [--predictor-module src/predictor/inner_pam.py] \\
      [--out bcdd_results.json] \\
      [--device cuda]

If the InnerPAM constructor signature differs from the spec defaults
(hidden=512, n_heads=8, n_layers=4, mlp_dim=2048), state_dict load will
fail loud — adjust ARCH_DEFAULTS below.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Constants — verified against per_ordinal_cross_loop_input.json
# -----------------------------------------------------------------------------
BED_ORD_FRAMES_LOOP_30 = list(range(10800, 10811))   # 10800..10810 inclusive
BED_ORD_FRAMES_LOOP_100 = list(range(36000, 36011))  # 36000..36010 inclusive

WINDOW_W = 16
PREDICT_K = 16
EMBED_DIM = 1024

# Ords 9 and 10: pixel-MD5 identical across loops 30/50/75/100 per v0 diagnostic.
# These are the decisive sub-rows for M1/M2 vs M3 discrimination.
INVARIANT_ORDS = (9, 10)

# Spec §7.3 SCAFFOLDING defaults; if InnerPAM was trained with different
# values, state_dict load will fail loud. Adjust here if needed.
ARCH_DEFAULTS = {
    "hidden": 512,
    "n_heads": 8,
    "n_layers": 4,
    "mlp_dim": 2048,
}

# v0 published variance drifts at invariant ords (for sanity-check cross-ref).
# v0 reports mean over K of (lv_100 − lv_30); see
# run_phase2_variance_by_ordinal.py:157 (`log_var.mean()` over the K axis) and
# run_phase2_per_ordinal_cross_loop_input.py:303-306 (drift = lv_100 − lv_30
# on the per-ordinal mean_log_var_over_K).
V0_VARIANCE_DRIFT_REFERENCE = {
    "ord_9":  -0.4356198310852051,
    "ord_10": -0.4058599472045898,
    "source": "per_ordinal_cross_loop_input.json:primary_discriminator",
}


# -----------------------------------------------------------------------------
# Predictor loading
# -----------------------------------------------------------------------------
def load_predictor(
    checkpoint_path: Path,
    predictor_module_path: Path,
    device: torch.device,
):
    """Load InnerPAM from a checkpoint.

    Imports InnerPAM from the working repo's predictor module to ensure
    architectural match. Tries common state-dict key patterns.
    """
    # inner_pam.py uses absolute imports (from src.config import ...), so the
    # repo root — parent of src/ — must be on sys.path.
    # predictor_module_path = .../src/predictor/inner_pam.py
    repo_root = predictor_module_path.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.predictor.inner_pam import InnerPAM  # type: ignore[import-not-found]

    try:
        model = InnerPAM(
            embed_dim=EMBED_DIM,
            window_w=WINDOW_W,
            predict_k=PREDICT_K,
            **ARCH_DEFAULTS,
        ).to(device)
    except TypeError as e:
        raise RuntimeError(
            f"InnerPAM constructor rejected expected kwargs. "
            f"Either the class signature differs from the spec, or "
            f"ARCH_DEFAULTS need adjustment. Underlying error: {e}"
        )

    model.eval()

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "predictor_state" in raw:
        state_dict = raw["predictor_state"]
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict) and all(
        isinstance(v, torch.Tensor) for v in raw.values()
    ):
        state_dict = raw
    else:
        raise RuntimeError(
            f"Unrecognised checkpoint format at {checkpoint_path}. "
            f"Top-level keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys when loading {checkpoint_path}: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading {checkpoint_path}: {unexpected}")

    return model


# -----------------------------------------------------------------------------
# Window construction
# -----------------------------------------------------------------------------
def build_window(embeddings: np.ndarray, ordinal_frame_idx: int) -> torch.Tensor:
    """K-back convention: window ends K frames before target.
    The K-th predicted step then lands ON the target frame at ordinal_frame_idx."""
    target_end = ordinal_frame_idx
    window_end = target_end - PREDICT_K          # frames before target
    start = window_end - WINDOW_W + 1
    end = window_end + 1
    if start < 0:
        raise ValueError(
            f"Frame {ordinal_frame_idx} too early for K-back W={WINDOW_W} window "
            f"(needs frames {start}..{end-1})"
        )
    win = embeddings[start:end]
    if win.shape != (WINDOW_W, EMBED_DIM):
        raise RuntimeError(f"Built window shape {win.shape} ≠ expected ({WINDOW_W}, {EMBED_DIM})")
    return torch.from_numpy(win).float().unsqueeze(0)  # (1, W, D)


# -----------------------------------------------------------------------------
# Forward pass with last_token capture (Test A)
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_with_last_token(model, window: torch.Tensor, device: torch.device):
    """Run predictor; capture encoder output's last token via forward hook.

    Returns:
        mean:        (K, D) numpy array
        log_var:     (K,)   numpy array
        last_token:  (hidden,) numpy array — body's pooled vector at t=W-1
    """
    captured: list[torch.Tensor] = []

    def hook(module, inp, out):
        # nn.TransformerEncoder output: (B, W, hidden)
        captured.append(out[:, -1, :].detach().cpu())

    handle = model.encoder.register_forward_hook(hook)
    try:
        mean, log_var = model(window.to(device))
    finally:
        handle.remove()

    assert mean.shape == (1, PREDICT_K, EMBED_DIM)
    assert log_var.shape == (1, PREDICT_K)
    assert len(captured) == 1, f"Expected 1 hook call, got {len(captured)}"

    return (
        mean[0].cpu().numpy(),
        log_var[0].cpu().numpy(),
        captured[0][0].numpy(),
    )


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    )


def mean_drift_per_k(mean_a: np.ndarray, mean_b: np.ndarray) -> list[float]:
    """1 − cosine per K-step, comparing two (K, D) arrays."""
    return [1.0 - cosine(mean_a[k], mean_b[k]) for k in range(mean_a.shape[0])]


def var_drift_per_k(lv_newer: np.ndarray, lv_older: np.ndarray) -> list[float]:
    """Log-variance drift, newer − older. Matches v0's drift = lv_100 − lv_30."""
    return [float(lv_newer[k] - lv_older[k]) for k in range(lv_newer.shape[0])]


def descriptive(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std_population": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "per_k_values": [float(v) for v in arr],
    }


def coefficient_of_variation(values: list[float]) -> float:
    """std / |mean|. Low → uniform across k; high → varies across k.

    Returns inf when mean is near zero (no signal to express variation against).
    """
    arr = np.asarray(values, dtype=np.float64)
    m = abs(float(arr.mean()))
    if m < 1e-12:
        return float("inf")
    return float(arr.std(ddof=0) / m)


def uniformity_stats(per_ord_values: list[float]) -> dict:
    """Mirrors per_ordinal_cross_loop_input.json's 2σ-band format.

    Reports whether the values at INVARIANT_ORDS fall inside the
    mean ± 2σ band built from all 11 ordinals. If yes → the invariant
    ordinals don't stand out from the rest → uniform drift across all
    inputs, matching the v0 variance-drift uniformity pattern.
    """
    arr = np.asarray(per_ord_values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    lo, hi = mean - 2 * std, mean + 2 * std
    inv_vals = [per_ord_values[i] for i in INVARIANT_ORDS]
    in_band = [(lo <= v <= hi) for v in inv_vals]
    return {
        "n_ordinals": int(arr.size),
        "mean": mean,
        "std_population": std,
        "band_2sigma_lo": lo,
        "band_2sigma_hi": hi,
        "per_ordinal_values": [float(v) for v in arr],
        "invariant_ords": list(INVARIANT_ORDS),
        "invariant_ord_values": inv_vals,
        "invariant_within_2sigma_band": in_band,
        "all_invariant_within_band": all(in_band),
    }


# -----------------------------------------------------------------------------
# Main BCDD logic
# -----------------------------------------------------------------------------
def run_bcdd(
    ckpt_30_path: Path,
    ckpt_100_path: Path,
    embeddings_path: Path,
    predictor_module_path: Path,
    output_path: Path,
    fresh_stage_a_ckpt_path: Optional[Path] = None,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[BCDD] Device: {device}")

    # ---- Load embeddings + sanity-check L2 norms ----
    print(f"[BCDD] Loading embeddings: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBED_DIM:
        raise RuntimeError(
            f"Embeddings shape {embeddings.shape} ≠ expected (N, {EMBED_DIM})"
        )
    sample_norms = np.linalg.norm(embeddings[::1000], axis=1)
    print(f"  L2 norm sample: min={sample_norms.min():.6f}, max={sample_norms.max():.6f}")
    if abs(sample_norms.mean() - 1.0) > 1e-3:
        raise RuntimeError(
            "Embeddings do not appear L2-normalised (sample mean far from 1.0)."
        )

    # ---- Load checkpoints ----
    print(f"[BCDD] Loading ckpt loop-30: {ckpt_30_path}")
    model_30 = load_predictor(ckpt_30_path, predictor_module_path, device)
    print(f"[BCDD] Loading ckpt loop-100: {ckpt_100_path}")
    model_100 = load_predictor(ckpt_100_path, predictor_module_path, device)

    model_fresh = None
    if fresh_stage_a_ckpt_path is not None:
        print(f"[BCDD] Loading fresh Stage A ckpt: {fresh_stage_a_ckpt_path}")
        model_fresh = load_predictor(fresh_stage_a_ckpt_path, predictor_module_path, device)

    # ---- Build windows for each of the 11 Bed ordinals at loop-30 and loop-100 ----
    windows_30 = [build_window(embeddings, f) for f in BED_ORD_FRAMES_LOOP_30]
    windows_100 = [build_window(embeddings, f) for f in BED_ORD_FRAMES_LOOP_100]

    # ---- Per-ordinal: run all (ckpt × window) combinations ----
    per_ordinal: list[dict] = []
    for i in range(11):
        w30 = windows_30[i]
        w100 = windows_100[i]

        m30_on_w30,   lv30_on_w30,   lt30_on_w30   = predict_with_last_token(model_30,  w30,  device)
        m100_on_w30,  lv100_on_w30,  lt100_on_w30  = predict_with_last_token(model_100, w30,  device)
        m30_on_w100,  lv30_on_w100,  lt30_on_w100  = predict_with_last_token(model_30,  w100, device)
        m100_on_w100, lv100_on_w100, lt100_on_w100 = predict_with_last_token(model_100, w100, device)

        # Same-input mean/variance drift (pure parameter effect)
        mdrift_on_w30   = mean_drift_per_k(m30_on_w30,   m100_on_w30)
        mdrift_on_w100  = mean_drift_per_k(m30_on_w100,  m100_on_w100)
        vdrift_on_w30   = var_drift_per_k(lv100_on_w30,  lv30_on_w30)
        vdrift_on_w100  = var_drift_per_k(lv100_on_w100, lv30_on_w100)

        # Test A: last_token cosine across ckpts on the same window
        last_token_cos_on_w30  = cosine(lt30_on_w30,  lt100_on_w30)
        last_token_cos_on_w100 = cosine(lt30_on_w100, lt100_on_w100)

        # v0-style: different ckpt + different input window (mirrors v0 diagnostic)
        # newer − older to match v0's drift = lv_100 − lv_30.
        v0_style_vdrift = var_drift_per_k(lv100_on_w100, lv30_on_w30)
        v0_style_mdrift = mean_drift_per_k(m30_on_w30, m100_on_w100)

        entry = {
            "ordinal": i,
            "loop_30_frame_idx": BED_ORD_FRAMES_LOOP_30[i],
            "loop_100_frame_idx": BED_ORD_FRAMES_LOOP_100[i],

            # --- Test A: last_token cosine ---
            "last_token_cosine_on_loop_30_window":  last_token_cos_on_w30,
            "last_token_cosine_on_loop_100_window": last_token_cos_on_w100,

            # --- Same-input mean drift (Test B input) ---
            "mean_drift_on_loop_30_window":  descriptive(mdrift_on_w30),
            "mean_drift_on_loop_100_window": descriptive(mdrift_on_w100),

            # --- Same-input variance drift ---
            "variance_drift_on_loop_30_window":  descriptive(vdrift_on_w30),
            "variance_drift_on_loop_100_window": descriptive(vdrift_on_w100),

            # --- v0-style: different ckpt + different input window ---
            "v0_style_variance_drift":  descriptive(v0_style_vdrift),
            "v0_style_mean_drift":      descriptive(v0_style_mdrift),
        }

        # Fresh Stage A supplementary three-point reads
        if model_fresh is not None:
            mF_on_w30,  lvF_on_w30,  ltF_on_w30  = predict_with_last_token(model_fresh, w30,  device)
            mF_on_w100, lvF_on_w100, ltF_on_w100 = predict_with_last_token(model_fresh, w100, device)
            entry["fresh_to_loop_30_mean_drift_on_loop_30_window"] = descriptive(
                mean_drift_per_k(mF_on_w30, m30_on_w30)
            )
            entry["fresh_to_loop_100_mean_drift_on_loop_100_window"] = descriptive(
                mean_drift_per_k(mF_on_w100, m100_on_w100)
            )
            entry["fresh_to_loop_30_last_token_cosine_on_loop_30_window"] = cosine(ltF_on_w30, lt30_on_w30)
            entry["fresh_to_loop_100_last_token_cosine_on_loop_100_window"] = cosine(ltF_on_w100, lt100_on_w100)

        per_ordinal.append(entry)

        flag = " ⭐" if i in INVARIANT_ORDS else ""
        print(
            f"  ord {i}{flag}: "
            f"last_tok cos (w30/w100) = {last_token_cos_on_w30:.6f} / {last_token_cos_on_w100:.6f}; "
            f"mean drift = {entry['mean_drift_on_loop_30_window']['mean']:.4f} / {entry['mean_drift_on_loop_100_window']['mean']:.4f}; "
            f"var drift = {entry['variance_drift_on_loop_30_window']['mean']:.4f} / {entry['variance_drift_on_loop_100_window']['mean']:.4f}"
        )

    # ---- Aggregate uniformity (Test B) ----
    agg_mean_drift_w30  = [e["mean_drift_on_loop_30_window"]["mean"]  for e in per_ordinal]
    agg_mean_drift_w100 = [e["mean_drift_on_loop_100_window"]["mean"] for e in per_ordinal]
    agg_var_drift_w30   = [e["variance_drift_on_loop_30_window"]["mean"]  for e in per_ordinal]
    agg_var_drift_w100  = [e["variance_drift_on_loop_100_window"]["mean"] for e in per_ordinal]
    agg_last_token_cos_w30  = [e["last_token_cosine_on_loop_30_window"]  for e in per_ordinal]
    agg_last_token_cos_w100 = [e["last_token_cosine_on_loop_100_window"] for e in per_ordinal]
    agg_v0_style_vdrift = [e["v0_style_variance_drift"]["mean"] for e in per_ordinal]

    aggregate_uniformity = {
        "mean_drift_on_loop_30_window":      uniformity_stats(agg_mean_drift_w30),
        "mean_drift_on_loop_100_window":     uniformity_stats(agg_mean_drift_w100),
        "variance_drift_on_loop_30_window":  uniformity_stats(agg_var_drift_w30),
        "variance_drift_on_loop_100_window": uniformity_stats(agg_var_drift_w100),
        "last_token_cosine_on_loop_30_window":  uniformity_stats(agg_last_token_cos_w30),
        "last_token_cosine_on_loop_100_window": uniformity_stats(agg_last_token_cos_w100),
        "v0_style_variance_drift":           uniformity_stats(agg_v0_style_vdrift),
    }

    # ---- Test C: per-k variation at invariant ords ----
    per_k_at_invariant = {}
    for ord_idx in INVARIANT_ORDS:
        e = per_ordinal[ord_idx]
        per_k_at_invariant[f"ord_{ord_idx}"] = {
            "mean_drift_on_loop_30_window": {
                "per_k": e["mean_drift_on_loop_30_window"]["per_k_values"],
                "cov": coefficient_of_variation(e["mean_drift_on_loop_30_window"]["per_k_values"]),
            },
            "mean_drift_on_loop_100_window": {
                "per_k": e["mean_drift_on_loop_100_window"]["per_k_values"],
                "cov": coefficient_of_variation(e["mean_drift_on_loop_100_window"]["per_k_values"]),
            },
            "variance_drift_on_loop_30_window": {
                "per_k": e["variance_drift_on_loop_30_window"]["per_k_values"],
                "cov": coefficient_of_variation(e["variance_drift_on_loop_30_window"]["per_k_values"]),
            },
            "variance_drift_on_loop_100_window": {
                "per_k": e["variance_drift_on_loop_100_window"]["per_k_values"],
                "cov": coefficient_of_variation(e["variance_drift_on_loop_100_window"]["per_k_values"]),
            },
        }

    # ---- v0 sanity-check summary ----
    sanity_check = {
        "v0_published_variance_drift_at_invariant_ords": V0_VARIANCE_DRIFT_REFERENCE,
        "bcdd_v0_style_variance_drift_at_invariant_ords": {
            f"ord_{i}": per_ordinal[i]["v0_style_variance_drift"]["mean"]
            for i in INVARIANT_ORDS
        },
        "note": (
            "BCDD's v0_style_variance_drift uses (model_30, window_30) vs "
            "(model_100, window_100) per-k aggregated — same construction "
            "as v0's variance_drift_loop30_to_loop100. Values should land "
            "near v0's -0.4356 (ord 9) and -0.4059 (ord 10) if the predictor "
            "loaded matches the v0 trained predictor."
        ),
    }

    # ---- Final output ----
    output = {
        "method": (
            "BCDD: For each Bed close-up ordinal i, run both loop-30 and "
            "loop-100 predictor checkpoints on BOTH the loop-30 and loop-100 "
            "input windows. Same-input comparison isolates pure parameter "
            "drift. Forward hook captures encoder output's last_token "
            "(body's pooled representation under architecture (a)). Tests "
            "A (last_token cosine), B (mean drift uniformity), C (per-k "
            "coefficient of variation at invariant ordinals 9 and 10)."
        ),
        "item": "Bed",
        "viewing_position_id": 1,
        "window_w": WINDOW_W,
        "predict_k": PREDICT_K,
        "embed_dim": EMBED_DIM,
        "invariant_ords": list(INVARIANT_ORDS),
        "checkpoints": {
            "loop_30":   str(ckpt_30_path),
            "loop_100":  str(ckpt_100_path),
            "fresh_stage_a": str(fresh_stage_a_ckpt_path) if fresh_stage_a_ckpt_path else None,
        },
        "embeddings_path": str(embeddings_path),

        "v0_sanity_check": sanity_check,

        "test_a_summary": {
            "description": (
                "last_token cosine across checkpoints on the same input window. "
                "≈1.0 → body produces identical pooled representation → drift "
                "is downstream of body (M1 or M2). <1.0 → body itself has "
                "drifted (M3)."
            ),
            "invariant_ord_values_on_loop_30_window": [
                agg_last_token_cos_w30[i] for i in INVARIANT_ORDS
            ],
            "invariant_ord_values_on_loop_100_window": [
                agg_last_token_cos_w100[i] for i in INVARIANT_ORDS
            ],
        },

        "test_b_summary": {
            "description": (
                "Mean drift per ordinal, aggregated over K. If mean drift at "
                "invariant ords 9/10 falls within the 2σ band of all-11-ords "
                "mean drift (mirroring the v0 variance-drift uniformity "
                "pattern), parameter drift produces uniform across-ordinal "
                "effects on mean predictions → M3 supported."
            ),
            "uniformity_on_loop_30_window": uniformity_stats(agg_mean_drift_w30),
            "uniformity_on_loop_100_window": uniformity_stats(agg_mean_drift_w100),
        },

        "test_c_summary": {
            "description": (
                "Per-k coefficient of variation at invariant ords 9 and 10. "
                "Low CoV (per-k drift uniform across K) → drift driven by "
                "last_token shift (body or pooled-head). High CoV (per-k "
                "drift varies across K) → drift driven by individual "
                "output_proj rows."
            ),
            "per_k_at_invariant_ords": per_k_at_invariant,
        },

        "per_ordinal": per_ordinal,
        "aggregate_uniformity": aggregate_uniformity,

        "decision_logic": {
            "M1_or_M2_supported_when": (
                "Test A: last_token cosine ≈ 1.0 at invariant ords (body "
                "preserves representation). Test B: mean drift at invariant "
                "ords is small relative to variance drift, or falls outside "
                "the all-ords 2σ band (mean drift is NOT uniform). Conclusion: "
                "variance head is the dominant coupling locus; 8.2 in the "
                "anisotropic or mixture-density reading is the precise v1 "
                "intervention. Option (c)'s single ablation arm suffices for "
                "attribution."
            ),
            "M3_supported_when": (
                "Test A: last_token cosine < 1.0 at invariant ords (body has "
                "drifted). Test B: mean drift at invariant ords falls WITHIN "
                "the all-ords 2σ band (same uniformity pattern as variance "
                "drift). Conclusion: body or output_proj parameter "
                "accumulation is the dominant coupling channel. 8.2 alone is "
                "insufficient; v1 must include a readout topology change "
                "(per-K output queries, position-preserving readout, or "
                "training-dynamics intervention). Second ablation arm "
                "(scalar variance + v0-strength perturbation, reproducing v0 "
                "as a control) becomes load-bearing for full attribution."
            ),
            "mixed_signal_when": (
                "Test A: last_token cosine intermediate (e.g., 0.95–0.99). "
                "Test C: per-k CoV breakdown distinguishes last_token shift "
                "(low CoV) from output_proj-row drift (high CoV). The lower-"
                "in-the-stack mechanism wins for v1 scope; the higher-in-the-"
                "stack one becomes a secondary candidate for v2."
            ),
        },
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n[BCDD] Wrote results: {output_path}")

    # ---- Console summary for quick inspection ----
    print("\n[BCDD] === Sanity check against v0 published variance drift (mean-over-K) ===")
    for i in INVARIANT_ORDS:
        bcdd_mean = per_ordinal[i]["v0_style_variance_drift"]["mean"]
        v0_pub = V0_VARIANCE_DRIFT_REFERENCE[f"ord_{i}"]
        delta = abs(bcdd_mean - v0_pub)
        flag = "PASS" if delta < 0.05 else "FAIL"
        print(
            f"  ord {i}: BCDD mean-over-K = {bcdd_mean:+.4f}  |  "
            f"v0 published = {v0_pub:+.4f}  |  Δ = {delta:.4f}  [{flag}]"
        )

    print("\n[BCDD] === Test A: last_token cosine at invariant ords ===")
    for i in INVARIANT_ORDS:
        print(
            f"  ord {i}: on loop-30 win = {agg_last_token_cos_w30[i]:.6f}, "
            f"on loop-100 win = {agg_last_token_cos_w100[i]:.6f}"
        )

    print("\n[BCDD] === Test B: mean drift uniformity (loop-30 window) ===")
    u = aggregate_uniformity["mean_drift_on_loop_30_window"]
    print(f"  mean ± 2σ band: [{u['band_2sigma_lo']:.4f}, {u['band_2sigma_hi']:.4f}]")
    print(f"  invariant ord values: {u['invariant_ord_values']}")
    print(f"  all invariant ords inside 2σ band: {u['all_invariant_within_band']}")

    print("\n[BCDD] === Test C: per-k CoV at invariant ords (loop-30 window, mean drift) ===")
    for i in INVARIANT_ORDS:
        cov = per_k_at_invariant[f"ord_{i}"]["mean_drift_on_loop_30_window"]["cov"]
        print(f"  ord {i}: CoV = {cov:.4f}")

    return output


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BCDD — Body-Coupling Disambiguating Diagnostic for Weft Inner PAM v1 design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ckpt-30", type=Path, required=True,
        help="Path to predictor checkpoint at/near training loop 30 (~step 10800)")
    parser.add_argument("--ckpt-100", type=Path, required=True,
        help="Path to predictor checkpoint at/near training loop 100 (~step 36000)")
    parser.add_argument("--fresh-stage-a-ckpt", type=Path, default=None,
        help="Optional: end-of-Stage-A baseline checkpoint for three-point analysis")
    parser.add_argument("--embeddings", type=Path,
        default=Path("data/phase2_embeddings/embeddings.npy"),
        help="Path to Phase 2 DINOv2 embeddings .npy (N, 1024)")
    parser.add_argument("--predictor-module", type=Path,
        default=Path("src/predictor/inner_pam.py"),
        help="Path to inner_pam.py (used to locate src/ for import)")
    parser.add_argument("--out", type=Path,
        default=Path("bcdd_results.json"))
    parser.add_argument("--device", type=str, default="cuda",
        help="torch device string (cuda or cpu); falls back to cpu if cuda unavailable")
    args = parser.parse_args()

    run_bcdd(
        ckpt_30_path=args.ckpt_30,
        ckpt_100_path=args.ckpt_100,
        embeddings_path=args.embeddings,
        predictor_module_path=args.predictor_module,
        output_path=args.out,
        fresh_stage_a_ckpt_path=args.fresh_stage_a_ckpt,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
