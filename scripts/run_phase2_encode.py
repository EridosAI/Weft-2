"""Phase 2 DINOv2 encoding + §8.4 perturbation-effect check.

Reads PNG frames at `data/phase2_frames/`, encodes them via the verified
DINOv2 protocol (frozen, fp16, 224 center crop, ImageNet mean/std,
L2-normalised CLS), writes the (N, 1024) float32 embedding matrix to
`data/phase2_embeddings/embeddings.npy` unconditionally, then applies
the §8.4 verification suite (restructured 2026-05-14 per session-6
reviewer authorisation):

  (1) Shape + L2-norm check on the embedding matrix.

  (2) **Ratio gate** (clean-control formulation): per-item Stage B vs
      Stage A mean cosine "gap" on the apex frames. Clean controls are
      {Bed, Television}; DiningTable is reported as a record-only noisy
      control (h118-corrected pose still leaks residual doorway-bleed
      per the third-STOP framing). Gate passes iff
      `mean(gap on perturbed) / mean(gap on clean controls) ≥ 2.0`,
      or `mean(gap on clean controls) ≤ 1e-3` (locality clean by
      construction).

  (3) **Wilcoxon-signed-rank gate** (statistical-distinguishability,
      Reading C). Per perturbed item, compute the 31×150 = 4650 pair
      cross-stage cosines (Stage A apex × Stage B apex). Run
      `scipy.stats.wilcoxon(1.0 - cos, alternative='greater')` on the
      per-pair (1 − cosine) values, testing whether the distribution's
      median is significantly greater than zero. Apply a Bonferroni
      correction for two perturbed items: gate passes iff each
      `min(raw_p * 2, 1.0) < 0.001`. The same test is run on Bed and
      Television as a record-only sanity-check diagnostic (not gated);
      DiningTable also runs record-only.

The save-first-gate-second pattern (embeddings written before gating)
lets the statistical test be re-run cheaply on the saved matrix without
re-encoding 65k frames.

Usage:
  nohup python3.12 -u scripts/run_phase2_encode.py \\
      > logs/phase2_encode_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy import stats as _scipy_stats

_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.encoder.dinov2_encoder import (  # noqa: E402
    encode_frames,
    load_frozen_dinov2,
)


_DEFAULT_FRAMES_DIR = _ROOT / "data" / "phase2_frames"
_DEFAULT_ANNOT = _ROOT / "data" / "phase2_annotations.jsonl"
_DEFAULT_OUT = _ROOT / "data" / "phase2_embeddings" / "embeddings.npy"
_DEFAULT_REPORT = _ROOT / "data" / "phase2_embeddings" / "encode_report.json"

_EMBED_DIM = 1024
_NORM_TOL = 1e-5

# Per-stage apex-frame sample count for the mean-cosine gap calculation.
# Stage A occupies the first 31 collected loops
# (PHASE_2_PERTURBATION_START_LOOP=31) → 31 apex frames per item per stage in
# Stage A. 25 sits below that natural ceiling. The per-pair sampler draws 500
# pairs from the sampled set for the within-stage means; the cross-stage mean
# uses all 25×25 = 625 pairs. The Wilcoxon test below uses the full 31×150
# cross-stage pair set drawn from all apex frames (not the 25 sub-sample).
_PERTURBATION_SAMPLE_N = 25

# §8.4 ratio gate — clean controls {Bed, Television} only (DT excluded as a
# noisy control per third-STOP framing; the h118-corrected pose still leaks
# residual doorway-bleed). Gate passes iff
#   perturbed_mean_gap / clean_control_mean_gap >= _DIFFERENTIAL_RATIO_MIN
# or clean_control_mean_gap <= _CONTROL_GAP_NEAR_ZERO (locality clean by
# construction). DiningTable's gap is reported as a record-only diagnostic.
_DIFFERENTIAL_RATIO_MIN: float = 2.0
_CONTROL_GAP_NEAR_ZERO: float = 1e-3

# §8.4 Wilcoxon signed-rank gate (statistical distinguishability, Reading C
# per session-6 reviewer authorisation 2026-05-14). For each perturbed item,
# compute the n_a × n_b cross-stage cosine values (Stage A apex × Stage B
# apex), then run
#   scipy.stats.wilcoxon(1.0 - cos, alternative='greater')
# testing whether the median of (1 − cos) is greater than zero. Bonferroni-
# correct for the two perturbed items: corrected_p = min(raw_p * 2, 1.0); gate
# passes iff corrected_p < _WILCOXON_CORRECTED_P_MAX for each perturbed item.
# Same test is run on Bed and Television as record-only sanity diagnostic
# (and on DiningTable as a noisy-control diagnostic).
_WILCOXON_N_PERTURBED_ITEMS: int = 2
_WILCOXON_CORRECTED_P_MAX: float = 0.001


def _load_annotations(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _norm_check(emb: np.ndarray, tol: float) -> Dict[str, Any]:
    norms = np.linalg.norm(emb, axis=1)
    lo, hi = 1.0 - tol, 1.0 + tol
    in_range = (norms >= lo) & (norms <= hi)
    n_bad = int((~in_range).sum())
    return {
        "passed": n_bad == 0,
        "tolerance": float(tol),
        "n_total": int(norms.size),
        "n_out_of_range": n_bad,
        "norms_min": float(norms.min()),
        "norms_max": float(norms.max()),
    }


def _apex_indices(
    annotations: List[Dict[str, Any]],
    viewing_position_id: int,
    perturbation_active: bool,
) -> List[int]:
    out: List[int] = []
    for a in annotations:
        if not a.get("close_up_apex_flag", False):
            continue
        if int(a.get("viewing_position_id", -1)) != viewing_position_id:
            continue
        if bool(a.get("perturbation_active", False)) != perturbation_active:
            continue
        out.append(int(a["frame_idx"]))
    return out


def _mean_pairwise_cosine(emb: np.ndarray, n_pairs: int, rng: np.random.Generator) -> float:
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    a_idx = rng.integers(0, n, size=n_pairs)
    b_idx = rng.integers(0, n, size=n_pairs)
    mask = a_idx != b_idx
    a_idx = a_idx[mask]
    b_idx = b_idx[mask]
    cos = np.einsum("ij,ij->i", emb[a_idx], emb[b_idx]).astype(np.float64)
    return float(cos.mean())


def _perturbation_effect_check(
    emb: np.ndarray,
    annotations: List[Dict[str, Any]],
    item_label: str,
    viewing_position_id: int,
    sample_n: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Per-item Stage B vs Stage A mean-cosine gap (record-only; ratio gate
    consumes the aggregate). The gap is the sample-level summary used by the
    ratio criterion; the per-pair distribution feeds the Wilcoxon test below.
    """
    stage_a_idx = _apex_indices(annotations, viewing_position_id, perturbation_active=False)
    stage_b_idx = _apex_indices(annotations, viewing_position_id, perturbation_active=True)
    if len(stage_a_idx) < sample_n or len(stage_b_idx) < sample_n:
        return {
            "ok": False,
            "reason": (
                f"insufficient apex frames: "
                f"stage_a={len(stage_a_idx)} stage_b={len(stage_b_idx)} "
                f"required {sample_n} per stage"
            ),
            "n_stage_a": len(stage_a_idx),
            "n_stage_b": len(stage_b_idx),
        }
    stage_a_sample = rng.choice(stage_a_idx, size=sample_n, replace=False)
    stage_b_sample = rng.choice(stage_b_idx, size=sample_n, replace=False)
    a_emb = emb[stage_a_sample]
    b_emb = emb[stage_b_sample]

    within_a = _mean_pairwise_cosine(a_emb, 500, rng)
    within_b = _mean_pairwise_cosine(b_emb, 500, rng)
    cross_mat = a_emb @ b_emb.T  # both already L2-normed
    cross = float(cross_mat.mean())

    within_avg = 0.5 * (within_a + within_b)
    gap = within_avg - cross
    return {
        "ok": True,
        "item": item_label,
        "viewing_position_id": int(viewing_position_id),
        "n_stage_a": len(stage_a_idx),
        "n_stage_b": len(stage_b_idx),
        "within_stage_a_mean_cosine": float(within_a),
        "within_stage_b_mean_cosine": float(within_b),
        "cross_stage_mean_cosine": float(cross),
        "within_avg_minus_cross": float(gap),
    }


def _wilcoxon_cross_stage_check(
    emb: np.ndarray,
    annotations: List[Dict[str, Any]],
    item_label: str,
    viewing_position_id: int,
    n_perturbed_items: int,
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test on (1 − cross_stage_cosine) values per item.

    Construct the full n_a × n_b matrix of cross-stage cosines from ALL Stage
    A apex frames against ALL Stage B apex frames at this viewing position
    (Stage A is bit-identical-deterministic so n_a × n_b pairs are
    independent draws from the Stage-B-side distribution given each
    Stage-A-side reference frame). Apply Wilcoxon signed-rank against zero
    with `alternative='greater'`; this tests whether the median of (1 − cos)
    is significantly greater than zero — i.e. whether cross-stage cosines
    sit systematically below 1.0.

    Bonferroni-correct for `n_perturbed_items` (multiply raw p by the count,
    cap at 1.0). The corrected p is what the §8.4 gate compares against
    `_WILCOXON_CORRECTED_P_MAX`. For control items the correction factor is
    still applied so the reported corrected_p is directly comparable to the
    perturbed-item gate; whether to *act* on a control's p value is a
    record-only diagnostic, not a gate.
    """
    stage_a_idx = _apex_indices(annotations, viewing_position_id, perturbation_active=False)
    stage_b_idx = _apex_indices(annotations, viewing_position_id, perturbation_active=True)
    if len(stage_a_idx) < 2 or len(stage_b_idx) < 2:
        return {
            "ok": False,
            "reason": (
                f"insufficient apex frames for Wilcoxon: "
                f"stage_a={len(stage_a_idx)} stage_b={len(stage_b_idx)}"
            ),
        }
    a_emb = emb[np.asarray(stage_a_idx, dtype=np.int64)]
    b_emb = emb[np.asarray(stage_b_idx, dtype=np.int64)]
    cross_mat = a_emb @ b_emb.T  # (n_a, n_b), already L2-normed
    one_minus_cos = (1.0 - cross_mat.astype(np.float64)).reshape(-1)
    n_pairs = int(one_minus_cos.size)

    # scipy.stats.wilcoxon needs to handle zero-difference values; default
    # `zero_method="wilcox"` discards them. For our case the cross-stage
    # cosines virtually never hit exact 1.0 (DINOv2 + texture variation makes
    # ties at the float64 level vanishingly rare), so the default is fine.
    # Use the normal approximation (`method="approx"`) — exact mode is
    # O(2^n_pairs) and infeasible at n_pairs = 4650.
    res = _scipy_stats.wilcoxon(
        one_minus_cos,
        alternative="greater",
        zero_method="wilcox",
        method="approx",
    )
    raw_p = float(res.pvalue)
    corrected_p = min(raw_p * float(n_perturbed_items), 1.0)
    return {
        "ok": True,
        "item": item_label,
        "viewing_position_id": int(viewing_position_id),
        "n_pairs": n_pairs,
        "n_stage_a": int(len(stage_a_idx)),
        "n_stage_b": int(len(stage_b_idx)),
        "one_minus_cos_median": float(np.median(one_minus_cos)),
        "one_minus_cos_mean": float(one_minus_cos.mean()),
        "wilcoxon_statistic": float(res.statistic),
        "wilcoxon_raw_p": raw_p,
        "wilcoxon_corrected_p": float(corrected_p),
        "bonferroni_n_items": int(n_perturbed_items),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames_dir", type=Path, default=_DEFAULT_FRAMES_DIR)
    parser.add_argument("--annotations", type=Path, default=_DEFAULT_ANNOT)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing embeddings.npy at --out (save-first pattern).",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    if not args.frames_dir.is_dir():
        print(f"[encode2] FAIL: frames_dir not found: {args.frames_dir}",
              file=sys.stderr)
        return 1
    if not args.annotations.is_file():
        print(f"[encode2] FAIL: annotations not found: {args.annotations}",
              file=sys.stderr)
        return 1
    if args.out.exists() and not args.overwrite:
        print(f"[encode2] FAIL: out path already exists (use --overwrite): "
              f"{args.out}", file=sys.stderr)
        return 1

    annotations = _load_annotations(args.annotations)
    n_total = len(annotations)
    print(f"[encode2] annotations: {n_total} records", flush=True)

    indices = [int(a["frame_idx"]) for a in annotations]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[encode2] FAIL: CUDA not available", file=sys.stderr)
        return 1
    print(f"[encode2] device: {device} ({torch.cuda.get_device_name(0)})",
          flush=True)
    print(f"[encode2] CUDA memory free: "
          f"{torch.cuda.mem_get_info(0)[0] / 1024**3:.1f} GB", flush=True)

    print(f"[encode2] loading frozen DINOv2-large", flush=True)
    model = load_frozen_dinov2(device)
    t0 = time.time()
    emb = encode_frames(
        model, args.frames_dir, indices, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    encode_seconds = time.time() - t0
    print(f"[encode2] encoded {emb.shape[0]} frames in {encode_seconds:.1f}s "
          f"({emb.shape[0]/max(encode_seconds, 1e-3):.1f} f/s)", flush=True)

    norm = _norm_check(emb, _NORM_TOL)
    print(f"[encode2] norm check passed={norm['passed']} "
          f"n_out_of_range={norm['n_out_of_range']} "
          f"min={norm['norms_min']:.6f} max={norm['norms_max']:.6f}", flush=True)

    # Save-first pattern (session-6 directive 2026-05-14): persist embeddings
    # unconditionally so the statistical gate can be re-run cheaply without
    # re-encoding 65k frames. Norm failure still aborts gating, but the matrix
    # is on disk for inspection.
    np.save(args.out, emb)
    print(f"[encode2] saved embeddings {emb.shape} -> {args.out}", flush=True)

    if not norm["passed"]:
        report = {
            "config": {"frames_dir": str(args.frames_dir), "annotations": str(args.annotations)},
            "encode_seconds": float(encode_seconds),
            "norm_check": norm,
            "overall_pass": False,
            "reason": "L2-norm out of tolerance for at least one row",
        }
        args.report.write_text(json.dumps(report, indent=2))
        print("[encode2] FAIL: norm check", file=sys.stderr)
        return 2

    rng = np.random.default_rng(int(args.seed))

    # ---- §8.4 per-item Stage B vs Stage A mean-cosine gap (record-only;
    # feeds the ratio gate below). Run on perturbed items and on all three
    # potential controls (Bed, Television, DiningTable). DT moves to a
    # record-only noisy-control slot per the third-STOP framing.
    perturbed_items: list[tuple[str, int]] = [("Dresser", 3), ("Sofa", 4)]
    clean_control_items: list[tuple[str, int]] = [("Bed", 1), ("Television", 5)]
    noisy_control_items: list[tuple[str, int]] = [("DiningTable", 2)]

    pert_checks: Dict[str, Any] = {}
    for label, vp in perturbed_items:
        pert_checks[label] = _perturbation_effect_check(
            emb, annotations, label, vp,
            sample_n=_PERTURBATION_SAMPLE_N, rng=rng,
        )
        print(f"[encode2] [gap] perturbed {label}: "
              f"gap={pert_checks[label].get('within_avg_minus_cross', float('nan')):.4f}",
              flush=True)

    clean_control_checks: Dict[str, Any] = {}
    for label, vp in clean_control_items:
        clean_control_checks[label] = _perturbation_effect_check(
            emb, annotations, label, vp,
            sample_n=_PERTURBATION_SAMPLE_N, rng=rng,
        )
        gap_val = clean_control_checks[label].get("within_avg_minus_cross", float("nan"))
        print(f"[encode2] [gap] clean-control {label}: gap={gap_val:.4f}", flush=True)

    noisy_control_checks: Dict[str, Any] = {}
    for label, vp in noisy_control_items:
        noisy_control_checks[label] = _perturbation_effect_check(
            emb, annotations, label, vp,
            sample_n=_PERTURBATION_SAMPLE_N, rng=rng,
        )
        gap_val = noisy_control_checks[label].get("within_avg_minus_cross", float("nan"))
        print(f"[encode2] [gap] noisy-control {label}: gap={gap_val:.4f} "
              f"(record-only; excluded from ratio gate)", flush=True)

    perturbed_gaps = [
        c["within_avg_minus_cross"] for c in pert_checks.values()
        if c.get("ok") and "within_avg_minus_cross" in c
    ]
    clean_control_gaps = [
        c["within_avg_minus_cross"] for c in clean_control_checks.values()
        if c.get("ok") and "within_avg_minus_cross" in c
    ]
    perturbed_mean_gap = float(np.mean(perturbed_gaps)) if perturbed_gaps else float("nan")
    clean_control_mean_gap = (
        float(np.mean(clean_control_gaps)) if clean_control_gaps else float("nan")
    )

    if not math.isfinite(perturbed_mean_gap) or not math.isfinite(clean_control_mean_gap):
        ratio = float("nan")
        ratio_pass = False
        ratio_reason = "non-finite gap value (degenerate sample)"
    elif clean_control_mean_gap <= _CONTROL_GAP_NEAR_ZERO:
        ratio = float("inf")
        ratio_pass = bool(perturbed_mean_gap > _CONTROL_GAP_NEAR_ZERO)
        ratio_reason = (
            f"clean_control_mean_gap <= {_CONTROL_GAP_NEAR_ZERO} (controls essentially "
            f"unmoved); locality clean by construction; pass iff perturbed_mean_gap "
            f"clears the same near-zero floor"
        )
    else:
        ratio = perturbed_mean_gap / clean_control_mean_gap
        ratio_pass = bool(ratio >= _DIFFERENTIAL_RATIO_MIN)
        ratio_reason = (
            f"perturbed_mean_gap / clean_control_mean_gap = {ratio:.3f} "
            f"vs threshold {_DIFFERENTIAL_RATIO_MIN}"
        )
    print(
        f"[encode2] [ratio] perturbed_mean_gap={perturbed_mean_gap:.4f} "
        f"clean_control_mean_gap={clean_control_mean_gap:.4f} "
        f"ratio={ratio:.3f}  {'PASS' if ratio_pass else 'FAIL'} "
        f"({ratio_reason})",
        flush=True,
    )

    # ---- §8.4 Wilcoxon signed-rank gate (Reading C, session-6 authorised).
    # Per-item cross-stage (1 − cos) signed-rank test against zero; the
    # perturbed-item corrected p-values are gated.
    wilcoxon_checks: Dict[str, Any] = {}
    for label, vp in (
        perturbed_items + clean_control_items + noisy_control_items
    ):
        check = _wilcoxon_cross_stage_check(
            emb, annotations, label, vp,
            n_perturbed_items=_WILCOXON_N_PERTURBED_ITEMS,
        )
        wilcoxon_checks[label] = check
        if check.get("ok"):
            role = (
                "perturbed" if (label, vp) in perturbed_items else
                "clean-control" if (label, vp) in clean_control_items else
                "noisy-control"
            )
            print(f"[encode2] [wilcoxon] {role:14s} {label}: "
                  f"n_pairs={check['n_pairs']} "
                  f"median(1-cos)={check['one_minus_cos_median']:.6f} "
                  f"raw_p={check['wilcoxon_raw_p']:.3e} "
                  f"corrected_p={check['wilcoxon_corrected_p']:.3e}",
                  flush=True)
    perturbed_wilcoxon_pass_flags = {
        label: bool(
            wilcoxon_checks[label].get("ok")
            and wilcoxon_checks[label]["wilcoxon_corrected_p"]
                < _WILCOXON_CORRECTED_P_MAX
        )
        for label, _ in perturbed_items
    }
    wilcoxon_pass = all(perturbed_wilcoxon_pass_flags.values())
    print(
        f"[encode2] [wilcoxon] perturbed-item gate {'PASS' if wilcoxon_pass else 'FAIL'} "
        f"({perturbed_wilcoxon_pass_flags}) vs corrected_p threshold "
        f"{_WILCOXON_CORRECTED_P_MAX}",
        flush=True,
    )

    overall_pass = norm["passed"] and ratio_pass and wilcoxon_pass
    report = {
        "config": {
            "frames_dir": str(args.frames_dir),
            "annotations": str(args.annotations),
            "model_id": "facebook/dinov2-large",
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "seed": int(args.seed),
            "perturbation_sample_n": int(_PERTURBATION_SAMPLE_N),
            "differential_ratio_min": float(_DIFFERENTIAL_RATIO_MIN),
            "control_gap_near_zero": float(_CONTROL_GAP_NEAR_ZERO),
            "wilcoxon_corrected_p_max": float(_WILCOXON_CORRECTED_P_MAX),
            "wilcoxon_n_perturbed_items": int(_WILCOXON_N_PERTURBED_ITEMS),
            "norm_tolerance": float(_NORM_TOL),
        },
        "n_frames_encoded": int(emb.shape[0]),
        "encode_seconds": float(encode_seconds),
        "norm_check": norm,
        "perturbation_effect_checks": pert_checks,
        "clean_control_effect_checks": clean_control_checks,
        "noisy_control_effect_checks": noisy_control_checks,
        "ratio_gate_summary": {
            "perturbed_items": [label for label, _ in perturbed_items],
            "clean_control_items": [label for label, _ in clean_control_items],
            "noisy_control_items": [label for label, _ in noisy_control_items],
            "perturbed_mean_gap": perturbed_mean_gap,
            "clean_control_mean_gap": clean_control_mean_gap,
            "ratio_perturbed_over_clean_control": ratio,
            "ratio_threshold_min": float(_DIFFERENTIAL_RATIO_MIN),
            "control_gap_near_zero": float(_CONTROL_GAP_NEAR_ZERO),
            "ratio_gate_pass": ratio_pass,
            "ratio_gate_reason": ratio_reason,
            "note": (
                "Ratio gate (session-6 restructuring 2026-05-14). Clean controls "
                "= {Bed, Television}. DiningTable reported as record-only noisy "
                "control (h118-corrected pose still leaks residual doorway-bleed). "
                f"Gate passes iff perturbed_mean_gap / clean_control_mean_gap >= "
                f"{_DIFFERENTIAL_RATIO_MIN}, or clean_control_mean_gap <= "
                f"{_CONTROL_GAP_NEAR_ZERO}."
            ),
        },
        "wilcoxon_gate_summary": {
            "perturbed_items": [label for label, _ in perturbed_items],
            "clean_control_items": [label for label, _ in clean_control_items],
            "noisy_control_items": [label for label, _ in noisy_control_items],
            "wilcoxon_checks": wilcoxon_checks,
            "perturbed_pass_per_item": perturbed_wilcoxon_pass_flags,
            "corrected_p_threshold_max": float(_WILCOXON_CORRECTED_P_MAX),
            "wilcoxon_gate_pass": wilcoxon_pass,
            "note": (
                "Wilcoxon signed-rank on (1 - cross-stage-cosine), "
                "alternative='greater', method='approx'. Per-pair sample = "
                "n_stage_a × n_stage_b cross-stage pairs at the item's apex "
                "frames. Bonferroni-correction: corrected_p = min(raw_p * "
                f"{_WILCOXON_N_PERTURBED_ITEMS}, 1.0). Gate passes iff each "
                f"perturbed item has corrected_p < {_WILCOXON_CORRECTED_P_MAX}. "
                "Reading C per session-6 reviewer authorisation 2026-05-14. "
                "Control-item p-values are record-only (not gated)."
            ),
        },
        "overall_pass": bool(overall_pass),
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(f"[encode2] wrote report: {args.report}", flush=True)

    if not overall_pass:
        reasons = []
        if not norm["passed"]:
            reasons.append("norm check")
        if not ratio_pass:
            reasons.append(f"ratio gate: {ratio_reason}")
        if not wilcoxon_pass:
            reasons.append(
                f"wilcoxon perturbed-item gate: {perturbed_wilcoxon_pass_flags}"
            )
        print(f"[encode2] FAIL: {'; '.join(reasons)}. "
              f"Embeddings saved (gate failed; do not launch training).",
              file=sys.stderr)
        return 2

    print(f"[encode2] PASS. Embeddings on disk at {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
