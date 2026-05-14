"""Phase 2 DINOv2 encoding + §8.4 perturbation-effect check.

Reads PNG frames at `data/phase2_frames/`, encodes them via the verified
DINOv2 protocol (frozen, fp16, 224 center crop, ImageNet mean/std,
L2-normalised CLS), and writes the (N, 1024) float32 embedding matrix
to `data/phase2_embeddings/embeddings.npy`.

Performs §8.4 verification with **both absolute and differential metrics**
(per the 2026-05-14 experiment-chat directive):

  Absolute (gated): for each perturbed item (Dresser, Sofa), report the
    within-Stage-A and within-Stage-B mean apex-embedding cosines and the
    cross-stage mean cosine. The gap (within_avg - cross) must exceed
    0.05 — otherwise per-loop RandomizeMaterials did not produce a
    measurable encoder-level perturbation and the script exits non-zero.

  Differential (record-only, for reviewer assessment): the same Stage B
    vs Stage A comparison applied to the **control items** (Bed,
    DiningTable, Television). Bedroom items are not visually perturbed
    by the LivingRoom-scoped call, so the expected pattern is gap ≈ 0
    on control items. The "contrast" (perturbed_mean_gap - control_mean_gap)
    is the load-bearing read for whether the perturbation is item-
    specific rather than a global drift.

The experiment chat reviews both metric families before authorising the
launch of Phase 2 training.

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
_PERTURBATION_GAP_THRESHOLD = 0.05  # instr §8.4 (absolute per-perturbed-item gap)
_PERTURBATION_SAMPLE_N = 50         # instr §8.4

# Differential go/no-go gate — ratio criterion (reviewer-authorised 2026-05-14
# post-sixth-STOP fix). perturbed_mean_gap must be at least
# _DIFFERENTIAL_RATIO_MIN times control_mean_gap. The ratio formulation is
# robust to the absolute scale of the gaps: small absolute gaps with strong
# locality still pass (perturbed clearly separates more than controls), while
# large gaps without locality (everything moves together) fail.
#
# Edge case: if control_mean_gap is near zero (≤ _CONTROL_GAP_NEAR_ZERO), the
# ratio is unbounded; we treat that as PASS by short-circuit (controls show
# essentially no Stage A vs Stage B drift, so locality is clean by
# construction).
_DIFFERENTIAL_RATIO_MIN: float = 2.0
_CONTROL_GAP_NEAR_ZERO: float = 1e-3


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
    gap_threshold: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    stage_a_idx = _apex_indices(annotations, viewing_position_id, perturbation_active=False)
    stage_b_idx = _apex_indices(annotations, viewing_position_id, perturbation_active=True)
    if len(stage_a_idx) < sample_n or len(stage_b_idx) < sample_n:
        return {
            "passed": False,
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

    # Mean pairwise cosine within each set (random pair sampling).
    within_a = _mean_pairwise_cosine(a_emb, 500, rng)
    within_b = _mean_pairwise_cosine(b_emb, 500, rng)
    # Mean cross-set cosine (all-pairs).
    cross_mat = a_emb @ b_emb.T  # both already L2-normed
    cross = float(cross_mat.mean())

    within_avg = 0.5 * (within_a + within_b)
    gap = within_avg - cross
    passed = bool(gap > gap_threshold)
    return {
        "passed": passed,
        "item": item_label,
        "viewing_position_id": int(viewing_position_id),
        "n_stage_a": len(stage_a_idx),
        "n_stage_b": len(stage_b_idx),
        "within_stage_a_mean_cosine": float(within_a),
        "within_stage_b_mean_cosine": float(within_b),
        "cross_stage_mean_cosine": float(cross),
        "within_avg_minus_cross": float(gap),
        "gap_threshold": float(gap_threshold),
        "criterion": f"within - cross > {gap_threshold}",
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
    if args.out.exists():
        print(f"[encode2] FAIL: out path already exists (refusing to overwrite): "
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

    if not norm["passed"]:
        # Save report and refuse to write embeddings.
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

    # ---- §8.4 absolute metric (gated): perturbed items' Stage B vs Stage A gap.
    perturbed_items: list[tuple[str, int]] = [("Dresser", 3), ("Sofa", 4)]
    pert_checks: Dict[str, Any] = {}
    for label, vp in perturbed_items:
        pert_checks[label] = _perturbation_effect_check(
            emb, annotations, label, vp,
            sample_n=_PERTURBATION_SAMPLE_N,
            gap_threshold=_PERTURBATION_GAP_THRESHOLD,
            rng=rng,
        )
        print(f"[encode2] [absolute] perturbed {label}: "
              f"passed={pert_checks[label]['passed']} "
              f"gap={pert_checks[label].get('within_avg_minus_cross', float('nan')):.4f}",
              flush=True)
    pert_pass = all(c["passed"] for c in pert_checks.values())

    # ---- §8.4 differential metric (record-only, per 2026-05-14 directive).
    # Compute the same Stage B vs Stage A gap on the control items (Bed,
    # DiningTable, Television). Under the locality claim, control items
    # should show gap ≈ 0 — they aren't visually re-textured by the
    # LivingRoom-scoped RandomizeMaterials call. Any non-zero gap captures
    # global drift across the Stage A/B boundary (lighting, shadows,
    # background bleed). The contrast = perturbed_mean_gap - control_mean_gap
    # isolates the perturbation-specific component.
    control_items: list[tuple[str, int]] = [
        ("Bed", 1), ("DiningTable", 2), ("Television", 5),
    ]
    control_checks: Dict[str, Any] = {}
    for label, vp in control_items:
        control_checks[label] = _perturbation_effect_check(
            emb, annotations, label, vp,
            sample_n=_PERTURBATION_SAMPLE_N,
            gap_threshold=_PERTURBATION_GAP_THRESHOLD,   # same threshold for symmetry; not gated
            rng=rng,
        )
        gap_val = control_checks[label].get("within_avg_minus_cross", float("nan"))
        print(f"[encode2] [differential] control {label}: "
              f"gap={gap_val:.4f} (record-only; not gated)",
              flush=True)

    perturbed_gaps = [
        c["within_avg_minus_cross"] for c in pert_checks.values()
        if "within_avg_minus_cross" in c
    ]
    control_gaps = [
        c["within_avg_minus_cross"] for c in control_checks.values()
        if "within_avg_minus_cross" in c
    ]
    perturbed_mean_gap = float(np.mean(perturbed_gaps)) if perturbed_gaps else float("nan")
    control_mean_gap = float(np.mean(control_gaps)) if control_gaps else float("nan")
    contrast = perturbed_mean_gap - control_mean_gap
    # ≥2× ratio gate (reviewer-authorised 2026-05-14, post-sixth-STOP). The
    # ratio formulation is robust to absolute scale: locality is established
    # by perturbed items separating proportionally more than controls, not
    # by the absolute magnitude of the gap. Edge case: control_mean_gap
    # essentially zero (≤ _CONTROL_GAP_NEAR_ZERO) short-circuits to PASS
    # — controls show no Stage A vs Stage B drift, locality is clean by
    # construction.
    if not math.isfinite(perturbed_mean_gap) or not math.isfinite(control_mean_gap):
        ratio = float("nan")
        differential_pass = False
        gate_reason = "non-finite gap value (degenerate sample)"
    elif control_mean_gap <= _CONTROL_GAP_NEAR_ZERO:
        ratio = float("inf")
        differential_pass = bool(perturbed_mean_gap > _PERTURBATION_GAP_THRESHOLD)
        gate_reason = (
            f"control_mean_gap <= {_CONTROL_GAP_NEAR_ZERO} (controls essentially "
            f"unmoved); locality clean by construction; pass iff perturbed_mean_gap "
            f"clears the absolute gate threshold {_PERTURBATION_GAP_THRESHOLD}"
        )
    else:
        ratio = perturbed_mean_gap / control_mean_gap
        differential_pass = bool(ratio >= _DIFFERENTIAL_RATIO_MIN)
        gate_reason = (
            f"perturbed_mean_gap / control_mean_gap = {ratio:.3f} "
            f"vs threshold {_DIFFERENTIAL_RATIO_MIN}"
        )
    print(
        f"[encode2] [contrast] perturbed_mean_gap={perturbed_mean_gap:.4f} "
        f"control_mean_gap={control_mean_gap:.4f} "
        f"ratio={ratio:.3f}  {'PASS' if differential_pass else 'FAIL'} "
        f"({gate_reason})",
        flush=True,
    )

    overall_pass = norm["passed"] and pert_pass and differential_pass
    report = {
        "config": {
            "frames_dir": str(args.frames_dir),
            "annotations": str(args.annotations),
            "model_id": "facebook/dinov2-large",
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "seed": int(args.seed),
            "perturbation_gap_threshold": float(_PERTURBATION_GAP_THRESHOLD),
            "perturbation_sample_n": int(_PERTURBATION_SAMPLE_N),
            "differential_ratio_min": float(_DIFFERENTIAL_RATIO_MIN),
            "control_gap_near_zero": float(_CONTROL_GAP_NEAR_ZERO),
            "norm_tolerance": float(_NORM_TOL),
        },
        "n_frames_encoded": int(emb.shape[0]),
        "encode_seconds": float(encode_seconds),
        "norm_check": norm,
        # Absolute (gated): per-perturbed-item gap > _PERTURBATION_GAP_THRESHOLD
        # on each of Dresser and Sofa.
        "absolute_perturbation_effect_checks": pert_checks,
        # Differential (gated since 2026-05-14): control-item gap + contrast.
        "differential_control_item_effect_checks": control_checks,
        "differential_summary": {
            "perturbed_items": [label for label, _ in perturbed_items],
            "control_items": [label for label, _ in control_items],
            "perturbed_mean_gap": perturbed_mean_gap,
            "control_mean_gap": control_mean_gap,
            "contrast_perturbed_minus_control": contrast,
            "ratio_perturbed_over_control": ratio,
            "ratio_threshold_min": float(_DIFFERENTIAL_RATIO_MIN),
            "control_gap_near_zero": float(_CONTROL_GAP_NEAR_ZERO),
            "differential_gate_pass": differential_pass,
            "differential_gate_reason": gate_reason,
            "note": (
                "Differential go/no-go gate (ratio criterion, "
                "reviewer-authorised 2026-05-14 post-sixth-STOP fix). "
                f"Phase 2 training launches only if perturbed_mean_gap / "
                f"control_mean_gap >= {_DIFFERENTIAL_RATIO_MIN}, or "
                f"control_mean_gap <= {_CONTROL_GAP_NEAR_ZERO} (clean "
                "locality by construction). Ratio formulation is robust to "
                "absolute scale of the gaps."
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
        if not pert_pass:
            reasons.append("absolute per-perturbed-item gap")
        if not differential_pass:
            reasons.append(f"differential ratio gate: {gate_reason}")
        print(f"[encode2] FAIL: {'; '.join(reasons)}. NOT writing embeddings.",
              file=sys.stderr)
        return 2

    np.save(args.out, emb)
    print(f"[encode2] PASS. Wrote {emb.shape} float32 -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
