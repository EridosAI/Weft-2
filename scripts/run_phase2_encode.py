"""Phase 2 DINOv2 encoding + §8.4 perturbation-effect check.

Reads PNG frames at `data/phase2_frames/`, encodes them via the verified
DINOv2 protocol (frozen, fp16, 224 center crop, ImageNet mean/std,
L2-normalised CLS), and writes the (N, 1024) float32 embedding matrix
to `data/phase2_embeddings/embeddings.npy`.

Performs §8.4 verification:
  - Shape and norm check.
  - Stage B vs Stage A perturbation-effect check on Dresser-apex and
    Sofa-apex embeddings: cross-stage cosine must separate from within-
    stage cosines by at least 0.05 (otherwise the per-loop
    `RandomizeMaterials` did not produce a measurable encoder-level
    perturbation and the script exits non-zero).

Usage:
  nohup python3.12 -u scripts/run_phase2_encode.py \\
      > logs/phase2_encode_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
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
_PERTURBATION_GAP_THRESHOLD = 0.05  # instr §8.4
_PERTURBATION_SAMPLE_N = 50         # instr §8.4


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
    pert_checks: Dict[str, Any] = {}
    for label, vp in (("Dresser", 3), ("Sofa", 4)):
        pert_checks[label] = _perturbation_effect_check(
            emb, annotations, label, vp,
            sample_n=_PERTURBATION_SAMPLE_N,
            gap_threshold=_PERTURBATION_GAP_THRESHOLD,
            rng=rng,
        )
        print(f"[encode2] perturbation effect check {label}: "
              f"passed={pert_checks[label]['passed']} "
              f"gap={pert_checks[label].get('within_avg_minus_cross', float('nan')):.4f}",
              flush=True)

    pert_pass = all(c["passed"] for c in pert_checks.values())

    overall_pass = norm["passed"] and pert_pass
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
            "norm_tolerance": float(_NORM_TOL),
        },
        "n_frames_encoded": int(emb.shape[0]),
        "encode_seconds": float(encode_seconds),
        "norm_check": norm,
        "perturbation_effect_checks": pert_checks,
        "overall_pass": bool(overall_pass),
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(f"[encode2] wrote report: {args.report}", flush=True)

    if not overall_pass:
        print("[encode2] FAIL: perturbation effect check below gap threshold; "
              "NOT writing embeddings.", file=sys.stderr)
        return 2

    np.save(args.out, emb)
    print(f"[encode2] PASS. Wrote {emb.shape} float32 -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
