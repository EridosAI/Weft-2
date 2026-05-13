"""DINOv2 full-stream encoder for the seed-7 furniture rerender.

Encodes all 100,000 frames at `data/seed7_furniture_frames/` (dwell + transit)
via the same DINOv2 protocol used by the substrate-verification batch
(`scripts/run_encoder_verification_dinov2.py`):

  - `facebook/dinov2-large` (ViT-L/14), frozen, fp16 eval
  - 256x256 RGB -> center-crop 224x224 -> ImageNet mean/std normalisation
  - CLS token (last_hidden_state[:, 0, :]) -> L2-normalise
  - batch 64, num_workers 4

Rationale: the substrate-verification artefact at
`data/dinov2_embeddings/embeddings_dwell_only_v1.npy` only encoded the
32,760 dwell frames (sufficient for §5 verification, insufficient for
v0 training which requires a continuous stream per spec §2.3 and
EXPERIMENT_INSTRUCTIONS §7.2).

Outputs:
  - data/dinov2_embeddings/embeddings.npy        (100,000, 1024) fp32 L2-norm
  - data/dinov2_embeddings/encode_full_stream_report.json

Consistency check:
  - 50 random dwell-frame indices: cosine(new, archived) >= 0.9999.
  - Mismatch indicates encoder configuration drift since the substrate
    verification; the script reports the failing indices and exits non-zero
    without overwriting `embeddings.npy`.

Norm check:
  - All 100,000 rows have ||row||_2 in [1 - 1e-5, 1 + 1e-5].
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel


_DEFAULT_FRAMES_DIR = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/data/seed7_furniture_frames"
)
_DEFAULT_ANNOT = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/main/"
    "frame_annotations.jsonl"
)
_DEFAULT_ARCHIVE = Path(
    "data/dinov2_embeddings/embeddings_dwell_only_v1.npy"
)
_DEFAULT_OUT = Path("data/dinov2_embeddings/embeddings.npy")
_DEFAULT_REPORT = Path("data/dinov2_embeddings/encode_full_stream_report.json")
_DEFAULT_MODEL_ID = "facebook/dinov2-large"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CROP = 224
_N_TOTAL = 100_000
_EMBED_DIM = 1024

_CONSISTENCY_THRESHOLD = 0.9999
_CONSISTENCY_SAMPLE = 50
_NORM_TOL = 1e-5


class _FrameDataset(Dataset):
    def __init__(self, frames_dir: Path, indices: List[int]):
        self.frames_dir = frames_dir
        self.indices = indices
        self.mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        fi = self.indices[k]
        path = self.frames_dir / f"frame_{fi:08d}.png"
        im = Image.open(path).convert("RGB")
        w, h = im.size
        assert w == 256 and h == 256, f"unexpected frame size {im.size}"
        left = (w - _CROP) // 2
        top = (h - _CROP) // 2
        im = im.crop((left, top, left + _CROP, top + _CROP))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - self.mean) / self.std
        return fi, t


def _encode_all(
    frames_dir: Path,
    model_id: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, str, float]:
    print(f"[encode] loading model {model_id}", flush=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    p_sum = sum(p.numel() for p in model.parameters())
    p_first = next(model.parameters()).data.flatten()[:8].cpu().float().tolist()
    model_hash = f"params={p_sum} first8={['%.6f' % v for v in p_first]}"
    print(f"[encode] {model_hash}", flush=True)

    indices = list(range(_N_TOTAL))
    ds = _FrameDataset(frames_dir, indices)
    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=True,
    )

    out = np.zeros((_N_TOTAL, _EMBED_DIM), dtype=np.float32)
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for fi_batch, x_batch in loader:
            x_batch = x_batch.to(device, dtype=torch.float16, non_blocking=True)
            out_dict = model(pixel_values=x_batch)
            cls = out_dict.last_hidden_state[:, 0, :].float()
            cls = F.normalize(cls, dim=1, eps=1e-12)
            cls = cls.cpu().numpy()
            for k in range(cls.shape[0]):
                out[int(fi_batch[k].item())] = cls[k]
            n_done += cls.shape[0]
            if n_done % (batch_size * 50) == 0 or n_done == len(ds):
                dt = time.time() - t0
                rate = n_done / dt if dt > 0 else 0.0
                eta = (len(ds) - n_done) / rate if rate > 0 else float("nan")
                print(f"[encode] {n_done}/{len(ds)} ({rate:.1f} f/s, ETA {eta:.0f}s)",
                      flush=True)
    elapsed = time.time() - t0
    print(f"[encode] done in {elapsed:.1f}s", flush=True)
    return out, model_hash, elapsed


def _load_annotations(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _dwell_indices_for_consistency(
    annotations: List[Dict[str, Any]],
) -> List[int]:
    """All dwell-frame indices across all 5 viewing positions."""
    out: List[int] = []
    for a in annotations:
        if a.get("phase") != "dwell":
            continue
        vp = int(a.get("viewing_position_id", 0))
        if 1 <= vp <= 5:
            out.append(int(a["frame_idx"]))
    return sorted(out)


def _consistency_check(
    new_emb: np.ndarray,
    archive_path: Path,
    annotations: List[Dict[str, Any]],
    rng: np.random.Generator,
    n_sample: int,
    threshold: float,
) -> Dict[str, Any]:
    archive = np.load(archive_path)
    if archive.shape != (_N_TOTAL, _EMBED_DIM):
        return {
            "passed": False,
            "reason": f"archive shape {archive.shape} != ({_N_TOTAL}, {_EMBED_DIM})",
        }
    dwell_idx = np.asarray(_dwell_indices_for_consistency(annotations), dtype=np.int64)
    if dwell_idx.size < n_sample:
        return {
            "passed": False,
            "reason": f"only {dwell_idx.size} dwell frames available for sampling",
        }
    chosen = rng.choice(dwell_idx, size=n_sample, replace=False)
    new_vecs = new_emb[chosen]
    arch_vecs = archive[chosen]
    # Both are already L2-normalised, but renormalise for safety.
    new_n = new_vecs / (np.linalg.norm(new_vecs, axis=1, keepdims=True) + 1e-12)
    arch_n = arch_vecs / (np.linalg.norm(arch_vecs, axis=1, keepdims=True) + 1e-12)
    cosines = np.einsum("ij,ij->i", new_n, arch_n).astype(np.float64)
    n_below = int((cosines < threshold).sum())
    failing = [
        {"frame_idx": int(chosen[i]), "cosine": float(cosines[i])}
        for i in range(len(cosines)) if cosines[i] < threshold
    ]
    return {
        "passed": n_below == 0,
        "threshold": float(threshold),
        "n_sample": int(n_sample),
        "n_below_threshold": n_below,
        "cosines_min": float(cosines.min()),
        "cosines_max": float(cosines.max()),
        "cosines_mean": float(cosines.mean()),
        "cosines_std": float(cosines.std(ddof=0)),
        "failing_samples": failing,
        "sample_indices": [int(x) for x in chosen.tolist()],
    }


def _norm_check(emb: np.ndarray, tol: float) -> Dict[str, Any]:
    norms = np.linalg.norm(emb, axis=1)
    lo, hi = 1.0 - tol, 1.0 + tol
    in_range = (norms >= lo) & (norms <= hi)
    n_bad = int((~in_range).sum())
    failing = []
    if n_bad > 0:
        bad_idx = np.where(~in_range)[0]
        failing = [
            {"frame_idx": int(bad_idx[i]), "norm": float(norms[bad_idx[i]])}
            for i in range(min(20, len(bad_idx)))
        ]
    return {
        "passed": n_bad == 0,
        "tolerance": float(tol),
        "n_total": int(norms.size),
        "n_out_of_range": n_bad,
        "norms_min": float(norms.min()),
        "norms_max": float(norms.max()),
        "norms_mean": float(norms.mean()),
        "norms_std": float(norms.std(ddof=0)),
        "first_failing": failing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames_dir", type=Path, default=_DEFAULT_FRAMES_DIR)
    parser.add_argument("--annotations", type=Path, default=_DEFAULT_ANNOT)
    parser.add_argument("--archive", type=Path, default=_DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--model_id", type=str, default=_DEFAULT_MODEL_ID)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Preconditions ---------------------------------------------------------
    if not args.frames_dir.is_dir():
        print(f"[encode] FAIL: frames_dir not found: {args.frames_dir}",
              file=sys.stderr)
        return 1
    if not args.archive.is_file():
        print(f"[encode] FAIL: archive not found: {args.archive}", file=sys.stderr)
        return 1
    if args.out.exists():
        print(f"[encode] FAIL: out path already exists (refusing to overwrite): "
              f"{args.out}", file=sys.stderr)
        return 1
    if not args.annotations.is_file():
        print(f"[encode] FAIL: annotations not found: {args.annotations}",
              file=sys.stderr)
        return 1

    # Quick file-count check (cheap)
    sample_paths = [args.frames_dir / f"frame_{i:08d}.png" for i in (0, 50000, 99999)]
    for p in sample_paths:
        if not p.is_file():
            print(f"[encode] FAIL: missing frame {p}", file=sys.stderr)
            return 1

    annotations = _load_annotations(args.annotations)
    if len(annotations) != _N_TOTAL:
        print(f"[encode] FAIL: annotations length {len(annotations)} != {_N_TOTAL}",
              file=sys.stderr)
        return 1

    # Encode ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[encode] FAIL: CUDA not available; refusing to run on CPU "
              "(would take days).", file=sys.stderr)
        return 1
    print(f"[encode] device: {device} ({torch.cuda.get_device_name(0)})",
          flush=True)
    print(f"[encode] CUDA memory free: "
          f"{torch.cuda.mem_get_info(0)[0] / 1024**3:.1f} GB", flush=True)

    emb, model_hash, encode_seconds = _encode_all(
        args.frames_dir, args.model_id, device,
        args.batch_size, args.num_workers,
    )

    # Save the new embeddings (under a temp name; rename on full pass).
    # np.save auto-appends ".npy" if the path doesn't already end in it,
    # so the temp name must end in ".npy" already to avoid a double extension.
    tmp_out = args.out.parent / (args.out.stem + ".tmp.npy")
    np.save(tmp_out, emb)
    print(f"[encode] wrote temp embeddings to {tmp_out}", flush=True)

    # Norm check ------------------------------------------------------------
    norm_result = _norm_check(emb, _NORM_TOL)
    print(f"[encode] norm check: passed={norm_result['passed']} "
          f"n_out_of_range={norm_result['n_out_of_range']} "
          f"min={norm_result['norms_min']:.6f} "
          f"max={norm_result['norms_max']:.6f}", flush=True)

    # Consistency check vs archive -----------------------------------------
    rng = np.random.default_rng(int(args.seed))
    cons_result = _consistency_check(
        emb, args.archive, annotations, rng,
        _CONSISTENCY_SAMPLE, _CONSISTENCY_THRESHOLD,
    )
    print(f"[encode] consistency check: passed={cons_result['passed']} "
          f"min_cosine={cons_result.get('cosines_min', float('nan')):.6f} "
          f"threshold={_CONSISTENCY_THRESHOLD}", flush=True)

    overall_pass = norm_result["passed"] and cons_result["passed"]

    report = {
        "config": {
            "frames_dir": str(args.frames_dir),
            "annotations": str(args.annotations),
            "archive": str(args.archive),
            "out": str(args.out),
            "model_id": args.model_id,
            "model_identity": model_hash,
            "input_resolution": _CROP,
            "imagenet_normalisation": {
                "mean": list(_IMAGENET_MEAN),
                "std": list(_IMAGENET_STD),
            },
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "seed": int(args.seed),
            "consistency_threshold": _CONSISTENCY_THRESHOLD,
            "consistency_sample_n": _CONSISTENCY_SAMPLE,
            "norm_tolerance": _NORM_TOL,
            "n_total": _N_TOTAL,
            "embed_dim": _EMBED_DIM,
        },
        "encode_seconds": float(encode_seconds),
        "norm_check": norm_result,
        "consistency_check": cons_result,
        "overall_pass": bool(overall_pass),
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(f"[encode] wrote report to {args.report}", flush=True)

    if not overall_pass:
        print("[encode] FAIL: one or more checks failed. NOT renaming temp file.",
              file=sys.stderr)
        print(f"[encode] temp file retained at: {tmp_out}", file=sys.stderr)
        return 2

    # Atomic-ish rename on full pass.
    tmp_out.rename(args.out)
    print(f"[encode] PASS. Renamed {tmp_out} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
