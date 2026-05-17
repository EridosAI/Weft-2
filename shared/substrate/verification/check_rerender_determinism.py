"""Determinism check for the seed-7 furniture frame re-render.

Verifies that re-encoding the saved RGB frames through V-JEPA 2
reproduces the embeddings already in the previous repo's bank, to
within numerical tolerance. If this passes, the saved frames are
substrate-verifiable infrastructure for downstream encoder
verification batches (DINOv2 next).

Procedure:
  1. Load the original bank's L2-normalised V-JEPA 2 embeddings from
     `/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/main/memory_bank_embeddings.npy`.
  2. Load `frame_annotations.jsonl` (same path) to find dwell frames
     at viewing_position_id ∈ {1..5}.
  3. Sample 10 dwell frame indices per viewing position (50 total).
  4. For each sampled index:
       - Load `frame_{idx:08d}.png` from the rerender frames dir.
       - Apply the exact preprocessing the original training used
         (`src.env.push_t_staged.frame_to_encoder_tensor` from the
         previous repo).
       - Forward through V-JEPA 2 (same checkpoint as original).
       - L2-normalise.
       - Compute cosine similarity vs the bank embedding at the same
         frame_idx.
  5. Pass criterion: cosine > 0.9999 for every sampled frame
     (effectively bit-identical, modulo float-rounding).

Outputs:
  - `results/frame_rerender/determinism_check.json`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# The determinism check needs the previous repo's encoder + preprocessing
# to ensure bit-identical re-encoding.
_PREV_REPO = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft")
if str(_PREV_REPO) not in sys.path:
    sys.path.insert(0, str(_PREV_REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.encoders.frozen_vjepa2 import FrozenVJepa2Encoder  # noqa: E402
from src.env.push_t_staged import frame_to_encoder_tensor  # noqa: E402


_NUM_ITEMS = 5
_SAMPLES_PER_ITEM = 10
_DEFAULT_PASS_THRESHOLD = 0.9999


def _load_annotations(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _dwell_indices_by_item(annotations: List[Dict[str, Any]],
                            n_embeddings: int) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {i: [] for i in range(1, _NUM_ITEMS + 1)}
    for a in annotations:
        if a.get("phase") != "dwell":
            continue
        vp = int(a.get("viewing_position_id", 0))
        if 1 <= vp <= _NUM_ITEMS:
            idx = int(a["frame_idx"])
            if 0 <= idx < n_embeddings:
                out[vp].append(idx)
    return out


def _l2(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32, copy=True)
    return (v / n).astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--frames_dir",
        type=Path,
        default=Path("data") / "seed7_furniture_frames",
    )
    parser.add_argument(
        "--bank_dir",
        type=Path,
        default=_PREV_REPO / "results" / "stage_0b_furniture" / "main",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=Path("results") / "frame_rerender" / "determinism_check.json",
    )
    parser.add_argument("--samples_per_item", type=int, default=_SAMPLES_PER_ITEM)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pass_threshold", type=float, default=_DEFAULT_PASS_THRESHOLD)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    bank_emb_path = args.bank_dir / "memory_bank_embeddings.npy"
    bank_annot_path = args.bank_dir / "frame_annotations.jsonl"
    if not bank_emb_path.is_file():
        print(f"[verify] bank embeddings not found: {bank_emb_path}",
              flush=True, file=sys.stderr)
        return 1
    if not bank_annot_path.is_file():
        print(f"[verify] bank annotations not found: {bank_annot_path}",
              flush=True, file=sys.stderr)
        return 1
    if not args.frames_dir.is_dir():
        print(f"[verify] frames dir not found: {args.frames_dir}",
              flush=True, file=sys.stderr)
        return 1

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[verify] device={device}", flush=True)

    print(f"[verify] loading bank embeddings: {bank_emb_path}", flush=True)
    bank = np.load(bank_emb_path)
    if bank.dtype != np.float32:
        bank = bank.astype(np.float32)
    n, d = bank.shape
    print(f"[verify] bank: ({n}, {d}) {bank.dtype}", flush=True)

    print(f"[verify] loading annotations: {bank_annot_path}", flush=True)
    annotations = _load_annotations(bank_annot_path)
    if len(annotations) != n:
        print(f"[verify] FAIL: annotations length {len(annotations)} != bank N {n}",
              flush=True, file=sys.stderr)
        return 1

    by_item = _dwell_indices_by_item(annotations, n)
    rng = np.random.default_rng(int(args.seed))
    sampled: List[Dict[str, Any]] = []
    for vp in range(1, _NUM_ITEMS + 1):
        pool = by_item[vp]
        if len(pool) < args.samples_per_item:
            print(f"[verify] FAIL: item {vp} has only {len(pool)} dwell "
                  f"frames (< {args.samples_per_item} required)",
                  flush=True, file=sys.stderr)
            return 1
        chosen = rng.choice(pool, size=args.samples_per_item, replace=False).tolist()
        for c in chosen:
            sampled.append({"frame_idx": int(c), "viewing_position_id": int(vp)})
    print(f"[verify] sampled {len(sampled)} frames "
          f"({args.samples_per_item}/item × {_NUM_ITEMS} items)", flush=True)

    print(f"[verify] loading V-JEPA 2 encoder", flush=True)
    encoder = FrozenVJepa2Encoder(
        checkpoint="facebook/vjepa2-vitl-fpc64-256",
        device=device,
    ).eval()

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    cos_min = 1.0
    cos_max = 0.0
    fails: List[Dict[str, Any]] = []
    for i, s in enumerate(sampled):
        idx = s["frame_idx"]
        png = args.frames_dir / f"frame_{idx:08d}.png"
        if not png.is_file():
            print(f"[verify] FAIL: missing frame {png}", flush=True, file=sys.stderr)
            return 1
        # Round-trip the PNG → numpy uint8.
        img = Image.open(png).convert("RGB")
        frame = np.asarray(img, dtype=np.uint8)
        if frame.shape != (256, 256, 3):
            print(f"[verify] FAIL: unexpected frame shape {frame.shape} for {png}",
                  flush=True, file=sys.stderr)
            return 1

        with torch.no_grad():
            tensor = frame_to_encoder_tensor(frame).to(device)
            emb = encoder.encode_frame(tensor).squeeze(0).detach().cpu().numpy()

        # Bank stores L2-normalised; re-normalise the fresh embedding for
        # cosine = dot.
        emb_n = _l2(emb)
        bank_n = bank[idx]  # already L2-normalised by MemoryBank.append
        cos = float(np.dot(emb_n, bank_n))
        cos_min = min(cos_min, cos)
        cos_max = max(cos_max, cos)
        rec = {
            "frame_idx": idx,
            "viewing_position_id": s["viewing_position_id"],
            "cosine_vs_bank": cos,
            "passes": cos > args.pass_threshold,
        }
        results.append(rec)
        if not rec["passes"]:
            fails.append(rec)
        if (i + 1) % 10 == 0:
            print(f"[verify] {i + 1}/{len(sampled)} cos_min={cos_min:.6f} "
                  f"cos_max={cos_max:.6f} elapsed={time.time() - t0:.1f}s",
                  flush=True)

    all_pass = (len(fails) == 0)
    payload = {
        "config": {
            "frames_dir": str(args.frames_dir),
            "bank_dir": str(args.bank_dir),
            "samples_per_item": int(args.samples_per_item),
            "seed": int(args.seed),
            "pass_threshold": float(args.pass_threshold),
            "device": str(device),
        },
        "n_samples": len(sampled),
        "n_passes": len(results) - len(fails),
        "n_fails": len(fails),
        "cos_min": float(cos_min),
        "cos_max": float(cos_max),
        "verdict": "PASS" if all_pass else "FAIL",
        "fails": fails,
        "per_sample": results,
    }
    args.out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[verify] wrote {args.out_path}", flush=True)
    print(f"[verify] VERDICT: {payload['verdict']} "
          f"(n_pass={payload['n_passes']}/{payload['n_samples']}, "
          f"cos_min={cos_min:.6f}, cos_max={cos_max:.6f})", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
