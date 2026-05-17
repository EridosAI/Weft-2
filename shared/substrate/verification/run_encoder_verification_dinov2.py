"""DINOv2 Substrate Verification — DINOv2 ViT-L/14 on the seed-7
furniture-run rerendered frames.

Mirrors `scripts/run_encoder_verification.py`'s protocol exactly but
encodes RGB frames with DINOv2 instead of reading V-JEPA 2 bank
embeddings. The frames are the rerender's `data/seed7_furniture_frames/`,
the annotations are the rerender's bit-identical
`frame_annotations.jsonl`, and the sampling procedure is the same so the
verdicts are directly comparable to the V-JEPA 2 result in
`results/encoder_verification/`.

Three checks (spec §5; same logic as `run_encoder_verification.py`):

  1. Cross-instance stability — 50 within-instance pairs per
     viewing_position_id ∈ {1..5}, drawn from different loop_index values.
     PASS if aggregated mean > 0.75.
  2. Cross-element distinguishability — 50 cross-element pairs per
     ordered pair (i, j), i ≠ j.
     PASS if aggregated mean < 0.60.
  3. Combined gap = Check 1 mean − Check 2 mean. PASS if gap ≥ 0.15.

Sampling seed: 7 (Check 1) / 8 (Check 2). Same as the V-JEPA 2 script,
same `numpy.random.default_rng` usage, same per-item sampling order.
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


_NUM_ITEMS = 5
_DEFAULT_PAIRS = 50
_DEFAULT_SAMPLE_SEED = 7
_DEFAULT_FRAMES_DIR = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/data/seed7_furniture_frames"
)
_DEFAULT_ANNOT = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/main/"
    "frame_annotations.jsonl"
)
_DEFAULT_MODEL_ID = "facebook/dinov2-large"
# DINOv2 standard preprocessing:
#   resize shortest edge -> 256, center-crop 224x224, normalise ImageNet
#   mean/std. Source frames are already 256x256, so the resize is a no-op.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CROP = 224


# ---------- annotations -----------------------------------------------------


def _load_annotations(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _filter_dwell_indices_by_item(
    annotations: List[Dict[str, Any]], n_embeddings: int
) -> Dict[int, List[int]]:
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


def _frame_loop_index(annotations: List[Dict[str, Any]],
                       n_embeddings: int) -> np.ndarray:
    arr = np.full(n_embeddings, -1, dtype=np.int32)
    for a in annotations:
        idx = int(a["frame_idx"])
        if 0 <= idx < n_embeddings:
            arr[idx] = int(a.get("loop_index", -1))
    return arr


def _per_loop_breakdown(
    by_item: Dict[int, List[int]], loop_idx: np.ndarray
) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {i: {} for i in by_item}
    for vp, frames in by_item.items():
        for f in frames:
            li = int(loop_idx[f])
            out[vp][li] = out[vp].get(li, 0) + 1
    return out


# ---------- DINOv2 encoding -------------------------------------------------


class _FrameDataset(Dataset):
    def __init__(self, frames_dir: Path, indices: List[int]):
        self.frames_dir = frames_dir
        self.indices = indices
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        fi = self.indices[k]
        path = self.frames_dir / f"frame_{fi:08d}.png"
        im = Image.open(path).convert("RGB")
        # Source is 256x256; center-crop to 224x224. Resize is a no-op.
        w, h = im.size
        assert w == 256 and h == 256, f"unexpected frame size {im.size}"
        left = (w - _CROP) // 2
        top = (h - _CROP) // 2
        im = im.crop((left, top, left + _CROP, top + _CROP))
        arr = np.asarray(im, dtype=np.float32) / 255.0  # H, W, 3
        t = torch.from_numpy(arr).permute(2, 0, 1)       # 3, H, W
        t = (t - self.mean) / self.std
        return fi, t


def _encode_dinov2(
    frames_dir: Path,
    dwell_indices: List[int],
    n_total: int,
    model_id: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[np.ndarray, str]:
    """Returns (embeddings (n_total, 1024) float32 L2-normalised, model_hash)."""
    print(f"[dinov2] loading model {model_id}", flush=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()
    # Model identity for reproducibility.
    p_sum = sum(p.numel() for p in model.parameters())
    p_first = next(model.parameters()).data.flatten()[:8].cpu().float().tolist()
    model_hash = f"params={p_sum} first8={['%.6f' % v for v in p_first]}"
    print(f"[dinov2] {model_hash}", flush=True)

    ds = _FrameDataset(frames_dir, dwell_indices)
    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=True
    )

    out = np.zeros((n_total, 1024), dtype=np.float32)
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for fi_batch, x_batch in loader:
            x_batch = x_batch.to(device, dtype=torch.float16, non_blocking=True)
            # forward; CLS is last_hidden_state[:, 0, :]
            out_dict = model(pixel_values=x_batch)
            cls = out_dict.last_hidden_state[:, 0, :].float()
            cls = F.normalize(cls, dim=1, eps=1e-12)
            cls = cls.cpu().numpy()
            for k in range(cls.shape[0]):
                out[int(fi_batch[k].item())] = cls[k]
            n_done += cls.shape[0]
            if n_done % (batch_size * 20) == 0 or n_done == len(ds):
                dt = time.time() - t0
                print(f"[dinov2] encoded {n_done}/{len(ds)} "
                      f"({n_done / dt:.1f} f/s)", flush=True)
    print(f"[dinov2] encoding done in {time.time() - t0:.1f}s", flush=True)
    return out, model_hash


# ---------- sampling (verbatim from run_encoder_verification.py) ------------


def _sample_within_instance_pairs(
    frames: List[int], loop_of: np.ndarray, n_pairs: int,
    rng: np.random.Generator
) -> List[Tuple[int, int]]:
    if len(frames) < 2:
        return []
    frames_arr = np.asarray(frames, dtype=np.int64)
    loops = loop_of[frames_arr]
    pairs: List[Tuple[int, int]] = []
    attempts = 0
    max_attempts = n_pairs * 50
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(len(frames_arr), size=2, replace=False)
        if loops[i] == loops[j]:
            continue
        pairs.append((int(frames_arr[i]), int(frames_arr[j])))
    return pairs


def _sample_cross_element_pairs(
    frames_a: List[int], frames_b: List[int], n_pairs: int,
    rng: np.random.Generator
) -> List[Tuple[int, int]]:
    if not frames_a or not frames_b:
        return []
    a_arr = np.asarray(frames_a, dtype=np.int64)
    b_arr = np.asarray(frames_b, dtype=np.int64)
    pairs: List[Tuple[int, int]] = []
    for _ in range(n_pairs):
        ia = int(rng.integers(0, len(a_arr)))
        ib = int(rng.integers(0, len(b_arr)))
        pairs.append((int(a_arr[ia]), int(b_arr[ib])))
    return pairs


def _cosines(emb: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros(0, dtype=np.float64)
    a_idx = np.asarray([p[0] for p in pairs], dtype=np.int64)
    b_idx = np.asarray([p[1] for p in pairs], dtype=np.int64)
    a = emb[a_idx]
    b = emb[b_idx]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.einsum("ij,ij->i", a, b).astype(np.float64)


def _stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _format_md_table_check1(per_item_stats, item_types):
    lines = ["| viewing_position_id | object type | n pairs | mean | std | min | max |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for vp in sorted(per_item_stats):
        s = per_item_stats[vp]
        ot = item_types.get(vp, "?")
        lines.append(
            f"| {vp} | `{ot}` | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} |"
        )
    return lines


def _format_md_matrix_check2(pair_stats, item_types):
    lines = []
    header = "| probe \\ retrieve |"
    sep = "|---|"
    for j in range(1, _NUM_ITEMS + 1):
        ot = item_types.get(j, "?")
        header += f" {j} ({ot}) |"
        sep += "---:|"
    lines.append(header)
    lines.append(sep)
    for i in range(1, _NUM_ITEMS + 1):
        ot_i = item_types.get(i, "?")
        row = f"| {i} ({ot_i}) |"
        for j in range(1, _NUM_ITEMS + 1):
            if i == j:
                row += " — |"
            else:
                m = pair_stats.get((i, j), {"mean": np.nan}).get("mean", np.nan)
                row += f" {m:.4f} |"
        lines.append(row)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames_dir", type=Path, default=_DEFAULT_FRAMES_DIR)
    parser.add_argument("--annotations", type=Path, default=_DEFAULT_ANNOT)
    parser.add_argument("--out_dir", type=Path,
                        default=Path("results") / "encoder_verification_dinov2")
    parser.add_argument("--embeddings_out", type=Path,
                        default=Path("data") / "dinov2_embeddings" / "embeddings.npy")
    parser.add_argument("--model_id", type=str, default=_DEFAULT_MODEL_ID)
    parser.add_argument("--n_pairs", type=int, default=_DEFAULT_PAIRS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SAMPLE_SEED)
    parser.add_argument("--n_total", type=int, default=100_000,
                        help="bank size (number of total frames); embeddings "
                             "array is sized to this so frame_idx indexing matches")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--reuse_embeddings", action="store_true",
                        help="skip encoding and load existing embeddings_out")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)

    # ---- Annotations & filter --------------------------------------------
    if not args.annotations.is_file():
        print(f"[verify] annotations not found: {args.annotations}",
              file=sys.stderr, flush=True)
        return 1
    print(f"[verify] loading annotations from {args.annotations}", flush=True)
    annotations = _load_annotations(args.annotations)
    n = args.n_total
    if len(annotations) != n:
        print(f"[verify] FAIL: annotations length {len(annotations)} != n_total {n}",
              file=sys.stderr, flush=True)
        return 1

    by_item = _filter_dwell_indices_by_item(annotations, n)
    loop_of = _frame_loop_index(annotations, n)
    counts_per_item = {vp: len(frames) for vp, frames in by_item.items()}
    print(f"[verify] dwell frames per item: {counts_per_item}", flush=True)
    for vp, c in counts_per_item.items():
        if c < 100:
            print(f"[verify] FAIL: item {vp} has only {c} dwell frames (< 100)",
                  file=sys.stderr, flush=True)
            return 1

    per_loop = _per_loop_breakdown(by_item, loop_of)
    item_loops_summary: Dict[int, Dict[str, Any]] = {}
    for vp, lc in per_loop.items():
        loops_present = sorted(lc.keys())
        per_loop_counts = [lc[l] for l in loops_present]
        item_loops_summary[vp] = {
            "n_loops_with_frames": len(loops_present),
            "min_per_loop": int(min(per_loop_counts)) if per_loop_counts else 0,
            "max_per_loop": int(max(per_loop_counts)) if per_loop_counts else 0,
            "mean_per_loop": float(np.mean(per_loop_counts)) if per_loop_counts else 0.0,
        }
    item_types: Dict[int, str] = {}
    for a in annotations:
        if a.get("phase") != "dwell":
            continue
        vp = int(a.get("viewing_position_id", 0))
        if vp in by_item and vp not in item_types:
            ot = a.get("furniture_object_type")
            if ot:
                item_types[vp] = str(ot)
        if len(item_types) == _NUM_ITEMS:
            break

    # ---- Encode dwell frames with DINOv2 ----------------------------------
    dwell_indices_all: List[int] = []
    for vp in sorted(by_item):
        dwell_indices_all.extend(by_item[vp])
    dwell_indices_all = sorted(set(dwell_indices_all))
    print(f"[verify] total unique dwell frames to encode: {len(dwell_indices_all)}",
          flush=True)

    if args.reuse_embeddings and args.embeddings_out.is_file():
        print(f"[verify] reusing existing embeddings at {args.embeddings_out}",
              flush=True)
        emb = np.load(args.embeddings_out)
        if emb.shape != (n, 1024):
            print(f"[verify] FAIL: cached embeddings shape {emb.shape} != ({n}, 1024)",
                  file=sys.stderr, flush=True)
            return 1
        model_hash = "(reused from disk)"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[verify] device: {device}", flush=True)
        emb, model_hash = _encode_dinov2(
            args.frames_dir, dwell_indices_all, n, args.model_id,
            device, batch_size=args.batch_size, num_workers=args.num_workers
        )
        print(f"[verify] saving embeddings to {args.embeddings_out}", flush=True)
        np.save(args.embeddings_out, emb)

    # Sanity: norms of dwell-position embeddings should be ~1.
    sample_idx = np.asarray(dwell_indices_all[: min(1000, len(dwell_indices_all))])
    sample_norms = np.linalg.norm(emb[sample_idx], axis=1)
    print(f"[verify] dwell-frame norms (first 1000 sampled): "
          f"mean={sample_norms.mean():.6f} std={sample_norms.std():.6f} "
          f"min={sample_norms.min():.6f} max={sample_norms.max():.6f}", flush=True)

    # ---- Check 1: cross-instance stability -------------------------------
    rng = np.random.default_rng(int(args.seed))
    check1_per_item_pairs: Dict[int, List[Tuple[int, int]]] = {}
    check1_per_item_cosines: Dict[int, np.ndarray] = {}
    for vp in sorted(by_item):
        pairs = _sample_within_instance_pairs(by_item[vp], loop_of, args.n_pairs, rng)
        check1_per_item_pairs[vp] = pairs
        check1_per_item_cosines[vp] = _cosines(emb, pairs)
        s = _stats(check1_per_item_cosines[vp])
        print(f"[verify] Check1 item {vp} ({item_types.get(vp, '?')}): "
              f"n={s['n']} mean={s['mean']:.4f} std={s['std']:.4f}", flush=True)
    check1_aggregate = np.concatenate(list(check1_per_item_cosines.values()))
    check1_overall = _stats(check1_aggregate)
    check1_per_item_stats = {
        vp: _stats(check1_per_item_cosines[vp]) for vp in check1_per_item_cosines
    }
    print(f"[verify] Check1 aggregate: mean={check1_overall['mean']:.4f} "
          f"(n={check1_overall['n']} pairs)", flush=True)

    # ---- Check 2: cross-element distinguishability ------------------------
    check2_per_pair_pairs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    check2_per_pair_cosines: Dict[Tuple[int, int], np.ndarray] = {}
    rng_for_check2 = np.random.default_rng(int(args.seed) + 1)
    for i in range(1, _NUM_ITEMS + 1):
        for j in range(1, _NUM_ITEMS + 1):
            if i == j:
                continue
            pairs = _sample_cross_element_pairs(
                by_item[i], by_item[j], args.n_pairs, rng_for_check2
            )
            check2_per_pair_pairs[(i, j)] = pairs
            check2_per_pair_cosines[(i, j)] = _cosines(emb, pairs)
    check2_aggregate = np.concatenate(list(check2_per_pair_cosines.values()))
    check2_overall = _stats(check2_aggregate)
    check2_per_pair_stats = {
        k: _stats(v) for k, v in check2_per_pair_cosines.items()
    }
    print(f"[verify] Check2 aggregate: mean={check2_overall['mean']:.4f} "
          f"(n={check2_overall['n']} pairs)", flush=True)

    # ---- Check 3: combined gap -------------------------------------------
    gap = check1_overall["mean"] - check2_overall["mean"]
    print(f"[verify] Check3 gap = {check1_overall['mean']:.4f} − "
          f"{check2_overall['mean']:.4f} = {gap:.4f}", flush=True)

    # ---- Pass/fail & verdict ---------------------------------------------
    pass1 = check1_overall["mean"] > 0.75
    pass2 = check2_overall["mean"] < 0.60
    pass3 = gap >= 0.15
    print(f"[verify] vs starting thresholds: Check1 (>0.75): "
          f"{'PASS' if pass1 else 'FAIL'}; Check2 (<0.60): "
          f"{'PASS' if pass2 else 'FAIL'}; Check3 (≥0.15): "
          f"{'PASS' if pass3 else 'FAIL'}", flush=True)

    borderline1 = abs(check1_overall["mean"] - 0.75) < 0.05
    borderline2 = abs(check2_overall["mean"] - 0.60) < 0.05
    borderline_any = borderline1 or borderline2
    recalibration: Dict[str, Any] = {
        "applied": False,
        "rationale": None,
        "thresholds": {"check1": 0.75, "check2": 0.60, "gap": 0.15},
        "passes_after_recalibration": None,
    }

    if pass1 and pass2 and pass3:
        verdict = "PASS"
    elif borderline_any:
        verdict = "BORDERLINE"
    else:
        far_miss_1 = check1_overall["mean"] < 0.65
        far_miss_2 = check2_overall["mean"] > 0.70
        far_miss_3 = gap < 0.05
        if far_miss_1 or far_miss_2 or far_miss_3:
            verdict = "FAIL"
        else:
            verdict = "BORDERLINE"
    print(f"[verify] VERDICT: {verdict}", flush=True)

    # ---- V-JEPA 2 comparison ---------------------------------------------
    # Load prior result for direct comparison.
    vjepa_path = Path("results/encoder_verification/verification_data.json")
    comparison: Dict[str, Any] = {}
    if vjepa_path.is_file():
        vj = json.loads(vjepa_path.read_text())
        vj1 = vj["check1_cross_instance_stability"]["aggregate"]["mean"]
        vj2 = vj["check2_cross_element_distinguishability"]["aggregate"]["mean"]
        vj3 = vj["check3_combined_gap"]["gap"]
        vj_verdict = vj.get("verdict", "?")
        vj_per_pair = vj["check2_cross_element_distinguishability"]["per_pair_stats"]
        comparison = {
            "vjepa_check1_mean": vj1,
            "vjepa_check2_mean": vj2,
            "vjepa_check3_gap": vj3,
            "vjepa_verdict": vj_verdict,
            "vjepa_per_pair_means": {k: v["mean"] for k, v in vj_per_pair.items()},
        }

    # ---- JSON artifact ----------------------------------------------------
    payload = {
        "config": {
            "frames_dir": str(args.frames_dir),
            "annotations": str(args.annotations),
            "out_dir": str(out_dir),
            "embeddings_path": str(args.embeddings_out),
            "model_id": args.model_id,
            "model_identity": model_hash,
            "input_resolution": _CROP,
            "imagenet_normalisation": {"mean": list(_IMAGENET_MEAN),
                                       "std": list(_IMAGENET_STD)},
            "n_pairs_per_set": int(args.n_pairs),
            "seed": int(args.seed),
        },
        "setup": {
            "n_embeddings": int(n),
            "embed_dim": int(emb.shape[1]),
            "n_annotations": int(len(annotations)),
            "norm_stats_first_1000_dwell": {
                "mean": float(sample_norms.mean()),
                "std": float(sample_norms.std()),
                "min": float(sample_norms.min()),
                "max": float(sample_norms.max()),
            },
            "dwell_frames_per_item": counts_per_item,
            "per_item_loop_summary": item_loops_summary,
            "item_types": item_types,
        },
        "check1_cross_instance_stability": {
            "per_item_stats": {str(k): v for k, v in check1_per_item_stats.items()},
            "aggregate": check1_overall,
            "starting_threshold": 0.75,
            "passes_starting_threshold": pass1,
            "raw_cosines_per_item": {
                str(k): v.tolist() for k, v in check1_per_item_cosines.items()
            },
        },
        "check2_cross_element_distinguishability": {
            "per_pair_stats": {
                f"{i}->{j}": v for (i, j), v in check2_per_pair_stats.items()
            },
            "aggregate": check2_overall,
            "starting_threshold": 0.60,
            "passes_starting_threshold": pass2,
            "raw_cosines_per_pair": {
                f"{i}->{j}": v.tolist() for (i, j), v in check2_per_pair_cosines.items()
            },
        },
        "check3_combined_gap": {
            "gap": float(gap),
            "starting_threshold": 0.15,
            "passes_starting_threshold": pass3,
        },
        "recalibration": recalibration,
        "verdict": verdict,
        "vjepa_comparison": comparison,
    }
    json_path = out_dir / "verification_data.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[verify] wrote {json_path}", flush=True)

    # ---- Markdown report --------------------------------------------------
    md: List[str] = []
    md.append("# Encoder Substrate Verification (DINOv2) — Report")
    md.append("")
    md.append(f"- Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append(f"- Frames source: `{args.frames_dir}` (re-rendered seed-7 furniture run)")
    md.append(f"- Annotations: `{args.annotations}`")
    md.append(f"- Encoder: `{args.model_id}` (DINOv2 ViT-L/14, CLS token), frozen, eval mode, fp16")
    md.append(f"- Input: 256×256 RGB → center-crop {_CROP}×{_CROP} → ImageNet "
              f"mean/std normalisation. Output L2-normalised.")
    md.append(f"- Sampling seed: `{args.seed}` (Check 1) / `{int(args.seed) + 1}` "
              f"(Check 2) — matches the V-JEPA 2 verification.")
    md.append(f"- Pairs per set: `{args.n_pairs}`")
    md.append(f"- Model identity: `{model_hash}`")
    md.append("")
    md.append("## 1. Setup")
    md.append("")
    md.append(f"- Bank shape: `({n}, {emb.shape[1]})`, dtype `{emb.dtype}`. ")
    md.append(
        f"- Dwell-frame norms (first 1000 sampled): "
        f"mean `{sample_norms.mean():.6f}`, std `{sample_norms.std():.6f}`, "
        f"min `{sample_norms.min():.6f}`, max `{sample_norms.max():.6f}`."
    )
    md.append(f"- Annotations records: `{len(annotations)}` (matches bank length).")
    md.append("- Dwell frames retained per viewing-position (transit excluded):")
    md.append("")
    md.append("| viewing_position_id | object type | n dwell frames | "
              "n loops with ≥1 frame | mean / loop | min / loop | max / loop |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    for vp in sorted(counts_per_item):
        s = item_loops_summary[vp]
        md.append(
            f"| {vp} | `{item_types.get(vp, '?')}` | {counts_per_item[vp]} | "
            f"{s['n_loops_with_frames']} | {s['mean_per_loop']:.1f} | "
            f"{s['min_per_loop']} | {s['max_per_loop']} |"
        )
    md.append("")

    # Check 1
    md.append("## 2. Check 1 — Cross-instance stability (§5.1)")
    md.append("")
    md.append(
        f"- Aggregate (across all 5 viewing positions, {check1_overall['n']} pairs): "
        f"mean **`{check1_overall['mean']:.4f}`**, std `{check1_overall['std']:.4f}`, "
        f"min `{check1_overall['min']:.4f}`, max `{check1_overall['max']:.4f}`."
    )
    md.append(f"- Starting threshold: mean cosine > 0.75. "
              f"Result: **{'PASS' if pass1 else 'FAIL'}**.")
    md.append("")
    md.append("**Per-viewing-position breakdown:**")
    md.append("")
    md.extend(_format_md_table_check1(check1_per_item_stats, item_types))
    md.append("")
    md.append("**Degeneracy note (carries over from the V-JEPA 2 verification).** "
              "The rerender's dwell mechanism teleports the agent to the exact "
              "same pose every dwell frame in every loop (per "
              "[`results/frame_rerender/RERENDER_REPORT.md`](../frame_rerender/RERENDER_REPORT.md)), "
              "so AI2-THOR produces bit-identical pixels per viewing position, "
              "and DINOv2 (deterministic forward, frozen, eval) maps identical "
              "pixels to identical embeddings. Within-instance cosines therefore "
              "reflect rendering + encoder determinism, not encoder stability "
              "under natural instance variation. Reported honestly per the "
              "batch §2.4; not engineered around.")
    md.append("")

    # Check 2
    md.append("## 3. Check 2 — Cross-element distinguishability (§5.2)")
    md.append("")
    md.append(
        f"- Aggregate (across all 20 ordered pairs, {check2_overall['n']} pairs): "
        f"mean **`{check2_overall['mean']:.4f}`**, std `{check2_overall['std']:.4f}`, "
        f"min `{check2_overall['min']:.4f}`, max `{check2_overall['max']:.4f}`."
    )
    md.append(f"- Starting threshold: mean cosine < 0.60. "
              f"Result: **{'PASS' if pass2 else 'FAIL'}**.")
    md.append("")
    md.append("**Per-(probe, retrieve) ordered-pair mean cosine matrix:**")
    md.append("")
    md.extend(_format_md_matrix_check2(check2_per_pair_stats, item_types))
    md.append("")
    md.append("**Per-pair n / mean / std (full breakdown):**")
    md.append("")
    md.append("| pair (i→j) | i type | j type | n | mean | std | min | max |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|")
    for (i, j) in sorted(check2_per_pair_stats):
        s = check2_per_pair_stats[(i, j)]
        md.append(
            f"| {i}→{j} | `{item_types.get(i, '?')}` | `{item_types.get(j, '?')}` | "
            f"{s['n']} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} |"
        )
    md.append("")

    # Check 3
    md.append("## 4. Check 3 — Combined gap (§5.3)")
    md.append("")
    md.append(
        f"- gap = Check 1 mean − Check 2 mean = "
        f"`{check1_overall['mean']:.4f}` − `{check2_overall['mean']:.4f}` = "
        f"**`{gap:.4f}`**."
    )
    md.append(f"- Starting threshold: gap ≥ 0.15. "
              f"Result: **{'PASS' if pass3 else 'FAIL'}**.")
    md.append("")

    # Recalibration
    md.append("## 5. Recalibration decision")
    md.append("")
    if borderline_any:
        md.append("Empirical means are within ~0.05 of one or more starting "
                  f"thresholds (Check 1 borderline: {borderline1}; Check 2 "
                  f"borderline: {borderline2}). The script does not "
                  "auto-recalibrate. Verdict BORDERLINE pending human read.")
    else:
        md.append("No recalibration applied. Empirical values are not within "
                  "±0.05 of the starting thresholds; recalibration would not "
                  "be justified per the batch's §3 discipline.")
    md.append("")

    # Verdict
    md.append("## 6. Verdict")
    md.append("")
    md.append(f"**{verdict}**")
    md.append("")
    if verdict == "PASS":
        md.append("All three checks pass against the starting thresholds. "
                  "DINOv2 ViT-L/14 CLS produces embeddings whose "
                  "within-instance stability, cross-element distinguishability, "
                  "and combined gap meet the architectural requirements stated "
                  "in §5 of the spec on the seed-7 furniture-run rerendered frames.")
    elif verdict == "PASS-AFTER-RECALIBRATION":
        md.append("All three checks pass after a single justified recalibration "
                  "of the starting thresholds. See §5 for rationale.")
    elif verdict == "BORDERLINE":
        md.append("Empirical results are close to thresholds but the distribution "
                  "is mixed enough that calling pass or fail is judgment, not data. "
                  "Stops for human review.")
    else:
        md.append("Encoder does not meet the protocol on this bank. The empirical "
                  "values are sufficiently far from the starting thresholds that a "
                  "single justified recalibration cannot bring them within criteria.")
    md.append("")

    # V-JEPA 2 comparison
    md.append("## 7. Direct comparison to V-JEPA 2 mean-pool")
    md.append("")
    if comparison:
        md.append("Both verifications use the same sampling seed (7 / 8), same "
                  "viewing-position filter, same dwell-frame index pool, same "
                  "50 pairs per set. Differences are encoder-only.")
        md.append("")
        md.append("| metric | DINOv2 ViT-L/14 (this batch) | V-JEPA 2 mean-pool (prior) | difference |")
        md.append("|---|---:|---:|---:|")
        md.append(f"| Check 1 aggregate mean | `{check1_overall['mean']:.4f}` | "
                  f"`{comparison['vjepa_check1_mean']:.4f}` | "
                  f"`{check1_overall['mean'] - comparison['vjepa_check1_mean']:+.4f}` |")
        md.append(f"| Check 2 aggregate mean | `{check2_overall['mean']:.4f}` | "
                  f"`{comparison['vjepa_check2_mean']:.4f}` | "
                  f"`{check2_overall['mean'] - comparison['vjepa_check2_mean']:+.4f}` |")
        md.append(f"| Check 3 gap | `{gap:.4f}` | "
                  f"`{comparison['vjepa_check3_gap']:.4f}` | "
                  f"`{gap - comparison['vjepa_check3_gap']:+.4f}` |")
        md.append(f"| Verdict | **{verdict}** | **{comparison['vjepa_verdict']}** | — |")
        md.append("")
        # Per-pair direct comparison.
        md.append("**Per-ordered-pair Check 2 cosines (DINOv2 vs V-JEPA 2):**")
        md.append("")
        md.append("| pair (i→j) | DINOv2 mean | V-JEPA 2 mean | difference |")
        md.append("|---|---:|---:|---:|")
        for (i, j) in sorted(check2_per_pair_stats):
            d_mean = check2_per_pair_stats[(i, j)]["mean"]
            v_mean = comparison["vjepa_per_pair_means"].get(f"{i}->{j}", float("nan"))
            md.append(
                f"| {i}→{j} ({item_types.get(i,'?')}→{item_types.get(j,'?')}) | "
                f"`{d_mean:.4f}` | `{v_mean:.4f}` | "
                f"`{d_mean - v_mean:+.4f}` |"
            )
    else:
        md.append("(V-JEPA 2 verification_data.json not found — comparison skipped.)")
    md.append("")

    # Honest interpretation
    md.append("## 8. Honest interpretation")
    md.append("")
    md.append("**What the verdict means.** Verdict only — this batch does not "
              "propose architectural next steps. PASS on DINOv2 would mean an "
              "off-the-shelf encoder option exists that meets §5; FAIL means the "
              "encoder substrate problem is not solvable by swapping V-JEPA 2 "
              "mean-pool for DINOv2 on this bank, and the §5.5 path is left to "
              "human review.")
    md.append("")
    md.append("**Caveat from the re-render (carries from RERENDER_REPORT §note-for-next-batch).** "
              "Items 3 (Dresser) and 4 (Sofa) — both in LivingRoom — have "
              "constant per-item offsets from the original V-JEPA 2 bank's frames "
              "at the cosine `0.0005`–`0.0008` level (V-JEPA 2 reading). DINOv2 "
              "is its own encoder and re-encodes the rerender's frames "
              "directly, so its protocol numbers are internally consistent. The "
              "caveat is recorded in case any downstream analysis surfaces an "
              "unexplained discrepancy at that magnitude.")
    md.append("")
    md.append("**Check 1 is degenerate on this bank (same reason as V-JEPA 2).** "
              "The per-item std is `0.0000` because dwell frames at the same "
              "viewing position are bit-identical across loops within the "
              "rerender (verified in RERENDER_REPORT). Check 2 is load-bearing.")
    md.append("")

    md_path = out_dir / "ENCODER_VERIFICATION_DINOV2_REPORT.md"
    md_path.write_text("\n".join(md))
    print(f"[verify] wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
