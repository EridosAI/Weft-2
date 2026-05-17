"""DINOv2 cross-instance stability test on jittered seed-7 frames.

Loads the jittered-loop frames produced by the previous repo's
`scripts/run_furniture_stability_collect.py`, encodes the dwell
frames with DINOv2 ViT-L/14 (same model, same preprocessing as
`scripts/run_encoder_verification_dinov2.py`), and computes
cross-instance cosine similarities per viewing_position_id.

Why: the prior DINOv2 verification (Check 2 / Check 3 PASS) had a
degenerate Check 1 because the original rerender frames were bit-
identical per viewing position across loops. This script fills the
Check 1 gap by computing within-instance cosines on a substrate
where each dwell frame has genuinely-different pixel content (per-
frame position+heading jitter applied at teleport time inside the
explorer).

The §5.1 pass criterion (aggregated mean cosine > 0.75) is the test;
the comparison to the prior degenerate `1.0000` is purely diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel


_NUM_ITEMS = 5
_DEFAULT_PAIRS = 50
_DEFAULT_SEED = 7
_DEFAULT_FRAMES_DIR = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/data/seed7_dinov2_stability_frames"
)
_DEFAULT_ANNOT = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/"
    "data/seed7_dinov2_stability_annotations.jsonl"
)
_DEFAULT_MODEL_ID = "facebook/dinov2-large"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CROP = 224


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
    annotations: List[Dict[str, Any]],
) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    by_item: Dict[int, List[int]] = {i: [] for i in range(1, _NUM_ITEMS + 1)}
    item_types: Dict[int, str] = {}
    for a in annotations:
        if a.get("phase") != "dwell":
            continue
        vp = int(a.get("viewing_position_id", 0))
        if 1 <= vp <= _NUM_ITEMS:
            by_item[vp].append(int(a["frame_idx"]))
            if vp not in item_types and a.get("furniture_object_type"):
                item_types[vp] = str(a["furniture_object_type"])
    return by_item, item_types


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
        im = Image.open(self.frames_dir / f"frame_{fi:08d}.png").convert("RGB")
        w, h = im.size
        assert w == 256 and h == 256, f"unexpected frame size {im.size}"
        left = (w - _CROP) // 2
        top = (h - _CROP) // 2
        im = im.crop((left, top, left + _CROP, top + _CROP))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - self.mean) / self.std
        return fi, t


def _encode_dinov2(
    frames_dir: Path, indices: List[int], device: torch.device,
    model_id: str, batch_size: int = 64, num_workers: int = 4,
) -> Tuple[Dict[int, np.ndarray], str]:
    print(f"[stability] loading model {model_id}", flush=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()
    p_sum = sum(p.numel() for p in model.parameters())
    p_first = next(model.parameters()).data.flatten()[:8].cpu().float().tolist()
    model_hash = f"params={p_sum} first8={['%.6f' % v for v in p_first]}"
    print(f"[stability] {model_hash}", flush=True)

    ds = _FrameDataset(frames_dir, indices)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)
    out: Dict[int, np.ndarray] = {}
    t0 = time.time()
    with torch.no_grad():
        for fi_batch, x_batch in loader:
            x_batch = x_batch.to(device, dtype=torch.float16, non_blocking=True)
            cls = model(pixel_values=x_batch).last_hidden_state[:, 0, :].float()
            cls = F.normalize(cls, dim=1, eps=1e-12).cpu().numpy()
            for k in range(cls.shape[0]):
                out[int(fi_batch[k].item())] = cls[k]
    dt = time.time() - t0
    print(f"[stability] encoded {len(indices)} frames in {dt:.1f}s "
          f"({len(indices)/max(dt,1e-9):.1f} f/s)", flush=True)
    return out, model_hash


def _sample_pairs(
    frames: List[int], n_pairs: int, rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """All unique unordered pairs at this position. Sample up to n_pairs."""
    if len(frames) < 2:
        return []
    arr = np.asarray(frames, dtype=np.int64)
    # Build full pair list deterministically, then sample.
    pairs: List[Tuple[int, int]] = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            pairs.append((int(arr[i]), int(arr[j])))
    if len(pairs) <= n_pairs:
        return pairs
    pick = rng.choice(len(pairs), size=n_pairs, replace=False)
    return [pairs[i] for i in sorted(int(p) for p in pick)]


def _cosines(emb_map: Dict[int, np.ndarray],
             pairs: List[Tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros(0, dtype=np.float64)
    a = np.stack([emb_map[p[0]] for p in pairs])
    b = np.stack([emb_map[p[1]] for p in pairs])
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames_dir", type=Path, default=_DEFAULT_FRAMES_DIR)
    parser.add_argument("--annotations", type=Path, default=_DEFAULT_ANNOT)
    parser.add_argument("--out_dir", type=Path,
                        default=Path("results") / "encoder_verification_dinov2_stability")
    parser.add_argument("--model_id", type=str, default=_DEFAULT_MODEL_ID)
    parser.add_argument("--n_pairs", type=int, default=_DEFAULT_PAIRS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.annotations.is_file():
        print(f"[stability] annotations not found: {args.annotations}",
              file=sys.stderr, flush=True)
        return 1

    annotations = _load_annotations(args.annotations)
    print(f"[stability] loaded {len(annotations)} annotations from "
          f"{args.annotations}", flush=True)
    by_item, item_types = _filter_dwell_indices_by_item(annotations)
    counts = {vp: len(frs) for vp, frs in by_item.items()}
    print(f"[stability] dwell frames per item: {counts}", flush=True)
    for vp, c in counts.items():
        if c < 15:
            print(f"[stability] FAIL: item {vp} has only {c} dwell frames (< 15)",
                  file=sys.stderr, flush=True)
            return 1

    # Jitter summary by item (for the report).
    jitter_summary: Dict[int, Dict[str, float]] = {}
    by_item_jitter_scales: Dict[int, List[float]] = {vp: [] for vp in by_item}
    for a in annotations:
        if a.get("phase") != "dwell":
            continue
        vp = int(a.get("viewing_position_id", 0))
        if vp in by_item_jitter_scales:
            by_item_jitter_scales[vp].append(float(a.get("jitter_scale", 0.0)))
    for vp, scales in by_item_jitter_scales.items():
        arr = np.asarray(scales)
        jitter_summary[vp] = {
            "n_dwell": int(arr.size),
            "mean_scale": float(arr.mean()) if arr.size else 0.0,
            "n_full_scale": int(np.sum(arr == 1.0)),
            "n_half_scale": int(np.sum(arr == 0.5)),
            "n_quarter_scale": int(np.sum(arr == 0.25)),
            "n_zero_scale": int(np.sum(arr == 0.0)),
        }

    # Encode all retained dwell frames.
    all_dwell = sorted(set(fi for frs in by_item.values() for fi in frs))
    print(f"[stability] total dwell frames to encode: {len(all_dwell)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_map, model_hash = _encode_dinov2(
        args.frames_dir, all_dwell, device, args.model_id,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Sanity: norms ~1.
    sample_norms = np.linalg.norm(
        np.stack(list(emb_map.values())[: min(1000, len(emb_map))]), axis=1
    )
    print(f"[stability] dwell-frame norms (first 1000 sampled): "
          f"mean={sample_norms.mean():.6f} std={sample_norms.std():.6f}",
          flush=True)

    # Cross-instance stability: per item, sample pairs, compute cosines.
    rng = np.random.default_rng(int(args.seed))
    per_item_pairs: Dict[int, List[Tuple[int, int]]] = {}
    per_item_cosines: Dict[int, np.ndarray] = {}
    per_item_stats: Dict[int, Dict[str, float]] = {}
    for vp in sorted(by_item):
        pairs = _sample_pairs(by_item[vp], args.n_pairs, rng)
        per_item_pairs[vp] = pairs
        per_item_cosines[vp] = _cosines(emb_map, pairs)
        per_item_stats[vp] = _stats(per_item_cosines[vp])
        s = per_item_stats[vp]
        print(f"[stability] item {vp} ({item_types.get(vp, '?')}): "
              f"n={s['n']} mean={s['mean']:.4f} std={s['std']:.4f} "
              f"min={s['min']:.4f} max={s['max']:.4f}", flush=True)

    aggregate = np.concatenate(list(per_item_cosines.values()))
    aggregate_stats = _stats(aggregate)
    print(f"[stability] AGGREGATE: n={aggregate_stats['n']} "
          f"mean={aggregate_stats['mean']:.4f} std={aggregate_stats['std']:.4f}",
          flush=True)

    # Verdict.
    mean = aggregate_stats["mean"]
    if mean > 0.75:
        verdict = "PASS"
    elif mean >= 0.65:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    print(f"[stability] VERDICT: {verdict}", flush=True)

    payload = {
        "config": {
            "frames_dir": str(args.frames_dir),
            "annotations": str(args.annotations),
            "out_dir": str(out_dir),
            "model_id": args.model_id,
            "model_identity": model_hash,
            "input_resolution": _CROP,
            "imagenet_normalisation": {"mean": list(_IMAGENET_MEAN),
                                       "std": list(_IMAGENET_STD)},
            "n_pairs_per_item": int(args.n_pairs),
            "seed": int(args.seed),
        },
        "setup": {
            "n_annotations": int(len(annotations)),
            "dwell_frames_per_item": counts,
            "item_types": item_types,
            "jitter_summary_per_item": jitter_summary,
            "norm_stats_first_1000_dwell": {
                "mean": float(sample_norms.mean()),
                "std": float(sample_norms.std()),
            },
        },
        "per_item_stats": {str(k): v for k, v in per_item_stats.items()},
        "raw_cosines_per_item": {
            str(k): v.tolist() for k, v in per_item_cosines.items()
        },
        "aggregate": aggregate_stats,
        "thresholds": {"pass": 0.75, "borderline_floor": 0.65},
        "verdict": verdict,
    }
    json_path = out_dir / "stability_data.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[stability] wrote {json_path}", flush=True)

    # Markdown report.
    md: List[str] = []
    md.append("# DINOv2 Cross-Instance Stability — Report")
    md.append("")
    md.append(f"- Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append(f"- Frames source: `{args.frames_dir}` (jittered single-loop collection)")
    md.append(f"- Annotations: `{args.annotations}`")
    md.append(f"- Encoder: `{args.model_id}` (DINOv2 ViT-L/14, CLS token), "
              "frozen, eval mode, fp16")
    md.append(f"- Input: 256×256 RGB → center-crop {_CROP}×{_CROP} → ImageNet "
              "mean/std normalisation. Output L2-normalised.")
    md.append(f"- Sampling seed: `{args.seed}`")
    md.append(f"- Pairs sampled per item: `{args.n_pairs}`")
    md.append(f"- Model identity: `{model_hash}`")
    md.append("")
    md.append("## 1. Setup")
    md.append("")
    md.append("**Jitter parameters (set at collection time):**")
    md.append("")
    md.append("- `jitter_position_m = 0.2` — uniform on each horizontal axis")
    md.append("- `jitter_heading_deg = 10.0` — uniform on yaw")
    md.append("- `jitter_seed = 7` — deterministic per-frame jitter sequence")
    md.append("- Fallback ladder (per explorer): 100% → 50% → 25% → unjittered, "
              "on NavMesh-unreachable poses")
    md.append("")
    md.append("**Dwell frames retained per viewing position and jitter scales applied:**")
    md.append("")
    md.append("| viewing_position_id | object type | n dwell | jitter scale 1.0 | 0.5 | 0.25 | 0.0 (fallback) |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    for vp in sorted(counts):
        js = jitter_summary[vp]
        md.append(
            f"| {vp} | `{item_types.get(vp, '?')}` | {counts[vp]} | "
            f"{js['n_full_scale']} | {js['n_half_scale']} | "
            f"{js['n_quarter_scale']} | {js['n_zero_scale']} |"
        )
    md.append("")

    md.append("## 2. Per-viewing-position stability")
    md.append("")
    md.append("Cosines computed between all sampled within-instance pairs at each "
              "viewing position. Pairs sampled uniformly without replacement from "
              "the full C(n, 2) set per item; deterministic given the seed.")
    md.append("")
    md.append("| viewing_position_id | object type | n pairs | mean | std | min | max |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    for vp in sorted(per_item_stats):
        s = per_item_stats[vp]
        md.append(
            f"| {vp} | `{item_types.get(vp, '?')}` | {s['n']} | "
            f"{s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |"
        )
    md.append("")

    md.append("## 3. Aggregate")
    md.append("")
    md.append(
        f"- Across all {aggregate_stats['n']} sampled pairs (5 items × "
        f"{args.n_pairs} pairs each): mean **`{aggregate_stats['mean']:.4f}`**, "
        f"std `{aggregate_stats['std']:.4f}`, min `{aggregate_stats['min']:.4f}`, "
        f"max `{aggregate_stats['max']:.4f}`."
    )
    md.append(
        "- Spec §5.1 threshold: aggregated mean cosine > `0.75` (PASS), "
        "`[0.65, 0.75]` (BORDERLINE), `< 0.65` (FAIL)."
    )
    md.append(f"- Result: **{verdict}**.")
    md.append("")

    md.append("## 4. Verdict")
    md.append("")
    md.append(f"**{verdict}**")
    md.append("")
    if verdict == "PASS":
        md.append("DINOv2 ViT-L/14 CLS produces stable embeddings under per-frame "
                  "position+heading jitter on the seed-7 furniture items. Combined "
                  "with the prior DINOv2 PASS on Check 2 (cross-element "
                  "distinguishability) and Check 3 (combined gap), the §5 protocol "
                  "is met on a non-degenerate substrate.")
    elif verdict == "BORDERLINE":
        md.append("DINOv2 produces moderately stable embeddings under the configured "
                  "jitter. Outcome falls in the [0.65, 0.75] band. Recalibration / "
                  "additional variation tests / SIGReg decision is reviewer-only "
                  "per spec §5.5.")
    else:
        md.append("DINOv2's representations are not stable enough under per-frame "
                  "jitter to satisfy §5.1. Per spec §5.5, the human reviewer "
                  "decides the next path (alternative encoder, SIGReg fine-tuning, "
                  "or reframing the recurring unit).")
    md.append("")

    md.append("## 5. Comparison to prior degenerate Check 1")
    md.append("")
    md.append("The prior DINOv2 verification on the rerender's bit-identical "
              "frames returned aggregate Check 1 mean `1.0000` with std `0.0000` — "
              "a tautology measuring rendering+encoder determinism, not encoder "
              "stability under instance variation. This batch's frames carry "
              "deliberate per-frame variation; the difference between aggregate "
              f"`{aggregate_stats['mean']:.4f}` here and the prior `1.0000` "
              "is the magnitude of encoder response to the configured jitter.")
    md.append("")

    md.append("## 6. Honest interpretation")
    md.append("")
    md.append("Verdict only — this batch does not recommend an architectural path. "
              "The reviewer decides whether to proceed with DINOv2 as the v0 "
              "encoder, run additional variation tests at different jitter "
              "magnitudes, or move to SIGReg.")
    md.append("")
    md.append("**Caveat on the jitter magnitude.** `0.2 m` and `10°` are "
              "SCAFFOLDING values per batch §3 — chosen to produce visible framing "
              "variation without dramatically changing what's in view. The verdict "
              "is conditional on this jitter magnitude; a non-trivially different "
              "magnitude could produce a different aggregate. This is not a "
              "weakness specific to DINOv2 — it is inherent to any §5.1 test that "
              "doesn't first principle-derive the jitter range from a model of "
              "natural agent-instance variation. Flagged for review.")
    md.append("")

    md_path = out_dir / "STABILITY_REPORT.md"
    md_path.write_text("\n".join(md))
    print(f"[stability] wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
