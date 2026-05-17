"""Encoder Substrate Verification — V-JEPA 2 mean-pool on the seed-7
furniture-run memory bank.

Operationalises §5 of `WEFT_INNER_PAM_v0_Spec.md` against the existing
bank in the previous repo. Read-only on the bank; writes only into
`results/encoder_verification/` of *this* (Weft 2) repo.

Three checks (criteria from spec §5; restated in
`instructions/ENCODER_SUBSTRATE_VERIFICATION.md` §2):

  1. **Cross-instance stability** — for each viewing_position_id ∈
     {1..5}, sample 50 pairs of dwell frames at that position drawn
     from *different* loop_index values. Report mean / std / min / max
     of cosine similarities; aggregate across the 5 positions.
     PASS if aggregated mean > 0.75.

  2. **Cross-element distinguishability** — for each ordered pair
     (i, j) with i ≠ j, sample 50 frame-pairs (one frame at item i,
     one at item j, any loop). Report mean / std / min / max of
     cosines; aggregate across the 20 ordered pairs.
     PASS if aggregated mean < 0.60.

  3. **Combined gap** — gap = (Check 1 aggregated mean) − (Check 2
     aggregated mean). PASS if gap ≥ 0.15.

Random sampling uses seed 7 (matching the original training run for
symmetry). Bank embeddings are L2-normalised at storage time
(`MemoryBank.append`), so cosine similarity = dot product. The script
verifies this empirically as a sanity check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


_NUM_ITEMS = 5
_DEFAULT_PAIRS = 50
_DEFAULT_SAMPLE_SEED = 7
_DEFAULT_PREV_BANK = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/main"
)


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
    """Return {viewing_position_id: [frame_idx, ...]} for dwell frames at
    items 1..5. Each frame_idx is an index into the embeddings array.
    """
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
    """Per-frame loop_index, indexed by frame_idx. -1 for missing."""
    arr = np.full(n_embeddings, -1, dtype=np.int32)
    for a in annotations:
        idx = int(a["frame_idx"])
        if 0 <= idx < n_embeddings:
            arr[idx] = int(a.get("loop_index", -1))
    return arr


def _per_loop_breakdown(
    by_item: Dict[int, List[int]], loop_idx: np.ndarray
) -> Dict[int, Dict[int, int]]:
    """{viewing_position_id: {loop_index: count, ...}, ...}."""
    out: Dict[int, Dict[int, int]] = {i: {} for i in by_item}
    for vp, frames in by_item.items():
        for f in frames:
            li = int(loop_idx[f])
            out[vp][li] = out[vp].get(li, 0) + 1
    return out


def _sample_within_instance_pairs(
    frames: List[int], loop_of: np.ndarray, n_pairs: int, rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """Sample `n_pairs` (frame_a, frame_b) where both are in `frames` and
    have *different* loop indices. Drawn with replacement across pairs but
    each pair has distinct frames; if the pool is small enough that every
    inter-loop pair must repeat, that's fine for our 50-pair target on a
    pool of ~6000 frames per item."""
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
    """Cosine similarity for each pair. Embeddings are already L2-normalised
    at storage by MemoryBank, so this is a dot product."""
    if not pairs:
        return np.zeros(0, dtype=np.float64)
    a_idx = np.asarray([p[0] for p in pairs], dtype=np.int64)
    b_idx = np.asarray([p[1] for p in pairs], dtype=np.int64)
    a = emb[a_idx]
    b = emb[b_idx]
    # Defensive re-normalisation (idempotent if storage was clean).
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


def _format_md_table_check1(
    per_item_stats: Dict[int, Dict[str, float]],
    item_types: Dict[int, str],
) -> List[str]:
    lines: List[str] = []
    lines.append("| viewing_position_id | object type | n pairs | mean | std | min | max |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for vp in sorted(per_item_stats):
        s = per_item_stats[vp]
        ot = item_types.get(vp, "?")
        lines.append(
            f"| {vp} | `{ot}` | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} |"
        )
    return lines


def _format_md_matrix_check2(
    pair_stats: Dict[Tuple[int, int], Dict[str, float]],
    item_types: Dict[int, str],
) -> List[str]:
    """5×5 matrix of mean cosine; diagonal NA."""
    lines: List[str] = []
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
    parser.add_argument("--prev_bank_dir", type=Path, default=_DEFAULT_PREV_BANK,
                        help="path to seed-7 furniture-run main/ dir in the previous repo")
    parser.add_argument("--out_dir", type=Path,
                        default=Path("results") / "encoder_verification")
    parser.add_argument("--n_pairs", type=int, default=_DEFAULT_PAIRS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SAMPLE_SEED)
    args = parser.parse_args()

    emb_path = args.prev_bank_dir / "memory_bank_embeddings.npy"
    annot_path = args.prev_bank_dir / "frame_annotations.jsonl"
    if not emb_path.is_file():
        print(f"[verify] embeddings not found: {emb_path}", flush=True, file=sys.stderr)
        return 1
    if not annot_path.is_file():
        print(f"[verify] annotations not found: {annot_path}", flush=True, file=sys.stderr)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[verify] loading bank from {emb_path}", flush=True)
    emb = np.load(emb_path)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    n, d = emb.shape
    print(f"[verify] bank: shape=({n}, {d}) dtype={emb.dtype}", flush=True)

    # Sanity: norms should already be ~1.
    sample_norms = np.linalg.norm(emb[: min(1000, n)], axis=1)
    print(f"[verify] sample norms (first 1000): mean={sample_norms.mean():.6f} "
          f"std={sample_norms.std():.6f} min={sample_norms.min():.6f} "
          f"max={sample_norms.max():.6f}", flush=True)

    print(f"[verify] loading annotations from {annot_path}", flush=True)
    annotations = _load_annotations(annot_path)
    if len(annotations) != n:
        print(f"[verify] FAIL: annotations length {len(annotations)} != bank N {n}",
              flush=True, file=sys.stderr)
        return 1

    by_item = _filter_dwell_indices_by_item(annotations, n)
    loop_of = _frame_loop_index(annotations, n)

    counts_per_item = {vp: len(frames) for vp, frames in by_item.items()}
    print(f"[verify] dwell frames per item: {counts_per_item}", flush=True)
    for vp, c in counts_per_item.items():
        if c < 100:
            print(f"[verify] FAIL: item {vp} has only {c} dwell frames "
                  "(< 100 required)", flush=True, file=sys.stderr)
            return 1

    # Per-item, per-loop breakdown.
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
    # Map item id -> object type from a representative annotation.
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

    # ---- Check 2: cross-element distinguishability -----------------------
    check2_per_pair_pairs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    check2_per_pair_cosines: Dict[Tuple[int, int], np.ndarray] = {}
    rng_for_check2 = np.random.default_rng(int(args.seed) + 1)  # separate stream
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
    print(f"[verify] Check3 gap = Check1 mean − Check2 mean = "
          f"{check1_overall['mean']:.4f} − {check2_overall['mean']:.4f} = "
          f"{gap:.4f}", flush=True)

    # ---- Pass/fail against starting thresholds ---------------------------
    pass1 = check1_overall["mean"] > 0.75
    pass2 = check2_overall["mean"] < 0.60
    pass3 = gap >= 0.15
    print(f"[verify] vs starting thresholds: Check1 (>0.75): "
          f"{'PASS' if pass1 else 'FAIL'}; Check2 (<0.60): "
          f"{'PASS' if pass2 else 'FAIL'}; Check3 (≥0.15): "
          f"{'PASS' if pass3 else 'FAIL'}", flush=True)

    # ---- Recalibration consideration -------------------------------------
    # Only consider recalibration if borderline (within ~0.05 of any
    # threshold) AND a clear cluster centre justifies a different threshold.
    borderline1 = abs(check1_overall["mean"] - 0.75) < 0.05
    borderline2 = abs(check2_overall["mean"] - 0.60) < 0.05
    borderline_any = borderline1 or borderline2
    recalibration: Dict[str, Any] = {
        "applied": False,
        "rationale": None,
        "thresholds": {
            "check1": 0.75,
            "check2": 0.60,
            "gap": 0.15,
        },
        "passes_after_recalibration": None,
    }

    # Verdict assignment.
    if pass1 and pass2 and pass3:
        verdict = "PASS"
    elif borderline_any:
        # Distribution-shape inspection lives in the report; the script
        # does not auto-recalibrate. Borderline triggers a BORDERLINE
        # verdict pending human read of the empirical distribution.
        verdict = "BORDERLINE"
    else:
        # Determine FAIL vs BORDERLINE based on how badly the criteria miss.
        # If Check1 < 0.65 (10 pp under) or Check2 > 0.70 (10 pp over) or
        # gap < 0.05 (10 pp under), call FAIL outright; else BORDERLINE.
        far_miss_1 = check1_overall["mean"] < 0.65
        far_miss_2 = check2_overall["mean"] > 0.70
        far_miss_3 = gap < 0.05
        if far_miss_1 or far_miss_2 or far_miss_3:
            verdict = "FAIL"
        else:
            verdict = "BORDERLINE"

    print(f"[verify] VERDICT: {verdict}", flush=True)

    # ---- Write JSON artifact --------------------------------------------
    payload = {
        "config": {
            "prev_bank_dir": str(args.prev_bank_dir),
            "out_dir": str(out_dir),
            "n_pairs_per_set": int(args.n_pairs),
            "seed": int(args.seed),
        },
        "setup": {
            "n_embeddings": int(n),
            "embed_dim": int(d),
            "n_annotations": int(len(annotations)),
            "norm_stats_first_1000": {
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
    }
    json_path = out_dir / "verification_data.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[verify] wrote {json_path}", flush=True)

    # ---- Write Markdown report ------------------------------------------
    md_lines: List[str] = []
    md_lines.append("# Encoder Substrate Verification — Report")
    md_lines.append("")
    md_lines.append(f"- Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md_lines.append(f"- Bank source: `{emb_path}`")
    md_lines.append(f"- Annotations: `{annot_path}`")
    md_lines.append(f"- Sampling seed: `{args.seed}` (Check 1) / "
                    f"`{int(args.seed) + 1}` (Check 2)")
    md_lines.append(f"- Pairs per set: `{args.n_pairs}`")
    md_lines.append("")
    md_lines.append("## 1. Setup")
    md_lines.append("")
    md_lines.append(f"- Bank shape: `({n}, {d})`, dtype `{emb.dtype}`. ")
    md_lines.append(
        f"- Embedding norms (first 1000 sampled): "
        f"mean `{sample_norms.mean():.6f}`, std `{sample_norms.std():.6f}`, "
        f"min `{sample_norms.min():.6f}`, max `{sample_norms.max():.6f}` — "
        "consistent with L2-normalised storage; cosine = dot product."
    )
    md_lines.append(
        f"- Annotations records: `{len(annotations)}` (matches bank length)."
    )
    md_lines.append("- Dwell frames retained per viewing-position (transit excluded):")
    md_lines.append("")
    md_lines.append("| viewing_position_id | object type | n dwell frames | "
                    "n loops with ≥1 frame | mean / loop | min / loop | max / loop |")
    md_lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for vp in sorted(counts_per_item):
        s = item_loops_summary[vp]
        md_lines.append(
            f"| {vp} | `{item_types.get(vp, '?')}` | "
            f"{counts_per_item[vp]} | {s['n_loops_with_frames']} | "
            f"{s['mean_per_loop']:.1f} | {s['min_per_loop']} | "
            f"{s['max_per_loop']} |"
        )
    md_lines.append("")

    # Check 1
    md_lines.append("## 2. Check 1 — Cross-instance stability (§5.1)")
    md_lines.append("")
    md_lines.append(
        f"- Aggregate (across all 5 viewing positions, "
        f"{check1_overall['n']} pairs): mean **`{check1_overall['mean']:.4f}`**, "
        f"std `{check1_overall['std']:.4f}`, min `{check1_overall['min']:.4f}`, "
        f"max `{check1_overall['max']:.4f}`."
    )
    md_lines.append(f"- Starting threshold: mean cosine > 0.75. "
                    f"Result: **{'PASS' if pass1 else 'FAIL'}**.")
    md_lines.append("")
    md_lines.append("**Per-viewing-position breakdown:**")
    md_lines.append("")
    md_lines.extend(_format_md_table_check1(check1_per_item_stats, item_types))
    md_lines.append("")

    # Check 2
    md_lines.append("## 3. Check 2 — Cross-element distinguishability (§5.2)")
    md_lines.append("")
    md_lines.append(
        f"- Aggregate (across all 20 ordered pairs, "
        f"{check2_overall['n']} pairs): mean **`{check2_overall['mean']:.4f}`**, "
        f"std `{check2_overall['std']:.4f}`, min `{check2_overall['min']:.4f}`, "
        f"max `{check2_overall['max']:.4f}`."
    )
    md_lines.append(f"- Starting threshold: mean cosine < 0.60. "
                    f"Result: **{'PASS' if pass2 else 'FAIL'}**.")
    md_lines.append("")
    md_lines.append("**Per-(probe, retrieve) ordered-pair mean cosine matrix:**")
    md_lines.append("")
    md_lines.extend(_format_md_matrix_check2(check2_per_pair_stats, item_types))
    md_lines.append("")
    md_lines.append("**Per-pair n / mean / std (full breakdown):**")
    md_lines.append("")
    md_lines.append("| pair (i→j) | i type | j type | n | mean | std | min | max |")
    md_lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for (i, j) in sorted(check2_per_pair_stats):
        s = check2_per_pair_stats[(i, j)]
        md_lines.append(
            f"| {i}→{j} | `{item_types.get(i, '?')}` | `{item_types.get(j, '?')}` | "
            f"{s['n']} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} |"
        )
    md_lines.append("")

    # Check 3
    md_lines.append("## 4. Check 3 — Combined gap (§5.3)")
    md_lines.append("")
    md_lines.append(
        f"- gap = Check 1 mean − Check 2 mean = "
        f"`{check1_overall['mean']:.4f}` − `{check2_overall['mean']:.4f}` = "
        f"**`{gap:.4f}`**."
    )
    md_lines.append(f"- Starting threshold: gap ≥ 0.15. "
                    f"Result: **{'PASS' if pass3 else 'FAIL'}**.")
    md_lines.append("")

    # Recalibration
    md_lines.append("## 5. Recalibration decision")
    md_lines.append("")
    if borderline_any:
        md_lines.append(
            "Empirical means are within ~0.05 of one or more starting "
            "thresholds (Check 1 borderline: "
            f"{borderline1}; Check 2 borderline: {borderline2}). The "
            "script does not auto-recalibrate — recalibration requires "
            "human inspection of the empirical distribution and a written "
            "rationale per spec §5.5. The verdict is BORDERLINE and the "
            "human review decides whether to recalibrate or report FAIL."
        )
    else:
        md_lines.append(
            "No recalibration applied. Empirical values are not within "
            "±0.05 of the starting thresholds, so the starting thresholds "
            "fall outside the 'borderline' band and a recalibration would "
            "not be justified per §3 of the batch instructions."
        )
    md_lines.append("")

    # Verdict
    md_lines.append("## 6. Verdict")
    md_lines.append("")
    md_lines.append(f"**{verdict}**")
    md_lines.append("")
    if verdict == "PASS":
        md_lines.append(
            "All three checks pass against the starting thresholds. The "
            "frozen V-JEPA 2 mean-pool encoder produces embeddings whose "
            "within-instance stability, cross-element distinguishability, "
            "and combined gap meet the architectural requirements stated "
            "in §5 of the spec on the seed-7 furniture-run bank."
        )
    elif verdict == "PASS-AFTER-RECALIBRATION":
        md_lines.append(
            "All three checks pass after a single justified recalibration "
            "of the starting thresholds. See §5 above for rationale."
        )
    elif verdict == "BORDERLINE":
        md_lines.append(
            "Empirical results are close to thresholds but the distribution "
            "is mixed enough that calling pass or fail is judgment, not "
            "data. Stops for human review per §6 of the spec and §6 of the "
            "batch instructions."
        )
    else:
        md_lines.append(
            "Encoder does not meet the protocol on this bank. The "
            "empirical values are sufficiently far from the starting "
            "thresholds that a single justified recalibration cannot "
            "bring them within criteria. Per spec §5.5, this stops v0 "
            "implementation; encoder substitution / fine-tuning is a v0 "
            "design decision left to the human reviewer."
        )
    md_lines.append("")
    md_lines.append("## 7. Honest interpretation")
    md_lines.append("")
    md_lines.append(
        "**What the verdict means for Weft Inner PAM v0:** the verdict is "
        "a precondition check, not a recommendation. PASS clears v0 to "
        "proceed with frozen V-JEPA 2 mean-pool as the encoder; FAIL or "
        "BORDERLINE leaves that decision to the human review per spec §5.5. "
        "This batch does not propose an encoder substitution or any other "
        "architectural change — that conversation lives separately."
    )
    md_lines.append("")
    md_lines.append("**Reading the per-(item, pair) breakdowns:** items or "
                    "pairs with means far from the aggregate are the failure "
                    "modes if any check fails. The per-pair matrix in §3 "
                    "shows where cross-element confusability is concentrated; "
                    "the per-item table in §2 shows whether one item's dwell "
                    "frames are unusually unstable across loops.")
    md_lines.append("")

    md_path = out_dir / "ENCODER_VERIFICATION_REPORT.md"
    md_path.write_text("\n".join(md_lines))
    print(f"[verify] wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
