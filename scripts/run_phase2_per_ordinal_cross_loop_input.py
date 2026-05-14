"""Per-ordinal cross-loop input variation diagnostic on Bed (verdict-disambig).

Tests whether Bed's uniform-across-ordinals variance drift (loop 30 → 100)
is consistent with reading (i) "V2 stands — cross-item coupling produces
uniform drift regardless of per-pose input" or reading (ii) "V2 doesn't
stand — Bed's per-pose cross-loop input variation is itself uniform,
producing uniform drift via a working architectural mechanism".

Primary discriminator: ords 9 and 10 of Bed's close-up were established as
pixel-MD5 identical across Stage B loops 50/75/100 in the within-loop
invariance diagnostic. If they are ALSO pixel-MD5 identical at loop 30,
those ordinals had zero cross-loop input variation across the full
loop-30-to-loop-100 span. Their variance drift values from
variance_by_ordinal.json are then drift-under-zero-input-variation —
architecturally impossible under (ii) and expected under (i).

Supplementary: Pearson r between per-ordinal (1 − mean_cross_loop_cos) and
per-ordinal |drift|. r in [0.3, 0.7] is the non-load-bearing band per the
reviewer-chat spec (the primary discriminator is decisive there).

Embedding-path note: the reviewer-chat spec mentions
`data/dinov2_embeddings/embeddings.npy` but that file is the 100k Phase-1
substrate-degenerate baseline whose frame indices do not align with
`phase2_annotations.jsonl` (65k Phase-2 frames). The correct file for
Phase-2 frame indices — and the one used by every prior Phase-2 diagnostic
including `variance_by_ordinal.json` and `within_loop_invariance.json` —
is `data/phase2_embeddings/embeddings.npy`. The script uses the Phase-2
file and flags the discrepancy in the output JSON's `notes` field.

Output: results/inner_pam_v0/phase2_main/per_ordinal_cross_loop_input.json
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import stats as _scipy_stats

REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
FRAMES_DIR = REPO_ROOT / "data/phase2_frames"
ANNOT = REPO_ROOT / "data/phase2_annotations.jsonl"

# Path note: spec says data/dinov2_embeddings/embeddings.npy but that's the
# Phase 1 substrate-degenerate baseline (100k rows, different frame set).
# Phase 2 frame indices in phase2_annotations.jsonl map to
# data/phase2_embeddings/embeddings.npy (65k rows). All prior Phase 2
# diagnostics use the latter; we do the same.
EMB = REPO_ROOT / "data/phase2_embeddings/embeddings.npy"

VAR_ORD_JSON = REPO_ROOT / "results/inner_pam_v0/phase2_main/variance_by_ordinal.json"
WLOOP_JSON = REPO_ROOT / "results/inner_pam_v0/phase2_main/within_loop_invariance.json"
OUT = REPO_ROOT / "results/inner_pam_v0/phase2_main/per_ordinal_cross_loop_input.json"

ITEM_LABEL = "Bed"
ITEM_VP = 1
SAMPLE_LOOPS = (30, 50, 75, 100)
N_ORDINALS_EXPECTED = 11
PEARSON_NON_LOAD_BEARING_BAND = (0.3, 0.7)


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _close_up_frames(
    annotations: list[dict[str, Any]],
    viewing_position_id: int,
    loop_index: int,
) -> list[int]:
    """Return frame indices of close-up frames at (item, loop), in temporal
    order — matches the resolution logic used by within_loop_invariance.json
    and variance_by_ordinal.json."""
    out: list[int] = []
    for a in annotations:
        if (
            int(a.get("viewing_position_id", -1)) == viewing_position_id
            and int(a.get("loop_index", -1)) == loop_index
            and a.get("phase_segment") == "close_up"
        ):
            out.append(int(a["frame_idx"]))
    return out


def _frame_pixel_md5(frame_idx: int) -> str:
    p = FRAMES_DIR / f"frame_{frame_idx:08d}.png"
    arr = np.asarray(Image.open(p).convert("RGB"))
    return hashlib.md5(arr.tobytes()).hexdigest()


def _write_report_and_exit(report: dict[str, Any], exit_code: int) -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"[per-ord-input] wrote report: {OUT}", flush=True)
    return exit_code


def main() -> int:
    report: dict[str, Any] = {
        "method": (
            "For each ordinal i in Bed's 11 close-up positions, compute pairwise "
            "DINOv2 cosines across the four sample loops {30, 50, 75, 100} (6 "
            "loop pairs per ordinal). Compute per-ordinal pixel-MD5 at loop 30 "
            "and cross-reference against the 50/75/100 hashes captured in "
            "within_loop_invariance.json. Primary discriminator at ords 9 and "
            "10 (established as pixel-identical across loops 50/75/100): "
            "(MD5 identical at loop 30 too?) AND (cos(loop_30, loop_100) >= "
            "0.9999?) AND (variance drift within Bed's mean ± 2*std band?). "
            "Supplementary: Pearson r between (1 - mean cross-loop cosine) "
            "and |variance drift| across the 11 ordinals."
        ),
        "item": ITEM_LABEL,
        "viewing_position_id": int(ITEM_VP),
        "loops_sampled": list(SAMPLE_LOOPS),
        "notes": {
            "embedding_path_used": str(EMB),
            "embedding_path_note": (
                "Spec mentioned data/dinov2_embeddings/embeddings.npy but that "
                "is the Phase-1 substrate-degenerate baseline (100k rows; "
                "different frame set). Phase 2 frame indices align with "
                "data/phase2_embeddings/embeddings.npy (65k rows; matches "
                "phase2_annotations.jsonl row count). All prior Phase-2 "
                "diagnostics use the latter; we do the same."
            ),
        },
    }

    # ---- Inputs present -----------------------------------------------------
    for path in (ANNOT, EMB, VAR_ORD_JSON, WLOOP_JSON, FRAMES_DIR):
        if not (path.is_file() or path.is_dir()):
            report["sanity_check_failed"] = {
                "reason": f"missing input path: {path}",
            }
            return _write_report_and_exit(report, 1)

    annotations = _load_annotations(ANNOT)
    var_ord = json.loads(VAR_ORD_JSON.read_text())
    wloop = json.loads(WLOOP_JSON.read_text())
    embeddings = np.load(EMB)
    print(f"[per-ord-input] loaded annotations={len(annotations)}, "
          f"embeddings.shape={embeddings.shape}", flush=True)

    # ---- Task 1: resolve frame indices + cross-check against var_ord -------
    bed_var_ord = var_ord["per_item"].get(ITEM_LABEL)
    if bed_var_ord is None:
        report["sanity_check_failed"] = {
            "reason": "variance_by_ordinal.json has no Bed entry",
        }
        return _write_report_and_exit(report, 1)

    frame_idxs_by_loop: dict[int, list[int]] = {}
    for loop in SAMPLE_LOOPS:
        frames = _close_up_frames(annotations, ITEM_VP, loop)
        if len(frames) != N_ORDINALS_EXPECTED:
            report["sanity_check_failed"] = {
                "reason": (
                    f"Bed close-up at loop {loop} returned "
                    f"{len(frames)} frames, expected {N_ORDINALS_EXPECTED}"
                ),
                "frames_found": frames,
            }
            return _write_report_and_exit(report, 1)
        frame_idxs_by_loop[loop] = frames

    # Cross-check against variance_by_ordinal.json's target_frame_idx
    mismatches: list[dict[str, Any]] = []
    for loop in SAMPLE_LOOPS:
        var_ord_loop_block = bed_var_ord["per_loop"].get(str(loop))
        if var_ord_loop_block is None:
            mismatches.append({
                "loop": loop, "reason": "missing in variance_by_ordinal.json"
            })
            continue
        for i, entry in enumerate(var_ord_loop_block["per_ordinal"]):
            expected = int(entry["target_frame_idx"])
            actual = frame_idxs_by_loop[loop][i]
            if expected != actual:
                mismatches.append({
                    "loop": loop, "ordinal": i,
                    "expected": expected, "actual": actual,
                })
    if mismatches:
        report["sanity_check_failed"] = {
            "reason": "frame-index resolution mismatch vs variance_by_ordinal.json",
            "mismatches": mismatches,
        }
        return _write_report_and_exit(report, 1)
    print("[per-ord-input] sanity 1 PASS: frame indices match variance_by_ordinal.json",
          flush=True)

    # ---- Task 2: load + verify L2 norms ------------------------------------
    all_frame_idxs = [
        idx for loop in SAMPLE_LOOPS for idx in frame_idxs_by_loop[loop]
    ]
    emb_subset = embeddings[all_frame_idxs].astype(np.float64)
    norms = np.linalg.norm(emb_subset, axis=1)
    lo, hi = 1.0 - 1e-5, 1.0 + 1e-5
    n_out_of_range = int(((norms < lo) | (norms > hi)).sum())
    if n_out_of_range > 0:
        report["sanity_check_failed"] = {
            "reason": (
                f"embedding L2 norms out of tolerance: "
                f"{n_out_of_range} of {len(norms)} outside [1.0-1e-5, 1.0+1e-5]"
            ),
            "norms_min": float(norms.min()),
            "norms_max": float(norms.max()),
        }
        return _write_report_and_exit(report, 2)
    print(f"[per-ord-input] sanity 2 PASS: all {len(norms)} L2 norms in "
          f"[{norms.min():.7f}, {norms.max():.7f}]", flush=True)

    # ---- Task 4: pixel-MD5 per ordinal at loop 30 + cross-reference --------
    md5_by_loop_ordinal: dict[int, list[str]] = {}
    # Loop 30: compute from PNGs
    md5_by_loop_ordinal[30] = [
        _frame_pixel_md5(frame_idxs_by_loop[30][i])
        for i in range(N_ORDINALS_EXPECTED)
    ]
    # Loops 50, 75, 100: pull from within_loop_invariance.json
    wloop_per_loop = wloop.get("per_loop", {})
    for loop in (50, 75, 100):
        block = wloop_per_loop.get(str(loop))
        if block is None or "pixel_md5_per_ordinal" not in block:
            report["sanity_check_failed"] = {
                "reason": (
                    f"within_loop_invariance.json missing pixel_md5_per_ordinal "
                    f"for loop {loop}"
                ),
            }
            return _write_report_and_exit(report, 1)
        md5s = block["pixel_md5_per_ordinal"]
        if len(md5s) != N_ORDINALS_EXPECTED:
            report["sanity_check_failed"] = {
                "reason": (
                    f"within_loop_invariance.json loop {loop} has "
                    f"{len(md5s)} MD5s, expected {N_ORDINALS_EXPECTED}"
                ),
            }
            return _write_report_and_exit(report, 1)
        md5_by_loop_ordinal[loop] = list(md5s)

    # Sanity 3: re-verify ord 9 and ord 10 MD5s for loops 50/75/100 match
    # within_loop_invariance.json by recomputing from PNGs on disk.
    md5_recheck: dict[str, Any] = {}
    for loop in (50, 75, 100):
        for ord_i in (9, 10):
            fidx = frame_idxs_by_loop[loop][ord_i]
            recomputed = _frame_pixel_md5(fidx)
            stored = md5_by_loop_ordinal[loop][ord_i]
            md5_recheck[f"loop{loop}_ord{ord_i}"] = {
                "frame_idx": int(fidx),
                "stored": stored,
                "recomputed": recomputed,
                "match": bool(stored == recomputed),
            }
    md5_recheck_mismatches = [
        k for k, v in md5_recheck.items() if not v["match"]
    ]
    if md5_recheck_mismatches:
        report["sanity_check_failed"] = {
            "reason": (
                "MD5 recheck mismatch vs within_loop_invariance.json at "
                f"{md5_recheck_mismatches}"
            ),
            "md5_recheck": md5_recheck,
        }
        return _write_report_and_exit(report, 1)
    print("[per-ord-input] sanity 3 PASS: ord 9/10 MD5s for loops 50/75/100 "
          "match within_loop_invariance.json", flush=True)

    # ---- Task 3: per-ordinal pairwise cosines across loops -----------------
    loop_pairs = [
        (30, 50), (30, 75), (30, 100), (50, 75), (50, 100), (75, 100),
    ]
    per_ordinal_entries: list[dict[str, Any]] = []
    for ord_i in range(N_ORDINALS_EXPECTED):
        # Embedding rows for this ordinal across the 4 loops, in SAMPLE_LOOPS order
        emb_at_ord = embeddings[
            [frame_idxs_by_loop[l][ord_i] for l in SAMPLE_LOOPS]
        ].astype(np.float64)
        cos_mat = emb_at_ord @ emb_at_ord.T  # already L2-normed
        pairwise_cosines: dict[str, float] = {}
        for la, lb in loop_pairs:
            ia = SAMPLE_LOOPS.index(la)
            ib = SAMPLE_LOOPS.index(lb)
            pairwise_cosines[f"{la}_{lb}"] = float(cos_mat[ia, ib])
        vals = list(pairwise_cosines.values())
        mean_cos = float(np.mean(vals))
        min_cos = float(np.min(vals))

        # Variance drift loop 30 → 100 at this ordinal from variance_by_ordinal.json
        lv_30 = bed_var_ord["per_loop"]["30"]["per_ordinal"][ord_i]["mean_log_var_over_K"]
        lv_100 = bed_var_ord["per_loop"]["100"]["per_ordinal"][ord_i]["mean_log_var_over_K"]
        drift = float(lv_100) - float(lv_30)

        per_ordinal_entries.append({
            "ordinal": int(ord_i),
            "frame_idxs_per_loop": {
                str(int(l)): int(frame_idxs_by_loop[l][ord_i]) for l in SAMPLE_LOOPS
            },
            "pairwise_cosines": pairwise_cosines,
            "mean_cross_loop_cosine": mean_cos,
            "min_cross_loop_cosine": min_cos,
            "variance_drift_loop30_to_loop100": drift,
        })

    # ---- Task 5: primary discriminator at ord 9 and ord 10 -----------------
    # Aggregate Bed drift stats across all 11 ordinals (for ±2σ band).
    all_drifts = [
        float(
            bed_var_ord["per_loop"]["100"]["per_ordinal"][i]["mean_log_var_over_K"]
            - bed_var_ord["per_loop"]["30"]["per_ordinal"][i]["mean_log_var_over_K"]
        )
        for i in range(N_ORDINALS_EXPECTED)
    ]
    mean_bed_drift = float(np.mean(all_drifts))
    std_bed_drift = float(np.std(all_drifts, ddof=0))
    band_lo = mean_bed_drift - 2.0 * std_bed_drift
    band_hi = mean_bed_drift + 2.0 * std_bed_drift

    primary_disc: dict[str, dict[str, Any]] = {}
    for ord_i in (9, 10):
        md5s_at_ord = {
            str(int(l)): md5_by_loop_ordinal[l][ord_i] for l in SAMPLE_LOOPS
        }
        md5_all_identical = len(set(md5s_at_ord.values())) == 1
        cos_30_100 = per_ordinal_entries[ord_i]["pairwise_cosines"]["30_100"]
        drift_ord = per_ordinal_entries[ord_i]["variance_drift_loop30_to_loop100"]
        primary_disc[f"ord_{ord_i}"] = {
            "frame_idxs_per_loop": per_ordinal_entries[ord_i]["frame_idxs_per_loop"],
            "pixel_md5_per_loop": md5s_at_ord,
            "md5_identical_across_4_loops": bool(md5_all_identical),
            "cos_loop30_loop100": float(cos_30_100),
            "variance_drift_loop30_to_loop100": float(drift_ord),
            "within_drift_band_2sigma": bool(band_lo <= drift_ord <= band_hi),
        }

    # ---- Task 6: supplementary Pearson r -----------------------------------
    x_per_ord = [1.0 - e["mean_cross_loop_cosine"] for e in per_ordinal_entries]
    y_per_ord = [abs(e["variance_drift_loop30_to_loop100"]) for e in per_ordinal_entries]
    pearson_res = _scipy_stats.pearsonr(x_per_ord, y_per_ord)
    r_val = float(pearson_res.statistic)
    p_val = float(pearson_res.pvalue)
    in_band = bool(PEARSON_NON_LOAD_BEARING_BAND[0] <= r_val <= PEARSON_NON_LOAD_BEARING_BAND[1])

    # ---- Task 7: verdict branch pointer ------------------------------------
    cond_i_md5 = all(
        primary_disc[f"ord_{i}"]["md5_identical_across_4_loops"] for i in (9, 10)
    )
    cond_i_cos = all(
        primary_disc[f"ord_{i}"]["cos_loop30_loop100"] >= 0.9999 for i in (9, 10)
    )
    cond_i_band = all(
        primary_disc[f"ord_{i}"]["within_drift_band_2sigma"] for i in (9, 10)
    )
    reading_i_supported = bool(cond_i_md5 and cond_i_cos and cond_i_band)

    cond_ii_cos_below = any(
        primary_disc[f"ord_{i}"]["cos_loop30_loop100"] < 0.999 for i in (9, 10)
    )
    cond_ii_pearson = bool(r_val >= 0.7 and p_val < 0.05)
    reading_ii_supported = bool(cond_ii_cos_below and cond_ii_pearson)

    if reading_i_supported and not reading_ii_supported:
        pointer = "reading_i_supported"
    elif reading_ii_supported and not reading_i_supported:
        pointer = "reading_ii_supported"
    else:
        pointer = "ambiguous_pending_review"

    report["sanity_checks"] = {
        "frame_indices_match_variance_by_ordinal": True,
        "embedding_L2_norms_in_tolerance": True,
        "ord_9_10_md5_match_within_loop_invariance": True,
        "norms_min": float(norms.min()),
        "norms_max": float(norms.max()),
        "md5_recheck": md5_recheck,
    }
    report["primary_discriminator"] = primary_disc
    report["bed_drift_distribution_across_11_ordinals"] = {
        "mean": mean_bed_drift,
        "std_population": std_bed_drift,
        "band_2sigma_lo": band_lo,
        "band_2sigma_hi": band_hi,
        "per_ordinal_drifts": all_drifts,
    }
    report["per_ordinal"] = per_ordinal_entries
    report["supplementary_pearson"] = {
        "x_per_ordinal": x_per_ord,
        "y_per_ordinal": y_per_ord,
        "r": r_val,
        "p": p_val,
        "in_non_load_bearing_band": in_band,
        "non_load_bearing_band": list(PEARSON_NON_LOAD_BEARING_BAND),
    }
    report["verdict_branch_pointer"] = pointer
    report["verdict_branch_logic_evaluation"] = {
        "reading_i_supported": reading_i_supported,
        "reading_i_components": {
            "ord_9_10_md5_identical_all_4_loops": cond_i_md5,
            "ord_9_10_cos_30_100_>=_0.9999": cond_i_cos,
            "ord_9_10_drift_within_2sigma_band": cond_i_band,
        },
        "reading_ii_supported": reading_ii_supported,
        "reading_ii_components": {
            "ord_9_or_10_cos_<_0.999": cond_ii_cos_below,
            "pearson_r_>=_0.7_and_p_<_0.05": cond_ii_pearson,
        },
    }

    # ---- Print summary -----------------------------------------------------
    print("[per-ord-input] --- primary discriminator ---", flush=True)
    for ord_i in (9, 10):
        d = primary_disc[f"ord_{ord_i}"]
        print(
            f"  ord{ord_i}: md5_identical_4loops={d['md5_identical_across_4_loops']} "
            f"cos(30,100)={d['cos_loop30_loop100']:.6f} "
            f"drift={d['variance_drift_loop30_to_loop100']:+.4f} "
            f"in_2sigma_band={d['within_drift_band_2sigma']}",
            flush=True,
        )
    print(
        f"[per-ord-input] bed drift dist: mean={mean_bed_drift:+.4f} "
        f"std={std_bed_drift:.4f} band=[{band_lo:+.4f}, {band_hi:+.4f}]",
        flush=True,
    )
    print(
        f"[per-ord-input] supplementary Pearson: r={r_val:.4f} p={p_val:.4g} "
        f"in_non_load_bearing_band={in_band}",
        flush=True,
    )
    print(f"[per-ord-input] verdict_branch_pointer = {pointer}", flush=True)

    return _write_report_and_exit(report, 0)


if __name__ == "__main__":
    raise SystemExit(main())
