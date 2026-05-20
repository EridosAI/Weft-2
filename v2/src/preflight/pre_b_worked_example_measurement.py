"""V2-PRE-B — worked-example measurement (instr §7.3; spec §6).

Applies the §6 measurement protocol to the DINOv2-on-AI2-THOR worked-example
trajectory collection. This module provides the PRE-B *verification gates*
(segmentation SSM + reference-state cluster check) and the §2.4 range
extraction; the full protocol application + distributions + W4 build on top
once segmentation is confirmed.

Worked-example data (verified correct substrate version — the
`_assign_close_up_ordinals` fix is driver-only and does not affect the DINOv2
forward-pass over pre-rendered frames; spec §6.1):
  embeddings  : v0/data/phase2_embeddings/embeddings.npy   (65000 x 1024, L2-normed)
  annotations : v0/data/phase2_annotations.jsonl

Trajectory structure (from annotations, authoritative):
  181 loops (0..180), ~359 frames/loop, monotonic frame order. Each loop
  traverses 5 items (Bed/DiningTable/Dresser/Sofa/Television) with transit +
  close-up segments. Stage A (clean) = loops 0..30; Stage B
  (livingroom_retexture active) = loops 31..180. NOTE: this is 181 loops, not
  the "5 loops" sketched in instr §7.3 — see the segmentation report.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from v2.config import REPO_ROOT

EMB_PATH = REPO_ROOT / "v0" / "data" / "phase2_embeddings" / "embeddings.npy"
ANN_PATH = REPO_ROOT / "v0" / "data" / "phase2_annotations.jsonl"


# --------------------------------------------------------------------------
# Loading + segmentation
# --------------------------------------------------------------------------


def load_worked_example() -> tuple[np.ndarray, list[dict]]:
    emb = np.load(EMB_PATH)
    ann = [json.loads(l) for l in ANN_PATH.read_text().splitlines() if l.strip()]
    if emb.shape[0] != len(ann):
        raise ValueError(f"embedding/annotation length mismatch: {emb.shape[0]} vs {len(ann)}")
    return emb, ann


def segment(ann: list[dict]) -> dict:
    """Derive per-frame segmentation arrays + the close_up_ordinal (v1 driver fix).

    close_up_ordinal: 0-indexed position within each contiguous (item, loop)
    close-up segment (the `_assign_close_up_ordinals` logic; populated here from
    the annotation stream since the explorer's observation dict omits it).
    """
    n = len(ann)
    loop = np.array([r["loop_index"] for r in ann], dtype=np.int64)
    item = np.array([r["viewing_position_id"] for r in ann], dtype=np.int64)  # 0=transit,1..5
    is_close = np.array([r["phase_segment"] == "close_up" for r in ann])
    is_apex = np.array([bool(r["close_up_apex_flag"]) for r in ann])
    pert_active = np.array([bool(r["perturbation_active"]) for r in ann])

    # Stage: A (clean) = no perturbation_active anywhere in the loop; B otherwise.
    loop_has_pert = defaultdict(bool)
    for lp, pa in zip(loop.tolist(), pert_active.tolist()):
        if pa:
            loop_has_pert[lp] = True
    stage = np.array(["B" if loop_has_pert[lp] else "A" for lp in loop.tolist()])

    # close_up_ordinal within each contiguous (item, loop) close-up run.
    close_up_ordinal = np.full(n, -1, dtype=np.int64)
    run_ord = 0
    prev_key = None
    for i in range(n):
        if not is_close[i]:
            prev_key = None
            continue
        key = (int(item[i]), int(loop[i]))
        if key != prev_key:
            run_ord = 0
            prev_key = key
        close_up_ordinal[i] = run_ord
        run_ord += 1

    return {
        "loop": loop, "item": item, "is_close": is_close, "is_apex": is_apex,
        "pert_active": pert_active, "stage": stage, "close_up_ordinal": close_up_ordinal,
        "n_loops": int(loop.max() + 1),
        "clean_loops": sorted(set(loop[stage == "A"].tolist())),
        "pert_loops": sorted(set(loop[stage == "B"].tolist())),
    }


# --------------------------------------------------------------------------
# Ask #1 — segmentation-verification SSM
# --------------------------------------------------------------------------


def coarse_ssm(emb: np.ndarray, block: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Block-mean-pooled cosine self-similarity matrix. Returns (ssm, block_loop_mode)."""
    n = emb.shape[0]
    nb = n // block
    pooled = np.stack([emb[i * block:(i + 1) * block].mean(axis=0) for i in range(nb)])
    pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
    ssm = pooled @ pooled.T
    return ssm, pooled


def characterize_ssm(ssm: np.ndarray, seg: dict, block: int) -> dict:
    """Quantify the SSM structure: loop period, stage boundary, item/loop coherence."""
    nb = ssm.shape[0]
    # Loop period in blocks via off-diagonal autocorrelation (skip lag 0).
    max_lag = min(nb // 2, 40)
    offdiag_mean = []
    for k in range(1, max_lag + 1):
        offdiag_mean.append(float(np.mean(np.diagonal(ssm, offset=k))))
    offdiag_mean = np.array(offdiag_mean)
    # Expected loop period in blocks ~ frames_per_loop / block.
    frames_per_loop = ssm.shape[0] * block / max(seg["n_loops"], 1)
    expected_period_blocks = frames_per_loop / block
    peak_lag = int(np.argmax(offdiag_mean)) + 1

    # Stage boundary: first perturbed loop -> approximate block index.
    first_pert_loop = seg["pert_loops"][0] if seg["pert_loops"] else None
    boundary_block = None
    if first_pert_loop is not None:
        first_pert_frame = int(np.flatnonzero(seg["loop"] == first_pert_loop)[0])
        boundary_block = first_pert_frame // block

    # Cross-stage vs within-stage block similarity (does the perturbation show up?).
    if boundary_block is not None and 0 < boundary_block < nb:
        a = slice(0, boundary_block)
        b = slice(boundary_block, nb)
        within_a = float(ssm[a, a].mean())
        within_b = float(ssm[b, b].mean())
        cross = float(ssm[a, b].mean())
    else:
        within_a = within_b = cross = None

    return {
        "n_blocks": nb, "block_frames": block,
        "expected_loop_period_blocks": round(expected_period_blocks, 3),
        "autocorr_peak_lag_blocks": peak_lag,
        "autocorr_peak_value": float(offdiag_mean.max()),
        "autocorr_profile_first10": [round(x, 4) for x in offdiag_mean[:10].tolist()],
        "stage_boundary_block": boundary_block,
        "within_stageA_block_sim": None if within_a is None else round(within_a, 4),
        "within_stageB_block_sim": None if within_b is None else round(within_b, 4),
        "cross_stage_block_sim": None if cross is None else round(cross, 4),
        "diag_minus_offdiag": round(float(1.0 - offdiag_mean.mean()), 4),
    }


# --------------------------------------------------------------------------
# Ask #2 — reference-state cluster check
# --------------------------------------------------------------------------


def reference_cluster_check(emb: np.ndarray, seg: dict, item_id: int, stage: str) -> dict:
    """Per-loop apex embeddings of one (item, stage) tuple; cosine cluster tightness.

    Empirical analogue of spec §2.5's by-construction repetition-coverage claim:
    the same (item, ordinal) viewed across loops should cluster tightly. With 181
    loops (not 5), Stage A has 31 apex samples, Stage B has up to 150.
    """
    mask = (seg["item"] == item_id) & seg["is_apex"] & (seg["stage"] == stage)
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return {"item_id": item_id, "stage": stage, "n_samples": int(idx.size),
                "note": "insufficient apex samples"}
    vecs = emb[idx]
    vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    sim = vecs @ vecs.T
    iu = np.triu_indices(sim.shape[0], k=1)
    pair = sim[iu]
    centroid = vecs.mean(axis=0)
    centroid /= np.clip(np.linalg.norm(centroid), 1e-12, None)
    to_centroid = vecs @ centroid
    return {
        "item_id": item_id, "stage": stage, "n_samples": int(idx.size),
        "pairwise_cosine_mean": round(float(pair.mean()), 5),
        "pairwise_cosine_min": round(float(pair.min()), 5),
        "pairwise_cosine_std": round(float(pair.std()), 5),
        "to_centroid_cosine_mean": round(float(to_centroid.mean()), 5),
        "to_centroid_cosine_min": round(float(to_centroid.min()), 5),
    }


# --------------------------------------------------------------------------
# Ask #4 — §2.4 per-axis range extraction (with traces)
# --------------------------------------------------------------------------


def extract_s24_ranges() -> dict:
    """Extract per-axis first-principles ranges for W4, tracing each bound.

    §2.4 (pass 1) prose enumerates explicit bounds ONLY for magnitude,
    dimensionality, and repetition; it states the ranges are 'operationalised in
    pass 2 §4'. Continuity and locality bounds therefore trace to §4; repetition
    is multi-dimensional (period, fidelity, coverage) with no single numerical
    range even in §4 — flagged ambiguous for design-chat resolution (ask #4 / §8).
    """
    axes = {
        "magnitude": {
            "range": [0.0, 1.0], "ambiguous": False,
            "trace": ("pass1 §2.4 lower-bound 'For magnitude: zero (bit-identical "
                      "streams)'; upper-bound 'For magnitude: full-stream replacement "
                      "(cosine drop of ~1.0 magnitude)'; operationalised pass2 §4.1 "
                      "'M = 0 ... M = 1 when they are orthogonal'."),
        },
        "manifold_dim": {
            "range": [1.0, 1024.0], "ambiguous": False,
            "trace": ("pass1 §2.4 lower-bound 'For dimensionality: 1 (degenerate "
                      "manifold)'; upper-bound 'For dimensionality: d=1024 (full ambient "
                      "space)'; operationalised pass2 §4.5 'D = 1 ... D = d = 1024'."),
        },
        "continuity": {
            "range": [0.0, 1.0], "ambiguous": False,
            "trace": ("pass1 §2.4 does NOT enumerate continuity explicitly but states "
                      "ranges are 'operationalised in pass 2 §4'; pass2 §4.3 'C = 0 "
                      "corresponds to bit-identical consecutive positions (zero motion); "
                      "C near 1 corresponds to consecutive positions being orthogonal'."),
        },
        "locality": {
            "range": [0.0, 1.0], "ambiguous": False,
            "trace": ("pass1 §2.4 does NOT enumerate locality explicitly but states "
                      "ranges are 'operationalised in pass 2 §4'; pass2 §4.2 'L = 1 when "
                      "the perturbation affects only positions in S_pert (perfect "
                      "locality); L = 0 when the perturbation shifts every other "
                      "position's relations (no locality)'."),
        },
        "repetition": {
            "range": None, "ambiguous": True,
            "trace": ("pass1 §2.4 gives only qualitative bounds 'For repetition: no "
                      "repetition (each stream position unique)' .. 'full pattern "
                      "saturation'; pass2 §4.4 defines repetition as a multi-dimensional "
                      "(period P, fidelity F, coverage) structure with no single "
                      "numerical range. Period sweep grid is uncommitted SCAFFOLDING "
                      "(§2.6 / Phase 0.5). AMBIGUOUS — surfaced for design-chat per ask #4."),
        },
    }
    return axes


# --------------------------------------------------------------------------
# Protocol application — whole-stream trajectory, within-trajectory distributions
# (design-chat decision: trajectory unit = whole stream; reference estimated by
#  annotation alignment (item, close_up_ordinal) over Stage-A clean loops, since
#  loops are variable-length (200-360) so fixed-period i mod P does not align.)
# --------------------------------------------------------------------------

ITEM_ROOM = {1: "Bedroom", 2: "Bedroom", 3: "Bedroom", 4: "LivingRoom", 5: "LivingRoom"}
ITEM_NAME = {1: "Bed", 2: "DiningTable", 3: "Dresser", 4: "Sofa", 5: "Television"}


def estimate_reference_close_up(emb: np.ndarray, seg: dict, min_clean: int = 10):
    """Reference per close-up frame = median over Stage-A loops at same (item, ordinal).

    Returns (reference (N,d) for close-up frames with a reference, has_ref mask).
    Transit frames get no reference (connective, not repeating viewing states).
    """
    n = emb.shape[0]
    clean_by_key: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(n):
        if seg["is_close"][i] and seg["stage"][i] == "A":
            clean_by_key[(int(seg["item"][i]), int(seg["close_up_ordinal"][i]))].append(i)
    ref_by_key: dict[tuple[int, int], np.ndarray] = {}
    for key, idxs in clean_by_key.items():
        if len(idxs) >= min_clean:
            v = np.median(emb[idxs], axis=0)
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            ref_by_key[key] = v
    ref = np.zeros((n, emb.shape[1]), dtype=np.float64)
    has_ref = np.zeros(n, dtype=bool)
    for i in range(n):
        if seg["is_close"][i]:
            key = (int(seg["item"][i]), int(seg["close_up_ordinal"][i]))
            if key in ref_by_key:
                ref[i] = ref_by_key[key]
                has_ref[i] = True
    return ref, has_ref


def _summarise(values: np.ndarray, bic_threshold: float) -> dict:
    """median + IQR + §6.3 multimodal detection (GMM/BIC)."""
    from v2.src.protocol.grid_mapping import detect_multimodal
    v = np.asarray(values, dtype=np.float64)
    out = {
        "n": int(v.size),
        "median": float(np.median(v)) if v.size else None,
        "iqr": float(np.subtract(*np.percentile(v, [75, 25]))) if v.size else None,
        "min": float(v.min()) if v.size else None,
        "max": float(v.max()) if v.size else None,
    }
    if v.size >= 4:
        out["multimodal"] = detect_multimodal(v, bic_threshold)
    return out


def build_worked_example_region(emb: np.ndarray, seg: dict, thresholds: dict) -> dict:
    """Per-axis empirical distributions for the worked example (spec §6, §4)."""
    import numpy as _np
    embn = emb / _np.clip(_np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    th = thresholds
    bic = th["BIC_IMPROVEMENT_THRESHOLD"]
    n = emb.shape[0]
    is_close, item, stage = seg["is_close"], seg["item"], seg["stage"]

    ref, has_ref = estimate_reference_close_up(emb, seg)

    # --- Magnitude (§4.1): deviation from clean reference over Stage-B close-up
    #     frames; distribution over items reveals LivingRoom-vs-other bimodality.
    mag_idx = _np.flatnonzero(is_close & (stage == "B") & has_ref)
    mag_vals = 1.0 - _np.sum(embn[mag_idx] * ref[mag_idx], axis=1)
    per_item_mag = {ITEM_NAME[it]: float(_np.median(
        mag_vals[item[mag_idx] == it])) for it in range(1, 6)
        if _np.any(item[mag_idx] == it)}
    s_pert = mag_idx[(1.0 - _np.sum(embn[mag_idx] * ref[mag_idx], axis=1)) > th["TAU_PERT"]]
    magnitude = {
        "distribution": _summarise(mag_vals, bic),
        "per_item_median": per_item_mag,
        "per_item_room": {ITEM_NAME[i]: ITEM_ROOM[i] for i in range(1, 6)},
        "detected_perturbed_n": int(s_pert.size),
        "axis_value_detected_median": float(_np.median(
            mag_vals[(mag_vals > th["TAU_PERT"])])) if _np.any(mag_vals > th["TAU_PERT"]) else None,
    }

    # --- Locality (§4.2): fraction of non-perturbed close-up frames whose relations
    #     to radius-R neighbours shift. Per-Stage-B-loop, over close-up frames.
    R = int(th["REF_NEIGHBOURHOOD_WINDOW"])
    tau_L = th["TAU_L"]
    pert_mask = _np.zeros(n, dtype=bool)
    pert_mask[s_pert] = True
    loc_per_loop = []
    for lp in seg["pert_loops"]:
        fi = _np.flatnonzero((seg["loop"] == lp) & is_close & has_ref)
        if fi.size < 2:
            continue
        nonpert = [i for i in fi if not pert_mask[i]]
        if not nonpert:
            continue
        affected = 0
        fi_set = set(fi.tolist())
        for i in nonpert:
            shifted = False
            for j in range(i - R, i + R + 1):
                if j == i or j not in fi_set:
                    continue
                actual = float(embn[i] @ embn[j])
                reference = float(ref[i] @ ref[j]) if (has_ref[i] and has_ref[j]) else actual
                if abs(actual - reference) > tau_L:
                    shifted = True
                    break
            affected += int(shifted)
        loc_per_loop.append(1.0 - affected / len(nonpert))
    locality = {
        "distribution": _summarise(_np.array(loc_per_loop), bic),
        "defined": True,
        "note": ("perturbation is item-localised (LivingRoom items); items are "
                 "transit-separated so frame-adjacency relations rarely cross items "
                 "-> locality reads high. §4.2 repetition-coverage prerequisite met "
                 "(close-up states repeat across loops)."),
    }

    # --- Continuity (§4.3): per-loop C over consecutive frames.
    from v2.src.property_measure.continuity import measure_continuity
    c_per_loop = []
    for lp in range(seg["n_loops"]):
        fi = _np.flatnonzero(seg["loop"] == lp)
        if fi.size >= 2:
            c_per_loop.append(measure_continuity(embn[fi])["C"])
    continuity = {
        "distribution": _summarise(_np.array(c_per_loop), bic),
        "global_C": measure_continuity(embn)["C"],
        "note": "per-loop C over consecutive frames (transit + close-up).",
    }

    # --- Manifold dim (§4.5): global PR over whole stream + per-loop local PR.
    from v2.src.property_measure.manifold_dim import measure_manifold_dim
    md_global = measure_manifold_dim(embn, window=th["LOCAL_PCA_WINDOW"],
                                     subsample_rate=th["MANIFOLD_SUBSAMPLE_RATE"])
    d_per_loop = []
    for lp in range(seg["n_loops"]):
        fi = _np.flatnonzero(seg["loop"] == lp)
        if fi.size >= th["LOCAL_PCA_WINDOW"]:
            d_per_loop.append(measure_manifold_dim(
                embn[fi], window=th["LOCAL_PCA_WINDOW"],
                subsample_rate=th["MANIFOLD_SUBSAMPLE_RATE"])["D_local"])
    manifold_dim = {
        "distribution_local_per_loop": _summarise(_np.array(d_per_loop), bic),
        "global_D": md_global["D_global"],
        "local_D_whole_stream": md_global["D_local"],
    }

    # --- Repetition (§4.4): whole-stream period/fidelity/coverage + per-item coverage.
    from v2.src.property_measure.repetition import measure_repetition
    rep = measure_repetition(embn, tau_R=th["TAU_R"], noise_floor=th["REPETITION_NOISE_FLOOR"],
                             max_lag=min(n // 2, 800))  # period ~359; 800 lags covers ~2 loops
    # Per-item cross-loop coverage = fraction of (item,ordinal) keys whose Stage-A
    # cross-loop pairwise cosine exceeds tau_R (generalises the cluster check).
    repetition = {
        "whole_stream": {k: rep[k] for k in ("period", "fidelity", "coverage", "peak_offdiag_mean")},
        "w4_status": "undetermined — §2.4 repetition range unspecified (deferred to Phase 0.5 per ask #4)",
        "note": "period reflects mean loop length (~359); loops are variable-length so the peak is smeared.",
    }

    # --- W4 detection (4 axes vs §2.4 ranges; repetition deferred).
    s24 = extract_s24_ranges()
    def w4(axis_key, median):
        rng = s24[axis_key]["range"]
        if rng is None or median is None:
            return {"w4_status": "undetermined", "range": rng, "median": median}
        within = rng[0] <= median <= rng[1]
        return {"w4_status": "within-range" if within else "OUTSIDE-RANGE (W4)",
                "range": rng, "median": median}
    w4_detection = {
        "magnitude": w4("magnitude", magnitude["distribution"]["median"]),
        "locality": w4("locality", locality["distribution"]["median"]),
        "continuity": w4("continuity", continuity["distribution"]["median"]),
        "manifold_dim": w4("manifold_dim", manifold_dim["global_D"]),
        "repetition": {"w4_status": "undetermined (deferred to Phase 0.5)"},
    }
    any_w4 = any(v.get("w4_status", "").startswith("OUTSIDE") for v in w4_detection.values())

    return {
        "trajectory_unit": "whole_stream (design-chat decision); within-trajectory distributions",
        "reference_estimation": "annotation-aligned (item, close_up_ordinal) median over Stage-A clean loops",
        "axes": {
            "magnitude": magnitude, "locality": locality, "continuity": continuity,
            "manifold_dim": manifold_dim, "repetition": repetition,
        },
        "w4_detection": w4_detection,
        "any_w4_triggered": any_w4,
        "grid_mapping_status": "deferred — per-axis sweep grid is uncommitted SCAFFOLDING (Phase 0.5)",
        "s24_ranges": s24,
    }
