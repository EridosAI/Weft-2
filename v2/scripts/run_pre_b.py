"""V2-PRE-B — worked-example measurement runner (instr §7.3; spec §6).

Trajectory unit = whole stream; reference estimated by annotation alignment
(design-chat decision). Produces worked_example_region.json containing the
forensic record (coarse + close-up SSM, reference-state clusters), the per-axis
empirical distributions with §6.3 multi-modal detection, and W4 detection (4
axes vs §2.4 ranges; repetition deferred to Phase 0.5 per ask #4).
"""

from __future__ import annotations

import json

import numpy as np

from v2.config import RESULTS_PRE_B, load_calibrated_thresholds
from v2.src.preflight import pre_b_worked_example_measurement as pb


def closeup_ssm(emb: np.ndarray, seg: dict) -> dict:
    """One vector per (item, loop) apex viewing state -> compact SSM characterization.

    Shows the 5-item x 181-loop repetition structure and Stage A->B boundary that
    the transit-diluted coarse whole-stream SSM cannot resolve.
    """
    rows, meta = [], []
    for lp in range(seg["n_loops"]):
        for it in range(1, 6):
            idx = np.flatnonzero((seg["loop"] == lp) & (seg["item"] == it) & seg["is_apex"])
            if idx.size:
                v = emb[idx].mean(axis=0)
                rows.append(v / max(float(np.linalg.norm(v)), 1e-12))
                meta.append({"loop": lp, "item": pb.ITEM_NAME[it],
                             "stage": str(seg["stage"][idx[0]])})
    M = np.stack(rows)
    ssm = M @ M.T
    # Within-item (same item, across loops) vs cross-item mean similarity.
    items = np.array([m["item"] for m in meta])
    iu = np.triu_indices(len(meta), k=1)
    same_item = items[iu[0]] == items[iu[1]]
    return {
        "n_states": len(meta),
        "within_item_mean_sim": round(float(ssm[iu][same_item].mean()), 4),
        "cross_item_mean_sim": round(float(ssm[iu][~same_item].mean()), 4),
        "interpretation": ("within-item >> cross-item confirms 5 item-blocks; "
                           "within-item Stage-A clustering ~1.0 confirms loop repetition"),
        "matrix_shape": list(ssm.shape),
    }, ssm, meta


def main() -> int:
    RESULTS_PRE_B.mkdir(parents=True, exist_ok=True)
    th = load_calibrated_thresholds()
    print("[pre_b] loading worked-example trajectory ...")
    emb, ann = pb.load_worked_example()
    seg = pb.segment(ann)

    # L2-norm contract (STOP trigger 1).
    rng = np.random.default_rng(0)
    sidx = rng.choice(emb.shape[0], 1000, replace=False)
    norms = np.linalg.norm(emb[sidx], axis=1)
    contract_ok = bool(np.all(np.abs(norms - 1.0) < 1e-5))
    print(f"[pre_b] n_loops={seg['n_loops']} clean={len(seg['clean_loops'])} "
          f"pert={len(seg['pert_loops'])} | L2 contract={contract_ok}")
    if not contract_ok:
        print("[pre_b] STOP: L2-norm contract failed on worked-example cache")
        return 1

    print("[pre_b] coarse + close-up SSM ...")
    coarse, pooled = pb.coarse_ssm(emb, block=100)
    coarse_char = pb.characterize_ssm(coarse, seg, block=100)
    cu_char, cu_ssm, cu_meta = closeup_ssm(emb, seg)
    np.save(RESULTS_PRE_B / "closeup_ssm.npy", cu_ssm)

    print("[pre_b] reference-state cluster checks (5 items x 2 stages) ...")
    clusters = {}
    for it in range(1, 6):
        for st in ("A", "B"):
            clusters[f"{pb.ITEM_NAME[it]}_{st}"] = pb.reference_cluster_check(emb, seg, it, st)

    print("[pre_b] applying §6 protocol (whole-stream, within-trajectory distributions) ...")
    region = pb.build_worked_example_region(emb, seg, th)

    report = {
        "data": {"embeddings": str(pb.EMB_PATH), "annotations": str(pb.ANN_PATH),
                 "n_frames": int(emb.shape[0]),
                 "substrate_version": "correct (_assign_close_up_ordinals driver-only; §6.1)"},
        "segmentation": {"n_loops": seg["n_loops"],
                         "stage_A_loops": "0..30 (31 clean)",
                         "stage_B_loops": "31..180 (150, livingroom_retexture)",
                         "frames_per_loop_mean": 359.1,
                         "note": "5 items x 181 loops (NOT the 5x5 / 5-loop §7.3 sketch)"},
        "ask1_segmentation_ssm": {"coarse_block100": coarse_char, "closeup_state": cu_char},
        "ask2_reference_state_clusters": clusters,
        "ask4_s24_ranges": region["s24_ranges"],
        "trajectory_unit": region["trajectory_unit"],
        "reference_estimation": region["reference_estimation"],
        "per_axis_distributions": region["axes"],
        "w4_detection": region["w4_detection"],
        "any_w4_triggered": region["any_w4_triggered"],
        "grid_mapping_status": region["grid_mapping_status"],
    }
    (RESULTS_PRE_B / "worked_example_region.json").write_text(json.dumps(report, indent=2, default=str))

    # Summary.
    print("\n[pre_b] === SEGMENTATION SSM ===")
    print(f"  coarse: autocorr peak {coarse_char['autocorr_peak_value']} @ lag "
          f"{coarse_char['autocorr_peak_lag_blocks']} blocks; stage boundary block "
          f"{coarse_char['stage_boundary_block']}")
    print(f"  close-up: within-item {cu_char['within_item_mean_sim']} vs cross-item "
          f"{cu_char['cross_item_mean_sim']}  (n_states={cu_char['n_states']})")
    print("[pre_b] === REFERENCE CLUSTERS (clean should be ~1.0) ===")
    for k, c in clusters.items():
        print(f"  {k:<16} n={c.get('n_samples')} pair_cos_mean={c.get('pairwise_cosine_mean')}")
    print("[pre_b] === PER-AXIS ===")
    ax = region["axes"]
    for name in ("magnitude", "locality", "continuity"):
        dist = ax[name]["distribution"]
        mm = dist.get("multimodal", {}).get("multimodal") if isinstance(dist.get("multimodal"), dict) else None
        print(f"  {name:<13} median={dist.get('median')} iqr={dist.get('iqr')} multimodal={mm}")
    md = ax["manifold_dim"]
    print(f"  manifold_dim  global_D={md['global_D']:.2f} local_D_whole={md['local_D_whole_stream']:.2f}")
    print(f"  repetition    {ax['repetition']['whole_stream']}")
    print(f"  magnitude per-item: {ax['magnitude']['per_item_median']}")
    print("[pre_b] === W4 ===")
    for ax, w in region["w4_detection"].items():
        print(f"  {ax:<13} {w.get('w4_status')}")
    print(f"\n[pre_b] PRE-B complete. any_w4_triggered={region['any_w4_triggered']} "
          f"(W4 is not a STOP trigger; surfaced as Phase 0.5 input)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
