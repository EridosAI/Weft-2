"""Controls and the shuffle-sanity check (instr §6.3, §6.5).

C1 — Pure cosine baseline. For each probe: L2-normalised mean of the window
embeddings is queried against the bank; the cosine to the actual y_1 (first
target) gives a baseline "is the most-similar bank entry close to the next
frame?" number.

C2 — Shuffle predictor lives in `src/trainer/online_trainer.py` (just a
seeded permutation of the training-step order). This module computes the
S1-S4 sanity-check diagnostics over its outputs.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from v0.src.config import (
    S4_MEAN_LOG_VAR_MIN,
    S4_MEAN_NORM_MAX,
    SEED_PROBE_SAMPLING,
)
from v0.src.eval.metrics import (
    ProbeResult,
    _normalize_rows,
    m1_centreline_accuracy,
    m3_cluster_sharpness,
)
from v0.src.eval.probes import Probe
from v0.src.memory.memory_bank import MemoryBank


# ---------- C1: cosine baseline --------------------------------------------


def cosine_baseline_top1(
    probes: Sequence[Probe],
    embeddings: np.ndarray,        # (N, d) — full stream
    bank: MemoryBank,
) -> dict[str, Any]:
    """For each probe, top-1 bank cosine vs the actual y_1.

    "Close to y_1" is measured by cosine(retrieved_embedding, y_1). The baseline
    cannot produce a centreline path; what it can do is say "given the current
    window, here is the most similar past frame, is that frame near where the
    actual trajectory is heading?"
    """
    cos_to_target: list[float] = []
    per_probe: list[dict[str, Any]] = []
    for p in probes:
        window = embeddings[p.window_start : p.window_end + 1]
        win_mean = window.mean(axis=0)
        win_mean = win_mean / max(float(np.linalg.norm(win_mean)), 1e-12)
        cosines, indices = bank.retrieve_by_cosine(
            win_mean.astype(np.float32), k=1
        )
        top_idx = int(indices[0, 0])
        if top_idx < 0:
            continue
        retrieved = bank.vectors[top_idx]
        y1 = embeddings[p.target_start]
        retrieved = retrieved / max(float(np.linalg.norm(retrieved)), 1e-12)
        y1n = y1 / max(float(np.linalg.norm(y1)), 1e-12)
        c = float(np.dot(retrieved, y1n))
        cos_to_target.append(c)
        per_probe.append(
            {
                "probe_type": p.probe_type,
                "from_item": p.from_item,
                "to_item": p.to_item,
                "loop_idx": p.loop_idx,
                "retrieved_bank_idx": int(top_idx),
                "cosine_retrieved_vs_y1": c,
            }
        )
    arr = np.asarray(cos_to_target, dtype=np.float64)
    return {
        "n_probes": int(arr.size),
        "mean_cosine_top1_vs_y1": float(arr.mean()) if arr.size else float("nan"),
        "std_cosine_top1_vs_y1": float(arr.std(ddof=0)) if arr.size else float("nan"),
        "per_probe": per_probe,
    }


# ---------- S1-S4: shuffle sanity check ------------------------------------


def s1_log_var_distribution(shuffle_results: Sequence[ProbeResult]) -> dict[str, float]:
    vals = np.concatenate([r.pred_log_var for r in shuffle_results])
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "median": float(np.median(vals)),
    }


def s2_m1_distribution(shuffle_results: Sequence[ProbeResult]) -> dict[str, float]:
    rows: list[np.ndarray] = []
    for r in shuffle_results:
        mu = _normalize_rows(r.pred_mean)
        y = _normalize_rows(r.target)
        rows.append((mu * y).sum(axis=-1))
    arr = np.stack(rows, axis=0) if rows else np.zeros((0, 0))
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n_probes": int(arr.shape[0]),
        "mean_aggregate": float(arr.mean()),
        "std_aggregate": float(arr.std(ddof=0)),
        "mean_per_k": arr.mean(axis=0).tolist(),
        "std_per_k": arr.std(axis=0, ddof=0).tolist(),
    }


def s3_cluster_sharpness(shuffle_results: Sequence[ProbeResult]) -> dict[str, Any]:
    return m3_cluster_sharpness(shuffle_results)


def s4_quantitative_collapse_check(
    shuffle_results: Sequence[ProbeResult],
    seed: int = SEED_PROBE_SAMPLING,
) -> dict[str, Any]:
    """Quantitative collapse-to-mean check (instr §6.5).

    Reads: collapse if mean(||mu_shuffle||_2) < S4_MEAN_NORM_MAX
                AND mean(log sigma^2_shuffle) > S4_MEAN_LOG_VAR_MIN.
    Otherwise reads non-collapse; verdict from S1-S3 + qualitative S4 sample.
    """
    if not shuffle_results:
        return {
            "n_probes": 0,
            "mean_pred_norm": float("nan"),
            "mean_pred_log_var": float("nan"),
            "verdict": "no_data",
        }
    # ||mu||_2 averaged across all (probe, step) (instr §6.5 quantitative).
    norms = np.concatenate([
        np.linalg.norm(r.pred_mean, axis=-1) for r in shuffle_results
    ])
    log_vars = np.concatenate([r.pred_log_var for r in shuffle_results])
    mean_norm = float(norms.mean())
    mean_log_var = float(log_vars.mean())
    collapse = (mean_norm < S4_MEAN_NORM_MAX) and (mean_log_var > S4_MEAN_LOG_VAR_MIN)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(
        len(shuffle_results), size=min(5, len(shuffle_results)), replace=False
    )
    qualitative_samples = []
    for i in sample_indices:
        r = shuffle_results[int(i)]
        qualitative_samples.append(
            {
                "probe_type": r.probe.probe_type,
                "from_item": r.probe.from_item,
                "pred_norm_per_k": np.linalg.norm(r.pred_mean, axis=-1).tolist(),
                "pred_log_var_per_k": r.pred_log_var.tolist(),
            }
        )
    return {
        "n_probes": int(len(shuffle_results)),
        "mean_pred_norm": mean_norm,
        "mean_pred_log_var": mean_log_var,
        "threshold_norm_max": S4_MEAN_NORM_MAX,
        "threshold_log_var_min": S4_MEAN_LOG_VAR_MIN,
        "quantitative_verdict": (
            "collapse_to_mean_expected" if collapse else "non_collapse"
        ),
        "qualitative_samples": qualitative_samples,
    }


def shuffle_sanity_check(
    main_results: Sequence[ProbeResult],
    shuffle_results: Sequence[ProbeResult],
) -> dict[str, Any]:
    """Run S1-S4; final verdict 'expected' if all four agree, else 'unexpected'."""
    s1 = s1_log_var_distribution(shuffle_results)
    s2 = s2_m1_distribution(shuffle_results)
    s3 = s3_cluster_sharpness(shuffle_results)
    s4 = s4_quantitative_collapse_check(shuffle_results)
    main_s1 = s1_log_var_distribution(main_results)
    main_s2 = s2_m1_distribution(main_results)
    main_s3 = s3_cluster_sharpness(main_results)

    expected_s1 = float(s1["mean"]) > float(main_s1["mean"])     # shuffle less confident
    expected_s2 = float(s2["mean_aggregate"]) < float(main_s2["mean_aggregate"])
    expected_s3 = abs(float(s3["cluster_sharpness"])) < abs(
        float(main_s3["cluster_sharpness"])
    )
    expected_s4 = s4["quantitative_verdict"] == "collapse_to_mean_expected"
    all_expected = expected_s1 and expected_s2 and expected_s3 and expected_s4
    verdict = "expected" if all_expected else "unexpected"
    return {
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "S4": s4,
        "individual_expected": {
            "S1_shuffle_less_confident": bool(expected_s1),
            "S2_shuffle_lower_m1": bool(expected_s2),
            "S3_shuffle_lower_sharpness": bool(expected_s3),
            "S4_quantitative_collapse": bool(expected_s4),
        },
        "verdict": verdict,
    }
