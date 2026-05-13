"""M1-M7 metrics for v0 evaluation (instr §6.2).

All metrics consume the per-probe records produced by `run_probes_through_predictor`
below: for each probe we record the predicted (mean, log_var) and the actual K-step
target embeddings, plus the probe's metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.config import EMBED_DIM, PREDICT_K
from src.eval.probes import Probe
from src.predictor.inner_pam import InnerPAM


# ---------- per-probe runner ------------------------------------------------


@dataclass
class ProbeResult:
    probe: Probe
    pred_mean: np.ndarray        # (K, d)
    pred_log_var: np.ndarray     # (K,)
    target: np.ndarray           # (K, d)


def run_probes_through_predictor(
    probes: Sequence[Probe],
    embeddings: np.ndarray,        # (N, d) — full stream, source of windows + targets
    predictor: InnerPAM,
    device: torch.device,
    batch_size: int = 32,
) -> list[ProbeResult]:
    if not probes:
        return []
    results: list[ProbeResult] = []
    predictor.eval()
    with torch.no_grad():
        for i in range(0, len(probes), batch_size):
            chunk = probes[i : i + batch_size]
            windows = np.stack(
                [embeddings[p.window_start : p.window_end + 1] for p in chunk],
                axis=0,
            )
            targets = np.stack(
                [embeddings[p.target_start : p.target_start + PREDICT_K] for p in chunk],
                axis=0,
            )
            w = torch.from_numpy(windows).to(device)
            mean, log_var = predictor(w)
            mean_np = mean.detach().cpu().numpy()
            log_var_np = log_var.detach().cpu().numpy()
            for j, p in enumerate(chunk):
                results.append(
                    ProbeResult(
                        probe=p,
                        pred_mean=mean_np[j],
                        pred_log_var=log_var_np[j],
                        target=targets[j],
                    )
                )
    return results


# ---------- helpers --------------------------------------------------------


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _cluster_key(p: Probe) -> tuple[str, int]:
    return (p.probe_type, p.from_item)


# ---------- M1: centreline accuracy per step -------------------------------


def m1_centreline_accuracy(results: Sequence[ProbeResult]) -> dict[str, Any]:
    """Cosine(mu_k, y_k) per step, disaggregated by (probe_type, from_item)."""
    out: dict[tuple[str, int], list[np.ndarray]] = {}
    aggregate_per_k: list[list[float]] = [[] for _ in range(PREDICT_K)]
    for r in results:
        mu = _normalize_rows(r.pred_mean)
        y = _normalize_rows(r.target)
        per_k_cos = (mu * y).sum(axis=-1)  # (K,)
        key = _cluster_key(r.probe)
        out.setdefault(key, []).append(per_k_cos)
        for k, c in enumerate(per_k_cos):
            aggregate_per_k[k].append(float(c))
    per_cluster_stats: dict[str, dict[str, list[float]]] = {}
    for (probe_type, item), rows in out.items():
        arr = np.stack(rows, axis=0)
        per_cluster_stats[f"{probe_type}:item_{item}"] = {
            "mean_per_k": arr.mean(axis=0).tolist(),
            "std_per_k": arr.std(axis=0, ddof=0).tolist(),
            "n": int(arr.shape[0]),
        }
    aggregate_stats = {
        "mean_per_k": [float(np.mean(v)) for v in aggregate_per_k],
        "std_per_k": [float(np.std(v, ddof=0)) for v in aggregate_per_k],
        "n_per_k": [int(len(v)) for v in aggregate_per_k],
    }
    return {"per_cluster": per_cluster_stats, "aggregate": aggregate_stats}


def m1_per_probe_centreline_at_k(
    results: Sequence[ProbeResult], k: int
) -> tuple[np.ndarray, list[Probe]]:
    """Cosines at step k (1-indexed in the spec, 0-indexed here); returns paired with probes."""
    assert 0 <= k < PREDICT_K
    cos_list: list[float] = []
    probes: list[Probe] = []
    for r in results:
        mu = r.pred_mean[k]
        y = r.target[k]
        mu_n = mu / max(float(np.linalg.norm(mu)), 1e-12)
        y_n = y / max(float(np.linalg.norm(y)), 1e-12)
        cos_list.append(float(np.dot(mu_n, y_n)))
        probes.append(r.probe)
    return np.asarray(cos_list, dtype=np.float64), probes


# ---------- M2: variance calibration ---------------------------------------


def m2_variance_calibration(results: Sequence[ProbeResult]) -> dict[str, Any]:
    """Per-step (squared_error/d) / sigma^2; ~1.0 if calibrated."""
    ratios_per_k: list[list[float]] = [[] for _ in range(PREDICT_K)]
    sq_err_per_k: list[list[float]] = [[] for _ in range(PREDICT_K)]
    sigma2_per_k: list[list[float]] = [[] for _ in range(PREDICT_K)]
    for r in results:
        diff = r.target - r.pred_mean
        sq = (diff * diff).sum(axis=-1) / EMBED_DIM  # (K,)
        sigma2 = np.exp(r.pred_log_var)               # (K,)
        ratio = sq / np.maximum(sigma2, 1e-12)
        for k in range(PREDICT_K):
            sq_err_per_k[k].append(float(sq[k]))
            sigma2_per_k[k].append(float(sigma2[k]))
            ratios_per_k[k].append(float(ratio[k]))
    return {
        "mean_ratio_per_k": [float(np.mean(v)) for v in ratios_per_k],
        "median_ratio_per_k": [float(np.median(v)) for v in ratios_per_k],
        "mean_sq_err_normed_per_k": [float(np.mean(v)) for v in sq_err_per_k],
        "mean_sigma2_per_k": [float(np.mean(v)) for v in sigma2_per_k],
        "n_per_k": [int(len(v)) for v in ratios_per_k],
    }


# ---------- M3: shape clustering -------------------------------------------


def m3_cluster_sharpness(results: Sequence[ProbeResult]) -> dict[str, Any]:
    """Within-cluster vs cross-cluster cosine on predicted means (k=0 / first step)."""
    by_cluster: dict[tuple[str, int], list[np.ndarray]] = {}
    for r in results:
        key = _cluster_key(r.probe)
        # Use the full K-step predicted-mean trajectory, flattened, as the cluster vector.
        v = r.pred_mean.reshape(-1)
        by_cluster.setdefault(key, []).append(v)

    within_per_cluster: dict[str, float] = {}
    within_vals: list[float] = []
    for key, vecs_list in by_cluster.items():
        if len(vecs_list) < 2:
            continue
        vecs = _normalize_rows(np.stack(vecs_list, axis=0))
        sim = vecs @ vecs.T
        n = sim.shape[0]
        iu = np.triu_indices(n, k=1)
        within = sim[iu]
        within_per_cluster[f"{key[0]}:item_{key[1]}"] = float(within.mean())
        within_vals.extend(within.tolist())

    cross_vals: list[float] = []
    cluster_keys = list(by_cluster.keys())
    for i in range(len(cluster_keys)):
        for j in range(i + 1, len(cluster_keys)):
            a_vecs = _normalize_rows(np.stack(by_cluster[cluster_keys[i]], axis=0))
            b_vecs = _normalize_rows(np.stack(by_cluster[cluster_keys[j]], axis=0))
            sim = a_vecs @ b_vecs.T
            cross_vals.extend(sim.flatten().tolist())

    within_mean = float(np.mean(within_vals)) if within_vals else 0.0
    cross_mean = float(np.mean(cross_vals)) if cross_vals else 0.0
    sharpness = within_mean - cross_mean

    # Predictor variance at steady-state probes (spec §10.4).
    steady_log_var: list[float] = []
    for r in results:
        if r.probe.probe_type == "steady":
            steady_log_var.append(float(r.pred_log_var.mean()))
    return {
        "within_cluster_cosine_mean": within_mean,
        "cross_cluster_cosine_mean": cross_mean,
        "cluster_sharpness": float(sharpness),
        "within_per_cluster": within_per_cluster,
        "steady_state_mean_log_var": (
            float(np.mean(steady_log_var)) if steady_log_var else float("nan")
        ),
        "n_clusters": int(len([k for k in by_cluster if len(by_cluster[k]) >= 2])),
    }


# ---------- M4: repetition-stratified accuracy -----------------------------


REP_BINS: tuple[tuple[int, int], ...] = (
    (1, 5), (6, 19), (20, 50), (51, 99), (100, 10_000_000),
)


def _bin_for_count(c: int) -> Optional[str]:
    for lo, hi in REP_BINS:
        if lo <= c <= hi:
            return f"{lo}_to_{hi if hi < 10_000_000 else 'inf'}"
    return None


def m4_repetition_stratified_accuracy(
    results: Sequence[ProbeResult],
    rep_counts: dict[tuple[str, int], int],
) -> dict[str, Any]:
    """M1 per bin of (probe_type, from_item) repetition count."""
    by_bin: dict[str, list[np.ndarray]] = {}
    for r in results:
        key = _cluster_key(r.probe)
        c = int(rep_counts.get(key, 0))
        b = _bin_for_count(c)
        if b is None:
            continue
        mu = _normalize_rows(r.pred_mean)
        y = _normalize_rows(r.target)
        cos = (mu * y).sum(axis=-1)
        by_bin.setdefault(b, []).append(cos)
    out: dict[str, dict[str, Any]] = {}
    for b, rows in by_bin.items():
        arr = np.stack(rows, axis=0)
        out[b] = {
            "n_probes": int(arr.shape[0]),
            "mean_per_k": arr.mean(axis=0).tolist(),
            "std_per_k": arr.std(axis=0, ddof=0).tolist(),
        }
    return out


# ---------- M5: bank-vs-weights recall fraction ----------------------------


def m5_recall_fractions(modes: Sequence[str]) -> dict[str, float]:
    if not modes:
        return {"predictor_only_fraction": 0.0, "predictor_plus_bank_fraction": 0.0}
    n = len(modes)
    n_only = sum(1 for m in modes if m == "predictor_only")
    return {
        "predictor_only_fraction": float(n_only / n),
        "predictor_plus_bank_fraction": float((n - n_only) / n),
        "n_probes": int(n),
    }


# ---------- M6 / M7 ---------------------------------------------------------


def m6_cluster_accommodation(
    results_perturbed: Sequence[ProbeResult],
    results_original: Sequence[ProbeResult],
) -> dict[str, Any]:
    """Within-original, within-perturbed, cross-cluster predicted-mean cosines per item."""
    out: dict[str, Any] = {}
    for item in sorted({r.probe.from_item for r in results_perturbed}):
        per = [r for r in results_perturbed if r.probe.from_item == item]
        orig = [r for r in results_original if r.probe.from_item == item]
        if not per or not orig:
            continue
        per_vecs = _normalize_rows(
            np.stack([r.pred_mean.reshape(-1) for r in per], axis=0)
        )
        orig_vecs = _normalize_rows(
            np.stack([r.pred_mean.reshape(-1) for r in orig], axis=0)
        )
        within_per = (per_vecs @ per_vecs.T)
        within_per = within_per[np.triu_indices(within_per.shape[0], k=1)]
        within_orig = (orig_vecs @ orig_vecs.T)
        within_orig = within_orig[np.triu_indices(within_orig.shape[0], k=1)]
        cross = (per_vecs @ orig_vecs.T).flatten()
        out[f"item_{item}"] = {
            "within_original_mean": float(within_orig.mean()) if within_orig.size else float("nan"),
            "within_perturbed_mean": float(within_per.mean()) if within_per.size else float("nan"),
            "cross_cluster_mean": float(cross.mean()) if cross.size else float("nan"),
            "n_perturbed": int(len(per)),
            "n_original": int(len(orig)),
        }
    if out:
        out["aggregate"] = {
            "within_original_mean": float(
                np.mean([v["within_original_mean"] for v in out.values()])
            ),
            "within_perturbed_mean": float(
                np.mean([v["within_perturbed_mean"] for v in out.values()])
            ),
            "cross_cluster_mean": float(
                np.mean([v["cross_cluster_mean"] for v in out.values()])
            ),
        }
    return out


def m7_compounding_accommodation(
    results_phase1: Sequence[ProbeResult],
    results_phase2: Sequence[ProbeResult],
    results_phase3: Sequence[ProbeResult],
) -> dict[str, Any]:
    """Cluster sharpness in each phase plus cross-phase predicted-mean cosines per item."""
    out: dict[str, Any] = {
        "phase1_sharpness": m3_cluster_sharpness(results_phase1)["cluster_sharpness"],
        "phase2_sharpness": m3_cluster_sharpness(results_phase2)["cluster_sharpness"],
        "phase3_sharpness": m3_cluster_sharpness(results_phase3)["cluster_sharpness"],
    }
    cross: dict[str, dict[str, float]] = {}
    by_phase = {"phase1": results_phase1, "phase2": results_phase2, "phase3": results_phase3}
    for item in sorted({r.probe.from_item for r in results_phase3}):
        vecs_by_phase: dict[str, np.ndarray] = {}
        for ph_name, ph_results in by_phase.items():
            v = [r.pred_mean.reshape(-1) for r in ph_results if r.probe.from_item == item]
            if v:
                vecs_by_phase[ph_name] = _normalize_rows(np.stack(v, axis=0))
        per_item: dict[str, float] = {}
        for a in vecs_by_phase:
            for b in vecs_by_phase:
                if a >= b:
                    continue
                sim = vecs_by_phase[a] @ vecs_by_phase[b].T
                per_item[f"{a}_vs_{b}_mean"] = float(sim.mean())
        cross[f"item_{item}"] = per_item
    out["cross_phase_per_item"] = cross
    return out
