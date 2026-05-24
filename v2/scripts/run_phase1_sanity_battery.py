"""V2 Phase 1 — mechanical sanity battery (drift detection, research_ops §15).

The five pilot cross-checks tested methodology *around* classification; none verified the
training+eval *pipe* produces meaningful numbers. A universal null with CV up to 12 is equally
consistent with (a) a real architectural variance limit, or (b) consistently-broken mechanics
all five cross-checks share. This battery distinguishes them. ~5 trained predictors (re-trained;
the pilot discarded weights) + direct prediction-error probes. ~20 min.

Primary probe is PREDICTION ERROR = mean(1 - cos(predicted_mean_k, actual_target_k)) — the
direct "did it learn to predict the trajectory" signal — reported alongside Diff_μ (whose
working-region semantics are subtler). All predictors use the mag=0 mid config (cleanest
base-trajectory fit test): a smooth period-256 loop a working predictor should nail.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from v1.src.config import PREDICT_K, WINDOW_W
from v2.config import RESULTS_ROOT
from v2.src.phase1.eval_perk import compute_diff_metrics_perk, train_one_perk
from v2.src.phase1.sweep_grid import CONTINUITY_CENTER, FIDELITY_F
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams, build_stream

MID = StreamParams(period_P=256, manifold_dim_D=16, continuity_center=CONTINUITY_CENTER[256]["mid"],
                   magnitude_M=0.0, locality_L=0.5, fidelity_F=0.97)
STEPS = 10000


@torch.no_grad()
def pred_error(pred, stream_t, device, max_windows: int = 2000) -> dict:
    """Mean (1 - cos(predicted_mean, target)) over windows × K; also k=1 and k=16."""
    L = stream_t.shape[0]
    ts = list(range(WINDOW_W - 1, L - PREDICT_K))[:max_windows]
    windows = torch.stack([stream_t[t - WINDOW_W + 1: t + 1] for t in ts])
    targets = torch.stack([stream_t[t + 1: t + 1 + PREDICT_K] for t in ts])    # (N,K,d)
    means = []
    for s in range(0, windows.shape[0], 512):
        m, _ = pred(windows[s:s + 512]); means.append(m)
    M = torch.cat(means)
    Mn = M / M.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    Tn = targets / targets.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos = (Mn * Tn).sum(-1)                                                     # (N,K)
    err = (1.0 - cos)
    return {"err_mean": float(err.mean()), "err_k1": float(err[:, 0].mean()),
            "err_k16": float(err[:, -1].mean()), "cos_k1": float(cos[:, 0].mean())}


def _fresh_orthogonal(seed: int, n: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    return (Q * np.sign(np.diag(R))).astype(np.float64)


def _norm_rows(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = load_or_create_U()
    out = {"config": "mag=0 mid (P=256,D=16,center=39); training_steps=10000", "checks": {}}
    print(f"[battery] device={device}; training predictors (mag=0 mid) ...")

    # Train predictors (return model + fine loss log). L_d1 s0/s1, L_d2 s0, L_d4 s0.
    r1, p1, emb1, evalt1 = train_one_perk(MID, 1, 0, U, device, STEPS, "Ld1s0",
                                          loss_log_every=50, return_model=True)
    r1b, p1b, _, _ = train_one_perk(MID, 1, 1, U, device, STEPS, "Ld1s1",
                                    loss_log_every=50, return_model=True)
    r2, p2, _, _ = train_one_perk(MID, 2, 0, U, device, STEPS, "Ld2s0",
                                  loss_log_every=50, return_model=True)
    r4, p4, _, _ = train_one_perk(MID, 4, 0, U, device, STEPS, "Ld4s0",
                                  loss_log_every=50, return_model=True)
    for p in (p1, p1b, p2, p4):
        p.eval()
    print("[battery] predictors trained.")

    # ---- Sanity 1 — predictor on its own training stream vs held-out ----
    pe_train = pred_error(p1, emb1, device)
    pe_held = pred_error(p1, evalt1, device)
    dm_train = compute_diff_metrics_perk(p1, emb1, device)["diff_mu_aggregate"]
    dm_held = compute_diff_metrics_perk(p1, evalt1, device)["diff_mu_aggregate"]
    s1_pass = pe_train["err_k1"] < 0.2          # next-frame should be nailed on training data
    out["checks"]["sanity1_fit_training_stream"] = {
        "pred_error_training": pe_train, "pred_error_heldout": pe_held,
        "diff_mu_training": dm_train, "diff_mu_heldout": dm_held,
        "verdict": "PASS (fits training: next-frame err<0.2)" if s1_pass
                   else "FAIL (does not fit even training stream -> upstream of perturbation)"}

    # ---- Sanity 2 — sensitivity to a fully-different trajectory ----
    U2 = _fresh_orthogonal(seed=999)
    B = build_stream(StreamParams(period_P=200, manifold_dim_D=24, continuity_center=50,
                                  magnitude_M=0.0, locality_L=0.5, fidelity_F=0.97,
                                  n_repetitions=12, seed=4242), U2).stream.astype(np.float32)
    Bt = torch.from_numpy(B).to(device)
    pe_familiar = pred_error(p1, evalt1, device)        # config A held-out (familiar structure)
    pe_novel = pred_error(p1, Bt, device)               # config B (never seen)
    s2_pass = pe_novel["err_mean"] > pe_familiar["err_mean"] + 0.05
    out["checks"]["sanity2_full_trajectory_swap"] = {
        "pred_error_familiar_A": pe_familiar, "pred_error_novel_B": pe_novel,
        "delta_err_mean": pe_novel["err_mean"] - pe_familiar["err_mean"],
        "verdict": "PASS (registers novel trajectory: err_B>err_A)" if s2_pass
                   else "FAIL (insensitive to full-trajectory replacement)"}

    # ---- Sanity 3 — eval metric on analytically-known streams ----
    embA = emb1.cpu().numpy()
    off = 0.5 * _norm_rows(np.random.default_rng(7).standard_normal((1, embA.shape[1])))
    b_offset = torch.from_numpy(_norm_rows(embA[:4096] + off).astype(np.float32)).to(device)
    c_noise = torch.from_numpy(_norm_rows(
        np.random.default_rng(11).standard_normal((4096, embA.shape[1]))).astype(np.float32)).to(device)
    a = {"diff_mu": compute_diff_metrics_perk(p1, emb1, device)["diff_mu_aggregate"],
         "pred_error": pred_error(p1, emb1, device)["err_mean"]}
    b = {"diff_mu": compute_diff_metrics_perk(p1, b_offset, device)["diff_mu_aggregate"],
         "pred_error": pred_error(p1, b_offset, device)["err_mean"]}
    c = {"diff_mu": compute_diff_metrics_perk(p1, c_noise, device)["diff_mu_aggregate"],
         "pred_error": pred_error(p1, c_noise, device)["err_mean"]}
    s3_pass = (c["pred_error"] > a["pred_error"] + 0.1) and (abs(c["diff_mu"] - a["diff_mu"]) > 1e-3)
    out["checks"]["sanity3_metric_known_cases"] = {
        "a_training_exact": a, "b_orthogonal_offset": b, "c_random_noise": c,
        "verdict": "PASS (metric+eval discriminate structured vs noise)" if s3_pass
                   else "FAIL (metric does not discriminate analytically-distinct streams)"}

    # ---- Sanity 4 — loss trajectory (fine) across capacity ----
    def traj(r):
        ll = r["loss_log"]
        return {"initial": ll[0][1] if ll else None, "final": ll[-1][1] if ll else None,
                "min": min(x[1] for x in ll) if ll else None,
                "n_points": len(ll), "decreased": (ll[0][1] - ll[-1][1]) if ll else None,
                "every_2000": [round(x[1], 4) for x in ll[::40]]}
    t1, t2, t4 = traj(r1), traj(r2), traj(r4)
    s4_pass = all(t["decreased"] is not None and t["decreased"] > 0 for t in (t1, t2, t4))
    out["checks"]["sanity4_loss_trajectory"] = {
        "L_d1": t1, "L_d2": t2, "L_d4": t4,
        "capacity_monotone_final": (t1["final"] >= t2["final"] >= t4["final"]),
        "verdict": "PASS (loss decreases at all capacities)" if s4_pass
                   else "FLAG (loss not decreasing -> stuck / not learning)"}

    # ---- Sanity 5 — inter-seed predictor output correlation ----
    L = evalt1.shape[0]
    ts = list(range(WINDOW_W - 1, L - PREDICT_K))[:1000]
    win = torch.stack([evalt1[t - WINDOW_W + 1: t + 1] for t in ts])
    with torch.no_grad():
        m0, _ = p1(win); m1, _ = p1b(win)               # two seeds, same windows
    m0n = m0 / m0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    m1n = m1 / m1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    inter_cos = float((m0n * m1n).sum(-1).mean())        # mean cos between seed-0/seed-1 outputs
    out["checks"]["sanity5_inter_seed_correlation"] = {
        "inter_seed_output_cos": inter_cos,
        "diff_mu_seed0_heldout": dm_held,
        "diff_mu_seed1_heldout": compute_diff_metrics_perk(p1b, evalt1, device)["diff_mu_aggregate"],
        "regime": ("high_corr (>0.8): converging to similar function" if inter_cos > 0.8
                   else "low_corr (<0.3): seeds learn DIFFERENT functions (unviable-egg, concrete)"
                   if inter_cos < 0.3 else "mid_corr: partial convergence")}

    (RESULTS_ROOT / "phase1").mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "phase1" / "sanity_battery.json").write_text(json.dumps(out, indent=2))

    print("\n[battery] === RESULTS ===")
    for k, v in out["checks"].items():
        print(f"  {k}: {v.get('verdict') or v.get('regime')}")
    print(f"\n[battery] S1 fit: train_err_k1={pe_train['err_k1']:.3f} cos_k1={pe_train['cos_k1']:.3f} "
          f"(held err_k1={pe_held['err_k1']:.3f})")
    print(f"[battery] S2 swap: err_familiar={pe_familiar['err_mean']:.3f} err_novel={pe_novel['err_mean']:.3f}")
    print(f"[battery] S3 metric: a_err={a['pred_error']:.3f} c_noise_err={c['pred_error']:.3f} | "
          f"a_dm={a['diff_mu']:.3f} b_dm={b['diff_mu']:.3f} c_dm={c['diff_mu']:.3f}")
    print(f"[battery] S4 loss: L_d1 {t1['initial']:.2f}->{t1['final']:.2f} | "
          f"L_d2 {t2['initial']:.2f}->{t2['final']:.2f} | L_d4 {t4['initial']:.2f}->{t4['final']:.2f}")
    print(f"[battery] S5 inter-seed output cos = {inter_cos:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
