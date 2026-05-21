"""V2 Phase 1 — single arm-run.

A thin, faithful wrapper over the proven PRE-D1a recipe (`train_one`: v1 Primary +
v1 path_prediction_loss + the trainer's per-step contract; per-stream-point Diff_μ /
Diff_σ on a held-out eval stream). No new training mechanism.

Two entry points:
  * `run_arm(...)`  — in-process (used by the smoke + aggregate paths).
  * CLI `--spec-file` — one arm-run as an independent subprocess, so the parallel
    harness (§7.7) can dispatch 2x/3x concurrent runs on the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

# CUDA determinism (instr §1) — before torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from v2.config import get_v2_training_steps
from v2.src.preflight.pre_d1a_endpoint_stability import train_one
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams

# The construction fields a cell spec carries (StreamParams; n_repetitions/seed are
# set by the runner / train_one internally).
_PARAM_KEYS = ("period_P", "manifold_dim_D", "continuity_center",
               "fidelity_F", "magnitude_M", "locality_L")


def params_from_dict(d: dict) -> StreamParams:
    return StreamParams(**{k: d[k] for k in _PARAM_KEYS if k in d})


def run_arm(params: StreamParams, L_d_main: int, seed: int, *,
            training_steps: int | None = None, U=None, device=None,
            label: str = "", axis: str = "phase1", endpoint: str = "cell"):
    """Train one Primary arm at (params, L_d_main, seed); return PRE-D1a ArmRunResult."""
    if training_steps is None:
        training_steps = get_v2_training_steps()
    if U is None:
        U = load_or_create_U()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train_one(params, L_d_main, seed, U, device, training_steps,
                     label, axis, endpoint)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one Phase 1 arm-run from a spec file.")
    ap.add_argument("--spec-file", required=True,
                    help="JSON: {params:{...}, L_d_main, seed, training_steps?, out_file, label?}")
    args = ap.parse_args()
    spec = json.loads(open(args.spec_file).read())

    params = params_from_dict(spec["params"])
    r = run_arm(params, int(spec["L_d_main"]), int(spec["seed"]),
                training_steps=spec.get("training_steps"),
                label=spec.get("label", ""),
                axis=spec.get("axis", "phase1"),
                endpoint=spec.get("endpoint", "cell"))

    res = asdict(r)
    res["spec"] = spec
    with open(spec["out_file"], "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[arm] {spec.get('label', '')} seed={spec['seed']} L_d={spec['L_d_main']} "
          f"diff_mu={r.diff_mu} diff_sigma={r.diff_sigma} flag={r.stability_flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
