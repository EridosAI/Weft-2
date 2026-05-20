"""V2-PRE-C — v1 architectural assertions on the v2 synthetic substrate (instr §7.2).

Re-verifies v1 PRE-D's 11 architectural property assertions (4 Primary + 4
Ablation 1 + 3 Ablation 2; spec §§7.2.4 / 7.3.4 / 7.4.4) under v2's stream
construction, and confirms the three arms produce in-contract outputs when fed
v2 synthetic-stream input.

Instruction-vs-reality note (recorded in the output JSON): instr §7.2 references
`run_assertions(predictor, stream)`, but v1 exposes
`assert_{primary,ablation1,ablation2}(model, *, device)`, and these generate
their own random L2-normalised window internally. The 11 assertions are
architecture-level (parameter shapes, autograd-Jacobian structure, source-text
inspection) and therefore input-distribution-independent — they pass identically
on any input. The genuinely input-dependent check is the forward-pass smoke, so
PRE-C additionally runs that smoke on a window drawn from a mid-parameter v2
synthetic stream (PRE-A midpoints), confirming finite, shape- and clamp-contract
outputs on synthetic substrate input. This is a faithful interpretation, not a
spec change (no v1 source is modified; assertions are invoked unchanged).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from v1.src.config import WINDOW_W
from v1.src.predictor.inner_pam_v1_ablation1 import InnerPAM_v1_Ablation1
from v1.src.predictor.inner_pam_v1_ablation2 import InnerPAM_v1_Ablation2
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.preflight.pre_d_arch_property_assertions import (
    _smoke_check,
    assert_ablation1,
    assert_ablation2,
    assert_primary,
)
from v2.config import REPO_ROOT, RESULTS_PRE_C
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams, build_stream

# v1's committed parameter counts at decoder_n_layers=2 (PRE-C placeholder /
# v1 PRE-D configuration), for the cross-check (instr §7.2 STOP trigger 2).
V1_PARAM_COUNTS_PATH = (
    REPO_ROOT / "v1" / "results" / "inner_pam_v1" / "pre_d_arch_assertions"
    / "parameter_counts.json"
)

# Mid-parameter synthetic stream = PRE-A sweep midpoints across all five axes.
MID_PARAMS = StreamParams(
    period_P=256, manifold_dim_D=16, continuity_center=24,
    fidelity_F=0.97, magnitude_M=0.5, locality_L=0.5,
)
L_D_PLACEHOLDER = 2  # matches v1's PRE-D configuration so results are comparable


def _v2_synthetic_window_batch(U: np.ndarray, device: torch.device) -> torch.Tensor:
    """A batch of W-length windows drawn from a mid-parameter v2 synthetic stream."""
    bs = build_stream(MID_PARAMS, U)
    offsets = [0, MID_PARAMS.period_P, 2 * MID_PARAMS.period_P, 3 * MID_PARAMS.period_P]
    windows = np.stack([bs.stream[o : o + WINDOW_W] for o in offsets], axis=0)  # (4, W, d)
    return torch.from_numpy(windows).float().to(device)


def run_pre_c(U: Optional[np.ndarray] = None, device: Optional[torch.device] = None) -> dict:
    """Run the 11 v1 architectural assertions + v2-substrate forward smoke."""
    if U is None:
        U = load_or_create_U()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    primary = InnerPAM_v1_Primary(decoder_n_layers=L_D_PLACEHOLDER).to(device)
    ablation1 = InnerPAM_v1_Ablation1(decoder_n_layers=L_D_PLACEHOLDER).to(device)
    ablation2 = InnerPAM_v1_Ablation2().to(device)

    # (a) v1 architectural assertions (unchanged; internal random window).
    arm_reports = [
        assert_primary(primary, device=device),
        assert_ablation1(ablation1, device=device),
        assert_ablation2(ablation2, device=device),
    ]

    # (b) v2-synthetic-window forward smoke per arm (the input-dependent check).
    window = _v2_synthetic_window_batch(U, device)
    b = window.shape[0]
    v2_smoke = {}
    for name, model in (("primary", primary), ("ablation1", ablation1), ("ablation2", ablation2)):
        model.eval()
        with torch.no_grad():
            mean, log_var = model(window)
        v2_smoke[name] = _smoke_check(mean, log_var, b)

    # (c) parameter-count cross-check vs v1's committed counts (STOP trigger 2).
    v1_counts = json.loads(V1_PARAM_COUNTS_PATH.read_text())
    param_check = {}
    counts_match = True
    for r in arm_reports:
        v1_total = int(v1_counts[r.arm]["total"])
        match = (r.parameter_count == v1_total)
        counts_match = counts_match and match
        param_check[r.arm] = {
            "v2_count": r.parameter_count, "v1_committed_count": v1_total, "match": match,
        }

    all_assertions_pass = all(r.all_passed() for r in arm_reports)
    v2_smoke_ok = all(s["shape_ok"] and s["finite_ok"] and s["clamp_ok"]
                      for s in v2_smoke.values())
    total_assertions = sum(len(r.assertions) for r in arm_reports)

    return {
        "substrate_source": "v2_synthetic",
        "l_d": L_D_PLACEHOLDER,
        "mid_params": vars(MID_PARAMS) | {"perturbed_reps": None},
        "total_assertions": total_assertions,  # 11 = 4 primary + 4 ablation1 + 3 ablation2
        "all_assertions_passed": all_assertions_pass,
        "v2_substrate_forward_smoke_ok": v2_smoke_ok,
        "parameter_counts_match_v1": counts_match,
        "arms": [
            {
                "arm": r.arm,
                "parameter_count": r.parameter_count,
                "source_file": r.source_file,
                "assertions": [asdict(a) for a in r.assertions],
                "random_window_smoke": r.smoke_outputs,
                "v2_synthetic_window_smoke": v2_smoke[r.arm],
            }
            for r in arm_reports
        ],
        "parameter_count_check": param_check,
        "instruction_note": (
            "v1 exposes assert_{primary,ablation1,ablation2}(model, device=...), not "
            "run_assertions(predictor, stream); the 11 assertions are architectural "
            "(input-independent). v2-substrate input is exercised via the "
            "v2_synthetic_window_smoke. §7.2 '33 (11x3)' corrected to 11 total (4+4+3)."
        ),
    }


def write_report(report: dict) -> Path:
    RESULTS_PRE_C.mkdir(parents=True, exist_ok=True)
    path = RESULTS_PRE_C / "arch_assertions_v2_substrate.json"
    path.write_text(json.dumps(report, indent=2))
    return path
