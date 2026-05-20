#!/usr/bin/env python3
"""Run PRE-D architectural property assertions on all three v1 arms.

Spec §7.2.4, §7.3.4, §7.4.4; instr §6.4.

Output:
  results/inner_pam_v1/pre_d_arch_assertions/pre_d_report.json
  results/inner_pam_v1/pre_d_arch_assertions/parameter_counts.json

The script reads PRE-C's selected `decoder_n_layers` from
`results/inner_pam_v1/pre_c_decoder_calibration/selected.json` if present;
otherwise accepts `--decoder-n-layers` to allow running PRE-D ahead of
PRE-C for build-out validation (spec §7.2.1 forbids silent default).

Exit code: 0 if all assertions PASS, 1 otherwise (CC stop condition).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from v1.src.config import PATHS, get_v1_decoder_n_layers
from v1.src.predictor.inner_pam_v1_ablation1 import InnerPAM_v1_Ablation1
from v1.src.predictor.inner_pam_v1_ablation2 import InnerPAM_v1_Ablation2
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.preflight.pre_d_arch_property_assertions import (
    assert_ablation1,
    assert_ablation2,
    assert_primary,
    compute_l_d_envelope,
    write_l_d_envelope,
    write_parameter_counts,
    write_report,
)


def main() -> int:
    p = argparse.ArgumentParser(description="PRE-D architectural property assertions")
    p.add_argument(
        "--decoder-n-layers",
        type=int,
        default=None,
        help=(
            "Override PRE-C-selected decoder_n_layers (spec §7.2.1 requires "
            "explicit value; use this when PRE-C has not yet run)."
        ),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    device = torch.device(args.device)

    # Resolve decoder_n_layers.
    if args.decoder_n_layers is not None:
        decoder_n_layers = args.decoder_n_layers
        source = f"--decoder-n-layers={decoder_n_layers}"
    else:
        decoder_n_layers = get_v1_decoder_n_layers()
        source = "PRE-C lock file"
    print(f"[pre_d] decoder_n_layers={decoder_n_layers} (source: {source})")

    # Construct all three arms.
    torch.manual_seed(0)
    primary = InnerPAM_v1_Primary(decoder_n_layers=decoder_n_layers).to(device)
    ablation1 = InnerPAM_v1_Ablation1(decoder_n_layers=decoder_n_layers).to(device)
    ablation2 = InnerPAM_v1_Ablation2().to(device)

    print("[pre_d] running assertions...")
    reports = [
        assert_primary(primary, device=device),
        assert_ablation1(ablation1, device=device),
        assert_ablation2(ablation2, device=device),
    ]

    report_path = PATHS.results_pre_d / "pre_d_report.json"
    pcount_path = PATHS.results_pre_d / "parameter_counts.json"
    envelope_path = PATHS.results_pre_d / "parameter_counts_l_d_envelope.json"
    all_pass = write_report(reports, report_path)
    write_parameter_counts(reports, pcount_path)

    # L_d capacity envelope (per reviewer-chat request, 2026-05-19): pre-compute
    # Primary + Ablation 1 parameter breakdowns at each PRE-C candidate L_d
    # so the realistic capacity range is visible alongside PRE-D rather
    # than after PRE-C completes.
    print("[pre_d] computing L_d capacity envelope across PRE_C_L_D_CANDIDATES...")
    from v1.src.config import PRE_C_L_D_CANDIDATES
    envelope = compute_l_d_envelope(tuple(PRE_C_L_D_CANDIDATES))
    write_l_d_envelope(envelope, envelope_path)

    print(f"[pre_d] report written: {report_path}")
    print(f"[pre_d] parameter counts: {pcount_path}")
    print(f"[pre_d] L_d envelope: {envelope_path}")
    for r in reports:
        print(f"[pre_d]   {r.arm}: {trainable_summary(r)}")
        for a in r.assertions:
            mark = "PASS" if a.passed else "FAIL"
            print(f"[pre_d]     [{mark}] {a.name}: {a.detail}")
    print(f"[pre_d] all assertions passed: {all_pass}")
    return 0 if all_pass else 1


def trainable_summary(report) -> str:
    return f"params={report.parameter_count:,}, source={report.source_file}"


if __name__ == "__main__":
    sys.exit(main())
