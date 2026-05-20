"""V2-PRE-C unit tests — wrapper import/call-signature + assertion run (instr §7.2)."""

from __future__ import annotations

import torch

from v2.src.preflight import pre_c_arch_assertions_v2_substrate as pre_c


def test_pre_c_wrapper_imports_and_signature():
    # The wrapper exposes run_pre_c / write_report and binds v1's assert_* fns.
    assert callable(pre_c.run_pre_c)
    assert callable(pre_c.write_report)
    from v1.src.preflight.pre_d_arch_property_assertions import (
        assert_ablation1, assert_ablation2, assert_primary,
    )
    assert pre_c.assert_primary is assert_primary
    assert pre_c.assert_ablation1 is assert_ablation1
    assert pre_c.assert_ablation2 is assert_ablation2


def test_pre_c_all_assertions_pass_on_v2_substrate():
    report = pre_c.run_pre_c(device=torch.device("cpu"))
    assert report["substrate_source"] == "v2_synthetic"
    assert report["total_assertions"] == 11        # 4 primary + 4 ablation1 + 3 ablation2
    assert report["all_assertions_passed"] is True
    assert report["v2_substrate_forward_smoke_ok"] is True
    assert report["parameter_counts_match_v1"] is True
    # Per-arm assertion counts: 4 / 4 / 3.
    by_arm = {a["arm"]: len(a["assertions"]) for a in report["arms"]}
    assert by_arm == {"primary": 4, "ablation1": 4, "ablation2": 3}
