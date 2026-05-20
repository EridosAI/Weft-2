"""Preflight tests for v1 (spec §6.1, §7-§8.3, instr §6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from v1.src.config import EMBED_DIM, WINDOW_W, PREDICT_K
from v1.src.predictor.inner_pam_v1_ablation1 import InnerPAM_v1_Ablation1
from v1.src.predictor.inner_pam_v1_ablation2 import InnerPAM_v1_Ablation2
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.preflight.pre_a_substrate_verification import (
    combined_gap,
    continuous_motion_check,
    cross_element_distinguishability,
    cross_instance_stability,
    embeddings_full_population_check,
)
from v1.src.preflight.pre_b_perturbation_mechanism import (
    CANDIDATE_PRIORITY,
    CandidateMeasurements,
    evaluate_candidate,
    select_candidate,
)
from v1.src.preflight.pre_c_decoder_layer_calibration import (
    CalibrationMetrics,
    compute_smoothness_ratio,
    select_l_d,
)
from v1.src.preflight.pre_d_arch_property_assertions import (
    assert_ablation1,
    assert_ablation2,
    assert_primary,
    write_parameter_counts,
    write_report,
)


DECODER_N_LAYERS = 2


# --------------------------------------------------------------------------
# PRE-A
# --------------------------------------------------------------------------


def test_pre_a_cross_instance_stability_pass():
    embs = {
        1: np.tile(np.eye(EMBED_DIM)[0], (5, 1)).astype(np.float32),  # all identical
        2: np.tile(np.eye(EMBED_DIM)[1], (5, 1)).astype(np.float32),
    }
    r = cross_instance_stability(embs)
    assert r.passed
    assert abs(r.value - 1.0) < 1e-6


def test_pre_a_cross_element_distinguishability_pass():
    embs = {
        1: np.tile(np.eye(EMBED_DIM)[0], (3, 1)).astype(np.float32),
        2: np.tile(np.eye(EMBED_DIM)[1], (3, 1)).astype(np.float32),
        3: np.tile(np.eye(EMBED_DIM)[2], (3, 1)).astype(np.float32),
    }
    r = cross_element_distinguishability(embs)
    assert r.passed
    assert abs(r.value) < 1e-6


def test_pre_a_combined_gap():
    stab = cross_instance_stability(
        {1: np.tile(np.eye(EMBED_DIM)[0], (3, 1)).astype(np.float32)}
    )
    dist = cross_element_distinguishability(
        {
            1: np.tile(np.eye(EMBED_DIM)[0], (3, 1)).astype(np.float32),
            2: np.tile(np.eye(EMBED_DIM)[1], (3, 1)).astype(np.float32),
        }
    )
    gap = combined_gap(stab, dist)
    assert gap.passed


def test_pre_a_continuous_motion_check_pass():
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((100, EMBED_DIM)).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    r = continuous_motion_check(embs)
    assert r.passed


def test_pre_a_continuous_motion_check_fail_on_dwell():
    embs = np.tile(np.eye(EMBED_DIM)[0], (50, 1)).astype(np.float32)
    r = continuous_motion_check(embs)
    assert not r.passed


def test_pre_a_full_population_check_pass():
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((100, EMBED_DIM)).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    r = embeddings_full_population_check(embs)
    assert r.passed


def test_pre_a_full_population_check_fails_on_zero_row():
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((100, EMBED_DIM)).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    embs[42] = 0.0
    r = embeddings_full_population_check(embs)
    assert not r.passed


# --------------------------------------------------------------------------
# PRE-B
# --------------------------------------------------------------------------


def _good_measurements(candidate: str) -> CandidateMeasurements:
    return CandidateMeasurements(
        candidate=candidate,
        perturbed_item_cosine_drops={3: 0.07, 4: 0.08},
        unperturbed_item_cosine_drops={1: 0.001, 2: 0.002, 5: 0.003},
        reproducibility_run1={3: 0.07, 4: 0.08},
        reproducibility_run2={3: 0.071, 4: 0.079},
        eight_finding_checks={f"check_{i}": True for i in range(1, 9)},
        frames_per_loop=360,
        api_success=True,
    )


def test_pre_b_evaluate_candidate_pass():
    m = _good_measurements("per_object_material_setting")
    v = evaluate_candidate(m)
    assert v.overall_pass
    assert v.magnitude_ok and v.locality_ok and v.reproducibility_ok


def test_pre_b_evaluate_candidate_fail_magnitude_low():
    m = _good_measurements("per_object_material_setting")
    m.perturbed_item_cosine_drops = {3: 0.02, 4: 0.03}
    v = evaluate_candidate(m)
    assert not v.magnitude_ok
    assert not v.overall_pass


def test_pre_b_evaluate_candidate_fail_locality():
    m = _good_measurements("per_object_material_setting")
    m.unperturbed_item_cosine_drops = {1: 0.05}  # too high
    v = evaluate_candidate(m)
    assert not v.locality_ok
    assert not v.overall_pass


def test_pre_b_select_priority_first_pass():
    """Priority order respected: per_object_material_setting wins over asset_replacement."""
    verdicts = [
        evaluate_candidate(_good_measurements(c))
        for c in ("asset_replacement", "per_object_material_setting")
    ]
    selected = select_candidate(verdicts)
    assert selected is not None
    assert selected.candidate == "per_object_material_setting"


def test_pre_b_select_none_when_no_pass():
    m = _good_measurements("per_object_material_setting")
    m.api_success = False
    v = evaluate_candidate(m)
    assert select_candidate([v]) is None


# --------------------------------------------------------------------------
# PRE-C
# --------------------------------------------------------------------------


def test_pre_c_smoothness_ratio_low_for_stable_signal():
    # Constant-ish loss trace
    trace = np.full(10_000, 1.0) + np.random.default_rng(0).normal(0, 0.001, 10_000)
    r = compute_smoothness_ratio(trace)
    assert r < 0.5


def test_pre_c_smoothness_ratio_high_for_noisy_signal():
    trace = np.random.default_rng(0).normal(0, 1, 10_000) + 0.01  # very noisy, small mean
    r = compute_smoothness_ratio(trace)
    assert r > 0.5


def test_pre_c_select_l_d_tiebreak_smaller():
    metrics = [
        CalibrationMetrics(1, 0.1, 0.85, 1.0, True, False),
        CalibrationMetrics(2, 0.1, 0.86, 1.0, True, False),
        CalibrationMetrics(3, 0.1, 0.84, 1.0, True, False),  # best by absolute
        CalibrationMetrics(4, 0.1, 0.85, 1.0, True, False),
    ]
    v = select_l_d(metrics)
    # L_d=3 is best by 0.02 band; L_d=1 (0.85), L_d=2 (0.86), L_d=4 (0.85) are
    # within tie-break band of best (0.84). All four within band of best.
    # Tie-breaking favors smaller L_d.
    assert v.selected_decoder_n_layers == 1


def test_pre_c_select_l_d_no_stable():
    metrics = [
        CalibrationMetrics(1, 0.7, 0.85, 1.0, False, False),
        CalibrationMetrics(2, 0.8, 0.86, 1.0, False, False),
    ]
    v = select_l_d(metrics)
    assert v.selected_decoder_n_layers is None
    assert "Escalate" in v.reason


# --------------------------------------------------------------------------
# PRE-D
# --------------------------------------------------------------------------


def test_pre_d_primary_all_assertions_pass():
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    report = assert_primary(model, device=torch.device("cpu"))
    assert report.all_passed(), [
        (a.name, a.detail) for a in report.assertions if not a.passed
    ]


def test_pre_d_ablation1_all_assertions_pass():
    model = InnerPAM_v1_Ablation1(decoder_n_layers=DECODER_N_LAYERS)
    report = assert_ablation1(model, device=torch.device("cpu"))
    assert report.all_passed(), [
        (a.name, a.detail) for a in report.assertions if not a.passed
    ]


def test_pre_d_ablation2_all_assertions_pass():
    model = InnerPAM_v1_Ablation2()
    report = assert_ablation2(model, device=torch.device("cpu"))
    assert report.all_passed(), [
        (a.name, a.detail) for a in report.assertions if not a.passed
    ]


def test_pre_d_report_writing(tmp_path: Path):
    p = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    a1 = InnerPAM_v1_Ablation1(decoder_n_layers=DECODER_N_LAYERS)
    a2 = InnerPAM_v1_Ablation2()
    reports = [
        assert_primary(p, device=torch.device("cpu")),
        assert_ablation1(a1, device=torch.device("cpu")),
        assert_ablation2(a2, device=torch.device("cpu")),
    ]
    all_pass = write_report(reports, tmp_path / "pre_d_report.json")
    write_parameter_counts(reports, tmp_path / "parameter_counts.json")
    assert all_pass
    assert (tmp_path / "pre_d_report.json").exists()
    assert (tmp_path / "parameter_counts.json").exists()
