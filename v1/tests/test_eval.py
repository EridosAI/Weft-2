"""Evaluation tests for v1 (spec §10, instr §5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from v1.src.config import EMBED_DIM, PREDICT_K, WINDOW_W
from v1.src.eval.arm_comparison_matrix import ArmRecords, build_matrix, write_matrix_csv, write_matrix_json
from v1.src.eval.per_item_ordinal_metrics import (
    build_canonical_pairs,
    compute_body_repr_drift,
    compute_drift_metrics,
    cosine_distance,
    evaluate_per_item_ordinal,
    write_per_item_ordinal_json,
)
from v1.src.eval.threshold_calibration import (
    calibrate_thresholds,
    classify_cells,
)
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary


DECODER_N_LAYERS = 2


def _make_annotations(n: int, items: int = 3, ordinals: int = 3) -> list[dict]:
    """Build a fake annotation stream with `items` items × `ordinals` close-up
    ordinals each. Each (item, ordinal) appears once, spaced so canonical
    windows fit in the stream.
    """
    annotations = []
    # Reserve the first 2 * (W + K) frames for the windows to fit.
    base = WINDOW_W + PREDICT_K + 5
    idx = 0
    for f in range(n):
        ann = {"frame_index": f, "loop_index": f // 50, "phase": "transit"}
        annotations.append(ann)
    # Drop in canonical (item, ordinal) entries at spaced positions.
    positions = []
    for i, item in enumerate(range(1, items + 1)):
        for j, ordinal in enumerate(range(1, ordinals + 1)):
            pos = base + (i * ordinals + j) * (WINDOW_W + PREDICT_K + 2)
            if pos >= n:
                continue
            annotations[pos] = {
                "frame_index": pos,
                "loop_index": pos // 50,
                "phase": "close_up",
                "viewing_position_id": item,
                "close_up_ordinal": ordinal,
            }
            positions.append(pos)
    return annotations


def test_build_canonical_pairs_basic():
    n = 500
    ann = _make_annotations(n, items=3, ordinals=3)
    pairs = build_canonical_pairs(ann)
    # 3 × 3 = 9 pairs expected
    assert len(pairs) == 9
    for p in pairs:
        assert p.window_end + 1 - p.window_start == WINDOW_W
        assert p.target_index - p.window_end == PREDICT_K


def test_evaluate_per_item_ordinal_smoke():
    n = 500
    ann = _make_annotations(n)
    pairs = build_canonical_pairs(ann)
    rng = np.random.default_rng(0)
    embeds = rng.standard_normal((n, EMBED_DIM)).astype(np.float32)
    embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    records = evaluate_per_item_ordinal(model, embeds, pairs, device=torch.device("cpu"))
    assert len(records) == len(pairs)
    for r in records:
        assert len(r["mean"]) == EMBED_DIM
        assert len(r["per_k_mean"]) == PREDICT_K
        assert len(r["per_k_log_var"]) == PREDICT_K
        assert len(r["per_k_loss"]) == PREDICT_K


def test_compute_drift_metrics_finite():
    """Drift metrics between two different model states should be finite."""
    n = 500
    ann = _make_annotations(n)
    pairs = build_canonical_pairs(ann)
    rng = np.random.default_rng(0)
    embeds = rng.standard_normal((n, EMBED_DIM)).astype(np.float32)
    embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    m_a = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    torch.manual_seed(99)
    m_b = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    ra = evaluate_per_item_ordinal(m_a, embeds, pairs, device=torch.device("cpu"))
    rb = evaluate_per_item_ordinal(m_b, embeds, pairs, device=torch.device("cpu"))
    drift = compute_drift_metrics(ra, rb)
    assert len(drift) == len(pairs)
    for d in drift:
        assert np.isfinite(d["mean_drift"])
        assert np.isfinite(d["variance_drift"])
        assert len(d["per_k_mean_drift"]) == PREDICT_K


def test_cosine_distance_identity():
    v = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_distance(v, v)) < 1e-9


def test_threshold_calibration_smoke():
    """End-to-end: build matrix from fake drift records, calibrate, classify."""
    arm1 = ArmRecords(
        arm="primary",
        drift_records=[
            {"item": 1, "ordinal": 1, "mean_drift": 0.4, "variance_drift": -0.4,
             "per_k_mean_drift": [0.4] * PREDICT_K, "per_k_variance_drift": [-0.4] * PREDICT_K},
            {"item": 2, "ordinal": 1, "mean_drift": 0.01, "variance_drift": -0.05,
             "per_k_mean_drift": [0.01] * PREDICT_K, "per_k_variance_drift": [-0.05] * PREDICT_K},
        ],
        body_repr_records=[
            {"item": 1, "ordinal": 1, "body_repr_cosine": 0.8},
            {"item": 2, "ordinal": 1, "body_repr_cosine": 0.95},
        ],
    )
    arm2 = ArmRecords(
        arm="ablation2",
        drift_records=[
            {"item": 1, "ordinal": 1, "mean_drift": 0.4, "variance_drift": -0.4,
             "per_k_mean_drift": [0.4] * PREDICT_K, "per_k_variance_drift": [-0.4] * PREDICT_K},
            {"item": 2, "ordinal": 1, "mean_drift": 0.4, "variance_drift": -0.4,
             "per_k_mean_drift": [0.4] * PREDICT_K, "per_k_variance_drift": [-0.4] * PREDICT_K},
        ],
        body_repr_records=[
            {"item": 1, "ordinal": 1, "body_repr_cosine": 0.7},
            {"item": 2, "ordinal": 1, "body_repr_cosine": 0.75},
        ],
    )
    matrix = build_matrix([arm1, arm2], bit_identical_pairs=[(2, 1)])
    assert matrix["arms"] == ["primary", "ablation2"]
    thresholds = calibrate_thresholds(matrix)
    classified = classify_cells(matrix, thresholds)
    # Row for item=2 ordinal=1 is bit-identical. Primary's mean_drift=0.01
    # should be 'stable'. Ablation 2's mean_drift=0.4 (at a bit-identical
    # pair) should NOT be classified as 'coupling' (coupling only applies to
    # input-varying pairs); it will be 'differentiated' or 'indeterminate'.
    bit_id_row = next(r for r in classified["rows"] if r["item"] == 2 and r["ordinal"] == 1)
    assert bit_id_row["bit_identical"] is True
    assert bit_id_row["cells"]["primary"]["classifications"]["mean_drift"] == "stable"


def test_matrix_csv_round_trip(tmp_path: Path):
    """CSV writer doesn't raise on a non-trivial matrix."""
    arm = ArmRecords(
        arm="primary",
        drift_records=[
            {"item": 1, "ordinal": 1, "mean_drift": 0.4, "variance_drift": -0.4,
             "per_k_mean_drift": [0.4] * PREDICT_K, "per_k_variance_drift": [-0.4] * PREDICT_K},
        ],
        body_repr_records=[
            {"item": 1, "ordinal": 1, "body_repr_cosine": 0.8},
        ],
    )
    matrix = build_matrix([arm], bit_identical_pairs=[])
    write_matrix_csv(matrix, tmp_path / "m.csv")
    write_matrix_json(matrix, tmp_path / "m.json")
    csv_text = (tmp_path / "m.csv").read_text()
    assert "primary_mean_drift" in csv_text
