"""Threshold calibration procedure (spec §10.4.3, instr §5.4 / §9.2).

Maps the per-(item, ordinal) × arm matrix to verdict-pattern classifications
via empirically-anchored percentile thresholds.

Procedure (instr §5.4):
  1. Build stability reference distribution: metric values at bit-identical
     pairs across all three arms.
  2. Build differentiation reference distribution: metric values at
     input-varying pairs across all three arms.
  3. Anchor thresholds at 75th-percentile-of-stability /
     25th-percentile-of-differentiation per spec §10.4.3.
  4. Classify each (item, ordinal) × arm × metric cell into:
     - "stable"           (below stability threshold)
     - "differentiated"   (above differentiation threshold)
     - "coupling"         (input-varying pair sitting near stability range)
     - "indeterminate"    (otherwise)

The 75/25 percentile choices are SCAFFOLDING; recalibrate post-first-run if
distributions are degenerate (instr §5.4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from v1.src.config import (
    VERDICT_DIFFERENTIATION_PERCENTILE,
    VERDICT_STABILITY_PERCENTILE,
)


METRICS_FOR_CALIBRATION: tuple[str, ...] = (
    "mean_drift",
    "variance_drift",
    "per_k_mean_drift_mean",
    "per_k_variance_drift_mean",
)


def _collect_metric_values(
    matrix: dict, *, bit_identical: bool, metric: str
) -> np.ndarray:
    """Collect metric values across all arms for rows matching bit_identical flag."""
    values = []
    for row in matrix["rows"]:
        if bool(row["bit_identical"]) != bit_identical:
            continue
        for arm, cell in row["cells"].items():
            if metric in cell:
                values.append(float(cell[metric]))
    return np.asarray(values, dtype=np.float64)


@dataclass
class MetricThresholds:
    metric: str
    stability_distribution_n: int
    differentiation_distribution_n: int
    stability_threshold: float
    differentiation_threshold: float


def calibrate_thresholds(matrix: dict) -> dict[str, MetricThresholds]:
    """Compute stability + differentiation thresholds per metric.

    Variance drift is signed; calibration uses absolute value so a strongly
    negative drift on input-varying pairs counts as differentiation. Other
    metrics (cosine distance, drift-magnitude means) are non-negative.
    """
    thresholds: dict[str, MetricThresholds] = {}
    for metric in METRICS_FOR_CALIBRATION:
        stab = _collect_metric_values(matrix, bit_identical=True, metric=metric)
        diff = _collect_metric_values(matrix, bit_identical=False, metric=metric)
        if metric in ("variance_drift", "per_k_variance_drift_mean"):
            stab = np.abs(stab)
            diff = np.abs(diff)
        stab_threshold = (
            float(np.percentile(stab, VERDICT_STABILITY_PERCENTILE))
            if stab.size > 0
            else float("nan")
        )
        diff_threshold = (
            float(np.percentile(diff, VERDICT_DIFFERENTIATION_PERCENTILE))
            if diff.size > 0
            else float("nan")
        )
        thresholds[metric] = MetricThresholds(
            metric=metric,
            stability_distribution_n=int(stab.size),
            differentiation_distribution_n=int(diff.size),
            stability_threshold=stab_threshold,
            differentiation_threshold=diff_threshold,
        )
    return thresholds


def classify_cells(matrix: dict, thresholds: dict[str, MetricThresholds]) -> dict:
    """Tag each (item, ordinal) × arm × metric with a classification label.

    Returns a new dict mirroring the matrix structure with per-cell
    `classifications` sub-dict appended (metric -> label).
    """
    out_rows = []
    for row in matrix["rows"]:
        new_row = dict(row)
        new_row["cells"] = {}
        for arm, cell in row["cells"].items():
            new_cell = dict(cell)
            cls_per_metric: dict[str, str] = {}
            for metric, t in thresholds.items():
                if metric not in cell:
                    continue
                v = float(cell[metric])
                if metric in ("variance_drift", "per_k_variance_drift_mean"):
                    v = abs(v)
                # Classification logic per instr §5.4 step 4.
                if v <= t.stability_threshold:
                    label = "stable"
                elif v >= t.differentiation_threshold:
                    label = "differentiated"
                else:
                    label = "indeterminate"
                # Coupling overlay: an input-varying pair whose value sits
                # below the stability threshold is "coupling" (v0 BCDD
                # pattern). This is logged for the verdict-pattern matcher.
                if not row["bit_identical"] and label == "stable":
                    label = "coupling"
                cls_per_metric[metric] = label
            new_cell["classifications"] = cls_per_metric
            new_row["cells"][arm] = new_cell
        out_rows.append(new_row)
    return {**matrix, "rows": out_rows}


def write_thresholds(
    thresholds: dict[str, MetricThresholds],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {k: v.__dict__ for k, v in thresholds.items()},
            indent=2,
        )
    )


def write_distributions(
    matrix: dict,
    path_json: Path,
) -> None:
    """Write the raw stability + differentiation per-metric distributions to JSON.

    PNG rendering is left to a downstream script (matplotlib not assumed
    available in every CC session).
    """
    out: dict = {}
    for metric in METRICS_FOR_CALIBRATION:
        stab = _collect_metric_values(matrix, bit_identical=True, metric=metric)
        diff = _collect_metric_values(matrix, bit_identical=False, metric=metric)
        if metric in ("variance_drift", "per_k_variance_drift_mean"):
            stab = np.abs(stab)
            diff = np.abs(diff)
        out[metric] = {
            "stability_values": stab.tolist(),
            "differentiation_values": diff.tolist(),
        }
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(out, indent=2))
