"""Per-(item, ordinal) × arm matrix construction (spec §10.4, instr §5.3, §9.1).

For each metric in spec §10.3 (1-6) and each arm × each (item, ordinal),
produce a cell. Output is both JSON (for downstream programmatic threshold
calibration) and CSV (for human inspection by reviewer / verdict-assignment
chats).

Metric 7 (body representation cosine) is added as a per-pair, per-arm
quantity (single value per pair × arm, not Stage A vs B drift).

Bit-identical (item, ordinal) pairs are flagged in a separate column so
threshold calibration (`threshold_calibration.py`) can stratify.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


ARMS: tuple[str, ...] = ("primary", "ablation1", "ablation2")


@dataclass
class ArmRecords:
    """Drift-metric records + body-repr records for a single arm."""

    arm: str
    drift_records: list[dict]            # output of compute_drift_metrics
    body_repr_records: list[dict]        # output of compute_body_repr_drift


def build_matrix(
    arm_records_list: list[ArmRecords],
    bit_identical_pairs: list[tuple[int, int]],
) -> dict:
    """Construct the per-(item, ordinal) × arm matrix data structure.

    Returns:
      {
        "rows": [
          {"item": int, "ordinal": int, "bit_identical": bool,
           "cells": {arm_name: {metric_name: value, ...}, ...}},
          ...
        ],
        "arms": [...],
        "metrics": [...],
      }
    """
    bit_id_set = {(int(i), int(o)) for (i, o) in bit_identical_pairs}
    # Union of (item, ordinal) keys present in any arm
    all_keys: set[tuple[int, int]] = set()
    for ar in arm_records_list:
        for r in ar.drift_records:
            all_keys.add((int(r["item"]), int(r["ordinal"])))

    metric_names = [
        "mean_drift",
        "variance_drift",
        "per_k_mean_drift_mean",        # scalar summary of per-K vector
        "per_k_variance_drift_mean",    # scalar summary of per-K vector
        "body_repr_cosine",
    ]

    rows: list[dict] = []
    for (item, ordinal) in sorted(all_keys):
        row: dict = {
            "item": item,
            "ordinal": ordinal,
            "bit_identical": (item, ordinal) in bit_id_set,
            "cells": {},
        }
        for ar in arm_records_list:
            drift = next(
                (r for r in ar.drift_records if r["item"] == item and r["ordinal"] == ordinal),
                None,
            )
            body = next(
                (r for r in ar.body_repr_records if r["item"] == item and r["ordinal"] == ordinal),
                None,
            )
            cell: dict = {}
            if drift is not None:
                cell["mean_drift"] = float(drift["mean_drift"])
                cell["variance_drift"] = float(drift["variance_drift"])
                cell["per_k_mean_drift_mean"] = float(np.mean(drift["per_k_mean_drift"]))
                cell["per_k_variance_drift_mean"] = float(np.mean(drift["per_k_variance_drift"]))
                cell["per_k_mean_drift"] = list(drift["per_k_mean_drift"])
                cell["per_k_variance_drift"] = list(drift["per_k_variance_drift"])
            if body is not None:
                cell["body_repr_cosine"] = float(body["body_repr_cosine"])
            row["cells"][ar.arm] = cell
        rows.append(row)

    return {
        "rows": rows,
        "arms": [ar.arm for ar in arm_records_list],
        "metrics": metric_names,
    }


def write_matrix_json(matrix: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix, indent=2))


def write_matrix_csv(matrix: dict, path: Path) -> None:
    """Flatten the matrix to a CSV with one row per (item, ordinal) and
    one column per (arm, metric)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arms = matrix["arms"]
    metric_columns = [
        "mean_drift",
        "variance_drift",
        "per_k_mean_drift_mean",
        "per_k_variance_drift_mean",
        "body_repr_cosine",
    ]
    fieldnames = ["item", "ordinal", "bit_identical"]
    for arm in arms:
        for m in metric_columns:
            fieldnames.append(f"{arm}_{m}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in matrix["rows"]:
            csv_row = {
                "item": row["item"],
                "ordinal": row["ordinal"],
                "bit_identical": row["bit_identical"],
            }
            for arm in arms:
                cell = row["cells"].get(arm, {})
                for m in metric_columns:
                    csv_row[f"{arm}_{m}"] = cell.get(m, "")
            writer.writerow(csv_row)
