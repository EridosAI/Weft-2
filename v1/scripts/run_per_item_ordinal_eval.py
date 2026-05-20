#!/usr/bin/env python3
"""Build the per-(item, ordinal) × arm matrix from end-of-Stage-A and
end-of-Stage-B checkpoints across all three arms (spec §10.4, instr §5.3, §9.1).

For each arm, this script:
  1. Loads the end-of-Stage-A and end-of-Stage-B checkpoints.
  2. Constructs the per-arm canonical (item, ordinal) pair list from the
     Stage A annotations.
  3. Computes per-(item, ordinal) drift metrics (mean drift, variance
     drift, per-K mean/variance drift; spec §10.3 metrics 1–4) between the
     two checkpoints.
  4. Computes body representation cosine across the two checkpoints
     (spec §10.3 metric 7).

It then:
  5. Loads the bit-identical (item, ordinal) set from
     `data/v1_shared/bit_identical_item_ordinal.json` (produced separately
     by the substrate / collection scripts).
  6. Builds the (item, ordinal) × arm matrix per instr §5.3.
  7. Writes matrix.{json,csv} under
     `results/inner_pam_v1/arm_comparison_matrix/`.

The verdict-recommendation script reads the matrix.json to apply
threshold calibration + verdict-pattern recognition (separate script).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from v1.src.config import PATHS, get_v1_decoder_n_layers
from v1.src.eval.arm_comparison_matrix import (
    ArmRecords,
    build_matrix,
    write_matrix_csv,
    write_matrix_json,
)
from v1.src.eval.per_item_ordinal_metrics import (
    build_canonical_pairs,
    compute_body_repr_drift,
    compute_drift_metrics,
    evaluate_per_item_ordinal,
    load_annotations,
)
from v1.src.predictor.inner_pam_v1_ablation1 import InnerPAM_v1_Ablation1
from v1.src.predictor.inner_pam_v1_ablation2 import InnerPAM_v1_Ablation2
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary


ARMS: tuple[str, ...] = ("primary", "ablation1", "ablation2")


def _arm_output_dir(arm: str) -> Path:
    return {
        "primary": PATHS.results_arm_primary,
        "ablation1": PATHS.results_arm_ablation1,
        "ablation2": PATHS.results_arm_ablation2,
    }[arm]


def _construct(arm: str) -> torch.nn.Module:
    if arm == "primary":
        return InnerPAM_v1_Primary(decoder_n_layers=get_v1_decoder_n_layers())
    if arm == "ablation1":
        return InnerPAM_v1_Ablation1(decoder_n_layers=get_v1_decoder_n_layers())
    return InnerPAM_v1_Ablation2()


def _load_checkpoint(predictor: torch.nn.Module, ckpt: Path) -> torch.nn.Module:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    predictor.load_state_dict(state["predictor_state"])
    return predictor


def main() -> int:
    p = argparse.ArgumentParser(description="Build per-(item, ordinal) × arm matrix")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()
    device = torch.device(args.device)

    # Load shared embeddings + annotations.
    embeddings_a = np.load(PATHS.embeddings_stage_a)
    embeddings_b = np.load(PATHS.embeddings_stage_b)
    annotations_a = load_annotations(PATHS.annotations_stage_a)
    pairs = build_canonical_pairs(annotations_a)
    print(f"[eval] canonical pairs: {len(pairs)}")

    # Load bit-identical set.
    bit_id_path = PATHS.bit_identical_item_ordinal
    if bit_id_path.exists():
        bit_id_pairs = [
            (int(e["item"]), int(e["ordinal"]))
            for e in json.loads(bit_id_path.read_text())
        ]
    else:
        print(f"[eval] WARNING: {bit_id_path} not found; treating no pairs as bit-identical")
        bit_id_pairs = []
    print(f"[eval] bit-identical pairs: {len(bit_id_pairs)}")

    arm_records_list: list[ArmRecords] = []
    for arm in ARMS:
        outdir = _arm_output_dir(arm)
        ckpt_a = outdir / "ckpt_end_stage_a.pt"
        ckpt_b = outdir / "ckpt_end_stage_b.pt"
        if not (ckpt_a.exists() and ckpt_b.exists()):
            raise SystemExit(
                f"[eval] missing checkpoints for {arm}: a={ckpt_a.exists()}, "
                f"b={ckpt_b.exists()}"
            )
        print(f"[eval] {arm}: loading checkpoints...")
        predictor_a = _load_checkpoint(_construct(arm), ckpt_a).to(device)
        predictor_b = _load_checkpoint(_construct(arm), ckpt_b).to(device)
        # Per-(item, ordinal) eval at both checkpoints. Stage A and Stage B
        # share canonical windows by index (same shared frame collection);
        # we use the Stage B embeddings for the windows-and-targets at the
        # canonical pairs.
        records_a = evaluate_per_item_ordinal(predictor_a, embeddings_b, pairs, device)
        records_b = evaluate_per_item_ordinal(predictor_b, embeddings_b, pairs, device)
        drift_records = compute_drift_metrics(records_a, records_b)
        body_records = compute_body_repr_drift(
            predictor_a, predictor_b, embeddings_b, pairs, device
        )
        arm_records_list.append(
            ArmRecords(arm=arm, drift_records=drift_records, body_repr_records=body_records)
        )
        print(f"[eval] {arm}: drift+body records computed ({len(drift_records)} pairs)")

    matrix = build_matrix(arm_records_list, bit_id_pairs)
    write_matrix_json(matrix, PATHS.results_matrix / "matrix.json")
    write_matrix_csv(matrix, PATHS.results_matrix / "matrix.csv")
    print(f"[eval] matrix written: {PATHS.results_matrix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
