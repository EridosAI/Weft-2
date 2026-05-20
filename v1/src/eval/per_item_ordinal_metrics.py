"""Per-(item, ordinal) metrics for the v1 verdict workflow.

Spec §10.2 (canonical windows / target frames), §10.3 (the seven metrics),
§10.4 (arm-comparison structure). Instr §5.1–5.3.

Canonical artefact contract (instr §5.1):
  - Viewing position 1 for each of the 5 items is the canonical evaluation
    position (v0 convention).
  - Canonical window: W=16 embeddings ending K=16 frames before the
    canonical target frame.
  - Canonical target: the frame at canonical viewing-position-1 for each
    (item, ordinal).

The frame-index mapping comes from `annotations_stage_{a,b}.jsonl` and is
arm-agnostic (instr §1.4: a single shared frame collection backs all three
arms).

Output schema for a per-(item, ordinal) checkpoint JSON:
{
  "step": int,
  "arm": str,
  "stage": "A" | "B",
  "pairs": [
    {
      "item": int,        # viewing_position_id (1..5)
      "ordinal": int,     # close-up-segment ordinal
      "mean": [float; d], # K=K-1 predicted mean head
      "log_var": float,   # K=K-1 predicted log_var
      "loss": float,      # path-prediction loss on canonical window
      "per_k_mean":    [[float; d]; K],
      "per_k_log_var": [float; K],
      "per_k_loss":    [float; K],
    },
    ...
  ]
}

Spec §10.3 metric 7 (body representation cosine) is computed separately and
written to `body_repr_{step}.json` (requires comparing two checkpoints'
representations on the same window; not a single-checkpoint quantity).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from v1.src.config import EMBED_DIM, PREDICT_K, WINDOW_W
from v1.src.predictor.inner_pam_v1_shared import path_prediction_loss


# --------------------------------------------------------------------------
# Canonical-window resolution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalPair:
    """A canonical (item, ordinal) evaluation point in the embedding stream.

    All indices are absolute into the per-arm embeddings.npy array.
    """

    item: int                       # viewing_position_id (1..5)
    ordinal: int                    # close-up-segment ordinal
    target_index: int               # absolute embedding-stream index of canonical target frame
    window_start: int               # inclusive
    window_end: int                 # inclusive; target_index = window_end + K

    def window_slice(self) -> slice:
        return slice(self.window_start, self.window_end + 1)


def build_canonical_pairs(
    annotations: list[dict],
    items: Iterable[int] = (1, 2, 3, 4, 5),
) -> list[CanonicalPair]:
    """Build canonical (item, ordinal) pairs from an annotations stream.

    Annotation schema (carried forward from v0 with v1's `perturbation_active`
    field):
      {"frame_index": int, "loop_index": int, "phase": str,
       "viewing_position_id": int, "close_up_ordinal": int?, ...}

    A canonical pair exists for each (item, ordinal) where:
      - `viewing_position_id == item`,
      - `close_up_ordinal` is set,
      - the frame is in the close-up segment (not transit),
      - and `target_index - WINDOW_W - PREDICT_K + 1 >= 0` (window fits
        before the stream).

    For each (item, ordinal) we use the FIRST appearance of that ordinal at
    that item in the stream as the canonical target.
    """
    seen: dict[tuple[int, int], int] = {}
    for ann in annotations:
        vp = ann.get("viewing_position_id")
        ord_idx = ann.get("close_up_ordinal")
        if vp is None or ord_idx is None:
            continue
        if vp not in items:
            continue
        if ann.get("phase") not in ("close_up", "closeup", None):
            # Some annotation streams omit phase; treat the ordinal field as
            # authoritative when present.
            if ann.get("phase") is not None:
                continue
        key = (int(vp), int(ord_idx))
        if key in seen:
            continue
        seen[key] = int(ann["frame_index"])

    pairs: list[CanonicalPair] = []
    for (item, ordinal), target_idx in sorted(seen.items()):
        window_end = target_idx - PREDICT_K
        window_start = window_end - WINDOW_W + 1
        if window_start < 0:
            continue
        pairs.append(
            CanonicalPair(
                item=item,
                ordinal=ordinal,
                target_index=target_idx,
                window_start=window_start,
                window_end=window_end,
            )
        )
    return pairs


def load_annotations(path: Path) -> list[dict]:
    """Load a JSONL annotations file as a list of dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Per-checkpoint per-(item, ordinal) evaluation
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate_per_item_ordinal(
    predictor: nn.Module,
    embeddings: np.ndarray,
    pairs: list[CanonicalPair],
    device: torch.device,
) -> list[dict]:
    """Run the predictor on each canonical pair; return a list of records.

    The returned schema matches the module-level "Output schema" docstring's
    `pairs` list element.
    """
    predictor.eval()
    records: list[dict] = []
    for pair in pairs:
        window_np = embeddings[pair.window_start : pair.window_end + 1]
        target_np = embeddings[pair.target_index - PREDICT_K + 1 : pair.target_index + 1]
        if window_np.shape != (WINDOW_W, EMBED_DIM):
            raise ValueError(
                f"window shape {window_np.shape} != expected ({WINDOW_W}, {EMBED_DIM}) "
                f"for pair item={pair.item} ordinal={pair.ordinal}"
            )
        if target_np.shape != (PREDICT_K, EMBED_DIM):
            raise ValueError(
                f"target shape {target_np.shape} != expected ({PREDICT_K}, {EMBED_DIM}) "
                f"for pair item={pair.item} ordinal={pair.ordinal}"
            )
        window = torch.from_numpy(window_np).float().unsqueeze(0).to(device)
        target = torch.from_numpy(target_np).float().unsqueeze(0).to(device)

        mean, log_var = predictor(window)
        per_k_loss = []
        # Per-K loss: same formula as path_prediction_loss but kept per K.
        for k in range(PREDICT_K):
            l_k = path_prediction_loss(
                mean[:, k : k + 1, :], log_var[:, k : k + 1], target[:, k : k + 1, :]
            )
            per_k_loss.append(float(l_k.item()))
        loss_total = float(np.mean(per_k_loss))

        record = {
            "item": pair.item,
            "ordinal": pair.ordinal,
            "target_index": pair.target_index,
            "window_start": pair.window_start,
            "window_end": pair.window_end,
            "mean": mean[0, PREDICT_K - 1].cpu().numpy().tolist(),
            "log_var": float(log_var[0, PREDICT_K - 1].item()),
            "loss": loss_total,
            "per_k_mean": mean[0].cpu().numpy().tolist(),
            "per_k_log_var": log_var[0].cpu().numpy().tolist(),
            "per_k_loss": per_k_loss,
        }
        records.append(record)
    return records


def write_per_item_ordinal_json(
    records: list[dict],
    output_path: Path,
    *,
    step: int,
    arm: str,
    stage: str,
) -> None:
    """Write the per-checkpoint per-(item, ordinal) JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"step": step, "arm": arm, "stage": stage, "pairs": records}
    output_path.write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------
# Drift metrics from two checkpoints (spec §10.3 metrics 1-6)
# --------------------------------------------------------------------------


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity between two 1-D vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return 1.0 - cos


def compute_drift_metrics(
    records_stage_a: list[dict],
    records_stage_b: list[dict],
) -> list[dict]:
    """Compute per-(item, ordinal) drift metrics between two checkpoints.

    Metrics produced per pair (subset; spec §10.3 metrics 1-4):
      - mean_drift            (cosine distance between K-averaged means)
      - variance_drift        (mean log_var difference B − A; v0 sign convention)
      - per_k_mean_drift      (per-K cosine distance)
      - per_k_variance_drift  (per-K log_var difference)
    Metrics 5/6 (stability at bit-identical pairs) and 7 (body repr cosine)
    are computed elsewhere; this function returns the per-pair drift records.
    """
    idx_a = {(r["item"], r["ordinal"]): r for r in records_stage_a}
    idx_b = {(r["item"], r["ordinal"]): r for r in records_stage_b}
    common = sorted(set(idx_a.keys()) & set(idx_b.keys()))
    out: list[dict] = []
    for key in common:
        ra, rb = idx_a[key], idx_b[key]
        # K-averaged mean drift: cosine distance between mean-over-K of stage A
        # vs stage B predicted means.
        mean_a_avg = np.array(ra["per_k_mean"]).mean(axis=0)
        mean_b_avg = np.array(rb["per_k_mean"]).mean(axis=0)
        mean_drift = cosine_distance(mean_a_avg, mean_b_avg)

        # K-averaged variance drift: log_var(B) − log_var(A), mean over K
        # (v0 sign convention per BCDD; instr §5.2 metric 2).
        log_var_a = np.array(ra["per_k_log_var"]).mean()
        log_var_b = np.array(rb["per_k_log_var"]).mean()
        variance_drift = float(log_var_b - log_var_a)

        per_k_mean_drift = [
            cosine_distance(
                np.array(ra["per_k_mean"][k]), np.array(rb["per_k_mean"][k])
            )
            for k in range(PREDICT_K)
        ]
        per_k_variance_drift = [
            float(rb["per_k_log_var"][k] - ra["per_k_log_var"][k])
            for k in range(PREDICT_K)
        ]
        out.append(
            {
                "item": key[0],
                "ordinal": key[1],
                "mean_drift": mean_drift,
                "variance_drift": variance_drift,
                "per_k_mean_drift": per_k_mean_drift,
                "per_k_variance_drift": per_k_variance_drift,
            }
        )
    return out


# --------------------------------------------------------------------------
# Bit-identical pair identification (instr §5.1)
# --------------------------------------------------------------------------


def compute_bit_identical_pairs(
    annotations_stage_a: list[dict],
    annotations_stage_b: list[dict],
    frames_dir_stage_a: Path,
    frames_dir_stage_b: Path,
    *,
    frame_name_pattern: str = "frame_{idx:08d}.png",
) -> list[tuple[int, int]]:
    """Identify (item, ordinal) pairs whose canonical target frames are
    pixel-MD5 identical across Stage A → Stage B.

    A pair qualifies as bit-identical iff (a) both Stage A and Stage B have
    a canonical target frame for the same (item, ordinal) and (b) the
    pixel-MD5 of those two frames are identical.

    The pixel-MD5 comparison must be on raw PNG bytes (the v0 substrate
    finding 8 convention); cosine similarity is NOT used here.
    """
    import hashlib

    pairs_a = {(p.item, p.ordinal): p.target_index for p in build_canonical_pairs(annotations_stage_a)}
    pairs_b = {(p.item, p.ordinal): p.target_index for p in build_canonical_pairs(annotations_stage_b)}
    common = sorted(set(pairs_a.keys()) & set(pairs_b.keys()))

    def md5_of(idx: int, frames_dir: Path) -> str:
        path = frames_dir / frame_name_pattern.format(idx=idx)
        return hashlib.md5(path.read_bytes()).hexdigest()

    out: list[tuple[int, int]] = []
    for key in common:
        ta = pairs_a[key]
        tb = pairs_b[key]
        if md5_of(ta, frames_dir_stage_a) == md5_of(tb, frames_dir_stage_b):
            out.append(key)
    return out


def write_bit_identical_pairs(
    pairs: list[tuple[int, int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([{"item": i, "ordinal": o} for (i, o) in pairs], indent=2)
    )


# --------------------------------------------------------------------------
# Spec §10.3 metric 7: body representation cosine across stages
# --------------------------------------------------------------------------


@torch.no_grad()
def body_representation_at_window(
    predictor: nn.Module,
    window: torch.Tensor,
) -> torch.Tensor:
    """Return the encoder body's pre-readout representation at window position W-1.

    Spec §10.3 metric 7:
      - Primary, Ablation 1: `memory[:, -1, :]` where `memory = self.encoder(x)`
        — the encoder body's final-window-position output before
        cross-attention into the K queries.
      - Ablation 2: `last_token = x[:, -1, :]` after encoder; identical to v0
        by construction.

    Implementation: re-run the input projection + positional embedding +
    encoder body (skipping the readout entirely). Works for all three arms
    because they share the same body family / scaffolding (instr §1.1).

    Returns (B, hidden) tensor.
    """
    p = predictor
    if hasattr(p, "input_proj") and hasattr(p, "pos_emb") and hasattr(p, "encoder"):
        x = p.input_proj(window)
        positions = torch.arange(window.shape[1], device=window.device)
        x = x + p.pos_emb(positions).unsqueeze(0)
        memory = p.encoder(x)
        return memory[:, -1, :]
    raise AttributeError(
        f"predictor {type(p).__name__} lacks input_proj/pos_emb/encoder attributes "
        f"required for body-representation extraction"
    )


@torch.no_grad()
def compute_body_repr_drift(
    predictor_stage_a: nn.Module,
    predictor_stage_b: nn.Module,
    embeddings: np.ndarray,
    pairs: list[CanonicalPair],
    device: torch.device,
) -> list[dict]:
    """Per-canonical-pair body-representation cosine across two checkpoints.

    The two predictor instances must share the same architecture / weights
    only differing in checkpoint state (end-of-Stage-A vs end-of-Stage-B).
    """
    predictor_stage_a.eval()
    predictor_stage_b.eval()
    out: list[dict] = []
    for pair in pairs:
        window_np = embeddings[pair.window_start : pair.window_end + 1]
        window = torch.from_numpy(window_np).float().unsqueeze(0).to(device)
        repr_a = body_representation_at_window(predictor_stage_a, window)[0].cpu().numpy()
        repr_b = body_representation_at_window(predictor_stage_b, window)[0].cpu().numpy()
        cos = 1.0 - cosine_distance(repr_a, repr_b)
        out.append(
            {"item": pair.item, "ordinal": pair.ordinal, "body_repr_cosine": float(cos)}
        )
    return out
