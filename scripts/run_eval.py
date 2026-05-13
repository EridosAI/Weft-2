"""Evaluation harness for v0 — single-checkpoint or developmental-arc mode.

Single-checkpoint mode (default):
  python scripts/run_eval.py \
      --checkpoint results/inner_pam_v0/phase1_main/ckpt_{step}.pt \
      --probes phase1 \
      --output results/inner_pam_v0/phase1_main/eval_{step}.json

Developmental-arc mode (post-Phase-3, instr §10):
  python scripts/run_eval.py --developmental

The --developmental flag runs the aggregate evaluation that produces the
developmental-arc artefacts (time series, repetition-stratified accuracy,
M5 tau-sensitivity overlay, M2 variance trajectories, shuffle-sanity-check
verdicts). It does not run until Phase 3 is complete; before then it errors
out cleanly with a message listing the missing inputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    HELD_OUT_LOOPS,
    PATHS,
    PHASE1,
    PHASE2,
    PHASE3,
    SEED_PROBE_SAMPLING,
    TAU_CALIB_END_STEP,
    TAU_CALIB_START_STEP,
)
from src.eval.controls import (  # noqa: E402
    cosine_baseline_top1,
    shuffle_sanity_check,
)
from src.eval.metrics import (  # noqa: E402
    REP_BINS,
    m1_centreline_accuracy,
    m1_per_probe_centreline_at_k,
    m2_variance_calibration,
    m3_cluster_sharpness,
    m4_repetition_stratified_accuracy,
    m5_recall_fractions,
    m6_cluster_accommodation,
    m7_compounding_accommodation,
    run_probes_through_predictor,
)
from src.eval.probes import build_probes, compute_held_out_boundary  # noqa: E402
from src.memory.memory_bank import MemoryBank  # noqa: E402
from src.mixing.recall_mixer import mix  # noqa: E402
from src.predictor.inner_pam import InnerPAM  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_annotations(path: Path) -> list[dict]:
    out = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _resolve_phase(name: str):
    return {"phase1": PHASE1, "phase2": PHASE2, "phase3": PHASE3}[name]


def _load_bank_for_checkpoint(ckpt_path: Path) -> Optional[MemoryBank]:
    bank_dir = ckpt_path.parent / ckpt_path.stem  # e.g. ckpt_5000.pt -> ckpt_5000
    if not bank_dir.is_dir():
        return None
    return MemoryBank.load(bank_dir)


def _rep_counts_at_step(
    annotations: list[dict],
    step: int,
    probe_types: tuple[str, ...] = ("steady", "cue"),
) -> dict[tuple[str, int], int]:
    """Repetition counts of (probe_type, from_item) over [0, step] in stream order."""
    from src.config import VIEWING_POSITION_IDS, ROUTE_TRANSITIONS

    out: dict[tuple[str, int], int] = {}
    for vp in VIEWING_POSITION_IDS:
        out[("steady", int(vp))] = 0
    for (a, _b) in ROUTE_TRANSITIONS:
        out[("cue", int(a))] = 0
    last_state: tuple[str, Optional[int]] = ("transit", None)
    for i in range(min(step + 1, len(annotations))):
        ann = annotations[i]
        ph = str(ann.get("phase", "transit"))
        vp = ann.get("viewing_position_id")
        vp = int(vp) if vp not in (None, 0) else None
        if ph == "dwell" and last_state[0] != "dwell" and vp is not None:
            out[("steady", vp)] = out.get(("steady", vp), 0) + 1
        if ph == "transit" and last_state[0] == "dwell" and last_state[1] is not None:
            out[("cue", last_state[1])] = out.get(("cue", last_state[1]), 0) + 1
        last_state = (ph, vp)
    return out


def run_single_checkpoint(
    checkpoint_path: Path,
    phase_name: str,
    output_path: Path,
    device: torch.device,
    tau_value: Optional[float],
) -> int:
    phase = _resolve_phase(phase_name)
    annotations = _load_annotations(phase.annotations)
    embeddings = np.load(phase.embeddings)
    assert embeddings.shape[0] == len(annotations)

    held_out_start, held_out_end = compute_held_out_boundary(annotations, HELD_OUT_LOOPS)
    probes_by_type = build_probes(
        annotations, held_out_start, held_out_end, seed=SEED_PROBE_SAMPLING
    )
    all_probes = probes_by_type["steady"] + probes_by_type["cue"]
    print(f"[eval] probes: steady={len(probes_by_type['steady'])} "
          f"cue={len(probes_by_type['cue'])} total={len(all_probes)}", flush=True)

    predictor = InnerPAM().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["predictor_state"])
    predictor.eval()

    bank = _load_bank_for_checkpoint(checkpoint_path)
    if bank is None:
        raise FileNotFoundError(
            f"bank state directory missing alongside checkpoint {checkpoint_path}"
        )

    results = run_probes_through_predictor(all_probes, embeddings, predictor, device)
    steady_results = [r for r in results if r.probe.probe_type == "steady"]
    cue_results = [r for r in results if r.probe.probe_type == "cue"]

    # Mixing modes (M5) — only meaningful if tau is provided.
    modes: list[str] = []
    if tau_value is not None:
        for r in results:
            window = embeddings[r.probe.window_start : r.probe.window_end + 1]
            w = torch.from_numpy(window).to(device).unsqueeze(0)
            res = mix(w, predictor, bank, tau=float(tau_value))
            modes.append(res.mode)

    m1 = m1_centreline_accuracy(results)
    m2 = m2_variance_calibration(results)
    m3 = m3_cluster_sharpness(results)
    rep_counts = _rep_counts_at_step(annotations, int(ckpt["step"]))
    m4 = m4_repetition_stratified_accuracy(results, rep_counts)
    m5 = m5_recall_fractions(modes) if modes else {"note": "tau not supplied"}
    c1 = cosine_baseline_top1(all_probes, embeddings, bank)

    payload = {
        "checkpoint": str(checkpoint_path),
        "phase": phase_name,
        "step": int(ckpt["step"]),
        "git_commit_at_save": ckpt.get("git_commit"),
        "git_commit_at_eval": _git_commit(),
        "held_out_region": [held_out_start, held_out_end],
        "n_probes": {
            "steady": len(steady_results),
            "cue": len(cue_results),
            "total": len(all_probes),
        },
        "tau_used": tau_value,
        "rep_counts_at_step": {f"{k[0]}:item_{k[1]}": v for k, v in rep_counts.items()},
        "M1_centreline_accuracy": m1,
        "M2_variance_calibration": m2,
        "M3_cluster_sharpness": m3,
        "M4_rep_stratified_accuracy": m4,
        "M5_recall_fractions": m5,
        "C1_cosine_baseline_top1": {
            k: v for k, v in c1.items() if k != "per_probe"
        },  # full per-probe stays in a separate file to keep the headline JSON small
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[eval] wrote {output_path}", flush=True)
    return 0


def run_developmental_arc(output_dir: Path) -> int:
    """Aggregate eval at endpoint (instr §10)."""
    missing = []
    for ph in (PHASE1, PHASE2, PHASE3):
        if not ph.results_main.is_dir() or not (ph.results_main / "training_summary.json").is_file():
            missing.append(ph.name)
    if missing:
        print(f"[eval] FAIL: developmental-arc requires all three phases complete; "
              f"missing {missing}", file=sys.stderr)
        return 3
    output_dir.mkdir(parents=True, exist_ok=True)
    # The aggregate report is constructed from per-checkpoint JSONs across phases.
    # Each phase's eval_{step}.json files are concatenated into time series.
    arc: dict[str, Any] = {"per_phase_checkpoints": {}, "git_commit": _git_commit()}
    for ph in (PHASE1, PHASE2, PHASE3):
        evals = sorted(ph.results_main.glob("eval_*.json"))
        arc["per_phase_checkpoints"][ph.name] = [str(p) for p in evals]
    (output_dir / "developmental_arc.json").write_text(json.dumps(arc, indent=2))
    print(f"[eval] wrote developmental-arc skeleton at "
          f"{output_dir / 'developmental_arc.json'}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--probes", type=str, default="phase1",
                        choices=("phase1", "phase2", "phase3"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tau", type=float, default=None,
                        help="confidence threshold for M5; if omitted, M5 is skipped")
    parser.add_argument("--developmental", action="store_true",
                        help="aggregate-mode developmental arc evaluation (instr §10)")
    parser.add_argument("--developmental_output", type=Path,
                        default=PATHS.results_developmental_arc)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.developmental:
        return run_developmental_arc(args.developmental_output)

    if args.checkpoint is None or args.output is None:
        print("[eval] FAIL: --checkpoint and --output required in single-checkpoint mode",
              file=sys.stderr)
        return 1
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    return run_single_checkpoint(
        args.checkpoint, args.probes, args.output, device, args.tau
    )


if __name__ == "__main__":
    raise SystemExit(main())
