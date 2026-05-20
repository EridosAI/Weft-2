"""V2-PRE-D1c — corner reachability characterisation (instr §7.4; analytical).

Consumes PRE-B's worked_example_region.json plus the §2.4 first-principles
ranges and characterises, per axis, the position of the empirical worked-example
region relative to the sweep range:

  * central     — median within the middle 60% of range AND IQR contained in range
  * off-center  — median in the outer 40% (but not outer 10%)
  * near-extreme — median in the outer 10%, or a W4 boundary

5D combined reading: all central → §3.5 corner-avoidance retained; any
near-extreme → corner regions along that axis may be reachable by (env, enc)
pairs distinct from DINOv2-on-AI2-THOR, and Phase 0.5 considers corner-sampled
crosses as Phase 3 probes. This is structured input for the Phase 0.5 design
chat — NOT a verdict (zero arm-runs).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from v2.config import RESULTS_PRE_B, RESULTS_PRE_D1C


def _position(median: float, rng: list[float]) -> float:
    lo, hi = rng
    return (median - lo) / (hi - lo) if hi > lo else 0.5


def _classify(pos: float) -> str:
    if pos < 0.10 or pos > 0.90:
        return "near-extreme"
    if pos < 0.20 or pos > 0.80:
        return "off-center"
    return "central"


def assess() -> dict:
    region = json.loads((RESULTS_PRE_B / "worked_example_region.json").read_text())
    axes = region["per_axis_distributions"]
    s24 = region["ask4_s24_ranges"]

    # Per-axis median used for positioning.
    medians = {
        "magnitude": axes["magnitude"]["distribution"]["median"],
        "locality": axes["locality"]["distribution"]["median"],
        "continuity": axes["continuity"]["distribution"]["median"],
        "manifold_dim": axes["manifold_dim"]["global_D"],
    }
    iqrs = {
        "magnitude": axes["magnitude"]["distribution"]["iqr"],
        "locality": axes["locality"]["distribution"]["iqr"],
        "continuity": axes["continuity"]["distribution"]["iqr"],
        "manifold_dim": axes["manifold_dim"]["distribution_local_per_loop"].get("iqr"),
    }

    out = {}
    for ax in ("magnitude", "locality", "continuity", "manifold_dim"):
        rng = s24[ax]["range"]
        med = medians[ax]
        pos = _position(med, rng)
        entry = {"median": med, "range": rng, "iqr": iqrs[ax],
                 "position_fraction_linear": round(pos, 4),
                 "classification_linear": _classify(pos)}
        # Manifold dim plausibly sweeps on a log scale; report both readings.
        if ax == "manifold_dim":
            log_pos = (math.log(med) - math.log(rng[0])) / (math.log(rng[1]) - math.log(rng[0]))
            entry["position_fraction_log"] = round(log_pos, 4)
            entry["classification_log"] = _classify(log_pos)
            entry["scale_note"] = ("axis-scale (linear vs log) is uncommitted; linear "
                                   "reads near-extreme-low, log reads central — Phase 0.5 "
                                   "should commit the dimensionality sweep scale.")
        out[ax] = entry

    out["repetition"] = {"classification": "undetermined",
                         "note": "§2.4 range ambiguous (deferred to Phase 0.5); position not assessable."}

    near_extreme = [a for a in ("magnitude", "locality", "continuity", "manifold_dim")
                    if out[a]["classification_linear"] == "near-extreme"]
    corner_avoidance_holds = len(near_extreme) == 0
    return {"per_axis": out, "near_extreme_axes_linear": near_extreme,
            "corner_avoidance_holds_linear": corner_avoidance_holds}


def write_markdown(a: dict) -> Path:
    RESULTS_PRE_D1C.mkdir(parents=True, exist_ok=True)
    L = ["# V2-PRE-D1c — Corner Reachability Assessment (analytical; instr §7.4)\n",
         "Structured input for the Phase 0.5 design chat — **not a verdict**. Consumes "
         "PRE-B's `worked_example_region.json` and the §2.4 first-principles ranges.\n",
         "## Per-axis position of the DINOv2-on-AI2-THOR empirical region\n",
         "| axis | median | range | linear pos | classification |",
         "|---|---|---|---|---|"]
    for ax in ("magnitude", "locality", "continuity", "manifold_dim"):
        e = a["per_axis"][ax]
        cls = e["classification_linear"]
        if ax == "manifold_dim":
            cls += f" (log: {e['classification_log']}, pos {e['position_fraction_log']})"
        L.append(f"| {ax} | {e['median']:.4g} | {e['range']} | {e['position_fraction_linear']} | {cls} |")
    L.append(f"| repetition | — | ambiguous | — | undetermined (deferred to Phase 0.5) |\n")

    L.append("## 5D combined reading\n")
    if a["corner_avoidance_holds_linear"]:
        L.append("All assessable axes are central → §3.5 corner-avoidance is retained; "
                 "corner-sampled crosses are likely **not** justified as Phase 3 probes.\n")
    else:
        ne = ", ".join(a["near_extreme_axes_linear"])
        L.append(f"Near-extreme axes (linear scale): **{ne}**. The worked example sits near "
                 f"the low end on these axes (subtle perturbation, smooth trajectory, low-dim "
                 f"manifold). Per §7.4, corner regions along these axes **may be reachable** by "
                 f"(env, enc) pairs distinct from DINOv2-on-AI2-THOR; Phase 0.5 should consider "
                 f"whether to add corner-sampled crosses as Phase 3 probes, or accept §3.5 "
                 f"corner-avoidance and record that the worked example occupies a low-corner "
                 f"region of the swept space.\n")
    L.append("**Scale caveat (manifold dim).** Linear-scale positioning reads manifold-dim as "
             "near-extreme-low (global D≈13.75 of [1,1024]); a log-scale sweep would read it as "
             "central. Phase 0.5 should commit the dimensionality sweep scale before finalising "
             "this axis's reachability reading.\n")
    L.append("**Recommendation framing (not commitment).** Several axes near-extreme-low → do "
             "not assume §3.5 corner-avoidance fully covers where real (env, enc) pairs land; "
             "Phase 0.5 weighs corner-sampled crosses against the reliability-over-coverage "
             "principle (§9.8).\n")
    path = RESULTS_PRE_D1C / "corner_reachability_assessment.md"
    path.write_text("\n".join(L))
    return path
