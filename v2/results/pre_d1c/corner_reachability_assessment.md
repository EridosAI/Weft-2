# V2-PRE-D1c — Corner Reachability Assessment (analytical; instr §7.4)

Structured input for the Phase 0.5 design chat — **not a verdict**. Consumes PRE-B's `worked_example_region.json` and the §2.4 first-principles ranges.

## Per-axis position of the DINOv2-on-AI2-THOR empirical region

| axis | median | range | linear pos | classification |
|---|---|---|---|---|
| magnitude | 0.01714 | [0.0, 1.0] | 0.0171 | near-extreme |
| locality | 0.8364 | [0.0, 1.0] | 0.8364 | off-center |
| continuity | 0.07745 | [0.0, 1.0] | 0.0774 | near-extreme |
| manifold_dim | 13.75 | [1.0, 1024.0] | 0.0125 | near-extreme (log: central, pos 0.3781) |
| repetition | — | ambiguous | — | undetermined (deferred to Phase 0.5) |

## 5D combined reading

Near-extreme axes (linear scale): **magnitude, continuity, manifold_dim**. The worked example sits near the low end on these axes (subtle perturbation, smooth trajectory, low-dim manifold). Per §7.4, corner regions along these axes **may be reachable** by (env, enc) pairs distinct from DINOv2-on-AI2-THOR; Phase 0.5 should consider whether to add corner-sampled crosses as Phase 3 probes, or accept §3.5 corner-avoidance and record that the worked example occupies a low-corner region of the swept space.

**Scale caveat (manifold dim).** Linear-scale positioning reads manifold-dim as near-extreme-low (global D≈13.75 of [1,1024]); a log-scale sweep would read it as central. Phase 0.5 should commit the dimensionality sweep scale before finalising this axis's reachability reading.

**Recommendation framing (not commitment).** Several axes near-extreme-low → do not assume §3.5 corner-avoidance fully covers where real (env, enc) pairs land; Phase 0.5 weighs corner-sampled crosses against the reliability-over-coverage principle (§9.8).
