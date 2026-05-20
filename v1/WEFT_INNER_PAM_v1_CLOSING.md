# Weft Inner PAM v1 — Closing Document

**Purpose.** Records the v1 verdict, the institutional memory v1 produced, and the conceptual reframing v2 inherits. Companion to `WEFT_INNER_PAM_v1_Spec_pass1_sections_1_to_6.md`, `WEFT_INNER_PAM_v1_Spec_pass2_sections_7_to_11.md`, `WEFT_INNER_PAM_v1_EXPERIMENT_INSTRUCTIONS.md`, and `HANDOFF.md`; written to be the strategic input for v2 design without prejudging v2's specific axis choices.

**Status.** Verdict recorded after PRE-B established that v1's substrate prerequisite is not constructible on the configurations available. Stage A and Stage B never ran. v2 scoping is open (separate chat).

---

## 1. The verdict

**V1P-with-content — Substrate prerequisite not constructible; architectural claim unresolved.**

v1's design committed to a controlled perturbation regime producing cross-stage DINOv2 cosine drops in [0.05, 0.10] (spec §1.2 commitment 3), satisfying both signal sufficiency (input-varying pair differentiation measurable above the bit-identical baseline) and coupling-pathway preservation (Ablation 2 has a substrate state in which to reproduce v0's pooled-readout pathology). PRE-B established empirically that no AI2-THOR 5.0.0 mutation mechanism produces this regime stably on the substrate v1 inherited from v0.

The architectural claim — that K learnable output queries with cross-attention, paired with per-K-position scalar variance heads, produces co-primary differentiation in mean and variance under perturbation where v0's pooled readout did not — was tested *only at the preflight layer*. PRE-D verified all 11 architectural property assertions across all three arms at decoder layer placeholder L_d=2. PRE-A verified inherited substrate (with finding 3 reclassified as substrate-as-feature; threshold updated 0.9999 → 0.999 per §4 below). PRE-B did not produce a usable perturbation mechanism. Stage A, Stage B, and the per-(item, ordinal) evaluation never ran.

**This is not "v1 failed to test the architectural claim." It is "v1 established that the substrate prerequisite the architectural claim requires is not available in AI2-THOR 5.0.0's mutation API."** The distinction matters for v2: the claim itself remains neither supported nor falsified, and v2's job is not to retry v1, but to characterise the conditions under which the claim (or a successor) can be tested at all.

---

## 2. What v1 demonstrated

**The preflight discipline worked.** Four preflight stages, each producing actionable output:

- **PRE-A** verified inherited substrate against eight findings checklist, surfaced the structural segment-handoff duplicates that triggered finding 3's substrate-as-feature determination, and required a driver fix (`_assign_close_up_ordinals` helper) that propagates forward to v2.

- **PRE-B** characterised every available AI2-THOR mutation mechanism against the §8.2.1 verification criteria, with the explicit "test all four candidates" discipline (instr §6.2.2) producing characterisation data on mechanisms v1 didn't ultimately select. The smoke test for asset replacement followed five paths in sequence; each yielded a distinct empirical signal; the substrate-property finding emerged at the fifth path.

- **PRE-C** was not run (perturbation-mechanism dependency unsatisfied), but the parameter-count envelope for L_d ∈ {1, 2, 3, 4} was pre-computed at PRE-D time and inherits to v2 as the capacity-envelope reference.

- **PRE-D** verified architectural property assertions per spec §§7.2.4 / 7.3.4 / 7.4.4 against constructed predictors for all three arms. All 11 assertions pass. The scaffold's architectural commitments are correctly realised in code.

**The scaffold is a functioning artifact.** Three predictor classes (Primary, Ablation 1, Ablation 2), arm-agnostic online trainer with skip-until-W contract and per-(item, ordinal) callback hooks, full evaluation framework, four preflight modules, and the run scripts gate execution by the same boundary conditions the instructions document specified. 72/72 tests pass at closing (21 v0 + 51 v1). What v1 built runs.

**The discipline catches problems before commitments.** Three substrate-property findings emerged during PRE-A/B execution, each surfaced by the discipline rather than by training-time failure:

1. Finding 3 segment-handoff structural duplicates (5 items × 5 loops = 25; not the 30-frame static dwell pathology the finding was designed to catch).
2. AI2-THOR 5.0.0 mutation API does not deliver the [0.05, 0.10] regime stably on inherited substrate (RandomizeMaterials at ~0.02 with failed locality and reproducibility; mechanisms 2/3/4 not available without substantial engineering).
3. `DisableObject` and `HideObject` succeed at the metadata layer but produce no render change on ProcTHOR-loaded geometry; iTHOR `FloorPlan1` verification confirms the limitation is procedural-geometry-scoped rather than API-fundamental.

Each finding surfaced through normal preflight execution, with the discipline producing structured stop-and-report outcomes rather than autonomous workarounds. The five-path smoke-test sequence in particular illustrates the value of stop-and-report: any one of the failure modes could have been silently absorbed (e.g., "DisableObject succeeded, lastActionSuccess=True, move on") had pixel-sum-diff verification not been a discipline requirement.

---

## 3. What v1 did not demonstrate

**The architectural claim is unresolved.** v1's spec set out to test whether the architectural interventions (K output queries with cross-attention, per-K-position scalar variance, stronger perturbation regime producing 0.05–0.10 cross-stage cosine drops) produce co-primary differentiation where v0's pooled readout did not. Stage A and Stage B never ran. No predictor in any of the three arms saw a single training step on Stage A frames. No evaluation matrix exists. The BCDD-identified pooled-readout coupling pathway remains hypothesised as the dominant locus of v0's coupling result; v1 produced no evidence on whether v1's architectural fix dissolves it.

What v1 closing does *not* say:

- That the v1 architecture is wrong, right, or anywhere in between. It was not tested.
- That the BCDD diagnosis of v0's coupling was wrong. The BCDD evidence (`bcdd_results.json`, v0 closing §7.1) stands; v1 simply did not advance it.
- That AI2-THOR-as-substrate is wrong for testing the claim. v1 established that AI2-THOR 5.0.0's *mutation API* doesn't deliver v1's specific perturbation regime; that's narrower than "AI2-THOR is wrong."
- That v0's substrate findings are invalidated. v0 substrate findings 1–8 carry forward; finding 3's interpretation is updated (substrate-as-feature; segment-handoff structural duplicate); finding 9 is added (ProcTHOR-scoped renderer binding under `DisableObject`/`HideObject`).
- That the v1 spec was wrong to set the [0.05, 0.10] band. The band's reasoning was sound (signal sufficiency + coupling-pathway preservation); the discipline learning is that the band should have been *empirically grounded against the substrate's actual mutation regime characterisation* during spec design, not after.

---

## 4. The new substrate finding (v1 finding 9)

v0 surfaced eight substrate findings (v0 closing §4). v1 adds a ninth and updates v0 finding 3's interpretation; both carry forward as institutional memory.

**v1 finding 9 (new). ProcTHOR-scoped renderer binding under `DisableObject` / `HideObject`.**

On `ai2thor==5.0.0` with ProcTHOR-10K seed-7 house, `DisableObject` and `HideObject` succeed at the metadata layer (`lastActionSuccess=True`, `objects[].visible=False`) but produce no change in the rendered camera output (pixel-sum-diff = 0 across pre/post frames at the same viewing position; verified by bit-identical saved PNGs). The same actions on default iTHOR `FloorPlan1` produce a visible render change (pixel-sum-diff = 935,069 on a 300×300 frame; DINOv2 cosine drop = 0.0207 at the disappeared `Cabinet`).

The limitation is procedural-geometry-scoped, not API-fundamental. AI2-THOR 5.0.0's scene loader binds ProcTHOR-described objects to renderable Unity GameObjects differently than built-in scene prefabs; `DisableObject`'s renderer-toggle path operates on the prefab pathway only. Build-version mismatch warning at house load (recommending either `ai2thor` upgrade or `procthor-10k` downgrade to `ab3cacd...`) is symptomatically related but not resolvable: 5.0.0 is the latest published `ai2thor`; the recommended `procthor-10k` revision predates 5.0.0 and binds the scene with the agent spawning at y = -38.86 (scene void), zero objects, broken NavMesh.

The mechanism is the same class as v0 findings 5 and 8: documented AI2-THOR behaviour that doesn't hold uniformly on this specific build/dataset configuration. The substrate's behaviour and the API's behaviour are two different properties.

**v0 finding 3 (updated). Substrate-as-feature reclassification: segment-handoff structural duplicates.**

PRE-A's finding-3 check (continuous motion, cos > 0.9999 between consecutive frames) flagged 25 violations on inherited substrate. CC's diagnostic established that all 25 high-cosine pairs are transit → close_up segment transitions at identical agent pose with locked heading: 5 items × 5 loops = 25, matching the count exactly. This is a structural-by-design property of the `ContinuousMotionExplorer`'s segment handoff (transit ends at the close-up entry point with heading rotated to viewing_heading; close-up begins at the same point), not the 30-frame static dwell pathology v0 finding 3 was designed to catch.

Reclassified as substrate-as-feature per the v0 STOP_REPORT precedent (§15 spec discipline). Threshold updated 0.9999 → 0.999 (single-frame structural duplicates remain visible above the new threshold should they reappear; sustained dwell would still trigger). The 25-violation pattern is single-frame duplicates, not consecutive runs; the cosine-collapse failure mode finding 3 guards against requires sustained runs that dominate the cross-attention window, which is structurally precluded by segment-handoff being a singleton phenomenon.

Inheritance to v2: any natural-substrate work using the `ContinuousMotionExplorer` inherits the 0.999 threshold and the segment-handoff-tolerant interpretation. A run-length-aware finding 3 check would separate dwell from segment-handoff without threshold recalibration; flagged as a v2 candidate refinement, not in v1 scope.

---

## 5. The PRE-B characterisation in full

PRE-B is the load-bearing v1 evidence. Recorded here at full fidelity for v2 inheritance.

### 5.1 Mechanism 1 — `RandomizeMaterials`

Only candidate with `api_success=True` end-to-end.

- Cross-stage cosine drop at perturbed items: **0.019, 0.022** (Dresser, Sofa). v1 band: [0.05, 0.10]. Under-band by ~3–5×.
- Cross-stage drop at unperturbed items (Bedroom, locality criterion < 0.015): **0.014–0.017**. Locality fails. Substrate finding 8 (cross-room visual leakage) reproduces under v1 at the same approximate magnitude as v0.
- Run-to-run reproducibility for Dresser: **|run1 − run2| = 0.0156**. Reproducibility tolerance 0.005. Fails by ~3×.

Three of four §8.2.1 criteria fail. Mechanism 1 is not a usable v1 perturbation source.

### 5.2 Mechanisms 2/3/4 — preflight deferral

- **M2 (asset replacement at fixed coordinates).** `RemoveFromScene` hangs (90s timeout, isolated probe). `SpawnTargetCircle` is not an asset spawner (`lastActionSuccess=False`, "circle failed to spawn"). `CreateObject(ArmChair, ...)` produces Unity-side `ArgumentOutOfRangeException` (ProcTHOR seed-7 prefab pool doesn't expose ArmChair). Requires `SpawnTargetCircle`/`CreateObject` capability beyond what AI2-THOR 5.0.0 exposes, or pre-built asset-pool selection engineering.

- **M3 (hand-built texture swaps).** Requires custom Unity material builder + shader path. Beyond v1 spec scope.

- **M4 (alternate ProcTHOR scene at Stage B).** Requires route-compatibility catalogue search across 10K procedural houses; structurally feasible but substantial engineering investment.

Each mechanism returned `api_success=False` with structured `deferred_to_design_chat` rationale rather than autonomous attempts at engineering workarounds.

### 5.3 Smoke test — asset replacement on Dresser

Authorised single-item characterisation probe (~30 min budget) to verify or refute CC's pre-smoke ~0.3–0.5 magnitude estimate for asset replacement. Five paths attempted in sequence; each successive blocker forced the next:

| path | outcome |
|---|---|
| `RemoveFromScene(Dresser)` | TIMEOUT at 90s |
| `SpawnTargetCircle` | `lastActionSuccess=False` |
| `CreateObject(ArmChair, ...)` | Unity ArgumentOutOfRangeException |
| `DisableObject(Dresser)` + `PlaceObjectAtPoint(Chair, Dresser.position)` | `DisableObject` ok=True render-NO-OP; `PlaceObjectAtPoint` blocked by wall |
| `HideObject(Dresser)` | ok=True render-NO-OP |

Cross-stage Dresser cosine drop measured: **0.0000** (run 1), **1.2 × 10⁻⁶** (run 2). The perturbation mechanically did not take effect. The pre-smoke ~0.3–0.5 estimate is unverified — no mechanism actually produced an asset-replacement cosine drop, so 0.0 doesn't characterise asset-replacement physics on this substrate.

### 5.4 Build alignment investigation

Following the smoke test's unexpected substrate property, build alignment was attempted to distinguish "build misconfiguration" from "API limitation."

Step 1 (capture): `ai2thor==5.0.0` (latest on PyPI; no upgrade exists); `procthor-10k` revision `43919352...` (current); recommended downgrade revision `ab3cacd0fc17754d4c080a3fd50b18395fae8647`.

Step 2 (alignment attempt — downgrade `procthor-10k`):

| measurement | working build | aligned build |
|---|---|---|
| AI2-THOR warning fires | yes | no |
| House json keys (objects/rooms/walls) | 32/4/30 | 32/4/30 (compatible) |
| Controller agent y position | ~0.9 (on floor) | **−38.86 (in scene void)** |
| `step("Pass").metadata["objects"]` count | 150 | **0** |
| `GetReachablePositions` | 1361 positions | **`lastActionSuccess=False`, 0 positions** |

The downgrade target predates AI2-THOR 5.0.0 and is incompatible with the working version. Scene loader cannot bind the older revision's geometry. Both branches of the warning's recommendation (upgrade `ai2thor`/downgrade `procthor-10k`) are empty paths.

### 5.5 iTHOR `DisableObject` verification

Final probe to refine the Reading B conclusion from "by elimination" to "directly verified." Single call on default iTHOR `FloorPlan1`:

| measurement | value |
|---|---|
| Target object | `Cabinet` |
| `DisableObject` wall time | 0.03 s (no hang) |
| `lastActionSuccess` | True |
| pixel-sum-diff | **935,069** |
| DINOv2 cosine drop | **0.0207** |
| Target visible in metadata after call | False |

`DisableObject` works as documented on iTHOR built-in geometry. The substrate-property finding (v1 finding 9 above) is ProcTHOR-scoped, not API-fundamental.

**Critical observation for v2 inheritance.** The iTHOR cosine drop (0.0207) lands in the same regime as ProcTHOR `RandomizeMaterials` (0.019–0.022) — still under the v1 band by ~3×. Switching v1 substrate from ProcTHOR to iTHOR would resolve the rendering question but not the middle-band question. The structural conclusion — that AI2-THOR 5.0.0's available mutation regimes are bimodal (under-band material variation at ~0.02 or out-of-band asset replacement requiring substantial engineering) — survives the probe. The middle band [0.05, 0.10] is not constructible across the configurations measured.

---

## 6. The substrate prerequisite finding

This is the conceptual core of v1's verdict. Stated separately from the empirical evidence because v2 inheritance depends on getting this right.

**The finding.** v1's spec set magnitude band [0.05, 0.10] to satisfy two design constraints simultaneously: **Goal A — signal sufficiency.** Stage A → Stage B input differentiation at perturbed items must exceed the bit-identical baseline by enough margin for the architectural-claim discrimination to be measurable. The v0 lesson on perturbation regime weakness (v0 closing §7.3) anchored the floor at 0.05. **Goal B — coupling-pathway preservation.** Body representations at perturbed windows must remain *near enough* to Stage A representations that Ablation 2's pooled-readout pathology can express the same v0-style coupling. A perturbation so strong that Stage B looks like an entirely new scene would mean Ablation 2 isn't reproducing v0's failure mode; it's doing a different computation on different inputs. The 0.10 ceiling was the design-time guess at where this ceiling sits.

PRE-B established that AI2-THOR 5.0.0's mutation API exposes two regimes:

- **Material-variation regime** (RandomizeMaterials, ~0.02 magnitude): satisfies Goal B (body coupling pathway preserved) but fails Goal A (signal too weak to measure architectural-claim discrimination above noise; locality and reproducibility additionally fail).
- **Asset-replacement regime** (estimated ~0.3–0.5+ when accessible; not directly measured on a working mechanism): would satisfy Goal A (strong signal) but uncertain on Goal B (Stage B may look like a different scene; Ablation 2's v0-coupling reproduction is not guaranteed at these magnitudes).

There is no measured AI2-THOR mechanism producing a stable, reproducible, well-localised cosine drop in [0.05, 0.10]. The gap is not search-failure — it is structural to how AI2-THOR exposes scene mutation. The middle band [0.05, 0.10] is not a stable operating region in vision-only AI2-THOR space because the available regimes are bimodal (weak material variation at ~0.02 or strong asset replacement at ~0.3–0.5+ when accessible). **Goal A and Goal B are in tension in vision-only AI2-THOR substrate, and the tension is what makes the middle band unconstructable on this substrate.**

This is a substantive finding, not a substrate failure. It says something about the relationship between visual-only encoders, available mutation APIs, and the architectural-claim structure v1 was testing. v0 surfaced the perturbation-weakness problem (closing §7.3) without measuring the alternative regime; v1 surfaced the bimodality and the structural tension. v2 inherits this as a constraint on what natural-substrate work would need to deliver, and as the empirical motivation for characterising the architectural claim in a substrate where stream properties can be controlled directly.

---

## 7. Spec §7.2.5 parameter erratum

v1 spec §7.2.5 estimated parameter counts (Primary at L_d=2: ~15.3M; Ablation 2: ~16.8M). Empirical counts from PRE-D scaffold construction: Primary at L_d=2 = 22,084,609; Ablation 1 = 22,084,097; Ablation 2 = 21,555,728.

Source of error: spec §7.2.5 estimated per-encoder-layer cost as ~2.1M (FFN only), omitting the ~1.05M self-attention QKVO projection block per layer. Same omission for decoder cross-attention (each decoder layer carries ~1.05M cross-attention on top of self-attention + FFN, producing 4.2M per decoder layer, not the ~3.2M the spec estimated).

Corrected envelope (from `parameter_counts_l_d_envelope.json`, all four L_d values):

| L_d | Primary | Ablation 1 | decoder body |
|---|---|---|---|
| 1 | 17,880,577 | 17,880,065 | 4,204,032 |
| 2 | 22,084,609 | 22,084,097 | 8,408,064 |
| 3 | 26,288,641 | 26,288,129 | 12,612,096 |
| 4 | 30,492,673 | 30,492,161 | 16,816,128 |

Ablation 1 − Primary delta is exactly −512 at every L_d (per-K `nn.Linear(512, 1)` = 513 params replaced by `nn.Parameter(torch.zeros(1))` = 1).

Spec text was not edited per instr §1.5 (no spec modification during execution). Carried forward to v2 as known-erratum; instr §3.5's ±10% tolerance band was staged to update targets to measured values but is also not committed (push hold). v2 inherits the corrected envelope as reference; v2 design intake decides whether to issue a retroactive spec patch.

**Discipline note.** The parameter-count check itself was load-bearing: it caught spec arithmetic error early, before scaffold-implementation drift could be misattributed to it. The discipline of "stop and report when implementation and spec disagree" (instr §1.5) is preserved; the erratum recorded for traceability, not silently absorbed.

---

## 8. Scaffold inheritance for v2

v1's scaffold is a functioning artifact and inherits forward.

**Three predictor classes** (`v1/src/predictor/`):
- `inner_pam_v1_primary.py` — full v1 architecture (K output queries with cross-attention, per-K-position scalar variance head, decoder).
- `inner_pam_v1_ablation1.py` — variance-head ablation (shared scalar log-variance, otherwise Primary).
- `inner_pam_v1_ablation2.py` — v0 InnerPAM subclassed with class-name override only.

`decoder_n_layers` is required kwarg, no None defaults. PRE-D architectural property assertions (`v1/src/preflight/pre_d_arch_property_assertions.py`) verify the architectural commitments per spec §§7.2.4 / 7.3.4 / 7.4.4. All 11 assertions PASS across all three arms.

**Arm-agnostic online trainer** (`v1/src/trainer/online_trainer_v1.py`) with skip-until-W early-trajectory contract, per-(item, ordinal) callback hook, NaN/Inf in-flight stop conditions, AdamW with grad-clip, resume-from-checkpoint.

**Evaluation framework** (`v1/src/eval/`): per-(item, ordinal) metric computation (mean drift, variance drift, per-K profiles, bit-identical stability, body-representation cosine), arm × matrix builder (JSON + CSV), percentile-anchored threshold calibration.

**Four preflight modules** (`v1/src/preflight/`): PRE-A substrate verification, PRE-B perturbation mechanism characterisation, PRE-C decoder-layer calibration, PRE-D architectural property assertions.

**Test suite.** 51 v1 unit tests covering shape contracts, per-K isolation (autograd check), loss value-correctness, trainer skip-until-W + resume, preflight pass/fail paths, end-to-end PRE-D. 72/72 tests pass at closing (51 v1 + 21 v0; no regression).

**Driver fix.** `_assign_close_up_ordinals` helper in the PRE-A driver (added during PRE-A execution; populates `close_up_ordinal` from the annotation stream when the explorer's observation dict doesn't). Carries forward to v2's annotation pipeline whenever the `ContinuousMotionExplorer` is used.

**What inherits cleanly.** The architectural property assertions, the per-(item, ordinal) callback discipline, the skip-until-W contract, the architectural-claim's three-arm contrast structure (Primary vs ablation-1 vs ablation-2), the evaluation framework's per-(item, ordinal) granularity.

**What inherits with reframing.** The verdict schema (V1A/V1B/V1C/V1D/V1E/V1P). v2's characterisation-map outcomes don't fit a yes/no/partial verdict structure; v2 needs its own verdict framework. The schema is preserved as v1 history; v2 designs new.

**What does not inherit.** The middle band [0.05, 0.10] as a substrate constraint (substrate-relative; in synthetic substrate, the magnitude axis is a sweep dimension, not a target). The three-arm comparison's specific verdict-interpretation rules (e.g., "Ablation 2 reproduces v0 coupling") that depend on substrate properties not constructible on this substrate.

---

## 9. Discipline learnings

Three substantive learnings from v1 execution, recorded for v2 inheritance as project-level discipline.

### 9.1 Substrate verification is build-coupled

v0 substrate findings 1–8 were established on a specific `ai2thor`/`procthor-10k` build configuration. v1 inherited "the substrate" without inheriting "the build pin." v1 finding 9 emerged precisely because the build's renderer-binding behaviour wasn't a property anyone had reason to verify until a mutation API needed to engage with it.

**Discipline replacement.** Substrate-property findings are build-relative until verified across builds. Inheritance of substrate verification should be accompanied by inheritance (and explicit re-verification) of build configuration. Any version-mismatch warnings emitted by the substrate environment at load time are first-class diagnostic signals, not warnings-to-suppress.

v2 inherits this as: verify substrate properties on v2's actual build configuration before treating v0/v1 substrate findings as inherited. If v2 changes encoder or environment, the verification re-runs in full.

### 9.2 Perturbation-mechanism PRE-B belongs in spec design

v1's spec §1.2 commitment 3 set the [0.05, 0.10] band without empirical substrate characterisation of available AI2-THOR mutation mechanisms. The band's reasoning was conceptually sound (signal sufficiency + coupling-pathway preservation); the empirical grounding came in PRE-B, after spec lock-in, scaffold construction, and reviewer sign-off.

The cost of this ordering: ~3 hours of preflight execution + the closing-document overhead for what is, in retrospect, a discoverable-at-design-time finding. A 30-minute substrate-characterisation probe at v1 spec design time would have surfaced the bimodality of AI2-THOR's available regimes and prompted the stack-fitting reframe (§10 below) before v1 was committed.

**Discipline replacement.** Substrate-prerequisite characterisation runs *during* spec design, not after. Any spec commitment that quantifies a substrate property (perturbation magnitudes, locality thresholds, reproducibility tolerances, signal-to-noise ratios) requires empirical grounding before the commitment is locked. The pattern: cheap exploratory characterisation → spec commitments anchored to characterisation findings → PRE-B as verification, not as discovery.

v2 inherits this as: v2 design intake includes a characterisation-during-design phase. Synthetic-substrate construction has the structural advantage that "characterisation" and "experiment" share infrastructure, but the discipline applies regardless of substrate choice.

### 9.3 Substrate-prerequisite vs architectural-claim

v1's V1P-with-content outcome is substantively informative. The substrate-prerequisite finding is a *real result*, not a substitute for an architectural verdict. It says something about the relationship between vision-only encoding, available mutation APIs, and the architectural-claim structure.

The discipline failure mode this addresses: treating V1P as "we couldn't run the experiment, no result." That framing collapses substantive substrate findings into protocol failures and loses the conceptual content. v1 closing's V1P-with-content framing preserves the substantive finding while remaining honest about the architectural claim's unresolved status.

**Discipline replacement.** When an experiment fails to produce its primary verdict due to substrate or prerequisite limitations, the closing artifact distinguishes (a) what was substantively learned about the prerequisite from (b) what remains unresolved about the primary claim. Both go into the closing record. Future-context readers should not have to derive "we found X about the substrate" from a closing that says only "we couldn't test Y."

v2 inherits this as: any future experiment's verdict schema includes substantive-prerequisite categories as first-class outcomes, not as protocol-failure subcategories.

---

## 10. The stack-fitting reframe (conceptual inheritance for v2)

This section paraphrases project-thinking that emerged during the v1 design chat on May 19, in response to the PRE-B substrate findings. The framing is the user's; the closing-prose articulation is mine and may smooth edges the user's original framing carried. Quoting fragments where the original phrasing is load-bearing.

**The reframe.** v1's substrate problem is not a substrate-selection problem (pick a better environment, pick a better mutation API). It is a *stack-fitting* problem. Four layers — environment, encoder, memory architecture, recall mechanism — co-evolved in biology and have to co-shape to each other in any synthetic instantiation. "Same sofa under different lighting" works for a biological agent not because the visual signal is sharp, but because multiple modalities are providing weakly-correlated confidence: same place, same people, same sounds, same proprioceptive cues. The visual signal is *allowed* to be ambiguous because context resolves it.

v1's Goal-A / Goal-B tension (§6 above) is what happens when we ask vision-only to support a discriminative operating point that biological vision was never asked to support alone. The middle band [0.05, 0.10] doesn't exist as a stable region in vision-only AI2-THOR space, and PRE-B suggests it doesn't exist as a stable region in *any* vision-only substrate built from natural-image-trained encoders against arbitrary environments. The stack-fitting framing predicts this: an off-the-shelf encoder applied to an unrelated off-the-shelf environment doesn't share the co-evolutionary fit that would make the middle band coherent.

**What this implies for v2.** v2 is not "v1 with a better substrate." v2 is a characterisation phase whose job is to map the encoder-environment-architecture coupling space, producing empirical constraints on what natural-substrate work would actually support testing the architectural claim (or any successor claim). The deliverable is a *characterisation map*, not an architectural verdict. The map informs subsequent decisions about encoder choice, substrate engineering, and possibly multimodal integration; it does not, by itself, decide them.

**The synthetic-substrate path.** The conversation converged on Option β (synthetic embeddings constructed directly in d=1024 space, no encoder in the loop) over Option α (synthetic images through DINOv2) and Option γ (synthetic images + designed encoder mapping). Option β isolates the memory-architecture layer of the stack by removing the encoder confound. The architecture's behaviour is then characterised against stream properties (perturbation magnitude, locality, coupling structure, manifold structure, continuity, repetition structure, noise floor, dimensionality), with the natural-substrate regime included as one anchored point in a larger property space rather than as the test point.

**Tension to engage at v2 design intake.** Option β has a conceptual seam with the stack-fitting frame itself, which v2 design must address rather than inherit as settled. The stack-fitting argument is that encoders are load-bearing because they co-evolved with everything else in the stack. Option β then removes encoders from v2's loop. The risk: v2's characterisation becomes encoder-agnostic by construction, and if the architectural claim's behaviour depends on encoder-specific manifold structure (which the stack-fitting framing implies it does — that is the framing's claim), then characterisation on Option β substrate may not transfer back to encoder-coupled naturalistic substrate. v2 design intake commits to one of: (a) Option β is justified because the architecture should work across a wide property region and the encoder selects a point within that region; (b) Option β is a first-pass characterisation, supplemented by Option γ for encoder-coupling characterisation against a narrower stream subspace; (c) something else. The closing does not pre-empt this decision; §12 carries it as an open question with elevated weight.

This is what v2 inherits as the conceptual frame, recorded here so v2 design intake reads it as project context rather than rederiving it.

**End-goal alignment.** The user's articulation, paraphrased: *the end goal is an encoder that works with reality itself.* Synthetic characterisation is the diagnostic phase, not the destination. v2 produces the map; v3+ uses the map to inform encoder design, substrate engineering, or multimodal integration; the project's long arc remains the Bush associative-trail thesis tested against continuous naturalistic experience.

---

## 11. What carries forward, what does not

**Carries forward to v2.**

- The operational discipline (`CODING_STANDARDS.md`, `research_operations_v1.md`).
- v0's eight substrate findings + v1 finding 9 (ProcTHOR-scoped renderer binding) + v0 finding 3's updated interpretation (substrate-as-feature, segment-handoff).
- v0's five SCAFFOLDING-threshold lessons + the methodological corollary.
- v0's BCDD evidence on pooled-readout coupling as the dominant locus of v0's coupling result (`bcdd_results.json`, v0 closing §7.1).
- The v1 scaffold: three predictor classes, online trainer, evaluation framework, four preflight modules, 51 v1 unit tests + 21 v0 unit tests.
- The architectural property assertions per spec §§7.2.4 / 7.3.4 / 7.4.4 (PRE-D, all PASS).
- The per-(item, ordinal) evaluation discipline.
- The parameter-count envelope for L_d ∈ {1, 2, 3, 4} (corrected, in `parameter_counts_l_d_envelope.json`).
- The stack-fitting reframe (§10 above) as conceptual frame for v2 design.
- The three v1 discipline learnings (§9.1–9.3) as project-level discipline.

**Carries forward with reframing.**

- The v1 architectural claim itself. v2 treats it as a hypothesis-to-characterise (Option β: characterise the architecture's behaviour across stream-property space), not a hypothesis-to-test (v1 framing: pass/fail against a substrate-derived perturbation regime).
- The three-arm contrast (Primary, Ablation 1, Ablation 2). v2 retains the three arms but characterises their behaviour across the property space rather than producing a single V1A/B/C verdict.
- The seven evaluation metrics from spec §10.3. Inherited as the characterisation map's measurement axes; verdict-pattern recognition (spec §10.4) does not inherit directly.

**Does not carry forward.**

- The middle band [0.05, 0.10] as a substrate constraint. Substrate-relative; in synthetic substrate the magnitude axis is a sweep dimension.
- The v1 verdict schema (V1A/B/C/D/E/P). Schema mismatched to characterisation outcomes; v2 designs new.
- The spec §1.2 commitments that quantify substrate properties (perturbation magnitudes, locality thresholds, reproducibility tolerances). v2's substrate is constructed; these become inputs to construction rather than constraints to verify.
- The PRE-A / PRE-B substrate-verification protocols *as v1 instantiated them*. Substrate verification persists as discipline (§9.1 above); the specific protocols re-derive against v2's actual substrate.
- AI2-THOR-as-substrate for v2's primary characterisation work. v2 is synthetic. Natural-substrate work resumes at v3+ informed by v2's characterisation map.

---

## 12. Open questions for v2 design intake

v2 design happens in a separate chat (per `research_operations_v1.md` §2.2). The questions below are project-level inputs to v2 design intake, not v2 design decisions.

Two of the questions below are flagged with **[elevated weight]**: they were identified during closing review as the most consequential of the set and warrant primary attention from v2 design intake before the others are engaged.

- **Stream-property axis selection.** Which axes does v2 sweep? Candidate list from v1 design chat (perturbation magnitude δ, perturbation locality, perturbation persistence, inter-item coupling, manifold structure, continuity/smoothness, repetition structure, noise floor, categorical vs continuous variation). The list is wide; v2 design decides which axes are independent vs coupled and which sweeps combine.

- **Sweep design.** How densely is each axis sampled? What's the anchor point on naturalistic regime (using v0/v1 DINOv2 statistical properties)? How does the sweep relate to the architectural claim's expected operating regime?

- **Verdict structure for characterisation outcomes.** What does "v2 succeeded" mean? Characterisation maps don't fit pass/fail verdicts. Candidate framings: regions-where-architecture-works as the deliverable; comparison of regions-where-Primary-works vs regions-where-Ablation-2-works as the architectural-claim contrast; characterisation of *failure modes* across the space as the architectural-revision input. v2 design picks the framing.

- **Negative controls.** What synthetic streams explicitly violate the architectural claim's assumptions, and what should the architecture do on them? Without negative controls, characterisation cannot distinguish "architecture works" from "architecture succeeds on streams that don't discriminate architectures." v0 closing's discipline (disambiguating-diagnostic over correlation thresholds) applies.

- **Compute envelope.** v1 was ~37 hours wall-clock for execution. v2's wider scope (nine candidate stream axes × seven measurement axes × negative controls × three arms) is structurally larger; design intake scopes the realistic envelope.

- **Encoder characterisation deferral.** v2 holds encoder constant (likely DINOv2 for continuity, possibly entirely encoder-less under Option β). v3+ characterises encoder variation against a narrower stream subspace, conditional on v2's findings. v2 design intake confirms the deferral (and the encoder-or-none decision for v2 itself).

- **[elevated weight] Anchoring synthetic to naturalistic — the Option β cross-check.** §10's "Tension to engage at v2 design intake" paragraph identifies a conceptual seam: the stack-fitting argument is that encoders are load-bearing because of co-evolution; Option β then removes encoders from v2's loop. v2 design intake commits explicitly to how Option β's characterisation grounds back to encoder-coupled naturalistic behaviour — whether (a) wide-region characterisation suffices because the encoder picks a point within the region, (b) Option β is first-pass supplemented by Option γ for encoder-coupling, or (c) something else. The closing does not pre-empt this; v2 must engage it rather than inherit Option β as settled.

- **[elevated weight] L_d selection rule continuity under the corrected parameter envelope.** v1's PRE-C selection rule was locked at "best-differentiation-within-stable" (instr §6.3.2, revision 7 during reviewer pass) on the assumption that L_d=2 was ~15M (spec §7.2.5 estimate) and the realistic L_d ∈ {1,2,3,4} range was ~11M → ~22M. The corrected envelope (§7 above) is 17.9M → 30.5M, with L_d=1 at 17.9M as the closest match to v0's empirical ~16.8M baseline. The rule itself still stands (compute is not the constraint and characterising the response across L_d remains valuable), but the rule reads differently under the new numbers: "best-differentiation-within-stable" now selects up to a 30.5M predictor against a ~16.8M v0 baseline, an 82% capacity increase. v2 design intake confirms whether best-differentiation-within-stable remains the right rule under the corrected envelope, or whether the rule should be reconsidered given the new capacity-envelope context. If v2 substrate is synthetic (Option β) and not encoder-coupled to v0, the v0 baseline comparison may carry less weight than it did at v1 design close; this is itself a question v2 should engage.

- **Empirical grounding of synthetic stream construction.** Given Option β (or whichever path the cross-check above settles on), how does v2 ground synthetic stream construction in measurable naturalistic-encoder statistical properties? v0/v1 frame collections + DINOv2 forward passes produce empirical anchors (cosine distributions within and across rooms, magnitude distributions of natural perturbations, manifold dimensionality estimates). Worth committing to during v2 design that synthetic spans naturalistic, not matches it. Distinct from the elevated-weight Option β cross-check above: that question is whether to use Option β at all; this question is how to construct streams empirically once the substrate path is settled.

---

## 13. Push hold

Push hold remains in effect on both repos through v2 design. v1 artifacts stay local until v2 scope is settled and a decision is made about which v1 outputs (substrate finding 9, parameter erratum, scaffold modules, discipline learnings) belong in any external-facing artefact.

---

*End of v1 closing document. Companion to `WEFT_INNER_PAM_v1_Spec_pass1_sections_1_to_6.md`, `WEFT_INNER_PAM_v1_Spec_pass2_sections_7_to_11.md`, `WEFT_INNER_PAM_v1_EXPERIMENT_INSTRUCTIONS.md`, and `HANDOFF.md`. v2 design begins in a separate chat with its own spec discipline.*
