# Weft Inner PAM v2 — HANDOFF

## v2 CLOSED — push hold lifted, record published 2026-05-26

**Decision:** v2 closes after Stage 1 of the recalibration phase. Stages 2–10 will not run. The grokking finding (mechanics CONFIRMED FIXED; mean head groks to 0.95–0.98 with sufficient steps; grok onset scales with manifold dimensionality D=4→~25k, D=16→~50–100k, D=128→~175k+) and the architectural-shape reading (transformer + path-prediction-NLL produces strong inter-seed convergence on stationary periodic substrate, which is function-approximator success rather than associative-memory phenomenology) are sufficient closing evidence. Further compute on this architecture is not justified for the Weft research line.

**Record state:**
- All Stage 1 commits published to `origin/main` (https://github.com/EridosAI/Weft-2)
- Published v0/v1/v2 record HEAD: `7892076` (this closing-marker commit is pushed on top as the final commit)
- Frozen-tree intact (`git diff 58e91d7 HEAD -- v0 v1 shared` empty)
- Canaries pass: **131 pytest** (121 inherited + 10 Stage 1 mean-head-plateau detector tests) + 11 PRE-D arch assertions
- Remote is a NEW repo (`EridosAI/Weft-2`); the pre-existing `EridosAI/Weft` is an unrelated V-JEPA/PAM-Tiered project (no shared history) and was NOT used. The new repo's init commit (LICENSE + .gitignore) was merged in (`--allow-unrelated-histories`); LICENSE retained, .gitignore kept ours.

**What did not happen, deliberately:**
- No `V2_TRAINING_STEPS` lock written (still the invalidated 10000; never written because Stage 1 STOP fired at C4 D=128)
- Stage 2 (sanity battery at lock) and Stages 3–10 of the recalibration cascade
- Phase 1 main effects, Phase 2, cross-encoder generalisation — all out of v2 scope from the start

**Closing document:** `WEFT_INNER_PAM_v2_CLOSING.md` (drafted in experiment chat; lands in the repo as a separate follow-up commit, not in this push batch).

---

## RECALIBRATION STAGE 1 COMPLETE — **STOP fired** (C4 D=128); lock-value decision is experiment-chat work

**Date:** 2026-05-26. **Stage 1 = multi-cell grokking detection** (recalibration instr §3). 7 cells × 3 predictor seeds = **21 grok-curve runs to 200k steps**, all `rc=0`. Ran ~12.1 h wall at 2× local on the RTX 4080 Super (23.9 h sequential-equivalent). Results: `results/recalibration/stage1/` (21 `grok_curve_{cell}_seed{N}.json`, 7 `_aggregate.json`, `lock_decision.json`). Code commit `02f9222`; results commit follows this HANDOFF edit.

### Headline: the broken-mechanics root cause is CONFIRMED FIXED
The mean head now **learns** — `cos(mean,target)` rises in a sharp grokking transition from ~0 to **0.95–0.98 on 6 of 7 cells** (at the broken `V2_TRAINING_STEPS=10000` it was ~0, below the trivial 0.56 baseline). 10k steps was simply far too short. **Grok onset is cleanly dimensionality-coupled:** D=4 onset ~25–50k, D=16 ~50–100k, **D=128 ~175k and still climbing at the 200k ceiling.**

### Per-cell results (mean-head-aware lock criterion, instr §3.3)
| cell | L_d | D | continuity | mag | lock_candidate / seed | cell max | max_cos (n=3) | within-seed spread |
|---|---|---|---|---|---|---|---|---|
| C1 (anchor) | 1 | 16 | c39 (C≈0.42) | 0 | 175k/175k/175k | 175k | 0.95–0.96 | 1.0× |
| C2 | 2 | 16 | c39 | 0 | 125k/100k/150k | 150k | 0.97 | 1.5× |
| C3 | 4 | 16 | c39 | 0 | 150k/175k/175k | 175k | 0.965 | 1.17× |
| **C4** | 1 | **128** | c39 (forced **C≈1.0**) | 0 | **200k/200k/200k** | **200k** | **0.82–0.85** | 1.0× |
| C5 | 1 | 16 | c39 | 0.9 | 175k/175k/150k | 175k | 0.955 | 1.17× |
| C6 | 1 | 4 | c5 (C≈0.42) | 0 | 50k/50k/75k | 75k | 0.979 | 1.5× |
| C7 | 1 | 16 | c59 (C≈0.87) | 0 | 100k/100k/100k | 100k | 0.965 | 1.0× |

### STOP condition (instr §3.5) — why Stage 2 is NOT entered
- **C4 (D=128), all three seeds `lock_step_candidate = 200000 > 175000`.** C4 is the `D == P//2` forced-continuity boundary cell (§3.1 caveat; measured C≈1.0). It groks very late and **never plateaus within the 200k budget** — seed0 stays cos≈0.01 through 150k, then 0.44→**0.82** at 200k; seed1 0.56→0.85; all still rising at the ceiling.
- Derived overflow: `max(lock_step_candidate_max)×1.1 = 200000×1.1 = 220000 > 200000` budget ceiling characterised by the confirmatory test (PHASE1_PROGRESS §16). **`lock_steps_proposed = null`.**
- **All other STOP guards clean:** within-cell across-seed spread ≤1.5× (n=3 seed-instability guard never tripped — seed convergence is tight, e.g. C1/C4/C7 identical across seeds); inter-cell `lock_step_candidate_max` spread = **2.67×** (C6 75k → C4 200k), under the 4× threshold. No not-grokked-within-budget cell (C4 clears trivial+0.10; it is grokked-but-not-plateaued).

Per §3.5: **CC has NOT proceeded to Stage 2, has NOT written the `V2_TRAINING_STEPS` config lock** (`results/pre_a/v2_training_steps.json` still holds the invalidated 10000; `--write-lock` refuses while a STOP stands). The lock value is decided by the experiment chat with these curves in hand.

### Decision needed from the experiment chat (the C4 question)
Options CC sees (presented, not chosen):
1. **Exclude C4** as the flagged `D==P//2` forced-continuity boundary special case (§3.1 already caveats it; it is not a "normal" cell). Remaining cells' max = **175k** (C1/C3/C5) → `175000×1.1 = 192500` → round-to-clean = **`V2_TRAINING_STEPS = 200000`**. Clean, in-budget. *(Note: if the downstream pilot's dim-depth axis reaches D=128 at L_d=1, the slow-grok behaviour resurfaces there — worth weighing.)*
2. **Extend the budget for C4** (>200k) to find its true plateau — exceeds the 200k ceiling and needs a fresh budget characterisation.
3. **Lock at 200000 and accept C4 as "barely grokked at the ceiling"** — but C4 has not reached post-asymptotic, violating the lock criterion for that cell.

### Fork 2 decision input (instr §3.8) — measured per-arm cost + cascade estimate
Measured at the 200k grok-run length (RTX 4080 Super; cost scales ~linearly in steps): **L_d=1 ≈ 58.8 min/arm, L_d=2 ≈ 75.7, L_d=4 ≈ 107.4** (peak VRAM 1.65–1.85 GB/arm; 2× concurrency safe). Per 1k steps: L_d=1 ≈ 0.294 min, L_d=2 ≈ 0.378, L_d=4 ≈ 0.537.

Cascade ballpark **at a 200k lock** (2× local; round figures, experiment chat refines):
| stage | arm-runs | ~2× local |
|---|---|---|
| 3 (unviable-egg, 30: 20×L1+10×L4) | 30 | ~19 h |
| 4 (PRE-D1a L_d-specific, §6 = 60) | 60 | ~40 h |
| 6 (PRE-D2, 200×L2) | 200 | ~5.3 d |
| 7 (controls, 120) | 120 | ~3.4 d |
| 8 (baseline-var, 40×L1) | 40 | ~20 h |
| 9 (pilot L_d=1, 240×L1) | 240 | ~4.9 d |
| **through Stage 9** | ~690 | **~17 days** |
| 10 (capacity, 480) | 480 | ~15 days |

**Implication:** a ~200k lock is **20× the original 10k assumption**, so the cascade is *weeks* at 2× local (dominated by Stages 6/9/10). This makes the **vast.ai case strong** (nominal 8× → through Stage 9 ≈ 2 days, ≈ $160 at $3.20/hr; +Stage 10 ≈ +$150) — *or* motivates reconsidering whether downstream stages need the full 200k (most downstream cells are L_d=1 mid configs that grok by ~175k). **CC does not make the Fork 2 call;** it continues only when the experiment chat sets the lock and compute strategy. If vast.ai is chosen, the §3.9 cross-hardware smoke gates the rental session.

---

## PHASE 0 COMPLETE — hands off to the Phase 0.5 design chat

**Outcome: all seven Phase 0 sub-phases completed; no unresolved STOP.** PRE-A (0.1) · PRE-C (0.2) · PRE-B (0.3) · PRE-D1c (0.4) · PRE-D1a (0.5) · PRE-E (0.6) · PRE-D2 (0.7). Commits: `60c8680, dbe6e34, 65fecf3, c1c9d16, ca3972d, de80e76, 63eb9c5, b2d0ec7`. Arm-runs consumed: PRE-A 1 (smoke), PRE-C 0, PRE-B 0, PRE-D1c 0, PRE-D1a 40, PRE-E 0, PRE-D2 200 = **~241**.

**Regression canaries — all 5 green:** v0 21/21 + v1 51/51 + v2 37/37 = **109 pytest PASS**; v1 PRE-D 11 assertions PASS (`run_pre_d_arch_assertions.py --decoder-n-layers 2`); v0/v1/shared unmodified (`git diff 58e91d7 HEAD -- v0 v1 shared` empty); push hold preserved (no `git push`). Working tree clean.

**Calibrated SCAFFOLDING:** `results/pre_e/scaffolding_calibration.json` (runtime source via `config.load_calibrated_thresholds()`; `config.py` constants synced). τ_W per head μ≈0.0613 / σ≈1.68e-7. `V2_TRAINING_STEPS=10000` (lock file; PRE-A-calibrated, PRE-D1a-confirmed adequate). **Worked-example region:** `results/pre_b/worked_example_region.json`.

### The four Phase 0.5 inputs (§10)

1. **Per-arm cost** (PRE-D1a → `results/pre_d1a/endpoint_stability_report.json`): L_d=1 ≈95s, L_d=4 ≈165s, mean 130s on RTX 4080 Super. Phase 1 n=10 envelope (~5220 arm-runs) ≈ **~8 days serial** — the empirical wall-clock for the density/n trade-off.
2. **n=10 vs n=20 CI** (PRE-D2 → `results/pre_d2/n_validation_report.json`): **variance-limited** — n=20 gives NO resolution gain over n=10 (band-residence 5/10→6/10); ~50-60% of points band-resident at both n because per-rep Diff_μ CV is high (2.09 overall) with several medians near the threshold. **n=20 not justified over n=10**, but per-point reliability at L_d=2 is fragile (a §9.8 reliability-over-coverage / closing concern). Does not fit the spec's three pre-committed framings — a fourth reading.
3. **Corner reachability** (PRE-D1c → `results/pre_d1c/corner_reachability_assessment.md`): magnitude/continuity/manifold-dim **near-extreme-low** → §3.5 corner-avoidance does NOT hold on the linear reading; manifold-dim log-scale reads central → **Phase 0.5 must commit the dimensionality sweep scale** before finalising this. Phase 0.5 weighs corner-sampled crosses vs §9.8.
4. **Worked-example outcome / W4** (PRE-B → `results/pre_b/worked_example_region.json`): magnitude 0.017 (genuinely bimodal, item-specific), locality 0.836, continuity 0.077, manifold-dim global≈13.75, repetition period 360 / coverage 1.0. All 4 determinable axes within §2.4 range → **no W4**. **Repetition W4 deferred to Phase 0.5** (§2.4 range ambiguous, ask #4) — design chat must define the repetition axis's W4 range (which sub-property; numerical bounds).

### Next immediate action + Phase 1 hard-gate

Phase 0.5 design chat reviews the four inputs above and commits: density (3/4/5 crosses; 5 vs fewer per-axis values; axis count), sample size (n=10 retained per Input 2, or n=20 with density reduction per the §9.1 hard floor), corner-sampling (retain §3.5 avoidance or add corner crosses per Input 3), the **dimensionality sweep scale** (Input 3 caveat), and the **repetition W4 range** (Input 4 deferral). **Phase 1 HARD-GATE (adversarial-review Finding 4):** CC does not initiate any Phase 1 work until the Phase 0.5 chat produces the Phase 1 CC instructions document — there is no Phase 1 instructions doc in scope; attempting Phase 1 is itself a STOP.

---

## Current state (per-sub-phase detail)

**Phase 0 sub-phase 0.1 (V2-PRE-A) COMPLETE — no STOP triggers.** Construction primitives (§5.1-5.3), property measurement (§4), measurement protocol (§6.2), grid mapping (§6.3), `config.py` (ARCHITECTURE/SCAFFOLDING), unit tests, and `scripts/run_pre_a.py` implemented. Library-import model: v2 imports v1 from `v1.src.*` (v1 frozen, not copied).

Sanity sweep: all 5 spec axes 3/3 within 10% tolerance (magnitude, locality, continuity, repetition period, manifold dim) + fidelity supplementary 3/3. Stream contract PASS (L2 norms 1±1e-5). Arch-forward smoke PASS (3 arms finite). `V2_TRAINING_STEPS = 10000` (loss-plateau-calibrated; lock-file-gated via `config.get_v2_training_steps()`). Outputs in `results/pre_a/`; `data/embedding_U.npy` persisted (spec §5.5).

**Sub-phase 0.2 (V2-PRE-C) COMPLETE — no STOP triggers.** 11/11 architectural assertions PASS on the v2 substrate (4 Primary + 4 Ablation 1 + 3 Ablation 2; param counts match v1 exactly: 22,084,609 / 22,084,097 / 21,555,728) + v2 synthetic-window forward smoke OK. Wrapper: `src/preflight/pre_c_arch_assertions_v2_substrate.py`; output `results/pre_c/arch_assertions_v2_substrate.json`. Note: v1 has no `run_assertions(predictor, stream)` — it exposes `assert_{primary,ablation1,ablation2}(model, device=...)` with internally-generated random windows; the 11 assertions are architecture-level (input-independent), so PRE-C runs them unchanged AND adds a genuine v2-synthetic-window forward smoke.

**Sub-phase 0.3 (V2-PRE-B) COMPLETE — no STOP, no W4.** Cache `v0/data/phase2_embeddings/embeddings.npy` (65k×1024, L2-normed, correct version). Trajectory = 5 items × **181 loops** (Stage A clean 0–30; Stage B `livingroom_retexture` 31–180) — NOT the "5 loops" §7.3 sketch. Design-chat decisions: trajectory unit = whole stream with within-trajectory distributions; reference estimated by annotation alignment `(item, close_up_ordinal)` over Stage-A loops; repetition W4 deferred to Phase 0.5 (§2.4 range ambiguous). Verification gates passed: close-up SSM within-item 0.984 vs cross-item 0.494; Stage-A reference clusters ~1.0 (coverage 1.0, confirms §2.5). Worked-example region: magnitude median 0.017 (bimodal, item-specific), locality 0.836, continuity 0.077 (smooth), manifold-dim global≈13.75, repetition period 360 / fidelity 0.974 / coverage 1.0. All 4 determinable axes within §2.4 range (no W4). Outputs: `results/pre_b/{worked_example_region.json, segmentation_verification.json, coarse_ssm_650.png}`. CORRECTION (per PRE-E): continuity's multimodal flag is GENUINE, not a false positive — BIC improvement 100.56 (n=181) → two real tight clusters at 0.0759/0.0778 (likely Stage-A vs Stage-B loops); calibrated BIC threshold 0.0 confirms it (magnitude BIC 4184 multimodal; locality BIC −14 unimodal). Methodology note for closing: BIC detects statistical multimodality even when component medians are practically close.

**Sub-phase 0.4 (V2-PRE-D1c) COMPLETE** (`c1c9d16`): corner reachability — magnitude/continuity/dim near-extreme-low → §3.5 corner-avoidance doesn't hold (linear); dim log-scale reads central (Phase 0.5 commits dim sweep scale). Phase 0.5 Input 3.

**Sub-phase 0.5 (V2-PRE-D1a) COMPLETE — no STOP.** 40 arm-runs (20 endpoint + 20 bit-identical baseline, Primary). All 40 **stable** (0 unstable, 0 divergent). Plateau caveat checked: **V2_TRAINING_STEPS=10000 adequate** — 5/20 endpoints mildly still-descending (max 9%, mostly L_d=1), 25% (not >25% surface threshold). NOTE: initial plateau/stability flags were buggy on negative NLL (">0" guard + multiplicative tolerance); fixed in `assess_trajectory` + regression-tested; reports recomputed from stored trajectories. **Per-arm cost (Input 1):** L_d=1 ≈95s, L_d=4 ≈165s (mean 130s). **Baseline (for τ_W):** Diff_μ median 0.0277 (IQR 0.061), Diff_σ median 1.29e-7 (IQR 3.05e-7). Outputs: `results/pre_d1a/{endpoint_stability_report.json, bit_identical_baseline.json}`.

**Sub-phase 0.6 (V2-PRE-E) COMPLETE — no STOP.** SCAFFOLDING thresholds calibrated (analytical; PRE-A primitives regenerated, no training): τ_R=0.552, repetition-noise-floor=0.395, τ_L=0.024, τ_pert=0.178, repetition-coverage-threshold=0.5 (PRE-B 1.0 clears), BIC=0.0 (single-mode refs max −3.09; BIC model-selection boundary), local-PCA-window=128 (recovers D=16.8, cv≈0), τ_W per-head μ≈0.0613 / σ≈1.68e-7 (one-sample Wilcoxon p=0.049 vs the D1a baseline). Source of truth: `results/pre_e/scaffolding_calibration.json` (read at runtime by `config.load_calibrated_thresholds()`; config.py constants synced). PRE-B used placeholders but its headline (medians, no W4, multimodal flags) is unchanged by the calibrated values (BIC 0 vs 10 identical for these BICs; coverage robust); local_D and detected-perturbed count would shift on a re-run (Phase 1/closing refinement).

Next sub-phase: **0.7 — V2-PRE-D2** (n=10 vs n=20 CI validation, 200 arm-runs, ~7hr): 10 sweep points (each axis at its 2nd & 4th value with others at midpoint; §11.3 first-10-by-seed subsample) × n=20 × Primary at **L_d=2 intermediate** (§11.6; CI extrapolation to L_d_main {1,4} approximate). Classify per head per point as discriminably-working / discriminably-non-working / band-resident vs (baseline + τ_W) at n=10 and n=20. Then Phase 0 HANDOFF (4 Phase 0.5 inputs).

Earlier setup (commit `8ebf068`): v2 directory + reference docs in `v2/docs/`.

## Environment

- Python: 3.12.3 (invoked as `python3`; bare `python` is not on PATH in this WSL2 environment — see Risks)
- Platform: Windows filesystem (`C:\Users\Jason\Desktop\Eridos\Weft 2\`) via WSL2 (Linux); torch 2.10.0+cu128
- Dependencies: parent-root `requirements.txt`; no v2-local dependency file
- Repository: subdirectory of parent `Weft 2/` repo (committed at `58e91d7` pre-v2-setup); no v2-local `.git`

## What runs

- v1 inheritance canary: **72/72 tests pass** when run from the parent repo root over both test trees: `python3 -m pytest v0/tests v1/tests` (21 v0 + 51 v1). Note: invoking `pytest` from `v1/` alone collects only v1's 51 tests; the 72 figure requires including `v0/tests`. Do not invoke from `v2/`.
- v2-specific imports (library model), from parent repo root:
  `python3 -c "from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary; from v1.src.trainer.online_trainer_v1 import OnlineTrainerV1; from v1.src.config import EMBED_DIM"` → succeeds.
- v2 empty packages importable from parent repo root:
  `python3 -c "import v2.src.preflight; import v2.src.substrate; import v2.src.protocol"` → succeeds (PEP 420 namespace package; `v2/` has no `__init__.py`).
- No v2-specific code yet — `src/preflight/`, `src/substrate/`, `src/protocol/` are empty packages.

## Working-tree state

Clean at parent repo root after setup. v2/ committed as a single setup commit on top of `58e91d7`; nothing outside `v2/` modified. Push hold in effect: local commit only, no remote.

## Phase boundary

End of repository setup. Next phase: **Phase 0 — Pre-sweep verification** (V2-PRE-A through V2-PRE-E per v2 spec §9.2). Phase 0 CC instructions drafted in a separate session.

## Immediate next action for Phase 0 chat

1. Confirm repository setup is complete and v1 canary passes from v1/.
2. Receive Phase 0 CC instructions (drafted separately).
3. Begin V2-PRE-A (construction-primitive sanity) implementation in `v2/src/preflight/`.

## Risks and flags

- Library-import model: any change to v1's source modules now affects v2 silently. v1 is treated as frozen per v2 spec §1.6; this is the discipline that protects the inheritance. v2 closing reviews whether the discipline held.
- CODING_STANDARDS.md not copied (stale, references PAM_Tiered_v0, Grok/Cursor orchestration — a different project). v2 inherits coding patterns from v1's codebase implicitly. If a Phase 0 coding question surfaces that v1 doesn't answer, write the rule then.
- `v1/src/env/`, `v1/src/evaluation/`, `v1/src/training/` exist alongside `v1/src/eval/`, `v1/src/trainer/`. CC verified at pre-flight that the predictor scaffold and scripts do not import from these; v2 can ignore them. If Phase 0 work surfaces a need, flag it.
- `bcdd_results.json` source: `v0/results/v1_design/bcdd_results.json` (not the `bcdd_results_failed_preflight.json` sibling).
- Canary 3 (v1 PRE-D assertions) requires the bypass arg — v1 PRE-C lock file was never created: `python3 v1/scripts/run_pre_d_arch_assertions.py --decoder-n-layers 2`. The 11 assertions also pass via `pytest v1/tests/test_preflight.py`. PRE-A canary status: 90 pytest PASS (21 v0 + 51 v1 + 18 v2), 11 PRE-D assertions PASS, v0/v1/shared unmodified, push hold held.
- Substrate-design notes (PRE-A): manifold-dim axis verified via `D_global` (`D_local` underestimates for a 1-D trajectory — local-PCA window calibrated at PRE-E); locality construction value is the ground-truth §4.2 measurement (protocol's estimated reference recovers it). A thin `src/substrate/stream_builder.py` composition helper was added (no new substrate mechanism).

### Deviations from the setup instructions (recorded per §7.2)

- **Class names corrected.** The instructions (README §4.1, this HANDOFF's smoke command, §6.2) used `InnerPAMv1Primary` / `InnerPAMv1Ablation1` / `InnerPAMv1Ablation2`. The actual v1 classes are `InnerPAM_v1_Primary` / `InnerPAM_v1_Ablation1` / `InnerPAM_v1_Ablation2` (with underscores), verified from `v1/src/predictor/*`. Corrected here and in the README; the §6.2 smoke passes with the corrected names (the original names would have produced a false ImportError, not a real inheritance failure). `OnlineTrainerV1` was already correct.
- **Config symbol corrected.** README example used `from v1.src.config import EMBED_DIM, W`. The actual symbols are `EMBED_DIM` and `WINDOW_W` (there is no `W`); corrected to `WINDOW_W`.
- **Interpreter.** Bare `python` is not on PATH in this WSL2 environment; `python3` (3.12.3) is. All verification/smoke commands were run with `python3`.
- **Canary invocation.** §6.1's literal "cd v1/; pytest" collects only v1's 51 tests; the 72/72 inheritance figure requires `v0/tests` + `v1/tests` from the parent root. Confirmed: 72 from parent root, 51 from `v1/` alone.
- **Verification ordering.** The file-based smokes (§6.1–6.3) were run before the git commit (they don't depend on it) so this HANDOFF could record real results; the git-based checks (§6.4–6.5) were run after the commit.
