# Weft Inner PAM v2 — HANDOFF

## Current state

**Phase 0 sub-phase 0.1 (V2-PRE-A) COMPLETE — no STOP triggers.** Construction primitives (§5.1-5.3), property measurement (§4), measurement protocol (§6.2), grid mapping (§6.3), `config.py` (ARCHITECTURE/SCAFFOLDING), unit tests, and `scripts/run_pre_a.py` implemented. Library-import model: v2 imports v1 from `v1.src.*` (v1 frozen, not copied).

Sanity sweep: all 5 spec axes 3/3 within 10% tolerance (magnitude, locality, continuity, repetition period, manifold dim) + fidelity supplementary 3/3. Stream contract PASS (L2 norms 1±1e-5). Arch-forward smoke PASS (3 arms finite). `V2_TRAINING_STEPS = 10000` (loss-plateau-calibrated; lock-file-gated via `config.get_v2_training_steps()`). Outputs in `results/pre_a/`; `data/embedding_U.npy` persisted (spec §5.5).

**Sub-phase 0.2 (V2-PRE-C) COMPLETE — no STOP triggers.** 11/11 architectural assertions PASS on the v2 substrate (4 Primary + 4 Ablation 1 + 3 Ablation 2; param counts match v1 exactly: 22,084,609 / 22,084,097 / 21,555,728) + v2 synthetic-window forward smoke OK. Wrapper: `src/preflight/pre_c_arch_assertions_v2_substrate.py`; output `results/pre_c/arch_assertions_v2_substrate.json`. Note: v1 has no `run_assertions(predictor, stream)` — it exposes `assert_{primary,ablation1,ablation2}(model, device=...)` with internally-generated random windows; the 11 assertions are architecture-level (input-independent), so PRE-C runs them unchanged AND adds a genuine v2-synthetic-window forward smoke.

**Sub-phase 0.3 (V2-PRE-B) COMPLETE — no STOP, no W4.** Cache `v0/data/phase2_embeddings/embeddings.npy` (65k×1024, L2-normed, correct version). Trajectory = 5 items × **181 loops** (Stage A clean 0–30; Stage B `livingroom_retexture` 31–180) — NOT the "5 loops" §7.3 sketch. Design-chat decisions: trajectory unit = whole stream with within-trajectory distributions; reference estimated by annotation alignment `(item, close_up_ordinal)` over Stage-A loops; repetition W4 deferred to Phase 0.5 (§2.4 range ambiguous). Verification gates passed: close-up SSM within-item 0.984 vs cross-item 0.494; Stage-A reference clusters ~1.0 (coverage 1.0, confirms §2.5). Worked-example region: magnitude median 0.017 (bimodal, item-specific), locality 0.836, continuity 0.077 (smooth), manifold-dim global≈13.75, repetition period 360 / fidelity 0.974 / coverage 1.0. All 4 determinable axes within §2.4 range (no W4). Outputs: `results/pre_b/{worked_example_region.json, segmentation_verification.json, coarse_ssm_650.png}`. PRE-E note: placeholder BIC threshold (10) gives a near-trivial "multimodal" on continuity (components 0.0759 vs 0.0778) → calibrate higher.

Next sub-phase: **0.4 — V2-PRE-D1c** (corner reachability, analytical, 0 arm-runs): inspect PRE-B per-axis medians/IQRs vs §2.4 ranges; characterise per-axis position (central / off-center / near-extreme); 5D corner-reachability reading → `results/pre_d1c/corner_reachability_assessment.md`. Then 0.5 PRE-D1a (40 arm-runs; heed the V2_TRAINING_STEPS=10000 plateau caveat — flag not-plateaued endpoints), 0.6 PRE-E, 0.7 PRE-D2 (200 arm-runs), Phase 0 HANDOFF.

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
