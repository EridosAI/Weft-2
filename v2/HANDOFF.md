# Weft Inner PAM v2 — HANDOFF

## Current state

Repository setup complete. v2 directory structure created with empty packages for `preflight/`, `substrate/`, `protocol/`. Reference documents in `v2/docs/`. Library-import model adopted: v2 imports v1 directly from `v1.src.*`; v1 scaffold not copied.

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
- No v2 implementation work has begun.

### Deviations from the setup instructions (recorded per §7.2)

- **Class names corrected.** The instructions (README §4.1, this HANDOFF's smoke command, §6.2) used `InnerPAMv1Primary` / `InnerPAMv1Ablation1` / `InnerPAMv1Ablation2`. The actual v1 classes are `InnerPAM_v1_Primary` / `InnerPAM_v1_Ablation1` / `InnerPAM_v1_Ablation2` (with underscores), verified from `v1/src/predictor/*`. Corrected here and in the README; the §6.2 smoke passes with the corrected names (the original names would have produced a false ImportError, not a real inheritance failure). `OnlineTrainerV1` was already correct.
- **Config symbol corrected.** README example used `from v1.src.config import EMBED_DIM, W`. The actual symbols are `EMBED_DIM` and `WINDOW_W` (there is no `W`); corrected to `WINDOW_W`.
- **Interpreter.** Bare `python` is not on PATH in this WSL2 environment; `python3` (3.12.3) is. All verification/smoke commands were run with `python3`.
- **Canary invocation.** §6.1's literal "cd v1/; pytest" collects only v1's 51 tests; the 72/72 inheritance figure requires `v0/tests` + `v1/tests` from the parent root. Confirmed: 72 from parent root, 51 from `v1/` alone.
- **Verification ordering.** The file-based smokes (§6.1–6.3) were run before the git commit (they don't depend on it) so this HANDOFF could record real results; the git-based checks (§6.4–6.5) were run after the commit.
