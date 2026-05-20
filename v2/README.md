# Weft Inner PAM v2

Synthetic-substrate characterisation methodology for the predictor architecture introduced in v1.

**Status.** Repository setup complete; pre-execution. Phase 0 (V2-PRE-A through V2-PRE-E) has not yet been run.

**Spec.** See `docs/WEFT_INNER_PAM_v2_Spec_pass1_sections_1_to_3.md` and `docs/WEFT_INNER_PAM_v2_Spec_pass2_sections_4_to_9.md`.

**Inheritance from v1 — library-import model.** v2 does not copy the v1 scaffold. v1's predictor classes, online trainer, and evaluation framework live at `v1/src/*` and are imported by v2 as a library:

    from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
    from v1.src.trainer.online_trainer_v1 import OnlineTrainerV1
    from v1.src.config import EMBED_DIM, WINDOW_W
    # etc.

v1 is the frozen reference baseline; v2 uses it unchanged per v2 spec §1.6. The inheritance canary is v1's own test suite (72/72 at `58e91d7`), invoked from `v1/`, not from `v2/`.

**v2-specific work.** Lives in `src/preflight/` (V2-PRE modules), `src/substrate/` (synthetic stream construction primitives), and `src/protocol/` (measurement protocol). Tests for v2 code live in `tests/`.

**Repository structure.** v2 is a subdirectory of the parent `Weft 2/` repository, which also contains `v0/`, `v1/`, and `shared/`. The parent repo's git history covers all subdirectories; v2 does not have its own `.git`. Dependencies are inherited from the parent-root `requirements.txt`; no v2-local dependency file.

**Push hold.** No commits pushed to remote until v2 closing per spec §9.7. Local commits within the parent repo are permitted per the v1-closing precedent.
