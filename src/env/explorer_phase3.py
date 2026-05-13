"""Phase 3 explorer — Phase 2 retexture + stronger perturbation. Implemented in session 4.

Plan (per EXPERIMENT_INSTRUCTIONS.md §9.2-§9.3):
  - Preferred: Television asset replacement-in-place
    (RemoveFromScene + PlaceObjectAtPoint or equivalent).
  - Fallback (if preflight rejects asset replacement): full-house RandomizeMaterials.
  - Apply Phase 2's LivingRoom retexture first, then Phase 3's mechanism.

Preflight (run via scripts/run_phase3_preflight.py) selects the mechanism.
"""


def __getattr__(name: str):  # pragma: no cover - intentional session-4 deferral
    raise NotImplementedError(
        "explorer_phase3 is implemented in session 4 (Phase 3 setup). "
        "See EXPERIMENT_INSTRUCTIONS.md §9 for the protocol."
    )
