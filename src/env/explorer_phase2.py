"""Phase 2 explorer — LivingRoom RandomizeMaterials wrapper. Implemented in session 3.

Plan (per EXPERIMENT_INSTRUCTIONS.md §8.2-§8.3):
  - Import prior repo's furniture_route_explorer.py (copy in, don't edit upstream).
  - At scene initialisation, before the explorer starts, call
    `controller.step(action='RandomizeMaterials', inRoomTypes=['LivingRoom'],
                     useTrainMaterials=True)`.
  - Record materials applied in `data/phase2_collection_metadata.json`.
  - Run the standard seed-7 route with per-frame jitter (0.2 m, 10 deg, fallback ladder).

Preflight (run via scripts/run_phase2_preflight.py) must pass before this is invoked.
"""


def __getattr__(name: str):  # pragma: no cover - intentional session-3 deferral
    raise NotImplementedError(
        "explorer_phase2 is implemented in session 3 (Phase 2 setup). "
        "See EXPERIMENT_INSTRUCTIONS.md §8 for the protocol."
    )
