"""Phase 2 explorer wrapper — Stage A baseline + Stage B LivingRoom retexture.

Wraps `ContinuousMotionEnv` (session 4) with a per-loop `RandomizeMaterials`
trigger that fires only for loops with `loop_index >= perturbation_start_loop`.

Per the curriculum framing recorded 2026-05-14 in instructions §1.4 and §8:
  - Loops 1..(perturbation_start_loop - 1) = **Stage A baseline**.
    No `RandomizeMaterials` is called. The route runs against the unperturbed
    seed-7 house. Frame annotations carry `perturbation_active = False`.
  - Loops perturbation_start_loop..end = **Stage B perturbed**.
    At the start of each loop, the wrapper fires
    `controller.step(action="RandomizeMaterials",
                     inRoomTypes=["LivingRoom"],
                     useTrainMaterials=True)`,
    producing fresh LivingRoom textures (Dresser + Sofa) for that loop.
    Frame annotations carry `perturbation_active = True`.

No agent-pose jitter, no furniture-position jitter, no per-frame perturbation
of any kind. Variation comes from the phase structure itself; see §1.5.

The preflight in `scripts/run_phase2_preflight.py` must pass before this
wrapper is used in a collection run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.env.continuous_motion_env import ContinuousMotionEnv


class Phase2RetextureEnv:
    """Wrap ContinuousMotionEnv with a per-loop LivingRoom RandomizeMaterials trigger.

    Public API matches ContinuousMotionEnv (delegates `next_frame`, `last_observation`,
    `episode_boundary_flag`, `current_room_name`, `explorer_stats`, `close`) and
    additionally exposes:

      - `perturbation_active_for_current_frame() -> bool`
      - `materials_by_loop` (dict[int, Any]): records the controller's last-action
        return for each loop where RandomizeMaterials fired.
    """

    _ACTION = "RandomizeMaterials"
    _ROOM_TYPES = ("LivingRoom",)

    def __init__(
        self,
        env: ContinuousMotionEnv,
        perturbation_start_loop: int,
    ) -> None:
        if perturbation_start_loop < 1:
            raise ValueError(
                f"perturbation_start_loop must be >= 1, got {perturbation_start_loop}"
            )
        self._env = env
        self._perturbation_start_loop = int(perturbation_start_loop)
        self._last_loop_seen: int = -1
        # Per-loop record of the metadata returned by RandomizeMaterials. The
        # controller's actionReturn shape is version-dependent; we store whatever
        # comes back for the audit trail.
        self.materials_by_loop: Dict[int, Any] = {}
        # Whether the current frame belongs to a perturbed loop (loop_index >= start).
        self._current_perturbation_active: bool = False

    # ---- delegated accessors ------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Delegate everything we don't override (current_room_name, explorer_stats,
        # episode_boundary_flag, close, ...).
        return getattr(self._env, name)

    @property
    def last_observation(self) -> Dict[str, Any]:
        return self._env.last_observation

    @property
    def episode_boundary_flag(self) -> bool:
        return self._env.episode_boundary_flag

    # ---- main step --------------------------------------------------------

    def next_frame(self):
        """Render the next frame.

        Before returning, check whether the loop_index for the next frame has
        advanced past the previous one; if so, and if the new loop is at or
        after perturbation_start_loop, fire RandomizeMaterials for that loop.

        Implementation note: the explorer's loop_index advances *after* the
        boundary frame is emitted (i.e. the first frame of a new loop carries
        the new loop_index). We trigger the per-loop RandomizeMaterials when
        we observe a fresh loop_index in the just-emitted frame's observation.
        """
        frame = self._env.next_frame()
        obs = self._env.last_observation
        loop_idx = int(obs.get("loop_index", -1))

        # Decide perturbation_active for this frame based on its loop_index.
        # (Stage A frames carry False; Stage B frames carry True.)
        self._current_perturbation_active = bool(
            loop_idx >= self._perturbation_start_loop
        )

        # When a new loop_index appears and that loop is Stage B, fire the
        # randomizer once for the loop. We deliberately fire *after* the first
        # frame of the loop is rendered: AI2-THOR materials change applies
        # from the next render onward, so the new textures will be visible
        # for the rest of that loop's frames. This deliberately accepts that
        # the boundary frame of each Stage B loop is rendered with the
        # *previous* loop's materials — a one-frame artefact at loop boundaries
        # that downstream analysis can filter by checking the loop_boundary_flag.
        if loop_idx != self._last_loop_seen:
            if loop_idx >= self._perturbation_start_loop:
                self._fire_randomize_materials(loop_idx)
            self._last_loop_seen = loop_idx

        return frame

    def perturbation_active_for_current_frame(self) -> bool:
        return self._current_perturbation_active

    # ---- RandomizeMaterials -------------------------------------------------

    def _fire_randomize_materials(self, loop_idx: int) -> None:
        controller = self._env._controller  # we deliberately reach into the env
        event = controller.step(
            action=self._ACTION,
            inRoomTypes=list(self._ROOM_TYPES),
            useTrainMaterials=True,
        )
        meta = event.metadata if hasattr(event, "metadata") else {}
        if not meta.get("lastActionSuccess", False):
            raise RuntimeError(
                f"RandomizeMaterials failed at loop_index={loop_idx}: "
                f"errorMessage={meta.get('errorMessage', '?')}"
            )
        self.materials_by_loop[int(loop_idx)] = {
            "actionReturn": meta.get("actionReturn"),
            "lastActionSuccess": True,
        }


__all__ = ["Phase2RetextureEnv"]
