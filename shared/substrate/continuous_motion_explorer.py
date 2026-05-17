"""Continuous-motion explorer (v0 successor to FurnitureRouteExplorer).

The session-3 reviewer note made the substrate-change explicit: a path-
prediction architecture requires path-shaped targets, and the 30-frame
static dwell (a Stage 0b inheritance) was producing identity-prediction
problems repeated 30 times, not trajectories. This explorer replaces
the static-dwell pattern with a continuous-motion segment at each item.

Per-item life cycle (replaces dwell+transit with close_up+transit):

  - **close_up phase.** Agent moves along a straight 2 m segment that
    passes through the item's viewing_position, perpendicular to its
    viewing_heading. Heading is locked at viewing_heading (looking at
    item) throughout the close-up so the item enters the frame from
    one side, centres at the apex (viewing_position), and slides out
    the other side. Densified at 0.20 m → ~10 frames per close-up.
    No frame is bit-identical to any other within the close-up.

  - **transit phase.** Agent moves from previous item's close-up end
    to the current item's close-up start via NavMesh-densified path
    (instead of the old viewing_position → next viewing_position).
    Same densification + corner-rotation mechanism as the prior
    explorer. Transit is unconditional motion.

Loop boundary: fires on the *first close-up frame at item 1 of a loop
other than the very first*. `loop_index` increments at that frame.

Coordinate convention matches AI2-THOR (positions {"x", "y", "z"},
heading_y degrees with 0 = +Z and 90 = +X). Heading is locked at
viewing_heading throughout the close-up; rotates over corners and to
align with the close-up entry heading during transit.

Tunable parameters (all SCAFFOLDING for v0):
  close_up_length_m   : total length of the close-up segment (default 2.0)
  close_up_step_m     : densification step inside the close-up (default 0.20)
  densify_step_m      : densification step inside transit (default 0.20)
  corner_rotation_step_deg : per micro-step rotation at NavMesh corners (default 5.0)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_DENSIFY_STEP_M = 0.20
_DEFAULT_CORNER_ROT_STEP_DEG = 5.0
_DEFAULT_CLOSE_UP_LENGTH_M = 2.0
_DEFAULT_CLOSE_UP_STEP_M = 0.20
_DEFAULT_PATH_ALLOWED_ERROR_M = 0.05
_MAX_TELEPORT_FAILURES = 3


def _bearing_deg(a_x: float, a_z: float, b_x: float, b_z: float) -> float:
    return math.degrees(math.atan2(b_x - a_x, b_z - a_z)) % 360.0


def _shortest_signed_delta(target: float, current: float) -> float:
    d = (target - current) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def _perp_unit_ccw(heading_deg: float) -> Tuple[float, float]:
    """Unit vector 90 deg CCW from forward (top-down screen sense)."""
    rad = math.radians(heading_deg)
    fx, fz = math.sin(rad), math.cos(rad)
    return -fz, fx


def _densify_segment(
    a: Dict[str, float], b: Dict[str, float], step_m: float,
) -> List[Dict[str, float]]:
    dx = b["x"] - a["x"]
    dz = b["z"] - a["z"]
    dy = b.get("y", 0.9) - a.get("y", 0.9)
    seg_len = math.hypot(dx, dz)
    if seg_len < 1e-9:
        return []
    n = max(1, int(math.ceil(seg_len / step_m)))
    out: List[Dict[str, float]] = []
    for i in range(1, n + 1):
        t = min(1.0, i / n)
        out.append({
            "x": a["x"] + t * dx,
            "y": a.get("y", 0.9) + t * dy,
            "z": a["z"] + t * dz,
        })
    return out


def _request_path(
    controller: Any,
    target: Dict[str, float],
    allowed_error_m: float = _DEFAULT_PATH_ALLOWED_ERROR_M,
) -> Optional[List[Dict[str, float]]]:
    for action_name, shape in (
        ("GetShortestPathToPoint", "target_dict"),
        ("GetShortestPathToPoint", "target_xyz"),
    ):
        kw: Dict[str, Any] = {"action": action_name, "allowedError": allowed_error_m}
        if shape == "target_dict":
            kw["target"] = dict(target)
        else:
            kw["x"] = float(target["x"])
            kw["y"] = float(target.get("y", 0.0))
            kw["z"] = float(target["z"])
        try:
            event = controller.step(**kw)
        except Exception:
            continue
        if not event.metadata.get("lastActionSuccess"):
            continue
        ar = event.metadata.get("actionReturn")
        if isinstance(ar, list) and ar and isinstance(ar[0], dict) \
                and {"x", "y", "z"}.issubset(ar[0].keys()):
            return [dict(p) for p in ar]
        if isinstance(ar, dict):
            cs = ar.get("corners") or ar.get("path")
            if isinstance(cs, list) and cs and {"x", "y", "z"}.issubset(cs[0].keys()):
                return [dict(p) for p in cs]
    return None


def compute_close_up_endpoints(
    viewing_position: Dict[str, float],
    viewing_heading_deg: float,
    length_m: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Endpoints of the close-up segment.

    The agent walks from (vp - L/2 * perp_ccw) through vp to (vp + L/2 * perp_ccw)
    with heading locked at viewing_heading_deg. CCW chosen so the item slides in
    a consistent direction across all five items.
    """
    px, pz = _perp_unit_ccw(viewing_heading_deg)
    half = length_m * 0.5
    start = {
        "x": viewing_position["x"] - half * px,
        "y": viewing_position.get("y", 0.9),
        "z": viewing_position["z"] - half * pz,
    }
    end = {
        "x": viewing_position["x"] + half * px,
        "y": viewing_position.get("y", 0.9),
        "z": viewing_position["z"] + half * pz,
    }
    return start, end


class ContinuousMotionExplorer:
    """Cycles through 5 furniture viewing positions with continuous motion at each."""

    def __init__(
        self,
        controller: Any,
        route_items: List[Dict[str, Any]],
        *,
        close_up_length_m: float = _DEFAULT_CLOSE_UP_LENGTH_M,
        close_up_step_m: float = _DEFAULT_CLOSE_UP_STEP_M,
        densify_step_m: float = _DEFAULT_DENSIFY_STEP_M,
        corner_rotation_step_deg: float = _DEFAULT_CORNER_ROT_STEP_DEG,
    ) -> None:
        if not route_items:
            raise ValueError("route_items must be non-empty")
        if close_up_length_m <= 0:
            raise ValueError("close_up_length_m must be positive")
        if close_up_step_m <= 0:
            raise ValueError("close_up_step_m must be positive")
        if densify_step_m <= 0:
            raise ValueError("densify_step_m must be positive")
        if corner_rotation_step_deg <= 0 or 360.0 % corner_rotation_step_deg > 1e-6:
            raise ValueError(
                "corner_rotation_step_deg must divide 360 evenly; got "
                f"{corner_rotation_step_deg}"
            )
        for it in route_items:
            for k in ("item_id", "object_id", "object_type",
                      "viewing_position", "viewing_heading_deg"):
                if k not in it:
                    raise ValueError(f"route_items entry missing key {k!r}")

        self._controller = controller
        self._items: List[Dict[str, Any]] = [dict(it) for it in route_items]
        self._close_up_length_m = float(close_up_length_m)
        self._close_up_step_m = float(close_up_step_m)
        self._densify_step_m = float(densify_step_m)
        self._corner_rot_step = float(corner_rotation_step_deg)

        # Derive the agent's NavMesh floor y. Required because _teleport uses
        # forceAction=True (needed for off-grid 0.20 m close-up steps), which
        # bypasses AI2-THOR's floor-snap validation. Without an explicit floor
        # y, the agent base would land at whatever y is supplied in the input
        # position dict — which is 0.901 for close-up steps (from route.json's
        # viewing_position.y, a stage_0b standing snapshot) and ~0.006 for
        # transit steps (NavMesh-planned waypoint y). That oscillation caused
        # the camera-elevation bug surfaced in the 2026-05-14 sixth-STOP
        # trajectory diagnostic (see results/phase2_calibration_v2/
        # trajectory_diagnostic.json). Modal-y across all reachable positions
        # is robust to scenes with stairs or sloped floors (it picks the
        # dominant floor level the route's items live on).
        self._agent_floor_y, self.floor_y_summary = self._derive_floor_y()

        # Precompute close-up endpoints for each item.
        self._close_up_endpoints: List[Tuple[Dict[str, float], Dict[str, float]]] = []
        for it in self._items:
            s, e = compute_close_up_endpoints(
                it["viewing_position"],
                float(it["viewing_heading_deg"]),
                self._close_up_length_m,
            )
            self._close_up_endpoints.append((s, e))

        self._current_item_idx: int = 0
        self._loop_index: int = 0
        self._first_frame_emitted: bool = False
        self._loop_boundary_pending: bool = False

        self._phase: str = "close_up"  # initial: enter directly at item 1's close-up
        self._step_queue: List[Tuple[Dict[str, float], float, str, int, bool]] = []
        self._step_idx: int = 0
        self._consecutive_failures: int = 0

        # Last pose for observation lookup.
        self._last_jitter: Dict[str, float] = {
            "jitter_x": 0.0, "jitter_z": 0.0,
            "jitter_heading": 0.0, "jitter_scale": 0.0,
        }

        self._stats: Dict[str, int] = {
            "frames_emitted": 0,
            "close_up_frames_emitted": 0,
            "transit_frames_emitted": 0,
            "loops_completed": 0,
            "transitions_planned": 0,
            "transitions_planner_fallback": 0,
            "teleport_failures": 0,
        }

        # Bootstrap: teleport to the start of item 1's close-up and prime the queue.
        s0, e0 = self._close_up_endpoints[0]
        h0 = float(self._items[0]["viewing_heading_deg"])
        self._teleport(s0, h0)
        self._enqueue_close_up(self._current_item_idx)

    # ---- public API -----------------------------------------------------

    def next_micro_step(self) -> Dict[str, Any]:
        if self._step_idx >= len(self._step_queue):
            self._advance_phase()
        if self._step_idx >= len(self._step_queue):
            raise RuntimeError("step queue empty after advance; explorer is stuck")
        position, heading, phase, vp_id, apex_flag = self._step_queue[self._step_idx]
        success = self._teleport(position, heading)
        if success:
            self._step_idx += 1
            self._consecutive_failures = 0
        else:
            self._stats["teleport_failures"] += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= _MAX_TELEPORT_FAILURES:
                self._step_idx += 1
                self._consecutive_failures = 0

        loop_boundary = False
        is_first_close_up_frame = (
            phase == "close_up"
            and (self._step_idx == 1)  # we just incremented
            and self._is_start_of_close_up()
        )
        if is_first_close_up_frame and self._loop_boundary_pending:
            loop_boundary = True
            self._loop_boundary_pending = False
            self._loop_index += 1
            self._stats["loops_completed"] += 1

        self._stats["frames_emitted"] += 1
        if phase == "close_up":
            self._stats["close_up_frames_emitted"] += 1
        else:
            self._stats["transit_frames_emitted"] += 1
        if not self._first_frame_emitted:
            self._first_frame_emitted = True

        item = self._items[self._current_item_idx] if vp_id != 0 else None
        return self._observation(
            phase=phase,
            viewing_position_id=int(vp_id),
            furniture_object_id=(item["object_id"] if item is not None else None),
            furniture_object_type=(item["object_type"] if item is not None else None),
            close_up_apex_flag=apex_flag,
            loop_boundary_flag=loop_boundary,
            action_success=success,
        )

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    @property
    def loop_index(self) -> int:
        return self._loop_index

    @property
    def current_item(self) -> Dict[str, Any]:
        return dict(self._items[self._current_item_idx])

    @property
    def phase(self) -> str:
        return self._phase

    # ---- queue construction ---------------------------------------------

    def _is_start_of_close_up(self) -> bool:
        """True if step_idx-1 corresponds to the first frame of the current close-up."""
        if self._step_idx < 1:
            return False
        _, _, ph, _, _ = self._step_queue[self._step_idx - 1]
        if ph != "close_up":
            return False
        # The first close_up step in any queue we build is at position 0
        # (we never mix close_up + transit in the same queue).
        return self._step_idx == 1

    def _enqueue_close_up(self, item_idx: int) -> None:
        """Build the close-up queue for the given item: start..end at close_up_step_m density."""
        self._step_queue = []
        self._step_idx = 0
        self._phase = "close_up"
        item = self._items[item_idx]
        viewing_position = dict(item["viewing_position"])
        heading = float(item["viewing_heading_deg"])
        start, end = self._close_up_endpoints[item_idx]

        # Build the close-up path as a single segment from start to end, then
        # densify. Include the start point and the apex point (closest to
        # viewing_position) as queue entries; densify the segment.
        path_positions = [start] + _densify_segment(start, end, self._close_up_step_m)
        # Identify the apex frame (the one closest to viewing_position).
        apex_idx = min(
            range(len(path_positions)),
            key=lambda i: (
                (path_positions[i]["x"] - viewing_position["x"]) ** 2
                + (path_positions[i]["z"] - viewing_position["z"]) ** 2
            ),
        )
        vp_id = int(item["item_id"])
        for i, pos in enumerate(path_positions):
            self._step_queue.append((pos, heading, "close_up", vp_id, i == apex_idx))

    def _enqueue_transit_to_next(self) -> None:
        """Build the transit queue from current close-up end to next close-up start."""
        cur_idx = self._current_item_idx
        next_idx = (cur_idx + 1) % len(self._items)
        _, cur_end = self._close_up_endpoints[cur_idx]
        next_start, _ = self._close_up_endpoints[next_idx]
        cur_heading = float(self._items[cur_idx]["viewing_heading_deg"])
        next_heading = float(self._items[next_idx]["viewing_heading_deg"])

        # Defensive: ensure we're at cur_end before planning.
        self._teleport(cur_end, cur_heading)

        corners = _request_path(self._controller, next_start)
        used_fallback = False
        if corners is None or len(corners) < 2:
            corners = [cur_end, next_start]
            used_fallback = True
            self._stats["transitions_planner_fallback"] += 1
        else:
            corners = [cur_end] + corners
        self._stats["transitions_planned"] += 1

        self._step_queue = []
        self._step_idx = 0
        self._phase = "transit"
        current_heading = cur_heading % 360.0
        for c_idx in range(len(corners) - 1):
            a = corners[c_idx]
            b = corners[c_idx + 1]
            seg_len = math.hypot(b["x"] - a["x"], b["z"] - a["z"])
            if seg_len < 1e-6:
                continue
            target_heading = _bearing_deg(a["x"], a["z"], b["x"], b["z"])
            self._append_rotation(a, current_heading, target_heading, phase="transit", vp_id=0)
            current_heading = target_heading
            for pos in _densify_segment(a, b, self._densify_step_m):
                self._step_queue.append((dict(pos), current_heading, "transit", 0, False))

        # Final rotation to align with next item's viewing_heading.
        if corners:
            self._append_rotation(
                corners[-1], current_heading, next_heading % 360.0,
                phase="transit", vp_id=0,
            )

    def _append_rotation(
        self,
        position: Dict[str, float],
        current_heading: float,
        target_heading: float,
        phase: str,
        vp_id: int,
    ) -> None:
        delta = _shortest_signed_delta(target_heading, current_heading)
        if abs(delta) <= self._corner_rot_step:
            return
        n = int(math.ceil(abs(delta) / self._corner_rot_step))
        sign = 1.0 if delta > 0 else -1.0
        for i in range(1, n):
            heading = (current_heading + sign * self._corner_rot_step * i) % 360.0
            self._step_queue.append((dict(position), heading, phase, vp_id, False))
        self._step_queue.append((dict(position), target_heading % 360.0, phase, vp_id, False))

    def _advance_phase(self) -> None:
        """Called when the current queue is exhausted."""
        if self._phase == "close_up":
            self._enqueue_transit_to_next()
        elif self._phase == "transit":
            prev_idx = self._current_item_idx
            self._current_item_idx = (self._current_item_idx + 1) % len(self._items)
            if self._current_item_idx == 0 and prev_idx == len(self._items) - 1:
                self._loop_boundary_pending = True
            self._enqueue_close_up(self._current_item_idx)
        else:
            raise RuntimeError(f"unknown phase: {self._phase}")

    # ---- low-level controller ops --------------------------------------

    def _teleport(self, position: Dict[str, float], heading_deg: float) -> bool:
        # Always teleport the agent base to the NavMesh floor y derived at
        # init (self._agent_floor_y). Drop the input position's y entirely —
        # forceAction=True bypasses AI2-THOR's floor-snap validation, so the
        # base lands wherever y is specified; routing the input y straight
        # through caused the camera-elevation bug fixed in the post-sixth-
        # STOP commit. forceAction=True is retained because the close-up's
        # 0.20 m step grid doesn't align to AI2-THOR's 0.25 m navigation
        # grid. `standing=True` makes the agent posture explicit (eye-height
        # camera offset above the base).
        try:
            event = self._controller.step(
                action="Teleport",
                position={
                    "x": float(position["x"]),
                    "y": float(self._agent_floor_y),
                    "z": float(position["z"]),
                },
                rotation={"x": 0.0, "y": float(heading_deg) % 360.0, "z": 0.0},
                horizon=0.0,
                standing=True,
                forceAction=True,
            )
        except Exception:
            return False
        return bool(event.metadata.get("lastActionSuccess"))

    def _derive_floor_y(self) -> Tuple[float, Dict[str, Any]]:
        """Query the NavMesh once, derive the modal floor y for the agent base.

        Returns (modal_y, summary_dict) where summary_dict captures the
        y-distribution for HANDOFF/audit-trail logging (n unique values, min,
        max, count at the mode, fraction of reachable positions at the mode).
        """
        from collections import Counter

        event = self._controller.step(action="GetReachablePositions")
        if not event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(
                "ContinuousMotionExplorer: GetReachablePositions failed at init; "
                "cannot derive agent floor y."
            )
        reach = event.metadata.get("actionReturn") or []
        if not reach:
            raise RuntimeError(
                "ContinuousMotionExplorer: GetReachablePositions returned an "
                "empty set; cannot derive agent floor y."
            )
        ys_rounded = [round(float(p["y"]), 4) for p in reach]
        y_counter = Counter(ys_rounded)
        modal_y, n_modal = y_counter.most_common(1)[0]
        summary: Dict[str, Any] = {
            "n_reachable_positions": int(len(reach)),
            "modal_y": float(modal_y),
            "n_positions_at_modal_y": int(n_modal),
            "fraction_at_modal_y": float(n_modal) / float(len(reach)),
            "n_unique_y_values_rounded_4dp": int(len(y_counter)),
            "y_min": float(min(ys_rounded)),
            "y_max": float(max(ys_rounded)),
            "y_histogram_top5": [
                {"y": float(y), "n": int(c)}
                for y, c in y_counter.most_common(5)
            ],
        }
        # Log to stdout so the calibration / collection / preflight scripts'
        # log files capture the derivation for HANDOFF audit (per the
        # reviewer's 2026-05-14 directive).
        print(
            f"[ContinuousMotionExplorer] floor_y_summary: modal_y={summary['modal_y']:.4f} "
            f"({summary['n_positions_at_modal_y']}/{summary['n_reachable_positions']} = "
            f"{summary['fraction_at_modal_y']*100:.1f}% of reachable positions), "
            f"unique_y_count={summary['n_unique_y_values_rounded_4dp']}, "
            f"y_range=[{summary['y_min']:.4f}, {summary['y_max']:.4f}]",
            flush=True,
        )
        return float(modal_y), summary

    def _observation(
        self,
        phase: str,
        viewing_position_id: int,
        furniture_object_id: Optional[str],
        furniture_object_type: Optional[str],
        close_up_apex_flag: bool,
        loop_boundary_flag: bool,
        action_success: bool,
    ) -> Dict[str, Any]:
        event = self._controller.last_event
        agent = event.metadata["agent"]
        return {
            "frame": event.frame,
            "position": dict(agent["position"]),
            "rotation_y": float(agent["rotation"]["y"]),
            "phase": phase,
            "viewing_position_id": int(viewing_position_id),
            "furniture_object_id": furniture_object_id,
            "furniture_object_type": furniture_object_type,
            "loop_index": int(self._loop_index),
            "loop_boundary_flag": bool(loop_boundary_flag),
            "close_up_apex_flag": bool(close_up_apex_flag),
            "action_success": bool(action_success),
            # Jitter slots are retained at 0.0 for schema compatibility with the
            # prior Stage 0b annotation files; the new substrate's motion does
            # not require per-frame jitter for variation. See HANDOFF for the
            # variation-strategy decision.
            "jitter_x": float(self._last_jitter["jitter_x"]),
            "jitter_z": float(self._last_jitter["jitter_z"]),
            "jitter_heading": float(self._last_jitter["jitter_heading"]),
            "jitter_scale": float(self._last_jitter["jitter_scale"]),
        }


__all__ = ["ContinuousMotionExplorer", "compute_close_up_endpoints"]
