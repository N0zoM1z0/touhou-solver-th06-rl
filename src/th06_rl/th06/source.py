"""Lower one coherent TH06 snapshot into observed-hazard geometry.

This adapter deliberately does not interpret stage timelines or ECL.  It
projects only hazards that already exist in the captured Wine state, using
their decoded motion fields.  Future births and script-driven mutations are
policy uncertainty and become factual evidence only after Wine executes them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.model import Action, Kinematics, movement_actions
from ..native import Aabb, PackedHazards
from ..retail.model import CONTROL_ACTIONS, action_from_input
from .observed_lasers import laser_rects_by_frame


SHIELD_HORIZON = 4
COLLISION_MARGIN = 0.35
OBSERVED_SHIELD_CONTRACT = "observed-hazard-kinematics-v1"
ALL_ACTIONS = movement_actions()
ACTION_BY_STATE = {
    (action.dx, action.dy, action.focused): action for action in ALL_ACTIONS
}


class ControlUnavailable(RuntimeError):
    """The physical observation/publication transaction cannot continue."""


@dataclass(frozen=True)
class ObservedHazardProjection:
    hazards: PackedHazards
    horizon: int
    contract: str = OBSERVED_SHIELD_CONTRACT


def core_action_from_input(input_mask: int) -> Action:
    source = action_from_input(input_mask)
    return ACTION_BY_STATE[(source.dx, source.dy, source.focused)]


def retail_action(action: Action):
    return next(
        item
        for item in CONTROL_ACTIONS
        if (item.dx, item.dy, item.focused)
        == (action.dx, action.dy, action.focused)
    )


def kinematics_from_snapshot(snapshot) -> Kinematics:
    return Kinematics(
        normal_speed=snapshot.normal_speed,
        focus_speed=snapshot.focus_speed,
        normal_diagonal_speed=snapshot.normal_diagonal_speed,
        focus_diagonal_speed=snapshot.focus_diagonal_speed,
    )


def _reachable_aabbs(snapshot, frames, margin: float):
    """Drop observed boxes that cannot intersect any player path."""
    speed = max(snapshot.normal_speed, snapshot.focus_speed)
    result = []
    for index, frame in enumerate(frames):
        steps = index + 1
        minimum_x = (
            max(8.0, snapshot.x - speed * steps)
            - snapshot.half_width
            - margin
        )
        maximum_x = (
            min(376.0, snapshot.x + speed * steps)
            + snapshot.half_width
            + margin
        )
        minimum_y = (
            max(16.0, snapshot.y - speed * steps)
            - snapshot.half_height
            - margin
        )
        maximum_y = (
            min(432.0, snapshot.y + speed * steps)
            + snapshot.half_height
            + margin
        )
        result.append(
            tuple(
                hazard
                for hazard in frame
                if not (
                    hazard[2] < minimum_x
                    or hazard[0] > maximum_x
                    or hazard[3] < minimum_y
                    or hazard[1] > maximum_y
                )
            )
        )
    return tuple(result)


def lower_observed_hazards(
    snapshot,
    requested_horizon: int = SHIELD_HORIZON,
    *,
    collision_margin: float = COLLISION_MARGIN,
) -> ObservedHazardProjection:
    """Project already-instantiated hazards without predicting script output.

    The returned action shield is exact only with respect to these supplied
    primitives and their decoded short-horizon kinematics.  It is not a claim
    that no unobserved bullet, laser, teleport, or global mutation can occur.
    """
    if requested_horizon < 1:
        raise ValueError("observed hazard horizon must be positive")
    from .observed_bullets import reachable_hazards_by_frame
    from ..retail.hazards.enemies import (
        hazards_by_frame as enemy_hazards_by_frame,
    )

    bullet_frames = reachable_hazards_by_frame(
        snapshot,
        requested_horizon,
        collision_margin,
    )
    enemy_frames = enemy_hazards_by_frame(
        snapshot.enemies,
        requested_horizon,
    )
    laser_frames = laser_rects_by_frame(
        snapshot.lasers,
        requested_horizon,
    )
    aabb_frames = tuple(
        bullet_frames[index] + enemy_frames[index]
        for index in range(requested_horizon)
    )
    reachable_frames = _reachable_aabbs(
        snapshot,
        aabb_frames,
        collision_margin,
    )
    return ObservedHazardProjection(
        hazards=PackedHazards(
            aabb_frames=tuple(
                tuple(Aabb(*hazard) for hazard in frame)
                for frame in reachable_frames
            ),
            laser_frames=laser_frames,
        ),
        horizon=requested_horizon,
    )
