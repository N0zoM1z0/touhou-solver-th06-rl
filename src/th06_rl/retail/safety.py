"""Hard authority over hazards already present in native memory."""

from __future__ import annotations

from .hazards.bullets import hazards_by_frame as bullet_hazards_by_frame
from .hazards.bullets import nearest_current_clearance
from .hazards.geometry import signed_clearance
from .hazards.enemies import hazards_by_frame as enemy_hazards_by_frame
from .hazards.lasers import hazards_by_frame as laser_hazards_by_frame
from .hazards.lasers import signed_laser_clearance
from .hazards.world import forecast_world_births
from .model import (
    ACTIONS,
    BUTTON_DOWN,
    BUTTON_FOCUS,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_UP,
    Action,
    SafeAction,
    Snapshot,
    action_from_input,
)


MOVEMENT_LEFT = 8.0
MOVEMENT_RIGHT = 376.0
MOVEMENT_TOP = 16.0
MOVEMENT_BOTTOM = 432.0
# Native pickup remains bounded to 0/1/2 frames after SendInput. A new command
# is published only from an age-zero snapshot, but SendInput may cross one game
# frame, so hard authority covers the combined 0..3-frame delivery window.
DELIVERY_DELAYS = (0, 1, 2, 3)
COLLISION_MARGIN = 0.35
_CONTROL_KEYS = (
    ("down", BUTTON_DOWN),
    ("focus", BUTTON_FOCUS),
    ("left", BUTTON_LEFT),
    ("right", BUTTON_RIGHT),
    ("up", BUTTON_UP),
)


def _action_mask(action: Action) -> int:
    mask = BUTTON_FOCUS if action.focused else 0
    if action.dx < 0:
        mask |= BUTTON_LEFT
    elif action.dx > 0:
        mask |= BUTTON_RIGHT
    if action.dy < 0:
        mask |= BUTTON_UP
    elif action.dy > 0:
        mask |= BUTTON_DOWN
    return mask


def transition_input_masks(current: Action, target: Action) -> tuple[int, ...]:
    """Exact control masks inside Keyboard's sorted release/press batch."""
    current_mask = _action_mask(current)
    target_mask = _action_mask(target)
    prefix_mask = current_mask
    prefixes: list[int] = []

    events = tuple(
        bit for _key, bit in _CONTROL_KEYS
        if current_mask & bit and not target_mask & bit
    ) + tuple(
        bit for _key, bit in _CONTROL_KEYS
        if target_mask & bit and not current_mask & bit
    )
    for bit in events:
        if prefix_mask & bit:
            prefix_mask &= ~bit
        else:
            prefix_mask |= bit
        if prefix_mask not in (current_mask, target_mask):
            prefixes.append(prefix_mask)
    return tuple(prefixes)


def transition_actions(current: Action, target: Action) -> tuple[Action, ...]:
    """Movement states observable inside the same physical input batch."""
    prefixes: list[Action] = []
    for mask in transition_input_masks(current, target):
        prefix = action_from_input(mask)
        if prefix not in (current, target) and prefix not in prefixes:
            prefixes.append(prefix)
    return tuple(prefixes)


def _step_player(
    x: float,
    y: float,
    action: Action,
    cardinal_speed: float,
    diagonal_speed: float,
) -> tuple[float, float]:
    speed = diagonal_speed if action.dx and action.dy else cardinal_speed
    x = min(MOVEMENT_RIGHT, max(MOVEMENT_LEFT, x + action.dx * speed))
    y = min(MOVEMENT_BOTTOM, max(MOVEMENT_TOP, y + action.dy * speed))
    return x, y


def candidate_path(
    snapshot: Snapshot,
    action: Action,
    delay: int,
    horizon: int,
    transition_action: Action | None = None,
) -> list[tuple[float, float]]:
    if transition_action is not None and delay <= 0:
        raise ValueError("a transition prefix requires a positive delivery delay")
    current = action_from_input(snapshot.input_mask)
    x, y = snapshot.x, snapshot.y
    path: list[tuple[float, float]] = []
    for frame in range(1, horizon + 1):
        if transition_action is not None and frame == delay:
            step_action = transition_action
        elif frame < delay or (transition_action is None and frame <= delay):
            step_action = current
        else:
            step_action = action
        cardinal = (
            snapshot.focus_speed if step_action.focused else snapshot.normal_speed
        )
        diagonal = (
            snapshot.focus_diagonal_speed
            if step_action.focused
            else snapshot.normal_diagonal_speed
        )
        x, y = _step_player(x, y, step_action, cardinal, diagonal)
        path.append((x, y))
    return path


def candidate_paths(
    snapshot: Snapshot,
    action: Action,
    delay: int,
    horizon: int,
) -> tuple[list[tuple[float, float]], ...]:
    paths = [candidate_path(snapshot, action, delay, horizon)]
    if delay > 0:
        current = action_from_input(snapshot.input_mask)
        paths.extend(
            candidate_path(snapshot, action, delay, horizon, prefix)
            for prefix in transition_actions(current, action)
        )
    return tuple(paths)


def certify_actions(
    snapshot: Snapshot,
    horizon: int,
    delivery_delays: tuple[int, ...] = DELIVERY_DELAYS,
    actions: tuple[Action, ...] = ACTIONS,
) -> tuple[SafeAction, ...]:
    if not delivery_delays:
        raise ValueError("delivery delays cannot be empty")
    bullet_frames = bullet_hazards_by_frame(snapshot, horizon)
    enemy_frames = enemy_hazards_by_frame(snapshot.enemies, horizon)
    laser_frames = laser_hazards_by_frame(snapshot.lasers, horizon)
    birth_forecast = forecast_world_births(
        snapshot,
        ((snapshot.x, snapshot.y),) * horizon,
    )
    if birth_forecast.covered_frames < horizon:
        return ()
    certified: list[SafeAction] = []
    for action in actions:
        action_clearance = 999.0
        valid = True
        final_x = snapshot.x
        final_y = snapshot.y
        for delay in delivery_delays:
            for path_index, path in enumerate(candidate_paths(snapshot, action, delay, horizon)):
                if delay == delivery_delays[-1] and path_index == 0:
                    final_x, final_y = path[-1]
                for frame_index, (x, y) in enumerate(path):
                    for hazard in bullet_frames[frame_index]:
                        clearance = signed_clearance(
                            x, y, snapshot.half_width, snapshot.half_height, hazard
                        )
                        action_clearance = min(action_clearance, clearance)
                        if clearance <= COLLISION_MARGIN:
                            valid = False
                            break
                    if valid:
                        for hazard in birth_forecast.hazards[frame_index]:
                            clearance = signed_clearance(
                                x, y, snapshot.half_width, snapshot.half_height, hazard
                            )
                            action_clearance = min(action_clearance, clearance)
                            if clearance <= COLLISION_MARGIN:
                                valid = False
                                break
                    if valid and birth_forecast.body_hazards:
                        for hazard in birth_forecast.body_hazards[frame_index]:
                            clearance = signed_clearance(
                                x, y, snapshot.half_width, snapshot.half_height, hazard
                            )
                            action_clearance = min(action_clearance, clearance)
                            if clearance <= COLLISION_MARGIN:
                                valid = False
                                break
                    if valid:
                        for hazard in enemy_frames[frame_index]:
                            clearance = signed_clearance(
                                x, y, snapshot.half_width, snapshot.half_height, hazard
                            )
                            action_clearance = min(action_clearance, clearance)
                            if clearance <= COLLISION_MARGIN:
                                valid = False
                                break
                    if valid:
                        for laser in laser_frames[frame_index]:
                            clearance = signed_laser_clearance(
                                x, y, snapshot.half_width, snapshot.half_height, laser
                            )
                            action_clearance = min(action_clearance, clearance)
                            if clearance <= COLLISION_MARGIN:
                                valid = False
                                break
                    if valid and birth_forecast.laser_hazards:
                        for laser in birth_forecast.laser_hazards[frame_index]:
                            clearance = signed_laser_clearance(
                                x,
                                y,
                                snapshot.half_width,
                                snapshot.half_height,
                                laser,
                            )
                            action_clearance = min(action_clearance, clearance)
                            if clearance <= COLLISION_MARGIN:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            certified.append(SafeAction(action, action_clearance, final_x, final_y))
    return tuple(certified)
