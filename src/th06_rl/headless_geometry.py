"""Source-grounded hazard lowering for TH06 headless observations.

The game process remains collision authority.  This module reconstructs the
same bounded, already-observed-hazard frontier used by the physical agent so a
headless learner may rank actions without acquiring collision authority.
Timeline/ECL births are deliberately retained as features, not interpreted in
this resident gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
import struct
from typing import Any, Iterable, Mapping

from .core.model import Action, Kinematics
from .native import ACTIONS, Aabb, NativeCertifiedAction, NativeKernel, PackedHazards
from .th06.observed_lasers import laser_rects_by_frame


OBSERVATION_SCHEMA = "th06-headless-observation-v2"
HARD_HORIZON = 4
COLLISION_MARGIN = 0.35
KINEMATICS = Kinematics(4.0, 2.0, 2.8284270763397217, 1.4142135381698608)
BY_NAME = {action.name: action for action in ACTIONS}

SLOWDOWN_FLAG = 0x001
SPAWN_EFFECT_FLAGS = 0x00E
ACCELERATION_FLAG = 0x010
CURVE_ACCELERATION_FLAG = 0x020
RELATIVE_TURN_FLAG = 0x040
PLAYER_AIM_TURN_FLAG = 0x080
ABSOLUTE_TURN_FLAG = 0x100
REFLECT_ALL_EDGES_FLAG = 0x400
REFLECT_NO_BOTTOM_FLAG = 0x800
KNOWN_EX_FLAGS = 0xFFF
_KEEP_OUT_OF_BOUNDS_FLAGS = 0xDC0


class HeadlessAuthorityUnavailable(RuntimeError):
    """The observation cannot support the fixed native collision contract."""


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _finite_number(row: Mapping[str, Any], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise HeadlessAuthorityUnavailable(f"invalid headless field {name}") from error
    if not math.isfinite(value):
        raise HeadlessAuthorityUnavailable(f"non-finite headless field {name}")
    return value


def _integer(row: Mapping[str, Any], name: str) -> int:
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise HeadlessAuthorityUnavailable(f"invalid headless integer {name}") from error


def validate_headless_observation(observation: Mapping[str, Any]) -> None:
    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise HeadlessAuthorityUnavailable("unsupported headless observation schema")
    player = observation.get("player")
    if not isinstance(player, Mapping):
        raise HeadlessAuthorityUnavailable("headless player state is missing")
    for name in ("x", "y", "half_width", "half_height"):
        _finite_number(player, name)
    for collection in ("bullets", "lasers", "enemies"):
        value = observation.get(collection)
        if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
            raise HeadlessAuthorityUnavailable(f"headless {collection} are incoherent")


def action_from_input(input_mask: int) -> Action:
    if input_mask & 0x02:
        raise HeadlessAuthorityUnavailable("headless observation contains Bomb input")
    dx = int(bool(input_mask & 0x80)) - int(bool(input_mask & 0x40))
    dy = int(bool(input_mask & 0x20)) - int(bool(input_mask & 0x10))
    focused = bool(input_mask & 0x04)
    name = next(
        action.name
        for action in ACTIONS
        if (action.dx, action.dy, action.focused) == (dx, dy, focused)
    )
    return BY_NAME[name]


@dataclass(frozen=True)
class _Bullet:
    x: float
    y: float
    vx: float
    vy: float
    half_width: float
    half_height: float
    sprite_half_width: float
    sprite_half_height: float
    state: int
    ex_flags: int
    acceleration_x: float
    acceleration_y: float
    speed: float
    angle: float
    curve_speed_acceleration: float
    curve_angular_velocity: float
    turn_speed: float
    direction_rotation: float
    timer: int
    timer_float: float
    acceleration_duration: int
    direction_interval: int
    direction_num_times: int
    direction_max_times: int

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> _Bullet:
        floating = (
            "x", "y", "vx", "vy", "half_width", "half_height",
            "sprite_half_width", "sprite_half_height", "acceleration_x",
            "acceleration_y", "speed", "angle", "curve_speed_acceleration",
            "curve_angular_velocity", "turn_speed", "direction_rotation",
            "timer_float",
        )
        integers = (
            "state", "ex_flags", "timer", "acceleration_duration",
            "direction_interval", "direction_num_times", "direction_max_times",
        )
        values: dict[str, float | int] = {
            name: _finite_number(row, name) for name in floating
        }
        values.update({name: _integer(row, name) for name in integers})
        bullet = cls(**values)  # type: ignore[arg-type]
        if bullet.state not in range(6):
            raise HeadlessAuthorityUnavailable("invalid headless bullet state")
        if bullet.ex_flags & ~KNOWN_EX_FLAGS:
            raise HeadlessAuthorityUnavailable(
                f"unknown headless bullet EX flags 0x{bullet.ex_flags:x}"
            )
        if min(bullet.half_width, bullet.half_height) <= 0.0:
            raise HeadlessAuthorityUnavailable("invalid headless bullet collision size")
        if bullet.ex_flags & (REFLECT_ALL_EDGES_FLAG | REFLECT_NO_BOTTOM_FLAG):
            if min(bullet.sprite_half_width, bullet.sprite_half_height) <= 0.0:
                raise HeadlessAuthorityUnavailable("reflecting bullet has no sprite size")
        return bullet


def _normalize_angle(value: float) -> float:
    value = _f32(value)
    while value > math.pi:
        value = _f32(value - math.tau)
    while value < -math.pi:
        value = _f32(value + math.tau)
    return value


def _in_bounds(bullet: _Bullet) -> bool:
    return not (
        bullet.sprite_half_width + bullet.x < 0.0
        or bullet.x - bullet.sprite_half_width > 384.0
        or bullet.sprite_half_height + bullet.y < 0.0
        or bullet.y - bullet.sprite_half_height > 448.0
    )


def _box(bullet: _Bullet) -> Aabb:
    return Aabb(
        bullet.x - bullet.half_width,
        bullet.y - bullet.half_height,
        bullet.x + bullet.half_width,
        bullet.y + bullet.half_height,
    )


def _maximum_uncertain_speed(bullet: _Bullet, remaining: int) -> float:
    base = max(math.hypot(bullet.vx, bullet.vy), abs(bullet.speed), abs(bullet.turn_speed))
    acceleration = math.hypot(bullet.acceleration_x, bullet.acceleration_y)
    curve = abs(bullet.curve_speed_acceleration)
    return max(0.0, base + remaining * max(acceleration, curve))


def _project_fired(bullet: _Bullet, horizon: int) -> tuple[Aabb, ...]:
    result: list[Aabb] = []
    current = replace(bullet, state=1)
    uncertain_radius: float | None = None
    uncertain_x = current.x
    uncertain_y = current.y
    for frame_index in range(horizon):
        remaining = horizon - frame_index
        if uncertain_radius is not None:
            uncertain_radius += _maximum_uncertain_speed(current, remaining)
            result.append(Aabb(
                uncertain_x - uncertain_radius - current.half_width,
                uncertain_y - uncertain_radius - current.half_height,
                uncertain_x + uncertain_radius + current.half_width,
                uncertain_y + uncertain_radius + current.half_height,
            ))
            continue

        flags = current.ex_flags
        vx, vy = current.vx, current.vy
        speed, angle = current.speed, current.angle
        direction_num_times = current.direction_num_times

        if flags & SLOWDOWN_FLAG:
            if current.timer <= 16:
                slowdown = _f32(5.0 - _f32(current.timer_float * 5.0 / 16.0))
                step_speed = _f32(slowdown + speed)
                vx = _f32(math.cos(angle) * step_speed)
                vy = _f32(math.sin(angle) * step_speed)
            else:
                flags ^= SLOWDOWN_FLAG
        elif flags & ACCELERATION_FLAG:
            if current.timer >= current.acceleration_duration:
                flags &= ~ACCELERATION_FLAG
            else:
                vx = _f32(vx + current.acceleration_x)
                vy = _f32(vy + current.acceleration_y)
                angle = _f32(math.atan2(vy, vx))
        elif flags & CURVE_ACCELERATION_FLAG:
            if current.timer >= current.acceleration_duration:
                flags &= ~CURVE_ACCELERATION_FLAG
            else:
                angle = _normalize_angle(_f32(angle + current.curve_angular_velocity))
                speed = _f32(speed + current.curve_speed_acceleration)
                vx = _f32(math.cos(angle) * speed)
                vy = _f32(math.sin(angle) * speed)

        direction_flag = (
            RELATIVE_TURN_FLAG if flags & RELATIVE_TURN_FLAG
            else ABSOLUTE_TURN_FLAG if flags & ABSOLUTE_TURN_FLAG
            else 0
        )
        if direction_flag:
            threshold = current.direction_interval * (direction_num_times + 1)
            if current.timer >= threshold:
                direction_num_times += 1
                if direction_num_times >= current.direction_max_times:
                    flags &= ~direction_flag
                angle = (
                    _f32(angle + current.direction_rotation)
                    if direction_flag == RELATIVE_TURN_FLAG
                    else _f32(current.direction_rotation)
                )
                speed = current.turn_speed
                step_speed = speed
            else:
                if current.direction_interval <= 0:
                    raise HeadlessAuthorityUnavailable("invalid bullet turn interval")
                step_speed = _f32(
                    speed
                    - _f32(
                        (current.timer_float - current.direction_interval * direction_num_times)
                        * speed
                        / current.direction_interval
                    )
                )
            vx = _f32(math.cos(angle) * step_speed)
            vy = _f32(math.sin(angle) * step_speed)
        elif flags & PLAYER_AIM_TURN_FLAG:
            threshold = current.direction_interval * (direction_num_times + 1)
            if current.timer >= threshold:
                # The source retargets to the candidate player position.  The
                # shared native hazard view cannot encode candidate-dependent
                # angles, so enclose every direction from this exact point.
                uncertain_x = current.x
                uncertain_y = current.y
                uncertain_radius = _maximum_uncertain_speed(current, remaining)
                result.append(Aabb(
                    uncertain_x - uncertain_radius - current.half_width,
                    uncertain_y - uncertain_radius - current.half_height,
                    uncertain_x + uncertain_radius + current.half_width,
                    uncertain_y + uncertain_radius + current.half_height,
                ))
                continue
            if current.direction_interval <= 0:
                raise HeadlessAuthorityUnavailable("invalid aimed-turn interval")
            step_speed = _f32(
                speed
                - _f32(
                    (current.timer_float - current.direction_interval * direction_num_times)
                    * speed
                    / current.direction_interval
                )
            )
            vx = _f32(math.cos(angle) * step_speed)
            vy = _f32(math.sin(angle) * step_speed)
        elif flags & (REFLECT_ALL_EDGES_FLAG | REFLECT_NO_BOTTOM_FLAG):
            if not _in_bounds(current):
                if current.x < 0.0 or current.x >= 384.0:
                    angle = _normalize_angle(_f32(-angle - math.pi))
                if current.y < 0.0 or (
                    flags & REFLECT_ALL_EDGES_FLAG and current.y >= 448.0
                ):
                    angle = _f32(-angle)
                speed = current.turn_speed
                vx = _f32(math.cos(angle) * speed)
                vy = _f32(math.sin(angle) * speed)
                direction_num_times += 1
                if direction_num_times >= current.direction_max_times:
                    flags &= ~(REFLECT_ALL_EDGES_FLAG | REFLECT_NO_BOTTOM_FLAG)

        current = replace(
            current,
            x=_f32(current.x + vx),
            y=_f32(current.y + vy),
            vx=vx,
            vy=vy,
            speed=speed,
            angle=angle,
            ex_flags=flags,
            timer=current.timer + 1,
            timer_float=_f32(current.timer_float + 1.0),
            direction_num_times=direction_num_times,
        )
        result.append(_box(current))
    return tuple(result)


def _project_bullet(bullet: _Bullet, horizon: int) -> tuple[Aabb | None, ...]:
    if bullet.state in (0, 5):
        return (None,) * horizon
    if bullet.state == 1:
        return _project_fired(bullet, horizon)
    if bullet.state not in (2, 3, 4):
        raise HeadlessAuthorityUnavailable("unsupported headless bullet state")

    divisor = {2: 2.0, 3: 2.5, 4: 3.0}[bullet.state]
    spawn_x, spawn_y = bullet.x, bullet.y
    branches: list[list[Aabb]] = [[] for _ in range(horizon)]
    for transition_frame in range(1, horizon + 1):
        spawn_x = _f32(spawn_x + _f32(bullet.vx / divisor))
        spawn_y = _f32(spawn_y + _f32(bullet.vy / divisor))
        fired = replace(
            bullet,
            x=spawn_x,
            y=spawn_y,
            state=1,
            timer=0,
            timer_float=0.0,
        )
        for offset, box in enumerate(_project_fired(fired, horizon - transition_frame + 1)):
            branches[transition_frame - 1 + offset].append(box)
    return tuple(
        Aabb(
            min(box.left for box in boxes),
            min(box.top for box in boxes),
            max(box.right for box in boxes),
            max(box.bottom for box in boxes),
        )
        for boxes in branches
    )


def _reachable(box: Aabb, *, x: float, y: float, frame: int, half_width: float, half_height: float) -> bool:
    reach = KINEMATICS.normal_speed * frame
    return not (
        box.right < x - reach - half_width - COLLISION_MARGIN
        or box.left > x + reach + half_width + COLLISION_MARGIN
        or box.bottom < y - reach - half_height - COLLISION_MARGIN
        or box.top > y + reach + half_height + COLLISION_MARGIN
    )


def _bullet_frames(
    rows: Iterable[Mapping[str, Any]],
    *,
    horizon: int,
    player_x: float,
    player_y: float,
    player_half_width: float,
    player_half_height: float,
) -> list[list[Aabb]]:
    frames: list[list[Aabb]] = [[] for _ in range(horizon)]
    for row in rows:
        bullet = _Bullet.from_json(row)
        for frame_index, box in enumerate(_project_bullet(bullet, horizon)):
            if box is not None and _reachable(
                box,
                x=player_x,
                y=player_y,
                frame=frame_index + 1,
                half_width=player_half_width,
                half_height=player_half_height,
            ):
                frames[frame_index].append(box)
    return frames


def _enemy_frames(rows: Iterable[Mapping[str, Any]], horizon: int) -> list[list[Aabb]]:
    frames: list[list[Aabb]] = [[] for _ in range(horizon)]
    for row in rows:
        if row.get("contact_active") is not True:
            continue
        x = _finite_number(row, "x")
        y = _finite_number(row, "y")
        vx = _finite_number(row, "vx")
        vy = _finite_number(row, "vy")
        half_width = _finite_number(row, "hitbox_width") / 2.0
        half_height = _finite_number(row, "hitbox_height") / 2.0
        if min(half_width, half_height) <= 0.0:
            raise HeadlessAuthorityUnavailable("invalid enemy contact hitbox")
        for frame in range(horizon):
            x = _f32(x + vx)
            y = _f32(y + vy)
            frames[frame].append(Aabb(
                x - half_width,
                y - half_height,
                x + half_width,
                y + half_height,
            ))
    return frames


def _laser_frames(rows: Iterable[Mapping[str, Any]], horizon: int):
    lasers = []
    for row in rows:
        state = _integer(row, "state")
        timer = _integer(row, "timer")
        graze_delay = _integer(row, "graze_delay")
        tracked = row.get("angle_tracked") is True
        could_be_lethal = state in (1, 2) or timer + horizon >= graze_delay
        if not tracked and could_be_lethal:
            raise HeadlessAuthorityUnavailable(
                f"laser slot {_integer(row, 'slot')} lacks angular history"
            )
        lasers.append(SimpleNamespace(
            x=_finite_number(row, "x"),
            y=_finite_number(row, "y"),
            angle=_finite_number(row, "angle"),
            angular_velocity=_finite_number(row, "angular_velocity"),
            start_offset=_finite_number(row, "start"),
            end_offset=_finite_number(row, "end"),
            start_length=_finite_number(row, "start_length"),
            width=_finite_number(row, "width"),
            speed=_finite_number(row, "speed"),
            start_time=_integer(row, "start_time"),
            hitbox_start_time=graze_delay,
            duration=_integer(row, "duration"),
            despawn_duration=_integer(row, "end_time"),
            hitbox_end_delay=_integer(row, "graze_interval"),
            timer=timer,
            timer_float=float(timer),
            flags=_integer(row, "flags"),
            state=state,
        ))
    return laser_rects_by_frame(lasers, horizon)


def lower_headless_hazards(
    observation: Mapping[str, Any],
    horizon: int = HARD_HORIZON,
) -> PackedHazards:
    if horizon < 1:
        raise ValueError("headless hazard horizon must be positive")
    validate_headless_observation(observation)
    player = observation["player"]
    assert isinstance(player, Mapping)
    x = _finite_number(player, "x")
    y = _finite_number(player, "y")
    half_width = _finite_number(player, "half_width")
    half_height = _finite_number(player, "half_height")
    bullet_frames = _bullet_frames(
        observation["bullets"],  # type: ignore[arg-type]
        horizon=horizon,
        player_x=x,
        player_y=y,
        player_half_width=half_width,
        player_half_height=half_height,
    )
    enemy_frames = _enemy_frames(observation["enemies"], horizon)  # type: ignore[arg-type]
    return PackedHazards(
        aabb_frames=tuple(
            tuple(bullets + enemies)
            for bullets, enemies in zip(bullet_frames, enemy_frames, strict=True)
        ),
        laser_frames=_laser_frames(observation["lasers"], horizon),  # type: ignore[arg-type]
    )


def certify_headless_actions(
    observation: Mapping[str, Any],
    *,
    kernel: NativeKernel | None = None,
    horizon: int = HARD_HORIZON,
) -> tuple[NativeCertifiedAction, ...]:
    hazards = lower_headless_hazards(observation, horizon)
    player = observation["player"]
    assert isinstance(player, Mapping)
    return (kernel or NativeKernel()).certify_actions(
        x=_finite_number(player, "x"),
        y=_finite_number(player, "y"),
        half_width=_finite_number(player, "half_width"),
        half_height=_finite_number(player, "half_height"),
        kinematics=KINEMATICS,
        current_action=action_from_input(_integer(observation, "input")),
        hazards=hazards,
        collision_margin=COLLISION_MARGIN,
    )


def reactive_headless_action(
    observation: Mapping[str, Any],
    certified: tuple[NativeCertifiedAction, ...],
) -> Action:
    if not certified:
        raise HeadlessAuthorityUnavailable("headless native safe set is empty")
    current = action_from_input(_integer(observation, "input"))

    def boundary_reserve(item: NativeCertifiedAction) -> float:
        return min(
            item.final_x - 8.0,
            376.0 - item.final_x,
            item.final_y - 16.0,
            432.0 - item.final_y,
        )

    return max(
        certified,
        key=lambda item: (
            item.min_clearance,
            boundary_reserve(item),
            item.action == current,
            item.action.dx == 0 and item.action.dy == 0,
            item.action.focused,
            item.action.name,
        ),
    ).action

