"""Source-exact projections for observed TH06 bullet motion modes.

The donor hazard model deliberately fails closed for EX modes that it does
not simulate.  That fallback is appropriate for unknown motion, but is far
too broad for the source's deterministic boundary-reflection flags: an
in-bounds 0x400/0x800 bullet continues linearly until its visual sprite has
fully left the 384x448 playfield.

This narrow adapter adds the missing deterministic modes without changing
the donor's conservative prefilter or its spawning-bullet uncertainty.
"""

from __future__ import annotations

import math
import struct
from dataclasses import replace

from .donor import enable_donor_imports


SLOWDOWN_FLAG = 0x001
ACCELERATION_FLAG = 0x010
CURVE_ACCELERATION_FLAG = 0x020
RELATIVE_TURN_FLAG = 0x040
PLAYER_AIM_TURN_FLAG = 0x080
ABSOLUTE_TURN_FLAG = 0x100
INERT_FLAG = 0x200
REFLECT_ALL_EDGES_FLAG = 0x400
REFLECT_NO_BOTTOM_FLAG = 0x800
SPAWN_EFFECT_FLAGS = 0x00E
DYNAMIC_EX_FLAGS = 0xDF1
KNOWN_EX_FLAGS = 0xFFF

_LOCAL_EXACT_FLAGS = (
    ABSOLUTE_TURN_FLAG
    | REFLECT_ALL_EDGES_FLAG
    | REFLECT_NO_BOTTOM_FLAG
)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _normalize_angle(value: float) -> float:
    value = _f32(value)
    iterations = 0
    while value > math.pi:
        value = _f32(value - math.tau)
        iterations += 1
        if iterations > 17:
            break
    while value < -math.pi:
        value = _f32(value + math.tau)
        iterations += 1
        if iterations > 17:
            break
    return value


def _is_in_bounds(
    x: float,
    y: float,
    sprite_half_width: float,
    sprite_half_height: float,
) -> bool:
    # GameManager::IsInBounds uses the visual sprite, not the kill box.
    return not (
        sprite_half_width + x < 0.0
        or x - sprite_half_width > 384.0
        or sprite_half_height + y < 0.0
        or y - sprite_half_height > 448.0
    )


def _supports_local_exact_projection(bullet) -> bool:
    dynamic = int(bullet.ex_flags) & DYNAMIC_EX_FLAGS
    if bullet.state != 1 or not dynamic & _LOCAL_EXACT_FLAGS:
        return False
    if dynamic & PLAYER_AIM_TURN_FLAG:
        # 0x80 retargets toward the future physical player.  A bullet-only
        # projection cannot know the candidate-conditioned target.
        return False
    if dynamic & (REFLECT_ALL_EDGES_FLAG | REFLECT_NO_BOTTOM_FLAG):
        return (
            math.isfinite(bullet.sprite_half_width)
            and math.isfinite(bullet.sprite_half_height)
            and bullet.sprite_half_width > 0.0
            and bullet.sprite_half_height > 0.0
        )
    return True


def _source_exact_positions(bullet, horizon: int) -> list[tuple[float, float]]:
    """Step the source's two ordered EX-motion chains at 1x game speed."""
    x, y = _f32(bullet.x), _f32(bullet.y)
    vx, vy = _f32(bullet.vx), _f32(bullet.vy)
    angle = _f32(bullet.angle)
    speed = _f32(bullet.speed)
    turn_speed = _f32(bullet.turn_speed)
    timer = int(bullet.timer)
    timer_float = _f32(bullet.timer_float)
    direction_num_times = int(bullet.direction_num_times)
    flags = int(bullet.ex_flags)
    result: list[tuple[float, float]] = []

    for _ in range(horizon):
        # BulletManager::OnUpdate: 0x1, else 0x10, else 0x20.
        if flags & SLOWDOWN_FLAG:
            if timer <= 16:
                slowdown = _f32(5.0 - _f32(timer_float * 5.0 / 16.0))
                current_speed = _f32(slowdown + speed)
                vx = _f32(math.cos(angle) * current_speed)
                vy = _f32(math.sin(angle) * current_speed)
            else:
                flags ^= SLOWDOWN_FLAG
        elif flags & ACCELERATION_FLAG:
            if timer >= bullet.acceleration_duration:
                flags &= ~ACCELERATION_FLAG
            else:
                vx = _f32(vx + bullet.acceleration_x)
                vy = _f32(vy + bullet.acceleration_y)
                angle = _f32(math.atan2(vy, vx))
        elif flags & CURVE_ACCELERATION_FLAG:
            if timer >= bullet.acceleration_duration:
                flags &= ~CURVE_ACCELERATION_FLAG
            else:
                angle = _normalize_angle(
                    _f32(angle + bullet.curve_angular_velocity)
                )
                speed = _f32(speed + bullet.curve_speed_acceleration)
                vx = _f32(math.cos(angle) * speed)
                vy = _f32(math.sin(angle) * speed)

        # Then 0x40, else 0x100, else 0x80, else 0x400, else 0x800.
        direction_flag = 0
        if flags & RELATIVE_TURN_FLAG:
            direction_flag = RELATIVE_TURN_FLAG
        elif flags & ABSOLUTE_TURN_FLAG:
            direction_flag = ABSOLUTE_TURN_FLAG

        if direction_flag:
            interval = int(bullet.direction_interval)
            threshold = interval * (direction_num_times + 1)
            if timer >= threshold:
                direction_num_times += 1
                if direction_num_times >= int(bullet.direction_max_times):
                    flags &= ~direction_flag
                if direction_flag == RELATIVE_TURN_FLAG:
                    angle = _f32(angle + bullet.direction_rotation)
                else:
                    angle = _f32(bullet.direction_rotation)
                speed = turn_speed
                current_speed = speed
            else:
                # A non-positive interval always takes the threshold branch.
                current_speed = _f32(
                    speed
                    - _f32(
                        (timer_float - interval * direction_num_times)
                        * speed
                        / interval
                    )
                )
            vx = _f32(math.cos(angle) * current_speed)
            vy = _f32(math.sin(angle) * current_speed)
        elif flags & PLAYER_AIM_TURN_FLAG:
            raise AssertionError("player-aimed 0x80 cannot be projected here")
        elif flags & (REFLECT_ALL_EDGES_FLAG | REFLECT_NO_BOTTOM_FLAG):
            reflect_flag = (
                REFLECT_ALL_EDGES_FLAG
                if flags & REFLECT_ALL_EDGES_FLAG
                else REFLECT_NO_BOTTOM_FLAG
            )
            if not _is_in_bounds(
                x,
                y,
                bullet.sprite_half_width,
                bullet.sprite_half_height,
            ):
                if x < 0.0 or x >= 384.0:
                    angle = _normalize_angle(_f32(-angle - math.pi))
                if y < 0.0 or (
                    reflect_flag == REFLECT_ALL_EDGES_FLAG and y >= 448.0
                ):
                    angle = _f32(-angle)
                speed = turn_speed
                vx = _f32(math.cos(angle) * speed)
                vy = _f32(math.sin(angle) * speed)
                direction_num_times += 1
                if direction_num_times >= int(bullet.direction_max_times):
                    flags &= ~reflect_flag

        x = _f32(x + vx)
        y = _f32(y + vy)
        timer += 1
        timer_float = _f32(timer_float + 1.0)
        result.append((x, y))
    return result


def hazard_boxes(bullet, horizon: int):
    enable_donor_imports()
    from th06.hazards.bullets import hazard_boxes as donor_hazard_boxes

    if bullet.state in (2, 3, 4):
        return _spawn_transition_hazard_boxes(
            bullet,
            horizon,
            donor_hazard_boxes,
        )
    if not _supports_local_exact_projection(bullet):
        return donor_hazard_boxes(bullet, horizon)
    half_width = bullet.half_width
    half_height = bullet.half_height
    return [
        (x - half_width, y - half_height, x + half_width, y + half_height)
        for x, y in _source_exact_positions(bullet, horizon)
    ]


def _fired_hazard_boxes(bullet, horizon: int, donor_hazard_boxes):
    if _supports_local_exact_projection(bullet):
        half_width = bullet.half_width
        half_height = bullet.half_height
        return [
            (x - half_width, y - half_height, x + half_width, y + half_height)
            for x, y in _source_exact_positions(bullet, horizon)
        ]
    return donor_hazard_boxes(bullet, horizon)


def _spawn_transition_hazard_boxes(bullet, horizon: int, donor_hazard_boxes):
    """Enclose every possible spawn-animation completion within ``horizon``.

    The compact control snapshot intentionally does not copy the large ANM VM
    program needed to know the exact completion tick.  Authoritative
    BulletManager behavior still gives a tight safe set: states 2/3/4 are not
    collidable, move at 1/2, 1/2.5, or 1/3 velocity, and on the completion
    update fall through to fired state with a reset timer.  Enumerating every
    possible completion update preserves that uncertainty without inventing
    arbitrary-direction motion.
    """
    divisor = {2: 2.0, 3: 2.5, 4: 3.0}[int(bullet.state)]
    spawn_dx = _f32(_f32(bullet.vx) / divisor)
    spawn_dy = _f32(_f32(bullet.vy) / divisor)
    spawn_x = _f32(bullet.x)
    spawn_y = _f32(bullet.y)
    possible: list[list[tuple[float, float, float, float]]] = [
        [] for _ in range(horizon)
    ]

    for transition_frame in range(1, horizon + 1):
        # The partial spawn movement happens even on the update whose ANM
        # script completes, before the source falls through to fired motion.
        spawn_x = _f32(spawn_x + spawn_dx)
        spawn_y = _f32(spawn_y + spawn_dy)
        fired = replace(
            bullet,
            x=spawn_x,
            y=spawn_y,
            state=1,
            timer=0,
            timer_float=0.0,
        )
        fired_boxes = _fired_hazard_boxes(
            fired,
            horizon - transition_frame + 1,
            donor_hazard_boxes,
        )
        for offset, box in enumerate(fired_boxes):
            possible[transition_frame - 1 + offset].append(box)

    result = []
    for boxes in possible:
        # There is always at least the branch that completes on this frame.
        result.append((
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ))
    return result


def hazard_box(bullet, frame: int):
    if frame < 1:
        raise ValueError("bullet hazard frame must be positive")
    return hazard_boxes(bullet, frame)[-1]


def reachable_hazards_by_frame(snapshot, horizon: int, collision_margin: float):
    """Keep the donor's broad prefilter, then use the exact local boxes."""
    enable_donor_imports()
    from th06.hazards.bullets import _may_reach_player

    frames: list[list[tuple[float, float, float, float]]] = [
        [] for _ in range(horizon)
    ]
    for bullet in snapshot.bullets:
        if not _may_reach_player(snapshot, bullet, horizon, collision_margin):
            continue
        for frame, box in zip(frames, hazard_boxes(bullet, horizon)):
            frame.append(box)
    return [tuple(frame) for frame in frames]


def classify_ex_flags(ex_flags: int) -> str:
    """Classify fired-bullet projection coverage for an offline census."""
    flags = int(ex_flags)
    if flags & ~KNOWN_EX_FLAGS:
        return "unknown-flag-fail-closed"
    dynamic = flags & DYNAMIC_EX_FLAGS
    if not dynamic:
        return "source-exact-linear"
    if dynamic & PLAYER_AIM_TURN_FLAG:
        return "conservative-player-retarget"
    if dynamic & _LOCAL_EXACT_FLAGS:
        return "source-exact-local"
    return "source-exact-donor"
