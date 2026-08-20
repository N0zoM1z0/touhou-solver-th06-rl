"""Source-grounded future boxes for native TH06 bullets."""

from __future__ import annotations

import math
import struct

from ..model import Bullet, Snapshot
from .geometry import signed_clearance


DYNAMIC_EX_FLAGS = 0xDF1
SLOWDOWN_FLAG = 0x01
ACCELERATION_FLAG = 0x10
CURVE_ACCELERATION_FLAG = 0x20
COMPLEX_MOTION_FLAGS = 0xDE1
DIRECTION_ROTATION_FLAG = 0x40
SOURCE_EXACT_DYNAMIC_FLAGS = (
    SLOWDOWN_FLAG
    | ACCELERATION_FLAG
    | CURVE_ACCELERATION_FLAG
    | DIRECTION_ROTATION_FLAG
)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _source_dynamic_positions(
    bullet: Bullet,
    horizon: int,
) -> list[tuple[float, float]]:
    """Step the source's ordered 0x01/0x10/0x20 plus 0x40 state machine."""
    x, y = _f32(bullet.x), _f32(bullet.y)
    vx, vy = _f32(bullet.vx), _f32(bullet.vy)
    angle = _f32(bullet.angle)
    speed = _f32(bullet.speed)
    turn_speed = _f32(bullet.turn_speed)
    flags = bullet.ex_flags
    timer = bullet.timer
    timer_float = _f32(bullet.timer_float)
    direction_num_times = bullet.direction_num_times
    result = []
    for _ in range(horizon):
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
                angle = _f32(math.remainder(
                    _f32(angle + bullet.curve_angular_velocity), math.tau
                ))
                speed = _f32(speed + bullet.curve_speed_acceleration)
                vx = _f32(math.cos(angle) * speed)
                vy = _f32(math.sin(angle) * speed)
        if flags & DIRECTION_ROTATION_FLAG:
            if timer >= bullet.direction_interval * (direction_num_times + 1):
                direction_num_times += 1
                if direction_num_times >= bullet.direction_max_times:
                    flags &= ~DIRECTION_ROTATION_FLAG
                angle = _f32(angle + bullet.direction_rotation)
                speed = turn_speed
                current_speed = speed
            else:
                phase = _f32(
                    timer_float - bullet.direction_interval * direction_num_times
                )
                current_speed = _f32(
                    speed - _f32(phase * speed / bullet.direction_interval)
                )
            vx = _f32(math.cos(angle) * current_speed)
            vy = _f32(math.sin(angle) * current_speed)
        x = _f32(x + vx)
        y = _f32(y + vy)
        timer += 1
        timer_float = _f32(timer_float + 1.0)
        result.append((x, y))
    return result


def _slowdown_positions(
    bullet: Bullet,
    horizon: int,
) -> list[tuple[float, float]]:
    """Reproduce source flag 0x01's deterministic first-17-tick speed."""
    x, y = _f32(bullet.x), _f32(bullet.y)
    vx, vy = _f32(bullet.vx), _f32(bullet.vy)
    angle = _f32(bullet.angle)
    speed = _f32(bullet.speed)
    timer = bullet.timer
    timer_float = _f32(bullet.timer_float)
    active = True
    cosine = _f32(math.cos(angle))
    sine = _f32(math.sin(angle))
    result = []
    for _ in range(horizon):
        if active:
            if timer <= 16:
                slowdown = _f32(5.0 - _f32(timer_float * 5.0 / 16.0))
                current_speed = _f32(slowdown + speed)
                vx = _f32(cosine * current_speed)
                vy = _f32(sine * current_speed)
            else:
                active = False
        x = _f32(x + vx)
        y = _f32(y + vy)
        timer += 1
        timer_float = _f32(timer_float + 1.0)
        result.append((x, y))
    return result


def _slowdown_position(bullet: Bullet, frame: int) -> tuple[float, float]:
    return _slowdown_positions(bullet, frame)[-1]


def _slowdown_acceleration_positions(
    bullet: Bullet,
    horizon: int,
) -> list[tuple[float, float]]:
    """Exact source priority for the common 0x01 then 0x10 chain."""
    x, y = _f32(bullet.x), _f32(bullet.y)
    vx, vy = _f32(bullet.vx), _f32(bullet.vy)
    speed = _f32(bullet.speed)
    timer = bullet.timer
    timer_float = _f32(bullet.timer_float)
    slowdown_active = True
    acceleration_active = True
    cosine = _f32(math.cos(_f32(bullet.angle)))
    sine = _f32(math.sin(_f32(bullet.angle)))
    result = []
    for _ in range(horizon):
        if slowdown_active:
            if timer <= 16:
                slowdown = _f32(5.0 - _f32(timer_float * 5.0 / 16.0))
                current_speed = _f32(slowdown + speed)
                vx = _f32(cosine * current_speed)
                vy = _f32(sine * current_speed)
            else:
                # Source clears 0x01 on this update; its else-if means 0x10
                # begins only on the following update.
                slowdown_active = False
        elif acceleration_active:
            if timer >= bullet.acceleration_duration:
                acceleration_active = False
            else:
                vx = _f32(vx + bullet.acceleration_x)
                vy = _f32(vy + bullet.acceleration_y)
        x = _f32(x + vx)
        y = _f32(y + vy)
        timer += 1
        timer_float = _f32(timer_float + 1.0)
        result.append((x, y))
    return result


def _direction_rotation_positions(
    bullet: Bullet,
    horizon: int,
) -> list[tuple[float, float]]:
    """Reproduce the source's timed 0x40 decelerate-and-rotate update."""
    x, y = bullet.x, bullet.y
    vx, vy = bullet.vx, bullet.vy
    angle = bullet.angle
    speed = bullet.speed
    timer = bullet.timer
    timer_float = bullet.timer_float
    direction_num_times = bullet.direction_num_times
    flags = bullet.ex_flags
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result = []
    for _ in range(horizon):
        if flags & DIRECTION_ROTATION_FLAG:
            if timer >= bullet.direction_interval * (direction_num_times + 1):
                direction_num_times += 1
                if direction_num_times >= bullet.direction_max_times:
                    flags &= ~DIRECTION_ROTATION_FLAG
                angle += bullet.direction_rotation
                cosine = math.cos(angle)
                sine = math.sin(angle)
                speed = bullet.turn_speed
                bullet_speed = speed
            else:
                phase = timer_float - bullet.direction_interval * direction_num_times
                bullet_speed = speed - phase * speed / bullet.direction_interval
            vx = cosine * bullet_speed
            vy = sine * bullet_speed
        x += vx
        y += vy
        timer += 1
        timer_float += 1.0
        result.append((x, y))
    return result


def _direction_rotation_position(bullet: Bullet, frame: int) -> tuple[float, float]:
    return _direction_rotation_positions(bullet, frame)[-1]


def _curve_acceleration_positions(
    bullet: Bullet,
    horizon: int,
) -> list[tuple[float, float]]:
    """Reproduce source flag 0x20 angular and speed acceleration."""
    x, y = bullet.x, bullet.y
    vx, vy = bullet.vx, bullet.vy
    angle = bullet.angle
    speed = bullet.speed
    timer = bullet.timer
    active = True
    result = []
    for _ in range(horizon):
        if active:
            if timer >= bullet.acceleration_duration:
                active = False
            else:
                angle += bullet.curve_angular_velocity
                speed += bullet.curve_speed_acceleration
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
        x += vx
        y += vy
        timer += 1
        result.append((x, y))
    return result


def _curve_acceleration_position(bullet: Bullet, frame: int) -> tuple[float, float]:
    return _curve_acceleration_positions(bullet, frame)[-1]


def hazard_box(bullet: Bullet, frame: int) -> tuple[float, float, float, float]:
    dynamic_flags = bullet.ex_flags & DYNAMIC_EX_FLAGS
    if (
        bullet.state == 1
        and dynamic_flags == SLOWDOWN_FLAG
    ):
        x, y = _slowdown_position(bullet, frame)
        return (
            x - bullet.half_width,
            y - bullet.half_height,
            x + bullet.half_width,
            y + bullet.half_height,
        )
    if (
        bullet.state == 1
        and dynamic_flags == (SLOWDOWN_FLAG | ACCELERATION_FLAG)
    ):
        x, y = _slowdown_acceleration_positions(bullet, frame)[-1]
        return (
            x - bullet.half_width,
            y - bullet.half_height,
            x + bullet.half_width,
            y + bullet.half_height,
        )
    if (
        bullet.state == 1
        and bullet.ex_flags & DIRECTION_ROTATION_FLAG
        and not bullet.ex_flags & (COMPLEX_MOTION_FLAGS & ~DIRECTION_ROTATION_FLAG)
    ):
        x, y = _direction_rotation_position(bullet, frame)
        return (
            x - bullet.half_width,
            y - bullet.half_height,
            x + bullet.half_width,
            y + bullet.half_height,
        )
    if (
        bullet.state == 1
        and bullet.ex_flags & ACCELERATION_FLAG
        and not bullet.ex_flags & COMPLEX_MOTION_FLAGS
        and bullet.acceleration_duration > 0
    ):
        # BulletManager::OnUpdate applies ex4Acceleration before position only
        # while timer.current < ex5Int0. Spawn-effect bits are inert once fired.
        applications = min(
            frame,
            max(0, bullet.acceleration_duration - bullet.timer),
        )
        acceleration_factor = (
            applications * frame - applications * (applications - 1) / 2.0
        )
        x = bullet.x + bullet.vx * frame + bullet.acceleration_x * acceleration_factor
        y = bullet.y + bullet.vy * frame + bullet.acceleration_y * acceleration_factor
        return (
            x - bullet.half_width,
            y - bullet.half_height,
            x + bullet.half_width,
            y + bullet.half_height,
        )
    if (
        bullet.state == 1
        and bullet.ex_flags & CURVE_ACCELERATION_FLAG
        and not bullet.ex_flags & (COMPLEX_MOTION_FLAGS & ~CURVE_ACCELERATION_FLAG)
    ):
        x, y = _curve_acceleration_position(bullet, frame)
        return (
            x - bullet.half_width,
            y - bullet.half_height,
            x + bullet.half_width,
            y + bullet.half_height,
        )
    if (
        bullet.state == 1
        and dynamic_flags
        and not dynamic_flags & ~SOURCE_EXACT_DYNAMIC_FLAGS
    ):
        x, y = _source_dynamic_positions(bullet, frame)[-1]
        return (
            x - bullet.half_width,
            y - bullet.half_height,
            x + bullet.half_width,
            y + bullet.half_height,
        )
    if bullet.ex_flags & DYNAMIC_EX_FLAGS:
        # Extended bullets may accelerate, turn, home, or bounce, but do not
        # teleport. Cover every direction using the source-visible speed fields.
        base_speed = max(math.hypot(bullet.vx, bullet.vy), abs(bullet.speed), abs(bullet.turn_speed))
        reach = (base_speed + 5.0) * frame + 0.5 * abs(bullet.acceleration) * frame * frame
        return (
            bullet.x - bullet.half_width - reach,
            bullet.y - bullet.half_height - reach,
            bullet.x + bullet.half_width + reach,
            bullet.y + bullet.half_height + reach,
        )

    if bullet.state == 1:
        minimum_factor = 1.0
    elif bullet.state == 2:
        minimum_factor = 0.5
    elif bullet.state == 3:
        minimum_factor = 0.4
    else:
        minimum_factor = 1.0 / 3.0
    x0 = bullet.x + bullet.vx * frame * minimum_factor
    y0 = bullet.y + bullet.vy * frame * minimum_factor
    x1 = bullet.x + bullet.vx * frame
    y1 = bullet.y + bullet.vy * frame
    return (
        min(x0, x1) - bullet.half_width,
        min(y0, y1) - bullet.half_height,
        max(x0, x1) + bullet.half_width,
        max(y0, y1) + bullet.half_height,
    )


def radial_hazard_box(
    bullet: Bullet,
    frame: int,
) -> tuple[float, float, float, float]:
    """Enclose a newborn bullet without assuming its future aim angle."""
    base_speed = max(
        math.hypot(bullet.vx, bullet.vy),
        abs(bullet.speed),
        abs(bullet.turn_speed),
    )
    acceleration = max(
        abs(bullet.acceleration),
        math.hypot(bullet.acceleration_x, bullet.acceleration_y),
        abs(bullet.curve_speed_acceleration),
    )
    reach = base_speed * frame + acceleration * frame * (frame + 1) / 2.0
    if bullet.ex_flags & (DYNAMIC_EX_FLAGS & ~(
        ACCELERATION_FLAG | CURVE_ACCELERATION_FLAG | DIRECTION_ROTATION_FLAG
    )):
        # Source-visible extended modes may turn, home, or bounce. The existing
        # hard model bounds those modes by five extra pixels per update.
        reach += 5.0 * frame
    return (
        bullet.x - bullet.half_width - reach,
        bullet.y - bullet.half_height - reach,
        bullet.x + bullet.half_width + reach,
        bullet.y + bullet.half_height + reach,
    )


def hazard_boxes(
    bullet: Bullet,
    horizon: int,
) -> list[tuple[float, float, float, float]]:
    """Project one bullet once while preserving ``hazard_box`` semantics."""
    state = bullet.state
    flags = bullet.ex_flags
    dynamic_flags = flags & DYNAMIC_EX_FLAGS
    half_width = bullet.half_width
    half_height = bullet.half_height
    if (
        state == 1
        and dynamic_flags == SLOWDOWN_FLAG
    ):
        return [
            (x - half_width, y - half_height, x + half_width, y + half_height)
            for x, y in _slowdown_positions(bullet, horizon)
        ]
    if (
        state == 1
        and dynamic_flags == (SLOWDOWN_FLAG | ACCELERATION_FLAG)
    ):
        return [
            (x - half_width, y - half_height, x + half_width, y + half_height)
            for x, y in _slowdown_acceleration_positions(bullet, horizon)
        ]
    if (
        state == 1
        and flags & DIRECTION_ROTATION_FLAG
        and not flags & (COMPLEX_MOTION_FLAGS & ~DIRECTION_ROTATION_FLAG)
    ):
        return [
            (x - half_width, y - half_height, x + half_width, y + half_height)
            for x, y in _direction_rotation_positions(bullet, horizon)
        ]
    if (
        state == 1
        and flags & CURVE_ACCELERATION_FLAG
        and not flags & (COMPLEX_MOTION_FLAGS & ~CURVE_ACCELERATION_FLAG)
    ):
        return [
            (x - half_width, y - half_height, x + half_width, y + half_height)
            for x, y in _curve_acceleration_positions(bullet, horizon)
        ]
    frames = range(1, horizon + 1)
    if (
        state == 1
        and flags & ACCELERATION_FLAG
        and not flags & COMPLEX_MOTION_FLAGS
        and bullet.acceleration_duration > 0
    ):
        remaining = max(0, bullet.acceleration_duration - bullet.timer)
        boxes = []
        for frame in frames:
            applications = min(frame, remaining)
            factor = (
                applications * frame - applications * (applications - 1) / 2.0
            )
            x = bullet.x + bullet.vx * frame + bullet.acceleration_x * factor
            y = bullet.y + bullet.vy * frame + bullet.acceleration_y * factor
            boxes.append((
                x - half_width,
                y - half_height,
                x + half_width,
                y + half_height,
            ))
        return boxes
    if (
        state == 1
        and dynamic_flags
        and not dynamic_flags & ~SOURCE_EXACT_DYNAMIC_FLAGS
    ):
        return [
            (x - half_width, y - half_height, x + half_width, y + half_height)
            for x, y in _source_dynamic_positions(bullet, horizon)
        ]
    if flags & DYNAMIC_EX_FLAGS:
        base_speed = max(
            math.hypot(bullet.vx, bullet.vy),
            abs(bullet.speed),
            abs(bullet.turn_speed),
        )
        acceleration = abs(bullet.acceleration)
        boxes = []
        for frame in frames:
            reach = (base_speed + 5.0) * frame + 0.5 * acceleration * frame * frame
            boxes.append((
                bullet.x - half_width - reach,
                bullet.y - half_height - reach,
                bullet.x + half_width + reach,
                bullet.y + half_height + reach,
            ))
        return boxes
    return _linear_hazard_boxes(bullet, 0, horizon)


def _linear_hazard_boxes(
    bullet: Bullet,
    completed_horizon: int,
    total_horizon: int,
) -> list[tuple[float, float, float, float]]:
    """Project only unseen ages for a bullet with no dynamic EX motion."""
    frames = range(completed_horizon + 1, total_horizon + 1)
    state = bullet.state
    half_width = bullet.half_width
    half_height = bullet.half_height
    if state == 1:
        x = bullet.x
        y = bullet.y
        vx = bullet.vx
        vy = bullet.vy
        return [
            (
                x + vx * frame - half_width,
                y + vy * frame - half_height,
                x + vx * frame + half_width,
                y + vy * frame + half_height,
            )
            for frame in frames
        ]
    if state == 2:
        minimum_factor = 0.5
    elif state == 3:
        minimum_factor = 0.4
    else:
        minimum_factor = 1.0 / 3.0
    boxes = []
    for frame in frames:
        x0 = bullet.x + bullet.vx * frame * minimum_factor
        y0 = bullet.y + bullet.vy * frame * minimum_factor
        x1 = bullet.x + bullet.vx * frame
        y1 = bullet.y + bullet.vy * frame
        boxes.append((
            min(x0, x1) - half_width,
            min(y0, y1) - half_height,
            max(x0, x1) + half_width,
            max(y0, y1) + half_height,
        ))
    return boxes


def _may_reach_player(
    snapshot: Snapshot,
    bullet: Bullet,
    horizon: int,
    collision_margin: float,
) -> bool:
    """Conservatively reject bullets outside the whole reachable sweep.

    ``BulletManager::OnUpdate`` changes only velocity/angle state before its
    per-frame translation.  The dynamic radius below encloses the broader
    fail-closed boxes already emitted by :func:`hazard_boxes`; linear motion
    uses an axis-wise displacement bound.  Intersecting envelopes are always
    retained and the later per-frame filter remains authoritative.
    """
    player_speed = max(
        abs(snapshot.normal_speed),
        abs(snapshot.focus_speed),
        abs(snapshot.normal_diagonal_speed),
        abs(snapshot.focus_diagonal_speed),
    )
    margin = max(0.0, collision_margin)
    player_left = (
        max(8.0, snapshot.x - player_speed * horizon)
        - snapshot.half_width
        - margin
    )
    player_right = (
        min(376.0, snapshot.x + player_speed * horizon)
        + snapshot.half_width
        + margin
    )
    player_top = (
        max(16.0, snapshot.y - player_speed * horizon)
        - snapshot.half_height
        - margin
    )
    player_bottom = (
        min(432.0, snapshot.y + player_speed * horizon)
        + snapshot.half_height
        + margin
    )

    if bullet.ex_flags & DYNAMIC_EX_FLAGS:
        base_speed = max(
            math.hypot(bullet.vx, bullet.vy),
            abs(bullet.speed),
            abs(bullet.turn_speed),
        )
        acceleration = max(
            abs(bullet.acceleration),
            math.hypot(
                bullet.acceleration_x,
                bullet.acceleration_y,
            ),
            abs(bullet.curve_speed_acceleration),
        )
        reach_x = reach_y = (
            (base_speed + 5.0) * horizon
            + acceleration * horizon * (horizon + 1) / 2.0
        )
    else:
        reach_x = abs(bullet.vx) * horizon
        reach_y = abs(bullet.vy) * horizon
    bullet_left = bullet.x - bullet.half_width - reach_x
    bullet_right = bullet.x + bullet.half_width + reach_x
    bullet_top = bullet.y - bullet.half_height - reach_y
    bullet_bottom = bullet.y + bullet.half_height + reach_y
    return not (
        bullet_right < player_left
        or bullet_left > player_right
        or bullet_bottom < player_top
        or bullet_top > player_bottom
    )


def reachable_hazards_by_frame(
    snapshot: Snapshot,
    horizon: int,
    collision_margin: float,
) -> list[tuple[tuple[float, float, float, float], ...]]:
    """Project only bullets whose conservative sweep can reach the player."""
    frames: list[list[tuple[float, float, float, float]]] = [
        [] for _ in range(horizon)
    ]
    for bullet in snapshot.bullets:
        if not _may_reach_player(
            snapshot, bullet, horizon, collision_margin
        ):
            continue
        for frame, box in zip(frames, hazard_boxes(bullet, horizon)):
            frame.append(box)
    return [tuple(frame) for frame in frames]


def extend_reachable_hazards_by_frame(
    snapshot: Snapshot,
    prefix: list[tuple[tuple[float, float, float, float], ...]],
    total_horizon: int,
    collision_margin: float,
) -> list[tuple[tuple[float, float, float, float], ...]]:
    """Extend a reachable prefix, admitting newly reachable bullet sweeps."""
    completed_horizon = len(prefix)
    if total_horizon < completed_horizon:
        raise ValueError("bullet hazard horizon cannot shrink during extension")
    frames = [list(frame) for frame in prefix]
    frames.extend([] for _ in range(total_horizon - completed_horizon))
    if total_horizon == completed_horizon:
        return [tuple(frame) for frame in frames]
    for bullet in snapshot.bullets:
        if not _may_reach_player(
            snapshot, bullet, total_horizon, collision_margin
        ):
            continue
        boxes = (
            hazard_boxes(bullet, total_horizon)[completed_horizon:]
            if bullet.ex_flags & DYNAMIC_EX_FLAGS
            else _linear_hazard_boxes(
                bullet,
                completed_horizon,
                total_horizon,
            )
        )
        for frame, box in zip(frames[completed_horizon:], boxes):
            frame.append(box)
    return [tuple(frame) for frame in frames]


def hazards_by_frame(snapshot: Snapshot, horizon: int) -> list[tuple[tuple[float, float, float, float], ...]]:
    frames: list[list[tuple[float, float, float, float]]] = [
        [] for _ in range(horizon)
    ]
    for bullet in snapshot.bullets:
        for frame, box in zip(frames, hazard_boxes(bullet, horizon)):
            frame.append(box)
    return [tuple(frame) for frame in frames]


def extend_hazards_by_frame(
    snapshot: Snapshot,
    prefix: list[tuple[tuple[float, float, float, float], ...]],
    total_horizon: int,
) -> list[tuple[tuple[float, float, float, float], ...]]:
    """Extend an exact current-bullet prefix without rebuilding its frames."""
    completed_horizon = len(prefix)
    if total_horizon < completed_horizon:
        raise ValueError("bullet hazard horizon cannot shrink during extension")
    frames = [list(frame) for frame in prefix]
    frames.extend([] for _ in range(total_horizon - completed_horizon))
    if total_horizon == completed_horizon:
        return [tuple(frame) for frame in frames]
    for bullet in snapshot.bullets:
        boxes = (
            hazard_boxes(bullet, total_horizon)[completed_horizon:]
            if bullet.ex_flags & DYNAMIC_EX_FLAGS
            else _linear_hazard_boxes(
                bullet,
                completed_horizon,
                total_horizon,
            )
        )
        for frame, box in zip(frames[completed_horizon:], boxes):
            frame.append(box)
    return [tuple(frame) for frame in frames]


def nearest_current_clearance(snapshot: Snapshot) -> float:
    nearest = 999.0
    for bullet in snapshot.bullets:
        hazard = (
            bullet.x - bullet.half_width,
            bullet.y - bullet.half_height,
            bullet.x + bullet.half_width,
            bullet.y + bullet.half_height,
        )
        nearest = min(
            nearest,
            signed_clearance(snapshot.x, snapshot.y, snapshot.half_width, snapshot.half_height, hazard),
        )
    return nearest
