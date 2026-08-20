"""Exact TH06 laser phase, segment, and rotated hitbox model."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..model import Laser, Snapshot
from .geometry import signed_clearance


@dataclass(frozen=True)
class LaserHazard:
    origin_x: float
    origin_y: float
    angle: float
    center_offset: float
    size_x: float
    size_y: float


def track_motion(
    previous: tuple[Laser, ...],
    current: tuple[Laser, ...],
    frame_delta: int,
) -> tuple[Laser, ...]:
    """Infer source ECL laser rotation from stable native pool slots."""
    if frame_delta <= 0:
        return current
    by_slot = {laser.slot: laser for laser in previous if laser.slot >= 0}
    tracked = []
    for laser in current:
        prior = by_slot.get(laser.slot)
        if prior is None:
            tracked.append(laser)
            continue
        angle_delta = math.remainder(laser.angle - prior.angle, math.tau)
        tracked.append(replace(
            laser,
            angular_velocity=angle_delta / frame_delta,
            motion_known=True,
        ))
    return tuple(tracked)


def unknown_motion_may_reach_player(
    snapshot: Snapshot,
    laser: Laser,
    horizon: int,
) -> bool:
    """Conservatively bound a new laser at every possible future angle."""
    if not any(future_hazards(laser, horizon)):
        return False
    future_end = laser.end_offset + max(0.0, laser.speed) * horizon
    beam_radius = max(
        0.0,
        abs(laser.start_offset),
        abs(laser.end_offset),
        abs(future_end),
    ) + laser.width / 2.0
    player_radius = (
        max(snapshot.normal_speed, snapshot.focus_speed) * horizon
        + math.hypot(snapshot.half_width, snapshot.half_height)
    )
    origin_distance = math.hypot(snapshot.x - laser.x, snapshot.y - laser.y)
    return origin_distance <= beam_radius + player_radius


def _geometry(
    laser: Laser,
    angle: float,
    start_offset: float,
    end_offset: float,
    size_x: float,
) -> LaserHazard:
    return LaserHazard(
        laser.x,
        laser.y,
        angle,
        (end_offset - start_offset) / 2.0 + start_offset,
        max(0.0, size_x),
        laser.width / 2.0,
    )


def advance_laser(
    laser: Laser,
) -> tuple[Laser | None, tuple[LaserHazard, ...]]:
    """Run one source BulletManager laser update and retain its next state."""
    start_offset = laser.start_offset
    end_offset = laser.end_offset
    state = laser.state
    timer = laser.timer
    timer_float = laser.timer_float
    # EclManager's LASERROTATE instruction updates the stored angle before
    # BulletManager builds and tests this frame's laser hitbox.  For a live
    # native root, ``angular_velocity`` is the measured ECL write per update;
    # an exact ECL world applies that write itself and stores zero here.
    angle = laser.angle + laser.angular_velocity
    active = True
    end_offset += laser.speed
    if laser.start_length < end_offset - start_offset:
        start_offset = end_offset - laser.start_length
    start_offset = max(0.0, start_offset)
    full_length = max(0.0, end_offset - start_offset)
    frame_hazards: list[LaserHazard] = []
    state_one_size_x = full_length

    if state == 0:
        if laser.flags & 1:
            size_x = full_length
        else:
            res = min(laser.start_time, 30)
            if laser.start_time - res < timer:
                width_now = timer_float * laser.width / max(1, laser.start_time)
            else:
                width_now = 1.2
            # Shipped bug: BulletManager assigns this to laserSize.x,
            # producing a small midpoint hitbox during warmup.
            size_x = width_now / 2.0
        if timer >= laser.hitbox_start_time:
            frame_hazards.append(
                _geometry(laser, angle, start_offset, end_offset, size_x)
            )
        if timer >= laser.start_time:
            state = 1
            timer = 0
            timer_float = 0.0
            # Shipped switch fallthrough does not restore the full length
            # after the warmup branch overwrites laserSize.x.
            state_one_size_x = size_x
        else:
            return replace(
                laser,
                angle=angle,
                start_offset=start_offset,
                end_offset=end_offset,
                timer=timer + 1,
                timer_float=timer_float + 1.0,
            ), tuple(frame_hazards)

    if state == 1:
        frame_hazards.append(
            _geometry(laser, angle, start_offset, end_offset, state_one_size_x)
        )
        if timer >= laser.duration:
            state = 2
            timer = 0
            timer_float = 0.0
            if laser.despawn_duration == 0:
                active = False

    if state == 2 and active:
        if laser.flags & 1:
            size_x = full_length
        else:
            width_now = laser.width
            if laser.despawn_duration > 0:
                width_now -= timer_float * laser.width / laser.despawn_duration
            # Same shipped midpoint-hitbox bug during despawn.
            size_x = width_now / 2.0
        if timer < laser.hitbox_end_delay:
            frame_hazards.append(
                _geometry(laser, angle, start_offset, end_offset, size_x)
            )
        if timer >= laser.despawn_duration:
            active = False

    if start_offset >= 640.0:
        active = False
    if not active:
        return None, tuple(frame_hazards)
    return replace(
        laser,
        angle=angle,
        start_offset=start_offset,
        end_offset=end_offset,
        state=state,
        timer=timer + 1,
        timer_float=timer_float + 1.0,
    ), tuple(frame_hazards)


def future_hazards(laser: Laser, horizon: int) -> list[tuple[LaserHazard, ...]]:
    active: Laser | None = laser
    result: list[tuple[LaserHazard, ...]] = []
    for _frame in range(horizon):
        if active is None:
            result.append(())
            continue
        active, hazards = advance_laser(active)
        result.append(hazards)
    return result


def hazards_by_frame(lasers: tuple[Laser, ...], horizon: int) -> list[tuple[LaserHazard, ...]]:
    frames: list[list[LaserHazard]] = [[] for _ in range(horizon)]
    for laser in lasers:
        for index, hazards in enumerate(future_hazards(laser, horizon)):
            frames[index].extend(hazards)
    return [tuple(frame) for frame in frames]


def signed_laser_clearance(
    player_x: float,
    player_y: float,
    player_half_width: float,
    player_half_height: float,
    laser: LaserHazard,
) -> float:
    dx = player_x - laser.origin_x
    dy = player_y - laser.origin_y
    sine = math.sin(laser.angle)
    cosine = math.cos(laser.angle)
    local_x = cosine * dx + sine * dy
    local_y = cosine * dy - sine * dx
    hazard = (
        laser.center_offset - laser.size_x / 2.0,
        -laser.size_y / 2.0,
        laser.center_offset + laser.size_x / 2.0,
        laser.size_y / 2.0,
    )
    return signed_clearance(
        local_x, local_y, player_half_width, player_half_height, hazard
    )
