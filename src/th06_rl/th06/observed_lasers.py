"""Allocation-light projection of already-observed TH06 laser hitboxes."""

from __future__ import annotations

from collections.abc import Iterable

from ..native import LaserRect


def laser_rects_by_frame(
    lasers: Iterable[object],
    horizon: int,
) -> tuple[tuple[LaserRect, ...], ...]:
    """Project live lasers without cloning a full Laser object each step.

    The state transitions mirror authoritative ``BulletManager.cpp`` lines
    967--1099 and the established donor laser model.  ECL rotation has already
    been inferred on each captured laser; this advances that write before the
    frame hitbox, as the shipped update order requires.
    """
    if horizon < 1:
        raise ValueError("laser forecast horizon must be positive")
    frames: list[list[LaserRect]] = [[] for _ in range(horizon)]
    for laser in lasers:
        angle = laser.angle
        start_offset = laser.start_offset
        end_offset = laser.end_offset
        state = laser.state
        timer = laser.timer
        timer_float = laser.timer_float
        alive = True
        for frame in frames:
            if not alive:
                continue
            angle += laser.angular_velocity
            end_offset += laser.speed
            if laser.start_length < end_offset - start_offset:
                start_offset = end_offset - laser.start_length
            start_offset = max(0.0, start_offset)
            full_length = max(0.0, end_offset - start_offset)
            state_one_size_x = full_length
            active = True

            def append_geometry(size_x: float) -> None:
                frame.append(LaserRect(
                    laser.x,
                    laser.y,
                    angle,
                    (end_offset - start_offset) / 2.0 + start_offset,
                    max(0.0, size_x),
                    laser.width / 2.0,
                ))

            if state == 0:
                if laser.flags & 1:
                    size_x = full_length
                else:
                    res = min(laser.start_time, 30)
                    if laser.start_time - res < timer:
                        width_now = (
                            timer_float
                            * laser.width
                            / max(1, laser.start_time)
                        )
                    else:
                        width_now = 1.2
                    # Preserve the shipped midpoint-hitbox bug.
                    size_x = width_now / 2.0
                if timer >= laser.hitbox_start_time:
                    append_geometry(size_x)
                if timer >= laser.start_time:
                    state = 1
                    timer = 0
                    timer_float = 0.0
                    # Source switch fallthrough retains the warmup size.
                    state_one_size_x = size_x
                else:
                    # The source returns from this case before its retirement
                    # check, so retain the same order here.
                    timer += 1
                    timer_float += 1.0
                    continue

            if state == 1:
                append_geometry(state_one_size_x)
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
                        width_now -= (
                            timer_float
                            * laser.width
                            / laser.despawn_duration
                        )
                    # Preserve the shipped midpoint-hitbox bug.
                    size_x = width_now / 2.0
                if timer < laser.hitbox_end_delay:
                    append_geometry(size_x)
                if timer >= laser.despawn_duration:
                    active = False

            if start_offset >= 640.0:
                active = False
            if not active:
                alive = False
                continue
            timer += 1
            timer_float += 1.0
    return tuple(tuple(frame) for frame in frames)
