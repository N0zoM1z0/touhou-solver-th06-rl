from __future__ import annotations

from th06_rl.native import LaserRect
from th06_rl.retail.hazards.lasers import hazards_by_frame
from th06_rl.retail.model import Laser
from th06_rl.th06.observed_lasers import laser_rects_by_frame


def test_scalar_laser_projection_matches_established_source_model() -> None:
    base = {
        "x": 192.0,
        "y": 96.0,
        "angle": 0.5,
        "start_offset": -4.0,
        "end_offset": 24.0,
        "start_length": 96.0,
        "width": 12.0,
        "speed": 3.0,
        "start_time": 30,
        "hitbox_start_time": 12,
        "duration": 45,
        "despawn_duration": 15,
        "hitbox_end_delay": 6,
        "timer": 0,
        "timer_float": 0.0,
        "flags": 0,
        "state": 0,
        "angular_velocity": 0.0,
        "motion_known": True,
    }

    def laser(slot: int, **changes) -> Laser:
        return Laser(slot=slot, **(base | changes))

    lasers = (
        laser(0),
        laser(1, timer=29, timer_float=29.0, angular_velocity=0.03),
        laser(2, state=1, timer=45, timer_float=45.0),
        laser(3, state=1, duration=0, despawn_duration=0),
        laser(4, state=2, timer=4, timer_float=4.0),
        laser(5, state=2, flags=1, start_offset=638.0, speed=3.0),
    )
    expected = tuple(
        tuple(LaserRect(
            hazard.origin_x,
            hazard.origin_y,
            hazard.angle,
            hazard.center_offset,
            hazard.size_x,
            hazard.size_y,
        ) for hazard in frame)
        for frame in hazards_by_frame(lasers, 16)
    )

    assert laser_rects_by_frame(lasers, 16) == expected
