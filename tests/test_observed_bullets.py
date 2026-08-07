from __future__ import annotations

import math

import pytest

from th06_rl.th06.donor import enable_donor_imports
from th06_rl.th06.observed_bullets import (
    classify_ex_flags,
    hazard_boxes,
)


enable_donor_imports()
from th06.model import Bullet  # noqa: E402


def _bullet(**changes) -> Bullet:
    values = dict(
        x=100.0,
        y=100.0,
        vx=1.0,
        vy=0.0,
        half_width=2.0,
        half_height=2.0,
        state=1,
        ex_flags=0,
        speed=1.0,
        turn_speed=1.0,
        angle=0.0,
        timer=0,
        timer_float=0.0,
        direction_interval=1,
        direction_num_times=0,
        direction_max_times=1,
        sprite_half_width=2.0,
        sprite_half_height=2.0,
    )
    values.update(changes)
    return Bullet(**values)


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def test_in_bounds_0xa00_needle_stays_small_and_linear() -> None:
    bullet = _bullet(
        x=274.222412109375,
        y=403.41217041015625,
        vx=0.362216,
        vy=1.248519,
        ex_flags=0xA00,
        speed=1.3,
        turn_speed=1.3,
        angle=1.28843,
    )

    boxes = hazard_boxes(bullet, 4)

    assert len(boxes) == 4
    assert boxes[-1][2] - boxes[-1][0] == pytest.approx(4.0)
    assert boxes[-1][3] - boxes[-1][1] == pytest.approx(4.0)
    assert _center(boxes[-1]) == pytest.approx((
        bullet.x + 4 * bullet.vx,
        bullet.y + 4 * bullet.vy,
    ), abs=2e-4)


def test_0x800_does_not_reflect_at_bottom_edge() -> None:
    bullet = _bullet(
        y=451.0,
        vx=0.0,
        vy=2.0,
        ex_flags=0x800,
        speed=2.0,
        turn_speed=2.0,
        angle=math.pi / 2.0,
    )

    assert _center(hazard_boxes(bullet, 1)[0]) == pytest.approx(
        (100.0, 453.0), abs=2e-5
    )


def test_0x400_reflects_at_bottom_edge() -> None:
    bullet = _bullet(
        y=451.0,
        vx=0.0,
        vy=2.0,
        ex_flags=0x400,
        speed=2.0,
        turn_speed=2.0,
        angle=math.pi / 2.0,
    )

    assert _center(hazard_boxes(bullet, 1)[0]) == pytest.approx(
        (100.0, 449.0), abs=2e-5
    )


def test_0x800_reflects_at_top_edge() -> None:
    bullet = _bullet(
        y=-3.0,
        vx=0.0,
        vy=-2.0,
        ex_flags=0x800,
        speed=2.0,
        turn_speed=2.0,
        angle=-math.pi / 2.0,
    )

    assert _center(hazard_boxes(bullet, 1)[0]) == pytest.approx(
        (100.0, -1.0), abs=2e-5
    )


def test_absolute_turn_0x100_is_source_exact() -> None:
    bullet = _bullet(
        ex_flags=0x100,
        turn_speed=2.0,
        direction_rotation=math.pi / 2.0,
        timer=1,
        timer_float=1.0,
    )

    assert _center(hazard_boxes(bullet, 1)[0]) == pytest.approx(
        (100.0, 102.0), abs=2e-5
    )
    assert classify_ex_flags(0x100) == "source-exact-local"
    assert classify_ex_flags(0xA00) == "source-exact-local"
    assert classify_ex_flags(0x80) == "conservative-player-retarget"
    assert classify_ex_flags(0x1000) == "unknown-flag-fail-closed"


def test_spawn_completion_uncertainty_uses_source_motion_not_radial_growth() -> None:
    bullet = _bullet(
        x=100.0,
        y=100.0,
        vx=3.0,
        vy=0.0,
        state=4,
        ex_flags=0x18,
        acceleration_duration=999,
        speed=3.0,
        turn_speed=3.0,
    )

    first, second = hazard_boxes(bullet, 2)

    # Completion on frame one: 1 px spawn motion, then 3 px fired motion.
    assert _center(first) == pytest.approx((104.0, 100.0), abs=2e-5)
    # Frame two encloses completion on either update: centers 105 and 107.
    assert second == pytest.approx((103.0, 98.0, 109.0, 102.0), abs=2e-5)
