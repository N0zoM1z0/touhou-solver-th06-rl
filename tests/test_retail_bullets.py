from __future__ import annotations

import struct

import pytest

from th06_rl.retail.hazards.bullets import (
    hazard_boxes,
    radial_hazard_box,
)
from th06_rl.retail.model import Bullet


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def test_supported_acceleration_uses_one_f32_source_stepper() -> None:
    bullet = Bullet(
        10.0, 20.0, 0.1, -0.2, 1.0, 1.0, 1,
        ex_flags=0x10,
        acceleration_x=0.03,
        acceleration_y=-0.04,
        acceleration_duration=8,
    )
    expected = []
    x, y, vx, vy = map(_f32, (10.0, 20.0, 0.1, -0.2))
    for _ in range(4):
        vx = _f32(vx + 0.03)
        vy = _f32(vy - 0.04)
        x = _f32(x + vx)
        y = _f32(y + vy)
        expected.append((x, y))

    assert [_center(box) for box in hazard_boxes(bullet, 4)] == expected


def test_unknown_dynamic_bound_includes_vector_and_curve_acceleration() -> None:
    bullet = Bullet(
        100.0, 100.0, 1.0, 0.0, 1.0, 1.0, 1,
        ex_flags=0x100,
        speed=1.0,
        acceleration=0.0,
        acceleration_x=3.0,
        acceleration_y=4.0,
        curve_speed_acceleration=6.0,
    )

    second = hazard_boxes(bullet, 2)[1]
    # (base 1 + unknown-mode reserve 5) * 2 + max acceleration 6 * 3.
    assert second == pytest.approx((69.0, 69.0, 131.0, 131.0))


def test_spawn_completion_box_includes_partial_plus_full_update() -> None:
    bullet = Bullet(0.0, 0.0, 10.0, 0.0, 1.0, 1.0, 2)

    first, second = hazard_boxes(bullet, 2)

    assert first == pytest.approx((14.0, -1.0, 16.0, 1.0))
    assert second == pytest.approx((19.0, -1.0, 26.0, 1.0))


def test_radial_spawn_bound_includes_the_completion_update() -> None:
    bullet = Bullet(
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 3,
        speed=10.0,
    )

    assert radial_hazard_box(bullet, 1) == pytest.approx(
        (-21.0, -21.0, 21.0, 21.0)
    )
