from __future__ import annotations

import math
from dataclasses import replace
import struct

import pytest

from th06_rl.retail.model import Bullet
from th06_rl.th06.observed_bullets import (
    classify_ex_flags,
    hazard_boxes,
)


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


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _spawn_reference(bullet, horizon):
    divisor = {2: 2.0, 3: 2.5, 4: 3.0}[bullet.state]
    dx = _f32(_f32(bullet.vx) / divisor)
    dy = _f32(_f32(bullet.vy) / divisor)
    x, y = _f32(bullet.x), _f32(bullet.y)
    possible = [[] for _ in range(horizon)]
    for transition in range(1, horizon + 1):
        x, y = _f32(x + dx), _f32(y + dy)
        fired = replace(
            bullet,
            x=x,
            y=y,
            state=1,
            timer=0,
            timer_float=0.0,
        )
        for offset, box in enumerate(
            hazard_boxes(fired, horizon - transition + 1)
        ):
            possible[transition - 1 + offset].append(box)
    return [
        (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        for boxes in possible
    ]


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


def test_pure_spawn_effect_includes_partial_and_fired_motion_same_frame() -> None:
    bullet = _bullet(
        x=100.0,
        y=200.0,
        vx=3.0,
        vy=0.0,
        half_width=0.5,
        half_height=0.5,
        state=3,
        ex_flags=0x4,
        speed=3.0,
        turn_speed=3.0,
    )

    first = hazard_boxes(bullet, 1)[0]

    # State 3 moves by velocity/2.5 before ANM completion, then the source
    # falls through and applies one full fired translation in the same pass.
    assert first == pytest.approx((103.7, 199.5, 104.7, 200.5), abs=2e-5)


def test_spawn_slowdown_batch_preserves_each_completion_branch() -> None:
    bullet = _bullet(
        x=208.43280029296875,
        y=86.51250457763672,
        vx=2.7388079166412354,
        vy=-1.5812424421310425,
        state=3,
        ex_flags=0x5,
        speed=3.1624984741210938,
        angle=-0.5235962867736816,
        timer=15,
        timer_float=15.0,
    )

    boxes = hazard_boxes(bullet, 12)

    assert boxes == _spawn_reference(bullet, 12)
    # Later envelopes contain branches which stayed in the slower spawn
    # animation longer; uncertainty must not collapse to one completion tick.
    assert boxes[-1][2] - boxes[-1][0] > 2 * bullet.half_width
    assert boxes[-1][3] - boxes[-1][1] > 2 * bullet.half_height


def test_spawn_absolute_turn_batch_preserves_each_completion_branch() -> None:
    bullet = _bullet(
        x=170.4848175048828,
        y=78.94924926757812,
        vx=-4.137522220611572,
        vy=-0.971293568611145,
        state=3,
        ex_flags=0x104,
        speed=4.25,
        turn_speed=1.399999976158142,
        angle=-2.911015272140503,
        direction_rotation=2.7488934993743896,
        direction_interval=60,
        direction_num_times=0,
        direction_max_times=1,
        timer=13,
        timer_float=13.0,
    )

    assert hazard_boxes(bullet, 12) == _spawn_reference(bullet, 12)
