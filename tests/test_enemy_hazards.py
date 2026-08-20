from __future__ import annotations

import pytest

from th06_rl.retail.hazards.enemies import future_boxes
from th06_rl.retail.model import EnemyBody


def _enemy(**changes) -> EnemyBody:
    values = dict(
        x=9.0,
        y=200.0,
        half_width=2.0,
        half_height=2.0,
        velocity_x=-10.0,
        velocity_y=0.0,
        angle=0.0,
        angular_velocity=0.0,
        speed=0.0,
        acceleration=0.0,
        movement_mode=0,
        movement_ease=0,
        invert_x=False,
        move_interp_x=0.0,
        move_interp_y=0.0,
        move_start_x=0.0,
        move_start_y=0.0,
        move_timer=0,
        move_timer_float=0.0,
        move_start_time=0,
    )
    values.update(changes)
    return EnemyBody(**values)


def test_enemy_body_applies_source_move_bounds_before_collision() -> None:
    enemy = _enemy(
        should_clamp_position=True,
        lower_move_x=8.0,
        lower_move_y=16.0,
        upper_move_x=376.0,
        upper_move_y=432.0,
    )

    assert future_boxes(enemy, 1)[0] == pytest.approx(
        (6.0, 198.0, 10.0, 202.0)
    )


def test_enemy_body_without_clamp_preserves_unbounded_source_motion() -> None:
    assert future_boxes(_enemy(), 1)[0] == pytest.approx(
        (-3.0, 198.0, 1.0, 202.0)
    )
