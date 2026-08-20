"""Source-grounded current-body motion and collision boxes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ..model import EnemyBody


class MovingEnemy(Protocol):
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    angle: float
    angular_velocity: float
    speed: float
    acceleration: float
    movement_mode: int
    movement_ease: int
    invert_x: bool
    move_interp_x: float
    move_interp_y: float
    move_start_x: float
    move_start_y: float
    move_timer: int
    move_timer_float: float
    move_start_time: int


@dataclass(frozen=True)
class MotionState:
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    angle: float
    speed: float
    movement_mode: int
    move_timer: int
    move_timer_float: float


def interpolation_progress(value: float, mode: int) -> float:
    if mode == 0:
        return 1.0 - value
    if mode == 1:
        return 1.0 - value * value
    if mode == 2:
        return 1.0 - value * value * value * value
    if mode == 3:
        value = 1.0 - value
        return value * value
    if mode == 4:
        value = 1.0 - value
        return value * value * value * value
    raise ValueError(f"unsupported enemy movement ease {mode}")


def advance_position(enemy: MovingEnemy) -> MotionState:
    """Apply Enemy::Move before this frame's ECL instructions."""
    x = enemy.x + (-enemy.velocity_x if enemy.invert_x else enemy.velocity_x)
    y = enemy.y + enemy.velocity_y
    return MotionState(
        x,
        y,
        enemy.velocity_x,
        enemy.velocity_y,
        enemy.angle,
        enemy.speed,
        enemy.movement_mode,
        enemy.move_timer,
        enemy.move_timer_float,
    )


def finish_motion(enemy: MovingEnemy) -> MotionState:
    """Apply RunEcl's post-instruction movement-mode update."""
    return finish_motion_values(
        enemy.x,
        enemy.y,
        enemy.velocity_x,
        enemy.velocity_y,
        enemy.angle,
        enemy.speed,
        enemy.angular_velocity,
        enemy.acceleration,
        enemy.movement_mode,
        enemy.movement_ease,
        enemy.move_interp_x,
        enemy.move_interp_y,
        enemy.move_start_x,
        enemy.move_start_y,
        enemy.move_timer,
        enemy.move_timer_float,
        enemy.move_start_time,
    )


def finish_motion_values(
    x: float,
    y: float,
    velocity_x: float,
    velocity_y: float,
    angle: float,
    speed: float,
    angular_velocity: float,
    acceleration: float,
    movement_mode: int,
    movement_ease: int,
    move_interp_x: float,
    move_interp_y: float,
    move_start_x: float,
    move_start_y: float,
    move_timer: int,
    move_timer_float: float,
    move_start_time: int,
) -> MotionState:
    """Source motion update over already-decoded scalar state."""

    if movement_mode == 1:
        angle += angular_velocity
        speed += acceleration
        velocity_x = math.cos(angle) * speed
        velocity_y = math.sin(angle) * speed
    elif movement_mode == 2:
        move_timer -= 1
        move_timer_float -= 1.0
        if move_start_time <= 0:
            raise ValueError("invalid active enemy interpolation time")
        interpolation = min(1.0, move_timer_float / move_start_time)
        interpolation = interpolation_progress(interpolation, movement_ease)
        velocity_x = interpolation * move_interp_x + move_start_x - x
        velocity_y = interpolation * move_interp_y + move_start_y - y
        if move_timer <= 0:
            movement_mode = 0
            x = move_start_x + move_interp_x
            y = move_start_y + move_interp_y
            velocity_x = 0.0
            velocity_y = 0.0
    return MotionState(
        x,
        y,
        velocity_x,
        velocity_y,
        angle,
        speed,
        movement_mode,
        move_timer,
        move_timer_float,
    )


def advance_motion(enemy: MovingEnemy) -> MotionState:
    """Advance a frame with no intervening ECL movement instruction."""
    positioned = advance_position(enemy)
    return finish_motion(_MotionView(enemy, positioned))


def future_positions(
    enemy: MovingEnemy,
    horizon: int,
) -> list[tuple[float, float]]:
    """Follow Enemy::Move and the source's no-instruction motion update."""
    x = enemy.x
    y = enemy.y
    velocity_x = enemy.velocity_x
    velocity_y = enemy.velocity_y
    angle = enemy.angle
    speed = enemy.speed
    movement_mode = enemy.movement_mode
    move_timer = enemy.move_timer
    move_timer_float = enemy.move_timer_float
    result: list[tuple[float, float]] = []
    for _frame in range(horizon):
        x += -velocity_x if enemy.invert_x else velocity_x
        y += velocity_y
        advanced = finish_motion_values(
            x,
            y,
            velocity_x,
            velocity_y,
            angle,
            speed,
            enemy.angular_velocity,
            enemy.acceleration,
            movement_mode,
            enemy.movement_ease,
            enemy.move_interp_x,
            enemy.move_interp_y,
            enemy.move_start_x,
            enemy.move_start_y,
            move_timer,
            move_timer_float,
            enemy.move_start_time,
        )
        x = advanced.x
        y = advanced.y
        velocity_x = advanced.velocity_x
        velocity_y = advanced.velocity_y
        angle = advanced.angle
        speed = advanced.speed
        movement_mode = advanced.movement_mode
        move_timer = advanced.move_timer
        move_timer_float = advanced.move_timer_float
        result.append((x, y))
    return result


class _MotionView:
    """Overlay mutable motion fields while retaining immutable parameters."""

    def __init__(self, source: MovingEnemy, state: MotionState) -> None:
        self._source = source
        self.__dict__.update(state.__dict__)

    def __getattr__(self, name: str):
        return getattr(self._source, name)


def future_boxes(
    enemy: EnemyBody,
    horizon: int,
) -> list[tuple[float, float, float, float]]:
    return [
        (
            x - enemy.half_width,
            y - enemy.half_height,
            x + enemy.half_width,
            y + enemy.half_height,
        )
        for x, y in future_positions(enemy, horizon)
    ]


def hazards_by_frame(
    enemies: tuple[EnemyBody, ...],
    horizon: int,
) -> list[tuple[tuple[float, float, float, float], ...]]:
    frames: list[list[tuple[float, float, float, float]]] = [
        [] for _ in range(horizon)
    ]
    for enemy in enemies:
        for index, hazard in enumerate(future_boxes(enemy, horizon)):
            frames[index].append(hazard)
    return [tuple(frame) for frame in frames]
