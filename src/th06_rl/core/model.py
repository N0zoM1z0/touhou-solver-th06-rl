"""Small movement and geometry value objects."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


@dataclass(frozen=True, order=True)
class Action:
    """One movement command. Bomb is deliberately not representable."""

    name: str
    dx: int
    dy: int
    focused: bool

    def __post_init__(self) -> None:
        if self.dx not in (-1, 0, 1) or self.dy not in (-1, 0, 1):
            raise ValueError("movement components must be -1, 0, or 1")


def movement_actions() -> tuple[Action, ...]:
    directions = (
        ("stay", 0, 0),
        ("up", 0, -1),
        ("down", 0, 1),
        ("left", -1, 0),
        ("right", 1, 0),
        ("up_left", -1, -1),
        ("up_right", 1, -1),
        ("down_left", -1, 1),
        ("down_right", 1, 1),
    )
    return tuple(
        Action(name if focused else f"{name}_fast", dx, dy, focused)
        for focused in (True, False)
        for name, dx, dy in directions
    )


@dataclass(frozen=True)
class CertifiedAction:
    """One action admitted by the observed-hazard shield."""

    action: Action
    min_clearance: float


@dataclass(frozen=True)
class Bounds:
    left: float
    right: float
    top: float
    bottom: float

    def __post_init__(self) -> None:
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("invalid movement bounds")

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, self.left), self.right),
            min(max(y, self.top), self.bottom),
        )

    def clearance(self, x: float, y: float) -> float:
        return min(
            x - self.left,
            self.right - x,
            y - self.top,
            self.bottom - y,
        )

    def control_reserve_deficit(
        self,
        x: float,
        y: float,
        reserve: float,
    ) -> float:
        """Penalize lost axis authority near edges and twice in corners."""
        return sum((
            max(reserve - (x - self.left), 0.0),
            max(reserve - (self.right - x), 0.0),
            max(reserve - (y - self.top), 0.0),
            max(reserve - (self.bottom - y), 0.0),
        ))


@dataclass(frozen=True)
class Kinematics:
    normal_speed: float
    focus_speed: float
    normal_diagonal_speed: float
    focus_diagonal_speed: float

    def __post_init__(self) -> None:
        if min(
            self.normal_speed,
            self.focus_speed,
            self.normal_diagonal_speed,
            self.focus_diagonal_speed,
        ) <= 0.0:
            raise ValueError("movement speeds must be positive")

    def advance(
        self,
        x: float,
        y: float,
        action: Action,
        bounds: Bounds,
    ) -> tuple[float, float]:
        diagonal = action.dx != 0 and action.dy != 0
        if action.focused:
            speed = (
                self.focus_diagonal_speed
                if diagonal
                else self.focus_speed
            )
        else:
            speed = (
                self.normal_diagonal_speed
                if diagonal
                else self.normal_speed
            )
        return bounds.clamp(
            x + action.dx * speed,
            y + action.dy * speed,
        )


@dataclass(frozen=True)
class HazardSample:
    """Source-model result at one projected player position and frame."""

    known: bool
    collisions: int
    clearance: float

    def __post_init__(self) -> None:
        if self.collisions < 0:
            raise ValueError("collision count cannot be negative")
        if self.known and math.isnan(self.clearance):
            raise ValueError("known clearance cannot be NaN")


class HazardOracle(Protocol):
    """Read-only projected hazards. Frame indices start at one."""

    def sample(self, x: float, y: float, frame: int) -> HazardSample:
        ...
