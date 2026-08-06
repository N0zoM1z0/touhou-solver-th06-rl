"""Stable hot-reload boundary above the non-learning reactive baseline."""

from __future__ import annotations

from dataclasses import dataclass


POLICY_API_VERSION = 1


@dataclass(frozen=True)
class PolicyContext:
    frame: int
    scope: tuple[int, int, int, int]
    source_context: str
    baseline_action: str
    locally_admissible_actions: tuple[str, ...]
    player_x: float
    player_y: float
    power: int
    bullet_count: int
    laser_count: int


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    policy_id: str
    behavior_probability: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.behavior_probability <= 1.0:
            raise ValueError("behavior probability must be in (0, 1]")

