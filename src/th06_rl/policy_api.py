"""Small game-neutral boundary for immutable online action ranking."""

from __future__ import annotations

from dataclasses import dataclass
import math


POLICY_API_VERSION = 5


@dataclass(frozen=True)
class PolicyContext:
    """Portable physical inputs; source/phase/episode identity is excluded."""

    baseline_action: str
    locally_admissible_actions: tuple[str, ...]
    player_x: float
    player_y: float
    power: int
    bullet_count: int
    laser_count: int
    shield_action_count: int
    current_action: str = "stay"
    shield_admissible_actions: tuple[str, ...] = ()
    shield_action_evaluations: tuple[
        tuple[str, float | None, float, float], ...
    ] = ()


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    policy_id: str
    behavior_probability: float = 1.0
    behavior_probabilities: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.behavior_probability <= 1.0:
            raise ValueError("behavior probability must be in (0, 1]")
        if not self.behavior_probabilities:
            return
        names = tuple(name for name, _value in self.behavior_probabilities)
        values = tuple(float(value) for _name, value in self.behavior_probabilities)
        probabilities = dict(self.behavior_probabilities)
        if (
            len(names) != len(probabilities)
            or self.action not in probabilities
            or any(not math.isfinite(value) or value < 0.0 for value in values)
            or not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(
                probabilities[self.action],
                self.behavior_probability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("complete behavior distribution is invalid")
