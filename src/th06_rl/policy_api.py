"""Small game-neutral boundary for immutable online action ranking."""

from __future__ import annotations

from dataclasses import dataclass
import math


POLICY_API_VERSION = 6
ACTION_EXPOSURE_SCHEMA = "th06-rl-action-exposure-v1"


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
class ActionExposure:
    """Run-local randomized intention metadata; never an actor observation."""

    schema: str
    group_id: int
    step: int
    horizon: int
    intended_action: str
    assignment_probability: float
    assignment_probabilities: tuple[tuple[str, float], ...]
    override_reason: str | None = None

    def __post_init__(self) -> None:
        names = tuple(name for name, _value in self.assignment_probabilities)
        values = tuple(float(value) for _name, value in self.assignment_probabilities)
        probabilities = dict(self.assignment_probabilities)
        if (
            self.schema != ACTION_EXPOSURE_SCHEMA
            or self.group_id < 0
            or self.horizon <= 0
            or not 0 <= self.step < self.horizon
            or not self.intended_action
            or len(names) != len(probabilities)
            or self.intended_action not in probabilities
            or any(not name for name in names)
            or any(not math.isfinite(value) or value < 0.0 for value in values)
            or not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isfinite(self.assignment_probability)
            or not 0.0 < self.assignment_probability <= 1.0
            or not math.isclose(
                probabilities[self.intended_action],
                self.assignment_probability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or (self.override_reason is not None and not self.override_reason)
        ):
            raise ValueError("action exposure metadata is invalid")


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    policy_id: str
    behavior_probability: float = 1.0
    behavior_probabilities: tuple[tuple[str, float], ...] = ()
    action_exposure: ActionExposure | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.behavior_probability <= 1.0:
            raise ValueError("behavior probability must be in (0, 1]")
        if not self.behavior_probabilities:
            if self.action_exposure is not None:
                raise ValueError("action exposure requires a behavior distribution")
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
        exposure = self.action_exposure
        if exposure is not None and exposure.step == 0:
            assigned = dict(exposure.assignment_probabilities)
            if (
                self.action != exposure.intended_action
                or set(assigned) != set(probabilities)
                or any(
                    not math.isclose(
                        assigned[name], probabilities[name], rel_tol=1e-12, abs_tol=1e-12
                    )
                    for name in assigned
                )
            ):
                raise ValueError("exposure assignment root differs from behavior draw")
