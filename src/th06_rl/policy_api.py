"""Stable immutable-policy boundary above the native reactive baseline."""

from __future__ import annotations

from dataclasses import dataclass
import math


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
    hard_action_count: int
    current_action: str = "stay"
    hard_admissible_actions: tuple[str, ...] = ()
    phase_elapsed_frames: int = 0
    # Already-computed four-frame native certificates.  Policies may use this
    # bounded geometry as ranking evidence, but cannot add an action to the
    # authoritative safe set.
    hard_action_evaluations: tuple[
        tuple[str, float | None, float, float], ...
    ] = ()
    # The controller's already-computed advisory horizon. Four means the
    # longer constant-action frontier closed and only the authoritative
    # Hard-4 set remains. It is generic pressure evidence, not a phase key.
    effort_horizon: int = 0
    # Adapter-normalized, game-neutral learner interface. The learner binds
    # feature names in its generation manifest and never reads game memory or
    # source context directly.
    observation_features: tuple[tuple[str, float], ...] = ()
    action_features: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ] = ()
    hazard_primitives: tuple[tuple[float, ...], ...] = ()
    history_features: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class PolicyOptionTrace:
    """Auditable assignment metadata for a temporally extended action intent."""

    option_id: str
    intent: str
    boundary: bool
    boundary_probability: float
    elapsed_frames: int
    termination_reason: str | None = None
    preceding_termination_reason: str | None = None
    behavior_probabilities: tuple[tuple[str, float], ...] = ()
    information_weights: tuple[tuple[str, float], ...] = ()
    propensity_ess: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.option_id or not self.intent:
            raise ValueError("option identity and intent cannot be empty")
        if (
            not math.isfinite(self.boundary_probability)
            or not 0.0 < self.boundary_probability <= 1.0
        ):
            raise ValueError("option boundary probability must be in (0, 1]")
        if self.elapsed_frames <= 0:
            raise ValueError("option elapsed frames must be positive")
        if self.behavior_probabilities:
            names = tuple(name for name, _value in self.behavior_probabilities)
            probabilities = tuple(
                float(value) for _name, value in self.behavior_probabilities
            )
            if (
                len(set(names)) != len(names)
                or self.intent not in names
                or any(
                    not math.isfinite(value) or value < 0.0
                    for value in probabilities
                )
                or dict(self.behavior_probabilities)[self.intent] <= 0.0
                or not math.isclose(
                    sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9
                )
                or not math.isclose(
                    dict(self.behavior_probabilities)[self.intent],
                    self.boundary_probability,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("complete option propensity vector is invalid")
            for diagnostics in (self.information_weights, self.propensity_ess):
                if diagnostics and (
                    tuple(name for name, _value in diagnostics) != names
                    or any(
                        not math.isfinite(float(value)) or float(value) < 0.0
                        for _name, value in diagnostics
                    )
                ):
                    raise ValueError("option propensity diagnostics are invalid")


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    policy_id: str
    behavior_probability: float = 1.0
    option: PolicyOptionTrace | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.behavior_probability <= 1.0:
            raise ValueError("behavior probability must be in (0, 1]")
        if self.option is not None:
            if self.option.intent != self.action:
                raise ValueError("option intent must equal the published decision")
            expected = (
                self.option.boundary_probability
                if self.option.boundary else 1.0
            )
            if not math.isclose(
                self.behavior_probability,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("decision probability disagrees with option assignment")
