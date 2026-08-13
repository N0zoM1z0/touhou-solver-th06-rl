"""One exact residual stochastic policy distribution for every consumer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


@dataclass(frozen=True)
class StochasticPolicyDecision:
    actions: tuple[str, ...]
    probabilities: tuple[float, ...]
    baseline_action: str
    native_collision_safe: tuple[bool, ...]
    statistically_supported: tuple[bool, ...]
    forecast_risky: tuple[bool, ...]
    policy_id: str

    def __post_init__(self) -> None:
        count = len(self.actions)
        if (
            not self.policy_id
            or count == 0
            or len(set(self.actions)) != count
            or self.baseline_action not in self.actions
            or len(self.probabilities) != count
            or len(self.native_collision_safe) != count
            or len(self.statistically_supported) != count
            or len(self.forecast_risky) != count
            or not all(self.native_collision_safe)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.probabilities
            )
            or not math.isclose(
                sum(self.probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError("stochastic policy distribution is invalid")

    def probability(self, action: str) -> float:
        try:
            return self.probabilities[self.actions.index(action)]
        except ValueError as error:
            raise KeyError(action) from error

    def sample(self, draw: float) -> str:
        if not math.isfinite(draw) or not 0.0 <= draw < 1.0:
            raise ValueError("policy draw must be in [0, 1)")
        cumulative = 0.0
        for action, probability in zip(
            self.actions, self.probabilities, strict=True
        ):
            cumulative += probability
            if draw < cumulative:
                return action
        return self.actions[-1]


class DistributionPolicy(Protocol):
    policy_id: str

    def distribution(
        self,
        *,
        safe_actions: tuple[str, ...],
        baseline_action: str,
        logits: tuple[float, ...],
        statistically_supported: tuple[bool, ...],
        forecast_risky: tuple[bool, ...],
    ) -> StochasticPolicyDecision: ...


def reference_probabilities(
    safe_actions: tuple[str, ...],
    baseline_action: str,
    *,
    epsilon: float,
) -> tuple[float, ...]:
    if (
        not safe_actions
        or len(set(safe_actions)) != len(safe_actions)
        or baseline_action not in safe_actions
        or not math.isfinite(epsilon)
        or not 0.0 <= epsilon <= 1.0
    ):
        raise ValueError("reference policy contract is invalid")
    uniform = epsilon / len(safe_actions)
    return tuple(
        uniform + (1.0 - epsilon if action == baseline_action else 0.0)
        for action in safe_actions
    )


@dataclass(frozen=True)
class ResidualStochasticPolicy:
    epsilon: float
    temperature: float
    maximum_log_tilt: float
    policy_id: str = "generation7-residual-stochastic-policy-v1"

    def __post_init__(self) -> None:
        if (
            not 0.0 < self.epsilon <= 1.0
            or not math.isfinite(self.temperature)
            or self.temperature <= 0.0
            or not math.isfinite(self.maximum_log_tilt)
            or self.maximum_log_tilt <= 0.0
        ):
            raise ValueError("residual stochastic policy parameters are invalid")

    def distribution(
        self,
        *,
        safe_actions: tuple[str, ...],
        baseline_action: str,
        logits: tuple[float, ...],
        statistically_supported: tuple[bool, ...],
        forecast_risky: tuple[bool, ...],
    ) -> StochasticPolicyDecision:
        count = len(safe_actions)
        if (
            len(logits) != count
            or len(statistically_supported) != count
            or len(forecast_risky) != count
            or any(not math.isfinite(value) for value in logits)
        ):
            raise ValueError("residual policy inputs have inconsistent shape")
        reference = reference_probabilities(
            safe_actions,
            baseline_action,
            epsilon=self.epsilon,
        )
        baseline_index = safe_actions.index(baseline_action)
        baseline_logit = logits[baseline_index]
        tilted = []
        for index, (probability, logit) in enumerate(
            zip(reference, logits, strict=True)
        ):
            allowed = (
                index != baseline_index
                and statistically_supported[index]
                and not forecast_risky[index]
            )
            delta = (logit - baseline_logit) / self.temperature if allowed else 0.0
            delta = max(-self.maximum_log_tilt, min(self.maximum_log_tilt, delta))
            tilted.append(probability * math.exp(delta))
        normalizer = sum(tilted)
        probabilities = tuple(value / normalizer for value in tilted)
        return StochasticPolicyDecision(
            actions=safe_actions,
            probabilities=probabilities,
            baseline_action=baseline_action,
            native_collision_safe=(True,) * count,
            statistically_supported=statistically_supported,
            forecast_risky=forecast_risky,
            policy_id=self.policy_id,
        )
