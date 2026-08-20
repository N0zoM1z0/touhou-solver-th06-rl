"""Pure Generation-7 policy/objective math shared by fit and deployment."""

from __future__ import annotations

from dataclasses import dataclass
import math


POLICY_DISTRIBUTION_SCHEMA = "th06-rl-g7-policy-distribution-v1"


def _actions(actions, *, label: str) -> tuple[str, ...]:
    result = tuple(sorted(str(action) for action in actions))
    if not result or any(not action for action in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} actions are empty or duplicated")
    return result


def reference_distribution(
    safe_actions,
    baseline_action: str,
    *,
    epsilon: float,
) -> tuple[tuple[str, float], ...]:
    """The common incumbent/uniform distribution used by fit and serving."""
    safe = _actions(safe_actions, label="physical-safe")
    baseline = str(baseline_action)
    if baseline not in safe:
        raise ValueError("reference baseline is outside the physical-safe set")
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("reference epsilon must be in [0, 1]")
    uniform = epsilon / len(safe)
    return tuple(
        (
            action,
            uniform + (1.0 - epsilon if action == baseline else 0.0),
        )
        for action in safe
    )


def _kl(
    probabilities: dict[str, float],
    reference: dict[str, float],
) -> float:
    return sum(
        probability * math.log(probability / reference[action])
        for action, probability in probabilities.items()
        if probability > 0.0
    )


def _tilt(
    reference: dict[str, float],
    costs: dict[str, float],
    *,
    inverse_temperature: float,
) -> dict[str, float]:
    minimum = min(costs.values())
    logits = {
        action: math.log(reference[action])
        - inverse_temperature * (costs[action] - minimum)
        for action in reference
    }
    maximum = max(logits.values())
    weights = {action: math.exp(value - maximum) for action, value in logits.items()}
    total = sum(weights.values())
    return {action: value / total for action, value in weights.items()}


@dataclass(frozen=True)
class ConstrainedDistribution:
    schema: str
    physical_safe_actions: tuple[str, ...]
    statistically_supported_actions: tuple[str, ...]
    forecast_accepted_actions: tuple[str, ...]
    reference_probabilities: tuple[tuple[str, float], ...]
    probabilities: tuple[tuple[str, float], ...]
    kl_from_reference: float
    tilt_fraction: float
    abstained: bool
    reason: str


def constrained_cost_distribution(
    *,
    safe_actions,
    baseline_action: str,
    predicted_costs,
    supported_actions,
    forecast_accepted_actions,
    epsilon: float,
    temperature: float,
    max_kl: float,
) -> ConstrainedDistribution:
    """Tilt only native-safe supported actions, otherwise abstain to incumbent.

    Physical safety, statistical support, and forecast acceptance are retained
    as distinct sets.  The latter two may remove a learned deviation but can
    never add an action to the native-safe set.
    """
    safe = _actions(safe_actions, label="physical-safe")
    baseline = str(baseline_action)
    reference_rows = reference_distribution(safe, baseline, epsilon=epsilon)
    reference = dict(reference_rows)
    supported = tuple(sorted(set(map(str, supported_actions)) & set(safe)))
    forecast = tuple(
        sorted(set(map(str, forecast_accepted_actions)) & set(safe))
    )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("actor temperature must be positive and finite")
    if not math.isfinite(max_kl) or max_kl < 0.0:
        raise ValueError("actor KL bound must be finite and nonnegative")

    def abstain(reason: str) -> ConstrainedDistribution:
        probabilities = tuple(
            (action, float(action == baseline)) for action in safe
        )
        return ConstrainedDistribution(
            POLICY_DISTRIBUTION_SCHEMA,
            safe,
            supported,
            forecast,
            reference_rows,
            probabilities,
            math.log(1.0 / reference[baseline]),
            0.0,
            True,
            reason,
        )

    eligible = tuple(sorted(set(safe) & set(supported) & set(forecast)))
    if baseline not in eligible:
        return abstain("baseline-not-supported-and-forecast-accepted")

    try:
        costs = {action: float(dict(predicted_costs)[action]) for action in eligible}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("predicted costs do not cover every eligible action") from error
    if any(not math.isfinite(value) for value in costs.values()):
        raise ValueError("predicted costs must be finite")

    eligible_mass = sum(reference[action] for action in eligible)
    restricted_reference = {
        action: reference[action] / eligible_mass for action in eligible
    }
    base_kl = _kl(restricted_reference, reference)
    if base_kl > max_kl + 1e-12:
        return abstain("support-or-risk-mask-exceeds-kl-bound")

    full_tilt = _tilt(
        restricted_reference,
        costs,
        inverse_temperature=1.0 / temperature,
    )
    if _kl(full_tilt, reference) <= max_kl + 1e-12:
        chosen = full_tilt
        fraction = 1.0
    else:
        low, high = 0.0, 1.0
        chosen = restricted_reference
        for _ in range(64):
            middle = (low + high) / 2.0
            candidate = _tilt(
                restricted_reference,
                costs,
                inverse_temperature=middle / temperature,
            )
            if _kl(candidate, reference) <= max_kl:
                low, chosen = middle, candidate
            else:
                high = middle
        fraction = low

    complete = tuple((action, chosen.get(action, 0.0)) for action in safe)
    return ConstrainedDistribution(
        POLICY_DISTRIBUTION_SCHEMA,
        safe,
        supported,
        forecast,
        reference_rows,
        complete,
        _kl(chosen, reference),
        fraction,
        False,
        "cost-tilt",
    )


def advantage_weighted_nll(
    logits,
    *,
    factual_index: int,
    weight: float,
) -> float:
    """Proper nonnegative weighted categorical negative log likelihood."""
    values = tuple(float(value) for value in logits)
    if not values or not 0 <= factual_index < len(values):
        raise ValueError("factual actor index is invalid")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("actor logits must be finite")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("actor weight must be finite and nonnegative")
    maximum = max(values)
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )
    return weight * (log_normalizer - values[factual_index])


def sample_action(probabilities, *, draw: float) -> tuple[str, float]:
    """Sample one explicitly logged distribution using a supplied uniform draw."""
    rows = tuple((str(action), float(value)) for action, value in probabilities)
    if (
        not rows
        or len({action for action, _value in rows}) != len(rows)
        or not math.isfinite(draw)
        or not 0.0 <= draw < 1.0
        or any(not action or not math.isfinite(value) or value < 0.0 for action, value in rows)
        or not math.isclose(
            sum(value for _action, value in rows),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("sample distribution or draw is invalid")
    cumulative = 0.0
    selected = next(action for action, value in reversed(rows) if value > 0.0)
    for action, probability in rows:
        cumulative += probability
        if probability > 0.0 and draw < cumulative:
            selected = action
            break
    return selected, dict(rows)[selected]


def bellman_cost_target(
    physical_hit_cost: int,
    *,
    terminal: bool,
    next_probabilities=(),
    next_costs=(),
    gamma: float = 1.0,
) -> float:
    """HIT-only gamma-one target; only physical episode end is terminal."""
    if (
        isinstance(physical_hit_cost, bool)
        or not isinstance(physical_hit_cost, int)
        or physical_hit_cost < 0
    ):
        raise ValueError("physical HIT cost must be a nonnegative integer")
    if gamma != 1.0:
        raise ValueError("Generation-7 factual cost requires gamma=1")
    if terminal:
        if next_probabilities or next_costs:
            raise ValueError("terminal target cannot bootstrap")
        return float(physical_hit_cost)
    probabilities = dict(next_probabilities)
    costs = dict(next_costs)
    if (
        not probabilities
        or set(probabilities) != set(costs)
        or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in probabilities.values()
        )
        or not math.isclose(
            sum(map(float, probabilities.values())),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or any(not math.isfinite(float(value)) for value in costs.values())
    ):
        raise ValueError("nonterminal target lacks a complete next policy/value")
    return float(physical_hit_cost) + sum(
        float(probabilities[action]) * float(costs[action])
        for action in probabilities
    )
