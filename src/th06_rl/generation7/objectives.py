"""Proper finite-lower-bound policy objectives used by Generation 7."""

from __future__ import annotations

import math


def _log_probabilities(logits: tuple[float, ...]) -> tuple[float, ...]:
    if not logits or any(not math.isfinite(value) for value in logits):
        raise ValueError("actor logits must be nonempty and finite")
    maximum = max(logits)
    normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
    return tuple(value - normalizer for value in logits)


def awr_weight(
    advantage: float,
    *,
    temperature: float,
    maximum_weight: float,
) -> float:
    if (
        not math.isfinite(advantage)
        or not math.isfinite(temperature)
        or temperature <= 0.0
        or not math.isfinite(maximum_weight)
        or maximum_weight < 1.0
    ):
        raise ValueError("AWR weight contract is invalid")
    exponent = advantage / temperature
    if exponent >= math.log(maximum_weight):
        return maximum_weight
    return math.exp(exponent)


def weighted_negative_log_likelihood(
    logits: tuple[float, ...],
    *,
    factual_index: int,
    weight: float,
) -> float:
    """Nonnegative proper loss with infimum zero for every nonnegative weight."""
    if not 0 <= factual_index < len(logits):
        raise IndexError("factual action index is invalid")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("actor optimization weight must be finite and nonnegative")
    loss = -weight * _log_probabilities(logits)[factual_index]
    if not math.isfinite(loss) or loss < -1e-12:
        raise RuntimeError("proper actor loss violated its lower bound")
    return max(0.0, loss)


def reference_kl(
    logits: tuple[float, ...],
    reference: tuple[float, ...],
) -> float:
    if (
        len(reference) != len(logits)
        or any(not math.isfinite(value) or value <= 0.0 for value in reference)
        or not math.isclose(sum(reference), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("reference policy must have complete positive support")
    log_policy = _log_probabilities(logits)
    value = sum(
        probability * (math.log(probability) - log_probability)
        for probability, log_probability in zip(reference, log_policy, strict=True)
    )
    return max(0.0, value)


def proper_actor_loss(
    logits: tuple[float, ...],
    *,
    factual_index: int,
    weight: float,
    reference: tuple[float, ...],
    kl_coefficient: float,
) -> float:
    if not math.isfinite(kl_coefficient) or kl_coefficient < 0.0:
        raise ValueError("KL coefficient must be finite and nonnegative")
    return weighted_negative_log_likelihood(
        logits,
        factual_index=factual_index,
        weight=weight,
    ) + kl_coefficient * reference_kl(logits, reference)


def extreme_logit_smoke() -> dict[str, object]:
    reference = (0.8, 0.1, 0.1)
    factual_probabilities = (1e-2, 1e-8, 1e-30)
    losses = []
    for probability in factual_probabilities:
        other = (1.0 - probability) / 2.0
        logits = tuple(math.log(value) for value in (probability, other, other))
        losses.append(proper_actor_loss(
            logits,
            factual_index=0,
            weight=0.25,
            reference=reference,
            kl_coefficient=0.05,
        ))
    passes = (
        all(value >= 0.0 and math.isfinite(value) for value in losses)
        and losses[0] < losses[1] < losses[2]
    )
    return {
        "schema": "generation7-extreme-logit-smoke-v1",
        "factual_probabilities": list(factual_probabilities),
        "proper_losses": losses,
        "passes": passes,
    }
