"""Game-neutral, rejection-only helpers for sequential evidence schedules."""

from __future__ import annotations


def minimum_count_gate_impossible(
    *,
    observed_successes: int,
    completed_units: int,
    total_units: int,
    required_successes: int,
) -> bool:
    """Return whether a monotone minimum-count gate can no longer pass.

    This helper may stop expensive evidence collection only at an already
    committed unit boundary.  It can never grant early acceptance: later
    aggregate, identity, safety, and runtime gates still require their complete
    predeclared evidence.
    """
    values = (
        observed_successes,
        completed_units,
        total_units,
        required_successes,
    )
    if any(type(value) is not int for value in values):
        raise TypeError("evidence count gates require integers")
    if not (
        0 <= observed_successes <= completed_units <= total_units
        and 0 <= required_successes <= total_units
    ):
        raise ValueError("evidence count gate shape is invalid")
    best_possible = observed_successes + total_units - completed_units
    return best_possible < required_successes
