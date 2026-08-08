from __future__ import annotations

from scripts.train_headless_cow_value import ordinal_outcome_labels


def outcome(terminal: str, survival: int, legal: int, reserve: float):
    return {
        "termination_reason": terminal,
        "survival_ticks": survival,
        "minimum_native_legal_actions": legal,
        "terminal_boundary_reserve": reserve,
    }


def test_ordinal_value_labels_preserve_physical_priority() -> None:
    labels = ordinal_outcome_labels([
        outcome("authority-failure", 179, 18, 100.0),
        outcome("tick-limit", 180, 2, 5.0),
        outcome("tick-limit", 180, 12, 40.0),
    ])

    assert labels[0] < labels[1] < labels[2]


def test_ordinal_value_labels_retain_exact_ties() -> None:
    value = outcome("tick-limit", 180, 12, 40.0)

    assert ordinal_outcome_labels([value, value]) == (0, 0)
