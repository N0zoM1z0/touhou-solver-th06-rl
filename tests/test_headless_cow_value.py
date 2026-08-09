from __future__ import annotations

from scripts.train_headless_cow_value import delivery_contract, ordinal_outcome_labels


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


def test_cow_value_delivery_contract_is_explicit_and_backward_auditable() -> None:
    assert delivery_contract({
        "runtime_delivery_contract": "synchronous-step-v1",
        "runtime_delivery_delays": [0],
    }) == ("synchronous-step-v1", (0,))
    assert delivery_contract({}) == ("legacy-unspecified-v0", ())
