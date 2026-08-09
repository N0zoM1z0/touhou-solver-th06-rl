from __future__ import annotations

from scripts.train_headless_cow_value import (
    behavior_value_groups,
    delivery_contract,
    ordinal_outcome_labels,
)
from scripts.train_headless_teacher import Decision


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


def test_ordinal_value_labels_do_not_invent_value_inside_equal_dead_end() -> None:
    labels = ordinal_outcome_labels([
        outcome("authority-failure", 1, 18, 4.0),
        outcome("authority-failure", 1, 2, 100.0),
    ])

    assert labels == (0, 0)


def test_ordinal_value_labels_retain_failed_survival_signal() -> None:
    labels = ordinal_outcome_labels([
        outcome("authority-failure", 61, 18, 100.0),
        outcome("authority-failure", 240, 1, 0.0),
    ])

    assert labels[0] < labels[1]


def test_ordinal_value_labels_keep_equivalent_completed_routes_tied() -> None:
    labels = ordinal_outcome_labels([
        outcome("tick-limit", 240, 16, 25.0),
        outcome("tick-limit", 240, 18, 31.0),
    ])

    assert labels == (0, 0)


def test_cow_value_delivery_contract_is_explicit_and_backward_auditable() -> None:
    assert delivery_contract({
        "runtime_delivery_contract": "synchronous-step-v1",
        "runtime_delivery_delays": [0],
    }) == ("synchronous-step-v1", (0,))
    assert delivery_contract({}) == ("legacy-unspecified-v0", ())


def test_behavior_regularization_yields_to_cow_observation() -> None:
    decision = Decision(
        run="run",
        seed=7,
        sequence=4,
        source_context="boss:0/1",
        state={},
        legal_actions=("left", "right"),
        candidates=(),
        teacher_action="left",
        selected_action="right",
        observation_sha256="digest",
    )

    assert behavior_value_groups([decision], stride=2)[0].labels == (0, 1)
    assert behavior_value_groups(
        [decision],
        excluded_observations=frozenset({"digest"}),
    ) == []
