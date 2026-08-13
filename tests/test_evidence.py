from __future__ import annotations

import pytest

from th06_rl.evidence import minimum_count_gate_impossible


def test_minimum_count_gate_rejects_only_when_success_is_unreachable() -> None:
    assert not minimum_count_gate_impossible(
        observed_successes=1,
        completed_units=3,
        total_units=6,
        required_successes=4,
    )
    assert minimum_count_gate_impossible(
        observed_successes=1,
        completed_units=4,
        total_units=6,
        required_successes=4,
    )


def test_minimum_count_gate_never_turns_a_partial_pass_into_acceptance() -> None:
    assert not minimum_count_gate_impossible(
        observed_successes=4,
        completed_units=4,
        total_units=6,
        required_successes=4,
    )


@pytest.mark.parametrize("values", (
    {"observed_successes": 2, "completed_units": 1,
     "total_units": 6, "required_successes": 4},
    {"observed_successes": 0, "completed_units": 7,
     "total_units": 6, "required_successes": 4},
    {"observed_successes": 0, "completed_units": 1,
     "total_units": 6, "required_successes": 7},
))
def test_minimum_count_gate_rejects_invalid_shapes(values) -> None:
    with pytest.raises(ValueError):
        minimum_count_gate_impossible(**values)
