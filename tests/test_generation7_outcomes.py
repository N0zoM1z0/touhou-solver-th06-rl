import pytest

from th06_rl.generation7.outcomes import (
    OptionOutcome,
    assert_hit_conservation,
    reward_from_hit_cost,
    undiscounted_hit_returns,
)


def test_hit_cost_sign_terminal_and_gamma_one_contract() -> None:
    outcomes = (
        OptionOutcome(1, 8, False),
        OptionOutcome(0, 8, False),
        OptionOutcome(2, 4, True),
    )
    assert undiscounted_hit_returns(outcomes) == (3.0, 2.0, 2.0)
    assert tuple(reward_from_hit_cost(row.hit_cost) for row in outcomes) == (
        -1.0, -0.0, -2.0
    )
    assert_hit_conservation(outcomes, pre_option_hits=1, manifest_hits=4)


def test_hit_conservation_rejects_missing_or_double_counted_hits() -> None:
    outcomes = (OptionOutcome(1, 8, True),)
    with pytest.raises(ValueError, match="conservation"):
        assert_hit_conservation(outcomes, pre_option_hits=0, manifest_hits=2)
