from __future__ import annotations

import pytest

from th06_rl.g7_policy_math import (
    advantage_weighted_nll,
    bellman_cost_target,
    constrained_cost_distribution,
    reference_distribution,
    sample_action,
)


def test_reference_distribution_matches_logged_safe_exploration_law() -> None:
    assert dict(reference_distribution(
        ("right", "left", "stay"),
        "stay",
        epsilon=0.3,
    )) == pytest.approx({"left": 0.1, "right": 0.1, "stay": 0.8})


def test_cost_tilt_is_complete_safe_and_kl_bounded() -> None:
    result = constrained_cost_distribution(
        safe_actions=("left", "stay", "right"),
        baseline_action="stay",
        predicted_costs={"left": 0.0, "stay": 1.0, "right": 2.0},
        supported_actions=("left", "stay", "right", "unsafe-extra"),
        forecast_accepted_actions=("left", "stay", "right", "unsafe-extra"),
        epsilon=0.3,
        temperature=0.1,
        max_kl=0.05,
    )

    probabilities = dict(result.probabilities)
    assert set(probabilities) == {"left", "stay", "right"}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["left"] > dict(result.reference_probabilities)["left"]
    assert result.kl_from_reference <= 0.05 + 1e-10
    assert result.abstained is False


def test_support_and_forecast_never_add_physical_action() -> None:
    result = constrained_cost_distribution(
        safe_actions=("left", "stay"),
        baseline_action="stay",
        predicted_costs={"left": -100.0, "stay": 0.0, "unsafe": -1000.0},
        supported_actions=("left", "stay", "unsafe"),
        forecast_accepted_actions=("left", "stay", "unsafe"),
        epsilon=0.2,
        temperature=1.0,
        max_kl=0.2,
    )

    assert set(dict(result.probabilities)) == {"left", "stay"}


def test_missing_baseline_support_abstains_to_incumbent() -> None:
    result = constrained_cost_distribution(
        safe_actions=("left", "stay"),
        baseline_action="stay",
        predicted_costs={"left": 0.0},
        supported_actions=("left",),
        forecast_accepted_actions=("left", "stay"),
        epsilon=0.2,
        temperature=1.0,
        max_kl=1.0,
    )

    assert result.abstained is True
    assert dict(result.probabilities) == {"left": 0.0, "stay": 1.0}


def test_awr_loss_cannot_reward_suppressing_the_factual_action() -> None:
    centered = advantage_weighted_nll((0.0, 0.0), factual_index=0, weight=2.0)
    suppressed = advantage_weighted_nll(
        (-100.0, 100.0), factual_index=0, weight=2.0
    )

    assert centered >= 0.0
    assert suppressed > centered
    with pytest.raises(ValueError, match="nonnegative"):
        advantage_weighted_nll((0.0, 0.0), factual_index=0, weight=-1.0)


def test_hit_cost_target_bootstraps_only_before_episode_end() -> None:
    assert bellman_cost_target(1, terminal=True) == 1.0
    assert bellman_cost_target(
        1,
        terminal=False,
        next_probabilities=(("left", 0.25), ("stay", 0.75)),
        next_costs=(("left", 2.0), ("stay", 4.0)),
    ) == pytest.approx(4.5)
    with pytest.raises(ValueError, match="gamma=1"):
        bellman_cost_target(0, terminal=True, gamma=0.99)
    with pytest.raises(ValueError, match="cannot bootstrap"):
        bellman_cost_target(
            0,
            terminal=True,
            next_probabilities=(("stay", 1.0),),
            next_costs=(("stay", 0.0),),
        )


def test_sampling_uses_the_exact_declared_probability() -> None:
    rows = (("left", 0.2), ("stay", 0.8), ("right", 0.0))

    assert sample_action(rows, draw=0.1) == ("left", 0.2)
    assert sample_action(rows, draw=0.2) == ("stay", 0.8)
