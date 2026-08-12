from __future__ import annotations

import pytest

from th06_rl.policies.propensity_aware_option_exploration import (
    INCUMBENT_MASS,
    INFORMATION_MASS,
    OPTION_HORIZON_FRAMES,
    STATE_SCHEMA,
    UNIFORM_MASS,
    PropensityAwareOptionExplorationPolicy,
)
from th06_rl.policy_api import PolicyContext


def _context(frame: int, legal=("stay", "left", "right")) -> PolicyContext:
    return PolicyContext(
        frame=frame,
        scope=(3, 0, 0, 6),
        source_context="opaque",
        baseline_action="stay",
        locally_admissible_actions=legal,
        player_x=192.0,
        player_y=400.0,
        power=64,
        bullet_count=0,
        laser_count=0,
        hard_action_count=len(legal),
        exploration_rate=0.0,
    )


def _policy() -> PropensityAwareOptionExplorationPolicy:
    policy = PropensityAwareOptionExplorationPolicy()
    policy.import_state({
        "schema": STATE_SCHEMA,
        "policy_seed": 260812,
        "option_horizon_frames": OPTION_HORIZON_FRAMES,
        "mixture": {
            "incumbent": INCUMBENT_MASS,
            "uniform": UNIFORM_MASS,
            "information": INFORMATION_MASS,
        },
    })
    return policy


def test_propensity_mixture_records_complete_bounded_distribution() -> None:
    policy = _policy()
    decision = policy.decide(_context(10))
    trace = decision.option
    assert trace is not None and trace.boundary is True
    probabilities = dict(trace.behavior_probabilities)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert min(probabilities.values()) >= UNIFORM_MASS / 3
    assert probabilities["stay"] > probabilities["left"]
    assert probabilities["stay"] > probabilities["right"]
    assert decision.behavior_probability == probabilities[decision.action]
    assert tuple(name for name, _value in trace.information_weights) == tuple(
        probabilities
    )
    assert tuple(name for name, _value in trace.propensity_ess) == tuple(
        probabilities
    )


def test_rejected_tentative_boundary_rolls_back_information_ess() -> None:
    policy = _policy()
    decision = policy.decide(_context(20))
    assert sum(policy.assignment_counts.values()) == 1
    policy.reject_publication(decision)
    assert sum(policy.assignment_counts.values()) == 0
    assert all(value == pytest.approx(0.0) for value in policy.importance_sum.values())
    assert all(
        value == pytest.approx(0.0)
        for value in policy.importance_square_sum.values()
    )


def test_information_mass_shifts_toward_lower_ess_without_naming_action() -> None:
    policy = _policy()
    for _index in range(25):
        policy.importance_sum["left"] += 2.0
        policy.importance_square_sum["left"] += 4.0
    probabilities, information, ess = policy._boundary_distribution(
        ("left", "right", "stay"), "stay"
    )
    assert ess["left"] > ess["right"]
    assert information["right"] > information["left"]
    assert probabilities["right"] > probabilities["left"]
    assert min(probabilities.values()) >= UNIFORM_MASS / 3


def test_bounded_population_disagreement_multiplies_ess_information() -> None:
    policy = _policy()
    probabilities, information, _ess = policy._boundary_distribution(
        ("left", "right", "stay"),
        "stay",
        {"left": 0.1, "right": 1.0, "stay": 0.5},
    )
    assert information["right"] > information["stay"] > information["left"]
    assert probabilities["right"] > probabilities["left"]
    assert min(probabilities.values()) >= UNIFORM_MASS / 3
