from __future__ import annotations

from types import SimpleNamespace

import pytest

from th06_rl.policies.safe_option_exploration import (
    OPTION_HORIZON_FRAMES,
    STATE_SCHEMA,
    SafeOptionExplorationPolicy,
)
from th06_rl.policy_api import PolicyDecision, PolicyOptionTrace


def _policy(*, seed: int = 7, exploration: float = 1.0):
    policy = SafeOptionExplorationPolicy()
    policy.import_state({
        "schema": STATE_SCHEMA,
        "policy_seed": seed,
        "exploration_probability": exploration,
        "option_horizon_frames": OPTION_HORIZON_FRAMES,
    })
    return policy


def _context(
    frame: int,
    *,
    legal: tuple[str, ...] = ("left", "right"),
    baseline: str = "left",
    scope: tuple[int, int, int, int] = (3, 0, 0, 6),
    learning_eligible: bool = True,
):
    return SimpleNamespace(
        frame=frame,
        scope=scope,
        locally_admissible_actions=legal,
        baseline_action=baseline,
        learning_eligible=learning_eligible,
    )


def test_option_assignment_persists_but_is_recensored_each_frame() -> None:
    policy = _policy()
    first = policy.decide(_context(100))
    second = policy.decide(_context(101, baseline="right"))

    assert second.action == first.action
    assert first.option is not None and first.option.boundary is True
    assert second.option is not None and second.option.boundary is False
    assert second.option.option_id == first.option.option_id
    assert second.option.elapsed_frames == 2
    assert second.behavior_probability == 1.0
    assert second.option.behavior_probabilities == first.option.behavior_probabilities
    probabilities = dict(first.option.behavior_probabilities)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities[first.action] == pytest.approx(
        first.behavior_probability
    )

    remaining = "right" if first.action == "left" else "left"
    replacement = policy.decide(_context(
        102,
        legal=(remaining,),
        baseline=remaining,
    ))
    assert replacement.action == remaining
    assert replacement.option is not None and replacement.option.boundary is True
    assert replacement.option.option_id != first.option.option_id
    assert replacement.option.preceding_termination_reason == "source-unsafe-intent"


def test_option_has_a_fixed_eight_physical_frame_horizon() -> None:
    policy = _policy(exploration=0.0)
    decisions = [policy.decide(_context(frame)) for frame in range(1, 10)]

    first_id = decisions[0].option.option_id
    assert dict(decisions[0].option.behavior_probabilities) == {
        "left": 1.0,
        "right": 0.0,
    }
    assert all(row.option.option_id == first_id for row in decisions[:8])
    assert decisions[7].option.termination_reason == "horizon"
    assert decisions[8].option.boundary is True
    assert decisions[8].option.option_id != first_id
    assert policy.metrics()["terminations"] == {"horizon": 1}


def test_observation_gap_starts_a_new_assignment() -> None:
    policy = _policy(exploration=0.0)
    first = policy.decide(_context(10))
    after_gap = policy.decide(_context(12))

    assert after_gap.option.option_id != first.option.option_id
    assert after_gap.option.preceding_termination_reason == "observation-gap"


def test_learning_eligibility_transition_splits_same_baseline_option() -> None:
    policy = _policy(exploration=0.0)
    first = policy.decide(_context(10, legal=("left",)))
    after_hit = policy.decide(_context(
        11,
        legal=("left",),
        learning_eligible=False,
    ))

    assert after_hit.option.option_id != first.option.option_id
    assert after_hit.option.boundary is True
    assert (
        after_hit.option.preceding_termination_reason
        == "learning-eligibility-transition"
    )
    assert after_hit.option.behavior_probabilities == (("left", 1.0),)


def test_input_lease_cannot_invent_a_boundary_after_option_horizon() -> None:
    policy = _policy(exploration=0.0)
    for frame in range(1, 9):
        policy.decide(_context(frame))

    forced = policy.continue_certified(_context(
        9,
        legal=("left",),
        baseline="left",
    ))
    assert forced.action == "left"
    assert forced.option is None
    assert policy.metrics()["option_boundaries"] == 1


def test_input_lease_ends_option_when_learning_eligibility_changes() -> None:
    policy = _policy(exploration=0.0)
    first = policy.decide(_context(10, legal=("left",)))

    forced = policy.continue_certified(_context(
        11,
        legal=("left",),
        learning_eligible=False,
    ))
    next_boundary = policy.decide(_context(
        12,
        legal=("left",),
        learning_eligible=False,
    ))

    assert first.option is not None
    assert forced.action == "left"
    assert forced.option is None
    assert next_boundary.option is not None
    assert next_boundary.option.boundary is True
    assert next_boundary.option.option_id != first.option.option_id
    assert policy.metrics()["terminations"][
        "learning-eligibility-transition"
    ] == 1


def test_rejected_publication_ends_tentative_option() -> None:
    policy = _policy(exploration=0.0)
    first = policy.decide(_context(10))

    policy.reject_publication(first)
    second = policy.decide(_context(11))

    assert first.option is not None
    assert second.option is not None
    assert second.option.boundary is True
    assert second.option.option_id != first.option.option_id
    assert policy.metrics()["terminations"]["publication-rejected"] == 1


def test_option_trace_binds_conditional_propensity() -> None:
    trace = PolicyOptionTrace("option-1", "left", False, 0.05, 2)
    with pytest.raises(ValueError, match="probability disagrees"):
        PolicyDecision("left", "test", 0.05, trace)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("policy_seed", True),
        ("policy_seed", 1.5),
        ("exploration_probability", True),
        ("option_horizon_frames", 8.5),
    ),
)
def test_state_refuses_coercive_numeric_values(field, value) -> None:
    state = {
        "schema": STATE_SCHEMA,
        "policy_seed": 7,
        "exploration_probability": 0.5,
        "option_horizon_frames": OPTION_HORIZON_FRAMES,
    }
    state[field] = value

    with pytest.raises(ValueError, match="numeric state"):
        SafeOptionExplorationPolicy().import_state(state)
