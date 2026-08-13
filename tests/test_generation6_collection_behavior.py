from __future__ import annotations

import math

from th06_rl.policy_api import PolicyContext
from th06_rl.policies.generation6_collection_behavior import (
    Generation6ActorEssCollectionPolicy,
)


class _Actor:
    def collection_proposal(self, _context) -> str:
        return "left"

    def metrics(self):
        return {"mode": "shadow"}


class _Draw:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _context(frame: int, legal=("stay", "left")) -> PolicyContext:
    return PolicyContext(
        frame=frame,
        scope=(6, 1, 1, 1),
        source_context="ignored",
        baseline_action="stay",
        locally_admissible_actions=legal,
        player_x=192,
        player_y=384,
        power=128,
        bullet_count=0,
        laser_count=0,
        hard_action_count=len(legal),
        exploration_rate=0,
    )


def _policy(draw: float = 0.0) -> Generation6ActorEssCollectionPolicy:
    policy = Generation6ActorEssCollectionPolicy()
    policy.loaded = True
    policy.actor = _Actor()  # type: ignore[assignment]
    policy.policy_seed = 1
    policy.random = _Draw(draw)  # type: ignore[assignment]
    return policy


def test_actor_anchored_distribution_is_complete_and_propensity_recorded() -> None:
    policy = _policy(0.99)
    decision = policy.decide(_context(100))
    assert decision.option is not None and decision.option.boundary
    probabilities = dict(decision.option.behavior_probabilities)
    assert set(probabilities) == {"stay", "left"}
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert probabilities["left"] == 0.75
    assert probabilities["stay"] == 0.25
    assert decision.behavior_probability == probabilities[decision.action]
    assert min(probabilities.values()) >= 0.25 / 2


def test_rejected_boundary_rolls_back_ess_assignment() -> None:
    policy = _policy()
    decision = policy.decide(_context(100))
    assert sum(policy.assignment_counts.values()) == 1
    policy.reject_publication(decision)
    assert sum(policy.assignment_counts.values()) == 0
    assert policy.active_id is None


def test_option_continues_for_eight_frames_and_ends() -> None:
    policy = _policy()
    first = policy.decide(_context(100))
    for frame in range(101, 108):
        current = policy.decide(_context(frame))
        assert current.action == first.action
        assert current.behavior_probability == 1.0
    assert policy.active_id is None
    assert policy.boundaries == 1
    assert policy.continuations == 7
