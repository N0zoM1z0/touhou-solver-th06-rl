from __future__ import annotations

from dataclasses import replace

import pytest

from th06_rl.policy_api import PolicyContext
from th06_rl.policies.adaptive import AdaptivePolicy
from th06_rl.policies.wine_intervention import (
    STATE_SCHEMA,
    WineInterventionPolicy,
)


def _context(**changes) -> PolicyContext:
    value = PolicyContext(
        frame=100,
        scope=(3, 0, 0, 6),
        source_context="automatic-context-is-metadata-only",
        baseline_action="left",
        locally_admissible_actions=("left", "up", "up_left"),
        player_x=100.0,
        player_y=432.0,
        power=128,
        bullet_count=400,
        laser_count=0,
        hard_action_count=8,
        exploration_rate=0.0,
        current_action="left",
        hard_admissible_actions=("left", "up", "up_left"),
        phase_elapsed_frames=10,
        hard_action_evaluations=(
            ("left", 2.0, 92.0, 432.0),
            ("up", 1.5, 100.0, 424.0),
            ("up_left", 1.0, 94.0, 424.0),
        ),
    )
    return replace(value, **changes)


def _state(arm: str) -> dict[str, object]:
    return {
        "schema": STATE_SCHEMA,
        "pair_id": "pair-01",
        "arm": arm,
        "alternative_probability": 0.5,
        "eligibility": {
            "min_player_y": 420.0,
            "min_bullets": 256,
            "max_hard_actions": 12,
            "min_reserve_gain": 4.0,
        },
        "incumbent_state": AdaptivePolicy().export_state(),
    }


@pytest.mark.parametrize(
    ("arm", "expected"),
    (("incumbent", "left"), ("alternative", "up")),
)
def test_balanced_pair_changes_only_the_one_eligible_action(
    arm: str, expected: str,
) -> None:
    policy = WineInterventionPolicy()
    policy.import_state(_state(arm))

    first = policy.decide(_context())
    second = policy.decide(_context(frame=101))

    assert first.action == expected
    assert first.behavior_probability == 0.5
    assert f":pair-01:{arm}:left:up" in first.policy_id
    assert second.action == "left"
    assert second.policy_id == "phase-local-hierarchical-ucb-v4"
    assert policy.metrics()["interventions"] == 1


def test_ineligible_physical_frontier_stays_with_incumbent() -> None:
    policy = WineInterventionPolicy()
    policy.import_state(_state("alternative"))

    decision = policy.decide(_context(player_y=400.0))

    assert decision.action == "left"
    assert decision.behavior_probability == 1.0
    assert policy.metrics()["interventions"] == 0


def test_alternative_must_be_in_local_native_safe_set() -> None:
    policy = WineInterventionPolicy()
    policy.import_state(_state("alternative"))

    decision = policy.decide(_context(locally_admissible_actions=("left",)))

    assert decision.action == "left"
    assert policy.metrics()["interventions"] == 0
