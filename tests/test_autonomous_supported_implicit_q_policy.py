from __future__ import annotations

from copy import deepcopy

from th06_rl.implicit_learning import STATE_SCHEMA
from th06_rl.policies.autonomous_supported_implicit_q import (
    AutonomousSupportedImplicitQPolicy,
)
from tests.test_autonomous_sequential_r_critic_policy import (
    _context,
    _model,
    _state,
)


def _implicit_state(mode: str) -> dict[str, object]:
    state = deepcopy(_state(mode))
    state["schema"] = STATE_SCHEMA
    state["selection"] = {
        "rule": "population-range-upper-bound-relative-to-incumbent",
        "baseline_advantage": 0.0,
        "uncertainty_range_multiplier": 1.0,
    }
    state["population"]["kind"] = (
        "whole-episode-bootstrap-action-centered-implicit-q"
    )
    return state


def test_active_implicit_population_publishes_supported_pessimistic_action() -> None:
    policy = AutonomousSupportedImplicitQPolicy()
    policy.import_state(_implicit_state("active"))

    assert policy.decide(_context()).action == "left"
    assert policy.metrics()["active_overrides"] == 1


def test_implicit_population_range_forces_abstention_despite_negative_members() -> None:
    state = _implicit_state("active")
    state["models"][-1] = _model(-0.1)
    policy = AutonomousSupportedImplicitQPolicy()
    policy.import_state(state)

    assert policy.decide(_context()).action == "stay"
    assert policy.metrics()["population_abstentions"] == 1


def test_shadow_implicit_population_records_but_does_not_publish() -> None:
    policy = AutonomousSupportedImplicitQPolicy()
    policy.import_state(_implicit_state("shadow"))

    assert policy.decide(_context()).action == "stay"
    assert policy.metrics()["shadow_proposals"] == 1
