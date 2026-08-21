from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from th06_rl.actions import ACTION_NAMES
from th06_rl.feature_contract import FEATURE_AVAILABILITY_SCHEMA
from th06_rl.g7_contract import (
    CANDIDATE_SCHEMA,
    ONLINE_AUTHORIZATION,
    ONLINE_QUALIFICATION_SCHEMA,
    ONLINE_STATE_SCHEMA,
)
from th06_rl.g7_forecast import build_forecast_artifact, forecast_accepted_actions
from th06_rl.g7_learner import (
    ACTOR_FEATURE_NAMES,
    LINEAR_ACTOR_SCHEMA,
    linear_actor_distribution,
)
from th06_rl.g7_policy_math import POLICY_DISTRIBUTION_SCHEMA, sample_action
from th06_rl.g7_support import SUPPORT_SCHEMA, locally_supported_actions
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.learning_features import CAUSAL_TREE_FEATURE_SCHEMA
from th06_rl.policies.g7_candidate import (
    G7CandidatePolicy,
    canonical_candidate_sha256,
)
from th06_rl.policy_api import PolicyContext
from th06_rl.policy_loader import ImmutablePolicy
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)


def _named(names, **overrides) -> tuple[tuple[str, float], ...]:
    return tuple((name, float(overrides.get(name, 0.0))) for name in names)


def _context(**changes) -> PolicyContext:
    actions = (
        ("left", _named(ACTION_FEATURE_NAMES, direction_x=-1.0)),
        ("stay", _named(ACTION_FEATURE_NAMES, stationary=1.0)),
    )
    values = {
        "frame": 10,
        "scope": (3, 0, 0, 6),
        "source_context": "source:test",
        "baseline_action": "stay",
        "locally_admissible_actions": ("left", "stay"),
        "player_x": 192.0,
        "player_y": 400.0,
        "power": 128,
        "bullet_count": 100,
        "laser_count": 0,
        "hard_action_count": 2,
        "current_action": "stay",
        "observation_features": _named(OBSERVATION_FEATURE_NAMES),
        "action_features": actions,
        "history_features": _named(HISTORY_FEATURE_NAMES),
        "learning_eligible": True,
    }
    values.update(changes)
    return PolicyContext(**values)


def _actor(left_weight: float = 2.0) -> dict[str, object]:
    weights = [0.0] * len(ACTOR_FEATURE_NAMES)
    weights[ACTOR_FEATURE_NAMES.index("action:direction_x")] = -left_weight
    return {
        "schema": LINEAR_ACTOR_SCHEMA,
        "feature_schema": CAUSAL_TREE_FEATURE_SCHEMA,
        "feature_availability_schema": FEATURE_AVAILABILITY_SCHEMA,
        "policy_distribution_schema": POLICY_DISTRIBUTION_SCHEMA,
        "feature_names": list(ACTOR_FEATURE_NAMES),
        "mean": [0.0] * len(ACTOR_FEATURE_NAMES),
        "scale": [1.0] * len(ACTOR_FEATURE_NAMES),
        "weights": weights,
        "reference_epsilon": 0.5,
    }


def _candidate() -> dict[str, object]:
    actor = _actor()
    support_actions = {}
    for action in ACTION_NAMES:
        supported = action in {"left", "stay"}
        support_actions[action] = {
            "supported": supported,
            "fit_samples": 100 if supported else 0,
            "calibration_samples": 100 if supported else 0,
            "fit_episodes": 100 if supported else 0,
            "calibration_episodes": 100 if supported else 0,
            "episode_effective_sample_size": 100.0 if supported else 0.0,
            "conformal_rank": 100 if supported else 1,
            "distance_threshold": 1_000_000.0 if supported else None,
            "prototypes": (
                [[0.0] * len(ACTOR_FEATURE_NAMES)] if supported else []
            ),
        }
    return {
        "schema": CANDIDATE_SCHEMA,
        "authorization": "offline-research-only",
        "actor": actor,
        "local_support": {
            "schema": SUPPORT_SCHEMA,
            "feature_schema": CAUSAL_TREE_FEATURE_SCHEMA,
            "feature_names": list(ACTOR_FEATURE_NAMES),
            "mean": [0.0] * len(ACTOR_FEATURE_NAMES),
            "scale": [1.0] * len(ACTOR_FEATURE_NAMES),
            "actions": support_actions,
            "calibration_unit": "physical-episode-maximum",
            "calibration_quantile": "split-conformal-ceil-(n+1)q",
            "distance_quantile": 0.99,
            "minimum_samples": 32,
            "minimum_effective_sample_size": 16.0,
        },
        "forecast": build_forecast_artifact([
            _actor(1.0),
            _actor(2.0),
            _actor(3.0),
        ]),
    }


def _state(seed: int = 7, max_kl: float = 0.5) -> dict[str, object]:
    candidate = _candidate()
    digest = canonical_candidate_sha256(candidate)
    return {
        "schema": ONLINE_STATE_SCHEMA,
        "authorization": ONLINE_AUTHORIZATION,
        "policy_seed": seed,
        "option_horizon_frames": 8,
        "target_max_kl": max_kl,
        "candidate": candidate,
        "qualification": {
            "schema": ONLINE_QUALIFICATION_SCHEMA,
            "candidate_sha256": digest,
            "target_max_kl": max_kl,
            "heldout_ope_passed": True,
            "offline_online_distribution_parity_passed": True,
            "windows_embedded_python_latency_passed": True,
            "original_wine_shadow_passed": True,
        },
    }


def test_candidate_boundary_is_exact_ope_distribution() -> None:
    seed = 7
    max_kl = 0.5
    state = _state(seed, max_kl)
    policy = G7CandidatePolicy()
    policy.import_state(state)
    context = _context()
    actor_state = policy._actor_state(
        context,
        context.locally_admissible_actions,
        context.baseline_action,
    )
    candidate = state["candidate"]
    assert isinstance(candidate, dict)
    support = locally_supported_actions(candidate["local_support"], actor_state)
    forecast = forecast_accepted_actions(
        candidate["forecast"],
        actor_state,
        supported_actions=support,
    )
    expected = linear_actor_distribution(
        candidate["actor"],
        actor_state,
        supported_actions=support,
        forecast_accepted_actions=forecast,
        max_kl=max_kl,
    )
    expected_action = sample_action(
        expected.probabilities,
        draw=random.Random(seed).random(),
    )[0]

    decision = policy.decide(context)

    assert decision.action == expected_action
    assert decision.option is not None
    assert decision.option.boundary is True
    assert decision.option.behavior_probabilities == expected.probabilities
    assert decision.behavior_probability == dict(expected.probabilities)[
        expected_action
    ]


def test_candidate_loads_through_immutable_runtime_boundary(tmp_path) -> None:
    state_path = tmp_path / "qualified-g7-state.json"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    plugin_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "policies"
        / "g7_candidate_policy.py"
    )
    policy = ImmutablePolicy(plugin_path, state_path=state_path)

    decision = policy.decide(_context())

    assert decision.option is not None
    assert decision.policy_id == "g7-qualified-candidate-v1"
    assert policy.status()["policy_id"] == "g7-qualified-candidate-v1"
    assert policy.status()["policy_failures"] == 0


def test_post_hit_extension_never_queries_candidate() -> None:
    policy = G7CandidatePolicy()
    policy.import_state(_state())
    policy.candidate["actor"] = {}

    decision = policy.decide(_context(
        locally_admissible_actions=("stay",),
        learning_eligible=False,
    ))

    assert decision.action == "stay"
    assert decision.behavior_probability == 1.0
    assert decision.option is not None
    assert decision.option.behavior_probabilities == (("stay", 1.0),)
    assert policy.metrics()["forced_ineligible_boundaries"] == 1


def test_mid_option_hit_splits_propensity_even_for_same_baseline() -> None:
    policy = G7CandidatePolicy()
    policy.import_state(_state())
    first = policy.decide(_context(
        locally_admissible_actions=("stay",),
        action_features=(
            ("stay", _named(ACTION_FEATURE_NAMES, stationary=1.0)),
        ),
    ))
    assert first.option is not None
    assert first.option.boundary is True
    policy.candidate["actor"] = {}

    after_hit = policy.decide(_context(
        frame=11,
        locally_admissible_actions=("stay",),
        action_features=(
            ("stay", _named(ACTION_FEATURE_NAMES, stationary=1.0)),
        ),
        learning_eligible=False,
    ))

    assert after_hit.option is not None
    assert after_hit.option.boundary is True
    assert after_hit.option.option_id != first.option.option_id
    assert (
        after_hit.option.preceding_termination_reason
        == "learning-eligibility-transition"
    )
    assert after_hit.option.behavior_probabilities == (("stay", 1.0),)
    assert policy.metrics()["terminations"][
        "learning-eligibility-transition"
    ] == 1


def test_candidate_refuses_unqualified_state() -> None:
    state = _state()
    state["qualification"]["original_wine_shadow_passed"] = False

    with pytest.raises(ValueError, match="qualification evidence"):
        G7CandidatePolicy().import_state(state)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("policy_seed", True),
        ("policy_seed", 1.5),
        ("option_horizon_frames", 8.5),
        ("target_max_kl", True),
    ),
)
def test_candidate_refuses_coercive_numeric_state(field, value) -> None:
    state = _state()
    state[field] = value

    with pytest.raises(ValueError, match="numeric contract"):
        G7CandidatePolicy().import_state(state)


def test_ineligible_extension_refuses_more_than_baseline() -> None:
    policy = G7CandidatePolicy()
    policy.import_state(_state())

    with pytest.raises(ValueError, match="baseline singleton"):
        policy.decide(_context(learning_eligible=False))
