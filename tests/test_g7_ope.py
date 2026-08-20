from __future__ import annotations

from th06_rl.feature_contract import FEATURE_AVAILABILITY_SCHEMA
from th06_rl.g7_forecast import build_forecast_artifact
from th06_rl.g7_learner import ACTOR_FEATURE_NAMES, build_critic_dataset
from th06_rl.g7_ope import OPE_SCHEMA, evaluate_candidate
from th06_rl.g7_policy_math import POLICY_DISTRIBUTION_SCHEMA
from th06_rl.g7_support import fit_local_support
from th06_rl.g7_training import CANDIDATE_SCHEMA
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.learning_features import CAUSAL_TREE_FEATURE_SCHEMA
from th06_rl.offline_options import ActorState, OfflineOptionTransition
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)


def _named(names, overrides=None):
    overrides = overrides or {}
    return tuple((name, float(overrides.get(name, 0.0))) for name in names)


def _state() -> ActorState:
    return ActorState(
        _named(OBSERVATION_FEATURE_NAMES),
        (
            ("left", _named(ACTION_FEATURE_NAMES, {"direction_x": -1.0})),
            ("stay", _named(ACTION_FEATURE_NAMES, {"stationary": 1.0})),
        ),
        _named(HISTORY_FEATURE_NAMES),
        ("left", "stay"),
        "stay",
        "stay",
    )


def _option(index: int) -> OfflineOptionTransition:
    action = "left" if index % 2 == 0 else "stay"
    cost = int(action == "stay")
    state = _state()
    return OfflineOptionTransition(
        "th06-rl-causal-options-v2",
        f"validation-{index:03d}",
        "complete-route",
        "safe-option-exploration-v2",
        f"{index:016x}:00000001",
        0,
        0,
        1,
        "fixture",
        action,
        0.5,
        (("left", 0.5), ("stay", 0.5)),
        state,
        None,
        cost,
        cost,
        0,
        1,
        0,
        True,
        True,
        (),
    )


def _actor() -> dict[str, object]:
    weights = [0.0] * len(ACTOR_FEATURE_NAMES)
    weights[ACTOR_FEATURE_NAMES.index("action:direction_x")] = -2.0
    return {
        "schema": "th06-rl-g7-linear-awr-actor-v1",
        "feature_schema": CAUSAL_TREE_FEATURE_SCHEMA,
        "feature_availability_schema": FEATURE_AVAILABILITY_SCHEMA,
        "policy_distribution_schema": POLICY_DISTRIBUTION_SCHEMA,
        "feature_names": list(ACTOR_FEATURE_NAMES),
        "mean": [0.0] * len(ACTOR_FEATURE_NAMES),
        "scale": [1.0] * len(ACTOR_FEATURE_NAMES),
        "weights": weights,
        "reference_epsilon": 1.0,
    }


def test_paired_heldout_pdis_recovers_known_lower_hit_policy() -> None:
    episodes = tuple((_option(index),) for index in range(200))
    support = fit_local_support(
        build_critic_dataset(episodes, reference_epsilon=1.0),
        seed=3,
        prototypes_per_action=2,
        minimum_samples=16,
        minimum_ess=16.0,
    )
    actor = _actor()
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "authorization": "offline-research-only",
        "actor": actor,
        "local_support": support,
        "forecast": build_forecast_artifact([actor, actor, actor]),
    }
    report = evaluate_candidate(
        candidate,
        episodes,
        max_kl=0.5,
        maximum_step_ratio=4.0,
        maximum_cumulative_ratio=4.0,
        minimum_effective_sample_size=50.0,
        minimum_episodes=100,
        confidence=0.95,
        bootstrap_resamples=1000,
        permutation_resamples=1000,
        maximum_null_p_value=0.05,
        seed=29,
    )

    assert report["schema"] == OPE_SCHEMA
    assert report["passed"] is True
    assert report["paired_difference_interval"][1] < 0.0
    assert report["candidate_pdis_mean"] < report["incumbent_pdis_mean"]
    assert report["depth_stratified_null_p_value"] <= 0.05
    assert report["authorization"] == "offline-evidence-only"
