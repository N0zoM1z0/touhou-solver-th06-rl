from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("xgboost")

from th06_rl.g7_learner import (
    build_critic_dataset,
    cross_fit_cost_critic,
    fit_linear_awr_actor,
    linear_actor_distribution,
    linear_actor_scores,
    whole_episode_folds,
)
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.offline_options import (
    NMNB_FORCED_EXCLUSION,
    ActorState,
    OfflineOptionTransition,
)
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


def _option(
    episode: str,
    index: int,
    action: str,
    cost: int,
    *,
    terminal: bool,
    eligible: bool = True,
) -> OfflineOptionTransition:
    state = _state()
    return OfflineOptionTransition(
        schema="th06-rl-causal-options-v3",
        episode_id=episode,
        episode_unit="route",
        behavior_policy_id="safe-option-exploration-v2",
        option_id=f"{episode}:option-{index}",
        start_sequence=index,
        end_sequence=index,
        start_stage=1,
        diagnostic_scope="diagnostic-only",
        action=action,
        behavior_probability=0.5,
        behavior_probabilities=(("left", 0.5), ("stay", 0.5)),
        state=state,
        next_state=None if terminal else state,
        physical_hit_cost=cost,
        controlled_hit_cost=cost,
        interstitial_hit_cost=0,
        elapsed_frames=1,
        interstitial_elapsed_frames=0,
        terminal=terminal,
        eligible=eligible,
        exclusion_reasons=() if eligible else ("fixture-exclusion",),
    )


def test_cost_to_go_keeps_hits_from_excluded_future_intervals() -> None:
    normal_state = _state()
    forced_state = replace(
        normal_state,
        action_features=(("stay", dict(normal_state.action_features)["stay"]),),
        legal_actions=("stay",),
    )
    eligible = replace(
        _option("episode-a", 0, "left", 0, terminal=False),
        next_state=forced_state,
    )
    forced = replace(
        _option("episode-a", 1, "stay", 1, terminal=True, eligible=False),
        state=forced_state,
        behavior_probability=1.0,
        behavior_probabilities=(("stay", 1.0),),
        exclusion_reasons=(NMNB_FORCED_EXCLUSION,),
    )
    episode = (eligible, forced)

    dataset = build_critic_dataset((episode,), reference_epsilon=1.0)

    assert len(dataset.examples) == 1
    assert dataset.examples[0].target_cost_to_go == 1.0
    assert dataset.excluded_options == 1
    assert dataset.exclusion_reasons == ((NMNB_FORCED_EXCLUSION, 1),)


def test_critic_rejects_non_hit_ineligible_continuation() -> None:
    option = _option(
        "episode-a",
        0,
        "stay",
        1,
        terminal=True,
        eligible=False,
    )

    with pytest.raises(ValueError, match="unsupported ineligible"):
        build_critic_dataset(((option,),), reference_epsilon=1.0)


def test_critic_rejects_reference_behavior_mismatch() -> None:
    option = _option("episode-a", 0, "left", 0, terminal=True)

    with pytest.raises(ValueError, match="reference differs from behavior"):
        build_critic_dataset(((option,),), reference_epsilon=0.2)


def test_episode_folds_are_deterministic_disjoint_assignments() -> None:
    episodes = tuple(f"episode-{index}" for index in range(12))
    first = whole_episode_folds(episodes, folds=4, seed=9)
    second = whole_episode_folds(tuple(reversed(episodes)), folds=4, seed=9)

    assert first == second
    assert set(dict(first)) == set(episodes)
    assert set(dict(first).values()) == {0, 1, 2, 3}


def test_synthetic_known_sign_survives_crossfit_and_proper_awr() -> None:
    episodes = tuple(
        (
            _option(
                f"episode-{index:03d}",
                0,
                "left" if index % 2 == 0 else "stay",
                0 if index % 2 == 0 else 1,
                terminal=True,
            ),
        )
        for index in range(64)
    )
    dataset = build_critic_dataset(episodes, reference_epsilon=1.0)
    critic = cross_fit_cost_critic(
        dataset,
        folds=4,
        seed=17,
        n_jobs=2,
        n_estimators=80,
    )
    artifact = fit_linear_awr_actor(
        dataset,
        critic,
        reference_epsilon=1.0,
        temperature=0.25,
    )
    scores = dict(linear_actor_scores(artifact, _state()))

    assert critic.example_count == 64
    assert critic.episode_count == 64
    assert critic.maximum_importance_ratio == pytest.approx(1.0)
    assert critic.factual_rmse < 0.2
    assert scores["left"] > scores["stay"]
    distribution = linear_actor_distribution(
        artifact,
        _state(),
        supported_actions=("left", "stay"),
        forecast_accepted_actions=("left", "stay"),
        max_kl=0.1,
    )
    assert dict(distribution.probabilities)["left"] > 0.5
    assert distribution.kl_from_reference <= 0.1 + 1e-10
