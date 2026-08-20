from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from th06_rl.g7_learner import build_critic_dataset
from th06_rl.g7_support import fit_local_support, locally_supported_actions
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
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


def _option(
    episode: str,
    index: int,
    action: str,
    cost: int,
    *,
    terminal: bool,
) -> OfflineOptionTransition:
    state = _state()
    return OfflineOptionTransition(
        schema="th06-rl-causal-options-v3",
        episode_id=episode,
        episode_unit="complete-route",
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
        eligible=True,
        exclusion_reasons=(),
    )


def test_episode_calibrated_support_accepts_seen_state_and_rejects_ood() -> None:
    episodes = tuple(
        (
            _option(
                f"episode-{index:03d}",
                0,
                "left" if index % 2 == 0 else "stay",
                0,
                terminal=True,
            ),
        )
        for index in range(80)
    )
    dataset = build_critic_dataset(episodes, reference_epsilon=1.0)
    artifact = fit_local_support(
        dataset,
        seed=11,
        prototypes_per_action=2,
        minimum_samples=8,
        minimum_ess=8.0,
    )

    assert locally_supported_actions(artifact, _state()) == ("left", "stay")
    state = _state()
    ood = state.__class__(
        tuple(
            (name, 100.0 if name == "position_x_unit" else value)
            for name, value in state.observation_features
        ),
        state.action_features,
        state.history_features,
        state.legal_actions,
        state.baseline_action,
        state.current_action,
    )
    assert locally_supported_actions(artifact, ood) == ()
