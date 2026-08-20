from __future__ import annotations

from types import SimpleNamespace

import pytest

import th06_rl.offline_options as offline_options
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.offline_options import (
    OfflineOptionError,
    iter_offline_options,
    whole_episode_split,
)
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)
from th06_rl.th06.source_dataset import SourceFrameBundle


def _features(names):
    return [[name, float(index)] for index, name in enumerate(names)]


def _bundle(sequence: int) -> SourceFrameBundle:
    return SourceFrameBundle(
        sequence,
        f"episode-1:{sequence:08d}",
        SimpleNamespace(stage=1),
        SimpleNamespace(),
        0,
        "test",
        {"stage": 1, "key": "3/0/0/1/test"},
        {},
    )


def _transition(
    sequence: int,
    *,
    option,
    hit: bool = False,
):
    return {
        "sequence": sequence,
        "scope": {"stage": 1, "key": "3/0/0/1/test"},
        "legal_actions": ["left", "stay"],
        "baseline_action": "stay",
        "executed_action": option["intent"] if option is not None else "stay",
        "behavior_probability": (
            option["conditional_probability"] if option is not None else 1.0
        ),
        "policy_context": {
            "current_action": "stay",
            "observation_features": _features(OBSERVATION_FEATURE_NAMES),
            "action_features": [
                ["left", _features(ACTION_FEATURE_NAMES)],
                ["stay", _features(ACTION_FEATURE_NAMES)],
            ],
            "history_features": _features(HISTORY_FEATURE_NAMES),
        },
        "option": option,
        "outcome_terms": {
            "elapsed_frames": 1,
            "life_lost": hit,
            "bomb_used": False,
            "authority_lost": False,
        },
        "learning_eligible": True,
        "learning_exclusion_reasons": [],
        "episode": {"id": "episode-1", "unit": "complete-route"},
    }


def _option(
    option_id: str,
    intent: str,
    *,
    boundary: bool,
    elapsed: int,
    termination=None,
    probabilities=None,
):
    probabilities = probabilities or [["left", 0.25], ["stay", 0.75]]
    selected = dict(probabilities)[intent]
    return {
        "option_id": option_id,
        "boundary": boundary,
        "intent": intent,
        "boundary_probability": selected,
        "conditional_probability": selected if boundary else 1.0,
        "elapsed_frames_at_decision": elapsed,
        "physical_elapsed_frames": 1,
        "termination_reason": termination,
        "preceding_termination_reason": None,
        "behavior_probabilities": probabilities,
    }


def test_whole_episode_split_is_disjoint_complete_and_deterministic() -> None:
    episodes = tuple(f"episode-{index}" for index in range(10))

    first = whole_episode_split(episodes, validation_fraction=0.2, seed=17)
    second = whole_episode_split(tuple(reversed(episodes)), validation_fraction=0.2, seed=17)

    assert first == second
    training, validation = first
    assert len(training) == 8
    assert len(validation) == 2
    assert not training & validation
    assert training | validation == set(episodes)


def test_whole_episode_split_refuses_row_level_or_single_episode_inference() -> None:
    with pytest.raises(ValueError, match=">=2 episodes"):
        whole_episode_split(("only-one",), validation_fraction=0.2, seed=0)


def test_option_aggregation_stops_at_termination_and_skips_forced_gap(monkeypatch) -> None:
    frames = tuple(_bundle(index) for index in range(5))
    transitions = (
        _transition(0, option=_option("a", "left", boundary=True, elapsed=1)),
        _transition(
            1,
            option=_option(
                "a",
                "left",
                boundary=False,
                elapsed=2,
                termination="physical-hit",
            ),
            hit=True,
        ),
        _transition(2, option=None),
        _transition(
            3,
            option=_option("b", "stay", boundary=True, elapsed=1),
        ),
    )
    joined = tuple(
        (frames[index], frames[index + 1], transition)
        for index, transition in enumerate(transitions)
    )
    monkeypatch.setattr(offline_options, "_joined_steps", lambda _path: iter(joined))

    first, second = tuple(iter_offline_options(SimpleNamespace(resolve=lambda: None)))

    assert first.option_id == "a"
    assert first.physical_hit_cost == 1
    assert first.elapsed_frames == 2
    assert first.next_state is not None
    assert first.terminal is False
    assert first.eligible is True
    assert second.option_id == "b"
    assert second.elapsed_frames == 1
    assert second.next_state is None
    assert second.terminal is True


def test_option_aggregation_requires_full_legal_probability_vector(monkeypatch) -> None:
    frames = (_bundle(0), _bundle(1))
    incomplete = _option(
        "a",
        "stay",
        boundary=True,
        elapsed=1,
        probabilities=[["stay", 1.0]],
    )
    joined = ((frames[0], frames[1], _transition(0, option=incomplete)),)
    monkeypatch.setattr(offline_options, "_joined_steps", lambda _path: iter(joined))

    with pytest.raises(OfflineOptionError, match="propensity vector"):
        tuple(iter_offline_options(SimpleNamespace(resolve=lambda: None)))
