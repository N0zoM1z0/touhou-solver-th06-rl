from __future__ import annotations

import pytest

from th06_rl.generation7.factual_options import aggregate_factual_episode
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.th06.learning_adapter import ACTION_FEATURE_NAMES, OBSERVATION_FEATURE_NAMES


def _features(names):
    return [[name, 0.0] for name in names]


def _boundary(sequence: int, frame: int, *, hit: bool = False):
    legal = ["stay", "up"]
    probabilities = [["stay", 0.8], ["up", 0.2]]
    return {
        "schema_version": "transition-v10",
        "sequence": sequence,
        "snapshot_ref": f"episode:00000000:f{frame}",
        "executed_action": "up",
        "published_action": "up",
        "proposed_action": "up",
        "baseline_action": "stay",
        "behavior_probability": 0.2,
        "policy_id": "fixture-behavior",
        "learning_eligible": True,
        "legal_actions": legal,
        "option": {
            "boundary": True,
            "boundary_probability": 0.2,
            "behavior_probabilities": probabilities,
            "elapsed_frames_at_decision": 1,
            "intent": "up",
            "option_id": f"option-{sequence}",
            "termination_reason": "horizon",
        },
        "outcome_terms": {
            "authority_lost": False,
            "bomb_used": False,
            "life_lost": hit,
            "elapsed_frames": 8,
        },
        "policy_context": {
            "current_action": "stay",
            "observation_features": _features(OBSERVATION_FEATURE_NAMES),
            "action_features": [
                ["stay", _features(ACTION_FEATURE_NAMES)],
                ["up", _features(ACTION_FEATURE_NAMES)],
            ],
            "history_features": _features(HISTORY_FEATURE_NAMES),
            "hazard_primitives": [],
        },
    }


def test_factual_options_use_only_decision_features_and_conserve_hits() -> None:
    episode = aggregate_factual_episode(
        (_boundary(0, 100, hit=True), _boundary(1, 108, hit=False)),
        episode_id="episode",
        source_id="source",
        transition_schema="transition-v10",
        stage=6,
        manifest_hits=1,
        reconstructible_propensity=False,
    )
    assert len(episode.options) == 2
    assert episode.options[0].hit_cost == 1
    assert episode.options[0].duration_frames == 8
    assert episode.options[-1].terminal is True
    assert episode.options[0].candidate_features[0][-1] == 0.0
    assert episode.options[1].candidate_features[0][-1] > 0.0
    assert episode.options[0].behavior_probabilities == (0.8, 0.2)
    assert episode.options[0].causal_context_features
    assert episode.options[0].proposal_action == "up"
    assert episode.options[0].boundary_executed_action == "up"
    assert episode.options[0].complied is True


def test_v9_propensity_is_reconstructed_from_frozen_behavior_contract() -> None:
    row = _boundary(0, 100)
    row["policy_id"] = "safe-option-exploration-v1"
    row["option"].pop("behavior_probabilities")
    row["behavior_probability"] = 0.05
    row["option"]["boundary_probability"] = 0.05
    episode = aggregate_factual_episode(
        (row,),
        episode_id="episode",
        source_id="source",
        transition_schema="transition-v9",
        stage=6,
        manifest_hits=0,
        reconstructible_propensity=True,
    )
    assert episode.options[0].behavior_probabilities == pytest.approx((0.95, 0.05))


def test_rejected_randomized_proposal_is_retained_as_intention_to_treat() -> None:
    rejected = _boundary(0, 100, hit=True)
    rejected["executed_action"] = "stay"
    rejected["published_action"] = "stay"
    rejected["learning_eligible"] = False
    rejected["option"]["termination_reason"] = "publication-rejected"
    episode = aggregate_factual_episode(
        (rejected, _boundary(1, 108)),
        episode_id="episode",
        source_id="source",
        transition_schema="transition-v10",
        stage=6,
        manifest_hits=1,
        reconstructible_propensity=False,
    )
    assert len(episode.options) == 2
    proposal = episode.options[0]
    assert proposal.proposal_action == "up"
    assert proposal.boundary_executed_action == "stay"
    assert proposal.complied is False
    assert proposal.factual_probability == 0.2
    assert proposal.hit_cost == 1
