from dataclasses import replace

import numpy as np

from th06_rl.generation7.orthogonal_learning import (
    CompactEpisode,
    ContextualEffectModel,
    _ACTION_CORE_INDICES,
    _paired_cluster_difference,
    proposal_propensity_calibration,
    orthogonal_randomization_nulls,
    episode_folds,
)
from th06_rl.generation7.feature_contract import compact_actor_feature_names


def _episode(name: str) -> CompactEpisode:
    return CompactEpisode(
        episode_id=name,
        source_id="source",
        cohort_id="cohort",
        stage=6,
        features=np.zeros((2, 3), dtype=np.float32),
        causal_context_features=np.zeros((1, 0), dtype=np.float32),
        action_indices=np.asarray([0, 1]),
        offsets=np.asarray([0, 2]),
        factual_positions=np.asarray([0]),
        baseline_positions=np.asarray([0]),
        behavior_probabilities=np.asarray([0.5, 0.5]),
        targets=np.asarray([0.0]),
        hit_costs=np.asarray([0.0]),
    )


def test_fold_assignment_is_deterministic_and_episode_atomic() -> None:
    episodes = tuple(_episode(f"episode-{index}") for index in range(10))
    left = episode_folds(episodes, folds=5, seed=7)
    right = episode_folds(episodes, folds=5, seed=7)
    assert left == right
    assert sorted(left) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_fold_identity_uses_episode_id_not_row_count() -> None:
    episodes = tuple(_episode(f"episode-{index}") for index in range(5))
    changed = (replace(episodes[0], targets=np.zeros(100)), *episodes[1:])
    assert episode_folds(episodes, folds=5, seed=9) == episode_folds(
        changed, folds=5, seed=9
    )


def test_contextual_randomization_null_builds_one_dimensional_interaction() -> None:
    width = len(compact_actor_feature_names())
    episode = CompactEpisode(
        episode_id="contextual",
        source_id="source",
        cohort_id="cohort",
        stage=6,
        features=np.zeros((2, width), dtype=np.float32),
        causal_context_features=np.zeros((1, 0), dtype=np.float32),
        action_indices=np.asarray([0, 1]),
        offsets=np.asarray([0, 2]),
        factual_positions=np.asarray([0]),
        baseline_positions=np.asarray([0]),
        behavior_probabilities=np.asarray([0.5, 0.5]),
        targets=np.asarray([0.0]),
        hit_costs=np.asarray([0.0]),
    )
    state_width = 2
    centered_width = width + state_width * len(_ACTION_CORE_INDICES)
    result = orthogonal_randomization_nulls(
        (episode,),
        {
            "nuisance_residuals": {"contextual": np.asarray([1.0])},
            "centered_features": {
                "contextual": np.zeros((1, centered_width))
            },
            "effect_states": {"contextual": np.zeros((1, state_width))},
            "effect_representation": "compact_bilinear",
        },
        replicates=20,
        seed=3,
    )
    assert result["observed_score_norm"] == 0.0


def test_richer_representation_enters_nuisance_and_fqe_design() -> None:
    width = len(compact_actor_feature_names())
    episode = CompactEpisode(
        episode_id="richer",
        source_id="source",
        cohort_id="cohort",
        stage=6,
        features=np.zeros((2, width), dtype=np.float32),
        causal_context_features=np.asarray([[3.0, 4.0]], dtype=np.float32),
        action_indices=np.asarray([0, 1]),
        offsets=np.asarray([0, 2]),
        factual_positions=np.asarray([0]),
        baseline_positions=np.asarray([0]),
        behavior_probabilities=np.asarray([0.5, 0.5]),
        targets=np.asarray([0.0]),
        hit_costs=np.asarray([0.0]),
    )
    compact_state = np.asarray([[1.0, 2.0]])
    model = ContextualEffectModel(estimator=None, representation="richer_bilinear")
    assert model.nuisance_state_rows(episode, compact_state).tolist() == [
        [1.0, 2.0, 3.0, 4.0]
    ]
    q_rows = model.q_candidate_rows(episode, compact_state)
    assert q_rows.shape == (2, width + 4 * len(_ACTION_CORE_INDICES) + 2)
    assert q_rows[:, -2:].tolist() == [[3.0, 4.0], [3.0, 4.0]]


def test_calibration_difference_is_episode_paired() -> None:
    result = _paired_cluster_difference(
        {"a": [3.0, 5.0], "b": [10.0]},
        {"a": [1.0, 1.0], "b": [7.0]},
    )
    assert result["episode_equal_mean"] == 3.0


def test_proposal_propensity_calibration_uses_randomized_assignments() -> None:
    baseline = _episode("baseline")
    alternative = replace(
        _episode("alternative"),
        factual_positions=np.asarray([1]),
    )
    result = proposal_propensity_calibration(
        (baseline, alternative),
        reference_epsilon=0.05,
    )
    assert result["aggregate"]["episode_equal_mean"] == 1.0
    assert result["strata"]["source:source"]["episode_equal_mean"] == 1.0
