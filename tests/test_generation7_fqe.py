import numpy as np

from th06_rl.generation7.fqe import evaluate_fqe_crosschecks
from th06_rl.generation7.orthogonal_learning import CompactEpisode


def _episode(name: str, costs: tuple[float, ...]) -> CompactEpisode:
    option_count = len(costs)
    return CompactEpisode(
        episode_id=name,
        source_id="source",
        cohort_id="cohort",
        stage=6,
        features=np.zeros((2 * option_count, 1)),
        causal_context_features=np.zeros((option_count, 0)),
        action_indices=np.tile(np.asarray([0, 1]), option_count),
        offsets=np.arange(0, 2 * option_count + 1, 2),
        factual_positions=np.zeros(option_count, dtype=np.int16),
        baseline_positions=np.zeros(option_count, dtype=np.int16),
        behavior_probabilities=np.tile(np.asarray([0.5, 0.5]), option_count),
        targets=np.zeros(option_count),
        hit_costs=np.asarray(costs),
    )


def test_identical_target_and_reference_have_zero_fqe_and_dr_difference() -> None:
    episodes = (_episode("train", (0.0, 1.0, 0.0)), _episode("held", (1.0, 0.0)))
    factual = {index: np.zeros((episode.option_count, 2)) for index, episode in enumerate(episodes)}
    expected = {index: np.zeros((episode.option_count, 2)) for index, episode in enumerate(episodes)}
    probabilities = {
        index: episode.behavior_probabilities.copy()
        for index, episode in enumerate(episodes)
    }
    result = evaluate_fqe_crosschecks(
        episodes,
        train_indices=(0,),
        held_indices=(1,),
        factual_rows=factual,
        behavior_expected_rows=expected,
        target_expected_rows=expected,
        reference_expected_rows=expected,
        target_candidate_probabilities=probabilities,
        reference_candidate_probabilities=probabilities,
        horizon=2,
        ridge_alpha=1.0,
    )
    for name in ("one_step_fqe", "sequential_fqe", "sequential_dr"):
        assert result["estimates"][name]["held"] == [0.0, 0.0]
    for diagnostic in result["cumulative_weight_diagnostics"].values():
        assert diagnostic["maximum"] == 1.0
