import numpy as np

from th06_rl.generation7.awr_actor import (
    AwrConfig,
    _varying_feature_indices,
    crossfit_proper_awr,
    fit_linear_awr_actor,
)
from th06_rl.generation7.feature_contract import compact_actor_feature_names
from th06_rl.generation7.orthogonal_learning import (
    CompactEpisode,
    OrthogonalConfig,
)
from th06_rl.generation7.policy_distribution import ResidualStochasticPolicy


def test_linear_actor_uses_only_within_choice_varying_features() -> None:
    names = compact_actor_feature_names()
    indices = _varying_feature_indices(names)
    selected = tuple(names[index] for index in indices)
    assert selected
    assert all(
        name.startswith("action:")
        or name.startswith("delta_from_baseline:")
        or name.startswith("matches_")
        for name in selected
    )
    assert not any(name.startswith("observation:") for name in selected)


def test_proper_awr_fit_is_finite_and_prefers_high_weight_action() -> None:
    names = compact_actor_feature_names()
    rows = np.zeros((4, len(names)), dtype=np.float32)
    direction_x = names.index("action:direction_x")
    delta_x = names.index("delta_from_baseline:direction_x")
    matches_baseline = names.index("matches_baseline")
    rows[[0, 2], matches_baseline] = 1.0
    rows[[1, 3], direction_x] = 1.0
    rows[[1, 3], delta_x] = 1.0
    episode = CompactEpisode(
        episode_id="synthetic",
        source_id="source",
        cohort_id="cohort",
        stage=6,
        features=rows,
        causal_context_features=np.zeros((2, 0), dtype=np.float32),
        action_indices=np.asarray([0, 1, 0, 1]),
        offsets=np.asarray([0, 2, 4]),
        factual_positions=np.asarray([0, 1]),
        baseline_positions=np.asarray([0, 0]),
        behavior_probabilities=np.asarray([0.5, 0.5, 0.5, 0.5]),
        targets=np.zeros(2),
        hit_costs=np.zeros(2),
    )
    policy = ResidualStochasticPolicy(
        epsilon=0.2,
        temperature=1.0,
        maximum_log_tilt=2.0,
    )
    actor, report = fit_linear_awr_actor(
        (episode,),
        weights={"synthetic": np.asarray([1.0, 10.0])},
        feature_names=names,
        supported_actions=frozenset({0, 1}),
        policy=policy,
        config=AwrConfig(
            temperature=1.0,
            maximum_weight=20.0,
            kl_coefficient=0.01,
            epochs=20,
            learning_rate=0.05,
            l2=0.001,
        ),
    )
    assert np.all(np.isfinite(actor.coefficients))
    assert report["epochs"][-1]["objective"] < report["epochs"][0]["objective"]
    assert actor.scores(rows)[1] > actor.scores(rows)[0]


def test_proper_awr_crossfit_exercises_exact_policy_ope_path() -> None:
    names = compact_actor_feature_names()
    direction_x = names.index("action:direction_x")
    delta_x = names.index("delta_from_baseline:direction_x")
    matches_baseline = names.index("matches_baseline")
    episodes = []
    for episode_index in range(5):
        rows = np.zeros((8, len(names)), dtype=np.float32)
        rows[::2, matches_baseline] = 1.0
        rows[1::2, direction_x] = 1.0
        rows[1::2, delta_x] = 1.0
        factual_positions = np.asarray([
            (episode_index + option) % 2 for option in range(4)
        ])
        costs = np.asarray([
            float(factual_positions[option] == 1 and option == 2)
            for option in range(4)
        ])
        episodes.append(CompactEpisode(
            episode_id=f"episode-{episode_index}",
            source_id="source",
            cohort_id="cohort",
            stage=6,
            features=rows,
            causal_context_features=np.zeros((4, 0), dtype=np.float32),
            action_indices=np.tile(np.asarray([0, 1]), 4),
            offsets=np.arange(0, 9, 2),
            factual_positions=factual_positions,
            baseline_positions=np.zeros(4, dtype=np.int16),
            behavior_probabilities=np.tile(np.asarray([0.5, 0.5]), 4),
            targets=np.asarray([
                costs[index:min(index + 2, 4)].sum() for index in range(4)
            ]),
            hit_costs=costs,
        ))
    result = crossfit_proper_awr(
        tuple(episodes),
        feature_names=names,
        orthogonal_config=OrthogonalConfig(
            folds=5,
            fold_seed=3,
            horizon=2,
            nuisance_ridge_alpha=1.0,
            effect_ridge_alpha=1.0,
            reference_epsilon=0.2,
            policy_temperature=1.0,
            maximum_log_tilt=2.0,
            minimum_action_assignments=1,
            minimum_action_episodes=1,
            fqe_horizon=2,
            fqe_ridge_alpha=1.0,
        ),
        awr_config=AwrConfig(
            temperature=1.0,
            maximum_weight=20.0,
            kl_coefficient=0.01,
            epochs=2,
            learning_rate=0.01,
            l2=0.001,
        ),
    )
    assert len(result["fold_reports"]) == 5
    assert set(result["estimates"]) == {
        "one_step_direct",
        "one_step_ips",
        "one_step_dr",
        "one_step_fqe",
        "sequential_fqe",
        "sequential_dr",
    }
