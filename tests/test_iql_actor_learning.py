from __future__ import annotations

from dataclasses import replace

import numpy as np

from th06_rl.implicit_learning import delayed_effect_episodes
from th06_rl.iql_actor_learning import (
    action_centered_actor_losses,
    actor_arrays,
    cross_fitted_factual_advantages,
)
from th06_rl.low_rank_learning import FeatureRoleLayout


def test_actor_arrays_keep_variable_safe_sets_and_factual_index() -> None:
    raw = delayed_effect_episodes(count=2, options=6, delay=2)
    samples = []
    for index, sample in enumerate(raw):
        if index % 2:
            samples.append(replace(
                sample,
                action="stay",
                behavior_probability=1.0,
                behavior_probabilities=(1.0,),
                vector=sample.candidate_vectors[0],
                legal_actions=("stay",),
                candidate_vectors=(sample.candidate_vectors[0],),
            ))
        else:
            samples.append(sample)
    layout = FeatureRoleLayout(
        names=("observation:p0", "observation:p1", "action:treatment"),
        state_indices=(0, 1),
        action_indices=(2,),
    )

    prepared = actor_arrays(samples, layout)

    assert prepared.actions.shape == (12, 2, 1)
    assert np.array_equal(prepared.masks.sum(axis=1), [2, 1] * 6)
    assert np.allclose(prepared.behavior_probabilities.sum(axis=1), 1.0)
    assert all(
        sample.legal_actions[index] == sample.action
        for sample, index in zip(samples, prepared.factual, strict=True)
    )


def test_action_centered_actor_loss_is_unbiased_without_local_weight_normalization(
) -> None:
    propensity = np.asarray([0.2, 0.3, 0.5])
    action_losses = np.asarray([1.7, 0.4, 2.2])
    advantage_weights = np.asarray([0.5, 2.0, 1.1])
    behavior_loss = np.sum(propensity * action_losses)

    estimates = action_centered_actor_losses(
        action_losses,
        np.full(3, behavior_loss),
        advantage_weights,
    )

    assert np.isclose(
        np.sum(propensity * estimates),
        np.sum(propensity * advantage_weights * action_losses),
    )


def test_actor_advantage_labels_are_cross_fitted_by_complete_episode() -> None:
    samples = delayed_effect_episodes(count=6, options=16, delay=3)

    advantages, report = cross_fitted_factual_advantages(
        samples,
        folds=3,
        critic_iterations=1,
        n_step_options=2,
        q_trees=2,
        value_trees=2,
        threads=1,
    )

    assert advantages.shape == (len(samples),)
    assert np.all(np.isfinite(advantages))
    assert report["all_labels_out_of_episode"] is True
    assert sum(row["options"] for row in report["folds"]) == len(samples)
    assert all(
        set(row["fit_episodes"]).isdisjoint(row["heldout_episodes"])
        for row in report["folds"]
    )
