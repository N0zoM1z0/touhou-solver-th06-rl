from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np

from th06_rl.implicit_learning import delayed_effect_episodes
from th06_rl.iql_actor_learning import (
    IqlActorModel,
    NativeIqlActorPopulation,
    action_centered_actor_losses,
    actor_arrays,
    cross_fitted_factual_advantages,
    iql_actor_model_artifact,
    iql_actor_model_from_artifact,
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


def test_native_iql_actor_population_matches_portable_models() -> None:
    library = (
        Path(__file__).resolve().parents[1]
        / "build/native/libth06_rl_ranker.so"
    )
    if not library.is_file():
        import pytest
        pytest.skip("native ranker build is absent")
    generator = np.random.default_rng(4813)
    layout = FeatureRoleLayout(
        names=("observation:x", "action:dx", "observation:y"),
        state_indices=(0, 2),
        action_indices=(1,),
    )

    def model() -> IqlActorModel:
        return IqlActorModel(
            layout=layout,
            state_mean=np.asarray([0.2, -0.1], dtype=np.float32),
            state_scale=np.asarray([1.3, 0.7], dtype=np.float32),
            action_mean=np.asarray([0.1], dtype=np.float32),
            action_scale=np.asarray([0.8], dtype=np.float32),
            state_hidden_weight=generator.normal(size=(2, 5)).astype(np.float32),
            state_hidden_bias=generator.normal(size=5).astype(np.float32),
            state_latent_weight=generator.normal(size=(5, 3)).astype(np.float32),
            state_latent_bias=generator.normal(size=3).astype(np.float32),
            action_hidden_weight=generator.normal(size=(1, 5)).astype(np.float32),
            action_hidden_bias=generator.normal(size=5).astype(np.float32),
            action_latent_weight=generator.normal(size=(5, 3)).astype(np.float32),
            action_latent_bias=generator.normal(size=3).astype(np.float32),
            action_score_weight=generator.normal(size=5).astype(np.float32),
            action_score_bias=float(generator.normal()),
        )

    portable = [model(), model(), model()]
    roundtrip = iql_actor_model_from_artifact(
        iql_actor_model_artifact(portable[0])
    )
    rows = [(0.7, value, -0.4) for value in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    assert np.allclose(roundtrip.predict(rows), portable[0].predict(rows))

    native = NativeIqlActorPopulation(
        library,
        expected_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        models=portable,
    )
    expected = np.asarray([item.predict(rows) for item in portable])
    actual = np.asarray(native.predict(rows))
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5)
