from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import numpy as np

from th06_rl.implicit_learning import delayed_effect_episodes
from th06_rl.iql_actor_learning import (
    IqlActorModel,
    NativeIqlActorPopulation,
    action_centered_actor_losses,
    actor_arrays,
    categorical_kl_from_logits,
    cross_fitted_factual_advantages,
    iql_actor_model_artifact,
    iql_actor_model_from_artifact,
    native_actor_prediction_tolerance_ratio,
    summarize_iql_actor_episodes,
)
from th06_rl.low_rank_learning import FeatureRoleLayout


def test_categorical_kl_stays_finite_for_extreme_actor_logits() -> None:
    value = categorical_kl_from_logits([0.5, 0.5], [0.0, -1000.0])
    assert math.isfinite(value)
    assert value > 400.0


def test_native_actor_prediction_tolerance_scales_with_float32_logits() -> None:
    assert native_actor_prediction_tolerance_ratio(
        [0.0, 1390.0], [0.00009, 1390.000244140625]
    ) <= 1.0
    assert native_actor_prediction_tolerance_ratio(
        [0.0, 1390.0], [0.001, 1390.01]
    ) > 1.0


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


def test_action_centered_actor_loss_is_not_a_proper_optimization_objective(
) -> None:
    """Keep the Generation-6 failure reproducible for successor smokes."""
    empirical = []
    proper = []
    for probability in (1e-2, 1e-6, 1e-12, 1e-30):
        action_losses = np.asarray([
            -np.log(probability), -np.log1p(-probability),
        ])
        behavior_loss = np.full(2, 0.5 * action_losses.sum())
        advantage_weights = np.asarray([0.1, 1.0])
        empirical.append(float(action_centered_actor_losses(
            action_losses, behavior_loss, advantage_weights,
        )[0]))
        proper.append(float(np.sum(
            np.asarray([0.5, 0.5]) * advantage_weights * action_losses
        )))

    assert all(left > right for left, right in zip(
        empirical[:-1], empirical[1:], strict=True
    ))
    assert empirical[-1] < -20.0
    assert all(value >= 0.0 for value in proper)
    assert proper[-1] > proper[0]


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


def test_policy_episode_bootstrap_is_invariant_to_mapping_order() -> None:
    rows = {
        f"episode-{index}": {
            "cohort": "stage-x",
            "options": 10,
            "full_proposals": 0,
            "full_proposal_loo_exact": 0,
            "loo_union": 0,
            "loo_exact": 0,
            "split_union": 0,
            "split_exact": 0,
            "individual_proposals": 0,
            "individual_union": 0,
            "individual_exact": 0,
            "mean_proposals": 1,
            "mean_loo_union": 1,
            "mean_loo_exact": 1,
            "policy_intervention_exposure": 0.1,
            "policy_model_effect": -0.1 * index,
            "policy_dr_effect": float(index - 4),
            "policy_max_abs_correction": 0.2,
            "policy_loo_dr_effect": [float(index - member) for member in range(7)],
            "behavior_kl_sum": 0.3,
        }
        for index in range(8)
    }

    forward = summarize_iql_actor_episodes(
        rows, cohort_names=("stage-x",)
    )
    reverse = summarize_iql_actor_episodes(
        dict(reversed(tuple(rows.items()))), cohort_names=("stage-x",)
    )

    assert forward == reverse
