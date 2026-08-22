from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from th06_rl.actions import ACTION_NAMES
from th06_rl.bc_target_diagnostics import (
    _distribution_metrics,
    _loss_and_gradient,
    diagnose_l1d_target_contract,
    load_propensity_dataset,
)
from th06_rl.corpus import CorpusRecorder, RunMetadata

from tests.test_episode_dataset import _decision, _snapshot


def _mixture_decision(
    *,
    published: str,
    baseline: str,
    current: str = "stay",
):
    legal = ("left", "right")
    probabilities = (("left", 0.9 if baseline == "left" else 0.1),
                     ("right", 0.9 if baseline == "right" else 0.1))
    return replace(
        _decision("ok", current=current, published=published, legal=legal),
        baseline_action=baseline,
        behavior_probability=dict(probabilities)[published],
        behavior_probabilities=probabilities,
        policy_id="uniform-shield-exploration-v1",
    )


def _mixture_episode(root: Path) -> Path:
    recorder = CorpusRecorder(
        root,
        RunMetadata(root.name, "exe", "native", "test", 3, 0, 0, 4, {}),
    )
    recorder.record(
        _snapshot(0, x=100.0),
        _mixture_decision(published="left", baseline="left"),
    )
    recorder.record(
        _snapshot(1, x=101.0),
        _mixture_decision(published="right", baseline="left", current="left"),
    )
    recorder.record(
        _snapshot(2, x=102.0, in_menu=True),
        _decision("passive", current="right"),
    )
    return recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "physical_hits": 0,
    })


def test_propensity_loader_retains_the_exact_declared_mixture(tmp_path: Path) -> None:
    episode = _mixture_episode(tmp_path / "episode")

    dataset = load_propensity_dataset(
        (episode,),
        exploration_probability=0.2,
    )

    left = ACTION_NAMES.index("left")
    right = ACTION_NAMES.index("right")
    assert dataset.rows == 2
    assert dataset.maximum_declared_mixture_error == 0.0
    assert dataset.policy_ids == (("uniform-shield-exploration-v1", 2),)
    assert np.allclose(dataset.behavior_targets[:, left], 0.9)
    assert np.allclose(dataset.behavior_targets[:, right], 0.1)
    assert tuple(dataset.targets) == (left, right)


def test_exact_propensity_is_zero_kl_even_when_sampled_action_differs(
    tmp_path: Path,
) -> None:
    dataset = load_propensity_dataset(
        (_mixture_episode(tmp_path / "episode"),),
        exploration_probability=0.2,
    )

    metrics = _distribution_metrics(dataset.behavior_targets, dataset)

    assert abs(metrics["kl_recorded_mu_to_model"]) < 1e-12
    assert metrics["soft_brier_to_recorded_mu"] == 0.0
    assert metrics["soft_top_label_calibration_error_10"] == 0.0
    assert metrics["hard_sample_nll"] > metrics["target_entropy"]


def test_hard_and_soft_targets_produce_distinct_finite_gradients() -> None:
    random = np.random.default_rng(4)
    rows = 5
    features = random.normal(size=(rows, 114))
    legal = np.zeros((rows, len(ACTION_NAMES)), dtype=np.bool_)
    left = ACTION_NAMES.index("left")
    right = ACTION_NAMES.index("right")
    legal[:, (left, right)] = True
    soft = np.zeros_like(legal, dtype=np.float64)
    soft[:, left] = 0.9
    soft[:, right] = 0.1
    hard = np.zeros_like(soft)
    hard[:, left] = 1.0
    parameters = (
        random.normal(scale=0.1, size=(32, 114)),
        np.zeros(32),
        random.normal(scale=0.1, size=(len(ACTION_NAMES), 32)),
        np.zeros(len(ACTION_NAMES)),
    )

    hard_values = _loss_and_gradient(features, legal, hard, parameters, l2=1e-4)
    soft_values = _loss_and_gradient(features, legal, soft, parameters, l2=1e-4)

    assert np.all(np.isfinite(hard_values[0]))
    assert np.all(np.isfinite(soft_values[0]))
    assert hard_values[3] > 0.0
    assert soft_values[3] > 0.0
    assert not np.isclose(hard_values[1], soft_values[1])
    assert any(
        not np.allclose(hard_gradient, soft_gradient)
        for hard_gradient, soft_gradient in zip(
            hard_values[2], soft_values[2], strict=True
        )
    )


def test_paired_continuations_never_evaluate_validation_branches(
    tmp_path: Path,
) -> None:
    train = load_propensity_dataset(
        (_mixture_episode(tmp_path / "train"),),
        exploration_probability=0.2,
    )
    validation = load_propensity_dataset(
        (_mixture_episode(tmp_path / "validation"),),
        exploration_probability=0.2,
    )
    random = np.random.default_rng(5)
    state = {
        "normalization": {
            "mean": np.mean(train.features, axis=0).tolist(),
            "scale": np.ones(train.features.shape[1]).tolist(),
        },
        "model": {
            "input_weights": random.normal(
                scale=0.01, size=(32, train.features.shape[1])
            ).tolist(),
            "hidden_biases": np.zeros(32).tolist(),
            "output_weights": random.normal(
                scale=0.01, size=(len(ACTION_NAMES), 32)
            ).tolist(),
            "output_biases": np.zeros(len(ACTION_NAMES)).tolist(),
        },
    }

    observed = diagnose_l1d_target_contract(
        train,
        validation,
        state,
        exploration_probability=0.2,
        continuation_updates=1,
        continuation_checkpoints=(0, 1),
        learning_rate=0.05,
        l2=1e-4,
        mixture_tolerance=1e-15,
        material_kl_reduction=0.01,
        material_soft_target_advantage=0.005,
    )

    continuations = observed["train_only_continuations"]
    assert continuations["validation_evaluated_for_continuations"] is False
    assert continuations["continuation_parameters_serialized"] is False
    assert observed["corpus_propensity_audit"]["exact"] is True
    assert "validation" in observed["frozen_l1d"]
