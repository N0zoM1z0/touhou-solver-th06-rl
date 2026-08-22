from __future__ import annotations

import numpy as np
import torch
import xgboost

from th06_rl.factual_hazard_model import fit_action_conditioned_hazard_models
from th06_rl.factual_primitive_log_hazard_model import (
    PRIMITIVE_LOG_HAZARD_FIT_SCHEMA,
    TRAINING_PROPER_SCORE,
    benchmark_primitive_log_hazard,
    evaluate_primitive_log_hazard_models,
    fit_primitive_log_hazard_models,
    primitive_log_hazard_predictions,
)

from tests.test_factual_primitive_hazard_model import _dataset, _fit as _fit_brier


def _fit_logscore(dataset):
    return fit_primitive_log_hazard_models(
        dataset,
        horizon=1,
        token_cap=8,
        token_hidden=4,
        head_hidden_1=6,
        head_hidden_2=4,
        epochs=2,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0001,
        gradient_clip_norm=5.0,
        seed=23,
        threads=2,
        expected_torch_version=torch.__version__,
    )


def test_logscore_fit_is_deterministic_and_changes_only_loss(tmp_path) -> None:
    dataset = _dataset(tmp_path, "train")
    first = _fit_logscore(dataset)
    second = _fit_logscore(dataset)

    assert first == second
    assert first["schema"] == PRIMITIVE_LOG_HAZARD_FIT_SCHEMA
    assert first["training_proper_score"] == TRAINING_PROPER_SCORE
    assert first["positive_class_weight"] == 1.0
    assert first["negative_class_weight"] == 1.0
    assert first["parameter_count_per_model"] == 275
    assert first["shared_initialization"] is True
    assert first["shared_minibatch_order"] is True
    assert set(first["models"]) == {
        "object_full",
        "scalar_only",
        "object_current_action_ablated",
    }
    view = dataset.horizons[0]
    probability, surface = primitive_log_hazard_predictions(
        first,
        view.scalar_features,
        view.primitive_tokens,
        view.primitive_masks,
        model_name="object_full",
        batch_size=4,
    )
    assert np.all((0.0 <= probability) & (probability <= 1.0))
    assert surface["saturated_rows"] == 0


def test_bce_retains_missed_positive_logit_gradient() -> None:
    logit = torch.tensor([-20.0], requires_grad=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, torch.ones_like(logit)
    )
    loss.backward()

    assert logit.grad is not None
    assert float(logit.grad[0]) < -0.999


def test_logscore_evaluation_retains_all_l2i_gates_and_no_policy(tmp_path) -> None:
    train = _dataset(tmp_path, "train")
    evaluation = _dataset(tmp_path, "evaluation")
    state = _fit_logscore(train)
    state["train"]["inference_benchmark"] = benchmark_primitive_log_hazard(
        state,
        train,
        batch_rows=4,
        warmup_repetitions=2,
        measured_repetitions=5,
        threads=1,
    )
    frozen_l2i = _fit_brier(train)
    view = train.horizons[0]
    frozen_l2f = fit_action_conditioned_hazard_models(
        type("Factual", (), {"horizons": (type("View", (), {
            "horizon": 1,
            "features": view.current_features,
            "hit_labels": view.hit_labels,
            "rows": view.rows,
        })(),)})(),
        horizon=1,
        boosted_rounds=4,
        maximum_depth=2,
        learning_rate=0.1,
        minimum_child_weight=1.0,
        l2_leaf_regularization=1.0,
        maximum_histogram_bins=16,
        seed=7,
        expected_xgboost_version=xgboost.__version__,
    )
    result = evaluate_primitive_log_hazard_models(
        state,
        frozen_l2f,
        frozen_l2i,
        evaluation,
        prediction_batch_size=4,
        bootstrap_samples=20,
        bootstrap_seed=29,
        calibration_bins=10,
        minimum_overall_positives=1,
        minimum_overall_negatives=1,
        minimum_nonbaseline_positives=1,
        minimum_low_propensity_positives=1,
        minimum_prefirst_hit_positives=1,
        minimum_object_gain_episodes=1,
        minimum_overall_episodes_favoring_full=1,
        minimum_nonbaseline_episodes_favoring_full=1,
        minimum_low_propensity_episodes_favoring_full=1,
        minimum_prefirst_episodes_favoring_full=1,
        calibration_in_the_large_absolute_max=1.0,
        full_ece_over_action_ablated_max=1.0,
        maximum_saturated_fraction=1.0,
        maximum_parameter_count=100_000,
        maximum_batch18_p99_ms=1000.0,
        minimum_loss_correction_episodes=1,
    )

    assert "frozen_l2i_object_full_same_rows" in result["metrics"]
    assert "loss_only_improves_frozen_l2i" in result["gates"]
    assert result["summary"]["logscore_probability_model_tested"] is True
    assert result["summary"]["independent_confirmation"] is False
    assert result["summary"]["counterfactual_successors"] is False
    assert result["summary"]["history_admitted"] is False
    assert result["summary"]["value_learning_admitted"] is False
    assert result["summary"]["online_policy_admitted"] is False
