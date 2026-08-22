"""Read-only diagnostics for a frozen masked behavior-cloning artifact."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np

from .actions import ACTION_NAMES
from .bc_features import FEATURE_NAMES, FEATURE_SCHEMA
from .bc_training import (
    BehaviorDataset,
    _bootstrap_delta_interval,
    _expected_calibration_error,
    _frequency_probabilities,
    _metrics,
)
from .core.model import movement_actions


DIAGNOSIS_SCHEMA = "th06-rl-l1c-residual-diagnosis-v1"
EXPLORATION_PROBABILITY = 0.2


def _state_logits(
    dataset: BehaviorDataset,
    state: dict[str, Any],
) -> np.ndarray:
    if (
        state.get("feature_schema") != FEATURE_SCHEMA
        or state.get("feature_names") != list(FEATURE_NAMES)
        or state.get("action_names") != list(ACTION_NAMES)
    ):
        raise ValueError("diagnostic model schema differs from current-observation BC")
    normalization = state.get("normalization")
    model = state.get("model")
    if not isinstance(normalization, dict) or not isinstance(model, dict):
        raise ValueError("diagnostic model lacks normalization or model state")
    mean = np.asarray(normalization.get("mean"), dtype=np.float64)
    scale = np.asarray(normalization.get("scale"), dtype=np.float64)
    weights = np.asarray(model.get("weights"), dtype=np.float64)
    biases = np.asarray(model.get("biases"), dtype=np.float64)
    expected_weights = (len(ACTION_NAMES), len(FEATURE_NAMES))
    if (
        mean.shape != (len(FEATURE_NAMES),)
        or scale.shape != mean.shape
        or weights.shape != expected_weights
        or biases.shape != (len(ACTION_NAMES),)
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(scale))
        or np.any(scale <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(~np.isfinite(biases))
    ):
        raise ValueError("diagnostic model arrays are malformed")
    normalized = (dataset.features - mean) / scale
    logits = normalized @ weights.T + biases
    if np.any(~np.isfinite(logits)):
        raise ValueError("diagnostic model produced non-finite logits")
    return logits


def scaled_probabilities(
    logits: np.ndarray,
    legal_masks: np.ndarray,
    logit_scale: float,
) -> np.ndarray:
    """Apply one positive global logit scale without changing legal actions."""
    if logits.shape != legal_masks.shape or logits.ndim != 2:
        raise ValueError("diagnostic logits and legal masks must align")
    if not math.isfinite(logit_scale) or logit_scale <= 0.0:
        raise ValueError("diagnostic logit scale must be finite and positive")
    masked = np.where(legal_masks, logits * logit_scale, -np.inf)
    maximum = np.max(masked, axis=1, keepdims=True)
    exponentials = np.where(legal_masks, np.exp(masked - maximum), 0.0)
    totals = exponentials.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("scaled diagnostic distribution is invalid")
    return exponentials / totals


def train_optimal_logit_scale(
    logits: np.ndarray,
    dataset: BehaviorDataset,
    *,
    bisection_steps: int = 80,
    maximum_scale: float = 1024.0,
) -> dict[str, float | bool]:
    """Minimize unregularized train NLL over one positive scalar."""
    if logits.shape != dataset.legal_masks.shape:
        raise ValueError("scale-fit logits and dataset do not align")
    if bisection_steps <= 0 or maximum_scale <= 1.0:
        raise ValueError("scale-fit bounds must be positive")
    rows = np.arange(dataset.rows)

    def derivative(value: float) -> float:
        probabilities = scaled_probabilities(logits, dataset.legal_masks, value)
        expected = np.sum(probabilities * logits, axis=1)
        selected = logits[rows, dataset.targets]
        return float(np.mean(expected - selected))

    lower = 1e-12
    lower_derivative = derivative(lower)
    if lower_derivative >= 0.0:
        return {
            "logit_scale": lower,
            "temperature": 1.0 / lower,
            "derivative": lower_derivative,
            "root_bracketed": False,
        }
    upper = 1.0
    upper_derivative = derivative(upper)
    while upper_derivative < 0.0 and upper < maximum_scale:
        upper = min(maximum_scale, upper * 2.0)
        upper_derivative = derivative(upper)
    bracketed = upper_derivative >= 0.0
    if bracketed:
        for _ in range(bisection_steps):
            middle = 0.5 * (lower + upper)
            middle_derivative = derivative(middle)
            if middle_derivative < 0.0:
                lower = middle
            else:
                upper = middle
        optimum = 0.5 * (lower + upper)
    else:
        optimum = upper
    return {
        "logit_scale": float(optimum),
        "temperature": float(1.0 / optimum),
        "derivative": derivative(optimum),
        "root_bracketed": bracketed,
    }


def _collector_probabilities(dataset: BehaviorDataset) -> np.ndarray:
    probabilities = np.zeros_like(dataset.legal_masks, dtype=np.float64)
    for row_index, (mask, baseline) in enumerate(
        zip(dataset.legal_masks, dataset.baseline_targets, strict=True)
    ):
        legal = np.flatnonzero(mask)
        if int(baseline) not in legal:
            raise ValueError("eligible collector row lacks a legal reactive baseline")
        probabilities[row_index, legal] = EXPLORATION_PROBABILITY / len(legal)
        probabilities[row_index, int(baseline)] += 1.0 - EXPLORATION_PROBABILITY
    return probabilities


def _reliability(probabilities: np.ndarray, dataset: BehaviorDataset) -> list[dict[str, float | int]]:
    predictions = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(dataset.rows), predictions]
    correct = predictions == dataset.targets
    indices = np.minimum(np.floor(confidence * 10).astype(np.int64), 9)
    rows = []
    for index in range(10):
        members = indices == index
        count = int(np.sum(members))
        if not count:
            continue
        mean_confidence = float(np.mean(confidence[members]))
        accuracy = float(np.mean(correct[members]))
        rows.append({
            "bin": index,
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "ece_contribution": float(count / dataset.rows * abs(mean_confidence - accuracy)),
        })
    return rows


def _episode_metrics(
    probabilities: np.ndarray,
    dataset: BehaviorDataset,
) -> list[dict[str, float | int | str]]:
    predictions = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(dataset.rows), predictions]
    selected = np.clip(
        probabilities[np.arange(dataset.rows), dataset.targets],
        1e-300,
        1.0,
    )
    result = []
    for index, episode_id in enumerate(dataset.episode_ids):
        members = dataset.episode_indices == index
        correct = predictions[members] == dataset.targets[members]
        result.append({
            "episode_id": episode_id,
            "rows": int(np.sum(members)),
            "negative_log_likelihood": float(-np.mean(np.log(selected[members]))),
            "accuracy": float(np.mean(correct)),
            "mean_top_confidence": float(np.mean(confidence[members])),
            "expected_calibration_error_10_bin": _expected_calibration_error(
                confidence[members], correct
            ),
        })
    return result


def _reactive_structure(
    probabilities: np.ndarray,
    dataset: BehaviorDataset,
) -> dict[str, object]:
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    actions = movement_actions()
    if tuple(action.name for action in actions) != ACTION_NAMES:
        raise ValueError("diagnostic action order differs from learner vocabulary")
    predictions = np.argmax(probabilities, axis=1)
    stage_counts: dict[str, int] = defaultdict(int)
    stage_model_matches: dict[str, int] = defaultdict(int)
    stage_target_matches: dict[str, int] = defaultdict(int)
    reconstructed = np.full(dataset.rows, -1, dtype=np.int64)

    for row_index, (features, legal_mask) in enumerate(
        zip(dataset.features, dataset.legal_masks, strict=True)
    ):
        current_bits = np.asarray([
            features[feature_index[f"current_action:{name}"]]
            for name in ACTION_NAMES
        ])
        if np.sum(current_bits == 1.0) != 1 or np.any(
            (current_bits != 0.0) & (current_bits != 1.0)
        ):
            raise ValueError("diagnostic current-action one-hot is malformed")
        current = int(np.argmax(current_bits))
        candidates = []
        for action_index in np.flatnonzero(legal_mask):
            name = ACTION_NAMES[int(action_index)]
            known = features[feature_index[f"shield_clearance_known:{name}"]]
            clearance = (
                features[feature_index[f"shield_clearance:{name}"]]
                if known == 1.0
                else math.inf
            )
            final_x = features[feature_index[f"shield_final_x:{name}"]]
            final_y = features[feature_index[f"shield_final_y:{name}"]]
            action = actions[int(action_index)]
            candidates.append((
                int(action_index),
                float(clearance),
                float(min(final_x - 8.0, 376.0 - final_x, final_y - 16.0, 432.0 - final_y)),
                int(action_index) == current,
                action.dx == 0 and action.dy == 0,
                action.focused,
                action.name,
            ))
        remaining = candidates
        stage = "lexicographic"
        for stage_name, column in (
            ("clearance", 1),
            ("boundary_reserve", 2),
            ("current_action", 3),
            ("stationary", 4),
            ("focused", 5),
            ("lexicographic", 6),
        ):
            best = max(row[column] for row in remaining)
            remaining = [row for row in remaining if row[column] == best]
            if len(remaining) == 1:
                stage = stage_name
                break
        selected = remaining[0][0]
        reconstructed[row_index] = selected
        stage_counts[stage] += 1
        stage_model_matches[stage] += int(predictions[row_index] == selected)
        stage_target_matches[stage] += int(dataset.targets[row_index] == selected)

    known = dataset.baseline_targets >= 0
    reconstruction_matches = reconstructed[known] == dataset.baseline_targets[known]
    stage_rows = {}
    for stage in sorted(stage_counts):
        count = stage_counts[stage]
        stage_rows[stage] = {
            "rows": count,
            "fraction": count / dataset.rows,
            "model_reactive_agreement": stage_model_matches[stage] / count,
            "sampled_target_reactive_rate": stage_target_matches[stage] / count,
        }
    return {
        "known_reactive_rows": int(np.sum(known)),
        "feature_only_reconstruction_accuracy": float(np.mean(reconstruction_matches)),
        "model_reactive_agreement": float(np.mean(predictions[known] == dataset.baseline_targets[known])),
        "sampled_target_reactive_rate": float(np.mean(dataset.targets[known] == dataset.baseline_targets[known])),
        "decision_stage": stage_rows,
    }


def select_ablation(
    *,
    current_feature_reconstruction_exact: bool,
    scaled_train_nll: float,
    unscaled_train_nll: float,
    scaled_train_ece: float,
    calibration_limit: float,
) -> tuple[str, str]:
    """Route using train-only transformed metrics and structural invariants."""
    if not current_feature_reconstruction_exact:
        return (
            "stop-diagnosis-current-feature-invariant-failed",
            "the frozen feature-only reactive reconstruction must be repaired before an ablation",
        )
    if scaled_train_nll < unscaled_train_nll and scaled_train_ece <= calibration_limit:
        return (
            "train-only-scalar-calibration",
            "one positive train-fitted logit scale repairs train calibration without changing rankings",
        )
    return (
        "small-current-observation-mlp",
        "current features determine the collector but one global scale does not repair train calibration",
    )


def diagnose_l1c_residuals(
    train: BehaviorDataset,
    validation: BehaviorDataset,
    state: dict[str, Any],
    *,
    calibration_limit: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Decompose the frozen L1c residual without fitting a deployable model."""
    train_logits = _state_logits(train, state)
    validation_logits = _state_logits(validation, state)
    train_probabilities = scaled_probabilities(train_logits, train.legal_masks, 1.0)
    validation_probabilities = scaled_probabilities(
        validation_logits, validation.legal_masks, 1.0
    )
    train_frequency = _frequency_probabilities(train, train)
    validation_frequency = _frequency_probabilities(train, validation)
    scale_fit = train_optimal_logit_scale(train_logits, train)
    logit_scale = float(scale_fit["logit_scale"])
    scaled_train_probabilities = scaled_probabilities(
        train_logits, train.legal_masks, logit_scale
    )
    train_metrics = _metrics(train_probabilities, train)
    validation_metrics = _metrics(validation_probabilities, validation)
    scaled_train_metrics = _metrics(scaled_train_probabilities, train)
    train_structure = _reactive_structure(train_probabilities, train)
    validation_structure = _reactive_structure(validation_probabilities, validation)
    reconstruction_exact = bool(
        train_structure["feature_only_reconstruction_accuracy"] == 1.0
        and validation_structure["feature_only_reconstruction_accuracy"] == 1.0
    )
    selected, reason = select_ablation(
        current_feature_reconstruction_exact=reconstruction_exact,
        scaled_train_nll=float(scaled_train_metrics["negative_log_likelihood"]),
        unscaled_train_nll=float(train_metrics["negative_log_likelihood"]),
        scaled_train_ece=float(
            scaled_train_metrics["expected_calibration_error_10_bin"]
        ),
        calibration_limit=calibration_limit,
    )
    collector_train = _collector_probabilities(train)
    collector_validation = _collector_probabilities(validation)
    validation_interval = _bootstrap_delta_interval(
        validation_probabilities,
        validation_frequency,
        validation,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    return {
        "schema": DIAGNOSIS_SCHEMA,
        "selection": {
            "ablation": selected,
            "reason": reason,
            "uses_transformed_validation_metrics": False,
        },
        "global_probability_scale": {
            "fit_split": "train-only",
            **scale_fit,
            "unscaled_train": train_metrics,
            "scaled_train": scaled_train_metrics,
            "calibration_limit": calibration_limit,
            "argmax_rows_changed": int(np.sum(
                np.argmax(train_probabilities, axis=1)
                != np.argmax(scaled_train_probabilities, axis=1)
            )),
            "scaled_validation": None,
        },
        "nonlinear_reactive_tie_breaking": {
            "train": train_structure,
            "validation": validation_structure,
        },
        "temporal_ambiguity": {
            "collector_distribution_determined_by_current_features": reconstruction_exact,
            "collector_random_component": "independent declared 20% uniform-shield draw",
            "history_admitted_for_behavior_target": False,
            "scope_note": "history remains a separate candidate for future-HIT risk",
        },
        "frozen_model_reproduction": {
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_reliability": _reliability(validation_probabilities, validation),
            "validation_by_episode": _episode_metrics(
                validation_probabilities, validation
            ),
            "validation_bc_minus_frequency_nll_95": list(validation_interval),
        },
        "collector_oracle": {
            "definition": "0.8 reactive baseline plus 0.2 uniform over the observed-shield set",
            "train": _metrics(collector_train, train),
            "validation": _metrics(collector_validation, validation),
        },
        "action_frequency": {
            "train": _metrics(train_frequency, train),
            "validation": _metrics(validation_frequency, validation),
        },
    }
