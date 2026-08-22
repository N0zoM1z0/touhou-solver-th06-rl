"""One fixed small nonlinear behavior-cloning ablation over frozen L1 data."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .actions import ACTION_NAMES
from .bc_features import FEATURE_NAMES, FEATURE_SCHEMA
from .bc_training import (
    CALIBRATION_SCHEMA,
    BehaviorDataset,
    _bootstrap_delta_interval,
    _frequency_probabilities,
    _masked_probabilities,
    _metrics,
    load_behavior_dataset,
)
from .policies.linear_behavior_clone import STATE_SCHEMA as LINEAR_STATE_SCHEMA
from .policies.small_mlp_behavior_clone import (
    DECISION_EPOCH_SCHEMA,
    HIDDEN_WIDTH,
    POLICY_NAME,
    STATE_SCHEMA,
    TARGET_SCHEMA,
    TRAINING_SCHEMA,
)
from .policy_api import POLICY_API_VERSION


INITIALIZATION_KIND = "fixed-seed-he-normal"
RNG_KIND = "numpy-pcg64"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mlp_probabilities(
    features: np.ndarray,
    legal_masks: np.ndarray,
    input_weights: np.ndarray,
    hidden_biases: np.ndarray,
    output_weights: np.ndarray,
    output_biases: np.ndarray,
) -> np.ndarray:
    hidden = np.maximum(0.0, features @ input_weights.T + hidden_biases)
    return _masked_probabilities(hidden, legal_masks, output_weights, output_biases)


def _load_linear_comparator(
    path: Path,
    validation: BehaviorDataset,
) -> tuple[dict[str, object], np.ndarray, str]:
    path = path.resolve()
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("schema") != LINEAR_STATE_SCHEMA:
        raise ValueError("frozen comparator is not a linear behavior-clone state")
    if (
        state.get("feature_schema") != FEATURE_SCHEMA
        or tuple(state.get("feature_names", ())) != FEATURE_NAMES
        or tuple(state.get("action_names", ())) != ACTION_NAMES
    ):
        raise ValueError("frozen comparator task vocabulary differs")
    normalization = state.get("normalization")
    model = state.get("model")
    if not isinstance(normalization, dict) or not isinstance(model, dict):
        raise ValueError("frozen comparator lacks numeric state")
    try:
        mean = np.asarray(normalization["mean"], dtype=np.float64)
        scale = np.asarray(normalization["scale"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        biases = np.asarray(model["biases"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen comparator numeric state is malformed") from error
    if (
        mean.shape != (len(FEATURE_NAMES),)
        or scale.shape != mean.shape
        or weights.shape != (len(ACTION_NAMES), len(FEATURE_NAMES))
        or biases.shape != (len(ACTION_NAMES),)
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(scale))
        or np.any(scale <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(~np.isfinite(biases))
    ):
        raise ValueError("frozen comparator numeric dimensions differ")
    probabilities = _masked_probabilities(
        (validation.features - mean) / scale,
        validation.legal_masks,
        weights,
        biases,
    )
    observed = _metrics(probabilities, validation)
    recorded_fit = state.get("fit")
    recorded = recorded_fit.get("validation") if isinstance(recorded_fit, dict) else None
    if not isinstance(recorded, dict):
        raise ValueError("frozen comparator lacks recorded validation metrics")
    for key in (
        "negative_log_likelihood",
        "accuracy",
        "brier_score",
        "expected_calibration_error_10_bin",
    ):
        if not math.isclose(
            float(observed[key]),
            float(recorded[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"frozen comparator did not reproduce {key}")
    return state, probabilities, _sha256(path)


def fit_small_mlp_behavior_clone(
    train_runs: tuple[Path, ...],
    validation_runs: tuple[Path, ...],
    *,
    linear_comparator_state: Path,
    epochs: int = 10_000,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    seed: int = 0,
    bootstrap_samples: int = 2_000,
    calibration_tolerance: float = 0.02,
    minimum_updates: int = 100,
    relative_gradient_l2_tolerance: float = 0.01,
    hidden_width: int = HIDDEN_WIDTH,
    max_rows: int = 2_000_000,
    code_commit: str = "fixture",
    policy_plugin_sha256: str = "fixture",
) -> dict[str, object]:
    """Fit the one preregistered 114--32--18 ReLU ablation."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if hidden_width != HIDDEN_WIDTH:
        raise ValueError(f"hidden width is frozen at {HIDDEN_WIDTH}")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be finite and nonnegative")
    if not math.isfinite(calibration_tolerance) or calibration_tolerance < 0.0:
        raise ValueError("calibration tolerance must be finite and nonnegative")
    if not 0 <= minimum_updates <= epochs:
        raise ValueError("minimum updates must be between zero and maximum epochs")
    if (
        not math.isfinite(relative_gradient_l2_tolerance)
        or not 0.0 < relative_gradient_l2_tolerance < 1.0
    ):
        raise ValueError("relative gradient L2 tolerance must be in (0, 1)")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if not code_commit or not policy_plugin_sha256:
        raise ValueError("fit provenance cannot be empty")

    train = load_behavior_dataset(train_runs, max_rows=max_rows)
    validation = load_behavior_dataset(validation_runs, max_rows=max_rows)
    overlap = set(train.episode_ids) & set(validation.episode_ids)
    if overlap:
        raise ValueError(f"train/validation episode leakage: {sorted(overlap)}")

    mean = np.mean(train.features, axis=0)
    scale = np.std(train.features, axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    train_features = (train.features - mean) / scale
    validation_features = (validation.features - mean) / scale

    random = np.random.default_rng(seed)
    input_weights = random.normal(
        0.0,
        math.sqrt(2.0 / len(FEATURE_NAMES)),
        size=(hidden_width, len(FEATURE_NAMES)),
    )
    hidden_biases = np.zeros(hidden_width, dtype=np.float64)
    output_weights = random.normal(
        0.0,
        math.sqrt(2.0 / hidden_width),
        size=(len(ACTION_NAMES), hidden_width),
    )
    output_biases = np.zeros(len(ACTION_NAMES), dtype=np.float64)

    def training_step_values() -> tuple[float, tuple[np.ndarray, ...], float]:
        hidden_pre = train_features @ input_weights.T + hidden_biases
        hidden = np.maximum(0.0, hidden_pre)
        probabilities = _masked_probabilities(
            hidden,
            train.legal_masks,
            output_weights,
            output_biases,
        )
        selected = np.clip(
            probabilities[np.arange(train.rows), train.targets],
            1e-300,
            1.0,
        )
        regularizer = 0.5 * l2 * (
            np.sum(input_weights ** 2) + np.sum(output_weights ** 2)
        )
        loss = float(-np.mean(np.log(selected)) + regularizer)
        errors = probabilities.copy()
        errors[np.arange(train.rows), train.targets] -= 1.0
        errors = np.where(train.legal_masks, errors, 0.0)
        gradient_output_weights = errors.T @ hidden / train.rows + l2 * output_weights
        gradient_output_biases = np.mean(errors, axis=0)
        hidden_errors = (errors @ output_weights) * (hidden_pre > 0.0)
        gradient_input_weights = (
            hidden_errors.T @ train_features / train.rows + l2 * input_weights
        )
        gradient_hidden_biases = np.mean(hidden_errors, axis=0)
        gradients = (
            gradient_input_weights,
            gradient_hidden_biases,
            gradient_output_weights,
            gradient_output_biases,
        )
        gradient_l2 = float(math.sqrt(sum(
            float(np.sum(gradient ** 2)) for gradient in gradients
        )))
        return loss, gradients, gradient_l2

    first_loss, gradients, initial_gradient_l2 = training_step_values()
    gradient_l2 = initial_gradient_l2
    updates_completed = 0
    optimization_converged = False
    while updates_completed < epochs:
        if (
            updates_completed >= minimum_updates
            and gradient_l2
            <= relative_gradient_l2_tolerance * initial_gradient_l2
        ):
            optimization_converged = True
            break
        (
            gradient_input_weights,
            gradient_hidden_biases,
            gradient_output_weights,
            gradient_output_biases,
        ) = gradients
        input_weights -= learning_rate * gradient_input_weights
        hidden_biases -= learning_rate * gradient_hidden_biases
        output_weights -= learning_rate * gradient_output_weights
        output_biases -= learning_rate * gradient_output_biases
        updates_completed += 1
        _, gradients, gradient_l2 = training_step_values()
    if (
        updates_completed >= minimum_updates
        and gradient_l2 <= relative_gradient_l2_tolerance * initial_gradient_l2
    ):
        optimization_converged = True
    gradient_ratio = (
        gradient_l2 / initial_gradient_l2 if initial_gradient_l2 > 0.0 else 0.0
    )

    train_probabilities = _mlp_probabilities(
        train_features,
        train.legal_masks,
        input_weights,
        hidden_biases,
        output_weights,
        output_biases,
    )
    validation_probabilities = _mlp_probabilities(
        validation_features,
        validation.legal_masks,
        input_weights,
        hidden_biases,
        output_weights,
        output_biases,
    )
    frequency_probabilities = _frequency_probabilities(train, validation)
    comparator_state, comparator_probabilities, comparator_sha256 = (
        _load_linear_comparator(linear_comparator_state, validation)
    )
    train_metrics = _metrics(train_probabilities, train)
    validation_metrics = _metrics(validation_probabilities, validation)
    frequency_metrics = _metrics(frequency_probabilities, validation)
    comparator_metrics = _metrics(comparator_probabilities, validation)
    direct_interval = _bootstrap_delta_interval(
        validation_probabilities,
        comparator_probabilities,
        validation,
        seed=seed,
        samples=bootstrap_samples,
    )
    frequency_interval = _bootstrap_delta_interval(
        validation_probabilities,
        frequency_probabilities,
        validation,
        seed=seed,
        samples=bootstrap_samples,
    )
    direct_gate_passed = bool(
        len(validation.episode_ids) >= 2 and direct_interval[1] < 0.0
    )
    frequency_gate_passed = bool(
        len(validation.episode_ids) >= 2 and frequency_interval[1] < 0.0
    )
    calibration_passed = bool(
        validation_metrics["expected_calibration_error_10_bin"]
        <= frequency_metrics["expected_calibration_error_10_bin"]
        + calibration_tolerance
    )
    gate_passed = bool(
        optimization_converged and direct_gate_passed and calibration_passed
    )

    base_state: dict[str, object] = {
        "schema": STATE_SCHEMA,
        "training_schema": TRAINING_SCHEMA,
        "decision_epoch_schema": DECISION_EPOCH_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "policy_api_version": POLICY_API_VERSION,
        "provenance": {
            "code_commit": code_commit,
            "policy_plugin_sha256": policy_plugin_sha256,
            "frozen_linear_comparator_sha256": comparator_sha256,
            "frozen_linear_comparator_policy_id": comparator_state.get("policy_id"),
        },
        "normalization": {
            "mean": [float(value) for value in mean],
            "scale": [float(value) for value in scale],
        },
        "model": {
            "kind": "masked-one-hidden-relu-softmax",
            "hidden_width": hidden_width,
            "input_weights": input_weights.tolist(),
            "hidden_biases": hidden_biases.tolist(),
            "output_weights": output_weights.tolist(),
            "output_biases": output_biases.tolist(),
        },
        "initialization": {
            "kind": INITIALIZATION_KIND,
            "rng": RNG_KIND,
            "seed": seed,
            "input_weight_standard_deviation": math.sqrt(
                2.0 / len(FEATURE_NAMES)
            ),
            "output_weight_standard_deviation": math.sqrt(2.0 / hidden_width),
            "biases": "all-zero",
        },
        "sampling": {"kind": "seeded-categorical", "seed": seed},
        "fit": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "calibration_tolerance": calibration_tolerance,
            "calibration_schema": CALIBRATION_SCHEMA,
            "initial_regularized_nll": first_loss,
            "optimization": {
                "kind": "full-batch-gradient-descent",
                "maximum_updates": epochs,
                "minimum_updates": minimum_updates,
                "relative_gradient_l2_tolerance": relative_gradient_l2_tolerance,
                "updates_completed": updates_completed,
                "converged": optimization_converged,
                "stop_reason": (
                    "relative-gradient-l2"
                    if optimization_converged
                    else "maximum-updates"
                ),
                "initial_gradient_l2": initial_gradient_l2,
                "final_gradient_l2": gradient_l2,
                "final_to_initial_gradient_l2_ratio": gradient_ratio,
            },
            "train": {
                "episodes": len(train.episode_ids),
                "rows": train.rows,
                **train_metrics,
            },
            "validation": {
                "episodes": len(validation.episode_ids),
                "rows": validation.rows,
                **validation_metrics,
            },
            "frozen_linear_validation": comparator_metrics,
            "action_frequency_validation": frequency_metrics,
            "episode_bootstrap_nll_delta_vs_frozen_linear_95": list(
                direct_interval
            ),
            "episode_bootstrap_nll_delta_vs_action_frequency_95": list(
                frequency_interval
            ),
            "direct_l1c_nll_gate_passed": direct_gate_passed,
            "frequency_nll_gate_passed": frequency_gate_passed,
            "calibration_gate_passed": calibration_passed,
            "optimization_gate_passed": optimization_converged,
            "learnability_gate_passed": gate_passed,
        },
        "inventory": {
            "train": list(train.inventory),
            "validation": list(validation.inventory),
        },
    }
    canonical = json.dumps(
        base_state,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    base_state["policy_id"] = (
        f"{POLICY_NAME}:{hashlib.sha256(canonical).hexdigest()[:16]}"
    )
    return base_state
