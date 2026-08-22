"""Fit one transparent shared per-action scorer on the frozen L1 corpus."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .actions import ACTION_NAMES
from .bc_features import FEATURE_NAMES, FEATURE_SCHEMA
from .bc_residual_diagnostics import _reactive_structure
from .bc_target_diagnostics import (
    PropensityDataset,
    _distribution_metrics,
    load_propensity_dataset,
)
from .core.model import movement_actions
from .mlp_bc_training import _mlp_probabilities
from .policies.shared_action_behavior_clone import (
    DECISION_EPOCH_SCHEMA,
    POLICY_NAME,
    STATE_SCHEMA,
    TARGET_SCHEMA,
    TRAINING_SCHEMA,
)
from .policy_api import POLICY_API_VERSION
from .shared_action_features import ACTION_FEATURE_NAMES, ACTION_FEATURE_SCHEMA


MODEL_KIND = "masked-shared-linear-action-softmax"
OPTIMIZER_KIND = "full-batch-gradient-descent"
INITIALIZATION_KIND = "all-zero-shared-linear"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_feature_tensor(dataset: PropensityDataset) -> np.ndarray:
    """Vectorize the exact online per-action projection for offline fitting."""
    index = {name: offset for offset, name in enumerate(FEATURE_NAMES)}
    known = dataset.features[:, [
        index[f"shield_clearance_known:{name}"] for name in ACTION_NAMES
    ]]
    shield_legal = dataset.features[:, [
        index[f"shield_legal:{name}"] for name in ACTION_NAMES
    ]]
    clearance = dataset.features[:, [
        index[f"shield_clearance:{name}"] for name in ACTION_NAMES
    ]]
    final_x = dataset.features[:, [
        index[f"shield_final_x:{name}"] for name in ACTION_NAMES
    ]]
    final_y = dataset.features[:, [
        index[f"shield_final_y:{name}"] for name in ACTION_NAMES
    ]]
    current = dataset.features[:, [
        index[f"current_action:{name}"] for name in ACTION_NAMES
    ]]
    if (
        np.any((known != 0.0) & (known != 1.0))
        or np.any((shield_legal != 0.0) & (shield_legal != 1.0))
        or not np.array_equal(shield_legal.astype(np.bool_), dataset.legal_masks)
    ):
        raise ValueError("shared-action vectorization found malformed shield bits")
    boundary = np.minimum.reduce((
        final_x - 8.0,
        376.0 - final_x,
        final_y - 16.0,
        432.0 - final_y,
    ))
    actions = movement_actions()
    if tuple(action.name for action in actions) != ACTION_NAMES:
        raise ValueError("shared-action movement vocabulary differs")
    stationary = np.asarray([
        float(action.dx == 0 and action.dy == 0) for action in actions
    ])
    focused = np.asarray([float(action.focused) for action in actions])
    lexical_priority = {
        name: rank for rank, name in enumerate(sorted(ACTION_NAMES))
    }
    lexical = np.asarray([
        float(lexical_priority[name]) for name in ACTION_NAMES
    ])
    rows = dataset.rows
    tensor = np.stack((
        1.0 - known,
        clearance,
        boundary,
        current,
        np.broadcast_to(stationary, (rows, len(ACTION_NAMES))),
        np.broadcast_to(focused, (rows, len(ACTION_NAMES))),
        np.broadcast_to(lexical, (rows, len(ACTION_NAMES))),
    ), axis=2)
    if tensor.shape != (rows, len(ACTION_NAMES), len(ACTION_FEATURE_NAMES)):
        raise ValueError("shared-action tensor dimensions differ")
    if np.any(~np.isfinite(tensor)):
        raise ValueError("shared-action tensor is not finite")
    return tensor


def _shared_probabilities(
    features: np.ndarray,
    legal_masks: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if (
        features.ndim != 3
        or features.shape[:2] != legal_masks.shape
        or features.shape[2] != len(ACTION_FEATURE_NAMES)
        or weights.shape != (len(ACTION_FEATURE_NAMES),)
    ):
        raise ValueError("shared-action probability dimensions differ")
    logits = np.einsum("raf,f->ra", features, weights, optimize=True)
    logits = np.where(legal_masks, logits, -np.inf)
    maximum = np.max(logits, axis=1, keepdims=True)
    exponentials = np.where(legal_masks, np.exp(logits - maximum), 0.0)
    totals = exponentials.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("shared-action model produced an invalid distribution")
    return exponentials / totals


def _load_l1d_probabilities(
    path: Path,
    dataset: PropensityDataset,
) -> tuple[np.ndarray, str]:
    path = path.resolve()
    state = json.loads(path.read_text(encoding="utf-8"))
    normalization = state.get("normalization")
    model = state.get("model")
    if (
        not isinstance(state, dict)
        or state.get("schema") != "th06-rl-small-mlp-behavior-clone-state-v1"
        or state.get("feature_schema") != FEATURE_SCHEMA
        or tuple(state.get("feature_names", ())) != FEATURE_NAMES
        or tuple(state.get("action_names", ())) != ACTION_NAMES
        or not isinstance(normalization, dict)
        or not isinstance(model, dict)
    ):
        raise ValueError("frozen L1d comparator contract differs")
    try:
        mean = np.asarray(normalization["mean"], dtype=np.float64)
        scale = np.asarray(normalization["scale"], dtype=np.float64)
        input_weights = np.asarray(model["input_weights"], dtype=np.float64)
        hidden_biases = np.asarray(model["hidden_biases"], dtype=np.float64)
        output_weights = np.asarray(model["output_weights"], dtype=np.float64)
        output_biases = np.asarray(model["output_biases"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen L1d comparator numeric state is malformed") from error
    if (
        mean.shape != (len(FEATURE_NAMES),)
        or scale.shape != mean.shape
        or input_weights.shape[1] != len(FEATURE_NAMES)
        or hidden_biases.shape != (input_weights.shape[0],)
        or output_weights.shape != (len(ACTION_NAMES), input_weights.shape[0])
        or output_biases.shape != (len(ACTION_NAMES),)
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ValueError("frozen L1d comparator numeric dimensions differ")
    probabilities = _mlp_probabilities(
        (dataset.features - mean) / scale,
        dataset.legal_masks,
        input_weights,
        hidden_biases,
        output_weights,
        output_biases,
    )
    return probabilities, _sha256(path)


def _episode_soft_cross_entropy(
    probabilities: np.ndarray,
    dataset: PropensityDataset,
) -> np.ndarray:
    losses = -np.sum(
        dataset.behavior_targets * np.log(np.clip(probabilities, 1e-300, 1.0)),
        axis=1,
    )
    return np.asarray([
        float(np.mean(losses[dataset.episode_indices == episode]))
        for episode in range(len(dataset.episode_ids))
    ])


def _bootstrap_soft_delta_interval(
    candidate: np.ndarray,
    comparator: np.ndarray,
    dataset: PropensityDataset,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    if samples <= 0 or len(dataset.episode_ids) < 2:
        raise ValueError("soft bootstrap needs positive draws and two episodes")
    deltas = _episode_soft_cross_entropy(candidate, dataset) - (
        _episode_soft_cross_entropy(comparator, dataset)
    )
    random = np.random.default_rng(seed)
    draws = random.choice(
        deltas,
        size=(samples, len(deltas)),
        replace=True,
    ).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def _pooled_final_tie_agreement(structure: dict[str, object]) -> float:
    stages = structure.get("decision_stage")
    if not isinstance(stages, dict):
        raise ValueError("reactive structure lacks decision stages")
    selected = [stages.get("focused"), stages.get("lexicographic")]
    if any(not isinstance(stage, dict) for stage in selected):
        raise ValueError("reactive structure lacks final tie stages")
    rows = sum(int(stage["rows"]) for stage in selected)
    matches = sum(
        int(stage["rows"]) * float(stage["model_reactive_agreement"])
        for stage in selected
    )
    if rows <= 0:
        raise ValueError("reactive structure has no final tie rows")
    return matches / rows


def fit_shared_action_behavior_clone(
    train_runs: tuple[Path, ...],
    validation_runs: tuple[Path, ...],
    *,
    l1d_comparator_state: Path,
    epochs: int = 10_000,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    seed: int = 0,
    bootstrap_samples: int = 2_000,
    minimum_updates: int = 100,
    relative_gradient_l2_tolerance: float = 0.01,
    exploration_probability: float = 0.2,
    maximum_validation_kl: float = 0.1,
    minimum_reactive_agreement: float = 0.95,
    minimum_final_tie_agreement: float = 0.5,
    max_rows: int = 2_000_000,
    code_commit: str = "fixture",
    policy_plugin_sha256: str = "fixture",
) -> dict[str, object]:
    """Fit the one preregistered seven-parameter structured BC ablation."""
    if epochs <= 0 or not 0 <= minimum_updates <= epochs:
        raise ValueError("shared-action update bounds are invalid")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be finite and nonnegative")
    if not 0.0 < relative_gradient_l2_tolerance < 1.0:
        raise ValueError("relative gradient tolerance must be in (0, 1)")
    if not 0 <= seed < 2**64 or bootstrap_samples <= 0:
        raise ValueError("seed or bootstrap sample count is invalid")
    if not 0.0 <= exploration_probability <= 1.0:
        raise ValueError("exploration probability must be in [0, 1]")
    if maximum_validation_kl <= 0.0:
        raise ValueError("maximum validation KL must be positive")
    if not 0.0 <= minimum_reactive_agreement <= 1.0:
        raise ValueError("minimum reactive agreement must be in [0, 1]")
    if not 0.0 <= minimum_final_tie_agreement <= 1.0:
        raise ValueError("minimum final-tie agreement must be in [0, 1]")
    if not code_commit or not policy_plugin_sha256:
        raise ValueError("fit provenance cannot be empty")

    train = load_propensity_dataset(
        train_runs,
        exploration_probability=exploration_probability,
        max_rows=max_rows,
    )
    validation = load_propensity_dataset(
        validation_runs,
        exploration_probability=exploration_probability,
        max_rows=max_rows,
    )
    overlap = set(train.episode_ids) & set(validation.episode_ids)
    if overlap:
        raise ValueError(f"train/validation episode leakage: {sorted(overlap)}")
    if train.maximum_declared_mixture_error != 0.0 or (
        validation.maximum_declared_mixture_error != 0.0
    ):
        raise ValueError("shared-action fit requires exact recorded propensities")

    train_raw = _action_feature_tensor(train)
    validation_raw = _action_feature_tensor(validation)
    train_legal_features = train_raw[train.legal_masks]
    mean = np.mean(train_legal_features, axis=0)
    scale = np.std(train_legal_features, axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    train_features = (train_raw - mean) / scale
    validation_features = (validation_raw - mean) / scale
    weights = np.zeros(len(ACTION_FEATURE_NAMES), dtype=np.float64)

    def training_step_values() -> tuple[float, np.ndarray, float]:
        probabilities = _shared_probabilities(
            train_features,
            train.legal_masks,
            weights,
        )
        selected = np.clip(
            probabilities[np.arange(train.rows), train.targets],
            1e-300,
            1.0,
        )
        loss = float(-np.mean(np.log(selected)) + 0.5 * l2 * np.sum(weights ** 2))
        errors = probabilities.copy()
        errors[np.arange(train.rows), train.targets] -= 1.0
        errors = np.where(train.legal_masks, errors, 0.0)
        gradient = (
            np.einsum("ra,raf->f", errors, train_features, optimize=True)
            / train.rows
            + l2 * weights
        )
        gradient_l2 = float(np.linalg.norm(gradient))
        return loss, gradient, gradient_l2

    first_loss, gradient, initial_gradient_l2 = training_step_values()
    gradient_l2 = initial_gradient_l2
    updates_completed = 0
    optimization_converged = False
    while updates_completed < epochs:
        if (
            updates_completed >= minimum_updates
            and gradient_l2 <= relative_gradient_l2_tolerance * initial_gradient_l2
        ):
            optimization_converged = True
            break
        weights -= learning_rate * gradient
        updates_completed += 1
        _, gradient, gradient_l2 = training_step_values()
    if (
        updates_completed >= minimum_updates
        and gradient_l2 <= relative_gradient_l2_tolerance * initial_gradient_l2
    ):
        optimization_converged = True

    train_probabilities = _shared_probabilities(
        train_features, train.legal_masks, weights
    )
    validation_probabilities = _shared_probabilities(
        validation_features, validation.legal_masks, weights
    )
    l1d_probabilities, l1d_sha256 = _load_l1d_probabilities(
        l1d_comparator_state, validation
    )
    train_metrics = _distribution_metrics(train_probabilities, train)
    validation_metrics = _distribution_metrics(validation_probabilities, validation)
    l1d_metrics = _distribution_metrics(l1d_probabilities, validation)
    train_structure = _reactive_structure(train_probabilities, train.behavior_view())
    validation_structure = _reactive_structure(
        validation_probabilities, validation.behavior_view()
    )
    l1d_structure = _reactive_structure(
        l1d_probabilities, validation.behavior_view()
    )
    interval = _bootstrap_soft_delta_interval(
        validation_probabilities,
        l1d_probabilities,
        validation,
        seed=seed,
        samples=bootstrap_samples,
    )
    proper_score_passed = interval[1] < 0.0
    absolute_kl_passed = (
        validation_metrics["kl_recorded_mu_to_model"] <= maximum_validation_kl
    )
    reactive_agreement_passed = (
        validation_structure["model_reactive_agreement"]
        >= minimum_reactive_agreement
    )
    final_tie_agreement = _pooled_final_tie_agreement(validation_structure)
    final_tie_passed = final_tie_agreement >= minimum_final_tie_agreement
    gate_passed = bool(
        optimization_converged
        and proper_score_passed
        and absolute_kl_passed
        and reactive_agreement_passed
        and final_tie_passed
    )

    base_state: dict[str, object] = {
        "schema": STATE_SCHEMA,
        "training_schema": TRAINING_SCHEMA,
        "decision_epoch_schema": DECISION_EPOCH_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "action_feature_schema": ACTION_FEATURE_SCHEMA,
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "policy_api_version": POLICY_API_VERSION,
        "provenance": {
            "code_commit": code_commit,
            "policy_plugin_sha256": policy_plugin_sha256,
            "frozen_l1d_comparator_sha256": l1d_sha256,
        },
        "normalization": {
            "population": "train legal action rows only",
            "mean": [float(value) for value in mean],
            "scale": [float(value) for value in scale],
        },
        "initialization": {"kind": INITIALIZATION_KIND},
        "model": {
            "kind": MODEL_KIND,
            "weights": [float(value) for value in weights],
        },
        "sampling": {"kind": "seeded-categorical", "seed": seed},
        "fit": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "exploration_probability": exploration_probability,
            "initial_regularized_nll": first_loss,
            "optimization": {
                "kind": OPTIMIZER_KIND,
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
                "final_to_initial_gradient_l2_ratio": (
                    gradient_l2 / initial_gradient_l2
                    if initial_gradient_l2 > 0.0 else 0.0
                ),
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
            "reactive_structure_train": train_structure,
            "reactive_structure_validation": validation_structure,
            "frozen_l1d_validation": l1d_metrics,
            "frozen_l1d_reactive_structure_validation": l1d_structure,
            "episode_bootstrap_soft_cross_entropy_delta_vs_l1d_95": list(interval),
            "maximum_validation_kl": maximum_validation_kl,
            "minimum_reactive_agreement": minimum_reactive_agreement,
            "minimum_final_tie_agreement": minimum_final_tie_agreement,
            "validation_final_tie_agreement": final_tie_agreement,
            "optimization_gate_passed": optimization_converged,
            "proper_score_gate_passed": proper_score_passed,
            "absolute_kl_gate_passed": absolute_kl_passed,
            "reactive_agreement_gate_passed": reactive_agreement_passed,
            "final_tie_gate_passed": final_tie_passed,
            "learnability_gate_passed": gate_passed,
        },
        "propensity_audit": {
            "train_maximum_declared_mixture_error": (
                train.maximum_declared_mixture_error
            ),
            "validation_maximum_declared_mixture_error": (
                validation.maximum_declared_mixture_error
            ),
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
