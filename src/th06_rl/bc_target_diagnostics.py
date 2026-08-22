"""Diagnose the L1d behavior-target and optimization contracts without Wine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from .actions import ACTION_NAMES
from .bc_features import FEATURE_NAMES, features_from_portable_root
from .bc_residual_diagnostics import _reactive_structure
from .bc_training import BehaviorDataset, _metrics
from .episode_dataset import iter_decision_epochs
from .mlp_bc_training import _mlp_probabilities


DIAGNOSIS_SCHEMA = "th06-rl-l1d-target-contract-diagnosis-v1"


@dataclass(frozen=True)
class PropensityDataset:
    episode_ids: tuple[str, ...]
    features: np.ndarray
    targets: np.ndarray
    legal_masks: np.ndarray
    baseline_targets: np.ndarray
    behavior_targets: np.ndarray
    episode_indices: np.ndarray
    inventory: tuple[dict[str, object], ...]
    policy_ids: tuple[tuple[str, int], ...]
    legal_size_counts: tuple[tuple[int, int], ...]
    maximum_declared_mixture_error: float

    @property
    def rows(self) -> int:
        return int(self.targets.shape[0])

    def behavior_view(self) -> BehaviorDataset:
        return BehaviorDataset(
            self.episode_ids,
            self.features,
            self.targets,
            self.legal_masks,
            self.baseline_targets,
            self.episode_indices,
            self.inventory,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_propensity_dataset(
    run_dirs: tuple[Path, ...],
    *,
    exploration_probability: float,
    max_rows: int = 2_000_000,
) -> PropensityDataset:
    """Load eligible rows while retaining their exact recorded distributions."""
    if not run_dirs:
        raise ValueError("propensity dataset requires complete episodes")
    if not 0.0 <= exploration_probability <= 1.0:
        raise ValueError("exploration probability must be in [0, 1]")
    action_index = {action: index for index, action in enumerate(ACTION_NAMES)}
    features = []
    targets = []
    legal_masks = []
    baseline_targets = []
    behavior_targets = []
    episode_indices = []
    episode_ids = []
    inventory = []
    policy_ids: Counter[str] = Counter()
    legal_size_counts: Counter[int] = Counter()
    maximum_mixture_error = 0.0

    for episode_index, unresolved in enumerate(run_dirs):
        run_dir = unresolved.resolve()
        episode_id = None
        episode_rows = 0
        for epoch in iter_decision_epochs(run_dir):
            if episode_id is None:
                episode_id = epoch.episode_id
            elif episode_id != epoch.episode_id:
                raise ValueError("one run directory exposed multiple episode identities")
            if not epoch.learning_eligible:
                continue
            if epoch.published_action is None or epoch.baseline_action is None:
                raise ValueError("eligible row lacks action or collector baseline")
            if len(targets) >= max_rows:
                raise ValueError("propensity dataset exceeds its in-memory limit")
            legal = tuple(epoch.observation.locally_admissible_actions)
            legal_set = frozenset(legal)
            if epoch.published_action not in legal_set or epoch.baseline_action not in legal_set:
                raise ValueError("recorded action or baseline is outside the shield mask")
            probability_map = dict(epoch.behavior_probabilities)
            if (
                len(probability_map) != len(epoch.behavior_probabilities)
                or set(probability_map) != set(legal)
            ):
                raise ValueError("recorded behavior distribution does not cover its mask")
            target = action_index[epoch.published_action]
            baseline = action_index[epoch.baseline_action]
            mask = np.asarray(
                [action in legal_set for action in ACTION_NAMES],
                dtype=np.bool_,
            )
            behavior = np.zeros(len(ACTION_NAMES), dtype=np.float64)
            for action, probability in probability_map.items():
                behavior[action_index[action]] = float(probability)
            if (
                np.any(~np.isfinite(behavior))
                or np.any(behavior < 0.0)
                or not math.isclose(float(np.sum(behavior)), 1.0, rel_tol=1e-9, abs_tol=1e-9)
                or not math.isclose(
                    float(behavior[target]),
                    float(epoch.behavior_probability),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("recorded behavior propensity is invalid")
            declared = np.zeros(len(ACTION_NAMES), dtype=np.float64)
            declared[mask] = exploration_probability / len(legal)
            declared[baseline] += 1.0 - exploration_probability
            maximum_mixture_error = max(
                maximum_mixture_error,
                float(np.max(np.abs(behavior - declared))),
            )
            features.append(features_from_portable_root(epoch.observation))
            targets.append(target)
            legal_masks.append(mask)
            baseline_targets.append(baseline)
            behavior_targets.append(behavior)
            episode_indices.append(episode_index)
            policy_ids[epoch.policy_id] += 1
            legal_size_counts[len(legal)] += 1
            episode_rows += 1
        if episode_id is None:
            raise ValueError(f"episode has no decision epochs: {run_dir.name}")
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode identity {episode_id}")
        episode_ids.append(episode_id)
        inventory.append({
            "episode_id": episode_id,
            "rows": episode_rows,
            "run_sha256": _sha256(run_dir / "run.json"),
            "manifest_sha256": _sha256(run_dir / "manifest.json"),
        })
    if not targets:
        raise ValueError("propensity dataset contains no eligible rows")
    return PropensityDataset(
        tuple(episode_ids),
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(legal_masks, dtype=np.bool_),
        np.asarray(baseline_targets, dtype=np.int64),
        np.asarray(behavior_targets, dtype=np.float64),
        np.asarray(episode_indices, dtype=np.int64),
        tuple(inventory),
        tuple(sorted(policy_ids.items())),
        tuple(sorted(legal_size_counts.items())),
        maximum_mixture_error,
    )


def _soft_top_label_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    predictions = np.argmax(probabilities, axis=1)
    rows = np.arange(probabilities.shape[0])
    confidence = probabilities[rows, predictions]
    target_probability = targets[rows, predictions]
    bin_indices = np.minimum(np.floor(confidence * bins).astype(np.int64), bins - 1)
    result = 0.0
    for bin_index in range(bins):
        members = bin_indices == bin_index
        if np.any(members):
            result += float(np.mean(members)) * abs(
                float(np.mean(confidence[members]))
                - float(np.mean(target_probability[members]))
            )
    return result


def _distribution_metrics(
    probabilities: np.ndarray,
    dataset: PropensityDataset,
) -> dict[str, float]:
    hard = _metrics(probabilities, dataset.behavior_view())
    rows = np.arange(dataset.rows)
    clipped = np.clip(probabilities, 1e-300, 1.0)
    target_clipped = np.clip(dataset.behavior_targets, 1e-300, 1.0)
    entropy = float(-np.mean(np.sum(
        dataset.behavior_targets * np.log(target_clipped), axis=1
    )))
    soft_cross_entropy = float(-np.mean(np.sum(
        dataset.behavior_targets * np.log(clipped), axis=1
    )))
    predictions = np.argmax(probabilities, axis=1)
    return {
        "hard_sample_nll": float(hard["negative_log_likelihood"]),
        "hard_sample_accuracy": float(hard["accuracy"]),
        "hard_sample_brier": float(hard["brier_score"]),
        "hard_sample_ece_10": float(hard["expected_calibration_error_10_bin"]),
        "target_entropy": entropy,
        "soft_cross_entropy": soft_cross_entropy,
        "kl_recorded_mu_to_model": soft_cross_entropy - entropy,
        "soft_brier_to_recorded_mu": float(np.mean(np.sum(
            (probabilities - dataset.behavior_targets) ** 2, axis=1
        ))),
        "soft_top_label_calibration_error_10": _soft_top_label_calibration_error(
            probabilities,
            dataset.behavior_targets,
        ),
        "argmax_baseline_agreement": float(np.mean(
            predictions == dataset.baseline_targets
        )),
        "mean_top_confidence": float(np.mean(
            probabilities[rows, predictions]
        )),
        "recorded_mu_probability_of_model_top_action": float(np.mean(
            dataset.behavior_targets[rows, predictions]
        )),
    }


def _model_arrays(state: dict[str, object]) -> tuple[np.ndarray, ...]:
    normalization = state.get("normalization")
    model = state.get("model")
    if not isinstance(normalization, dict) or not isinstance(model, dict):
        raise ValueError("L1d state lacks normalization or model arrays")
    arrays = (
        np.asarray(normalization.get("mean"), dtype=np.float64),
        np.asarray(normalization.get("scale"), dtype=np.float64),
        np.asarray(model.get("input_weights"), dtype=np.float64),
        np.asarray(model.get("hidden_biases"), dtype=np.float64),
        np.asarray(model.get("output_weights"), dtype=np.float64),
        np.asarray(model.get("output_biases"), dtype=np.float64),
    )
    mean, scale, input_weights, hidden_biases, output_weights, output_biases = arrays
    if (
        mean.shape != (len(FEATURE_NAMES),)
        or scale.shape != mean.shape
        or input_weights.shape != (32, len(FEATURE_NAMES))
        or hidden_biases.shape != (32,)
        or output_weights.shape != (len(ACTION_NAMES), 32)
        or output_biases.shape != (len(ACTION_NAMES),)
        or any(np.any(~np.isfinite(array)) for array in arrays)
        or np.any(scale <= 0.0)
    ):
        raise ValueError("L1d state arrays are malformed")
    return arrays


def _loss_and_gradient(
    features: np.ndarray,
    legal_masks: np.ndarray,
    target_probabilities: np.ndarray,
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    l2: float,
) -> tuple[np.ndarray, float, tuple[np.ndarray, ...], float]:
    input_weights, hidden_biases, output_weights, output_biases = parameters
    hidden_pre = features @ input_weights.T + hidden_biases
    hidden = np.maximum(0.0, hidden_pre)
    probabilities = _mlp_probabilities(
        features,
        legal_masks,
        input_weights,
        hidden_biases,
        output_weights,
        output_biases,
    )
    selected = np.clip(probabilities, 1e-300, 1.0)
    loss = float(
        -np.mean(np.sum(target_probabilities * np.log(selected), axis=1))
        + 0.5 * l2 * (
            np.sum(input_weights ** 2) + np.sum(output_weights ** 2)
        )
    )
    errors = np.where(legal_masks, probabilities - target_probabilities, 0.0)
    rows = features.shape[0]
    gradient_output_weights = errors.T @ hidden / rows + l2 * output_weights
    gradient_output_biases = np.mean(errors, axis=0)
    hidden_errors = (errors @ output_weights) * (hidden_pre > 0.0)
    gradient_input_weights = hidden_errors.T @ features / rows + l2 * input_weights
    gradient_hidden_biases = np.mean(hidden_errors, axis=0)
    gradients = (
        gradient_input_weights,
        gradient_hidden_biases,
        gradient_output_weights,
        gradient_output_biases,
    )
    norm = float(math.sqrt(sum(float(np.sum(value ** 2)) for value in gradients)))
    return probabilities, loss, gradients, norm


def _continue_branch(
    normalized_features: np.ndarray,
    dataset: PropensityDataset,
    initial_parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    objective_targets: np.ndarray,
    *,
    updates: int,
    checkpoints: tuple[int, ...],
    learning_rate: float,
    l2: float,
) -> dict[str, object]:
    parameters = tuple(value.copy() for value in initial_parameters)
    checkpoint_set = frozenset(checkpoints)
    records = []
    for update in range(updates + 1):
        probabilities, loss, gradients, gradient_l2 = _loss_and_gradient(
            normalized_features,
            dataset.legal_masks,
            objective_targets,
            parameters,
            l2=l2,
        )
        if update in checkpoint_set:
            records.append({
                "updates_after_frozen_l1d": update,
                "regularized_objective": loss,
                "objective_gradient_l2": gradient_l2,
                **_distribution_metrics(probabilities, dataset),
            })
        if update == updates:
            break
        parameters = tuple(
            value - learning_rate * gradient
            for value, gradient in zip(parameters, gradients, strict=True)
        )
    return {"checkpoints": records}


def diagnose_l1d_target_contract(
    train: PropensityDataset,
    validation: PropensityDataset,
    state: dict[str, object],
    *,
    exploration_probability: float,
    continuation_updates: int,
    continuation_checkpoints: tuple[int, ...],
    learning_rate: float,
    l2: float,
    mixture_tolerance: float,
    material_kl_reduction: float,
    material_soft_target_advantage: float,
) -> dict[str, object]:
    if set(train.episode_ids) & set(validation.episode_ids):
        raise ValueError("target diagnosis received episode leakage")
    if continuation_checkpoints[-1] != continuation_updates:
        raise ValueError("target diagnosis checkpoints omit the final update")
    mean, scale, input_weights, hidden_biases, output_weights, output_biases = (
        _model_arrays(state)
    )
    train_features = (train.features - mean) / scale
    validation_features = (validation.features - mean) / scale
    initial_parameters = (
        input_weights,
        hidden_biases,
        output_weights,
        output_biases,
    )
    initial_train = _mlp_probabilities(
        train_features,
        train.legal_masks,
        *initial_parameters,
    )
    initial_validation = _mlp_probabilities(
        validation_features,
        validation.legal_masks,
        *initial_parameters,
    )
    one_hot = np.zeros_like(train.behavior_targets)
    one_hot[np.arange(train.rows), train.targets] = 1.0
    _, _, hard_gradients, hard_gradient_l2 = _loss_and_gradient(
        train_features,
        train.legal_masks,
        one_hot,
        initial_parameters,
        l2=l2,
    )
    _, _, soft_gradients, soft_gradient_l2 = _loss_and_gradient(
        train_features,
        train.legal_masks,
        train.behavior_targets,
        initial_parameters,
        l2=l2,
    )
    hard_flat = np.concatenate([value.ravel() for value in hard_gradients])
    soft_flat = np.concatenate([value.ravel() for value in soft_gradients])
    cosine = float(np.dot(hard_flat, soft_flat) / (
        np.linalg.norm(hard_flat) * np.linalg.norm(soft_flat)
    ))
    hard_branch = _continue_branch(
        train_features,
        train,
        initial_parameters,
        one_hot,
        updates=continuation_updates,
        checkpoints=continuation_checkpoints,
        learning_rate=learning_rate,
        l2=l2,
    )
    soft_branch = _continue_branch(
        train_features,
        train,
        initial_parameters,
        train.behavior_targets,
        updates=continuation_updates,
        checkpoints=continuation_checkpoints,
        learning_rate=learning_rate,
        l2=l2,
    )
    initial_kl = float(hard_branch["checkpoints"][0]["kl_recorded_mu_to_model"])
    hard_final_kl = float(hard_branch["checkpoints"][-1]["kl_recorded_mu_to_model"])
    soft_final_kl = float(soft_branch["checkpoints"][-1]["kl_recorded_mu_to_model"])
    hard_reduction = initial_kl - hard_final_kl
    soft_advantage = hard_final_kl - soft_final_kl
    mixture_exact = bool(
        train.maximum_declared_mixture_error <= mixture_tolerance
        and validation.maximum_declared_mixture_error <= mixture_tolerance
    )
    early_stop_material = bool(hard_reduction >= material_kl_reduction)
    soft_target_material = bool(
        soft_advantage >= material_soft_target_advantage
    )
    if not mixture_exact:
        next_experiment = "stop-and-repair-propensity-pipeline"
    elif soft_target_material:
        next_experiment = "preregister-full-propensity-soft-target-bc"
    elif early_stop_material:
        next_experiment = "preregister-l1d-train-only-convergence-correction"
    else:
        next_experiment = "preregister-structured-current-observation-scorer"
    return {
        "schema": DIAGNOSIS_SCHEMA,
        "exploration_probability": exploration_probability,
        "corpus_propensity_audit": {
            "train_rows": train.rows,
            "validation_rows": validation.rows,
            "train_policy_ids": dict(train.policy_ids),
            "validation_policy_ids": dict(validation.policy_ids),
            "train_legal_size_counts": dict(train.legal_size_counts),
            "validation_legal_size_counts": dict(validation.legal_size_counts),
            "train_maximum_declared_mixture_error": train.maximum_declared_mixture_error,
            "validation_maximum_declared_mixture_error": validation.maximum_declared_mixture_error,
            "mixture_tolerance": mixture_tolerance,
            "exact": mixture_exact,
        },
        "frozen_l1d": {
            "train": _distribution_metrics(initial_train, train),
            "validation": _distribution_metrics(initial_validation, validation),
            "train_reactive_structure": _reactive_structure(
                initial_train,
                train.behavior_view(),
            ),
            "validation_reactive_structure": _reactive_structure(
                initial_validation,
                validation.behavior_view(),
            ),
        },
        "gradient_decomposition_at_frozen_l1d": {
            "hard_target_gradient_l2": hard_gradient_l2,
            "soft_target_gradient_l2": soft_gradient_l2,
            "hard_soft_gradient_cosine": cosine,
            "gradient_difference_l2": float(np.linalg.norm(hard_flat - soft_flat)),
        },
        "train_only_continuations": {
            "updates": continuation_updates,
            "checkpoints": list(continuation_checkpoints),
            "learning_rate": learning_rate,
            "l2": l2,
            "hard_published_action_target": hard_branch,
            "full_recorded_propensity_target": soft_branch,
            "hard_target_kl_reduction": hard_reduction,
            "soft_target_final_kl_advantage": soft_advantage,
            "validation_evaluated_for_continuations": False,
            "continuation_parameters_serialized": False,
        },
        "attribution": {
            "propensity_pipeline_exact": mixture_exact,
            "sampled_label_ece_is_sufficient_explanation": False,
            "premature_relative_gradient_stop_material": early_stop_material,
            "discarded_full_propensity_target_material": soft_target_material,
            "material_kl_reduction": material_kl_reduction,
            "material_soft_target_advantage": material_soft_target_advantage,
            "next_experiment": next_experiment,
        },
    }
