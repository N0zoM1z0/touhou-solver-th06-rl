"""Transparent masked linear behavior cloning over audited decision epochs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .actions import ACTION_NAMES
from .bc_features import FEATURE_NAMES, FEATURE_SCHEMA, features_from_portable_root
from .episode_dataset import (
    DECISION_EPOCH_SCHEMA as DATASET_DECISION_EPOCH_SCHEMA,
    iter_decision_epochs,
)
from .policies.linear_behavior_clone import (
    DECISION_EPOCH_SCHEMA,
    POLICY_NAME,
    STATE_SCHEMA,
    TARGET_SCHEMA,
    TRAINING_SCHEMA,
)
from .policy_api import POLICY_API_VERSION


if DECISION_EPOCH_SCHEMA != DATASET_DECISION_EPOCH_SCHEMA:
    raise RuntimeError("behavior-clone and dataset decision schemas disagree")


CALIBRATION_SCHEMA = "th06-rl-top-label-ece-10-equal-width-v2"


@dataclass(frozen=True)
class BehaviorDataset:
    episode_ids: tuple[str, ...]
    features: np.ndarray
    targets: np.ndarray
    legal_masks: np.ndarray
    baseline_targets: np.ndarray
    episode_indices: np.ndarray
    inventory: tuple[dict[str, object], ...]

    @property
    def rows(self) -> int:
        return int(self.targets.shape[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_behavior_dataset(
    run_dirs: tuple[Path, ...],
    *,
    max_rows: int = 2_000_000,
) -> BehaviorDataset:
    """Materialize only eligible factual decisions, failing rather than sampling."""
    if not run_dirs:
        raise ValueError("behavior dataset requires at least one complete episode")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    features = []
    targets = []
    legal_masks = []
    baseline_targets = []
    episode_indices = []
    episode_ids = []
    inventory = []
    action_index = {action: index for index, action in enumerate(ACTION_NAMES)}

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
            if epoch.published_action is None:
                raise ValueError("eligible decision lacks a behavior target")
            if len(targets) >= max_rows:
                raise ValueError(
                    f"behavior dataset exceeds explicit in-memory limit {max_rows}"
                )
            target = action_index[epoch.published_action]
            legal = frozenset(epoch.observation.locally_admissible_actions)
            if epoch.published_action not in legal:
                raise ValueError("behavior target is outside its shield mask")
            features.append(features_from_portable_root(epoch.observation))
            targets.append(target)
            legal_masks.append(tuple(action in legal for action in ACTION_NAMES))
            baseline_targets.append(
                action_index.get(epoch.baseline_action, -1)
            )
            episode_indices.append(episode_index)
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
        raise ValueError("behavior dataset contains no eligible decision epochs")
    return BehaviorDataset(
        tuple(episode_ids),
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(legal_masks, dtype=np.bool_),
        np.asarray(baseline_targets, dtype=np.int64),
        np.asarray(episode_indices, dtype=np.int64),
        tuple(inventory),
    )


def _masked_probabilities(
    features: np.ndarray,
    legal_masks: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    logits = features @ weights.T + biases
    logits = np.where(legal_masks, logits, -np.inf)
    maximum = np.max(logits, axis=1, keepdims=True)
    exponentials = np.where(legal_masks, np.exp(logits - maximum), 0.0)
    totals = exponentials.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("masked behavior model produced an invalid distribution")
    return exponentials / totals


def _expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Return equal-width top-label ECE with every boundary assigned once."""
    if bins <= 0:
        raise ValueError("calibration bin count must be positive")
    if confidence.ndim != 1 or correct.ndim != 1 or confidence.shape != correct.shape:
        raise ValueError("calibration inputs must be aligned one-dimensional arrays")
    if confidence.size == 0:
        raise ValueError("calibration inputs cannot be empty")
    if np.any(~np.isfinite(confidence)) or np.any(
        (confidence < 0.0) | (confidence > 1.0)
    ):
        raise ValueError("calibration confidence must be finite and in [0, 1]")
    bin_indices = np.minimum(
        np.floor(confidence * bins).astype(np.int64),
        bins - 1,
    )
    ece = 0.0
    for bin_index in range(bins):
        members = bin_indices == bin_index
        if np.any(members):
            ece += float(np.mean(members)) * abs(
                float(np.mean(confidence[members]))
                - float(np.mean(correct[members]))
            )
    return ece


def _metrics(
    probabilities: np.ndarray,
    dataset: BehaviorDataset,
) -> dict[str, float | None]:
    rows = np.arange(dataset.rows)
    selected = np.clip(probabilities[rows, dataset.targets], 1e-300, 1.0)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == dataset.targets
    one_hot = np.zeros_like(probabilities)
    one_hot[rows, dataset.targets] = 1.0
    confidence = probabilities[rows, predictions]
    ece = _expected_calibration_error(confidence, correct)
    baseline_known = dataset.baseline_targets >= 0
    return {
        "negative_log_likelihood": float(-np.mean(np.log(selected))),
        "accuracy": float(np.mean(correct)),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error_10_bin": ece,
        "reactive_action_accuracy": (
            float(np.mean(
                dataset.baseline_targets[baseline_known]
                == dataset.targets[baseline_known]
            ))
            if np.any(baseline_known)
            else None
        ),
    }


def _frequency_probabilities(
    train: BehaviorDataset,
    evaluation: BehaviorDataset,
) -> np.ndarray:
    counts: dict[tuple[bool, ...], Counter[int]] = defaultdict(Counter)
    global_counts: Counter[int] = Counter()
    for target, mask in zip(train.targets, train.legal_masks, strict=True):
        key = tuple(bool(value) for value in mask)
        counts[key][int(target)] += 1
        global_counts[int(target)] += 1
    probabilities = np.zeros(
        (evaluation.rows, len(ACTION_NAMES)),
        dtype=np.float64,
    )
    for row_index, mask in enumerate(evaluation.legal_masks):
        key = tuple(bool(value) for value in mask)
        legal = np.flatnonzero(mask)
        source = counts.get(key, global_counts)
        values = np.asarray(
            [float(source[int(action)]) + 1.0 for action in legal],
            dtype=np.float64,
        )
        probabilities[row_index, legal] = values / values.sum()
    return probabilities


def _episode_nll(
    probabilities: np.ndarray,
    dataset: BehaviorDataset,
) -> np.ndarray:
    selected = np.clip(
        probabilities[np.arange(dataset.rows), dataset.targets],
        1e-300,
        1.0,
    )
    losses = -np.log(selected)
    return np.asarray([
        float(np.mean(losses[dataset.episode_indices == index]))
        for index in range(len(dataset.episode_ids))
    ])


def _bootstrap_delta_interval(
    model_probabilities: np.ndarray,
    frequency_probabilities: np.ndarray,
    validation: BehaviorDataset,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    deltas = _episode_nll(model_probabilities, validation) - _episode_nll(
        frequency_probabilities,
        validation,
    )
    random = np.random.default_rng(seed)
    draws = random.choice(
        deltas,
        size=(samples, len(deltas)),
        replace=True,
    ).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def fit_behavior_clone(
    train_runs: tuple[Path, ...],
    validation_runs: tuple[Path, ...],
    *,
    epochs: int = 100,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    seed: int = 0,
    bootstrap_samples: int = 2_000,
    calibration_tolerance: float = 0.02,
    minimum_updates: int = 0,
    relative_gradient_l2_tolerance: float | None = None,
    max_rows: int = 2_000_000,
    code_commit: str = "fixture",
    policy_plugin_sha256: str = "fixture",
) -> dict[str, object]:
    """Fit one deterministic masked softmax baseline and return its state."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be finite and nonnegative")
    if not math.isfinite(calibration_tolerance) or calibration_tolerance < 0.0:
        raise ValueError("calibration tolerance must be finite and nonnegative")
    if not 0 <= minimum_updates <= epochs:
        raise ValueError("minimum updates must be between zero and maximum epochs")
    if relative_gradient_l2_tolerance is not None and (
        not math.isfinite(relative_gradient_l2_tolerance)
        or not 0.0 <= relative_gradient_l2_tolerance < 1.0
    ):
        raise ValueError(
            "relative gradient L2 tolerance must be finite and in [0, 1)"
        )
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
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
    weights = np.zeros((len(ACTION_NAMES), len(FEATURE_NAMES)), dtype=np.float64)
    biases = np.zeros(len(ACTION_NAMES), dtype=np.float64)
    first_loss = None

    def training_step_values() -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
        probabilities = _masked_probabilities(
            train_features,
            train.legal_masks,
            weights,
            biases,
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
        gradient_weights = errors.T @ train_features / train.rows + l2 * weights
        gradient_biases = np.mean(errors, axis=0)
        gradient_l2 = float(math.sqrt(
            float(np.sum(gradient_weights ** 2))
            + float(np.sum(gradient_biases ** 2))
        ))
        return probabilities, loss, gradient_weights, gradient_biases, gradient_l2

    _, first_loss, gradient_weights, gradient_biases, initial_gradient_l2 = (
        training_step_values()
    )
    gradient_l2 = initial_gradient_l2
    updates_completed = 0
    optimization_converged = False
    while updates_completed < epochs:
        if (
            relative_gradient_l2_tolerance is not None
            and updates_completed >= minimum_updates
            and gradient_l2
            <= relative_gradient_l2_tolerance * initial_gradient_l2
        ):
            optimization_converged = True
            break
        weights -= learning_rate * gradient_weights
        biases -= learning_rate * gradient_biases
        updates_completed += 1
        _, _, gradient_weights, gradient_biases, gradient_l2 = training_step_values()

    if relative_gradient_l2_tolerance is not None and (
        updates_completed >= minimum_updates
        and gradient_l2 <= relative_gradient_l2_tolerance * initial_gradient_l2
    ):
        optimization_converged = True
    final_gradient_l2 = gradient_l2
    gradient_ratio = (
        final_gradient_l2 / initial_gradient_l2
        if initial_gradient_l2 > 0.0
        else 0.0
    )

    train_probabilities = _masked_probabilities(
        train_features,
        train.legal_masks,
        weights,
        biases,
    )
    validation_probabilities = _masked_probabilities(
        validation_features,
        validation.legal_masks,
        weights,
        biases,
    )
    frequency_probabilities = _frequency_probabilities(train, validation)
    train_metrics = _metrics(train_probabilities, train)
    validation_metrics = _metrics(validation_probabilities, validation)
    frequency_metrics = _metrics(frequency_probabilities, validation)
    interval = _bootstrap_delta_interval(
        validation_probabilities,
        frequency_probabilities,
        validation,
        seed=seed,
        samples=bootstrap_samples,
    )
    proper_score_passed = bool(
        len(validation.episode_ids) >= 2
        and interval[1] < 0.0
    )
    calibration_passed = bool(
        validation_metrics["expected_calibration_error_10_bin"]
        <= frequency_metrics["expected_calibration_error_10_bin"]
        + calibration_tolerance
    )
    optimization_passed = bool(
        relative_gradient_l2_tolerance is None or optimization_converged
    )
    gate_passed = bool(
        proper_score_passed and calibration_passed and optimization_passed
    )
    action_major_weights = tuple(
        tuple(float(value) for value in row) for row in weights
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
        },
        "normalization": {
            "mean": [float(value) for value in mean],
            "scale": [float(value) for value in scale],
        },
        "model": {
            "kind": "masked-linear-softmax",
            "weights": [list(row) for row in action_major_weights],
            "biases": [float(value) for value in biases],
        },
        "sampling": {
            "kind": "seeded-categorical",
            "seed": seed,
        },
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
                "converged": (
                    optimization_converged
                    if relative_gradient_l2_tolerance is not None
                    else None
                ),
                "stop_reason": (
                    "relative-gradient-l2"
                    if optimization_converged
                    else "maximum-updates"
                ),
                "initial_gradient_l2": initial_gradient_l2,
                "final_gradient_l2": final_gradient_l2,
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
            "action_frequency_validation": frequency_metrics,
            "episode_bootstrap_nll_delta_95": list(interval),
            "proper_score_gate_passed": proper_score_passed,
            "calibration_gate_passed": calibration_passed,
            "optimization_gate_passed": optimization_passed,
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
    base_state["policy_id"] = f"{POLICY_NAME}:{hashlib.sha256(canonical).hexdigest()[:16]}"
    return base_state
