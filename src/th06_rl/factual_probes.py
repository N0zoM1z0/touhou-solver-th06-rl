"""Transparent factual probes over executed original-Wine transitions.

These probes are diagnostics, not rewards or policy-improvement models.  They
use only portable current-root facts, the action actually published at that
root, and factual successors from the immutable episode.  No counterfactual
successor is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .actions import ACTION_NAMES
from .core.model import movement_actions
from .episode_dataset import (
    EpisodeDatasetError,
    PortableDecisionRoot,
    iter_decision_epochs,
    iter_episode_transitions,
)


PROBE_FEATURE_SCHEMA = "th06-rl-current-root-action-probe-features-v1"
PROBE_FEATURE_NAMES = (
    "player_x",
    "player_y",
    "power",
    "log1p_bullet_count",
    "log1p_laser_count",
    "shield_action_count",
    "action_clearance_unknown",
    "action_clearance",
    "action_final_x",
    "action_final_y",
    "action_boundary_reserve",
    "action_is_current",
    "action_dx",
    "action_dy",
    "action_is_focused",
)
INVALID_HORIZON_REASONS = frozenset({
    "observation-gap",
    "bomb",
    "infrastructure-failure",
})

_ACTIONS = {action.name: action for action in movement_actions()}
if tuple(_ACTIONS) != ACTION_NAMES:
    raise RuntimeError("probe movement vocabulary differs from canonical actions")


@dataclass(frozen=True)
class DynamicsDataset:
    episode_ids: tuple[str, ...]
    episode_indices: np.ndarray
    current_actions: tuple[str, ...]
    published_actions: tuple[str, ...]
    executed_actions: tuple[str, ...]
    deltas: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.deltas.shape[0])


@dataclass(frozen=True)
class HorizonDataset:
    horizon: int
    episode_ids: tuple[str, ...]
    episode_indices: np.ndarray
    features: np.ndarray
    hit_labels: np.ndarray
    shield_collapse_labels: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class FactualProbeDataset:
    episode_ids: tuple[str, ...]
    inventory: tuple[dict[str, object], ...]
    dynamics: DynamicsDataset
    horizons: tuple[HorizonDataset, ...]


@dataclass(frozen=True)
class _TransitionFact:
    sequence: int
    elapsed_frames: int
    executed_action: str | None
    delta_x: float
    delta_y: float
    hit: bool
    shield_count_after: int
    control_dead_end: bool
    valid_for_horizon: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_conditioned_probe_features(
    root: PortableDecisionRoot,
    action_name: str,
) -> tuple[float, ...]:
    """Project one current root and one factual published action."""
    if action_name not in root.locally_admissible_actions:
        raise ValueError("probe action is outside the observed shield set")
    evaluations = {
        name: (clearance, final_x, final_y)
        for name, clearance, final_x, final_y in root.shield_action_evaluations
    }
    if len(evaluations) != len(root.shield_action_evaluations):
        raise ValueError("probe shield evaluations contain duplicate actions")
    try:
        clearance, final_x, final_y = evaluations[action_name]
        action = _ACTIONS[action_name]
    except KeyError as error:
        raise ValueError("probe action lacks a shield evaluation") from error
    boundary_reserve = min(
        final_x - 8.0,
        376.0 - final_x,
        final_y - 16.0,
        432.0 - final_y,
    )
    values = (
        float(root.player_x),
        float(root.player_y),
        float(root.power),
        math.log1p(root.bullet_count),
        math.log1p(root.laser_count),
        float(len(root.locally_admissible_actions)),
        float(clearance is None),
        0.0 if clearance is None else float(clearance),
        float(final_x),
        float(final_y),
        float(boundary_reserve),
        float(action_name == root.current_action),
        float(action.dx),
        float(action.dy),
        float(action.focused),
    )
    if len(values) != len(PROBE_FEATURE_NAMES) or any(
        not math.isfinite(value) for value in values
    ):
        raise ValueError("probe feature vector is non-finite or malformed")
    return values


def _transition_fact(transition) -> _TransitionFact:
    outcome = transition.outcome
    try:
        elapsed = int(outcome["elapsed_frames"])
        before_x = float(outcome["player_x_before"])
        before_y = float(outcome["player_y_before"])
        after_x = float(outcome["player_x_after"])
        after_y = float(outcome["player_y_after"])
        shield_after = int(outcome["shield_count_after"])
    except (KeyError, TypeError, ValueError) as error:
        raise EpisodeDatasetError("probe transition outcome is malformed") from error
    if (
        shield_after < 0
        or any(not math.isfinite(value) for value in (
            before_x, before_y, after_x, after_y,
        ))
    ):
        raise EpisodeDatasetError("probe transition geometry is invalid")
    reasons = frozenset(transition.learning_exclusion_reasons)
    return _TransitionFact(
        sequence=int(transition.sequence),
        elapsed_frames=elapsed,
        executed_action=transition.executed_action,
        delta_x=after_x - before_x,
        delta_y=after_y - before_y,
        hit=bool(outcome.get("life_lost")),
        shield_count_after=shield_after,
        control_dead_end=bool(outcome.get("control_dead_end")),
        valid_for_horizon=(
            elapsed == 1 and not reasons.intersection(INVALID_HORIZON_REASONS)
        ),
    )


def _load_episode(
    run_dir: Path,
    *,
    episode_index: int,
    horizons: tuple[int, ...],
) -> tuple[
    str,
    dict[str, object],
    list[tuple[int, str, str, str, float, float]],
    dict[int, list[tuple[int, tuple[float, ...], bool, bool]]],
]:
    # Decision epochs establish complete frame/transition linkage and exact HIT
    # conservation.  Raw one-frame transitions are then read for fixed physical
    # horizons; this deliberate second validated pass is prototype debt, not a
    # derived-data cache or a relaxed evidence path.
    epochs = tuple(iter_decision_epochs(run_dir))
    facts = tuple(_transition_fact(row) for row in iter_episode_transitions(run_dir))
    if not epochs or not facts or any(
        fact.sequence != index for index, fact in enumerate(facts)
    ):
        raise EpisodeDatasetError("probe episode is empty or non-contiguous")
    episode_id = epochs[0].episode_id
    if any(epoch.episode_id != episode_id for epoch in epochs):
        raise EpisodeDatasetError("probe run exposed multiple episode identities")

    dynamics: list[tuple[int, str, str, str, float, float]] = []
    horizon_rows = {horizon: [] for horizon in horizons}
    for epoch in epochs:
        if not epoch.learning_eligible or epoch.published_action is None:
            continue
        start = epoch.start_sequence
        if not 0 <= start < len(facts):
            raise EpisodeDatasetError("decision start falls outside transition stream")
        first = facts[start]
        if first.valid_for_horizon and first.executed_action is not None:
            dynamics.append((
                episode_index,
                epoch.observation.current_action,
                epoch.published_action,
                first.executed_action,
                first.delta_x,
                first.delta_y,
            ))

        features = action_conditioned_probe_features(
            epoch.observation,
            epoch.published_action,
        )
        initial_shield_count = len(epoch.observation.locally_admissible_actions)
        for horizon in horizons:
            stop = start + horizon
            if stop > len(facts):
                continue
            window = facts[start:stop]
            if any(not fact.valid_for_horizon for fact in window):
                continue
            # "Collapse" is frozen as a strict contraction of the nonempty
            # observed admissible set at a subsequent root, or an explicit
            # observed-shield control dead end.  Zero counts on HIT/passive
            # roots alone do not manufacture a collapse label.
            collapsed = any(
                fact.control_dead_end
                or 0 < fact.shield_count_after < initial_shield_count
                for fact in window
            )
            horizon_rows[horizon].append((
                episode_index,
                features,
                any(fact.hit for fact in window),
                collapsed,
            ))

    return (
        episode_id,
        {
            "episode_id": episode_id,
            "run_sha256": _sha256(run_dir / "run.json"),
            "manifest_sha256": _sha256(run_dir / "manifest.json"),
            "transitions": len(facts),
            "decision_epochs": len(epochs),
            "eligible_decision_epochs": sum(
                int(epoch.learning_eligible) for epoch in epochs
            ),
        },
        dynamics,
        horizon_rows,
    )


def load_factual_probe_dataset(
    run_dirs: Iterable[Path],
    *,
    horizons: tuple[int, ...],
    max_rows: int = 2_000_000,
) -> FactualProbeDataset:
    """Load complete episodes into fixed factual dynamics and risk views."""
    paths = tuple(Path(path).resolve() for path in run_dirs)
    if not paths:
        raise ValueError("factual probes require at least one complete episode")
    if (
        not horizons
        or tuple(sorted(set(horizons))) != horizons
        or any(not isinstance(horizon, int) or horizon <= 0 for horizon in horizons)
    ):
        raise ValueError("probe horizons must be unique increasing positive integers")

    episode_ids = []
    inventory = []
    dynamics_rows = []
    horizon_rows: dict[int, list[tuple[int, tuple[float, ...], bool, bool]]] = {
        horizon: [] for horizon in horizons
    }
    for episode_index, run_dir in enumerate(paths):
        episode_id, episode_inventory, episode_dynamics, episode_horizons = (
            _load_episode(
                run_dir,
                episode_index=episode_index,
                horizons=horizons,
            )
        )
        if episode_id in episode_ids:
            raise ValueError(f"duplicate probe episode identity {episode_id}")
        episode_ids.append(episode_id)
        inventory.append(episode_inventory)
        dynamics_rows.extend(episode_dynamics)
        for horizon in horizons:
            horizon_rows[horizon].extend(episode_horizons[horizon])
            if len(horizon_rows[horizon]) > max_rows:
                raise ValueError("factual horizon dataset exceeds its row limit")
    if not dynamics_rows or any(not horizon_rows[horizon] for horizon in horizons):
        raise ValueError("factual probe dataset contains an empty view")

    dynamics = DynamicsDataset(
        tuple(episode_ids),
        np.asarray([row[0] for row in dynamics_rows], dtype=np.int64),
        tuple(row[1] for row in dynamics_rows),
        tuple(row[2] for row in dynamics_rows),
        tuple(row[3] for row in dynamics_rows),
        np.asarray([(row[4], row[5]) for row in dynamics_rows], dtype=np.float64),
    )
    horizon_datasets = []
    for horizon in horizons:
        rows = horizon_rows[horizon]
        horizon_datasets.append(HorizonDataset(
            horizon,
            tuple(episode_ids),
            np.asarray([row[0] for row in rows], dtype=np.int64),
            np.asarray([row[1] for row in rows], dtype=np.float64),
            np.asarray([row[2] for row in rows], dtype=np.bool_),
            np.asarray([row[3] for row in rows], dtype=np.bool_),
        ))
    return FactualProbeDataset(
        tuple(episode_ids),
        tuple(inventory),
        dynamics,
        tuple(horizon_datasets),
    )


def _action_means(
    actions: tuple[str, ...],
    deltas: np.ndarray,
) -> dict[str, list[float]]:
    global_mean = np.mean(deltas, axis=0)
    result = {"__global__": global_mean.tolist()}
    action_array = np.asarray(actions, dtype=object)
    for action in ACTION_NAMES:
        members = action_array == action
        if np.any(members):
            result[action] = np.mean(deltas[members], axis=0).tolist()
    return result


def _dynamics_predictions(
    means: dict[str, list[float]],
    actions: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    fallback = means["__global__"]
    unseen = sum(action not in means for action in actions)
    return np.asarray(
        [means.get(action, fallback) for action in actions],
        dtype=np.float64,
    ), unseen


def _joint_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.sum((predictions - targets) ** 2, axis=1)))


def fit_factual_probe_models(
    dataset: FactualProbeDataset,
    *,
    ridge_l2: float,
) -> dict[str, object]:
    """Fit table-mean dynamics and ridge linear-probability risk probes."""
    if not math.isfinite(ridge_l2) or ridge_l2 <= 0.0:
        raise ValueError("probe ridge L2 must be positive")
    dynamics = dataset.dynamics
    dynamics_state = {
        "current_action_means": _action_means(
            dynamics.current_actions, dynamics.deltas,
        ),
        "published_action_means": _action_means(
            dynamics.published_actions, dynamics.deltas,
        ),
        "executed_action_means": _action_means(
            dynamics.executed_actions, dynamics.deltas,
        ),
    }
    horizons: dict[str, object] = {}
    for view in dataset.horizons:
        mean = np.mean(view.features, axis=0)
        scale = np.std(view.features, axis=0)
        scale[scale < 1e-9] = 1.0
        normalized = (view.features - mean) / scale
        design = np.column_stack((np.ones(view.rows), normalized))
        penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_l2
        penalty[0, 0] = 0.0
        gram = design.T @ design / view.rows + penalty
        labels = np.column_stack((
            view.hit_labels.astype(np.float64),
            view.shield_collapse_labels.astype(np.float64),
        ))
        coefficients = np.linalg.solve(gram, design.T @ labels / view.rows)
        if np.any(~np.isfinite(coefficients)):
            raise ValueError("probe ridge fit produced non-finite coefficients")
        horizons[str(view.horizon)] = {
            "normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
            "coefficients": {
                "hit": coefficients[:, 0].tolist(),
                "shield_collapse": coefficients[:, 1].tolist(),
            },
            "train_prevalence": {
                "hit": float(np.mean(view.hit_labels)),
                "shield_collapse": float(np.mean(view.shield_collapse_labels)),
            },
            "train_counts": {
                "rows": view.rows,
                "hit_positives": int(np.sum(view.hit_labels)),
                "shield_collapse_positives": int(np.sum(
                    view.shield_collapse_labels
                )),
            },
        }
    return {
        "feature_schema": PROBE_FEATURE_SCHEMA,
        "feature_names": list(PROBE_FEATURE_NAMES),
        "model": "standardized-ridge-linear-probability",
        "ridge_l2": ridge_l2,
        "dynamics": dynamics_state,
        "horizons": horizons,
    }


def _binary_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    targets = labels.astype(np.float64)
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    positives = int(np.sum(labels))
    negatives = int(labels.size - positives)
    order = np.argsort(probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        stop = start + 1
        while (
            stop < labels.size
            and sorted_probabilities[stop] == sorted_probabilities[start]
        ):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    auc = None
    if positives and negatives:
        auc = float(
            (np.sum(ranks[labels]) - positives * (positives + 1) / 2.0)
            / (positives * negatives)
        )
    descending = np.argsort(-probabilities, kind="stable")
    sorted_targets = targets[descending]
    average_precision = None
    if positives:
        average_precision = float(np.sum(
            np.cumsum(sorted_targets) / np.arange(1, labels.size + 1)
            * sorted_targets
        ) / positives)
    return {
        "rows": int(labels.size),
        "positives": positives,
        "negatives": negatives,
        "prevalence": float(np.mean(targets)),
        "brier": float(np.mean((probabilities - targets) ** 2)),
        "negative_log_likelihood": float(-np.mean(
            targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped)
        )),
        "roc_auc": auc,
        "average_precision": average_precision,
    }


def _episode_bootstrap_brier_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    episode_indices: np.ndarray,
    *,
    episode_count: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    if samples <= 0 or episode_count <= 0:
        raise ValueError("episode bootstrap settings must be positive")
    targets = labels.astype(np.float64)
    row_delta = (candidate - targets) ** 2 - (baseline - targets) ** 2
    sums = np.bincount(episode_indices, weights=row_delta, minlength=episode_count)
    counts = np.bincount(episode_indices, minlength=episode_count)
    if np.any(counts == 0):
        raise ValueError("whole-episode bootstrap received an empty episode")
    random = np.random.default_rng(seed)
    draws = random.integers(0, episode_count, size=(samples, episode_count))
    values = np.sum(sums[draws], axis=1) / np.sum(counts[draws], axis=1)
    return {
        "unit": "complete-physical-episode",
        "samples": samples,
        "seed": seed,
        "point": float(np.mean(row_delta)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def evaluate_factual_probe_models(
    state: dict[str, object],
    dataset: FactualProbeDataset,
    *,
    dynamics_mse_ratio_max: float,
    execution_match_rate_min: float,
    mismatch_rows_min: int,
    mismatch_mse_ratio_max: float,
    minimum_train_positives: int,
    minimum_validation_positives: int,
    minimum_validation_negatives: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Evaluate frozen train-only probe states once on complete episodes."""
    dynamics = dataset.dynamics
    dynamics_state = state["dynamics"]
    assert isinstance(dynamics_state, dict)
    predictions = {}
    unseen = {}
    for name, actions in (
        ("current", dynamics.current_actions),
        ("published", dynamics.published_actions),
        ("executed", dynamics.executed_actions),
    ):
        means = dynamics_state[f"{name}_action_means"]
        prediction, unseen_rows = _dynamics_predictions(means, actions)
        predictions[name] = prediction
        unseen[name] = unseen_rows
    global_prediction = np.repeat(
        np.asarray(
            dynamics_state["executed_action_means"]["__global__"],
            dtype=np.float64,
        )[None, :],
        dynamics.rows,
        axis=0,
    )
    global_mse = _joint_mse(global_prediction, dynamics.deltas)
    executed_mse = _joint_mse(predictions["executed"], dynamics.deltas)
    matches = np.asarray([
        published == executed
        for published, executed in zip(
            dynamics.published_actions,
            dynamics.executed_actions,
            strict=True,
        )
    ], dtype=np.bool_)
    mismatch = ~matches
    mismatch_rows = int(np.sum(mismatch))
    mismatch_ratio = None
    if mismatch_rows:
        mismatch_executed_mse = _joint_mse(
            predictions["executed"][mismatch], dynamics.deltas[mismatch]
        )
        mismatch_published_mse = _joint_mse(
            predictions["published"][mismatch], dynamics.deltas[mismatch]
        )
        mismatch_ratio = (
            mismatch_executed_mse / mismatch_published_mse
            if mismatch_published_mse > 0.0 else math.inf
        )
    match_rate = float(np.mean(matches))
    overall_ratio = executed_mse / global_mse if global_mse > 0.0 else math.inf
    alignment_passed = (
        match_rate >= execution_match_rate_min
        or (
            mismatch_rows >= mismatch_rows_min
            and mismatch_ratio is not None
            and mismatch_ratio <= mismatch_mse_ratio_max
        )
    )
    dynamics_passed = overall_ratio <= dynamics_mse_ratio_max and alignment_passed
    dynamics_result = {
        "rows": dynamics.rows,
        "episode_count": len(dynamics.episode_ids),
        "published_executed_match_rate": match_rate,
        "published_executed_mismatch_rows": mismatch_rows,
        "unseen_action_rows": unseen,
        "joint_mse": {
            "global": global_mse,
            "current_action": _joint_mse(predictions["current"], dynamics.deltas),
            "published_action": _joint_mse(
                predictions["published"], dynamics.deltas
            ),
            "executed_action": executed_mse,
        },
        "executed_to_global_mse_ratio": overall_ratio,
        "mismatch_executed_to_published_mse_ratio": mismatch_ratio,
        "alignment_passed": alignment_passed,
        "gate_passed": dynamics_passed,
    }

    horizon_state = state["horizons"]
    assert isinstance(horizon_state, dict)
    horizon_results = {}
    hit_passes = 0
    shield_passes = 0
    hit_sufficient = 0
    shield_sufficient = 0
    for horizon_index, view in enumerate(dataset.horizons):
        fitted = horizon_state[str(view.horizon)]
        normalization = fitted["normalization"]
        mean = np.asarray(normalization["mean"], dtype=np.float64)
        scale = np.asarray(normalization["scale"], dtype=np.float64)
        normalized = (view.features - mean) / scale
        design = np.column_stack((np.ones(view.rows), normalized))
        target_results = {}
        for target_index, (name, labels) in enumerate((
            ("hit", view.hit_labels),
            ("shield_collapse", view.shield_collapse_labels),
        )):
            coefficients = np.asarray(
                fitted["coefficients"][name], dtype=np.float64
            )
            candidate = np.clip(design @ coefficients, 0.0, 1.0)
            prevalence = float(fitted["train_prevalence"][name])
            baseline = np.full(view.rows, prevalence, dtype=np.float64)
            train_positives = int(fitted["train_counts"][f"{name}_positives"])
            validation_positives = int(np.sum(labels))
            validation_negatives = int(labels.size - validation_positives)
            sufficient = (
                train_positives >= minimum_train_positives
                and validation_positives >= minimum_validation_positives
                and validation_negatives >= minimum_validation_negatives
            )
            bootstrap = _episode_bootstrap_brier_delta(
                candidate,
                baseline,
                labels,
                view.episode_indices,
                episode_count=len(view.episode_ids),
                samples=bootstrap_samples,
                seed=bootstrap_seed + horizon_index * 2 + target_index,
            )
            passed = sufficient and float(bootstrap["upper_95"]) < 0.0
            if name == "hit":
                hit_sufficient += int(sufficient)
                hit_passes += int(passed)
            else:
                shield_sufficient += int(sufficient)
                shield_passes += int(passed)
            target_results[name] = {
                "sufficient_event_support": sufficient,
                "gate_passed": passed,
                "candidate": _binary_metrics(candidate, labels),
                "train_prevalence_baseline": _binary_metrics(baseline, labels),
                "whole_episode_bootstrap_candidate_minus_baseline_brier": bootstrap,
            }
        horizon_results[str(view.horizon)] = {
            "rows": view.rows,
            "episode_count": len(view.episode_ids),
            "targets": target_results,
        }

    if not dynamics_passed:
        decision = "stop-and-repair-action-time-boundary"
    elif hit_passes and shield_passes:
        decision = "proceed-current-observation-factual-signal"
    elif not hit_sufficient or not shield_sufficient:
        decision = "inconclusive-factual-event-support"
    else:
        decision = "modify-transparent-current-observation-probe"
    return {
        "dynamics": dynamics_result,
        "horizons": horizon_results,
        "summary": {
            "dynamics_gate_passed": dynamics_passed,
            "hit_horizons_with_sufficient_support": hit_sufficient,
            "hit_horizons_passed": hit_passes,
            "shield_horizons_with_sufficient_support": shield_sufficient,
            "shield_horizons_passed": shield_passes,
            "decision": decision,
            "history_admitted": False,
            "history_rule": (
                "No history follows from this pilot alone. A failed current-root "
                "linear probe first requires a separately frozen temporal-ambiguity "
                "diagnosis; a passing probe leaves history unsupported."
            ),
        },
    }
