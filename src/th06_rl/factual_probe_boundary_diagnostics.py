"""Boundary diagnostics for frozen factual HIT-risk probes.

This module does not fit a model or construct a counterfactual transition.  It
reconstructs the exact factual L2 rows while retaining three pieces of recorder
evidence that were intentionally absent from actor features: logged action
propensity, whether the published action differed from the reactive baseline,
and elapsed game frames since the most recent physical HIT.  They are used only
to diagnose support, lifecycle, and calibration boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .episode_dataset import (
    EpisodeDatasetError,
    iter_decision_epochs,
    iter_episode_transitions,
)
from .factual_probe_diagnostics import (
    _calibration_bins,
    _full_predictions,
    _state_only_predictions,
)
from .factual_probes import (
    FactualProbeDataset,
    _binary_metrics,
    _transition_fact,
    action_conditioned_probe_features,
    load_factual_probe_dataset,
)


BOUNDARY_DIAGNOSIS_SCHEMA = "th06-rl-l2c-boundary-diagnosis-v1"
LIFECYCLE_STRATA = (
    "pre-first-hit",
    "post-hit-0-63-frames",
    "post-hit-64-255-frames",
    "post-hit-256-plus-frames",
)
PROPENSITY_STRATA = (
    "p-lt-0.025",
    "p-0.025-to-lt-0.05",
    "p-0.05-to-lt-0.1",
    "p-0.1-to-lt-0.5",
    "p-ge-0.5",
)


@dataclass(frozen=True)
class BoundaryHorizonDataset:
    horizon: int
    episode_ids: tuple[str, ...]
    episode_indices: np.ndarray
    features: np.ndarray
    hit_labels: np.ndarray
    published_actions: tuple[str, ...]
    baseline_actions: tuple[str | None, ...]
    behavior_probabilities: np.ndarray
    shield_action_counts: np.ndarray
    lifecycle_strata: tuple[str, ...]
    frames_since_prior_hit: tuple[int | None, ...]

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class BoundaryProbeDataset:
    factual: FactualProbeDataset
    horizons: tuple[BoundaryHorizonDataset, ...]


def _lifecycle_stratum(frames_since_prior_hit: int | None) -> str:
    if frames_since_prior_hit is None:
        return LIFECYCLE_STRATA[0]
    if frames_since_prior_hit < 64:
        return LIFECYCLE_STRATA[1]
    if frames_since_prior_hit < 256:
        return LIFECYCLE_STRATA[2]
    return LIFECYCLE_STRATA[3]


def _propensity_stratum(probability: float) -> str:
    if probability < 0.025:
        return PROPENSITY_STRATA[0]
    if probability < 0.05:
        return PROPENSITY_STRATA[1]
    if probability < 0.1:
        return PROPENSITY_STRATA[2]
    if probability < 0.5:
        return PROPENSITY_STRATA[3]
    return PROPENSITY_STRATA[4]


def _prior_hit_distances(facts: tuple[object, ...]) -> tuple[int | None, ...]:
    """Return elapsed game frames since HIT at every transition-start state."""
    distances: list[int | None] = [None]
    distance: int | None = None
    for fact in facts:
        elapsed = int(getattr(fact, "elapsed_frames"))
        if elapsed <= 0:
            raise EpisodeDatasetError("lifecycle diagnosis saw nonpositive elapsed time")
        if bool(getattr(fact, "hit")):
            distance = 0
        elif distance is not None:
            distance += elapsed
        distances.append(distance)
    return tuple(distances)


def _load_boundary_episode(
    run_dir: Path,
    *,
    episode_index: int,
    horizons: tuple[int, ...],
) -> tuple[str, dict[int, list[tuple[object, ...]]]]:
    epochs = tuple(iter_decision_epochs(run_dir))
    facts = tuple(_transition_fact(row) for row in iter_episode_transitions(run_dir))
    if not epochs or not facts or any(
        fact.sequence != index for index, fact in enumerate(facts)
    ):
        raise EpisodeDatasetError("boundary diagnosis episode is empty or non-contiguous")
    episode_id = epochs[0].episode_id
    if any(epoch.episode_id != episode_id for epoch in epochs):
        raise EpisodeDatasetError("boundary diagnosis saw multiple episode identities")
    prior_hit_distances = _prior_hit_distances(facts)
    rows = {horizon: [] for horizon in horizons}
    for epoch in epochs:
        if not epoch.learning_eligible or epoch.published_action is None:
            continue
        start = epoch.start_sequence
        if not 0 <= start < len(facts):
            raise EpisodeDatasetError("boundary decision start is outside the transition stream")
        if epoch.baseline_action is None:
            raise EpisodeDatasetError("eligible boundary decision lacks a baseline action")
        probability_map = dict(epoch.behavior_probabilities)
        probability = float(epoch.behavior_probability)
        if (
            not math.isfinite(probability)
            or probability_map.get(epoch.published_action) != probability
        ):
            raise EpisodeDatasetError("boundary decision propensity disagrees with publication")
        features = action_conditioned_probe_features(
            epoch.observation,
            epoch.published_action,
        )
        frames_since_hit = prior_hit_distances[start]
        for horizon in horizons:
            stop = start + horizon
            if stop > len(facts):
                continue
            window = facts[start:stop]
            if any(not fact.valid_for_horizon for fact in window):
                continue
            rows[horizon].append((
                episode_index,
                features,
                any(fact.hit for fact in window),
                epoch.published_action,
                epoch.baseline_action,
                probability,
                len(epoch.observation.locally_admissible_actions),
                _lifecycle_stratum(frames_since_hit),
                frames_since_hit,
            ))
    return episode_id, rows


def load_boundary_probe_dataset(
    run_dirs: Iterable[Path],
    *,
    horizons: tuple[int, ...],
    max_rows: int = 2_000_000,
) -> BoundaryProbeDataset:
    """Load exact L2 rows plus recorder-only diagnostic strata."""
    paths = tuple(Path(path).resolve() for path in run_dirs)
    factual = load_factual_probe_dataset(paths, horizons=horizons, max_rows=max_rows)
    episode_ids: list[str] = []
    rows: dict[int, list[tuple[object, ...]]] = {horizon: [] for horizon in horizons}
    for episode_index, run_dir in enumerate(paths):
        episode_id, episode_rows = _load_boundary_episode(
            run_dir,
            episode_index=episode_index,
            horizons=horizons,
        )
        episode_ids.append(episode_id)
        for horizon in horizons:
            rows[horizon].extend(episode_rows[horizon])
            if len(rows[horizon]) > max_rows:
                raise ValueError("boundary horizon dataset exceeds its row limit")
    if tuple(episode_ids) != factual.episode_ids:
        raise EpisodeDatasetError("boundary and factual episode inventories differ")

    views = []
    for factual_view in factual.horizons:
        horizon_rows = rows[factual_view.horizon]
        episode_indices = np.asarray([row[0] for row in horizon_rows], dtype=np.int64)
        features = np.asarray([row[1] for row in horizon_rows], dtype=np.float64)
        labels = np.asarray([row[2] for row in horizon_rows], dtype=np.bool_)
        if (
            not np.array_equal(episode_indices, factual_view.episode_indices)
            or not np.array_equal(features, factual_view.features)
            or not np.array_equal(labels, factual_view.hit_labels)
        ):
            raise EpisodeDatasetError("boundary rows do not exactly reproduce L2")
        views.append(BoundaryHorizonDataset(
            factual_view.horizon,
            factual.episode_ids,
            episode_indices,
            features,
            labels,
            tuple(str(row[3]) for row in horizon_rows),
            tuple(None if row[4] is None else str(row[4]) for row in horizon_rows),
            np.asarray([row[5] for row in horizon_rows], dtype=np.float64),
            np.asarray([row[6] for row in horizon_rows], dtype=np.int64),
            tuple(str(row[7]) for row in horizon_rows),
            tuple(None if row[8] is None else int(row[8]) for row in horizon_rows),
        ))
    return BoundaryProbeDataset(factual, tuple(views))


def _calibration_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int,
) -> dict[str, object]:
    calibration = _calibration_bins(probabilities, labels, bins=bins)
    event_rate = float(np.mean(labels))
    mean_prediction = float(np.mean(probabilities))
    return {
        "mean_prediction": mean_prediction,
        "event_rate": event_rate,
        "calibration_in_the_large": mean_prediction - event_rate,
        "expected_calibration_error": calibration["expected_calibration_error"],
        "zero_clipped_rows": int(np.sum(probabilities == 0.0)),
        "one_clipped_rows": int(np.sum(probabilities == 1.0)),
        "equal_width_bins": calibration["bins"],
    }


def _stratum_result(
    mask: np.ndarray,
    *,
    full: np.ndarray,
    state_only: np.ndarray,
    labels: np.ndarray,
    episode_indices: np.ndarray,
    episode_ids: tuple[str, ...],
    calibration_bins: int,
) -> dict[str, object]:
    if not np.any(mask):
        return {
            "rows": 0,
            "positives": 0,
            "episode_count": 0,
            "full_minus_state_only_brier": None,
            "episodes_favoring_full": 0,
            "full": None,
            "state_only": None,
            "full_calibration": None,
            "state_only_calibration": None,
            "per_episode": [],
        }
    selected_episodes = sorted(set(int(value) for value in episode_indices[mask]))
    targets = labels[mask].astype(np.float64)
    row_delta = (full[mask] - targets) ** 2 - (state_only[mask] - targets) ** 2
    per_episode = []
    for episode_index in selected_episodes:
        members = mask & (episode_indices == episode_index)
        episode_targets = labels[members].astype(np.float64)
        episode_delta = (
            (full[members] - episode_targets) ** 2
            - (state_only[members] - episode_targets) ** 2
        )
        per_episode.append({
            "episode_id": episode_ids[episode_index],
            "rows": int(np.sum(members)),
            "positives": int(np.sum(labels[members])),
            "full_minus_state_only_brier": float(np.mean(episode_delta)),
        })
    return {
        "rows": int(np.sum(mask)),
        "positives": int(np.sum(labels[mask])),
        "episode_count": len(selected_episodes),
        "full_minus_state_only_brier": float(np.mean(row_delta)),
        "episodes_favoring_full": sum(
            row["full_minus_state_only_brier"] < 0.0 for row in per_episode
        ),
        "full": _binary_metrics(full[mask], labels[mask]),
        "state_only": _binary_metrics(state_only[mask], labels[mask]),
        "full_calibration": _calibration_summary(
            full[mask], labels[mask], bins=calibration_bins
        ),
        "state_only_calibration": _calibration_summary(
            state_only[mask], labels[mask], bins=calibration_bins
        ),
        "per_episode": per_episode,
    }


def diagnose_probe_boundaries(
    full_state: dict[str, object],
    state_only_state: dict[str, object],
    dataset: BoundaryProbeDataset,
    *,
    calibration_bins: int,
) -> dict[str, object]:
    """Describe support, lifecycle, and calibration without selecting a model."""
    horizon_results = {}
    for view in dataset.horizons:
        full = _full_predictions(full_state, view.horizon, "hit", view.features)
        state_only = _state_only_predictions(
            state_only_state, view.horizon, "hit", view.features
        )
        all_rows = np.ones(view.rows, dtype=np.bool_)
        baseline_equal = np.asarray([
            published == baseline
            for published, baseline in zip(
                view.published_actions, view.baseline_actions, strict=True
            )
        ], dtype=np.bool_)
        lifecycle = np.asarray(view.lifecycle_strata, dtype=object)
        propensity = np.asarray([
            _propensity_stratum(float(value))
            for value in view.behavior_probabilities
        ], dtype=object)
        actions = np.asarray(view.published_actions, dtype=object)

        def result(mask: np.ndarray) -> dict[str, object]:
            return _stratum_result(
                mask,
                full=full,
                state_only=state_only,
                labels=view.hit_labels,
                episode_indices=view.episode_indices,
                episode_ids=view.episode_ids,
                calibration_bins=calibration_bins,
            )

        horizon_results[str(view.horizon)] = {
            "overall": result(all_rows),
            "support": {
                "published_equals_baseline": result(baseline_equal),
                "published_differs_from_baseline": result(~baseline_equal),
                "propensity_strata": {
                    name: result(propensity == name)
                    for name in PROPENSITY_STRATA
                    if np.any(propensity == name)
                },
                "published_action": {
                    name: result(actions == name)
                    for name in sorted(set(view.published_actions))
                },
                "behavior_probability": {
                    "minimum": float(np.min(view.behavior_probabilities)),
                    "median": float(np.median(view.behavior_probabilities)),
                    "maximum": float(np.max(view.behavior_probabilities)),
                },
                "shield_action_count": {
                    "minimum": int(np.min(view.shield_action_counts)),
                    "median": float(np.median(view.shield_action_counts)),
                    "maximum": int(np.max(view.shield_action_counts)),
                },
                "identification_limit": (
                    "published-differs-from-baseline proves a non-baseline draw; "
                    "published-equals-baseline does not identify mixture membership"
                ),
            },
            "lifecycle": {
                name: result(lifecycle == name)
                for name in LIFECYCLE_STRATA
                if np.any(lifecycle == name)
            },
        }
    return {
        "schema": BOUNDARY_DIAGNOSIS_SCHEMA,
        "horizons": horizon_results,
        "summary": {
            "independent_confirmation": False,
            "causal_action_effect_identified": False,
            "history_admitted": False,
            "value_learning_admitted": False,
            "fresh_confirmation_required": True,
            "decision": "freeze-fresh-independent-confirmation",
        },
    }
