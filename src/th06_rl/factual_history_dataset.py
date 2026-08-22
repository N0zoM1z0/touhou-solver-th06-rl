"""Causal fixed-root histories for factual physical-HIT prediction.

Every model field existed at or before the current paused policy root.  Rows
without the full consecutive eligible prefix are dropped rather than padded.
The loader also retains the complete current-root view so frozen L2f source
scores can be reproduced before any history comparison is interpreted.
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
from .factual_probe_boundary_diagnostics import (
    _lifecycle_stratum,
    _prior_hit_distances,
)
from .factual_probes import (
    PROBE_FEATURE_NAMES,
    _sha256,
    _transition_fact,
    action_conditioned_probe_features,
)


HISTORY_FEATURE_SCHEMA = "th06-rl-fixed-portable-root-history-features-v1"


def history_feature_names(history_length: int) -> tuple[str, ...]:
    if history_length <= 0:
        raise ValueError("history length must be positive")
    return tuple(
        f"history_minus_{lag}:{name}"
        for lag in range(history_length, 0, -1)
        for name in PROBE_FEATURE_NAMES
    ) + tuple(f"current:{name}" for name in PROBE_FEATURE_NAMES)


@dataclass(frozen=True)
class HistoryHorizonDataset:
    horizon: int
    history_length: int
    episode_ids: tuple[str, ...]
    all_current_episode_indices: np.ndarray
    all_current_features: np.ndarray
    all_current_hit_labels: np.ndarray
    episode_indices: np.ndarray
    all_current_row_indices: np.ndarray
    features: np.ndarray
    current_features: np.ndarray
    hit_labels: np.ndarray
    published_actions: tuple[str, ...]
    baseline_actions: tuple[str | None, ...]
    behavior_probabilities: np.ndarray
    shield_action_counts: np.ndarray
    lifecycle_strata: tuple[str, ...]
    history_elapsed_game_frames: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def all_current_rows(self) -> int:
        return int(self.all_current_features.shape[0])


@dataclass(frozen=True)
class HistoryProbeDataset:
    episode_ids: tuple[str, ...]
    inventory: tuple[dict[str, object], ...]
    feature_schema: str
    feature_names: tuple[str, ...]
    horizons: tuple[HistoryHorizonDataset, ...]


def _history_prefix(epochs: tuple[object, ...], position: int, length: int):
    if position < length:
        return None
    prefix = epochs[position - length:position]
    current = epochs[position]
    if any(
        not bool(getattr(epoch, "learning_eligible"))
        or getattr(epoch, "published_action") is None
        or int(getattr(epoch, "elapsed_game_frames")) != 1
        for epoch in prefix
    ):
        return None
    chain = prefix + (current,)
    if any(
        int(getattr(left, "next_sequence"))
        != int(getattr(right, "start_sequence"))
        for left, right in zip(chain, chain[1:])
    ):
        raise EpisodeDatasetError("history policy roots are not causally consecutive")
    return prefix


def _load_history_episode(
    run_dir: Path,
    *,
    episode_index: int,
    horizons: tuple[int, ...],
    history_length: int,
) -> tuple[
    str,
    dict[str, object],
    dict[int, dict[str, list[object]]],
]:
    epochs = tuple(iter_decision_epochs(run_dir))
    facts = tuple(_transition_fact(row) for row in iter_episode_transitions(run_dir))
    if not epochs or not facts or any(
        fact.sequence != index for index, fact in enumerate(facts)
    ):
        raise EpisodeDatasetError("history episode is empty or non-contiguous")
    episode_id = epochs[0].episode_id
    if any(epoch.episode_id != episode_id for epoch in epochs):
        raise EpisodeDatasetError("history episode exposed multiple identities")
    prior_hit_distances = _prior_hit_distances(facts)
    rows = {
        horizon: {
            "all": [],
            "history": [],
        }
        for horizon in horizons
    }
    for position, epoch in enumerate(epochs):
        if not epoch.learning_eligible or epoch.published_action is None:
            continue
        start = int(epoch.start_sequence)
        if not 0 <= start < len(facts):
            raise EpisodeDatasetError("history decision start is outside transitions")
        current = action_conditioned_probe_features(
            epoch.observation, epoch.published_action
        )
        prefix = _history_prefix(epochs, position, history_length)
        history_features = None
        history_elapsed = None
        if prefix is not None:
            past = tuple(
                value
                for prior in prefix
                for value in action_conditioned_probe_features(
                    prior.observation, prior.published_action
                )
            )
            history_features = past + current
            history_elapsed = sum(int(prior.elapsed_game_frames) for prior in prefix)
        frames_since_hit = prior_hit_distances[start]
        if epoch.baseline_action is None:
            raise EpisodeDatasetError("eligible history row lacks a baseline action")
        probability_map = dict(epoch.behavior_probabilities)
        probability = float(epoch.behavior_probability)
        if (
            not math.isfinite(probability)
            or probability_map.get(epoch.published_action) != probability
        ):
            raise EpisodeDatasetError("history row propensity disagrees with publication")
        for horizon in horizons:
            stop = start + horizon
            if stop > len(facts):
                continue
            window = facts[start:stop]
            if any(
                fact.elapsed_frames != 1 or not fact.valid_for_horizon
                for fact in window
            ):
                continue
            label = any(fact.hit for fact in window)
            all_index = len(rows[horizon]["all"])
            rows[horizon]["all"].append((episode_index, current, label))
            if history_features is not None:
                rows[horizon]["history"].append((
                    episode_index,
                    all_index,
                    history_features,
                    current,
                    label,
                    epoch.published_action,
                    epoch.baseline_action,
                    probability,
                    len(epoch.observation.locally_admissible_actions),
                    _lifecycle_stratum(frames_since_hit),
                    history_elapsed,
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
        rows,
    )


def load_history_probe_dataset(
    run_dirs: Iterable[Path],
    *,
    horizons: tuple[int, ...],
    history_length: int,
    max_rows: int = 2_000_000,
) -> HistoryProbeDataset:
    """Load exact factual h16 rows and one fixed causal history subset."""
    paths = tuple(Path(path).resolve() for path in run_dirs)
    if not paths or history_length <= 0:
        raise ValueError("history dataset requires episodes and positive history")
    if (
        not horizons
        or tuple(sorted(set(horizons))) != horizons
        or any(not isinstance(horizon, int) or horizon <= 0 for horizon in horizons)
    ):
        raise ValueError("history horizons must be unique increasing positive integers")
    names = history_feature_names(history_length)
    episode_ids = []
    inventory = []
    accumulated = {
        horizon: {"all": [], "history": []}
        for horizon in horizons
    }
    all_offsets = {horizon: 0 for horizon in horizons}
    for episode_index, path in enumerate(paths):
        episode_id, episode_inventory, episode_rows = _load_history_episode(
            path,
            episode_index=episode_index,
            horizons=horizons,
            history_length=history_length,
        )
        if episode_id in episode_ids:
            raise ValueError(f"duplicate history episode identity {episode_id}")
        episode_ids.append(episode_id)
        inventory.append(episode_inventory)
        for horizon in horizons:
            offset = all_offsets[horizon]
            accumulated[horizon]["all"].extend(episode_rows[horizon]["all"])
            accumulated[horizon]["history"].extend(
                row[:1] + (int(row[1]) + offset,) + row[2:]
                for row in episode_rows[horizon]["history"]
            )
            all_offsets[horizon] += len(episode_rows[horizon]["all"])
            if len(accumulated[horizon]["all"]) > max_rows:
                raise ValueError("history all-current view exceeds its row limit")
    views = []
    for horizon in horizons:
        all_rows = accumulated[horizon]["all"]
        rows = accumulated[horizon]["history"]
        if not all_rows or not rows:
            raise ValueError("history dataset contains an empty view")
        all_features = np.asarray([row[1] for row in all_rows], dtype=np.float64)
        all_labels = np.asarray([row[2] for row in all_rows], dtype=np.bool_)
        indices = np.asarray([row[1] for row in rows], dtype=np.int64)
        features = np.asarray([row[2] for row in rows], dtype=np.float64)
        current = np.asarray([row[3] for row in rows], dtype=np.float64)
        labels = np.asarray([row[4] for row in rows], dtype=np.bool_)
        if (
            features.shape[1] != len(names)
            or current.shape[1] != len(PROBE_FEATURE_NAMES)
            or set(int(row[0]) for row in all_rows) != set(range(len(episode_ids)))
            or set(int(row[0]) for row in rows) != set(range(len(episode_ids)))
            or np.any(indices < 0)
            or np.any(indices >= len(all_rows))
            or not np.array_equal(current, all_features[indices])
            or not np.array_equal(labels, all_labels[indices])
            or not np.all(np.isfinite(features))
        ):
            raise EpisodeDatasetError("history subset does not reproduce current rows")
        views.append(HistoryHorizonDataset(
            horizon,
            history_length,
            tuple(episode_ids),
            np.asarray([row[0] for row in all_rows], dtype=np.int64),
            all_features,
            all_labels,
            np.asarray([row[0] for row in rows], dtype=np.int64),
            indices,
            features,
            current,
            labels,
            tuple(str(row[5]) for row in rows),
            tuple(None if row[6] is None else str(row[6]) for row in rows),
            np.asarray([row[7] for row in rows], dtype=np.float64),
            np.asarray([row[8] for row in rows], dtype=np.int64),
            tuple(str(row[9]) for row in rows),
            np.asarray([row[10] for row in rows], dtype=np.int64),
        ))
    return HistoryProbeDataset(
        tuple(episode_ids),
        tuple(inventory),
        HISTORY_FEATURE_SCHEMA,
        names,
        tuple(views),
    )
