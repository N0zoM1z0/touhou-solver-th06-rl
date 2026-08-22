"""Learner view for factual randomized action-intention outcomes.

Each row is one accepted step-zero assignment from an immutable complete Wine
episode.  The assigned action is an observed randomized treatment; the label is
the factual h12 intention-to-treat HIT outcome.  No alternate successor is
constructed.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import math
import multiprocessing
from pathlib import Path
from typing import Iterable

import numpy as np

from .episode_dataset import (
    PortableDecisionRoot,
    iter_decision_epochs,
    iter_episode_transitions,
)
from .factual_probes import PROBE_FEATURE_NAMES, action_conditioned_probe_features


DATASET_SCHEMA = "th06-rl-h12-action-intention-dataset-v1"


@dataclass(frozen=True)
class ActionExposureTargetRow:
    """One group start under the L2l-frozen factual ITT rule."""

    episode_id: str
    group_id: int
    start_sequence: int
    observation: PortableDecisionRoot
    intended_action: str
    assignment_probability: float
    status: str
    label: bool | None
    hit_offset: int | None
    any_override: bool
    control_dead_end_before_outcome: bool


@dataclass(frozen=True)
class ActionIntentionDataset:
    schema: str
    exposure_roots: int
    episode_ids: tuple[str, ...]
    inventory: tuple[dict[str, object], ...]
    episode_indices: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    intended_actions: tuple[str, ...]
    assignment_probabilities: np.ndarray
    any_overrides: np.ndarray
    control_dead_ends: np.ndarray
    hit_offsets: tuple[int | None, ...]

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_exposure_target_rows(
    run_dir: Path,
    *,
    exposure_roots: int,
) -> tuple[ActionExposureTargetRow, ...]:
    """Project exact group outcomes without mutating the frozen L2l auditor."""
    starts = []
    for epoch in iter_decision_epochs(run_dir):
        exposure = epoch.action_exposure
        if exposure is not None and exposure.step == 0:
            starts.append((
                exposure.group_id,
                epoch.start_sequence,
                exposure.intended_action,
                exposure.assignment_probability,
                epoch.observation,
                epoch.episode_id,
            ))
    needed = {
        sequence + offset
        for _group, sequence, _action, _probability, _observation, _episode in starts
        for offset in range(exposure_roots)
    }
    transitions = {}
    for item in iter_episode_transitions(run_dir):
        if item.sequence not in needed:
            continue
        exposure = item.action_exposure
        transitions[item.sequence] = {
            "elapsed": int(item.outcome.get("elapsed_frames", -1)),
            "hit": bool(item.outcome.get("life_lost")),
            "dead_end": bool(item.outcome.get("control_dead_end")),
            "executed": item.executed_action,
            "group": None if exposure is None else exposure.group_id,
            "step": None if exposure is None else exposure.step,
            "override": None if exposure is None else exposure.override_reason,
        }

    rows = []
    for group_id, start, intended, probability, observation, episode_id in starts:
        window = [transitions.get(start + offset) for offset in range(exposure_roots)]
        if any(item is None for item in window):
            rows.append(ActionExposureTargetRow(
                episode_id, group_id, start, observation, intended, probability,
                "unsupported-end", None, None, False, False,
            ))
            continue
        assert all(item is not None for item in window)
        any_override = any(
            item is not None and item["override"] is not None for item in window
        )
        any_dead_end = any(
            item is not None and bool(item["dead_end"]) for item in window
        )
        if any(int(item["elapsed"]) != 1 for item in window if item is not None):
            rows.append(ActionExposureTargetRow(
                episode_id, group_id, start, observation, intended, probability,
                "censored-observation-gap", None, None, any_override, any_dead_end,
            ))
            continue
        hits = [
            offset for offset, item in enumerate(window)
            if item is not None and bool(item["hit"])
        ]
        first_hit = hits[0] if hits else None
        outcome_length = exposure_roots if first_hit is None else first_hit + 1
        observed = window[:outcome_length]
        observed_override = any(
            item is not None and item["override"] is not None for item in observed
        )
        observed_dead_end = any(
            item is not None and bool(item["dead_end"]) for item in observed
        )
        first = observed[0]
        assert first is not None
        if first["executed"] != intended:
            status = "censored-assignment-not-executed"
        elif any(
            item is not None
            and item["group"] is not None
            and item["group"] != group_id
            for item in observed
        ):
            status = "censored-next-assignment"
        elif first_hit is None and [
            item["step"] if item is not None else None for item in window
        ] != list(range(exposure_roots)):
            status = "censored-incomplete-protocol"
        else:
            label = first_hit is not None
            rows.append(ActionExposureTargetRow(
                episode_id, group_id, start, observation, intended, probability,
                f"accepted-label-{int(label)}", label, first_hit,
                observed_override, observed_dead_end,
            ))
            continue
        rows.append(ActionExposureTargetRow(
            episode_id, group_id, start, observation, intended, probability,
            status, None, first_hit, observed_override, observed_dead_end,
        ))
    return tuple(rows)


def summarize_target_rows(rows: tuple[ActionExposureTargetRow, ...]) -> dict[str, object]:
    """Expose a regression-comparable summary of the frozen target rule."""
    status: Counter[str] = Counter(row.status for row in rows)
    accepted: Counter[str] = Counter(
        row.intended_action for row in rows if row.label is not None
    )
    positives: Counter[str] = Counter(
        row.intended_action for row in rows if row.label is True
    )
    offsets: Counter[int] = Counter(
        row.hit_offset for row in rows if row.label is True and row.hit_offset is not None
    )
    positive_dead_ends = sum(
        row.label is True and row.control_dead_end_before_outcome for row in rows
    )
    negative_overrides = sum(
        row.label is False and row.any_override for row in rows
    )
    if positive_dead_ends:
        status["positive-after-control-dead-end"] = positive_dead_ends
    if negative_overrides:
        status["accepted-negative-with-shield-override"] = negative_overrides
    return {
        "group_starts": len(rows),
        "status": dict(sorted(status.items())),
        "accepted_actions": dict(sorted(accepted.items())),
        "positive_actions": dict(sorted(positives.items())),
        "hit_offsets": {str(key): value for key, value in sorted(offsets.items())},
    }


def _load_episode(
    task: tuple[Path, int, int],
) -> tuple[str, dict[str, object], list[tuple[object, ...]]]:
    run_dir, episode_index, exposure_roots = task
    targets = action_exposure_target_rows(run_dir, exposure_roots=exposure_roots)
    if not targets:
        raise ValueError("action-intention episode contains no group starts")
    episode_id = targets[0].episode_id
    if any(row.episode_id != episode_id for row in targets):
        raise ValueError("action-intention run exposed multiple episode identities")
    accepted = []
    for row in targets:
        if row.label is None:
            continue
        expected_probability = 1.0 / len(row.observation.locally_admissible_actions)
        if not math.isclose(
            row.assignment_probability,
            expected_probability,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("action-intention assignment is not uniform at step zero")
        accepted.append((
            episode_index,
            action_conditioned_probe_features(
                row.observation,
                row.intended_action,
            ),
            row.label,
            row.intended_action,
            row.assignment_probability,
            row.any_override,
            row.control_dead_end_before_outcome,
            row.hit_offset,
        ))
    if not accepted:
        raise ValueError("action-intention episode contains no accepted labels")
    positives = sum(bool(row[2]) for row in accepted)
    return (
        episode_id,
        {
            "episode_id": episode_id,
            "run_sha256": _sha256(run_dir / "run.json"),
            "manifest_sha256": _sha256(run_dir / "manifest.json"),
            "group_starts": len(targets),
            "accepted_rows": len(accepted),
            "positives": positives,
            "negatives": len(accepted) - positives,
            "positive_after_control_dead_end": sum(
                bool(row[2]) and bool(row[6]) for row in accepted
            ),
        },
        accepted,
    )


def load_action_intention_dataset(
    run_dirs: Iterable[Path],
    *,
    exposure_roots: int,
    max_rows: int = 100_000,
    workers: int = 1,
) -> ActionIntentionDataset:
    """Load exact accepted group rows from complete immutable Wine episodes."""
    paths = tuple(Path(path).resolve() for path in run_dirs)
    if (
        not paths
        or exposure_roots <= 0
        or max_rows <= 0
        or workers <= 0
        or workers > len(paths)
    ):
        raise ValueError("action-intention dataset settings are invalid")
    tasks = tuple(
        (path, episode_index, exposure_roots)
        for episode_index, path in enumerate(paths)
    )
    if workers == 1:
        episodes = [_load_episode(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            episodes = list(executor.map(_load_episode, tasks))
    episode_ids = tuple(row[0] for row in episodes)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate action-intention episode identity")
    inventory = tuple(row[1] for row in episodes)
    rows = [sample for episode in episodes for sample in episode[2]]
    if len(rows) > max_rows:
        raise ValueError("action-intention dataset exceeds its row limit")
    features = np.asarray([row[1] for row in rows], dtype=np.float64)
    labels = np.asarray([row[2] for row in rows], dtype=np.bool_)
    if (
        features.shape != (len(rows), len(PROBE_FEATURE_NAMES))
        or not np.all(np.isfinite(features))
        or not np.any(labels)
        or np.all(labels)
    ):
        raise ValueError("action-intention dataset is malformed or single-class")
    return ActionIntentionDataset(
        DATASET_SCHEMA,
        exposure_roots,
        episode_ids,
        inventory,
        np.asarray([row[0] for row in rows], dtype=np.int64),
        features,
        labels,
        tuple(str(row[3]) for row in rows),
        np.asarray([row[4] for row in rows], dtype=np.float64),
        np.asarray([row[5] for row in rows], dtype=np.bool_),
        np.asarray([row[6] for row in rows], dtype=np.bool_),
        tuple(None if row[7] is None else int(row[7]) for row in rows),
    )
