"""Bounded observed-primitive sets for factual physical-HIT prediction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import multiprocessing
from pathlib import Path
from typing import Iterable

import numpy as np

from .episode_dataset import (
    EpisodeDatasetError,
    iter_decision_epochs,
    iter_episode_frames,
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


PRIMITIVE_FEATURE_SCHEMA = "th06-rl-observed-primitive-set-features-v1"
PRIMITIVE_CONTRACT = "observed-hazard-kinematics-v1"
PRIMITIVE_HORIZON = 4
TOKEN_FEATURE_NAMES = (
    "is_aabb",
    "is_laser",
    "projection_frame_1",
    "projection_frame_2",
    "projection_frame_3",
    "projection_frame_4",
    "relative_x",
    "relative_y",
    "geometry_1",
    "geometry_2",
    "geometry_3",
    "geometry_4",
    "geometry_5",
)


@dataclass(frozen=True)
class PrimitiveHorizonDataset:
    horizon: int
    token_cap: int
    episode_ids: tuple[str, ...]
    episode_indices: np.ndarray
    current_features: np.ndarray
    scalar_features: np.ndarray
    primitive_tokens: np.ndarray
    primitive_masks: np.ndarray
    token_counts: np.ndarray
    truncated_token_counts: np.ndarray
    hit_labels: np.ndarray
    published_actions: tuple[str, ...]
    baseline_actions: tuple[str | None, ...]
    behavior_probabilities: np.ndarray
    lifecycle_strata: tuple[str, ...]

    @property
    def rows(self) -> int:
        return int(self.current_features.shape[0])


@dataclass(frozen=True)
class PrimitiveProbeDataset:
    episode_ids: tuple[str, ...]
    inventory: tuple[dict[str, object], ...]
    feature_schema: str
    scalar_feature_names: tuple[str, ...]
    token_feature_names: tuple[str, ...]
    horizons: tuple[PrimitiveHorizonDataset, ...]


def normalized_scalar_features(features: tuple[float, ...]) -> tuple[float, ...]:
    """Apply fixed portable geometry scales, never train-fitted statistics."""
    if len(features) != len(PROBE_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in features
    ):
        raise ValueError("primitive scalar features are malformed")
    return (
        (features[0] - 192.0) / 192.0,
        (features[1] - 224.0) / 224.0,
        features[2] / 128.0,
        features[3] / math.log1p(640.0),
        features[4] / math.log1p(64.0),
        features[5] / 18.0,
        features[6],
        features[7] / 448.0,
        (features[8] - 192.0) / 192.0,
        (features[9] - 224.0) / 224.0,
        features[10] / 184.0,
        features[11],
        features[12],
        features[13],
        features[14],
    )


def _finite_tuple(value: object, width: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != width:
        raise EpisodeDatasetError(f"primitive {name} width changed")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise EpisodeDatasetError(f"primitive {name} is non-finite")
    return result


def primitive_tokens_from_frame(
    frame,
    *,
    token_cap: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Project one recorder root into a deterministic cap-checked token set."""
    if token_cap <= 0:
        raise ValueError("primitive token cap must be positive")
    decision = frame.decision
    snapshot = frame.snapshot
    aabb_frames = decision.get("shield_aabb_frames")
    laser_frames = decision.get("shield_laser_frames")
    if (
        decision.get("shield_contract") != PRIMITIVE_CONTRACT
        or int(decision.get("shield_horizon", 0)) != PRIMITIVE_HORIZON
        or not isinstance(aabb_frames, list)
        or not isinstance(laser_frames, list)
        or len(aabb_frames) != PRIMITIVE_HORIZON
        or len(laser_frames) != PRIMITIVE_HORIZON
    ):
        raise EpisodeDatasetError("observed primitive recorder contract changed")
    try:
        player_x = float(snapshot["x"])
        player_y = float(snapshot["y"])
    except (KeyError, TypeError, ValueError) as error:
        raise EpisodeDatasetError("primitive player root is malformed") from error
    if not math.isfinite(player_x) or not math.isfinite(player_y):
        raise EpisodeDatasetError("primitive player root is non-finite")

    keyed: list[tuple[tuple[object, ...], tuple[float, ...]]] = []
    for frame_index, (aabbs, lasers) in enumerate(
        zip(aabb_frames, laser_frames, strict=True)
    ):
        if not isinstance(aabbs, list) or not isinstance(lasers, list):
            raise EpisodeDatasetError("primitive frame is not a list")
        one_hot = tuple(float(index == frame_index) for index in range(4))
        for unresolved in aabbs:
            left, top, right, bottom = _finite_tuple(unresolved, 4, "AABB")
            if right < left or bottom < top:
                raise EpisodeDatasetError("primitive AABB is inverted")
            center_x = (left + right) * 0.5
            center_y = (top + bottom) * 0.5
            token = (
                1.0,
                0.0,
                *one_hot,
                (center_x - player_x) / 384.0,
                (center_y - player_y) / 448.0,
                (right - left) * 0.5 / 64.0,
                (bottom - top) * 0.5 / 64.0,
                0.0,
                0.0,
                0.0,
            )
            keyed.append(((frame_index, 0, left, top, right, bottom), token))
        for unresolved in lasers:
            x, y, angle, extent_1, extent_2, width = _finite_tuple(
                unresolved, 6, "laser"
            )
            if width < 0.0 or extent_1 < 0.0 or extent_2 < 0.0:
                raise EpisodeDatasetError("primitive laser extent is negative")
            token = (
                0.0,
                1.0,
                *one_hot,
                (x - player_x) / 384.0,
                (y - player_y) / 448.0,
                math.cos(angle),
                math.sin(angle),
                extent_1 / 512.0,
                extent_2 / 512.0,
                width / 32.0,
            )
            keyed.append((
                (frame_index, 1, x, y, angle, extent_1, extent_2, width),
                token,
            ))
    keyed.sort(key=lambda item: item[0])
    observed = len(keyed)
    retained = min(observed, token_cap)
    tokens = np.zeros((token_cap, len(TOKEN_FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros(token_cap, dtype=np.bool_)
    if retained:
        tokens[:retained] = np.asarray(
            [item[1] for item in keyed[:retained]], dtype=np.float32
        )
        mask[:retained] = True
    if not np.all(np.isfinite(tokens)):
        raise EpisodeDatasetError("primitive token projection is non-finite")
    return tokens, mask, observed, observed - retained


def _load_primitive_episode(
    run_dir: Path,
    *,
    episode_index: int,
    horizons: tuple[int, ...],
    token_cap: int,
) -> tuple[str, dict[str, object], dict[int, list[tuple[object, ...]]]]:
    epochs = tuple(iter_decision_epochs(run_dir))
    facts = tuple(_transition_fact(row) for row in iter_episode_transitions(run_dir))
    if not epochs or not facts or any(
        fact.sequence != index for index, fact in enumerate(facts)
    ):
        raise EpisodeDatasetError("primitive episode is empty or non-contiguous")
    episode_id = epochs[0].episode_id
    if any(epoch.episode_id != episode_id for epoch in epochs):
        raise EpisodeDatasetError("primitive episode exposed multiple identities")
    wanted = {
        int(epoch.start_sequence)
        for epoch in epochs
        if epoch.learning_eligible and epoch.published_action is not None
    }
    projected: dict[int, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for frame in iter_episode_frames(run_dir):
        if frame.sequence in wanted:
            if frame.sequence in projected:
                raise EpisodeDatasetError("duplicate primitive policy root")
            projected[frame.sequence] = primitive_tokens_from_frame(
                frame, token_cap=token_cap
            )
    if set(projected) != wanted:
        raise EpisodeDatasetError("primitive roots do not match decision epochs")

    prior_hit_distances = _prior_hit_distances(facts)
    rows = {horizon: [] for horizon in horizons}
    for epoch in epochs:
        if not epoch.learning_eligible or epoch.published_action is None:
            continue
        start = int(epoch.start_sequence)
        current = action_conditioned_probe_features(
            epoch.observation, epoch.published_action
        )
        scalar = normalized_scalar_features(current)
        tokens, mask, token_count, truncated = projected[start]
        if epoch.baseline_action is None:
            raise EpisodeDatasetError("eligible primitive row lacks baseline action")
        probability_map = dict(epoch.behavior_probabilities)
        probability = float(epoch.behavior_probability)
        if (
            not math.isfinite(probability)
            or probability_map.get(epoch.published_action) != probability
        ):
            raise EpisodeDatasetError("primitive propensity disagrees with publication")
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
            rows[horizon].append((
                episode_index,
                current,
                scalar,
                tokens,
                mask,
                token_count,
                truncated,
                any(fact.hit for fact in window),
                epoch.published_action,
                epoch.baseline_action,
                probability,
                _lifecycle_stratum(prior_hit_distances[start]),
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


def _load_primitive_episode_task(task):
    path, episode_index, horizons, token_cap = task
    return _load_primitive_episode(
        path,
        episode_index=episode_index,
        horizons=horizons,
        token_cap=token_cap,
    )


def load_primitive_probe_dataset(
    run_dirs: Iterable[Path],
    *,
    horizons: tuple[int, ...],
    token_cap: int,
    max_rows: int = 2_000_000,
    workers: int = 1,
) -> PrimitiveProbeDataset:
    """Load factual h16 rows with one bounded observed-primitive set."""
    paths = tuple(Path(path).resolve() for path in run_dirs)
    if not paths or token_cap <= 0 or not 1 <= workers <= 32:
        raise ValueError("primitive dataset requires episodes and a positive cap")
    if (
        not horizons
        or tuple(sorted(set(horizons))) != horizons
        or any(not isinstance(horizon, int) or horizon <= 0 for horizon in horizons)
    ):
        raise ValueError("primitive horizons must be increasing positive integers")
    episode_ids = []
    inventory = []
    accumulated = {horizon: [] for horizon in horizons}
    tasks = tuple(
        (path, episode_index, horizons, token_cap)
        for episode_index, path in enumerate(paths)
    )

    def consume(results) -> None:
        for episode_id, episode_inventory, episode_rows in results:
            if episode_id in episode_ids:
                raise ValueError(f"duplicate primitive episode identity {episode_id}")
            episode_ids.append(episode_id)
            inventory.append(episode_inventory)
            for horizon in horizons:
                accumulated[horizon].extend(episode_rows[horizon])
                if len(accumulated[horizon]) > max_rows:
                    raise ValueError("primitive view exceeds its row limit")

    if workers == 1:
        consume(map(_load_primitive_episode_task, tasks))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=context
        ) as executor:
            # executor.map preserves the frozen episode order.
            consume(executor.map(_load_primitive_episode_task, tasks))

    views = []
    for horizon in horizons:
        rows = accumulated[horizon]
        if not rows:
            raise ValueError("primitive dataset contains an empty horizon")
        current = np.asarray([row[1] for row in rows], dtype=np.float64)
        scalar = np.asarray([row[2] for row in rows], dtype=np.float32)
        tokens = np.stack([row[3] for row in rows])
        masks = np.stack([row[4] for row in rows])
        labels = np.asarray([row[7] for row in rows], dtype=np.bool_)
        episode_indices = np.asarray([row[0] for row in rows], dtype=np.int64)
        if (
            current.shape[1] != len(PROBE_FEATURE_NAMES)
            or scalar.shape != current.shape
            or tokens.shape != (len(rows), token_cap, len(TOKEN_FEATURE_NAMES))
            or masks.shape != (len(rows), token_cap)
            or set(map(int, episode_indices)) != set(range(len(episode_ids)))
            or not np.all(np.isfinite(current))
            or not np.all(np.isfinite(scalar))
            or not np.all(np.isfinite(tokens))
            or np.any(tokens[~masks] != 0.0)
        ):
            raise EpisodeDatasetError("primitive tensor contract changed")
        views.append(PrimitiveHorizonDataset(
            horizon,
            token_cap,
            tuple(episode_ids),
            episode_indices,
            current,
            scalar,
            tokens,
            masks,
            np.asarray([row[5] for row in rows], dtype=np.int64),
            np.asarray([row[6] for row in rows], dtype=np.int64),
            labels,
            tuple(str(row[8]) for row in rows),
            tuple(None if row[9] is None else str(row[9]) for row in rows),
            np.asarray([row[10] for row in rows], dtype=np.float64),
            tuple(str(row[11]) for row in rows),
        ))
    return PrimitiveProbeDataset(
        tuple(episode_ids),
        tuple(inventory),
        PRIMITIVE_FEATURE_SCHEMA,
        PROBE_FEATURE_NAMES,
        TOKEN_FEATURE_NAMES,
        tuple(views),
    )
