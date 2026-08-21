"""Episode-calibrated local statistical support for a Generation-7 actor."""

from __future__ import annotations

import math

from th06_rl.actions import ACTION_NAMES
from th06_rl.g7_learner import (
    ACTOR_FEATURE_NAMES,
    CriticDataset,
    _actor_vector_from_critic,
    _actor_vector,
    whole_episode_folds,
)
from th06_rl.learning_features import CAUSAL_TREE_FEATURE_SCHEMA
from th06_rl.offline_options import ActorState


SUPPORT_SCHEMA = "th06-rl-g7-local-prototype-support-v2"
CALIBRATION_UNIT = "physical-episode-maximum"
CALIBRATION_QUANTILE = "split-conformal-ceil-(n+1)q"


def _nearest_distance(row, prototypes) -> float:
    return min(
        sum((left - right) ** 2 for left, right in zip(row, prototype, strict=True))
        / len(row)
        for prototype in prototypes
    )


def _kmeans(rows, *, count: int, seed: int):
    import numpy as np

    matrix = np.asarray(rows, dtype=np.float64)
    count = min(count, len(matrix))
    order = sorted(
        range(len(matrix)),
        key=lambda index: (
            sum(matrix[index] * matrix[index]),
            hashlib_sha(seed, index),
        ),
    )
    initial = [order[index * len(order) // count] for index in range(count)]
    centers = matrix[initial].copy()
    for _ in range(64):
        distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
        assignments = distances.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(count):
            members = matrix[assignments == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centers, rtol=0.0, atol=1e-10):
            centers = updated
            break
        centers = updated
    return tuple(tuple(float(value) for value in row) for row in centers)


def hashlib_sha(seed: int, index: int) -> bytes:
    import hashlib

    return hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()


def fit_local_support(
    dataset: CriticDataset,
    *,
    seed: int,
    prototypes_per_action: int = 16,
    calibration_fraction: float = 0.2,
    distance_quantile: float = 0.99,
    minimum_samples: int = 32,
    minimum_ess: float = 16.0,
) -> dict[str, object]:
    """Fit prototypes on episodes disjoint from distance calibration episodes."""
    if (
        prototypes_per_action < 1
        or not 0.0 < calibration_fraction < 1.0
        or not 0.5 <= distance_quantile < 1.0
        or minimum_samples < 1
        or minimum_ess <= 0.0
    ):
        raise ValueError("local support bounds are invalid")
    if len(dataset.episode_ids) < 2:
        raise ValueError("local support requires at least two physical episodes")
    import numpy as np

    calibration_folds = max(2, min(
        len(dataset.episode_ids),
        round(1.0 / calibration_fraction),
    ))
    fold_rows = whole_episode_folds(
        dataset.episode_ids,
        folds=calibration_folds,
        seed=seed,
    )
    fold_by_episode = dict(fold_rows)
    calibration_fold = 0
    fit_examples = [
        example for example in dataset.examples
        if fold_by_episode[example.episode_id] != calibration_fold
    ]
    calibration_examples = [
        example for example in dataset.examples
        if fold_by_episode[example.episode_id] == calibration_fold
    ]
    if not fit_examples or not calibration_examples:
        raise ValueError("support fit/calibration episode split is empty")
    fit_vectors = [
        _actor_vector_from_critic(
            dict(example.candidate_vectors)[example.factual_action]
        )
        for example in fit_examples
    ]
    matrix = np.asarray(fit_vectors, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0

    actions = {}
    for action in ACTION_NAMES:
        action_fit = [
            example for example in fit_examples
            if example.factual_action == action
        ]
        action_calibration = [
            example for example in calibration_examples
            if example.factual_action == action
        ]
        inverse_propensity_by_episode: dict[str, float] = {}
        for example in action_fit:
            inverse_propensity_by_episode[example.episode_id] = (
                inverse_propensity_by_episode.get(example.episode_id, 0.0)
                + 1.0 / example.behavior_probability
            )
        inverse_propensities = tuple(inverse_propensity_by_episode.values())
        ess = (
            sum(inverse_propensities) ** 2
            / sum(weight * weight for weight in inverse_propensities)
            if inverse_propensities
            else 0.0
        )
        supported = (
            len(action_fit) >= minimum_samples
            and ess >= minimum_ess
        )
        normalized_fit = [
            tuple((value - center) / width for value, center, width in zip(
                _actor_vector_from_critic(
                    dict(example.candidate_vectors)[action]
                ),
                mean,
                scale,
                strict=True,
            ))
            for example in action_fit
        ]
        prototypes = (
            _kmeans(
                normalized_fit,
                count=prototypes_per_action,
                seed=seed + ACTION_NAMES.index(action),
            )
            if normalized_fit
            else ()
        )
        calibration_distance_by_episode: dict[str, float] = {}
        for example in action_calibration:
            normalized = tuple(
                (value - center) / width
                for value, center, width in zip(
                    _actor_vector_from_critic(
                        dict(example.candidate_vectors)[action]
                    ),
                    mean,
                    scale,
                    strict=True,
                )
            )
            distance = _nearest_distance(normalized, prototypes) if prototypes else 0.0
            calibration_distance_by_episode[example.episode_id] = max(
                distance,
                calibration_distance_by_episode.get(example.episode_id, 0.0),
            )
        episode_distances = sorted(calibration_distance_by_episode.values())
        conformal_rank = math.ceil((len(episode_distances) + 1) * distance_quantile)
        supported = supported and conformal_rank <= len(episode_distances)
        threshold = episode_distances[conformal_rank - 1] if supported else None
        actions[action] = {
            "supported": supported,
            "fit_samples": len(action_fit),
            "calibration_samples": len(action_calibration),
            "fit_episodes": len(inverse_propensity_by_episode),
            "calibration_episodes": len(calibration_distance_by_episode),
            "episode_effective_sample_size": ess,
            "conformal_rank": conformal_rank,
            "distance_threshold": threshold,
            "prototypes": [list(row) for row in prototypes] if supported else [],
        }
    return {
        "schema": SUPPORT_SCHEMA,
        "feature_schema": CAUSAL_TREE_FEATURE_SCHEMA,
        "feature_names": list(ACTOR_FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "actions": actions,
        "fit_fold_by_episode": [list(row) for row in fold_rows],
        "calibration_fold": calibration_fold,
        "calibration_unit": CALIBRATION_UNIT,
        "calibration_quantile": CALIBRATION_QUANTILE,
        "distance_quantile": distance_quantile,
        "minimum_samples": minimum_samples,
        "minimum_effective_sample_size": minimum_ess,
    }


def locally_supported_actions(
    artifact: dict[str, object],
    state: ActorState,
) -> tuple[str, ...]:
    """Evaluate only statistical support; native safety remains external."""
    distance_quantile_value = artifact.get("distance_quantile")
    minimum_samples = artifact.get("minimum_samples")
    minimum_ess_value = artifact.get("minimum_effective_sample_size")
    if (
        artifact.get("schema") != SUPPORT_SCHEMA
        or artifact.get("feature_schema") != CAUSAL_TREE_FEATURE_SCHEMA
        or tuple(artifact.get("feature_names", ())) != ACTOR_FEATURE_NAMES
        or artifact.get("calibration_unit") != CALIBRATION_UNIT
        or artifact.get("calibration_quantile") != CALIBRATION_QUANTILE
        or not isinstance(distance_quantile_value, (int, float))
        or isinstance(distance_quantile_value, bool)
        or not isinstance(minimum_samples, int)
        or isinstance(minimum_samples, bool)
        or not isinstance(minimum_ess_value, (int, float))
        or isinstance(minimum_ess_value, bool)
    ):
        raise ValueError("local support artifact contract mismatch")
    distance_quantile = float(distance_quantile_value)
    minimum_ess = float(minimum_ess_value)
    mean = tuple(float(value) for value in artifact.get("mean", ()))
    scale = tuple(float(value) for value in artifact.get("scale", ()))
    actions = artifact.get("actions")
    if (
        len(mean) != len(ACTOR_FEATURE_NAMES)
        or len(scale) != len(mean)
        or any(not math.isfinite(value) for value in mean)
        or any(not math.isfinite(value) or value <= 0.0 for value in scale)
        or not isinstance(actions, dict)
        or set(actions) != set(ACTION_NAMES)
        or not 0.5 <= distance_quantile < 1.0
        or minimum_samples < 1
        or not math.isfinite(minimum_ess)
        or minimum_ess <= 0.0
    ):
        raise ValueError("local support numeric artifact is invalid")
    for specification in actions.values():
        if not isinstance(specification, dict):
            raise ValueError("local support action artifact is malformed")
        integer_names = (
            "fit_samples",
            "calibration_samples",
            "fit_episodes",
            "calibration_episodes",
            "conformal_rank",
        )
        integers = tuple(specification.get(name) for name in integer_names)
        ess_value = specification.get("episode_effective_sample_size")
        if (
            not isinstance(specification.get("supported"), bool)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in integers
            )
            or not isinstance(ess_value, (int, float))
            or isinstance(ess_value, bool)
        ):
            raise ValueError("local support action counts are invalid")
        fit_samples, calibration_samples, fit_episodes, calibration_episodes, rank = (
            integers
        )
        ess = float(ess_value)
        supported = specification["supported"]
        threshold = specification.get("distance_threshold")
        prototypes = specification.get("prototypes")
        if (
            fit_episodes > fit_samples
            or calibration_episodes > calibration_samples
            or rank != math.ceil((calibration_episodes + 1) * distance_quantile)
            or not math.isfinite(ess)
            or ess < 0.0
            or supported
            and (
                fit_samples < minimum_samples
                or ess < minimum_ess
                or rank > calibration_episodes
                or not isinstance(prototypes, list)
                or not prototypes
                or not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)
                or not math.isfinite(float(threshold))
                or float(threshold) < 0.0
            )
            or not supported and (threshold is not None or prototypes != [])
        ):
            raise ValueError("local support action evidence is inconsistent")
    result = []
    for action in state.legal_actions:
        specification = actions.get(action)
        if not isinstance(specification, dict) or specification.get("supported") is not True:
            continue
        prototypes = specification.get("prototypes")
        threshold = specification.get("distance_threshold")
        if not isinstance(prototypes, list) or not prototypes:
            raise ValueError("supported action lacks prototypes")
        converted = tuple(tuple(float(value) for value in row) for row in prototypes)
        if any(
            len(row) != len(mean)
            or any(not math.isfinite(value) for value in row)
            for row in converted
        ):
            raise ValueError("support prototype feature length mismatch")
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("support distance threshold is invalid")
        vector = _actor_vector(state, action)
        normalized = tuple(
            (value - center) / width
            for value, center, width in zip(vector, mean, scale, strict=True)
        )
        if _nearest_distance(normalized, converted) <= threshold + 1e-12:
            result.append(action)
    return tuple(result)
