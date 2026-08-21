"""Pure-Python, actor-portable features for the first behavior clone."""

from __future__ import annotations

import math
from typing import Protocol

from .actions import ACTION_NAMES


FEATURE_SCHEMA = "th06-rl-current-observation-features-v1"
STATE_FEATURE_NAMES = (
    "player_x",
    "player_y",
    "power",
    "bullet_count",
    "laser_count",
    "shield_action_count",
)
FEATURE_NAMES = (
    *STATE_FEATURE_NAMES,
    *(f"current_action:{action}" for action in ACTION_NAMES),
    *(
        name
        for action in ACTION_NAMES
        for name in (
            f"shield_legal:{action}",
            f"shield_clearance_known:{action}",
            f"shield_clearance:{action}",
            f"shield_final_x:{action}",
            f"shield_final_y:{action}",
        )
    ),
)


class PortableRootLike(Protocol):
    player_x: float
    player_y: float
    power: int
    bullet_count: int
    laser_count: int
    current_action: str
    locally_admissible_actions: tuple[str, ...]
    shield_action_evaluations: tuple[
        tuple[str, float | None, float, float], ...
    ]


def _features(
    *,
    player_x: float,
    player_y: float,
    power: int,
    bullet_count: int,
    laser_count: int,
    current_action: str,
    legal_actions: tuple[str, ...],
    evaluations: tuple[tuple[str, float | None, float, float], ...],
) -> tuple[float, ...]:
    if not math.isfinite(player_x) or not math.isfinite(player_y):
        raise ValueError("player position must be finite")
    if min(power, bullet_count, laser_count) < 0:
        raise ValueError("portable counters must be nonnegative")
    if current_action not in ACTION_NAMES:
        raise ValueError("current action is outside the canonical vocabulary")
    if (
        not legal_actions
        or len(set(legal_actions)) != len(legal_actions)
        or any(action not in ACTION_NAMES for action in legal_actions)
    ):
        raise ValueError("shield action set is empty, duplicated, or unknown")
    evaluation_map = {}
    for action, clearance, final_x, final_y in evaluations:
        if action in evaluation_map or action not in ACTION_NAMES:
            raise ValueError("shield evaluation action is duplicated or unknown")
        if (
            (clearance is not None and not math.isfinite(clearance))
            or not math.isfinite(final_x)
            or not math.isfinite(final_y)
        ):
            raise ValueError("shield evaluation is not finite")
        evaluation_map[action] = (clearance, final_x, final_y)
    if set(evaluation_map) != set(legal_actions):
        raise ValueError("shield evaluations disagree with the legal action set")

    result = [
        float(player_x),
        float(player_y),
        float(power),
        float(bullet_count),
        float(laser_count),
        float(len(legal_actions)),
    ]
    result.extend(float(current_action == action) for action in ACTION_NAMES)
    for action in ACTION_NAMES:
        evaluation = evaluation_map.get(action)
        if evaluation is None:
            result.extend((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        clearance, final_x, final_y = evaluation
        result.extend((
            1.0,
            float(clearance is not None),
            0.0 if clearance is None else float(clearance),
            float(final_x),
            float(final_y),
        ))
    if len(result) != len(FEATURE_NAMES) or any(
        not math.isfinite(value) for value in result
    ):
        raise ValueError("feature vector does not satisfy its frozen schema")
    return tuple(result)


def features_from_portable_root(root: PortableRootLike) -> tuple[float, ...]:
    """Build offline features without reading source, stage, or future facts."""
    return _features(
        player_x=float(root.player_x),
        player_y=float(root.player_y),
        power=int(root.power),
        bullet_count=int(root.bullet_count),
        laser_count=int(root.laser_count),
        current_action=str(root.current_action),
        legal_actions=tuple(root.locally_admissible_actions),
        evaluations=tuple(root.shield_action_evaluations),
    )


def features_from_policy_context(context) -> tuple[float, ...]:
    """Build the identical vector from the bounded online PolicyContext."""
    legal = tuple(str(action) for action in context.locally_admissible_actions)
    shield_legal = tuple(str(action) for action in context.shield_admissible_actions)
    evaluations = tuple(
        (
            str(row[0]),
            None if row[1] is None else float(row[1]),
            float(row[2]),
            float(row[3]),
        )
        for row in context.shield_action_evaluations
    )
    if shield_legal != legal:
        raise ValueError("online local and shield action sets disagree")
    if int(context.shield_action_count) != len(legal):
        raise ValueError("online shield action count disagrees with its set")
    # baseline_action is intentionally excluded: the L1 behavior target is an
    # 80/20 mixture around that control, so exposing it would make the
    # learnability test tautological rather than testing physical features.
    return _features(
        player_x=float(context.player_x),
        player_y=float(context.player_y),
        power=int(context.power),
        bullet_count=int(context.bullet_count),
        laser_count=int(context.laser_count),
        current_action=str(context.current_action),
        legal_actions=legal,
        evaluations=evaluations,
    )


def normalized_features(
    features: tuple[float, ...],
    mean: tuple[float, ...],
    scale: tuple[float, ...],
) -> tuple[float, ...]:
    if not (len(features) == len(mean) == len(scale) == len(FEATURE_NAMES)):
        raise ValueError("normalization vector length disagrees with feature schema")
    if any(not math.isfinite(value) for value in (*features, *mean, *scale)):
        raise ValueError("normalization inputs must be finite")
    if any(value <= 0.0 for value in scale):
        raise ValueError("normalization scale must be positive")
    return tuple(
        (value - center) / width
        for value, center, width in zip(features, mean, scale, strict=True)
    )


def linear_action_scores(
    features: tuple[float, ...],
    weights: tuple[tuple[float, ...], ...],
    biases: tuple[float, ...],
) -> tuple[float, ...]:
    if len(weights) != len(ACTION_NAMES) or len(biases) != len(ACTION_NAMES):
        raise ValueError("linear output does not cover the canonical action set")
    scores = []
    for row, bias in zip(weights, biases, strict=True):
        if len(row) != len(features):
            raise ValueError("linear weight width disagrees with feature schema")
        score = float(bias) + sum(
            weight * value for weight, value in zip(row, features, strict=True)
        )
        if not math.isfinite(score):
            raise ValueError("linear action score is not finite")
        scores.append(score)
    return tuple(scores)


def masked_softmax_probabilities(
    scores: tuple[float, ...],
    legal_actions: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    """Convert canonical logits to a complete distribution over one shield set."""
    if len(scores) != len(ACTION_NAMES) or any(
        not math.isfinite(score) for score in scores
    ):
        raise ValueError("softmax scores do not cover the canonical action set")
    if (
        not legal_actions
        or len(set(legal_actions)) != len(legal_actions)
        or any(action not in ACTION_NAMES for action in legal_actions)
    ):
        raise ValueError("softmax shield action set is empty, duplicated, or unknown")
    score_by_action = dict(zip(ACTION_NAMES, scores, strict=True))
    maximum = max(score_by_action[action] for action in legal_actions)
    exponentials = tuple(
        math.exp(score_by_action[action] - maximum) for action in legal_actions
    )
    total = sum(exponentials)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("softmax normalization is invalid")
    return tuple(
        (action, value / total)
        for action, value in zip(legal_actions, exponentials, strict=True)
    )
