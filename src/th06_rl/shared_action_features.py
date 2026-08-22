"""Portable per-action features for one shared current-observation scorer."""

from __future__ import annotations

import math

from .actions import ACTION_NAMES
from .bc_features import FEATURE_NAMES
from .core.model import movement_actions


ACTION_FEATURE_SCHEMA = "th06-rl-shared-action-features-v1"
ACTION_FEATURE_NAMES = (
    "clearance_unknown",
    "clearance",
    "boundary_reserve",
    "is_current_action",
    "is_stationary",
    "is_focused",
    "lexical_priority",
)

_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
_ACTIONS = movement_actions()
if tuple(action.name for action in _ACTIONS) != ACTION_NAMES:
    raise RuntimeError("shared-action feature vocabulary differs from movement actions")
_LEXICAL_PRIORITY = {
    name: rank for rank, name in enumerate(sorted(ACTION_NAMES))
}


def action_feature_rows(
    features: tuple[float, ...],
    legal_actions: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    """Project one frozen 114-vector into one transparent row per action."""
    if len(features) != len(FEATURE_NAMES) or any(
        not math.isfinite(value) for value in features
    ):
        raise ValueError("shared-action input does not satisfy the frozen schema")
    if (
        not legal_actions
        or len(set(legal_actions)) != len(legal_actions)
        or any(action not in ACTION_NAMES for action in legal_actions)
    ):
        raise ValueError("shared-action legal set is empty, duplicated, or unknown")
    legal = frozenset(legal_actions)
    rows = []
    for action in _ACTIONS:
        name = action.name
        known = features[_FEATURE_INDEX[f"shield_clearance_known:{name}"]]
        shield_legal = features[_FEATURE_INDEX[f"shield_legal:{name}"]]
        if known not in (0.0, 1.0) or shield_legal not in (0.0, 1.0):
            raise ValueError("shared-action shield bits are not binary")
        if bool(shield_legal) != (name in legal):
            raise ValueError("shared-action legal set differs from frozen features")
        clearance = features[_FEATURE_INDEX[f"shield_clearance:{name}"]]
        final_x = features[_FEATURE_INDEX[f"shield_final_x:{name}"]]
        final_y = features[_FEATURE_INDEX[f"shield_final_y:{name}"]]
        boundary_reserve = min(
            final_x - 8.0,
            376.0 - final_x,
            final_y - 16.0,
            432.0 - final_y,
        )
        row = (
            1.0 - known,
            clearance,
            boundary_reserve,
            features[_FEATURE_INDEX[f"current_action:{name}"]],
            float(action.dx == 0 and action.dy == 0),
            float(action.focused),
            float(_LEXICAL_PRIORITY[name]),
        )
        if len(row) != len(ACTION_FEATURE_NAMES) or any(
            not math.isfinite(value) for value in row
        ):
            raise ValueError("derived shared-action row is invalid")
        rows.append(row)
    return tuple(rows)


def normalized_action_feature_rows(
    rows: tuple[tuple[float, ...], ...],
    mean: tuple[float, ...],
    scale: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    if (
        len(rows) != len(ACTION_NAMES)
        or len(mean) != len(ACTION_FEATURE_NAMES)
        or len(scale) != len(ACTION_FEATURE_NAMES)
        or any(len(row) != len(ACTION_FEATURE_NAMES) for row in rows)
        or any(value <= 0.0 for value in scale)
    ):
        raise ValueError("shared-action normalization dimensions differ")
    numeric = (*mean, *scale, *(value for row in rows for value in row))
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("shared-action normalization is not finite")
    return tuple(
        tuple(
            (value - center) / width
            for value, center, width in zip(row, mean, scale, strict=True)
        )
        for row in rows
    )


def shared_action_scores(
    rows: tuple[tuple[float, ...], ...],
    weights: tuple[float, ...],
) -> tuple[float, ...]:
    if (
        len(rows) != len(ACTION_NAMES)
        or len(weights) != len(ACTION_FEATURE_NAMES)
        or any(len(row) != len(weights) for row in rows)
    ):
        raise ValueError("shared-action scorer dimensions differ")
    scores = tuple(
        sum(weight * value for weight, value in zip(weights, row, strict=True))
        for row in rows
    )
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("shared-action score is not finite")
    return scores
