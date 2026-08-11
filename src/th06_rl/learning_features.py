"""Game-neutral construction of numeric state-action learner vectors."""

from __future__ import annotations

import math


FEATURE_SCHEMA = "safe-action-relative-interactions-v1"


def _feature_map(
    rows: tuple[tuple[str, float], ...] | list[list[object]],
    *,
    label: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw in rows:
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise TypeError(f"{label} feature row must contain name and value")
        name = str(raw[0])
        value = float(raw[1])
        if not name or name in values or not math.isfinite(value):
            raise ValueError(f"invalid or duplicate {label} feature {name!r}")
        values[name] = value
    if not values:
        raise ValueError(f"{label} features cannot be empty")
    return values


def feature_names(
    observation_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> tuple[str, ...]:
    if not observation_names or not action_names:
        raise ValueError("observation and action feature schemas must be nonempty")
    if len(set(observation_names)) != len(observation_names):
        raise ValueError("observation feature names contain duplicates")
    if len(set(action_names)) != len(action_names):
        raise ValueError("action feature names contain duplicates")
    return (
        *(f"observation:{name}" for name in observation_names),
        *(f"action:{name}" for name in action_names),
        *(f"delta_from_baseline:{name}" for name in action_names),
        "matches_baseline",
        "matches_current",
        *(
            f"observation_action:{observation}*{action}"
            for observation in observation_names
            for action in action_names
        ),
        *(
            f"observation_delta:{observation}*{action}"
            for observation in observation_names
            for action in action_names
        ),
    )


def candidate_vector(
    *,
    observation_features,
    action_features,
    action: str,
    baseline_action: str,
    current_action: str,
    observation_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> tuple[float, ...]:
    observation = _feature_map(observation_features, label="observation")
    if tuple(observation) != observation_names:
        raise ValueError("adapter observation feature schema mismatch")
    raw_actions = dict(action_features)
    if action not in raw_actions or baseline_action not in raw_actions:
        raise ValueError("candidate or baseline action features are absent")
    selected = _feature_map(raw_actions[action], label=f"action {action}")
    baseline = _feature_map(
        raw_actions[baseline_action], label=f"action {baseline_action}"
    )
    if tuple(selected) != action_names or tuple(baseline) != action_names:
        raise ValueError("adapter action feature schema mismatch")
    observation_values = [observation[name] for name in observation_names]
    action_values = [selected[name] for name in action_names]
    delta_values = [
        selected[name] - baseline[name] for name in action_names
    ]
    return tuple((
        *observation_values,
        *action_values,
        *delta_values,
        float(action == baseline_action),
        float(action == current_action),
        *(
            observation_value * action_value
            for observation_value in observation_values
            for action_value in action_values
        ),
        *(
            observation_value * delta_value
            for observation_value in observation_values
            for delta_value in delta_values
        ),
    ))
