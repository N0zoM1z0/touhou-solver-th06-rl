"""Feature and reward construction for CPU-only offline experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
from typing import Iterable

from .offline import (
    ACTION_NAMES,
    ACTION_SET,
    HIT_HORIZONS,
    RunDescriptor,
    iter_run_transitions,
)


FEATURE_SCHEMA = "th06-rl-offline-feature-v1"
LABEL_SCHEMA = "th06-rl-offline-label-survival-reserve-hit-trace-v2"
HIT_CREDIT_DISCOUNT = 0.97
HIT_CREDIT_PENALTY = 100.0
HIT_CREDIT_HORIZON_FRAMES = 120
_FRAME_RE = re.compile(r":f(\d+)$")
_NUMBER_RE = re.compile(r"-?\d+")
_ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}

CATEGORICAL_FEATURES = (
    "source_context",
    "action",
    "baseline_action",
    "current_action",
    "legal_mask",
    "hard_mask",
    "context_quality",
    "transition_schema",
)
NUMERIC_FEATURES = (
    "player_x",
    "player_y",
    "edge_reserve",
    "power",
    "bullet_count",
    "laser_count",
    "hard_action_count",
    "legal_action_count",
    "phase_elapsed_frames",
    "action_dx",
    "action_dy",
    "action_focused",
    "action_stationary",
    "action_diagonal",
    "baseline_dx",
    "baseline_dy",
    "baseline_focused",
    "matches_baseline",
    "matches_current",
    "phase_number_0",
    "phase_number_1",
    "phase_number_2",
    "phase_number_3",
    "phase_number_4",
    "phase_number_5",
)
FEATURE_NAMES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)


@dataclass(frozen=True)
class LabeledTransition:
    run_id: str
    sequence: int
    frame: int
    source_context: str
    action: str
    baseline_action: str
    legal_actions: tuple[str, ...]
    behavior_probability: float
    features: dict[str, str | float]
    reward: float
    hit_within_30: bool = False
    hit_within_60: bool = False
    hit_within_120: bool = False

    def hit_within(self, horizon: int) -> bool:
        return bool(getattr(self, f"hit_within_{horizon}"))


def _frame(value: object, fallback: int) -> int:
    match = _FRAME_RE.search(str(value))
    return int(match.group(1)) if match else fallback


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _number(mapping: dict[str, object], name: str, default: float = -1.0) -> float:
    try:
        value = mapping.get(name, default)
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _action_components(name: str) -> tuple[float, float, float]:
    core = name.removesuffix("_fast")
    return (
        float("right" in core) - float("left" in core),
        float("down" in core) - float("up" in core),
        float(not name.endswith("_fast")),
    )


def _mask(actions: Iterable[str]) -> str:
    value = 0
    for action in actions:
        index = _ACTION_INDEX.get(action)
        if index is not None:
            value |= 1 << index
    return f"{value:05x}"


def _edge(x: float, y: float) -> float:
    return min(x - 8.0, 376.0 - x, y - 16.0, 432.0 - y)


def _immediate_reward(outcome: dict[str, object]) -> float:
    reward = 1.0
    if outcome.get("bomb_used"):
        reward -= 100.0
    if outcome.get("control_dead_end"):
        reward -= 25.0
    if outcome.get("authority_lost"):
        reward -= 25.0
    hard_after = int(outcome.get("hard_count_after", -1))
    if hard_after >= 0:
        reward += 0.25 * min(1.0, hard_after / 18.0)
    x_after = _number(outcome, "player_x_after")
    y_after = _number(outcome, "player_y_after")
    if x_after >= 0.0 and y_after >= 0.0:
        reward += 0.10 * max(0.0, min(1.0, _edge(x_after, y_after) / 32.0))
    if outcome.get("phase_changed") and not any(
        outcome.get(name)
        for name in ("life_lost", "bomb_used", "control_dead_end", "authority_lost")
    ):
        reward += 5.0
    return reward


def _candidate_features(
    row: dict[str, object],
    *,
    phase_elapsed: int,
    previous_action: str,
    action: str,
) -> dict[str, str | float]:
    scope = _mapping(row.get("scope"))
    outcome = _mapping(row.get("outcome_terms"))
    policy = _mapping(row.get("policy_context"))
    source_context = str(scope.get("phase_id", "unknown"))
    legal_raw = row.get("legal_actions")
    legal = tuple(str(item) for item in legal_raw) if isinstance(legal_raw, list) else ()
    hard_raw = policy.get("hard_admissible_actions")
    hard = tuple(str(item) for item in hard_raw) if isinstance(hard_raw, list) else ()
    exact = bool(policy)
    player_x = _number(policy, "player_x", _number(outcome, "player_x_before"))
    player_y = _number(policy, "player_y", _number(outcome, "player_y_before"))
    baseline = str(row.get("baseline_action") or "unknown")
    current = str(policy.get("current_action") or previous_action or "unknown")
    action_dx, action_dy, action_focused = _action_components(action)
    baseline_dx, baseline_dy, baseline_focused = _action_components(baseline)
    phase_numbers = [float(value) for value in _NUMBER_RE.findall(source_context)[:6]]
    phase_numbers.extend([-1.0] * (6 - len(phase_numbers)))
    features: dict[str, str | float] = {
        "source_context": source_context,
        "action": action,
        "baseline_action": baseline,
        "current_action": current,
        "legal_mask": _mask(legal),
        "hard_mask": _mask(hard) if exact else "unknown",
        "context_quality": "exact-v5" if exact else "common-derived",
        "transition_schema": str(row.get("schema_version", "unknown")),
        "player_x": player_x,
        "player_y": player_y,
        "edge_reserve": _edge(player_x, player_y) if player_x >= 0.0 and player_y >= 0.0 else -1.0,
        "power": _number(policy, "power"),
        "bullet_count": _number(policy, "bullet_count"),
        "laser_count": _number(policy, "laser_count"),
        "hard_action_count": _number(
            policy,
            "hard_action_count",
            _number(outcome, "hard_count_before"),
        ),
        "legal_action_count": float(len(legal)),
        "phase_elapsed_frames": _number(policy, "phase_elapsed_frames", float(phase_elapsed)),
        "action_dx": action_dx,
        "action_dy": action_dy,
        "action_focused": action_focused,
        "action_stationary": float(action_dx == 0.0 and action_dy == 0.0),
        "action_diagonal": float(action_dx != 0.0 and action_dy != 0.0),
        "baseline_dx": baseline_dx,
        "baseline_dy": baseline_dy,
        "baseline_focused": baseline_focused,
        "matches_baseline": float(action == baseline),
        "matches_current": float(action == current),
    }
    features.update({f"phase_number_{index}": value for index, value in enumerate(phase_numbers)})
    return features


def features_for_candidate(row: LabeledTransition, action: str) -> dict[str, str | float]:
    if action not in row.legal_actions:
        raise ValueError("candidate action is outside the recorded native safe set")
    features = dict(row.features)
    dx, dy, focused = _action_components(action)
    features.update({
        "action": action,
        "action_dx": dx,
        "action_dy": dy,
        "action_focused": focused,
        "action_stationary": float(dx == 0.0 and dy == 0.0),
        "action_diagonal": float(dx != 0.0 and dy != 0.0),
        "matches_baseline": float(action == row.baseline_action),
        "matches_current": float(action == features["current_action"]),
    })
    return features


def label_transitions(
    raw_rows: Iterable[dict[str, object]],
    run: RunDescriptor,
    *,
    exact_context_only: bool,
) -> list[LabeledTransition]:
    labeled: list[LabeledTransition] = []
    trace: deque[tuple[int, str, int]] = deque()
    phase = None
    phase_start_frame = 0
    previous_action = "stay"
    for row in raw_rows:
        sequence = int(row.get("sequence", -1))
        frame = _frame(row.get("snapshot_ref"), sequence)
        scope = _mapping(row.get("scope"))
        source_context = str(scope.get("phase_id", "unknown"))
        if source_context != phase:
            phase = source_context
            phase_start_frame = frame
        policy = _mapping(row.get("policy_context"))
        outcome = _mapping(row.get("outcome_terms"))
        legal_raw = row.get("legal_actions")
        legal = tuple(str(item) for item in legal_raw) if isinstance(legal_raw, list) else ()
        action = row.get("published_action")
        proposal = row.get("proposed_action")
        action = str(action) if action is not None else None
        proposal = str(proposal) if proposal is not None else None
        try:
            propensity = float(row.get("behavior_probability", 0.0))
        except (TypeError, ValueError):
            propensity = 0.0
        trainable = bool(
            run.training_eligible
            and row.get("learning_eligible") is True
            and action in ACTION_SET
            and action in legal
            and proposal == action
            and not outcome.get("bomb_used")
            and 0.0 < propensity <= 1.0
            and (policy or not exact_context_only)
        )
        if trainable:
            assert action is not None
            entry = LabeledTransition(
                run_id=run.run_id,
                sequence=sequence,
                frame=frame,
                source_context=source_context,
                action=action,
                baseline_action=str(row.get("baseline_action") or "unknown"),
                legal_actions=legal,
                behavior_probability=propensity,
                features=_candidate_features(
                    row,
                    phase_elapsed=frame - phase_start_frame,
                    previous_action=previous_action,
                    action=action,
                ),
                reward=_immediate_reward(outcome),
            )
            labeled.append(entry)
            trace.append((frame, source_context, len(labeled) - 1))
        while trace and frame - trace[0][0] > HIT_CREDIT_HORIZON_FRAMES:
            trace.popleft()
        if outcome.get("life_lost"):
            hit_frame = _frame(row.get("next_snapshot_ref"), frame + 1)
            next_scope = _mapping(row.get("next_scope"))
            hit_context = str(next_scope.get("phase_id", source_context))
            for action_frame, action_context, index in trace:
                lag = hit_frame - action_frame
                if action_context != hit_context or not 0 <= lag <= HIT_CREDIT_HORIZON_FRAMES:
                    continue
                labels = {
                    f"hit_within_{horizon}": True
                    for horizon in HIT_HORIZONS
                    if lag <= horizon
                }
                labeled[index] = replace(
                    labeled[index],
                    reward=(
                        labeled[index].reward
                        - HIT_CREDIT_PENALTY * HIT_CREDIT_DISCOUNT ** lag
                    ),
                    **labels,
                )
            trace.clear()
        if action is not None:
            previous_action = action
    return labeled


def load_labeled_run(
    root: Path,
    run: RunDescriptor,
    *,
    exact_context_only: bool,
    verify_sha256: bool = False,
) -> list[LabeledTransition]:
    return label_transitions(
        iter_run_transitions(root, run, verify_sha256=verify_sha256),
        run,
        exact_context_only=exact_context_only,
    )


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("metric vectors must be nonempty and equal length")
    errors = [right - left for left, right in zip(actual, predicted, strict=True)]
    return {
        "mae": sum(abs(value) for value in errors) / len(errors),
        "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
    }
