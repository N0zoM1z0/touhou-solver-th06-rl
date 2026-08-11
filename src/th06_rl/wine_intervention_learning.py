"""Shared action-relative contract for Wine intervention fit and scoring."""

from __future__ import annotations

import math


FEATURE_SCHEMA = "th06-rl-wine-action-relative-v1"
FEATURE_NAMES = (
    "player_x",
    "player_y",
    "bullet_count",
    "hard_action_count",
    "local_action_count",
    "effort_horizon",
    "action_dx",
    "action_dy",
    "action_focused",
    "delta_dx_incumbent",
    "delta_dy_incumbent",
    "delta_focus_incumbent",
    "hard_clearance",
    "clearance_delta_incumbent",
    "final_edge_reserve",
    "edge_delta_incumbent",
    "matches_current",
    "matches_baseline",
)


def action_components(name: str) -> tuple[float, float, float]:
    core = name.removesuffix("_fast")
    return (
        float("right" in core) - float("left" in core),
        float("down" in core) - float("up" in core),
        float(not name.endswith("_fast")),
    )


def edge_reserve(x: float, y: float) -> float:
    return min(x - 8.0, 376.0 - x, y - 16.0, 432.0 - y)


def action_relative_features(
    *,
    player_x: float,
    player_y: float,
    bullet_count: int,
    hard_action_count: int,
    local_action_count: int,
    effort_horizon: int,
    current_action: str,
    baseline_action: str,
    action: str,
    incumbent_action: str,
    evaluations,
) -> dict[str, float]:
    decoded: dict[str, tuple[float, float, float]] = {}
    for raw in evaluations:
        if not isinstance(raw, (tuple, list)) or len(raw) != 4:
            raise TypeError("native action evaluation must have four fields")
        name = str(raw[0])
        if name in decoded or raw[1] is None:
            continue
        values = (float(raw[1]), float(raw[2]), float(raw[3]))
        if all(math.isfinite(value) for value in values):
            decoded[name] = values
    if action not in decoded or incumbent_action not in decoded:
        raise ValueError("selected/incumbent action lacks a finite Hard evaluation")
    clearance, final_x, final_y = decoded[action]
    inc_clearance, inc_x, inc_y = decoded[incumbent_action]
    dx, dy, focused = action_components(action)
    inc_dx, inc_dy, inc_focused = action_components(incumbent_action)
    final_edge = edge_reserve(final_x, final_y)
    incumbent_edge = edge_reserve(inc_x, inc_y)
    features = {
        "player_x": float(player_x),
        "player_y": float(player_y),
        "bullet_count": float(bullet_count),
        "hard_action_count": float(hard_action_count),
        "local_action_count": float(local_action_count),
        "effort_horizon": float(effort_horizon),
        "action_dx": dx,
        "action_dy": dy,
        "action_focused": focused,
        "delta_dx_incumbent": dx - inc_dx,
        "delta_dy_incumbent": dy - inc_dy,
        "delta_focus_incumbent": focused - inc_focused,
        "hard_clearance": clearance,
        "clearance_delta_incumbent": clearance - inc_clearance,
        "final_edge_reserve": final_edge,
        "edge_delta_incumbent": final_edge - incumbent_edge,
        "matches_current": float(action == current_action),
        "matches_baseline": float(action == baseline_action),
    }
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("Wine intervention feature schema mismatch")
    return features
