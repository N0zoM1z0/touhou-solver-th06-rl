"""TH06 adapter projection for the game-neutral autonomous learner."""

from __future__ import annotations

import math


OBSERVATION_FEATURE_NAMES = (
    "position_x_unit",
    "position_y_unit",
    "edge_reserve_unit",
    "bullet_count_log",
    "laser_count_log",
    "power_log",
    "hard_action_fraction",
    "local_action_fraction",
    "effort_horizon_log",
)
ACTION_FEATURE_NAMES = (
    "direction_x",
    "direction_y",
    "focused",
    "stationary",
    "diagonal",
    "clearance_log",
    "clearance_unbounded",
    "terminal_delta_x_unit",
    "terminal_delta_y_unit",
    "terminal_edge_reserve_unit",
)
LEFT = 8.0
RIGHT = 376.0
TOP = 16.0
BOTTOM = 432.0
WIDTH = RIGHT - LEFT
HEIGHT = BOTTOM - TOP
RESERVE_SCALE = min(WIDTH, HEIGHT)
ACTION_CAPACITY = 18.0


def _edge_reserve(x: float, y: float) -> float:
    return min(x - LEFT, RIGHT - x, y - TOP, BOTTOM - y)


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def project_learning_features(
    snapshot,
    hard_evaluations,
    locally_admissible_actions: tuple[str, ...],
    effort_horizon: int,
):
    """Return stable named scalars; no scalar is an exploration eligibility gate."""
    observation = (
        ("position_x_unit", (float(snapshot.x) - LEFT) / WIDTH),
        ("position_y_unit", (float(snapshot.y) - TOP) / HEIGHT),
        (
            "edge_reserve_unit",
            _edge_reserve(float(snapshot.x), float(snapshot.y)) / RESERVE_SCALE,
        ),
        ("bullet_count_log", math.log1p(int(snapshot.live_bullet_count))),
        ("laser_count_log", math.log1p(int(snapshot.laser_count))),
        ("power_log", math.log1p(max(0, int(snapshot.current_power)))),
        ("hard_action_fraction", len(hard_evaluations) / ACTION_CAPACITY),
        (
            "local_action_fraction",
            len(locally_admissible_actions) / ACTION_CAPACITY,
        ),
        ("effort_horizon_log", math.log1p(max(0, int(effort_horizon)))),
    )
    actions = []
    for evaluation in hard_evaluations:
        action = evaluation.action
        clearance = float(evaluation.min_clearance)
        unbounded = not math.isfinite(clearance)
        actions.append((
            action.name,
            (
                ("direction_x", float(action.dx)),
                ("direction_y", float(action.dy)),
                ("focused", float(action.focused)),
                ("stationary", float(action.dx == 0 and action.dy == 0)),
                ("diagonal", float(action.dx != 0 and action.dy != 0)),
                (
                    "clearance_log",
                    0.0 if unbounded else _signed_log1p(clearance),
                ),
                ("clearance_unbounded", float(unbounded)),
                (
                    "terminal_delta_x_unit",
                    (float(evaluation.final_x) - float(snapshot.x)) / WIDTH,
                ),
                (
                    "terminal_delta_y_unit",
                    (float(evaluation.final_y) - float(snapshot.y)) / HEIGHT,
                ),
                (
                    "terminal_edge_reserve_unit",
                    _edge_reserve(
                        float(evaluation.final_x), float(evaluation.final_y)
                    )
                    / RESERVE_SCALE,
                ),
            ),
        ))
    if tuple(name for name, _ in observation) != OBSERVATION_FEATURE_NAMES:
        raise RuntimeError("TH06 observation adapter schema drift")
    if any(
        tuple(name for name, _ in values) != ACTION_FEATURE_NAMES
        for _action, values in actions
    ):
        raise RuntimeError("TH06 action adapter schema drift")
    return observation, tuple(actions)
