"""TH06 projection into the game-neutral trajectory-profile interface."""

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
    "profile_action_fraction",
    "profile_mean_clearance_h1",
    "profile_best_clearance_h1",
    "profile_unbounded_fraction_h1",
    "profile_mean_clearance_h2",
    "profile_best_clearance_h2",
    "profile_unbounded_fraction_h2",
    "profile_mean_clearance_h3",
    "profile_best_clearance_h3",
    "profile_unbounded_fraction_h3",
    "profile_mean_clearance_h4",
    "profile_best_clearance_h4",
    "profile_unbounded_fraction_h4",
    "profile_mean_clearance_h6",
    "profile_best_clearance_h6",
    "profile_unbounded_fraction_h6",
    "profile_mean_clearance_h8",
    "profile_best_clearance_h8",
    "profile_unbounded_fraction_h8",
    "profile_mean_clearance_h10",
    "profile_best_clearance_h10",
    "profile_unbounded_fraction_h10",
    "profile_mean_clearance_h12",
    "profile_best_clearance_h12",
    "profile_unbounded_fraction_h12",
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
    "profile_clearance_h1",
    "profile_unbounded_h1",
    "profile_rank_h1",
    "profile_viable_h1",
    "profile_clearance_h2",
    "profile_unbounded_h2",
    "profile_rank_h2",
    "profile_viable_h2",
    "profile_clearance_h3",
    "profile_unbounded_h3",
    "profile_rank_h3",
    "profile_viable_h3",
    "profile_clearance_h4",
    "profile_unbounded_h4",
    "profile_rank_h4",
    "profile_viable_h4",
    "profile_clearance_h6",
    "profile_unbounded_h6",
    "profile_rank_h6",
    "profile_viable_h6",
    "profile_clearance_h8",
    "profile_unbounded_h8",
    "profile_rank_h8",
    "profile_viable_h8",
    "profile_clearance_h10",
    "profile_unbounded_h10",
    "profile_rank_h10",
    "profile_viable_h10",
    "profile_clearance_h12",
    "profile_unbounded_h12",
    "profile_rank_h12",
    "profile_viable_h12",
)
PROFILE_CHECKPOINTS = (1, 2, 3, 4, 6, 8, 10, 12)
LEFT = 8.0
RIGHT = 376.0
TOP = 16.0
BOTTOM = 432.0
WIDTH = RIGHT - LEFT
HEIGHT = BOTTOM - TOP
RESERVE_SCALE = min(WIDTH, HEIGHT)
ACTION_CAPACITY = 18.0
NATIVE_COLLISION_MARGIN = 0.35


def _edge_reserve(x: float, y: float) -> float:
    return min(x - LEFT, RIGHT - x, y - TOP, BOTTOM - y)


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _profile_value(value: float) -> tuple[float, float]:
    return (
        (0.0, 1.0)
        if math.isinf(value) and value > 0.0
        else (_signed_log1p(value), 0.0)
    )


def project_learning_features(
    snapshot,
    hard_evaluations,
    locally_admissible_actions: tuple[str, ...],
    effort_horizon: int,
    action_profiles,
):
    """Return bounded native trajectory evidence, never an eligibility gate."""
    profiles = {item.action.name: item for item in action_profiles}
    hard_names = tuple(item.action.name for item in hard_evaluations)
    if set(profiles) != set(hard_names):
        raise ValueError("native trajectory profiles must cover the exact Hard set")
    if any(tuple(item.checkpoints) != PROFILE_CHECKPOINTS for item in profiles.values()):
        raise ValueError("native trajectory profile checkpoints drifted")
    by_checkpoint = {
        checkpoint: tuple(
            profiles[name].min_clearances[index] for name in hard_names
        )
        for index, checkpoint in enumerate(PROFILE_CHECKPOINTS)
    }
    summary = []
    for checkpoint in PROFILE_CHECKPOINTS:
        values = by_checkpoint[checkpoint]
        finite = tuple(value for value in values if math.isfinite(value))
        encoded = tuple(_signed_log1p(value) for value in finite)
        summary.extend((
            (
                f"profile_mean_clearance_h{checkpoint}",
                sum(encoded) / len(encoded) if encoded else 0.0,
            ),
            (
                f"profile_best_clearance_h{checkpoint}",
                max(encoded) if encoded else 0.0,
            ),
            (
                f"profile_unbounded_fraction_h{checkpoint}",
                (len(values) - len(finite)) / len(values),
            ),
        ))
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
        ("profile_action_fraction", len(profiles) / ACTION_CAPACITY),
        *summary,
    )
    actions = []
    for evaluation in hard_evaluations:
        action = evaluation.action
        clearance = float(evaluation.min_clearance)
        clearance_unbounded = not math.isfinite(clearance)
        profile = profiles[action.name]
        profile_features = []
        for index, checkpoint in enumerate(PROFILE_CHECKPOINTS):
            value = float(profile.min_clearances[index])
            encoded, profile_unbounded = _profile_value(value)
            peers = by_checkpoint[checkpoint]
            rank = (
                sum(peer < value for peer in peers) / max(1, len(peers) - 1)
                if len(peers) > 1
                else 1.0
            )
            profile_features.extend((
                (f"profile_clearance_h{checkpoint}", encoded),
                (f"profile_unbounded_h{checkpoint}", profile_unbounded),
                (f"profile_rank_h{checkpoint}", rank),
                (
                    f"profile_viable_h{checkpoint}",
                    float(value > NATIVE_COLLISION_MARGIN),
                ),
            ))
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
                    0.0 if clearance_unbounded else _signed_log1p(clearance),
                ),
                ("clearance_unbounded", float(clearance_unbounded)),
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
                *profile_features,
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
