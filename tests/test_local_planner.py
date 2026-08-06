from __future__ import annotations

from dataclasses import dataclass
import math

from th06_rl.core import (
    Action,
    Bounds,
    CertifiedAction,
    HazardSample,
    Kinematics,
    LocalPlanner,
    LocalPlannerConfig,
    LocalPlanningRequest,
    movement_actions,
)


ACTIONS = movement_actions()
BY_NAME = {action.name: action for action in ACTIONS}
BOUNDS = Bounds(8.0, 376.0, 16.0, 432.0)
KINEMATICS = Kinematics(4.0, 2.0, 2.828427, 1.414214)


@dataclass(frozen=True)
class MovingCircles:
    circles: tuple[tuple[float, float, float, float, float], ...]
    player_radius: float = 1.25
    known_through: int = 10_000

    def sample(self, x: float, y: float, frame: int) -> HazardSample:
        if frame > self.known_through:
            return HazardSample(False, 0, math.inf)
        clearance = BOUNDS.clearance(x, y)
        for start_x, start_y, vx, vy, radius in self.circles:
            hazard_x = start_x + vx * frame
            hazard_y = start_y + vy * frame
            clearance = min(
                clearance,
                math.hypot(x - hazard_x, y - hazard_y)
                - radius
                - self.player_radius,
            )
        return HazardSample(True, int(clearance < 0.0), clearance)


def request(
    hazards,
    *,
    x=192.0,
    y=380.0,
    previous="stay",
    hard=("stay", "left", "right"),
    continuation=("stay", "left", "right"),
):
    return LocalPlanningRequest(
        x=x,
        y=y,
        previous_action=BY_NAME[previous],
        hard_first_actions=tuple(
            CertifiedAction(BY_NAME[name], 100.0) for name in hard
        ),
        continuation_actions=tuple(BY_NAME[name] for name in continuation),
        kinematics=KINEMATICS,
        bounds=BOUNDS,
        hazards=hazards,
    )


def test_straight_incoming_bullet_is_dodged_without_learning():
    # The bullet reaches the initial player lane near the end of the horizon.
    # Staying is locally legal now, but moving horizontally creates a safe
    # continuation. No stage, phase, frame identity, or learned value exists.
    hazards = MovingCircles(((192.0, 344.0, 0.0, 3.0, 4.0),))
    planner = LocalPlanner(LocalPlannerConfig(
        horizon=12,
        action_hold_frames=2,
        beam_width=64,
    ))

    proposal = planner.propose(request(hazards))

    assert proposal.available
    assert proposal.effort_horizon == 12
    assert proposal.action in (BY_NAME["left"], BY_NAME["right"])
    assert all(
        evaluation.survived_frames == 12
        for evaluation in proposal.evaluations
    )


def test_uncertified_escape_direction_can_never_be_proposed():
    hazards = MovingCircles(((192.0, 344.0, 0.0, 3.0, 4.0),))
    planner = LocalPlanner(LocalPlannerConfig(horizon=12))

    proposal = planner.propose(request(
        hazards,
        hard=("stay", "left"),
        continuation=("stay", "left", "right"),
    ))

    assert proposal.action in (BY_NAME["stay"], BY_NAME["left"])
    assert BY_NAME["right"] not in {
        evaluation.action for evaluation in proposal.evaluations
    }


def test_boundary_reserve_prefers_inward_motion_in_an_open_field():
    planner = LocalPlanner(LocalPlannerConfig(
        horizon=8,
        action_hold_frames=2,
        boundary_reserve=24.0,
    ))
    hazards = MovingCircles(())

    proposal = planner.propose(request(
        hazards,
        x=10.0,
        previous="stay",
        hard=("stay", "left", "right"),
    ))

    assert proposal.action == BY_NAME["right"]
    by_action = {
        evaluation.action.name: evaluation
        for evaluation in proposal.evaluations
    }
    assert (
        by_action["right"].terminal_boundary_deficit
        < by_action["stay"].terminal_boundary_deficit
    )


def test_equal_open_space_keeps_direction_instead_of_reversing():
    planner = LocalPlanner(LocalPlannerConfig(horizon=6))
    hazards = MovingCircles(())

    proposal = planner.propose(request(
        hazards,
        previous="left",
        hard=("left", "right"),
        continuation=("stay", "left", "right"),
    ))

    assert proposal.action == BY_NAME["left"]
    assert proposal.evaluations[0].reversed_direction is False


def test_unknown_forecast_fails_closed_at_the_root():
    planner = LocalPlanner(LocalPlannerConfig(horizon=8))
    hazards = MovingCircles((), known_through=0)

    proposal = planner.propose(request(hazards))

    assert not proposal.available
    assert proposal.reason == "forecast-has-no-known-safe-root"


def test_partial_forecast_reports_the_longest_known_safe_prefix():
    planner = LocalPlanner(LocalPlannerConfig(horizon=8))
    hazards = MovingCircles((), known_through=4)

    proposal = planner.propose(request(hazards))

    assert proposal.available
    assert proposal.effort_horizon == 4
    assert proposal.reason == "longest-known-safe-prefix"


def test_bomb_is_not_a_representable_planner_action():
    assert {action.name for action in ACTIONS} == {
        "stay",
        "up",
        "down",
        "left",
        "right",
        "up_left",
        "up_right",
        "down_left",
        "down_right",
        "stay_fast",
        "up_fast",
        "down_fast",
        "left_fast",
        "right_fast",
        "up_left_fast",
        "up_right_fast",
        "down_left_fast",
        "down_right_fast",
    }
