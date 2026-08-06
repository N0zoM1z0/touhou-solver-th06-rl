"""A small collision-first receding-horizon beam for ordinary dodging.

This keeps only the useful local part of the TH08 design. It has no stage or
phase branches, no route targets, no learning, and no authority to add a first
action. The caller owns capture, source hazard projection, Hard certification,
fresh issue certification, and input publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .model import (
    Action,
    Bounds,
    CertifiedAction,
    HazardOracle,
    Kinematics,
)


@dataclass(frozen=True)
class LocalPlannerConfig:
    horizon: int = 12
    action_hold_frames: int = 2
    beam_width: int = 64
    position_quantization: float = 0.5
    comfort_clearance: float = 8.0
    boundary_reserve: float = 16.0
    risk_scale: float = 12.0
    direction_switch_cost: float = 0.08
    direction_reverse_cost: float = 24.0
    focus_switch_cost: float = 0.12

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("planner horizon must be positive")
        if self.action_hold_frames <= 0:
            raise ValueError("action hold must be positive")
        if self.beam_width <= 0:
            raise ValueError("beam width must be positive")
        if self.position_quantization <= 0.0:
            raise ValueError("position quantization must be positive")
        if min(
            self.comfort_clearance,
            self.boundary_reserve,
            self.risk_scale,
        ) <= 0.0:
            raise ValueError("planner distance scales must be positive")


@dataclass(frozen=True)
class LocalPlanningRequest:
    x: float
    y: float
    previous_action: Action
    hard_first_actions: tuple[CertifiedAction, ...]
    continuation_actions: tuple[Action, ...]
    kinematics: Kinematics
    bounds: Bounds
    hazards: HazardOracle

    def __post_init__(self) -> None:
        actions = tuple(item.action for item in self.hard_first_actions)
        if len(actions) != len(frozenset(actions)):
            raise ValueError("Hard first actions must be unique")
        if len(self.continuation_actions) != len(
            frozenset(self.continuation_actions)
        ):
            raise ValueError("continuation actions must be unique")


@dataclass(frozen=True)
class _SearchNode:
    x: float
    y: float
    first_action: Action
    last_action: Action
    min_clearance: float
    risk: float
    direction_switches: int
    reversals: int


@dataclass(frozen=True)
class ActionEvaluation:
    action: Action
    survived_frames: int
    min_clearance: float
    cumulative_risk: float
    terminal_x: float
    terminal_y: float
    terminal_boundary_deficit: float
    endpoint_count: int
    continuation_action_count: int
    changed_direction: bool
    reversed_direction: bool


@dataclass(frozen=True)
class LocalProposal:
    action: Action | None
    effort_horizon: int
    evaluations: tuple[ActionEvaluation, ...]
    reason: str

    @property
    def available(self) -> bool:
        return self.action is not None


def _opposed(first: Action, second: Action) -> bool:
    return (
        (first.dx != 0 or first.dy != 0)
        and first.dx == -second.dx
        and first.dy == -second.dy
    )


class LocalPlanner:
    """Rank fresh Hard actions by short-horizon physical maneuverability."""

    def __init__(self, config: LocalPlannerConfig | None = None) -> None:
        self.config = config or LocalPlannerConfig()

    def _node_key(
        self,
        node: _SearchNode,
        bounds: Bounds,
    ) -> tuple[float | int | str, ...]:
        config = self.config
        return (
            max(config.comfort_clearance - node.min_clearance, 0.0),
            bounds.control_reserve_deficit(
                node.x,
                node.y,
                config.boundary_reserve,
            ),
            node.risk,
            node.reversals,
            node.direction_switches,
            -node.min_clearance,
            node.first_action.name,
            node.last_action.name,
        )

    def _dedup_key(self, node: _SearchNode) -> tuple[int, int, Action, Action]:
        scale = 1.0 / self.config.position_quantization
        return (
            int(round(node.x * scale)),
            int(round(node.y * scale)),
            node.first_action,
            node.last_action,
        )

    def _expand(
        self,
        request: LocalPlanningRequest,
        beam: tuple[_SearchNode, ...] | None,
        frame: int,
    ) -> tuple[_SearchNode, ...]:
        config = self.config
        drafts: dict[tuple[int, int, Action, Action], _SearchNode] = {}
        if beam is None:
            roots = tuple(
                _SearchNode(
                    request.x,
                    request.y,
                    item.action,
                    request.previous_action,
                    math.inf,
                    0.0,
                    0,
                    0,
                )
                for item in request.hard_first_actions
            )
        else:
            roots = beam

        for node in roots:
            if frame == 1:
                actions = (node.first_action,)
            elif (frame - 1) % config.action_hold_frames == 0:
                actions = request.continuation_actions
            else:
                actions = (node.last_action,)
            for action in actions:
                x, y = request.kinematics.advance(
                    node.x,
                    node.y,
                    action,
                    request.bounds,
                )
                sample = request.hazards.sample(x, y, frame)
                if (
                    not sample.known
                    or sample.collisions > 0
                    or sample.clearance < 0.0
                ):
                    continue
                changed = action != node.last_action
                reversed_direction = _opposed(action, node.last_action)
                transition_risk = math.exp(
                    -max(sample.clearance, 0.0) / config.risk_scale
                )
                if changed:
                    transition_risk += config.direction_switch_cost
                if reversed_direction:
                    transition_risk += config.direction_reverse_cost
                if action.focused != node.last_action.focused:
                    transition_risk += config.focus_switch_cost
                candidate = _SearchNode(
                    x=x,
                    y=y,
                    first_action=node.first_action,
                    last_action=action,
                    min_clearance=min(node.min_clearance, sample.clearance),
                    risk=node.risk + transition_risk,
                    direction_switches=(
                        node.direction_switches + int(changed)
                    ),
                    reversals=node.reversals + int(reversed_direction),
                )
                key = self._dedup_key(candidate)
                retained = drafts.get(key)
                if (
                    retained is None
                    or self._node_key(candidate, request.bounds)
                    < self._node_key(retained, request.bounds)
                ):
                    drafts[key] = candidate

        return tuple(sorted(
            drafts.values(),
            key=lambda node: self._node_key(node, request.bounds),
        )[:config.beam_width])

    def _evaluations(
        self,
        request: LocalPlanningRequest,
        beam: tuple[_SearchNode, ...],
        survived_frames: int,
    ) -> tuple[ActionEvaluation, ...]:
        config = self.config
        result = []
        for certified in request.hard_first_actions:
            nodes = tuple(
                node for node in beam
                if node.first_action == certified.action
            )
            if not nodes:
                continue
            best = min(
                nodes,
                key=lambda node: self._node_key(node, request.bounds),
            )
            result.append(ActionEvaluation(
                action=certified.action,
                survived_frames=survived_frames,
                min_clearance=min(
                    certified.min_clearance,
                    best.min_clearance,
                ),
                cumulative_risk=best.risk,
                terminal_x=best.x,
                terminal_y=best.y,
                terminal_boundary_deficit=(
                    request.bounds.control_reserve_deficit(
                        best.x,
                        best.y,
                        config.boundary_reserve,
                    )
                ),
                endpoint_count=len({
                    (
                        int(round(node.x / config.position_quantization)),
                        int(round(node.y / config.position_quantization)),
                    )
                    for node in nodes
                }),
                continuation_action_count=len({
                    node.last_action for node in nodes
                }),
                changed_direction=(
                    certified.action != request.previous_action
                ),
                reversed_direction=_opposed(
                    certified.action,
                    request.previous_action,
                ),
            ))
        return tuple(result)

    def _evaluation_key(
        self,
        value: ActionEvaluation,
    ) -> tuple[float | int | str, ...]:
        config = self.config
        return (
            -value.survived_frames,
            max(config.comfort_clearance - value.min_clearance, 0.0),
            value.terminal_boundary_deficit,
            -value.continuation_action_count,
            -value.endpoint_count,
            value.cumulative_risk,
            int(value.reversed_direction),
            int(value.changed_direction),
            -value.min_clearance,
            value.action.name,
        )

    def propose(self, request: LocalPlanningRequest) -> LocalProposal:
        """Return one proposal; the caller must freshly certify it again."""
        if not request.hard_first_actions:
            return LocalProposal(None, 0, (), "hard-safe-set-empty")
        if not request.continuation_actions:
            return LocalProposal(None, 0, (), "no-continuation-actions")

        beam: tuple[_SearchNode, ...] | None = None
        last_complete: tuple[_SearchNode, ...] = ()
        effort_horizon = 0
        for frame in range(1, self.config.horizon + 1):
            expanded = self._expand(request, beam, frame)
            if not expanded:
                break
            beam = expanded
            last_complete = expanded
            effort_horizon = frame

        if not last_complete:
            return LocalProposal(None, 0, (), "forecast-has-no-known-safe-root")

        evaluations = self._evaluations(
            request,
            last_complete,
            effort_horizon,
        )
        if not evaluations:
            return LocalProposal(None, effort_horizon, (), "no-surviving-first-action")
        chosen = min(evaluations, key=self._evaluation_key)
        return LocalProposal(
            action=chosen.action,
            effort_horizon=effort_horizon,
            evaluations=tuple(sorted(evaluations, key=self._evaluation_key)),
            reason=(
                "full-horizon"
                if effort_horizon == self.config.horizon
                else "longest-known-safe-prefix"
            ),
        )

