"""One-shot paired Wine intervention above the frozen incumbent.

The wrapper never enlarges the controller's native/local safe set.  It asks
the incumbent for its ordinary action, looks for one generic physical
frontier, and records exactly one balanced incumbent/alternative assignment.
The paired state file chooses the arm; no game RNG, frame, phase, or run
identity is consulted by movement logic.
"""

from __future__ import annotations

import math

from ..policy_api import POLICY_API_VERSION, PolicyDecision
from .adaptive import AdaptivePolicy


STATE_SCHEMA = "th06-rl-wine-intervention-pair-v1"
POLICY_NAME = "wine-one-shot-intervention-v1"
ARMS = ("incumbent", "alternative")


def _boundary_reserve(x: float, y: float) -> float:
    return min(x - 8.0, 376.0 - x, y - 16.0, 432.0 - y)


class WineInterventionPolicy:
    api_version = POLICY_API_VERSION
    name = POLICY_NAME

    def __init__(self) -> None:
        self.incumbent = AdaptivePolicy()
        self.loaded = False
        self.pair_id = "unloaded"
        self.arm = "incumbent"
        self.alternative_probability = 0.5
        self.min_player_y = 420.0
        self.min_bullets = 256
        self.max_hard_actions = 12
        self.max_reserve_deficit = 4.0
        self.required_effort_horizon = 4
        self.intervened = False
        self.eligible_frontiers = 0
        self.event: dict[str, object] | None = None

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("Wine intervention state schema mismatch")
        pair_id = state.get("pair_id")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or any(not (character.isalnum() or character in "_.-") for character in pair_id)
        ):
            raise ValueError("pair_id must use only letters, digits, dot, dash, underscore")
        arm = state.get("arm")
        if arm not in ARMS:
            raise ValueError(f"intervention arm must be one of {ARMS}")
        probability = float(state.get("alternative_probability", 0.5))
        if not 0.0 < probability < 1.0:
            raise ValueError("alternative_probability must be in (0, 1)")
        incumbent_state = state.get("incumbent_state")
        if not isinstance(incumbent_state, dict):
            raise TypeError("intervention state must embed the incumbent state")
        eligibility = state.get("eligibility", {})
        if not isinstance(eligibility, dict):
            raise TypeError("eligibility must be an object")

        self.incumbent.import_state(incumbent_state)
        self.pair_id = pair_id
        self.arm = str(arm)
        self.alternative_probability = probability
        self.min_player_y = float(eligibility.get("min_player_y", 420.0))
        self.min_bullets = int(eligibility.get("min_bullets", 256))
        self.max_hard_actions = int(eligibility.get("max_hard_actions", 12))
        self.max_reserve_deficit = float(
            eligibility.get("max_reserve_deficit", 4.0)
        )
        self.required_effort_horizon = int(
            eligibility.get("required_effort_horizon", 4)
        )
        if not (
            math.isfinite(self.min_player_y)
            and 16.0 <= self.min_player_y <= 432.0
            and self.min_bullets >= 0
            and 1 <= self.max_hard_actions <= 18
            and math.isfinite(self.max_reserve_deficit)
            and self.max_reserve_deficit >= 0.0
            and self.required_effort_horizon == 4
        ):
            raise ValueError("invalid generic intervention eligibility")
        self.loaded = True

    @staticmethod
    def _evaluation_rows(context) -> dict[str, tuple[float, float, float]]:
        rows = {}
        for action, clearance, final_x, final_y in context.hard_action_evaluations:
            if clearance is None:
                continue
            numbers = (float(clearance), float(final_x), float(final_y))
            if all(math.isfinite(value) for value in numbers):
                rows[str(action)] = numbers
        return rows

    def _alternative(self, context, incumbent_action: str) -> str | None:
        if (
            self.intervened
            or context.player_y < self.min_player_y
            or context.bullet_count < self.min_bullets
            or context.hard_action_count > self.max_hard_actions
            or context.effort_horizon != self.required_effort_horizon
        ):
            return None
        evaluations = self._evaluation_rows(context)
        incumbent = evaluations.get(incumbent_action)
        if incumbent is None:
            return None
        incumbent_reserve = _boundary_reserve(incumbent[1], incumbent[2])
        local = set(context.locally_admissible_actions)
        alternatives = [
            (action, values)
            for action, values in evaluations.items()
            if action in local and action != incumbent_action
        ]
        if not alternatives:
            return None
        action, values = max(
            alternatives,
            key=lambda item: (
                _boundary_reserve(item[1][1], item[1][2]),
                item[1][0],
                item[0] == context.baseline_action,
                item[0],
            ),
        )
        candidate_reserve = _boundary_reserve(values[1], values[2])
        if candidate_reserve < incumbent_reserve - self.max_reserve_deficit:
            return None
        return action

    def decide(self, context):
        if not self.loaded:
            raise RuntimeError("Wine intervention policy requires a state file")
        incumbent = self.incumbent.decide(context)
        candidate = self._alternative(context, incumbent.action)
        if candidate is None:
            return incumbent

        self.eligible_frontiers += 1
        self.intervened = True
        chosen = candidate if self.arm == "alternative" else incumbent.action
        probability = (
            self.alternative_probability
            if self.arm == "alternative"
            else 1.0 - self.alternative_probability
        )
        self.event = {
            "frame": int(context.frame),
            "arm": self.arm,
            "incumbent_action": incumbent.action,
            "alternative_action": candidate,
            "published_action": chosen,
            "behavior_probability": probability,
        }
        event_id = (
            f"{POLICY_NAME}:{self.pair_id}:{self.arm}:"
            f"{incumbent.action}:{candidate}"
        )
        return PolicyDecision(chosen, event_id, probability)

    def metrics(self) -> dict[str, object]:
        return {
            "schema": STATE_SCHEMA,
            "pair_id": self.pair_id,
            "arm": self.arm,
            "alternative_probability": self.alternative_probability,
            "intervention_budget": 1,
            "eligible_frontiers": self.eligible_frontiers,
            "interventions": int(self.intervened),
            "event": self.event,
            "incumbent": self.incumbent.metrics(),
        }


def create_policy() -> WineInterventionPolicy:
    return WineInterventionPolicy()
