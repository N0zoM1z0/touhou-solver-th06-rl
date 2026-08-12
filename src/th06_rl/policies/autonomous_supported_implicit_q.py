"""Immutable Generation-5 supported pessimistic implicit-Q population."""

from __future__ import annotations

import math

from ..implicit_learning import POPULATION_MEMBERS, STATE_SCHEMA
from ..sequential_learning import RICH_FEATURE_SCHEMA
from .autonomous_sequential_r_critic import AutonomousSequentialRCriticPolicy


class AutonomousSupportedImplicitQPolicy(AutonomousSequentialRCriticPolicy):
    name = "autonomous-supported-implicit-q-uninitialized"
    state_schema = STATE_SCHEMA
    feature_schema = RICH_FEATURE_SCHEMA
    population_members = POPULATION_MEMBERS
    selection_rule = "population-range-upper-bound-relative-to-incumbent"
    population_kind = "whole-episode-bootstrap-action-centered-implicit-q"
    policy_slug = "autonomous-supported-implicit-q"
    generation_label = "Generation-5"

    def _selection_contract(self, selection: dict[str, object]) -> bool:
        return math.isclose(
            float(selection.get("uncertainty_range_multiplier", math.nan)),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )

    def _advantage_bound(self, member_advantages: list[float]) -> float:
        maximum = max(member_advantages)
        return maximum + maximum - min(member_advantages)

    def _panel_candidate(
        self,
        *,
        indices: tuple[int, ...],
        legal: tuple[str, ...],
        baseline: str,
        supported: list[int],
        predictions,
    ) -> str:
        baseline_index = legal.index(baseline)
        candidates = []
        for action_index in supported:
            values = [
                predictions[member][action_index]
                - predictions[member][baseline_index]
                for member in indices
            ]
            bound = self._advantage_bound(values)
            if bound < 0.0:
                candidates.append((bound, legal[action_index]))
        return min(candidates, default=(0.0, baseline))[1]

    def _select_candidate(
        self,
        *,
        legal: tuple[str, ...],
        baseline: str,
        supported: list[int],
        predictions,
    ) -> str:
        if len(predictions) != 7:
            raise ValueError("Generation-5 panel split needs seven members")
        left = self._panel_candidate(
            indices=(0, 1, 2),
            legal=legal,
            baseline=baseline,
            supported=supported,
            predictions=predictions,
        )
        right = self._panel_candidate(
            indices=(3, 4, 5, 6),
            legal=legal,
            baseline=baseline,
            supported=supported,
            predictions=predictions,
        )
        return left if left == right else baseline


def create_policy() -> AutonomousSupportedImplicitQPolicy:
    return AutonomousSupportedImplicitQPolicy()
