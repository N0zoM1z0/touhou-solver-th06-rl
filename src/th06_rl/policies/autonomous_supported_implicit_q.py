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


def create_policy() -> AutonomousSupportedImplicitQPolicy:
    return AutonomousSupportedImplicitQPolicy()
