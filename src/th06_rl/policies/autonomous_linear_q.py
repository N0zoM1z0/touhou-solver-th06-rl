"""Immutable, game-neutral linear Q committee above a native safe set."""

from __future__ import annotations

from collections import Counter
import math

from ..autonomous_learning import (
    MODEL_SCHEMA,
    POLICY_STATE_SCHEMA,
    predict_model,
)
from ..learning_features import FEATURE_SCHEMA, candidate_vector, feature_names
from ..policy_api import POLICY_API_VERSION, PolicyDecision


class AutonomousLinearQPolicy:
    api_version = POLICY_API_VERSION
    name = "autonomous-linear-q-uninitialized"

    def __init__(self) -> None:
        self.loaded = False
        self.mode = "shadow"
        self.observation_names: tuple[str, ...] = ()
        self.action_names: tuple[str, ...] = ()
        self.full: dict[str, object] = {}
        self.committee: tuple[dict[str, object], ...] = ()
        self.supported_actions: frozenset[str] = frozenset()
        self.margin = 0.0
        self.active_override_budget: int | None = 64
        self.decisions = 0
        self.supported_decisions = 0
        self.committee_abstentions = 0
        self.margin_abstentions = 0
        self.shadow_proposals = 0
        self.active_overrides = 0
        self.budget_abstentions = 0
        self.proposals: Counter[str] = Counter()

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != POLICY_STATE_SCHEMA:
            raise ValueError("unsupported autonomous policy state")
        if state.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("autonomous policy feature schema mismatch")
        mode = state.get("mode")
        if mode not in ("shadow", "active"):
            raise ValueError("autonomous policy mode must be shadow or active")
        observation_names = tuple(state.get("observation_feature_names", ()))
        action_names = tuple(state.get("action_feature_names", ()))
        expected_names = feature_names(observation_names, action_names)
        if tuple(state.get("feature_names", ())) != expected_names:
            raise ValueError("autonomous policy feature order mismatch")
        model = state.get("model")
        if not isinstance(model, dict) or model.get("schema") != MODEL_SCHEMA:
            raise ValueError("autonomous policy model contract is invalid")
        full = model.get("full")
        committee = model.get("committee")
        if not isinstance(full, dict) or not isinstance(committee, list) or len(committee) < 2:
            raise TypeError("autonomous policy committee is absent")
        selection = state.get("selection")
        support = state.get("support")
        authorization = state.get("authorization")
        if not isinstance(selection, dict) or not isinstance(support, dict):
            raise TypeError("autonomous policy support contract is absent")
        supported = frozenset(
            str(action) for action, row in support.items()
            if isinstance(row, dict) and row.get("authorized") is True
        )
        margin = float(selection.get("score_margin", float("nan")))
        raw_budget = selection.get("active_override_budget", 64)
        budget = None if raw_budget is None else int(raw_budget)
        if not supported or not math.isfinite(margin) or margin < 0.0:
            raise ValueError("autonomous policy support or margin is invalid")
        if budget is not None and budget <= 0:
            raise ValueError("active override budget must be positive or null")
        if mode == "active":
            if (
                not isinstance(authorization, dict)
                or authorization.get("fit_eligible") is not True
                or not isinstance(authorization.get("active_canary"), dict)
            ):
                raise ValueError("active autonomous policy lacks canary authorization")
            audit_sha = authorization["active_canary"].get("shadow_audit_sha256")
            if not isinstance(audit_sha, str) or len(audit_sha) != 64:
                raise ValueError("active autonomous policy shadow binding is invalid")
            if budget is None:
                evaluation = authorization.get("full_evaluation")
                if not isinstance(evaluation, dict):
                    raise ValueError(
                        "unbounded active policy lacks canary-bound evaluation authorization"
                    )
                canary_sha = evaluation.get("canary_audit_sha256")
                if not isinstance(canary_sha, str) or len(canary_sha) != 64:
                    raise ValueError("full evaluation canary binding is invalid")
        # Conformance-check every model using a zero vector before accepting it.
        zero = (0.0,) * len(expected_names)
        for member in (full, *committee):
            value = predict_model(member, zero)
            if not math.isfinite(value):
                raise ValueError("autonomous policy model is non-finite")
        self.mode = str(mode)
        self.observation_names = observation_names
        self.action_names = action_names
        self.full = full
        self.committee = tuple(committee)
        self.supported_actions = supported
        self.margin = margin
        self.active_override_budget = budget
        self.name = f"autonomous-linear-q-{mode}"
        self.loaded = True

    def _vector(self, context, action: str) -> tuple[float, ...]:
        return candidate_vector(
            observation_features=context.observation_features,
            action_features=context.action_features,
            action=action,
            baseline_action=context.baseline_action,
            current_action=context.current_action,
            observation_names=self.observation_names,
            action_names=self.action_names,
        )

    def decide(self, context) -> PolicyDecision:
        if not self.loaded:
            raise RuntimeError("autonomous policy requires a state file")
        legal = tuple(sorted(set(context.locally_admissible_actions)))
        baseline = str(context.baseline_action)
        if not legal or baseline not in legal:
            raise ValueError("autonomous policy received an invalid safe set")
        candidates = tuple(
            action for action in legal
            if action == baseline or action in self.supported_actions
        )
        vectors = {action: self._vector(context, action) for action in candidates}

        def scores(model):
            return {
                action: predict_model(model, vector)
                for action, vector in vectors.items()
            }

        full_scores = scores(self.full)
        committee_scores = [scores(model) for model in self.committee]
        best = max(candidates, key=lambda action: (full_scores[action], action))
        member_best = [
            max(candidates, key=lambda action: (row[action], action))
            for row in committee_scores
        ]
        proposed = baseline
        if best != baseline:
            self.supported_decisions += 1
            if any(action != best for action in member_best):
                self.committee_abstentions += 1
            elif (
                full_scores[best] - full_scores[baseline] < self.margin
                or any(
                    row[best] - row[baseline] < self.margin
                    for row in committee_scores
                )
            ):
                self.margin_abstentions += 1
            else:
                proposed = best
        self.decisions += 1
        if proposed != baseline:
            self.proposals[proposed] += 1
            if self.mode == "shadow":
                self.shadow_proposals += 1
            else:
                if (
                    self.active_override_budget is not None
                    and self.active_overrides >= self.active_override_budget
                ):
                    proposed = baseline
                    self.budget_abstentions += 1
                else:
                    self.active_overrides += 1
        published = proposed if self.mode == "active" else baseline
        return PolicyDecision(published, self.name, 1.0)

    def metrics(self) -> dict[str, object]:
        return {
            "schema": POLICY_STATE_SCHEMA,
            "mode": self.mode,
            "decisions": self.decisions,
            "supported_decisions": self.supported_decisions,
            "committee_abstentions": self.committee_abstentions,
            "margin_abstentions": self.margin_abstentions,
            "shadow_proposals": self.shadow_proposals,
            "active_overrides": self.active_overrides,
            "active_override_budget": self.active_override_budget,
            "budget_abstentions": self.budget_abstentions,
            "proposals": dict(sorted(self.proposals.items())),
        }


def create_policy() -> AutonomousLinearQPolicy:
    return AutonomousLinearQPolicy()
