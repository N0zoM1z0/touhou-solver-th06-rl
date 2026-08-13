"""Immutable conservative fitted-Q committee over native-safe trajectories."""

from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import zlib

from ..conservative_learning import MODEL_CODEC, STATE_SCHEMA
from ..learning_features import (
    TREE_FEATURE_SCHEMA,
    tree_candidate_vector,
    tree_feature_names,
)
from ..actions import ACTION_NAMES
from ..policy_api import POLICY_API_VERSION, PolicyDecision
from .offline_ranker import (
    NATIVE_SCORER_ENV,
    NativePrototypeSupport,
    NativeXGBoostRegressor,
    PortablePrototypeSupport,
    PortableXGBoostRegressor,
)


_ACTION_INDEX = {action: index for index, action in enumerate(ACTION_NAMES)}


def _pessimistic_candidate(
    predictions: list[list[float]],
    *,
    supported: list[int],
    baseline_index: int,
    legal: tuple[str, ...],
    uncertainty_scale: float,
) -> tuple[int, str]:
    """Select the strongest robust improvement across every supported action."""
    candidates = [index for index in supported if index != baseline_index]
    if not candidates:
        return baseline_index, "no-candidate"
    unanimous = [
        index for index in candidates
        if all(
            member[index] < member[baseline_index]
            for member in predictions
        )
    ]
    if not unanimous:
        return baseline_index, "committee"

    def moments(index: int) -> tuple[float, float]:
        values = [member[index] for member in predictions]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return mean, math.sqrt(variance)

    baseline_mean, baseline_std = moments(baseline_index)
    baseline_lower = baseline_mean - uncertainty_scale * baseline_std
    robust = []
    for index in unanimous:
        candidate_mean, candidate_std = moments(index)
        candidate_upper = candidate_mean + uncertainty_scale * candidate_std
        margin = baseline_lower - candidate_upper
        if margin > 0.0:
            robust.append((margin, legal[index], index))
    if not robust:
        return baseline_index, "bound"
    # Maximize the pessimistic improvement, with a stable action-name tie break.
    best_margin = max(row[0] for row in robust)
    best = min(row for row in robust if row[0] == best_margin)
    return best[2], "selected"


def _decode_model(row: object) -> dict[str, object]:
    if not isinstance(row, dict) or row.get("codec") != MODEL_CODEC:
        raise TypeError("conservative model payload is invalid")
    payload = row.get("payload")
    expected = row.get("sha256")
    if not isinstance(payload, str) or not isinstance(expected, str):
        raise TypeError("conservative model payload binding is absent")
    raw = zlib.decompress(base64.b64decode(payload, validate=True))
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("conservative model payload SHA-256 mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("conservative model artifact is not an object")
    return value


class AutonomousConservativeQPolicy:
    api_version = POLICY_API_VERSION
    name = "autonomous-conservative-q-uninitialized"

    def __init__(self) -> None:
        self.mode = "shadow"
        self.observation_names: tuple[str, ...] = ()
        self.action_names: tuple[str, ...] = ()
        self.scorers = []
        self.support = None
        self.support_threshold = 0.0
        self.uncertainty_scale = 1.0
        self.active_override_budget: int | None = 0
        self.decisions = 0
        self.support_abstentions = 0
        self.committee_abstentions = 0
        self.bound_abstentions = 0
        self.shadow_proposals = 0
        self.active_overrides = 0
        self.budget_abstentions = 0
        self.proposals: Counter[str] = Counter()
        self.scorer_backend = "uninitialized"

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("unsupported conservative-Q policy state")
        mode = state.get("mode")
        if mode not in ("shadow", "active"):
            raise ValueError("conservative-Q mode must be shadow or active")
        authorization = state.get("authorization")
        if (
            not isinstance(authorization, dict)
            or authorization.get("fit_eligible") is not True
            or (mode == "active" and not isinstance(
                authorization.get("active_canary"), dict
            ))
        ):
            raise ValueError("conservative-Q authorization is absent")
        observation_names = tuple(str(value) for value in state.get(
            "observation_feature_names", ()
        ))
        action_names = tuple(str(value) for value in state.get(
            "action_feature_names", ()
        ))
        names = tree_feature_names(observation_names, action_names)
        if (
            state.get("feature_schema") != TREE_FEATURE_SCHEMA
            or tuple(state.get("feature_names", ())) != names
        ):
            raise ValueError("conservative-Q feature schema mismatch")
        raw_models = state.get("models")
        if not isinstance(raw_models, list) or len(raw_models) < 3:
            raise ValueError("conservative-Q committee is too small")
        portable_models = [
            PortableXGBoostRegressor(
                _decode_model(row),
                expected_feature_schema=TREE_FEATURE_SCHEMA,
                expected_feature_names=names,
            )
            for row in raw_models
        ]
        support_artifact = state.get("support")
        if not isinstance(support_artifact, dict):
            raise TypeError("conservative-Q support artifact is absent")
        portable_support = PortablePrototypeSupport(
            support_artifact, feature_count=len(names)
        )
        threshold = float(support_artifact.get("threshold", -1.0))
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("conservative-Q support threshold is invalid")
        scorers = list(portable_models)
        support_scorer = portable_support
        backend = "python-portable"
        native_contract = state.get("native_scorer")
        path_value = os.environ.get(NATIVE_SCORER_ENV)
        if path_value:
            if not isinstance(native_contract, dict):
                raise TypeError("conservative-Q native scorer contract is absent")
            path = Path(path_value)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            compatible = native_contract.get("compatible_sha256", ())
            if not isinstance(compatible, list):
                raise TypeError("native scorer compatibility list is invalid")
            allowed = {str(native_contract.get("sha256", "")), *map(str, compatible)}
            if actual not in allowed:
                raise ValueError("native conservative-Q scorer is not compatible")
            expected = actual
            scorers = [
                NativeXGBoostRegressor(
                    path, expected_sha256=expected, portable=portable
                )
                for portable in portable_models
            ]
            support_scorer = NativePrototypeSupport(
                path, expected_sha256=expected, portable=portable_support
            )
            for scorer, artifact in zip(scorers, map(_decode_model, raw_models), strict=True):
                conformance = artifact.get("conformance")
                if not isinstance(conformance, list) or not conformance:
                    raise ValueError("conservative-Q conformance vectors are absent")
                rows = [[float(value) for value in row["features"]] for row in conformance]
                actual = scorer.predict_many(rows)
                expected_values = [float(row["prediction"]) for row in conformance]
                if any(
                    not math.isclose(left, right, rel_tol=2e-5, abs_tol=2e-5)
                    for left, right in zip(actual, expected_values, strict=True)
                ):
                    raise ValueError("native conservative-Q conformance failed")
            backend = "native-batch"
        elif os.name == "nt":
            raise RuntimeError(
                f"Windows conservative-Q requires {NATIVE_SCORER_ENV}"
            )
        selection = state.get("selection")
        if not isinstance(selection, dict):
            raise TypeError("conservative-Q selection contract is absent")
        uncertainty = float(selection.get("uncertainty_scale", -1.0))
        budget = selection.get("active_override_budget")
        if uncertainty < 0.0 or (
            budget is not None and int(budget) <= 0
        ):
            raise ValueError("conservative-Q selection bounds are invalid")
        self.mode = str(mode)
        self.observation_names = observation_names
        self.action_names = action_names
        self.scorers = scorers
        self.support = support_scorer
        self.support_threshold = threshold
        self.uncertainty_scale = uncertainty
        self.active_override_budget = None if budget is None else int(budget)
        self.scorer_backend = backend
        self.name = f"autonomous-conservative-q-{mode}"

    def _vector(self, context, action: str) -> list[float]:
        return list(tree_candidate_vector(
            observation_features=context.observation_features,
            action_features=context.action_features,
            action=action,
            baseline_action=context.baseline_action,
            current_action=context.current_action,
            observation_names=self.observation_names,
            action_names=self.action_names,
        ))

    def decide(self, context) -> PolicyDecision:
        if not self.scorers or self.support is None:
            raise RuntimeError("conservative-Q state was not loaded")
        legal = tuple(sorted(
            set(context.locally_admissible_actions),
            key=_ACTION_INDEX.__getitem__,
        ))
        baseline = str(context.baseline_action)
        if not legal or baseline not in legal:
            raise ValueError("conservative-Q received an invalid native-safe set")
        rows = [self._vector(context, action) for action in legal]
        distances = self.support.distances(
            rows, [_ACTION_INDEX[action] for action in legal]
        )
        supported = [
            index for index, action in enumerate(legal)
            if action == baseline or distances[index] <= self.support_threshold
        ]
        self.support_abstentions += len(legal) - len(supported)
        predictions = [scorer.predict_many(rows) for scorer in self.scorers]
        baseline_index = legal.index(baseline)
        proposed = baseline
        best, reason = _pessimistic_candidate(
            predictions,
            supported=supported,
            baseline_index=baseline_index,
            legal=legal,
            uncertainty_scale=self.uncertainty_scale,
        )
        if reason == "committee":
            self.committee_abstentions += 1
        elif reason == "bound":
            self.bound_abstentions += 1
        elif reason == "selected":
            proposed = legal[best]
        self.decisions += 1
        if proposed != baseline:
            self.proposals[proposed] += 1
            if self.mode == "shadow":
                self.shadow_proposals += 1
            elif (
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
            "mode": self.mode,
            "decisions": self.decisions,
            "support_abstentions": self.support_abstentions,
            "committee_abstentions": self.committee_abstentions,
            "bound_abstentions": self.bound_abstentions,
            "shadow_proposals": self.shadow_proposals,
            "active_overrides": self.active_overrides,
            "active_override_budget": self.active_override_budget,
            "budget_abstentions": self.budget_abstentions,
            "support_threshold": self.support_threshold,
            "scorer_backend": self.scorer_backend,
            "proposals": dict(sorted(self.proposals.items())),
        }


def create_policy() -> AutonomousConservativeQPolicy:
    return AutonomousConservativeQPolicy()
