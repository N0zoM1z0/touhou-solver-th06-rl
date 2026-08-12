"""Immutable Generation-4 full-population sequential R critic."""

from __future__ import annotations

from collections import Counter, deque
import hashlib
import math
import os
from pathlib import Path
import time

from ..advantage_learning import (
    encode_hazard_set,
    hazard_codebook_feature_names,
    rich_candidate_vector_from_encoding,
    rich_feature_names,
)
from ..hazard_representation import HISTORY_FEATURE_NAMES, NativeHazardCodebookEncoder
from ..learning_features import tree_candidate_vector
from ..offline import ACTION_NAMES
from ..policy_api import POLICY_API_VERSION, PolicyDecision
from ..sequential_learning import (
    POPULATION_MEMBERS,
    RICH_FEATURE_SCHEMA,
    STATE_SCHEMA,
)
from .autonomous_conservative_q import _decode_model
from .offline_ranker import (
    NATIVE_SCORER_ENV,
    NativePrototypeSupport,
    NativeXGBoostPopulation,
    PortablePrototypeSupport,
    PortableXGBoostRegressor,
)


_ACTION_INDEX = {action: index for index, action in enumerate(ACTION_NAMES)}


def _p95(values) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)])


class AutonomousSequentialRCriticPolicy:
    api_version = POLICY_API_VERSION
    name = "autonomous-sequential-r-critic-uninitialized"
    state_schema = STATE_SCHEMA
    feature_schema = RICH_FEATURE_SCHEMA
    population_members = POPULATION_MEMBERS
    selection_rule = "all-members-negative-relative-to-incumbent"
    population_kind = "whole-episode-bootstrap-action-centered-r-critic"
    policy_slug = "autonomous-sequential-r-critic"
    generation_label = "Generation-4"

    def __init__(self) -> None:
        self.mode = "shadow"
        self.observation_names: tuple[str, ...] = ()
        self.action_names: tuple[str, ...] = ()
        self.scorers = []
        self.population_scorer = None
        self.support = None
        self.support_threshold = 0.0
        self.factual_supported_actions: frozenset[str] = frozenset()
        self.hazard_encoder = None
        self.decisions = 0
        self.shadow_proposals = 0
        self.active_overrides = 0
        self.support_abstentions = 0
        self.population_abstentions = 0
        self.proposals: Counter[str] = Counter()
        self.timing_ms = deque(maxlen=4096)
        self.over_four_ms = 0
        self.deadline_misses = 0
        self.scorer_backend = "uninitialized"
        self.trees_per_member = 0

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != self.state_schema:
            raise ValueError(
                f"unsupported {self.generation_label} population state"
            )
        mode = state.get("mode")
        authorization = state.get("authorization")
        if (
            mode not in ("shadow", "active")
            or not isinstance(authorization, dict)
            or authorization.get("fit_eligible") is not True
            or not isinstance(authorization.get("policy_calibration"), dict)
            or (mode == "active" and not isinstance(
                authorization.get("active_canary"), dict
            ))
        ):
            raise ValueError("Generation-4 authorization is absent")
        names = rich_feature_names()
        if (
            state.get("feature_schema") != self.feature_schema
            or tuple(state.get("feature_names", ())) != names
        ):
            raise ValueError("Generation-4 rich feature schema mismatch")
        raw_models = state.get("models")
        if (
            not isinstance(raw_models, list)
            or len(raw_models) != self.population_members
        ):
            raise ValueError(
                f"{self.generation_label} population member count differs"
            )
        portable_models = [
            PortableXGBoostRegressor(
                _decode_model(row),
                expected_feature_schema=self.feature_schema,
                expected_feature_names=names,
            )
            for row in raw_models
        ]
        representation = state.get("representation")
        if not isinstance(representation, dict):
            raise TypeError("Generation-4 representation is absent")
        codebook = representation.get("hazard_codebook")
        if (
            not isinstance(codebook, dict)
            or representation.get("kind")
            != "learned-permutation-invariant-hazard-codebook-plus-factual-history"
            or tuple(representation.get("history_feature_names", ()))
            != HISTORY_FEATURE_NAMES
        ):
            raise ValueError("Generation-4 representation contract mismatch")
        support_artifact = state.get("support")
        if not isinstance(support_artifact, dict):
            raise TypeError("Generation-4 support artifact is absent")
        supported = frozenset(map(
            str, support_artifact.get("factual_supported_actions", ())
        ))
        threshold = float(support_artifact.get("threshold", -1.0))
        if (
            not supported
            or not supported <= set(ACTION_NAMES)
            or not math.isfinite(threshold)
            or threshold < 0.0
        ):
            raise ValueError("Generation-4 factual support is invalid")
        portable_support = PortablePrototypeSupport(
            support_artifact, feature_count=len(names)
        )
        selection = state.get("selection")
        population = state.get("population")
        if (
            not isinstance(selection, dict)
            or selection.get("rule") != self.selection_rule
            or float(selection.get("baseline_advantage", math.nan)) != 0.0
            or not isinstance(population, dict)
            or population.get("kind") != self.population_kind
            or int(population.get("members", -1)) != self.population_members
            or not self._selection_contract(selection)
        ):
            raise ValueError("Generation-4 population selection contract mismatch")

        scorers = list(portable_models)
        support_scorer = portable_support
        encoder = lambda primitives: encode_hazard_set(primitives, codebook)
        backend = "python-portable"
        path_value = os.environ.get(NATIVE_SCORER_ENV)
        if path_value:
            native_contract = state.get("native_scorer")
            if not isinstance(native_contract, dict):
                raise TypeError("Generation-4 native scorer contract is absent")
            path = Path(path_value)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            compatible = native_contract.get("compatible_sha256", ())
            if not isinstance(compatible, list):
                raise TypeError("native scorer compatibility list is invalid")
            allowed = {str(native_contract.get("sha256", "")), *map(str, compatible)}
            if actual not in allowed:
                raise ValueError("native Generation-4 scorer is incompatible")
            population_scorer = NativeXGBoostPopulation(
                path,
                expected_sha256=actual,
                portable=portable_models,
            )
            support_scorer = NativePrototypeSupport(
                path, expected_sha256=actual, portable=portable_support
            )
            native_encoder = NativeHazardCodebookEncoder(
                path,
                expected_sha256=actual,
                artifact=codebook,
                output_count=len(hazard_codebook_feature_names()),
            )
            conformance = codebook.get("conformance")
            if not isinstance(conformance, list) or not conformance:
                raise ValueError("native hazard conformance vectors are absent")
            for row in conformance:
                primitives = tuple(tuple(map(float, item)) for item in row["primitives"])
                observed = native_encoder.encode(primitives)
                expected = tuple(map(float, row["encoding"]))
                if any(
                    not math.isclose(left, right, rel_tol=2e-5, abs_tol=2e-5)
                    for left, right in zip(observed, expected, strict=True)
                ):
                    raise ValueError("native Generation-4 hazard encoding differs")
            encoder = native_encoder.encode
            artifacts = list(map(_decode_model, raw_models))
            rows = [
                [float(value) for value in row["features"]]
                for row in artifacts[0].get("conformance", ())
            ]
            if not rows or any(
                [list(map(float, row["features"])) for row in artifact.get(
                    "conformance", ()
                )] != rows
                for artifact in artifacts
            ):
                raise ValueError("population conformance inputs differ")
            actual_predictions = population_scorer.predict_many(rows)
            for observed, artifact in zip(
                actual_predictions, artifacts, strict=True
            ):
                expected = [
                    float(row["prediction"])
                    for row in artifact.get("conformance", ())
                ]
                if any(
                    not math.isclose(left, right, rel_tol=2e-5, abs_tol=2e-5)
                    for left, right in zip(observed, expected, strict=True)
                ):
                    raise ValueError("native Generation-4 population differs")
            backend = "native-batch"
        elif os.name == "nt":
            raise RuntimeError(
                f"Windows Generation-4 policy requires {NATIVE_SCORER_ENV}"
            )

        self.mode = str(mode)
        self.observation_names = tuple(map(
            str, state.get("observation_feature_names", ())
        ))
        self.action_names = tuple(map(
            str, state.get("action_feature_names", ())
        ))
        self.scorers = scorers
        self.population_scorer = (
            population_scorer if path_value else None
        )
        self.support = support_scorer
        self.support_threshold = threshold
        self.factual_supported_actions = supported
        self.hazard_encoder = encoder
        self.scorer_backend = backend
        self.trees_per_member = int(population.get("trees_per_member", 0))
        self.name = f"{self.policy_slug}-{mode}"

    def _selection_contract(self, selection: dict[str, object]) -> bool:
        return True

    def _advantage_bound(self, member_advantages: list[float]) -> float:
        return max(member_advantages)

    def _history(self, context) -> tuple[float, ...]:
        names = tuple(str(name) for name, _value in context.history_features)
        values = tuple(float(value) for _name, value in context.history_features)
        if names != HISTORY_FEATURE_NAMES:
            raise ValueError("runtime factual history schema mismatch")
        return values

    def _score_context(self, context):
        if not self.scorers or self.support is None or self.hazard_encoder is None:
            raise RuntimeError("Generation-4 policy state was not loaded")
        legal = tuple(sorted(
            set(context.locally_admissible_actions),
            key=_ACTION_INDEX.__getitem__,
        ))
        baseline = str(context.baseline_action)
        if not legal or baseline not in legal:
            raise ValueError("Generation-4 policy received an invalid native-safe set")
        hazard = tuple(self.hazard_encoder(tuple(context.hazard_primitives)))
        history = self._history(context)
        rows = [
            list(rich_candidate_vector_from_encoding(
                tree_candidate_vector(
                    observation_features=context.observation_features,
                    action_features=context.action_features,
                    action=action,
                    baseline_action=baseline,
                    current_action=context.current_action,
                    observation_names=self.observation_names,
                    action_names=self.action_names,
                ),
                hazard,
                history,
            ))
            for action in legal
        ]
        distances = self.support.distances(
            rows, [_ACTION_INDEX[action] for action in legal]
        )
        supported = [
            index for index, action in enumerate(legal)
            if action != baseline
            and action in self.factual_supported_actions
            and distances[index] <= self.support_threshold
        ]
        self.support_abstentions += len(legal) - 1 - len(supported)
        predictions = (
            self.population_scorer.predict_many(rows)
            if self.population_scorer is not None
            else tuple(scorer.predict_many(rows) for scorer in self.scorers)
        )
        return legal, baseline, supported, predictions

    def information_disagreement(self, context) -> dict[str, float]:
        """Return bounded epistemic information weights for safe exploration."""
        legal, baseline, _supported, predictions = self._score_context(context)
        baseline_index = legal.index(baseline)
        result = {}
        for index, action in enumerate(legal):
            advantages = [
                member[index] - member[baseline_index] for member in predictions
            ]
            result[action] = max(1e-6, min(1.0, max(advantages) - min(advantages)))
        return result

    def decide(self, context) -> PolicyDecision:
        started = time.perf_counter()
        legal, baseline, supported, predictions = self._score_context(context)
        baseline_index = legal.index(baseline)
        candidates = []
        for index in supported:
            member_advantages = [
                member[index] - member[baseline_index] for member in predictions
            ]
            upper = self._advantage_bound(member_advantages)
            if upper < 0.0:
                candidates.append((upper, legal[index]))
        proposed = baseline
        if candidates:
            proposed = min(candidates)[1]
        elif supported:
            self.population_abstentions += 1
        self.decisions += 1
        if proposed != baseline:
            self.proposals[proposed] += 1
            if self.mode == "shadow":
                self.shadow_proposals += 1
            else:
                self.active_overrides += 1
        elapsed = (time.perf_counter() - started) * 1000.0
        self.timing_ms.append(elapsed)
        self.over_four_ms += elapsed > 4.0
        self.deadline_misses += elapsed > (1000.0 / 60.0)
        return PolicyDecision(
            proposed if self.mode == "active" else baseline,
            self.name,
            1.0,
        )

    def metrics(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "decisions": self.decisions,
            "population_members": len(self.scorers),
            "trees_per_member": self.trees_per_member,
            "factual_supported_actions": sorted(self.factual_supported_actions),
            "support_abstentions": self.support_abstentions,
            "population_abstentions": self.population_abstentions,
            "shadow_proposals": self.shadow_proposals,
            "active_overrides": self.active_overrides,
            "proposals": dict(sorted(self.proposals.items())),
            "scorer_backend": self.scorer_backend,
            "decision_latency_p95_ms": _p95(self.timing_ms),
            "decision_latency_max_ms": max(self.timing_ms, default=None),
            "decision_latency_over_four_ms": self.over_four_ms,
            "controller_deadline_misses": self.deadline_misses,
        }


def create_policy() -> AutonomousSequentialRCriticPolicy:
    return AutonomousSequentialRCriticPolicy()
