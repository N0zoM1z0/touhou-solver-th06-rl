"""Immutable Generation-3 calibrated residual population over native options."""

from __future__ import annotations

from collections import Counter, deque
import hashlib
import math
import os
from pathlib import Path
import time

from ..advantage_learning import (
    POPULATION_MEMBERS,
    RICH_FEATURE_SCHEMA,
    STATE_SCHEMA,
    encode_hazard_set,
    hazard_codebook_feature_names,
    rich_candidate_vector_from_encoding,
    rich_feature_names,
)
from ..hazard_representation import (
    HISTORY_FEATURE_NAMES,
    NativeHazardCodebookEncoder,
)
from ..learning_features import tree_candidate_vector
from ..actions import ACTION_NAMES
from ..policy_api import POLICY_API_VERSION, PolicyDecision
from .autonomous_conservative_q import _decode_model
from .offline_ranker import (
    NATIVE_SCORER_ENV,
    NativePrototypeSupport,
    NativeXGBoostRegressor,
    PortablePrototypeSupport,
    PortableXGBoostRegressor,
)


_ACTION_INDEX = {action: index for index, action in enumerate(ACTION_NAMES)}


def _p95(values) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)])


class AutonomousDROptionAdvantagePolicy:
    api_version = POLICY_API_VERSION
    name = "autonomous-dr-option-advantage-uninitialized"

    def __init__(self) -> None:
        self.mode = "shadow"
        self.observation_names: tuple[str, ...] = ()
        self.action_names: tuple[str, ...] = ()
        self.feature_names: tuple[str, ...] = ()
        self.scorers = []
        self.support = None
        self.support_threshold = 0.0
        self.factual_supported_actions: frozenset[str] = frozenset()
        self.codebook: dict[str, object] = {}
        self.hazard_encoder = None
        self.conformal_radius = 0.0
        self.decisions = 0
        self.shadow_proposals = 0
        self.active_overrides = 0
        self.support_abstentions = 0
        self.bound_abstentions = 0
        self.proposals: Counter[str] = Counter()
        self.timing_ms = deque(maxlen=4096)
        self.over_four_ms = 0
        self.deadline_misses = 0
        self.scorer_backend = "uninitialized"

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("unsupported Generation-3 advantage state")
        mode = state.get("mode")
        if mode not in ("shadow", "active"):
            raise ValueError("Generation-3 mode must be shadow or active")
        authorization = state.get("authorization")
        if (
            not isinstance(authorization, dict)
            or authorization.get("fit_eligible") is not True
            or not isinstance(authorization.get("calibration"), dict)
            or (mode == "active" and not isinstance(
                authorization.get("active_canary"), dict
            ))
        ):
            raise ValueError("Generation-3 authorization is absent")
        observation_names = tuple(str(value) for value in state.get(
            "observation_feature_names", ()
        ))
        action_names = tuple(str(value) for value in state.get(
            "action_feature_names", ()
        ))
        names = rich_feature_names()
        if (
            state.get("feature_schema") != RICH_FEATURE_SCHEMA
            or tuple(state.get("feature_names", ())) != names
        ):
            raise ValueError("Generation-3 rich feature schema mismatch")
        raw_models = state.get("models")
        if not isinstance(raw_models, list) or len(raw_models) != POPULATION_MEMBERS:
            raise ValueError("Generation-3 population must contain seven members")
        portable_models = [
            PortableXGBoostRegressor(
                _decode_model(row),
                expected_feature_schema=RICH_FEATURE_SCHEMA,
                expected_feature_names=names,
            )
            for row in raw_models
        ]
        representation = state.get("representation")
        if not isinstance(representation, dict):
            raise TypeError("Generation-3 representation artifact is absent")
        codebook = representation.get("hazard_codebook")
        if (
            not isinstance(codebook, dict)
            or representation.get("kind")
            != "learned-permutation-invariant-hazard-codebook-plus-factual-history"
            or tuple(representation.get("history_feature_names", ()))
            != HISTORY_FEATURE_NAMES
        ):
            raise ValueError("Generation-3 representation contract mismatch")
        support_artifact = state.get("support")
        if not isinstance(support_artifact, dict):
            raise TypeError("Generation-3 support artifact is absent")
        supported = frozenset(map(
            str, support_artifact.get("factual_supported_actions", ())
        ))
        if (
            not supported
            or not supported <= set(ACTION_NAMES)
            or not isinstance(support_artifact.get("threshold"), (int, float))
        ):
            raise ValueError("Generation-3 factual action support is invalid")
        portable_support = PortablePrototypeSupport(
            support_artifact, feature_count=len(names)
        )
        threshold = float(support_artifact["threshold"])
        calibration = authorization["calibration"]
        radius = float(calibration.get("radius", -1.0))
        selection = state.get("selection")
        population = state.get("population")
        if (
            not isinstance(selection, dict)
            or selection.get("rule")
            != "minimum-calibrated-population-upper-advantage"
            or float(selection.get("baseline_advantage", float("nan"))) != 0.0
            or not isinstance(population, dict)
            or population.get("kind")
            != "whole-episode-bootstrap-cross-fitted-dr"
            or int(population.get("members", -1)) != POPULATION_MEMBERS
        ):
            raise ValueError("Generation-3 population selection contract mismatch")
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("Generation-3 conformal radius is invalid")
        if not math.isclose(
            float(selection.get("conformal_radius", -1.0)),
            radius,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("selection and calibration radii disagree")

        scorers = list(portable_models)
        support_scorer = portable_support
        encoder = lambda primitives: encode_hazard_set(primitives, codebook)
        backend = "python-portable"
        native_contract = state.get("native_scorer")
        path_value = os.environ.get(NATIVE_SCORER_ENV)
        if path_value:
            if not isinstance(native_contract, dict):
                raise TypeError("Generation-3 native scorer contract is absent")
            path = Path(path_value)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            compatible = native_contract.get("compatible_sha256", ())
            if not isinstance(compatible, list):
                raise TypeError("native scorer compatibility list is invalid")
            allowed = {str(native_contract.get("sha256", "")), *map(str, compatible)}
            if actual not in allowed:
                raise ValueError("native Generation-3 scorer is not compatible")
            scorers = [
                NativeXGBoostRegressor(
                    path, expected_sha256=actual, portable=portable
                )
                for portable in portable_models
            ]
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
                raise ValueError("hazard encoder conformance vectors are absent")
            for row in conformance:
                primitives = tuple(tuple(map(float, item)) for item in row["primitives"])
                observed = native_encoder.encode(primitives)
                expected = tuple(map(float, row["encoding"]))
                if any(
                    not math.isclose(left, right, rel_tol=2e-5, abs_tol=2e-5)
                    for left, right in zip(observed, expected, strict=True)
                ):
                    raise ValueError("native hazard encoder conformance failed")
            encoder = native_encoder.encode
            for scorer, artifact in zip(
                scorers, map(_decode_model, raw_models), strict=True
            ):
                rows = [
                    [float(value) for value in row["features"]]
                    for row in artifact.get("conformance", ())
                ]
                expected = [
                    float(row["prediction"])
                    for row in artifact.get("conformance", ())
                ]
                if not rows or any(
                    not math.isclose(left, right, rel_tol=2e-5, abs_tol=2e-5)
                    for left, right in zip(
                        scorer.predict_many(rows), expected, strict=True
                    )
                ):
                    raise ValueError("native population scorer conformance failed")
            backend = "native-batch"
        elif os.name == "nt":
            raise RuntimeError(
                f"Windows Generation-3 policy requires {NATIVE_SCORER_ENV}"
            )

        self.mode = str(mode)
        self.observation_names = observation_names
        self.action_names = action_names
        self.feature_names = names
        self.scorers = scorers
        self.support = support_scorer
        self.support_threshold = threshold
        self.factual_supported_actions = supported
        self.codebook = codebook
        self.hazard_encoder = encoder
        self.conformal_radius = radius
        self.scorer_backend = backend
        self.name = f"autonomous-dr-option-advantage-{mode}"

    def _history(self, context) -> tuple[float, ...]:
        names = []
        values = []
        for name, value in context.history_features:
            names.append(str(name))
            values.append(float(value))
        if tuple(names) != HISTORY_FEATURE_NAMES:
            raise ValueError("runtime factual history schema mismatch")
        return tuple(values)

    def decide(self, context) -> PolicyDecision:
        if not self.scorers or self.support is None or self.hazard_encoder is None:
            raise RuntimeError("Generation-3 policy state was not loaded")
        started = time.perf_counter()
        legal = tuple(sorted(
            set(context.locally_admissible_actions),
            key=_ACTION_INDEX.__getitem__,
        ))
        baseline = str(context.baseline_action)
        if not legal or baseline not in legal:
            raise ValueError("Generation-3 policy received an invalid native-safe set")
        hazard_encoding = tuple(self.hazard_encoder(tuple(context.hazard_primitives)))
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
                hazard_encoding,
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
        predictions = [scorer.predict_many(rows) for scorer in self.scorers]
        bounds = [
            (
                max(member[index] for member in predictions)
                + self.conformal_radius,
                legal[index],
                index,
            )
            for index in supported
        ]
        proposed = baseline
        if bounds:
            upper, action, _index = min(bounds)
            if upper < 0.0:
                proposed = action
            else:
                self.bound_abstentions += 1
        self.decisions += 1
        if proposed != baseline:
            self.proposals[proposed] += 1
            if self.mode == "shadow":
                self.shadow_proposals += 1
            else:
                self.active_overrides += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.timing_ms.append(elapsed_ms)
        self.over_four_ms += elapsed_ms > 4.0
        self.deadline_misses += elapsed_ms > (1000.0 / 60.0)
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
            "conformal_radius": self.conformal_radius,
            "factual_supported_actions": sorted(self.factual_supported_actions),
            "support_abstentions": self.support_abstentions,
            "bound_abstentions": self.bound_abstentions,
            "shadow_proposals": self.shadow_proposals,
            "active_overrides": self.active_overrides,
            "proposals": dict(sorted(self.proposals.items())),
            "scorer_backend": self.scorer_backend,
            "decision_latency_p95_ms": _p95(self.timing_ms),
            "decision_latency_max_ms": max(self.timing_ms, default=None),
            "decision_latency_over_four_ms": self.over_four_ms,
            "controller_deadline_misses": self.deadline_misses,
        }


def create_policy() -> AutonomousDROptionAdvantagePolicy:
    return AutonomousDROptionAdvantagePolicy()
