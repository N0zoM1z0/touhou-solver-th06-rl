"""Factual Wine risk guard around the frozen UCB incumbent.

The model scores only the action selected by the incumbent.  It cannot add an
action to the native-gated set.  In shadow mode it never changes publication;
in active mode its only possible intervention is the already-computed native
reactive baseline.
"""

from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import zlib

from ..policy_api import POLICY_API_VERSION, PolicyDecision
from ..wine_risk import (
    RISK_CATEGORICAL_FEATURES,
    RISK_FEATURE_NAMES,
    RISK_FEATURE_SCHEMA,
    risk_feature_contract,
    risk_features_for_context,
)
from .adaptive import (
    AdaptivePolicy,
    LEGACY_STATE_SCHEMAS as INCUMBENT_LEGACY_SCHEMAS,
    REWARD_VERSION as INCUMBENT_REWARD_VERSION,
    STATE_SCHEMA as INCUMBENT_STATE_SCHEMA,
    unpack_state,
)
from .offline_ranker import (
    MODEL_CODEC,
    NATIVE_SCORER_ENV,
    NATIVE_SCORER_SCHEMA,
    NativeXGBoostRegressor,
    PortableXGBoostRegressor,
)


STATE_SCHEMA = "th06-rl-offline-wine-risk-guard-policy-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class OfflineRiskGuardPolicy:
    api_version = POLICY_API_VERSION
    name = "offline-wine-risk-guard-uninitialized"

    def __init__(self) -> None:
        self.incumbent = AdaptivePolicy()
        self.model: PortableXGBoostRegressor | None = None
        self.scorer: PortableXGBoostRegressor | NativeXGBoostRegressor | None = None
        self.encoder: dict[str, dict[str, int]] = {}
        self.feature_schema = RISK_FEATURE_SCHEMA
        self.feature_names = RISK_FEATURE_NAMES
        self.categorical_features = RISK_CATEGORICAL_FEATURES
        self.scope: tuple[int, int, int, int] | None = None
        self.mode = "shadow"
        self.threshold = math.inf
        self.source_model_sha256 = ""
        self.portable_model_sha256 = ""
        self.scorer_backend = "uninitialized"
        self.loaded_state: dict[str, object] | None = None
        self.decisions = 0
        self.scored_decisions = 0
        self.shadow_candidates = 0
        self.active_fallbacks = 0
        self.incumbent_choices: Counter[str] = Counter()
        self.selected: Counter[str] = Counter()
        self.maximum_score: float | None = None
        self.score_sum = 0.0

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("unsupported offline Wine risk-guard state")
        mode = state.get("mode")
        if mode not in ("shadow", "active"):
            raise ValueError("risk-guard mode must be shadow or active")
        raw_scope = state.get("scope")
        if not isinstance(raw_scope, list) or len(raw_scope) != 4:
            raise TypeError("risk-guard scope must contain four integers")
        scope = tuple(int(value) for value in raw_scope)
        threshold = float(state.get("threshold", math.nan))
        if not math.isfinite(threshold):
            raise ValueError("risk-guard threshold must be finite")

        incumbent_state = state.get("incumbent_state")
        if not isinstance(incumbent_state, dict):
            raise TypeError("risk-guard incumbent state is absent")
        incumbent_sha = hashlib.sha256(_canonical(incumbent_state)).hexdigest()
        if incumbent_sha != state.get("incumbent_state_sha256"):
            raise ValueError("risk-guard incumbent state SHA-256 mismatch")
        unpacked = unpack_state(incumbent_state)
        if (
            unpacked.get("schema")
            not in (INCUMBENT_STATE_SCHEMA, *INCUMBENT_LEGACY_SCHEMAS)
            or unpacked.get("reward_version") != INCUMBENT_REWARD_VERSION
        ):
            raise ValueError("risk-guard incumbent is not the supported frozen UCB")
        incumbent = AdaptivePolicy()
        incumbent.import_state(incumbent_state)

        if state.get("model_codec") != MODEL_CODEC:
            raise ValueError("unsupported portable risk-model codec")
        payload = state.get("model_payload")
        if not isinstance(payload, str):
            raise TypeError("portable risk-model payload must be text")
        decoded = zlib.decompress(base64.b64decode(payload, validate=True))
        portable_sha = hashlib.sha256(decoded).hexdigest()
        if portable_sha != state.get("portable_model_sha256"):
            raise ValueError("portable risk-model SHA-256 mismatch")
        artifact = json.loads(decoded.decode("utf-8"))
        if not isinstance(artifact, dict):
            raise TypeError("portable risk-model root must be an object")
        feature_schema = str(artifact.get("feature_schema", ""))
        feature_names, categorical_features = risk_feature_contract(feature_schema)
        if tuple(artifact.get("categorical_features", ())) != categorical_features:
            raise ValueError("portable risk-model categorical schema mismatch")
        raw_encoder = artifact.get("encoder")
        if not isinstance(raw_encoder, dict):
            raise TypeError("portable risk-model encoder is absent")
        encoder: dict[str, dict[str, int]] = {}
        for feature in categorical_features:
            values = raw_encoder.get(feature)
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values))
            ):
                raise TypeError(f"portable risk encoder {feature!r} is invalid")
            encoder[feature] = {
                value: index for index, value in enumerate(values)
            }
        model = PortableXGBoostRegressor(
            artifact,
            expected_feature_schema=feature_schema,
            expected_feature_names=feature_names,
        )
        conformance = artifact.get("conformance")
        if not isinstance(conformance, list) or not conformance:
            raise TypeError("portable risk-model conformance vectors are absent")
        vectors = []
        for row in conformance:
            if not isinstance(row, dict) or not isinstance(row.get("features"), list):
                raise TypeError("portable risk-model conformance vector is invalid")
            features = [float(value) for value in row["features"]]
            actual = model.predict(features)
            expected = float(row["prediction"])
            if not math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-5):
                raise ValueError(
                    "portable risk-model conformance failed: "
                    f"expected {expected}, observed {actual}"
                )
            vectors.append(features)

        scorer: PortableXGBoostRegressor | NativeXGBoostRegressor = model
        scorer_backend = "python-portable"
        native_contract = state.get("native_scorer")
        if native_contract is not None:
            if (
                not isinstance(native_contract, dict)
                or native_contract.get("schema") != NATIVE_SCORER_SCHEMA
            ):
                raise ValueError("unsupported native risk-scorer contract")
            expected_sha = native_contract.get("sha256")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise ValueError("native risk-scorer SHA-256 is invalid")
            scorer_path = os.environ.get(NATIVE_SCORER_ENV)
            if scorer_path:
                scorer = NativeXGBoostRegressor(
                    Path(scorer_path),
                    expected_sha256=expected_sha,
                    portable=model,
                )
                for actual, row in zip(
                    scorer.predict_many(vectors), conformance, strict=True,
                ):
                    expected = float(row["prediction"])
                    if not math.isclose(
                        actual, expected, rel_tol=2e-5, abs_tol=2e-5,
                    ):
                        raise ValueError(
                            "native risk-scorer conformance failed: "
                            f"expected {expected}, observed {actual}"
                        )
                scorer_backend = "native-batch"
            elif os.name == "nt":
                raise RuntimeError(
                    f"Windows risk guard requires {NATIVE_SCORER_ENV}"
                )
        elif os.name == "nt":
            raise RuntimeError("Windows risk guard requires a native scorer contract")

        source_model_sha = state.get("source_model_sha256")
        if not isinstance(source_model_sha, str) or len(source_model_sha) != 64:
            raise ValueError("risk-guard source model SHA-256 is invalid")
        self.incumbent = incumbent
        self.model = model
        self.scorer = scorer
        self.encoder = encoder
        self.feature_schema = feature_schema
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.scope = scope  # type: ignore[assignment]
        self.mode = str(mode)
        self.threshold = threshold
        self.source_model_sha256 = source_model_sha
        self.portable_model_sha256 = portable_sha
        self.scorer_backend = scorer_backend
        self.loaded_state = json.loads(_canonical(state).decode("utf-8"))
        self.name = f"wine-risk-guard-{mode}-{source_model_sha[:12]}"

    def _encode_features(self, features: dict[str, str | float]) -> list[float]:
        row = []
        categorical = set(self.categorical_features)
        for name in self.feature_names:
            value = features[name]
            if name in categorical:
                row.append(float(self.encoder[name].get(str(value), -1)))
            else:
                row.append(float(value))
        return row

    def _encoded_row(self, context, action: str) -> list[float]:
        return self._encode_features(self.features_for_context(context, action))

    def features_for_context(
        self, context, action: str,
    ) -> dict[str, str | float]:
        return risk_features_for_context(
            context, action, feature_schema=self.feature_schema,
        )

    def score_features(self, features: dict[str, str | float]) -> float:
        """Score one already-constructed factual feature row."""
        if self.scorer is None:
            raise RuntimeError("risk-guard state was not loaded")
        score = self.scorer.predict_many([
            self._encode_features(features),
        ])[0]
        if not math.isfinite(score):
            raise ValueError("risk-guard score is not finite")
        return score

    def score_action(self, context, action: str) -> float:
        """Score one factual action without selecting or publishing an action."""
        return self.score_features(self.features_for_context(context, action))

    def decide(self, context) -> PolicyDecision:
        if self.scorer is None or self.scope is None:
            raise RuntimeError("risk-guard state was not loaded")
        if tuple(context.scope) != self.scope:
            raise ValueError(
                f"risk-guard scope mismatch: expected {self.scope}, got {context.scope}"
            )
        if float(context.exploration_rate) != 0.0:
            raise ValueError("risk guard requires a frozen zero-exploration incumbent")
        incumbent = self.incumbent.decide(context)
        if incumbent.action not in context.locally_admissible_actions:
            raise ValueError("risk-guard incumbent left the native local set")
        candidate = False
        if incumbent.action != context.baseline_action:
            score = self.score_action(context, incumbent.action)
            self.scored_decisions += 1
            self.score_sum += score
            self.maximum_score = (
                score if self.maximum_score is None else max(self.maximum_score, score)
            )
            candidate = score >= self.threshold
        selected = (
            context.baseline_action
            if candidate and self.mode == "active"
            else incumbent.action
        )
        self.decisions += 1
        self.incumbent_choices[incumbent.action] += 1
        self.selected[selected] += 1
        if candidate:
            if self.mode == "shadow":
                self.shadow_candidates += 1
            else:
                self.active_fallbacks += 1
        suffix = "candidate" if candidate else "pass"
        probability = 1.0 if selected != incumbent.action else incumbent.behavior_probability
        return PolicyDecision(selected, f"{self.name}-{suffix}", probability)

    def export_state(self) -> dict[str, object]:
        if self.loaded_state is None:
            raise RuntimeError("risk-guard state was not loaded")
        return json.loads(_canonical(self.loaded_state).decode("utf-8"))

    def metrics(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "decisions": self.decisions,
            "scored_decisions": self.scored_decisions,
            "shadow_candidates": self.shadow_candidates,
            "active_fallbacks": self.active_fallbacks,
            "mean_score": (
                self.score_sum / self.scored_decisions
                if self.scored_decisions else None
            ),
            "maximum_score": self.maximum_score,
            "scorer_backend": self.scorer_backend,
            "scope": list(self.scope) if self.scope is not None else None,
            "source_model_sha256": self.source_model_sha256,
            "portable_model_sha256": self.portable_model_sha256,
            "incumbent_choices": dict(self.incumbent_choices),
            "selected": dict(self.selected),
            "incumbent": self.incumbent.metrics(),
        }


def create_policy() -> OfflineRiskGuardPolicy:
    return OfflineRiskGuardPolicy()
