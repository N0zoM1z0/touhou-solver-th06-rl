"""Evidence-gated fixed-member consensus around the frozen Stage 6 UCB.

Every member scores only the incumbent's factual action.  A candidate exists
only when all retained leave-one-physical-run-out models cross the same risk
threshold.  The default schema never changes the published action.  A
separate active schema can only fall back to the already-computed native
reactive baseline and requires strict physical-shadow authorization evidence.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

from ..policy_api import POLICY_API_VERSION, PolicyDecision
from .offline_risk_guard import (
    STATE_SCHEMA as MEMBER_STATE_SCHEMA,
    OfflineRiskGuardPolicy,
)


STATE_SCHEMA = "th06-rl-offline-wine-risk-consensus-shadow-policy-v1"
LEGACY_ACTIVE_STATE_SCHEMA = "th06-rl-offline-wine-risk-consensus-active-policy-v1"
ACTIVE_STATE_SCHEMA = "th06-rl-offline-wine-risk-consensus-active-policy-v2"
MINIMUM_MEMBERS = 3
MAXIMUM_MEMBERS = 16


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class OfflineRiskConsensusPolicy:
    api_version = POLICY_API_VERSION
    name = "offline-wine-risk-consensus-uninitialized"

    def __init__(self) -> None:
        self.members: list[OfflineRiskGuardPolicy] = []
        self.scope: tuple[int, int, int, int] | None = None
        self.threshold = math.inf
        self.loaded_state: dict[str, object] | None = None
        self.decisions = 0
        self.scored_decisions = 0
        self.consensus_candidates = 0
        self.active_fallbacks = 0
        self.mode = "shadow"
        self.intervention_gate = "shadow"
        self.risk_latched = False
        self.minimum_score: float | None = None
        self.maximum_score: float | None = None
        self.minimum_score_sum = 0.0
        self.incumbent_choices: Counter[str] = Counter()

    def import_state(self, state: dict[str, object]) -> None:
        schema = state.get("schema")
        if schema == STATE_SCHEMA:
            mode = "shadow"
            intervention_gate = "shadow"
        elif schema == LEGACY_ACTIVE_STATE_SCHEMA:
            mode = "active"
            intervention_gate = "level-legacy"
        elif schema == ACTIVE_STATE_SCHEMA:
            mode = "active"
            intervention_gate = "rising-edge"
        else:
            raise ValueError("unsupported offline Wine risk-consensus state")
        if state.get("mode") != mode:
            raise ValueError("risk-consensus mode/schema mismatch")
        if mode == "active":
            authorization = state.get("active_authorization")
            if (
                not isinstance(authorization, dict)
                or authorization.get("publication")
                != "native-reactive-baseline-only"
                or int(authorization.get("validation_runs", 0)) < 2
                or int(authorization.get("candidate_positive", 0)) <= 0
                or int(authorization.get("candidate_negative", -1)) != 0
                or float(
                    authorization.get(
                        "precision_lower_bound_95_one_sided", 0.0,
                    )
                ) < 0.90
                or not isinstance(authorization.get("shadow_state_sha256"), str)
                or len(str(authorization["shadow_state_sha256"])) != 64
                or (
                    schema == ACTIVE_STATE_SCHEMA
                    and authorization.get("intervention_gate") != "rising-edge"
                )
            ):
                raise ValueError("risk-consensus active authorization is insufficient")
        consensus = state.get("consensus")
        if not isinstance(consensus, dict) or consensus.get("aggregation") != "minimum":
            raise ValueError("risk consensus requires minimum-score aggregation")
        raw_scope = state.get("scope")
        if not isinstance(raw_scope, list) or len(raw_scope) != 4:
            raise TypeError("risk-consensus scope must contain four integers")
        scope = tuple(int(value) for value in raw_scope)
        threshold = float(state.get("threshold", math.nan))
        if not math.isfinite(threshold):
            raise ValueError("risk-consensus threshold must be finite")
        incumbent_state = state.get("incumbent_state")
        if not isinstance(incumbent_state, dict):
            raise TypeError("risk-consensus incumbent state is absent")
        incumbent_sha = hashlib.sha256(_canonical(incumbent_state)).hexdigest()
        if incumbent_sha != state.get("incumbent_state_sha256"):
            raise ValueError("risk-consensus incumbent state SHA-256 mismatch")
        members = state.get("members")
        if not isinstance(members, list) or not (
            MINIMUM_MEMBERS <= len(members) <= MAXIMUM_MEMBERS
        ):
            raise ValueError("risk consensus has an invalid bounded member count")
        if int(consensus.get("members", -1)) != len(members):
            raise ValueError("risk-consensus member count contract mismatch")

        loaded = []
        portable_hashes = set()
        source_hashes = set()
        feature_schemas = set()
        for index, raw in enumerate(members):
            if not isinstance(raw, dict) or int(raw.get("index", -1)) != index:
                raise TypeError("risk-consensus member order is invalid")
            member_state = {
                "schema": MEMBER_STATE_SCHEMA,
                "mode": "shadow",
                "scope": list(scope),
                "threshold": threshold,
                "source_model_sha256": raw.get("source_model_sha256"),
                "model_codec": raw.get("model_codec"),
                "portable_model_sha256": raw.get("portable_model_sha256"),
                "model_payload": raw.get("model_payload"),
                "incumbent_state_sha256": incumbent_sha,
                "incumbent_state": incumbent_state,
                "native_scorer": state.get("native_scorer"),
            }
            member = OfflineRiskGuardPolicy()
            member.import_state(member_state)
            portable_hashes.add(member.portable_model_sha256)
            source_hashes.add(member.source_model_sha256)
            feature_schemas.add(member.feature_schema)
            loaded.append(member)
        if len(portable_hashes) != len(loaded) or len(source_hashes) != len(loaded):
            raise ValueError("risk-consensus members must be distinct")
        if len(feature_schemas) != 1:
            raise ValueError("risk-consensus members must share one feature schema")

        self.members = loaded
        self.scope = scope  # type: ignore[assignment]
        self.mode = mode
        self.intervention_gate = intervention_gate
        self.threshold = threshold
        self.loaded_state = json.loads(_canonical(state).decode("utf-8"))
        self.name = f"wine-risk-consensus-{mode}-{len(loaded)}x"

    def decide(self, context) -> PolicyDecision:
        if not self.members or self.scope is None:
            raise RuntimeError("risk-consensus state was not loaded")
        if tuple(context.scope) != self.scope:
            raise ValueError(
                f"risk-consensus scope mismatch: expected {self.scope}, got {context.scope}"
            )
        if float(context.exploration_rate) != 0.0:
            raise ValueError("risk consensus requires zero exploration")
        incumbent = self.members[0].incumbent.decide(context)
        if incumbent.action not in context.locally_admissible_actions:
            raise ValueError("risk-consensus incumbent left the native local set")
        if context.baseline_action not in context.locally_admissible_actions:
            raise ValueError("risk-consensus baseline left the native local set")

        raw_candidate = False
        if incumbent.action != context.baseline_action:
            features = self.members[0].features_for_context(
                context, incumbent.action,
            )
            scores = [
                member.score_features(features)
                for member in self.members
            ]
            minimum = min(scores)
            maximum = max(scores)
            raw_candidate = minimum >= self.threshold
            self.scored_decisions += 1
            self.minimum_score_sum += minimum
            self.minimum_score = (
                minimum if self.minimum_score is None
                else min(self.minimum_score, minimum)
            )
            self.maximum_score = (
                maximum if self.maximum_score is None
                else max(self.maximum_score, maximum)
            )
        candidate = raw_candidate
        if self.intervention_gate == "rising-edge":
            candidate = raw_candidate and not self.risk_latched
            self.risk_latched = raw_candidate
        self.decisions += 1
        self.incumbent_choices[incumbent.action] += 1
        if raw_candidate:
            self.consensus_candidates += 1
        selected = (
            context.baseline_action
            if candidate and self.mode == "active"
            else incumbent.action
        )
        if selected != incumbent.action:
            self.active_fallbacks += 1
        suffix = "candidate" if candidate else "pass"
        return PolicyDecision(
            selected,
            f"{self.name}-{suffix}",
            (
                1.0 if selected != incumbent.action
                else incumbent.behavior_probability
            ),
        )

    def export_state(self) -> dict[str, object]:
        if self.loaded_state is None:
            raise RuntimeError("risk-consensus state was not loaded")
        return json.loads(_canonical(self.loaded_state).decode("utf-8"))

    def metrics(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "aggregation": "minimum",
            "intervention_gate": self.intervention_gate,
            "members": len(self.members),
            "threshold": self.threshold,
            "decisions": self.decisions,
            "scored_decisions": self.scored_decisions,
            "consensus_candidates": self.consensus_candidates,
            "active_fallbacks": self.active_fallbacks,
            "mean_minimum_score": (
                self.minimum_score_sum / self.scored_decisions
                if self.scored_decisions else None
            ),
            "minimum_score": self.minimum_score,
            "maximum_member_score": self.maximum_score,
            "scope": list(self.scope) if self.scope is not None else None,
            "scorer_backends": [member.scorer_backend for member in self.members],
            "portable_model_sha256": [
                member.portable_model_sha256 for member in self.members
            ],
            "incumbent_choices": dict(self.incumbent_choices),
            "incumbent": (
                self.members[0].incumbent.metrics() if self.members else None
            ),
        }


def create_policy() -> OfflineRiskConsensusPolicy:
    return OfflineRiskConsensusPolicy()
