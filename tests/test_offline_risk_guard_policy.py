from __future__ import annotations

import base64
import hashlib
import json
import zlib

from th06_rl.offline import ACTION_NAMES
from th06_rl.policies.adaptive import REWARD_VERSION, STATE_SCHEMA as UCB_STATE_SCHEMA
from th06_rl.policies.offline_ranker import MODEL_CODEC, MODEL_SCHEMA
from th06_rl.policies.offline_risk_guard import (
    STATE_SCHEMA,
    OfflineRiskGuardPolicy,
)
from th06_rl.policies.offline_risk_consensus import (
    ACTIVE_STATE_SCHEMA as CONSENSUS_ACTIVE_STATE_SCHEMA,
    STATE_SCHEMA as CONSENSUS_STATE_SCHEMA,
    OfflineRiskConsensusPolicy,
)
from th06_rl.policy_api import PolicyContext
from th06_rl.wine_risk import (
    RISK_CATEGORICAL_FEATURES,
    RISK_CATEGORICAL_FEATURES_V2,
    RISK_FEATURE_NAMES,
    RISK_FEATURE_NAMES_V2,
    RISK_FEATURE_SCHEMA,
    RISK_FEATURE_SCHEMA_V2,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _context() -> PolicyContext:
    return PolicyContext(
        frame=100,
        scope=(3, 0, 0, 6),
        source_context="boss:test",
        baseline_action="stay",
        locally_admissible_actions=("stay", "right"),
        player_x=192.0,
        player_y=400.0,
        power=128,
        bullet_count=42,
        laser_count=0,
        hard_action_count=2,
        exploration_rate=0.0,
        current_action="stay",
        hard_admissible_actions=("stay", "right"),
        phase_elapsed_frames=20,
        hard_action_evaluations=(
            ("stay", None, 192.0, 400.0),
            ("right", None, 194.0, 400.0),
        ),
    )


def _incumbent_state() -> dict[str, object]:
    from th06_rl.policies.adaptive import AdaptivePolicy

    policy = AdaptivePolicy()
    context = _context()
    coarse = policy._context_key(context)
    middle = policy._middle_context_key(context)
    fine = policy._fine_context_key(context)
    coarse_stay = policy._action_key(coarse, "stay")
    coarse_right = policy._action_key(coarse, "right")
    middle_stay = policy._action_key(middle, "stay")
    middle_right = policy._action_key(middle, "right")
    fine_stay = policy._action_key(fine, "stay")
    fine_right = policy._action_key(fine, "right")
    return {
        "schema": UCB_STATE_SCHEMA,
        "reward_version": REWARD_VERSION,
        "decisions": 0,
        "trials": {coarse_stay: 1, coarse_right: 1},
        "reward_sum": {coarse_stay: 0.0, coarse_right: 10.0},
        "middle_trials": {middle_stay: 1, middle_right: 1},
        "middle_reward_sum": {middle_stay: 0.0, middle_right: 10.0},
        "fine_trials": {fine_stay: 1, fine_right: 1},
        "fine_reward_sum": {fine_stay: 0.0, fine_right: 10.0},
    }


def _state(
    mode: str,
    *,
    prediction: float = 1.0,
    source_character: str = "a",
    feature_schema: str = RISK_FEATURE_SCHEMA,
) -> dict[str, object]:
    if feature_schema == RISK_FEATURE_SCHEMA_V2:
        feature_names = RISK_FEATURE_NAMES_V2
        categorical_features = RISK_CATEGORICAL_FEATURES_V2
    else:
        feature_names = RISK_FEATURE_NAMES
        categorical_features = RISK_CATEGORICAL_FEATURES
    categories = {
        name: (["boss:test"] if name == "source_context" else list(ACTION_NAMES))
        for name in categorical_features
    }
    artifact = {
        "schema": MODEL_SCHEMA,
        "feature_schema": feature_schema,
        "feature_names": list(feature_names),
        "categorical_features": list(categorical_features),
        "encoder": categories,
        "base_score": 0.0,
        "trees": [[[-1, 0.0, -1, -1, -1, prediction]]],
        "conformance": [{
            "features": [0.0] * len(feature_names),
            "prediction": prediction,
        }],
    }
    decoded = _canonical(artifact)
    incumbent = _incumbent_state()
    return {
        "schema": STATE_SCHEMA,
        "mode": mode,
        "scope": [3, 0, 0, 6],
        "threshold": 0.5,
        "source_model_sha256": source_character * 64,
        "model_codec": MODEL_CODEC,
        "portable_model_sha256": hashlib.sha256(decoded).hexdigest(),
        "model_payload": base64.b64encode(zlib.compress(decoded)).decode(),
        "incumbent_state_sha256": hashlib.sha256(_canonical(incumbent)).hexdigest(),
        "incumbent_state": incumbent,
    }


def test_shadow_risk_guard_scores_incumbent_without_changing_it() -> None:
    policy = OfflineRiskGuardPolicy()
    policy.import_state(_state("shadow"))

    decision = policy.decide(_context())

    assert decision.action == "right"
    assert decision.policy_id.endswith("-candidate")
    assert policy.metrics()["shadow_candidates"] == 1
    assert policy.metrics()["active_fallbacks"] == 0


def test_active_risk_guard_can_only_fallback_to_native_baseline() -> None:
    policy = OfflineRiskGuardPolicy()
    policy.import_state(_state("active"))

    decision = policy.decide(_context())

    assert decision.action == "stay"
    assert policy.metrics()["shadow_candidates"] == 0
    assert policy.metrics()["active_fallbacks"] == 1


def test_shadow_risk_guard_supports_context_reactive_v2_features() -> None:
    policy = OfflineRiskGuardPolicy()
    policy.import_state(_state("shadow", feature_schema=RISK_FEATURE_SCHEMA_V2))

    decision = policy.decide(_context())

    assert decision.action == "right"
    assert decision.policy_id.endswith("-candidate")
    assert policy.feature_schema == RISK_FEATURE_SCHEMA_V2


def _consensus_state() -> dict[str, object]:
    incumbent = _incumbent_state()
    members = []
    for index, (prediction, character) in enumerate(
        ((1.0, "a"), (0.9, "b"), (0.8, "c")),
    ):
        member = _state(
            "shadow", prediction=prediction, source_character=character,
        )
        members.append({
            "index": index,
            "source_model_sha256": member["source_model_sha256"],
            "model_codec": member["model_codec"],
            "portable_model_sha256": member["portable_model_sha256"],
            "model_payload": member["model_payload"],
        })
    return {
        "schema": CONSENSUS_STATE_SCHEMA,
        "mode": "shadow",
        "scope": [3, 0, 0, 6],
        "threshold": 0.75,
        "consensus": {"aggregation": "minimum", "members": len(members)},
        "members": members,
        "incumbent_state_sha256": hashlib.sha256(
            _canonical(incumbent),
        ).hexdigest(),
        "incumbent_state": incumbent,
    }


def test_risk_consensus_requires_every_fixed_member_to_cross_threshold() -> None:
    policy = OfflineRiskConsensusPolicy()
    policy.import_state(_consensus_state())

    decision = policy.decide(_context())

    assert decision.action == "right"
    assert decision.policy_id.endswith("-candidate")
    assert policy.metrics()["consensus_candidates"] == 1
    assert policy.metrics()["members"] == 3


def test_risk_consensus_v1_refuses_active_mode() -> None:
    state = _consensus_state()
    state["mode"] = "active"

    policy = OfflineRiskConsensusPolicy()

    import pytest

    with pytest.raises(ValueError, match="mode/schema mismatch"):
        policy.import_state(state)


def test_active_risk_consensus_can_only_fallback_to_native_baseline() -> None:
    state = _consensus_state()
    state["schema"] = CONSENSUS_ACTIVE_STATE_SCHEMA
    state["mode"] = "active"
    state["active_authorization"] = {
        "publication": "native-reactive-baseline-only",
        "intervention_gate": "rising-edge",
        "validation_runs": 2,
        "candidate_positive": 46,
        "candidate_negative": 0,
        "precision_lower_bound_95_one_sided": 0.94,
        "shadow_state_sha256": "d" * 64,
    }
    policy = OfflineRiskConsensusPolicy()
    policy.import_state(state)

    decision = policy.decide(_context())

    assert decision.action == "stay"
    assert decision.policy_id.endswith("-candidate")
    assert policy.metrics()["active_fallbacks"] == 1

    second = policy.decide(_context())

    assert second.action == "right"
    assert second.policy_id.endswith("-pass")
    assert policy.metrics()["active_fallbacks"] == 1
