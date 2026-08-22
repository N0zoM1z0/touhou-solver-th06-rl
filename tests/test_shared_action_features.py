from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from th06_rl.actions import ACTION_NAMES
from th06_rl.bc_features import FEATURE_NAMES, _features
from th06_rl.policies.shared_action_behavior_clone import (
    DECISION_EPOCH_SCHEMA,
    POLICY_NAME,
    STATE_SCHEMA,
    TARGET_SCHEMA,
    TRAINING_SCHEMA,
    SharedActionBehaviorClonePolicy,
)
from th06_rl.policy_api import POLICY_API_VERSION
from th06_rl.shared_action_features import (
    ACTION_FEATURE_NAMES,
    ACTION_FEATURE_SCHEMA,
    action_feature_rows,
)


def _fixture_features():
    evaluations = tuple(
        (name, 10.0 + index, 100.0 + index, 200.0 - index)
        for index, name in enumerate(ACTION_NAMES)
    )
    return _features(
        player_x=100.0,
        player_y=200.0,
        power=64,
        bullet_count=3,
        laser_count=1,
        current_action="right",
        legal_actions=ACTION_NAMES,
        evaluations=evaluations,
    ), evaluations


def test_shared_rows_expose_only_action_relative_reactive_facts() -> None:
    features, _evaluations = _fixture_features()
    rows = action_feature_rows(features, ACTION_NAMES)
    assert len(rows) == len(ACTION_NAMES)
    assert all(len(row) == len(ACTION_FEATURE_NAMES) for row in rows)
    right = rows[ACTION_NAMES.index("right")]
    assert right[0] == 0.0
    assert right[1] == 14.0
    assert right[2] == 96.0
    assert right[3] == 1.0
    assert right[4] == 0.0
    assert right[5] == 1.0
    lexical = {name: rank for rank, name in enumerate(sorted(ACTION_NAMES))}
    assert right[6] == lexical["right"]


def test_shared_rows_fail_closed_on_a_mask_mismatch() -> None:
    features, _evaluations = _fixture_features()
    with pytest.raises(ValueError, match="legal set differs"):
        action_feature_rows(features, ACTION_NAMES[:-1])


def test_shared_policy_import_hash_and_distribution() -> None:
    features, evaluations = _fixture_features()
    state = {
        "schema": STATE_SCHEMA,
        "training_schema": TRAINING_SCHEMA,
        "decision_epoch_schema": DECISION_EPOCH_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "feature_schema": "th06-rl-current-observation-features-v1",
        "feature_names": list(FEATURE_NAMES),
        "action_feature_schema": ACTION_FEATURE_SCHEMA,
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "policy_api_version": POLICY_API_VERSION,
        "normalization": {
            "mean": [0.0] * len(ACTION_FEATURE_NAMES),
            "scale": [1.0] * len(ACTION_FEATURE_NAMES),
        },
        "model": {
            "kind": "masked-shared-linear-action-softmax",
            "weights": [0.0] * len(ACTION_FEATURE_NAMES),
        },
        "sampling": {"kind": "seeded-categorical", "seed": 0},
    }
    canonical = json.dumps(
        state, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    state["policy_id"] = (
        f"{POLICY_NAME}:{hashlib.sha256(canonical).hexdigest()[:16]}"
    )
    policy = SharedActionBehaviorClonePolicy()
    policy.import_state(state)
    context = SimpleNamespace(
        player_x=100.0,
        player_y=200.0,
        power=64,
        bullet_count=3,
        laser_count=1,
        current_action="right",
        locally_admissible_actions=ACTION_NAMES,
        shield_admissible_actions=ACTION_NAMES,
        shield_action_count=len(ACTION_NAMES),
        shield_action_evaluations=evaluations,
    )
    decision = policy.decide(context)
    assert decision.action in ACTION_NAMES
    assert sum(
        probability for _, probability in decision.behavior_probabilities
    ) == pytest.approx(1.0)
    assert all(
        probability == pytest.approx(1.0 / len(ACTION_NAMES))
        for _, probability in decision.behavior_probabilities
    )
    assert features
