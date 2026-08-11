from __future__ import annotations

import base64
import hashlib
import json
import zlib

from th06_rl.conservative_learning import MODEL_CODEC, STATE_SCHEMA
from th06_rl.learning_features import TREE_FEATURE_SCHEMA, tree_feature_names
from th06_rl.offline import ACTION_NAMES
from th06_rl.policies.autonomous_conservative_q import (
    AutonomousConservativeQPolicy,
)
from th06_rl.policies.offline_ranker import MODEL_SCHEMA
from th06_rl.policy_api import PolicyContext
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)
from tests.test_learning_features import _projection


def _state(mode: str) -> dict[str, object]:
    names = tree_feature_names(OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES)
    direction = names.index("action:direction_x")
    tree = [
        [direction, -0.5, 1, 2, 2, 0.0],
        [-1, 0.0, -1, -1, -1, 0.0],
        [-1, 0.0, -1, -1, -1, 2.0],
    ]
    zero = [0.0] * len(names)
    artifact = {
        "schema": MODEL_SCHEMA,
        "feature_schema": TREE_FEATURE_SCHEMA,
        "feature_names": list(names),
        "base_score": 0.0,
        "trees": [tree],
        "conformance": [{"features": zero, "prediction": 2.0}],
    }
    raw = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    encoded = {
        "codec": MODEL_CODEC,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": base64.b64encode(zlib.compress(raw)).decode(),
    }
    authorization = {
        "fit_eligible": True,
        "active_canary": {"shadow_audit_sha256": "a" * 64}
        if mode == "active" else None,
    }
    return {
        "schema": STATE_SCHEMA,
        "mode": mode,
        "feature_schema": TREE_FEATURE_SCHEMA,
        "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "feature_names": list(names),
        "models": [encoded, encoded, encoded],
        "support": {
            "mean": zero,
            "scale": [1.0] * len(names),
            "prototypes": [[zero] for _action in ACTION_NAMES],
            "threshold": 1e9,
        },
        "selection": {
            "uncertainty_scale": 1.0,
            "active_override_budget": 8,
        },
        "authorization": authorization,
    }


def _context() -> PolicyContext:
    observation, actions = _projection()
    return PolicyContext(
        frame=1,
        scope=(3, 0, 0, 6),
        source_context="hidden",
        baseline_action="stay",
        locally_admissible_actions=("stay", "left"),
        player_x=192.0,
        player_y=400.0,
        power=64,
        bullet_count=320,
        laser_count=1,
        hard_action_count=2,
        exploration_rate=0.0,
        current_action="stay",
        observation_features=observation,
        action_features=actions,
    )


def test_active_conservative_q_uses_unanimous_lower_cost_action() -> None:
    policy = AutonomousConservativeQPolicy()
    policy.import_state(_state("active"))
    assert policy.decide(_context()).action == "left"
    assert policy.metrics()["active_overrides"] == 1


def test_shadow_conservative_q_never_publishes_proposal() -> None:
    policy = AutonomousConservativeQPolicy()
    policy.import_state(_state("shadow"))
    assert policy.decide(_context()).action == "stay"
    assert policy.metrics()["shadow_proposals"] == 1
