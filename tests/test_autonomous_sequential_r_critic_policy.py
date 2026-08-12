from __future__ import annotations

from copy import deepcopy

from th06_rl.advantage_learning import (
    HAZARD_CODEBOOK_SCHEMA,
    _encoded_model,
    hazard_codebook_feature_names,
    rich_feature_names,
)
from th06_rl.hazard_representation import (
    HAZARD_PRIMITIVE_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
)
from th06_rl.offline import ACTION_NAMES
from th06_rl.policies.autonomous_sequential_r_critic import (
    AutonomousSequentialRCriticPolicy,
)
from th06_rl.policies.offline_ranker import MODEL_SCHEMA
from th06_rl.policy_api import PolicyContext
from th06_rl.sequential_learning import RICH_FEATURE_SCHEMA, STATE_SCHEMA
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)
from tests.test_learning_features import _projection


def _model(candidate_leaf: float) -> dict[str, object]:
    names = rich_feature_names()
    direction = names.index("action:direction_x")
    zero = [0.0] * len(names)
    return _encoded_model({
        "schema": MODEL_SCHEMA,
        "feature_schema": RICH_FEATURE_SCHEMA,
        "feature_names": list(names),
        "base_score": 0.0,
        "trees": [[
            [direction, -0.5, 1, 2, 2, 0.0],
            [-1, 0.0, -1, -1, -1, candidate_leaf],
            [-1, 0.0, -1, -1, -1, 0.0],
        ]],
        "conformance": [{"features": zero, "prediction": 0.0}],
    })


def _state(mode: str, *, optimistic_member: bool = False) -> dict[str, object]:
    names = rich_feature_names()
    zero = [0.0] * len(names)
    codebook = {
        "schema": HAZARD_CODEBOOK_SCHEMA,
        "primitive_feature_names": list(HAZARD_PRIMITIVE_FEATURE_NAMES),
        "maximum_primitives": 256,
        "prototype_count": 24,
        "mean": [0.0] * len(HAZARD_PRIMITIVE_FEATURE_NAMES),
        "scale": [1.0] * len(HAZARD_PRIMITIVE_FEATURE_NAMES),
        "prototypes": [
            [0.0] * len(HAZARD_PRIMITIVE_FEATURE_NAMES) for _ in range(24)
        ],
        "conformance": [{
            "primitives": [],
            "encoding": [
                0.0
            ] * (len(hazard_codebook_feature_names()) - 1) + [1.0],
        }],
    }
    models = [_model(-1.0) for _ in range(7)]
    if optimistic_member:
        models[-1] = _model(0.2)
    return {
        "schema": STATE_SCHEMA,
        "mode": mode,
        "feature_schema": RICH_FEATURE_SCHEMA,
        "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "feature_names": list(names),
        "models": models,
        "representation": {
            "kind": "learned-permutation-invariant-hazard-codebook-plus-factual-history",
            "hazard_codebook": codebook,
            "history_feature_names": list(HISTORY_FEATURE_NAMES),
        },
        "support": {
            "mean": zero,
            "scale": [1.0] * len(names),
            "prototypes": [[zero] for _action in ACTION_NAMES],
            "threshold": 1e9,
            "factual_supported_actions": ["stay", "left"],
        },
        "selection": {
            "rule": "all-members-negative-relative-to-incumbent",
            "baseline_advantage": 0.0,
        },
        "population": {
            "kind": "whole-episode-bootstrap-action-centered-r-critic",
            "members": 7,
            "trees_per_member": 128,
        },
        "native_scorer": {
            "sha256": "a" * 64,
            "compatible_sha256": ["a" * 64],
        },
        "authorization": {
            "fit_eligible": True,
            "policy_calibration": {"relative_r_loss": 0.8},
            "active_canary": {"audit": "bound"} if mode == "active" else None,
        },
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
        hazard_primitives=(),
        history_features=tuple((name, 0.0) for name in HISTORY_FEATURE_NAMES),
    )


def test_active_full_population_requires_every_member_negative() -> None:
    policy = AutonomousSequentialRCriticPolicy()
    policy.import_state(_state("active"))
    assert policy.decide(_context()).action == "left"
    assert policy.metrics()["active_overrides"] == 1


def test_shadow_full_population_never_publishes_proposal() -> None:
    policy = AutonomousSequentialRCriticPolicy()
    policy.import_state(_state("shadow"))
    assert policy.decide(_context()).action == "stay"
    assert policy.metrics()["shadow_proposals"] == 1


def test_one_optimistic_member_forces_population_abstention() -> None:
    policy = AutonomousSequentialRCriticPolicy()
    policy.import_state(deepcopy(_state("active", optimistic_member=True)))
    assert policy.decide(_context()).action == "stay"
    assert policy.metrics()["population_abstentions"] == 1
