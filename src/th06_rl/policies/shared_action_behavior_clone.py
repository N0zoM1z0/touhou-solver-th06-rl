"""Immutable bounded online policy for the shared per-action BC scorer."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import random

from ..actions import ACTION_NAMES
from ..bc_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    features_from_policy_context,
    masked_softmax_probabilities,
)
from ..policy_api import POLICY_API_VERSION, PolicyDecision
from ..shared_action_features import (
    ACTION_FEATURE_NAMES,
    ACTION_FEATURE_SCHEMA,
    action_feature_rows,
    normalized_action_feature_rows,
    shared_action_scores,
)


STATE_SCHEMA = "th06-rl-shared-action-behavior-clone-state-v1"
POLICY_NAME = "shared-action-behavior-clone-v1"
TRAINING_SCHEMA = "th06-rl-shared-action-behavior-clone-fit-v1"
DECISION_EPOCH_SCHEMA = "th06-rl-decision-epoch-v1"
TARGET_SCHEMA = "th06-rl-published-executed-action-target-v1"


class SharedActionBehaviorClonePolicy:
    api_version = POLICY_API_VERSION
    name = POLICY_NAME

    def __init__(self) -> None:
        self.loaded = False
        self.policy_id = POLICY_NAME
        self.mean: tuple[float, ...] = ()
        self.scale: tuple[float, ...] = ()
        self.weights: tuple[float, ...] = ()
        self.policy_seed = 0
        self.random = random.Random(0)
        self.decisions = 0
        self.selected: Counter[str] = Counter()

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("shared-action behavior-clone state schema mismatch")
        if (
            state.get("training_schema") != TRAINING_SCHEMA
            or state.get("decision_epoch_schema") != DECISION_EPOCH_SCHEMA
            or state.get("target_schema") != TARGET_SCHEMA
            or state.get("feature_schema") != FEATURE_SCHEMA
            or state.get("action_feature_schema") != ACTION_FEATURE_SCHEMA
        ):
            raise ValueError("shared-action behavior-clone task contract mismatch")
        if (
            tuple(state.get("feature_names", ())) != FEATURE_NAMES
            or tuple(state.get("action_feature_names", ())) != ACTION_FEATURE_NAMES
            or tuple(state.get("action_names", ())) != ACTION_NAMES
            or state.get("policy_api_version") != POLICY_API_VERSION
        ):
            raise ValueError("shared-action behavior-clone vocabulary mismatch")
        normalization = state.get("normalization")
        model = state.get("model")
        sampling = state.get("sampling")
        if (
            not isinstance(normalization, dict)
            or not isinstance(model, dict)
            or not isinstance(sampling, dict)
            or model.get("kind") != "masked-shared-linear-action-softmax"
            or sampling.get("kind") != "seeded-categorical"
        ):
            raise ValueError("shared-action behavior-clone state lacks model data")
        try:
            mean = tuple(float(value) for value in normalization["mean"])
            scale = tuple(float(value) for value in normalization["scale"])
            weights = tuple(float(value) for value in model["weights"])
            policy_seed = int(sampling["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("shared-action numeric state is malformed") from error
        if (
            len(mean) != len(ACTION_FEATURE_NAMES)
            or len(scale) != len(ACTION_FEATURE_NAMES)
            or len(weights) != len(ACTION_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in (*mean, *scale, *weights))
            or any(value <= 0.0 for value in scale)
            or not 0 <= policy_seed < 2**64
        ):
            raise ValueError("shared-action numeric state is invalid")
        policy_id = state.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("shared-action behavior-clone identity is missing")
        unhashed_state = dict(state)
        del unhashed_state["policy_id"]
        canonical = json.dumps(
            unhashed_state,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected = f"{POLICY_NAME}:{hashlib.sha256(canonical).hexdigest()[:16]}"
        if policy_id != expected:
            raise ValueError("shared-action behavior-clone identity hash mismatch")
        self.policy_id = policy_id
        self.name = policy_id
        self.mean = mean
        self.scale = scale
        self.weights = weights
        self.policy_seed = policy_seed
        self.random = random.Random(policy_seed)
        self.loaded = True

    def decide(self, context) -> PolicyDecision:
        if not self.loaded:
            raise RuntimeError("shared-action behavior clone requires a state file")
        legal = tuple(str(action) for action in context.locally_admissible_actions)
        features = features_from_policy_context(context)
        rows = normalized_action_feature_rows(
            action_feature_rows(features, legal),
            self.mean,
            self.scale,
        )
        probabilities = masked_softmax_probabilities(
            shared_action_scores(rows, self.weights),
            legal,
        )
        draw = self.random.random()
        cumulative = 0.0
        chosen = probabilities[-1][0]
        for action, probability in probabilities:
            cumulative += probability
            if draw < cumulative:
                chosen = action
                break
        probability = dict(probabilities)[chosen]
        if probability <= 0.0:
            raise RuntimeError("sampled an action with zero behavior probability")
        self.decisions += 1
        self.selected[chosen] += 1
        return PolicyDecision(chosen, self.policy_id, probability, probabilities)

    def metrics(self) -> dict[str, object]:
        return {
            "schema": STATE_SCHEMA,
            "feature_schema": FEATURE_SCHEMA,
            "action_feature_schema": ACTION_FEATURE_SCHEMA,
            "policy_id": self.policy_id,
            "policy_seed": self.policy_seed,
            "decisions": self.decisions,
            "selected": dict(sorted(self.selected.items())),
        }


def create_policy() -> SharedActionBehaviorClonePolicy:
    return SharedActionBehaviorClonePolicy()
