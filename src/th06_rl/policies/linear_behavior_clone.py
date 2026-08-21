"""Immutable bounded online scorer for the first linear behavior clone."""

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
    linear_action_scores,
    masked_softmax_probabilities,
    normalized_features,
)
from ..policy_api import POLICY_API_VERSION, PolicyDecision


STATE_SCHEMA = "th06-rl-linear-behavior-clone-state-v1"
POLICY_NAME = "linear-behavior-clone-v1"
TRAINING_SCHEMA = "th06-rl-linear-behavior-clone-fit-v1"
DECISION_EPOCH_SCHEMA = "th06-rl-decision-epoch-v1"
TARGET_SCHEMA = "th06-rl-published-executed-action-target-v1"


class LinearBehaviorClonePolicy:
    api_version = POLICY_API_VERSION
    name = POLICY_NAME

    def __init__(self) -> None:
        self.loaded = False
        self.policy_id = POLICY_NAME
        self.mean: tuple[float, ...] = ()
        self.scale: tuple[float, ...] = ()
        self.weights: tuple[tuple[float, ...], ...] = ()
        self.biases: tuple[float, ...] = ()
        self.policy_seed = 0
        self.random = random.Random(0)
        self.decisions = 0
        self.selected: Counter[str] = Counter()

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("linear behavior-clone state schema mismatch")
        if state.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("linear behavior-clone feature schema mismatch")
        if (
            state.get("training_schema") != TRAINING_SCHEMA
            or state.get("decision_epoch_schema") != DECISION_EPOCH_SCHEMA
            or state.get("target_schema") != TARGET_SCHEMA
        ):
            raise ValueError("linear behavior-clone task contract mismatch")
        if tuple(state.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("linear behavior-clone feature names mismatch")
        if tuple(state.get("action_names", ())) != ACTION_NAMES:
            raise ValueError("linear behavior-clone action vocabulary mismatch")
        if state.get("policy_api_version") != POLICY_API_VERSION:
            raise ValueError("linear behavior-clone policy API binding mismatch")
        normalization = state.get("normalization")
        model = state.get("model")
        sampling = state.get("sampling")
        if (
            not isinstance(normalization, dict)
            or not isinstance(model, dict)
            or not isinstance(sampling, dict)
            or model.get("kind") != "masked-linear-softmax"
            or sampling.get("kind") != "seeded-categorical"
        ):
            raise ValueError("linear behavior-clone state lacks model data")
        try:
            mean = tuple(float(value) for value in normalization["mean"])
            scale = tuple(float(value) for value in normalization["scale"])
            weights = tuple(
                tuple(float(value) for value in row)
                for row in model["weights"]
            )
            biases = tuple(float(value) for value in model["biases"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("linear behavior-clone numeric state is malformed") from error
        if (
            len(mean) != len(FEATURE_NAMES)
            or len(scale) != len(FEATURE_NAMES)
            or len(weights) != len(ACTION_NAMES)
            or any(len(row) != len(FEATURE_NAMES) for row in weights)
            or len(biases) != len(ACTION_NAMES)
            or any(
                not math.isfinite(value)
                for value in (
                    *mean,
                    *scale,
                    *biases,
                    *(item for row in weights for item in row),
                )
            )
            or any(value <= 0.0 for value in scale)
        ):
            raise ValueError("linear behavior-clone numeric state is invalid")
        policy_id = state.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("linear behavior-clone policy identity is missing")
        try:
            policy_seed = int(sampling["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("linear behavior-clone policy seed is malformed") from error
        if not 0 <= policy_seed < 2**64:
            raise ValueError("linear behavior-clone policy seed is invalid")
        unhashed_state = dict(state)
        del unhashed_state["policy_id"]
        canonical = json.dumps(
            unhashed_state,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected_policy_id = (
            f"{POLICY_NAME}:{hashlib.sha256(canonical).hexdigest()[:16]}"
        )
        if policy_id != expected_policy_id:
            raise ValueError("linear behavior-clone policy identity hash mismatch")
        self.policy_id = policy_id
        self.name = policy_id
        self.mean = mean
        self.scale = scale
        self.weights = weights
        self.biases = biases
        self.policy_seed = policy_seed
        self.random = random.Random(policy_seed)
        self.loaded = True

    def decide(self, context) -> PolicyDecision:
        if not self.loaded:
            raise RuntimeError("linear behavior clone requires a state file")
        legal = tuple(str(action) for action in context.locally_admissible_actions)
        if not legal:
            raise ValueError("linear behavior clone received an empty shield set")
        features = normalized_features(
            features_from_policy_context(context),
            self.mean,
            self.scale,
        )
        scores = linear_action_scores(features, self.weights, self.biases)
        probabilities = masked_softmax_probabilities(scores, legal)
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
        return PolicyDecision(
            chosen,
            self.policy_id,
            probability,
            probabilities,
        )

    def metrics(self) -> dict[str, object]:
        return {
            "schema": STATE_SCHEMA,
            "feature_schema": FEATURE_SCHEMA,
            "policy_id": self.policy_id,
            "policy_seed": self.policy_seed,
            "decisions": self.decisions,
            "selected": dict(sorted(self.selected.items())),
        }


def create_policy() -> LinearBehaviorClonePolicy:
    return LinearBehaviorClonePolicy()
