"""Immutable bounded online scorer for the fixed small behavior clone."""

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
    normalized_features,
)
from ..policy_api import POLICY_API_VERSION, PolicyDecision


STATE_SCHEMA = "th06-rl-small-mlp-behavior-clone-state-v1"
POLICY_NAME = "small-mlp-behavior-clone-v1"
TRAINING_SCHEMA = "th06-rl-small-mlp-behavior-clone-fit-v1"
DECISION_EPOCH_SCHEMA = "th06-rl-decision-epoch-v1"
TARGET_SCHEMA = "th06-rl-published-executed-action-target-v1"
HIDDEN_WIDTH = 32


def _dense_relu(
    features: tuple[float, ...],
    weights: tuple[tuple[float, ...], ...],
    biases: tuple[float, ...],
) -> tuple[float, ...]:
    if len(weights) != len(biases):
        raise ValueError("MLP layer weights and biases disagree")
    values = []
    for row, bias in zip(weights, biases, strict=True):
        if len(row) != len(features):
            raise ValueError("MLP layer width disagrees with its input")
        value = float(bias) + sum(
            weight * feature
            for weight, feature in zip(row, features, strict=True)
        )
        if not math.isfinite(value):
            raise ValueError("MLP hidden activation is not finite")
        values.append(max(0.0, value))
    return tuple(values)


def _dense_scores(
    features: tuple[float, ...],
    weights: tuple[tuple[float, ...], ...],
    biases: tuple[float, ...],
) -> tuple[float, ...]:
    if len(weights) != len(ACTION_NAMES) or len(biases) != len(ACTION_NAMES):
        raise ValueError("MLP output does not cover the canonical action set")
    scores = []
    for row, bias in zip(weights, biases, strict=True):
        if len(row) != len(features):
            raise ValueError("MLP output width disagrees with its hidden layer")
        score = float(bias) + sum(
            weight * feature
            for weight, feature in zip(row, features, strict=True)
        )
        if not math.isfinite(score):
            raise ValueError("MLP action score is not finite")
        scores.append(score)
    return tuple(scores)


class SmallMlpBehaviorClonePolicy:
    api_version = POLICY_API_VERSION
    name = POLICY_NAME

    def __init__(self) -> None:
        self.loaded = False
        self.policy_id = POLICY_NAME
        self.mean: tuple[float, ...] = ()
        self.scale: tuple[float, ...] = ()
        self.input_weights: tuple[tuple[float, ...], ...] = ()
        self.hidden_biases: tuple[float, ...] = ()
        self.output_weights: tuple[tuple[float, ...], ...] = ()
        self.output_biases: tuple[float, ...] = ()
        self.policy_seed = 0
        self.random = random.Random(0)
        self.decisions = 0
        self.selected: Counter[str] = Counter()

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("small MLP behavior-clone state schema mismatch")
        if state.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("small MLP behavior-clone feature schema mismatch")
        if (
            state.get("training_schema") != TRAINING_SCHEMA
            or state.get("decision_epoch_schema") != DECISION_EPOCH_SCHEMA
            or state.get("target_schema") != TARGET_SCHEMA
        ):
            raise ValueError("small MLP behavior-clone task contract mismatch")
        if tuple(state.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("small MLP behavior-clone feature names mismatch")
        if tuple(state.get("action_names", ())) != ACTION_NAMES:
            raise ValueError("small MLP behavior-clone action vocabulary mismatch")
        if state.get("policy_api_version") != POLICY_API_VERSION:
            raise ValueError("small MLP behavior-clone API binding mismatch")
        normalization = state.get("normalization")
        model = state.get("model")
        sampling = state.get("sampling")
        if (
            not isinstance(normalization, dict)
            or not isinstance(model, dict)
            or not isinstance(sampling, dict)
            or model.get("kind") != "masked-one-hidden-relu-softmax"
            or model.get("hidden_width") != HIDDEN_WIDTH
            or sampling.get("kind") != "seeded-categorical"
        ):
            raise ValueError("small MLP behavior-clone state lacks model data")
        try:
            mean = tuple(float(value) for value in normalization["mean"])
            scale = tuple(float(value) for value in normalization["scale"])
            input_weights = tuple(
                tuple(float(value) for value in row)
                for row in model["input_weights"]
            )
            hidden_biases = tuple(float(value) for value in model["hidden_biases"])
            output_weights = tuple(
                tuple(float(value) for value in row)
                for row in model["output_weights"]
            )
            output_biases = tuple(float(value) for value in model["output_biases"])
            policy_seed = int(sampling["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("small MLP behavior-clone numeric state is malformed") from error
        numeric = (
            *mean,
            *scale,
            *hidden_biases,
            *output_biases,
            *(value for row in input_weights for value in row),
            *(value for row in output_weights for value in row),
        )
        if (
            len(mean) != len(FEATURE_NAMES)
            or len(scale) != len(FEATURE_NAMES)
            or len(input_weights) != HIDDEN_WIDTH
            or any(len(row) != len(FEATURE_NAMES) for row in input_weights)
            or len(hidden_biases) != HIDDEN_WIDTH
            or len(output_weights) != len(ACTION_NAMES)
            or any(len(row) != HIDDEN_WIDTH for row in output_weights)
            or len(output_biases) != len(ACTION_NAMES)
            or any(not math.isfinite(value) for value in numeric)
            or any(value <= 0.0 for value in scale)
            or not 0 <= policy_seed < 2**64
        ):
            raise ValueError("small MLP behavior-clone numeric state is invalid")
        policy_id = state.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("small MLP behavior-clone identity is missing")
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
            raise ValueError("small MLP behavior-clone identity hash mismatch")
        self.policy_id = policy_id
        self.name = policy_id
        self.mean = mean
        self.scale = scale
        self.input_weights = input_weights
        self.hidden_biases = hidden_biases
        self.output_weights = output_weights
        self.output_biases = output_biases
        self.policy_seed = policy_seed
        self.random = random.Random(policy_seed)
        self.loaded = True

    def decide(self, context) -> PolicyDecision:
        if not self.loaded:
            raise RuntimeError("small MLP behavior clone requires a state file")
        legal = tuple(str(action) for action in context.locally_admissible_actions)
        if not legal:
            raise ValueError("small MLP behavior clone received an empty shield set")
        features = normalized_features(
            features_from_policy_context(context),
            self.mean,
            self.scale,
        )
        hidden = _dense_relu(features, self.input_weights, self.hidden_biases)
        scores = _dense_scores(hidden, self.output_weights, self.output_biases)
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
        return PolicyDecision(chosen, self.policy_id, probability, probabilities)

    def metrics(self) -> dict[str, object]:
        return {
            "schema": STATE_SCHEMA,
            "feature_schema": FEATURE_SCHEMA,
            "policy_id": self.policy_id,
            "policy_seed": self.policy_seed,
            "decisions": self.decisions,
            "selected": dict(sorted(self.selected.items())),
        }


def create_policy() -> SmallMlpBehaviorClonePolicy:
    return SmallMlpBehaviorClonePolicy()
