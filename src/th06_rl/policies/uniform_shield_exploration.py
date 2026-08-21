"""Simple identifiable exploration inside the observed-shield action set."""

from __future__ import annotations

from collections import Counter
import math
import random

from ..policy_api import POLICY_API_VERSION, PolicyDecision


STATE_SCHEMA = "th06-rl-uniform-shield-exploration-v1"
POLICY_NAME = "uniform-shield-exploration-v1"


class UniformShieldExplorationPolicy:
    api_version = POLICY_API_VERSION
    name = POLICY_NAME

    def __init__(self) -> None:
        self.loaded = False
        self.policy_seed = 0
        self.exploration_probability = 0.0
        self.random = random.Random(0)
        self.decisions = 0
        self.non_baseline = 0
        self.selected: Counter[str] = Counter()

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("uniform shield exploration state schema mismatch")
        seed = int(state.get("policy_seed", -1))
        probability = float(state.get("exploration_probability", float("nan")))
        if not 0 <= seed < 2**64:
            raise ValueError("policy_seed must be an unsigned 64-bit integer")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("exploration_probability must be in [0, 1]")
        self.policy_seed = seed
        self.exploration_probability = probability
        self.random = random.Random(seed)
        self.loaded = True

    def decide(self, context) -> PolicyDecision:
        if not self.loaded:
            raise RuntimeError("uniform shield exploration requires a state file")
        legal = tuple(sorted(set(context.locally_admissible_actions)))
        if not legal:
            raise ValueError("exploration policy received an empty shield set")
        baseline = str(context.baseline_action)
        if baseline not in legal:
            raise ValueError("adapter baseline is outside the shield set")

        exploratory = self.exploration_probability / len(legal)
        probabilities = {action: exploratory for action in legal}
        probabilities[baseline] += 1.0 - self.exploration_probability
        draw = self.random.random()
        cumulative = 0.0
        chosen = legal[-1]
        for action in legal:
            cumulative += probabilities[action]
            if draw <= cumulative:
                chosen = action
                break
        probability = probabilities[chosen]
        if probability <= 0.0:
            raise RuntimeError("sampled an action with zero behavior probability")
        self.decisions += 1
        self.non_baseline += int(chosen != baseline)
        self.selected[chosen] += 1
        return PolicyDecision(
            chosen,
            POLICY_NAME,
            probability,
            tuple((action, probabilities[action]) for action in legal),
        )

    def metrics(self) -> dict[str, object]:
        return {
            "schema": STATE_SCHEMA,
            "policy_seed": self.policy_seed,
            "exploration_probability": self.exploration_probability,
            "decisions": self.decisions,
            "non_baseline_decisions": self.non_baseline,
            "selected": dict(sorted(self.selected.items())),
        }


def create_policy() -> UniformShieldExplorationPolicy:
    return UniformShieldExplorationPolicy()
