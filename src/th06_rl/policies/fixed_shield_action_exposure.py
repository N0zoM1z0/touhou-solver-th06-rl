"""Uniform short action intentions with a fresh observed-shield check per root."""

from __future__ import annotations

from collections import Counter
import random

from ..policy_api import (
    ACTION_EXPOSURE_SCHEMA,
    POLICY_API_VERSION,
    ActionExposure,
    PolicyDecision,
)


STATE_SCHEMA = "th06-rl-fixed-shield-action-exposure-v1"
POLICY_NAME = "fixed-shield-action-exposure-v1"
OVERRIDE_REASON = "intended-action-not-shield-admissible"


class FixedShieldActionExposurePolicy:
    api_version = POLICY_API_VERSION
    name = POLICY_NAME

    def __init__(self) -> None:
        self.loaded = False
        self.policy_seed = 0
        self.exposure_roots = 0
        self.random = random.Random(0)
        self.group_id = -1
        self.steps_emitted = 0
        self.intended_action: str | None = None
        self.assignment_probability = 0.0
        self.assignment_probabilities: tuple[tuple[str, float], ...] = ()
        self.decisions = 0
        self.groups = 0
        self.overrides = 0
        self.interruptions: Counter[str] = Counter()
        self.assigned: Counter[str] = Counter()
        self.published: Counter[str] = Counter()

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("fixed action-exposure state schema mismatch")
        seed = int(state.get("policy_seed", -1))
        roots = int(state.get("exposure_roots", -1))
        if not 0 <= seed < 2**64:
            raise ValueError("policy_seed must be an unsigned 64-bit integer")
        if not 2 <= roots <= 16:
            raise ValueError("exposure_roots must be in [2, 16]")
        self.policy_seed = seed
        self.exposure_roots = roots
        self.random = random.Random(seed)
        self.loaded = True

    def _start_group(self, legal: tuple[str, ...]) -> tuple[str, float]:
        probability = 1.0 / len(legal)
        probabilities = tuple((action, probability) for action in legal)
        draw = self.random.random()
        chosen = legal[min(int(draw * len(legal)), len(legal) - 1)]
        self.group_id += 1
        self.steps_emitted = 0
        self.intended_action = chosen
        self.assignment_probability = probability
        self.assignment_probabilities = probabilities
        self.groups += 1
        self.assigned[chosen] += 1
        return chosen, probability

    def decide(self, context) -> PolicyDecision:
        if not self.loaded:
            raise RuntimeError("fixed action exposure requires a state file")
        legal = tuple(sorted(set(context.locally_admissible_actions)))
        if not legal:
            raise ValueError("action-exposure policy received an empty shield set")
        baseline = str(context.baseline_action)
        if baseline not in legal:
            raise ValueError("adapter baseline is outside the shield set")

        starting = self.intended_action is None or self.steps_emitted >= self.exposure_roots
        if starting:
            action, behavior_probability = self._start_group(legal)
            behavior_probabilities = self.assignment_probabilities
            override_reason = None
        else:
            assert self.intended_action is not None
            if self.intended_action in legal:
                action = self.intended_action
                override_reason = None
            else:
                action = baseline
                override_reason = OVERRIDE_REASON
                self.overrides += 1
            behavior_probability = 1.0
            behavior_probabilities = tuple(
                (candidate, float(candidate == action)) for candidate in legal
            )

        assert self.intended_action is not None
        step = self.steps_emitted
        exposure = ActionExposure(
            ACTION_EXPOSURE_SCHEMA,
            self.group_id,
            step,
            self.exposure_roots,
            self.intended_action,
            self.assignment_probability,
            self.assignment_probabilities,
            override_reason,
        )
        self.steps_emitted += 1
        self.decisions += 1
        self.published[action] += 1
        return PolicyDecision(
            action,
            POLICY_NAME,
            behavior_probability,
            behavior_probabilities,
            exposure,
        )

    def interrupt(self, reason: str) -> None:
        if self.intended_action is None or self.steps_emitted >= self.exposure_roots:
            return
        if not reason:
            raise ValueError("action-exposure interruption reason is empty")
        self.interruptions[reason] += 1
        self.intended_action = None
        self.steps_emitted = 0
        self.assignment_probability = 0.0
        self.assignment_probabilities = ()

    def metrics(self) -> dict[str, object]:
        return {
            "schema": STATE_SCHEMA,
            "policy_seed": self.policy_seed,
            "exposure_roots": self.exposure_roots,
            "decisions": self.decisions,
            "groups": self.groups,
            "overrides": self.overrides,
            "interruptions": dict(sorted(self.interruptions.items())),
            "assigned": dict(sorted(self.assigned.items())),
            "published": dict(sorted(self.published.items())),
        }


def create_policy() -> FixedShieldActionExposurePolicy:
    return FixedShieldActionExposurePolicy()
