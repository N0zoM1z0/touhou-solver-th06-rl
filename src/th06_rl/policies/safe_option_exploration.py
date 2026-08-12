"""Wine-only randomized safe options with fresh per-frame certification.

The policy never constructs or extends a safe set.  It may continue an intent
only while that intent remains in the adapter's freshly supplied native-safe
set.  Randomization occurs solely at option boundaries.
"""

from __future__ import annotations

from collections import Counter
import math
import random

from ..policy_api import (
    POLICY_API_VERSION,
    PolicyDecision,
    PolicyOptionTrace,
)


STATE_SCHEMA = "th06-rl-safe-option-exploration-v1"
POLICY_NAME = "safe-option-exploration-v1"
OPTION_HORIZON_FRAMES = 8


class SafeOptionExplorationPolicy:
    api_version = POLICY_API_VERSION
    name = POLICY_NAME

    def __init__(self) -> None:
        self.loaded = False
        self.policy_seed = 0
        self.exploration_probability = 0.0
        self.random = random.Random(0)
        self.option_counter = 0
        self.active_id: str | None = None
        self.active_intent: str | None = None
        self.active_boundary_probability = 1.0
        self.active_start_frame = 0
        self.active_last_frame = -1
        self.active_scope: tuple[int, int, int, int] | None = None
        self.decisions = 0
        self.boundaries = 0
        self.continuations = 0
        self.non_baseline_boundaries = 0
        self.selected: Counter[str] = Counter()
        self.terminations: Counter[str] = Counter()

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("safe option exploration state schema mismatch")
        seed = int(state.get("policy_seed", -1))
        probability = float(state.get("exploration_probability", float("nan")))
        horizon = int(state.get("option_horizon_frames", -1))
        if not 0 <= seed < 2**64:
            raise ValueError("policy_seed must be an unsigned 64-bit integer")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("exploration_probability must be in [0, 1]")
        if horizon != OPTION_HORIZON_FRAMES:
            raise ValueError("generation-3 option horizon must be eight frames")
        self.policy_seed = seed
        self.exploration_probability = probability
        self.random = random.Random(seed)
        self.loaded = True

    def _end_active(self, reason: str) -> str:
        if self.active_id is None:
            raise RuntimeError("cannot terminate an absent option")
        self.terminations[reason] += 1
        self.active_id = None
        self.active_intent = None
        self.active_scope = None
        return reason

    def _preceding_termination(self, context, legal: tuple[str, ...]) -> str | None:
        if self.active_id is None:
            return "episode-start" if self.decisions == 0 else None
        if tuple(context.scope) != self.active_scope:
            return self._end_active("stage-transition")
        if context.frame != self.active_last_frame + 1:
            return self._end_active("observation-gap")
        if self.active_intent not in legal:
            return self._end_active("source-unsafe-intent")
        return None

    def _boundary_probabilities(
        self, legal: tuple[str, ...], baseline: str
    ) -> dict[str, float]:
        if len(legal) == 1 or self.exploration_probability == 0.0:
            return {action: float(action == baseline) for action in legal}
        exploratory = self.exploration_probability / len(legal)
        result = {action: exploratory for action in legal}
        result[baseline] += 1.0 - self.exploration_probability
        return result

    def _sample(self, probabilities: dict[str, float]) -> str:
        draw = self.random.random()
        cumulative = 0.0
        chosen = next(reversed(probabilities))
        for action, probability in probabilities.items():
            cumulative += probability
            if probability > 0.0 and draw < cumulative:
                chosen = action
                break
        return chosen

    def continue_certified(self, context) -> PolicyDecision:
        """Advance option metadata inside a controller-certified input lease."""
        legal = tuple(sorted(set(context.locally_admissible_actions)))
        if not self.loaded or not legal:
            raise RuntimeError("certified continuation has no loaded safe option")
        preceding = self._preceding_termination(context, legal)
        if self.active_id is None:
            # A hardware lease may outlive the causal option horizon. The
            # controller still recertifies its single forced action, but that
            # is not a new randomized assignment without a choice boundary.
            return PolicyDecision(context.baseline_action, POLICY_NAME, 1.0)
        if preceding is not None:
            raise RuntimeError("active option survived an invalid continuation")
        return self.decide(context)

    def decide(self, context) -> PolicyDecision:
        if not self.loaded:
            raise RuntimeError("safe option exploration requires a state file")
        legal = tuple(sorted(set(context.locally_admissible_actions)))
        if not legal:
            raise ValueError("option exploration received an empty safe set")
        baseline = str(context.baseline_action)
        if baseline not in legal:
            raise ValueError("adapter baseline is outside the safe set")

        preceding = self._preceding_termination(context, legal)
        boundary = self.active_id is None
        if boundary:
            probabilities = self._boundary_probabilities(legal, baseline)
            chosen = self._sample(probabilities)
            probability = probabilities[chosen]
            if probability <= 0.0:
                raise RuntimeError("sampled an option with zero probability")
            self.option_counter += 1
            self.active_id = f"{self.policy_seed:016x}:{self.option_counter:08d}"
            self.active_intent = chosen
            self.active_boundary_probability = probability
            self.active_start_frame = int(context.frame)
            self.active_scope = tuple(context.scope)
            self.boundaries += 1
            self.non_baseline_boundaries += int(chosen != baseline)
        else:
            chosen = str(self.active_intent)
            probability = 1.0
            self.continuations += 1

        elapsed = int(context.frame) - self.active_start_frame + 1
        if not 1 <= elapsed <= OPTION_HORIZON_FRAMES:
            raise RuntimeError("option elapsed-frame invariant failed")
        termination = "horizon" if elapsed == OPTION_HORIZON_FRAMES else None
        trace = PolicyOptionTrace(
            option_id=str(self.active_id),
            intent=chosen,
            boundary=boundary,
            boundary_probability=self.active_boundary_probability,
            elapsed_frames=elapsed,
            termination_reason=termination,
            preceding_termination_reason=preceding,
        )
        self.active_last_frame = int(context.frame)
        if termination is not None:
            self._end_active(termination)

        self.decisions += 1
        self.selected[chosen] += 1
        return PolicyDecision(chosen, POLICY_NAME, probability, trace)

    def metrics(self) -> dict[str, object]:
        return {
            "schema": STATE_SCHEMA,
            "policy_seed": self.policy_seed,
            "exploration_probability": self.exploration_probability,
            "option_horizon_frames": OPTION_HORIZON_FRAMES,
            "decisions": self.decisions,
            "option_boundaries": self.boundaries,
            "option_continuations": self.continuations,
            "non_baseline_boundaries": self.non_baseline_boundaries,
            "selected": dict(sorted(self.selected.items())),
            "terminations": dict(sorted(self.terminations.items())),
        }


def create_policy() -> SafeOptionExplorationPolicy:
    return SafeOptionExplorationPolicy()
