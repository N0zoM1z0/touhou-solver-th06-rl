"""Generation-4 autonomous propensity-aware options inside native safety."""

from __future__ import annotations

from collections import Counter
import math
import random

from ..policy_api import POLICY_API_VERSION, PolicyDecision, PolicyOptionTrace


STATE_SCHEMA = "th06-rl-propensity-aware-option-exploration-v1"
POLICY_NAME = "propensity-aware-option-exploration-v1"
OPTION_HORIZON_FRAMES = 8
INCUMBENT_MASS = 0.50
UNIFORM_MASS = 0.25
INFORMATION_MASS = 0.25


class PropensityAwareOptionExplorationPolicy:
    api_version = POLICY_API_VERSION
    name = POLICY_NAME

    def __init__(self) -> None:
        self.loaded = False
        self.policy_seed = 0
        self.random = random.Random(0)
        self.option_counter = 0
        self.active_id: str | None = None
        self.active_intent: str | None = None
        self.active_boundary_probability = 1.0
        self.active_probabilities: tuple[tuple[str, float], ...] = ()
        self.active_information: tuple[tuple[str, float], ...] = ()
        self.active_ess: tuple[tuple[str, float], ...] = ()
        self.active_start_frame = 0
        self.active_last_frame = -1
        self.active_scope: tuple[int, int, int, int] | None = None
        self.pending_boundary_update: tuple[str, float] | None = None
        self.importance_sum: Counter[str] = Counter()
        self.importance_square_sum: Counter[str] = Counter()
        self.assignment_counts: Counter[str] = Counter()
        self.decisions = 0
        self.boundaries = 0
        self.continuations = 0
        self.non_baseline_boundaries = 0
        self.selected: Counter[str] = Counter()
        self.terminations: Counter[str] = Counter()
        self.minimum_probability = 1.0
        self.information_policy = None

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("Generation-4 exploration state schema mismatch")
        seed = int(state.get("policy_seed", -1))
        horizon = int(state.get("option_horizon_frames", -1))
        masses = state.get("mixture")
        if (
            not 0 <= seed < 2**64
            or horizon != OPTION_HORIZON_FRAMES
            or not isinstance(masses, dict)
            or not math.isclose(float(masses.get("incumbent", -1.0)), INCUMBENT_MASS)
            or not math.isclose(float(masses.get("uniform", -1.0)), UNIFORM_MASS)
            or not math.isclose(
                float(masses.get("information", -1.0)), INFORMATION_MASS
            )
        ):
            raise ValueError("Generation-4 exploration contract mismatch")
        self.policy_seed = seed
        self.random = random.Random(seed)
        information_state = state.get("information_policy")
        if information_state is not None:
            if not isinstance(information_state, dict):
                raise TypeError("Generation-4 information policy is invalid")
            from .autonomous_sequential_r_critic import (
                AutonomousSequentialRCriticPolicy,
            )
            critic = AutonomousSequentialRCriticPolicy()
            critic.import_state(information_state)
            if critic.mode != "shadow":
                raise ValueError("exploration information policy must be shadow-only")
            self.information_policy = critic
        self.loaded = True

    def _end_active(self, reason: str) -> str:
        if self.active_id is None:
            raise RuntimeError("cannot terminate an absent option")
        self.terminations[reason] += 1
        self.active_id = None
        self.active_intent = None
        self.active_scope = None
        self.active_probabilities = ()
        self.active_information = ()
        self.active_ess = ()
        self.pending_boundary_update = None
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

    def _ess(self, action: str) -> float:
        total = float(self.importance_sum[action])
        square = float(self.importance_square_sum[action])
        return total * total / square if square > 0.0 else 0.0

    def _boundary_distribution(
        self,
        legal: tuple[str, ...],
        baseline: str,
        disagreement: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        if len(legal) == 1:
            return ({baseline: 1.0}, {baseline: 1.0}, {baseline: self._ess(baseline)})
        ess = {action: self._ess(action) for action in legal}
        if disagreement is not None and (
            set(disagreement) != set(legal)
            or any(
                not math.isfinite(value) or not 0.0 < value <= 1.0
                for value in disagreement.values()
            )
        ):
            raise ValueError("Generation-4 population disagreement is invalid")
        raw_information = {
            action: (
                1.0 / math.sqrt(1.0 + ess[action])
                * (disagreement[action] if disagreement is not None else 1.0)
            )
            for action in legal
        }
        normalizer = sum(raw_information.values())
        information = {
            action: raw_information[action] / normalizer for action in legal
        }
        probabilities = {
            action: (
                UNIFORM_MASS / len(legal)
                + INFORMATION_MASS * information[action]
                + (INCUMBENT_MASS if action == baseline else 0.0)
            )
            for action in legal
        }
        if (
            not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-12)
            or min(probabilities.values()) + 1e-12 < UNIFORM_MASS / len(legal)
        ):
            raise RuntimeError("Generation-4 propensity mixture is invalid")
        return probabilities, information, ess

    def _sample(self, probabilities: dict[str, float]) -> str:
        draw = self.random.random()
        cumulative = 0.0
        chosen = next(reversed(probabilities))
        for action, probability in probabilities.items():
            cumulative += probability
            if draw < cumulative:
                chosen = action
                break
        return chosen

    def _record_assignment(self, action: str, probability: float) -> None:
        weight = 1.0 / probability
        self.importance_sum[action] += weight
        self.importance_square_sum[action] += weight * weight
        self.assignment_counts[action] += 1
        self.pending_boundary_update = (action, probability)

    def _rollback_assignment(self) -> None:
        if self.pending_boundary_update is None:
            return
        action, probability = self.pending_boundary_update
        weight = 1.0 / probability
        self.importance_sum[action] -= weight
        self.importance_square_sum[action] -= weight * weight
        self.assignment_counts[action] -= 1
        self.pending_boundary_update = None

    def continue_certified(self, context) -> PolicyDecision:
        legal = tuple(sorted(set(context.locally_admissible_actions)))
        if not self.loaded or not legal:
            raise RuntimeError("certified continuation has no loaded safe option")
        preceding = self._preceding_termination(context, legal)
        if self.active_id is None:
            return PolicyDecision(context.baseline_action, POLICY_NAME, 1.0)
        if preceding is not None:
            raise RuntimeError("active option survived an invalid continuation")
        return self.decide(context)

    def reject_publication(self, decision: PolicyDecision) -> None:
        trace = decision.option
        if trace is None or self.active_id is None:
            return
        if trace.option_id != self.active_id:
            raise RuntimeError("publication rejection named a stale option")
        if trace.boundary:
            self._rollback_assignment()
        self._end_active("publication-rejected")

    def decide(self, context) -> PolicyDecision:
        if not self.loaded:
            raise RuntimeError("Generation-4 exploration requires a state file")
        legal = tuple(sorted(set(context.locally_admissible_actions)))
        if not legal:
            raise ValueError("Generation-4 exploration received an empty safe set")
        baseline = str(context.baseline_action)
        if baseline not in legal:
            raise ValueError("adapter baseline is outside the safe set")
        preceding = self._preceding_termination(context, legal)
        boundary = self.active_id is None
        if boundary:
            disagreement = (
                self.information_policy.information_disagreement(context)
                if self.information_policy is not None else None
            )
            probabilities, information, ess = self._boundary_distribution(
                legal, baseline, disagreement
            )
            chosen = self._sample(probabilities)
            probability = probabilities[chosen]
            self.option_counter += 1
            self.active_id = f"{self.policy_seed:016x}:{self.option_counter:08d}"
            self.active_intent = chosen
            self.active_boundary_probability = probability
            self.active_probabilities = tuple(probabilities.items())
            self.active_information = tuple(information.items())
            self.active_ess = tuple(ess.items())
            self.active_start_frame = int(context.frame)
            self.active_scope = tuple(context.scope)
            self._record_assignment(chosen, probability)
            self.boundaries += 1
            self.non_baseline_boundaries += int(chosen != baseline)
            self.minimum_probability = min(
                self.minimum_probability, min(probabilities.values())
            )
        else:
            chosen = str(self.active_intent)
            probability = 1.0
            self.pending_boundary_update = None
            self.continuations += 1
        elapsed = int(context.frame) - self.active_start_frame + 1
        if not 1 <= elapsed <= OPTION_HORIZON_FRAMES:
            raise RuntimeError("Generation-4 option elapsed-frame invariant failed")
        termination = "horizon" if elapsed == OPTION_HORIZON_FRAMES else None
        trace = PolicyOptionTrace(
            option_id=str(self.active_id),
            intent=chosen,
            boundary=boundary,
            boundary_probability=self.active_boundary_probability,
            elapsed_frames=elapsed,
            termination_reason=termination,
            preceding_termination_reason=preceding,
            behavior_probabilities=self.active_probabilities,
            information_weights=self.active_information,
            propensity_ess=self.active_ess,
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
            "mixture": {
                "incumbent": INCUMBENT_MASS,
                "uniform": UNIFORM_MASS,
                "information": INFORMATION_MASS,
            },
            "decisions": self.decisions,
            "option_boundaries": self.boundaries,
            "option_continuations": self.continuations,
            "non_baseline_boundaries": self.non_baseline_boundaries,
            "minimum_probability": self.minimum_probability,
            "information_mode": (
                "population-disagreement-times-ess"
                if self.information_policy is not None else "ess-only"
            ),
            "information_scorer_backend": (
                self.information_policy.scorer_backend
                if self.information_policy is not None else None
            ),
            "assignment_counts": dict(sorted(self.assignment_counts.items())),
            "propensity_ess": {
                action: self._ess(action) for action in sorted(self.importance_sum)
            },
            "selected": dict(sorted(self.selected.items())),
            "terminations": dict(sorted(self.terminations.items())),
        }


def create_policy() -> PropensityAwareOptionExplorationPolicy:
    return PropensityAwareOptionExplorationPolicy()
