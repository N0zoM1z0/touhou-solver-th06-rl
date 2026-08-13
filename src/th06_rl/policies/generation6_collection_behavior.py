"""Propensity-recorded ESS exploration anchored to a frozen G6 actor.

This policy is deliberately a data-collection policy, not a promoted player.
At each eight-frame option boundary it gives half of the probability mass to
the frozen Generation-6 actor proposal, one quarter uniformly to every native-
safe action, and one quarter to an automatic inverse-ESS allocation.  No game,
Stage, phase, frame, RNG, HIT, or source-context identity enters the mixture.
"""

from __future__ import annotations

from collections import Counter
import math
import random

from ..actions import ACTION_NAMES
from ..policy_api import POLICY_API_VERSION, PolicyDecision, PolicyOptionTrace
from .autonomous_iql_actor import (
    AutonomousIqlActorPolicy,
    OPTION_HORIZON_FRAMES,
    STATE_SCHEMA as ACTOR_STATE_SCHEMA,
)


STATE_SCHEMA = "generation6-actor-ess-collection-state-v1"
# This is the stable complete-propensity option *interface* identifier. The
# distinct state schema, metrics schema, contract hash, and registry capability
# bind this actor-anchored mixture without changing historical loader semantics.
POLICY_NAME = "propensity-aware-option-exploration-v1"
ACTOR_MASS = 0.50
UNIFORM_MASS = 0.25
ESS_MASS = 0.25
ALLOWED_COLLECTION_CONTRACT_SHA256: frozenset[str] = frozenset((
    "d733d919726393b60d243b1be2501cc0a57888b1c8e588ed34d257a0aa081a52",
    "e49e363ba0da7a2b89ddb78116612a9ca164d022188db787376c0e0390c09c4f",
    "a4276c321d92ccf8ee17aa8cf7cad57934e7c0742135402bc781ea048ff6e960",
))
_ACTION_INDEX = {action: index for index, action in enumerate(ACTION_NAMES)}


class Generation6ActorEssCollectionPolicy:
    api_version = POLICY_API_VERSION
    name = POLICY_NAME

    def __init__(self) -> None:
        self.loaded = False
        self.policy_seed = 0
        self.random = random.Random(0)
        self.actor: AutonomousIqlActorPolicy | None = None
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
        self.actor_proposals = 0
        self.non_actor_assignments = 0
        self.selected: Counter[str] = Counter()
        self.actor_selected: Counter[str] = Counter()
        self.terminations: Counter[str] = Counter()
        self.minimum_probability = 1.0

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("Generation-6 collection state schema mismatch")
        contract = str(state.get("collection_contract_sha256", ""))
        seed = int(state.get("policy_seed", -1))
        mixture = state.get("mixture")
        actor_state = state.get("actor_state")
        if (
            contract not in ALLOWED_COLLECTION_CONTRACT_SHA256
            or not 0 <= seed < 2**64
            or int(state.get("option_horizon_frames", -1))
            != OPTION_HORIZON_FRAMES
            or not isinstance(mixture, dict)
            or not math.isclose(float(mixture.get("actor", -1)), ACTOR_MASS)
            or not math.isclose(float(mixture.get("uniform", -1)), UNIFORM_MASS)
            or not math.isclose(float(mixture.get("inverse_ess", -1)), ESS_MASS)
            or not isinstance(actor_state, dict)
            or actor_state.get("schema") != ACTOR_STATE_SCHEMA
            or actor_state.get("mode") != "shadow"
        ):
            raise ValueError("Generation-6 collection contract drifted")
        actor = AutonomousIqlActorPolicy()
        actor.import_state(actor_state)
        self.actor = actor
        self.policy_seed = seed
        self.random = random.Random(seed)
        self.loaded = True

    def _ess(self, action: str) -> float:
        total = float(self.importance_sum[action])
        square = float(self.importance_square_sum[action])
        return total * total / square if square > 0.0 else 0.0

    def _distribution(
        self, legal: tuple[str, ...], actor_action: str
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        if actor_action not in legal:
            raise ValueError("Generation-6 actor proposal escaped native safety")
        if len(legal) == 1:
            only = legal[0]
            return {only: 1.0}, {only: 1.0}, {only: self._ess(only)}
        ess = {action: self._ess(action) for action in legal}
        raw = {action: 1.0 / math.sqrt(1.0 + ess[action]) for action in legal}
        normalizer = sum(raw.values())
        information = {action: raw[action] / normalizer for action in legal}
        probabilities = {
            action: (
                UNIFORM_MASS / len(legal)
                + ESS_MASS * information[action]
                + (ACTOR_MASS if action == actor_action else 0.0)
            )
            for action in legal
        }
        if (
            not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-12)
            or min(probabilities.values()) + 1e-12
            < UNIFORM_MASS / len(legal)
        ):
            raise RuntimeError("Generation-6 collection propensity is invalid")
        return probabilities, information, ess

    def _sample(self, probabilities: dict[str, float]) -> str:
        draw = self.random.random()
        cumulative = 0.0
        chosen = next(reversed(probabilities))
        for action, probability in probabilities.items():
            cumulative += probability
            if draw < cumulative:
                return action
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

    def _end_active(self, reason: str) -> str:
        if self.active_id is None:
            raise RuntimeError("cannot terminate an absent collection option")
        self.terminations[reason] += 1
        self.active_id = None
        self.active_intent = None
        self.active_scope = None
        self.active_probabilities = ()
        self.active_information = ()
        self.active_ess = ()
        self.pending_boundary_update = None
        return reason

    def _preceding(self, context, legal: tuple[str, ...]) -> str | None:
        if self.active_id is None:
            return "episode-start" if self.decisions == 0 else None
        if tuple(context.scope) != self.active_scope:
            return self._end_active("stage-transition")
        if int(context.frame) != self.active_last_frame + 1:
            return self._end_active("observation-gap")
        if self.active_intent not in legal:
            return self._end_active("source-unsafe-intent")
        return None

    def continue_certified(self, context) -> PolicyDecision:
        legal = tuple(sorted(
            set(context.locally_admissible_actions), key=_ACTION_INDEX.__getitem__
        ))
        if not self.loaded or not legal:
            raise RuntimeError("collection continuation has no loaded safe option")
        preceding = self._preceding(context, legal)
        if self.active_id is None:
            return PolicyDecision(context.baseline_action, POLICY_NAME, 1.0)
        if preceding is not None:
            raise RuntimeError("collection option survived an invalid continuation")
        return self.decide(context)

    def reject_publication(self, decision: PolicyDecision) -> None:
        trace = decision.option
        if trace is None or self.active_id is None:
            return
        if trace.option_id != self.active_id:
            raise RuntimeError("collection rejection named a stale option")
        if trace.boundary:
            self._rollback_assignment()
        self._end_active("publication-rejected")

    def decide(self, context) -> PolicyDecision:
        if not self.loaded or self.actor is None:
            raise RuntimeError("Generation-6 collection policy is not loaded")
        legal = tuple(sorted(
            set(context.locally_admissible_actions), key=_ACTION_INDEX.__getitem__
        ))
        baseline = str(context.baseline_action)
        if not legal or baseline not in legal or not set(legal) <= set(ACTION_NAMES):
            raise ValueError("Generation-6 collection received an invalid safe set")
        preceding = self._preceding(context, legal)
        boundary = self.active_id is None
        if boundary:
            actor_action = self.actor.collection_proposal(context)
            probabilities, information, ess = self._distribution(
                legal, actor_action
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
            self.actor_proposals += actor_action != baseline
            self.non_actor_assignments += chosen != actor_action
            self.actor_selected[actor_action] += 1
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
            raise RuntimeError("Generation-6 collection option horizon drifted")
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
        actor_metrics = self.actor.metrics() if self.actor is not None else None
        return {
            "schema": STATE_SCHEMA,
            "policy_seed": self.policy_seed,
            "mixture": {
                "actor": ACTOR_MASS,
                "uniform": UNIFORM_MASS,
                "inverse_ess": ESS_MASS,
            },
            "decisions": self.decisions,
            "option_boundaries": self.boundaries,
            "option_continuations": self.continuations,
            "actor_proposals": self.actor_proposals,
            "non_actor_assignments": self.non_actor_assignments,
            "minimum_probability": self.minimum_probability,
            "assignment_counts": dict(sorted(self.assignment_counts.items())),
            "propensity_ess": {
                action: self._ess(action) for action in sorted(self.importance_sum)
            },
            "selected": dict(sorted(self.selected.items())),
            "actor_selected": dict(sorted(self.actor_selected.items())),
            "terminations": dict(sorted(self.terminations.items())),
            "actor": actor_metrics,
        }


def create_policy() -> Generation6ActorEssCollectionPolicy:
    return Generation6ActorEssCollectionPolicy()
