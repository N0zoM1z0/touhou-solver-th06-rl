"""Qualified, bounded Generation-7 option policy for the Wine controller.

The online path consumes only frozen portable artifacts. It never imports a
training framework, expands the controller's native-safe set, or queries the
learned actor outside the explicitly eligible NMNB state distribution.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

from th06_rl.g7_contract import (
    CANDIDATE_SCHEMA,
    ONLINE_AUTHORIZATION,
    ONLINE_POLICY_NAME,
    ONLINE_QUALIFICATION_SCHEMA,
    ONLINE_STATE_SCHEMA,
)
from th06_rl.g7_forecast import forecast_accepted_actions
from th06_rl.g7_learner import linear_actor_distribution
from th06_rl.g7_policy_math import sample_action
from th06_rl.g7_support import locally_supported_actions
from th06_rl.offline_options import ActorState
from th06_rl.policies.safe_option_exploration import (
    OPTION_HORIZON_FRAMES,
    SafeOptionExplorationPolicy,
)


def canonical_candidate_sha256(candidate: dict[str, object]) -> str:
    payload = json.dumps(
        candidate,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class G7CandidatePolicy(SafeOptionExplorationPolicy):
    """Sample the exact OPE distribution only at eight-frame boundaries."""

    name = ONLINE_POLICY_NAME

    def __init__(self) -> None:
        super().__init__()
        self.candidate: dict[str, object] = {}
        self.target_max_kl = 0.0
        self.candidate_sha256 = ""
        self.authorization = ""
        self.boundary_reasons: Counter[str] = Counter()
        self.supported_action_total = 0
        self.forecast_action_total = 0
        self.forced_ineligible_boundaries = 0
        self.abstained_boundaries = 0

    def import_state(self, state: dict[str, object]) -> None:
        candidate = state.get("candidate")
        qualification = state.get("qualification")
        seed_value = state.get("policy_seed")
        horizon_value = state.get("option_horizon_frames")
        max_kl_value = state.get("target_max_kl")
        if (
            not isinstance(seed_value, int)
            or isinstance(seed_value, bool)
            or not isinstance(horizon_value, int)
            or isinstance(horizon_value, bool)
            or not isinstance(max_kl_value, (int, float))
            or isinstance(max_kl_value, bool)
        ):
            raise ValueError("G7 online state numeric contract is invalid")
        seed = seed_value
        horizon = horizon_value
        target_max_kl = float(max_kl_value)
        if (
            state.get("schema") != ONLINE_STATE_SCHEMA
            or state.get("authorization") != ONLINE_AUTHORIZATION
            or not 0 <= seed < 2**64
            or horizon != OPTION_HORIZON_FRAMES
            or not math.isfinite(target_max_kl)
            or target_max_kl < 0.0
            or not isinstance(candidate, dict)
            or candidate.get("schema") != CANDIDATE_SCHEMA
            or candidate.get("authorization") != "offline-research-only"
            or not isinstance(qualification, dict)
        ):
            raise ValueError("G7 online state is not canary-qualified")
        candidate_sha256 = canonical_candidate_sha256(candidate)
        if (
            qualification.get("schema") != ONLINE_QUALIFICATION_SCHEMA
            or qualification.get("candidate_sha256") != candidate_sha256
            or qualification.get("target_max_kl") != target_max_kl
            or qualification.get("heldout_ope_passed") is not True
            or qualification.get("offline_online_distribution_parity_passed")
            is not True
            or qualification.get("windows_embedded_python_latency_passed")
            is not True
            or qualification.get("original_wine_shadow_passed") is not True
        ):
            raise ValueError("G7 online qualification evidence is incomplete")
        if not all(
            isinstance(candidate.get(name), dict)
            for name in ("actor", "local_support", "forecast")
        ):
            raise ValueError("G7 candidate lacks a portable serving artifact")

        self.policy_seed = seed
        self.random.seed(seed)
        self.target_max_kl = target_max_kl
        self.candidate = candidate
        self.candidate_sha256 = candidate_sha256
        self.authorization = ONLINE_AUTHORIZATION
        self.loaded = True

    @staticmethod
    def _actor_state(context, legal: tuple[str, ...], baseline: str) -> ActorState:
        return ActorState(
            tuple(context.observation_features),
            tuple(context.action_features),
            tuple(context.history_features),
            legal,
            baseline,
            str(context.current_action),
        )

    def _boundary_probabilities(
        self, context, legal: tuple[str, ...], baseline: str
    ) -> dict[str, float]:
        if not context.learning_eligible:
            if legal != (baseline,):
                raise ValueError(
                    "ineligible NMNB extension must be a baseline singleton"
                )
            self.forced_ineligible_boundaries += 1
            self.boundary_reasons["player-not-vulnerable"] += 1
            return {baseline: 1.0}

        state = self._actor_state(context, legal, baseline)
        support_artifact = self.candidate["local_support"]
        forecast_artifact = self.candidate["forecast"]
        actor = self.candidate["actor"]
        assert isinstance(support_artifact, dict)
        assert isinstance(forecast_artifact, dict)
        assert isinstance(actor, dict)
        supported = locally_supported_actions(support_artifact, state)
        forecast = forecast_accepted_actions(
            forecast_artifact,
            state,
            supported_actions=supported,
        )
        distribution = linear_actor_distribution(
            actor,
            state,
            supported_actions=supported,
            forecast_accepted_actions=forecast,
            max_kl=self.target_max_kl,
        )
        probabilities = dict(distribution.probabilities)
        if set(probabilities) != set(legal):
            raise RuntimeError("G7 distribution changed the native-safe action set")
        self.supported_action_total += len(supported)
        self.forecast_action_total += len(forecast)
        self.abstained_boundaries += int(distribution.abstained)
        self.boundary_reasons[distribution.reason] += 1
        return probabilities

    def _sample(self, probabilities: dict[str, float]) -> str:
        return sample_action(
            tuple(probabilities.items()),
            draw=self.random.random(),
        )[0]

    def metrics(self) -> dict[str, object]:
        base = super().metrics()
        base.update({
            "schema": ONLINE_STATE_SCHEMA,
            "policy_id": self.name,
            "authorization": self.authorization,
            "candidate_sha256": self.candidate_sha256,
            "target_max_kl": self.target_max_kl,
            "forced_ineligible_boundaries": self.forced_ineligible_boundaries,
            "abstained_boundaries": self.abstained_boundaries,
            "supported_action_total": self.supported_action_total,
            "forecast_action_total": self.forecast_action_total,
            "boundary_reasons": dict(sorted(self.boundary_reasons.items())),
        })
        return base


def create_policy() -> G7CandidatePolicy:
    return G7CandidatePolicy()
