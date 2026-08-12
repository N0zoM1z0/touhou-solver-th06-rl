"""Generation-6 native-safe, propensity-bounded IQL actor intervention."""

from __future__ import annotations

import base64
from array import array
from collections import Counter, deque
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import zlib

from ..advantage_learning import (
    hazard_codebook_feature_names,
    rich_feature_names,
)
from ..hazard_representation import (
    HISTORY_FEATURE_NAMES,
    NativeHazardCodebookEncoder,
)
from ..offline import ACTION_NAMES
from ..policy_api import (
    POLICY_API_VERSION,
    PolicyDecision,
    PolicyOptionTrace,
)
from ..th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)
from .offline_ranker import (
    NATIVE_SCORER_ENV,
    PortablePrototypeSupport,
)


STATE_SCHEMA = "autonomous-generation-6-iql-actor-policy-v1"
POLICY_NAME = "autonomous-generation-6-iql-actor"
CANDIDATE_SCHEMA = "autonomous-generation-6-candidate-v1"
MODEL_SCHEMA = "autonomous-iql-actor-model-v1"
MODEL_CODEC = "zlib-base64-json-v1"
OPTION_HORIZON_FRAMES = 8
INTERVENTION_CAP = 0.10
MINIMUM_UNIFORM_MASS = 0.10
DENSITY_RATIO_CAP = 2.0
EXPECTED_CANDIDATE_SHA256 = (
    "aea789ed9fe63aa4a2c0799092675fd287c9b66787ed968d82e82098fbb4ea64"
)
EXPECTED_QUALIFICATION_SHA256 = (
    "1da0212281902daf18c124d3e246a244ae19d4a92fa3177efd34711c460b3e34"
)
EXPECTED_DEPLOYABLE_AUDIT_SHA256 = (
    "f683abf05b0fc1165181c1b922882e8950eae48cb67060e54792e3bc6a86ba8f"
)
ALLOWED_CANARY_CONTRACT_SHA256 = frozenset((
    "161d6c0461dcd180777c020f97f701492061beda9610e0fe58bcf31269572f3c",
    "37f136deab0e162e76bb67bb4f55d88b76eeefb87e8af392fb165d9c24d99c6b",
    "cf51538bd8ccbf266a9442579078fc5373411f704814133c326203f25c6622a1",
))
ALLOWED_NATIVE_SCORER_SHA256 = frozenset((
    "8b99074e0d9eeae232d4a79286646b1688004d721ba288c605cde74743ef62ec",
    "0aa7c5a95b90b2df0d032ec02f21fcd3a39be3ba440819d00c0cb025bc641ef0",
    # Same frozen math, cache-contiguous loop traversal; admitted only after
    # portable/native exact-action and Wine latency preflight.
    "f0e34ad5b0929b3333e850028f814036786078193176cf08968d6975b3e220fa",
    "e794045cb89e9f6439e4bdfc354325f89a0771a57aa75a4aa654aac9197f2b87",
    # Fused rich-row support plus normalized actor entry point.
    "96255f659ac6ec3d79f4b6742abccc3e7ee9de9115f804b6629ca526381d14d0",
    "7010f61c7e4f56802e1caecabd18aea927ab7109d240d2025bc889cb0115a121",
    # Single adapter-array -> hazard/support/actor FFI path.
    "58c3a1aa82c73dba5f1200094546b16aa1d2044e0c5f046027719368ab5580ab",
    "507b7e2bb797b6d90b12dbebf1d77c431d6f3ce9086cf522c749f5f10305fa1b",
    # Fused seven-member mean and supported-row proposal selection.
    "377283b508e31a310fdfab32d2087a8da9d8d3fb70dd58433f3977caf4dc5cc4",
    "87be7d7c3f5711e3f744214031696257ee045be00a8f62c229e3519155ea8c92",
))
_ACTION_INDEX = {action: index for index, action in enumerate(ACTION_NAMES)}
_LEXICAL_RANK = {
    action: index for index, action in enumerate(sorted(ACTION_NAMES))
}


def _p95(values) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[math.ceil(0.95 * len(ordered)) - 1])


class _NativeActorPopulation:
    """Stdlib-only Windows facade over seven portable actor artifacts."""

    ARRAY_NAMES = (
        "state_hidden_weight", "state_hidden_bias",
        "state_latent_weight", "state_latent_bias",
        "action_hidden_weight", "action_hidden_bias",
        "action_latent_weight", "action_latent_bias",
        "action_score_weight", "action_score_bias",
    )

    def __init__(
        self,
        path: Path,
        artifacts: list[dict[str, object]],
        support: PortablePrototypeSupport,
        hazard_encoder: NativeHazardCodebookEncoder,
    ) -> None:
        if len(artifacts) != 7:
            raise ValueError("Generation-6 requires seven actor artifacts")
        reference = artifacts[0]
        names = tuple(map(str, reference.get("feature_names", ())))
        state_indices = tuple(map(int, reference.get("state_indices", ())))
        action_indices = tuple(map(int, reference.get("action_indices", ())))
        state_mean = tuple(map(float, reference.get("state_mean", ())))
        state_scale = tuple(map(float, reference.get("state_scale", ())))
        action_mean = tuple(map(float, reference.get("action_mean", ())))
        action_scale = tuple(map(float, reference.get("action_scale", ())))
        hidden = len(reference.get("state_hidden_bias", ()))
        rank = len(reference.get("state_latent_bias", ()))
        if (
            reference.get("schema") != MODEL_SCHEMA
            or names != rich_feature_names()
            or not state_indices or not action_indices
            or sorted((*state_indices, *action_indices)) != list(range(len(names)))
            or len(state_mean) != len(state_indices)
            or len(state_scale) != len(state_indices)
            or len(action_mean) != len(action_indices)
            or len(action_scale) != len(action_indices)
            or not 1 <= hidden <= 256
            or not 1 <= rank <= 128
            or any(value <= 0.0 or not math.isfinite(value)
                   for value in (*state_scale, *action_scale))
        ):
            raise ValueError("Generation-6 actor layout is invalid")
        common = (
            "feature_names", "state_indices", "action_indices",
            "state_mean", "state_scale", "action_mean", "action_scale",
        )
        if any(
            artifact.get("schema") != MODEL_SCHEMA
            or any(artifact.get(name) != reference.get(name) for name in common)
            for artifact in artifacts
        ):
            raise ValueError("Generation-6 actor population normalization drifted")

        def flatten(value):
            if isinstance(value, list):
                return [item for nested in value for item in flatten(nested)]
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Generation-6 actor contains a non-finite weight")
            return [number]

        arrays = []
        for name in self.ARRAY_NAMES:
            values = [
                value for artifact in artifacts
                for value in flatten(artifact.get(name))
            ]
            arrays.append((ctypes.c_float * len(values))(*values))
        library = ctypes.CDLL(str(path))
        function = library.th06_rl_evaluate_iql_policy_v1
        pointer = ctypes.POINTER(ctypes.c_float)
        integers = ctypes.POINTER(ctypes.c_int32)
        function.argtypes = [
            pointer, ctypes.c_int32,
            pointer, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32,
            pointer, ctypes.c_int32, ctypes.c_int32,
            pointer, pointer, pointer, ctypes.c_int32, ctypes.c_int32,
            pointer, ctypes.c_int32,
            integers, integers, integers, ctypes.c_double,
            pointer, pointer, pointer, ctypes.c_int32, integers, ctypes.c_int32,
            integers, ctypes.c_int32, integers, ctypes.c_int32,
            pointer, pointer, pointer, pointer,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            *(pointer for _index in range(10)),
            integers, integers,
        ]
        function.restype = ctypes.c_int
        support_flat = [
            value for group in support.groups for row in group for value in row
        ]
        support_offsets = [0]
        for group in support.groups:
            support_offsets.append(support_offsets[-1] + len(group))
        self.library = library
        self.function = function
        self.arrays = tuple(arrays)
        self.names = names
        self.state_indices = state_indices
        self.action_indices = action_indices
        self.state_mean = state_mean
        self.state_scale = state_scale
        self.action_mean = action_mean
        self.action_scale = action_scale
        self.hidden = hidden
        self.rank = rank
        self.model_count = len(artifacts)
        self.feature_count = len(names)
        self.state_indices_array = (
            ctypes.c_int32 * len(state_indices)
        )(*state_indices)
        self.action_indices_array = (
            ctypes.c_int32 * len(action_indices)
        )(*action_indices)
        self.state_mean_array = (ctypes.c_float * len(state_mean))(*state_mean)
        self.state_scale_array = (ctypes.c_float * len(state_scale))(*state_scale)
        self.action_mean_array = (ctypes.c_float * len(action_mean))(*action_mean)
        self.action_scale_array = (ctypes.c_float * len(action_scale))(*action_scale)
        self.support_mean = (
            ctypes.c_float * support.feature_count
        )(*support.mean)
        self.support_scale = (
            ctypes.c_float * support.feature_count
        )(*support.scale)
        self.support_prototypes = (
            ctypes.c_float * len(support_flat)
        )(*support_flat)
        self.support_offsets = (
            ctypes.c_int32 * len(support_offsets)
        )(*support_offsets)
        self.support_prototype_count = support_offsets[-1]
        self.support_action_count = len(support.groups)
        self.hazard_encoder = hazard_encoder

    def choose_context(
        self,
        observation: tuple[float, ...],
        actions: list[tuple[float, ...]],
        hazards: tuple[tuple[float, ...], ...],
        history: tuple[float, ...],
        *,
        baseline_row: int,
        current_row: int,
        action_indices: list[int],
        supported: list[int],
        tie_break_ranks: list[int],
        support_threshold: float,
    ) -> tuple[int, int]:
        if (
            not observation
            or not actions
            or len(observation) != len(OBSERVATION_FEATURE_NAMES)
            or any(len(row) != len(ACTION_FEATURE_NAMES) for row in actions)
            or len(history) != len(HISTORY_FEATURE_NAMES)
            or len(hazards) > 256
            or any(
                len(row) != self.hazard_encoder.feature_count for row in hazards
            )
            or not 0 <= baseline_row < len(actions)
            or not -1 <= current_row < len(actions)
        ):
            raise ValueError("Generation-6 policy input shape is invalid")
        if not (
            len(action_indices) == len(actions)
            == len(supported) == len(tie_break_ranks)
        ):
            raise ValueError("Generation-6 support action shape is invalid")
        observation_buffer = array("f", observation)
        action_buffer = array("f")
        for row in actions:
            action_buffer.extend(row)
        hazard_buffer = array("f")
        for row in hazards:
            hazard_buffer.extend(row)
        if not hazard_buffer:
            hazard_buffer.append(0.0)
        history_buffer = array("f", history)
        action_index_buffer = array("i", action_indices)
        supported_buffer = array("i", supported)
        tie_break_buffer = array("i", tie_break_ranks)
        if action_index_buffer.itemsize != ctypes.sizeof(ctypes.c_int32):
            raise RuntimeError("Generation-6 native integer width differs")
        observation_input = (
            ctypes.c_float * len(observation_buffer)
        ).from_buffer(observation_buffer)
        flat_actions = (
            ctypes.c_float * len(action_buffer)
        ).from_buffer(action_buffer)
        hazard_input = (
            ctypes.c_float * len(hazard_buffer)
        ).from_buffer(hazard_buffer)
        history_input = (
            ctypes.c_float * len(history_buffer)
        ).from_buffer(history_buffer)
        row_actions = (
            ctypes.c_int32 * len(action_index_buffer)
        ).from_buffer(action_index_buffer)
        row_supported = (
            ctypes.c_int32 * len(supported_buffer)
        ).from_buffer(supported_buffer)
        row_tie_break = (
            ctypes.c_int32 * len(tie_break_buffer)
        ).from_buffer(tie_break_buffer)
        proposal_row = ctypes.c_int32(-1)
        supported_count = ctypes.c_int32(-1)
        status = self.function(
            observation_input, len(observation),
            flat_actions, len(actions), len(actions[0]),
            baseline_row, current_row,
            hazard_input, len(hazards), self.hazard_encoder.feature_count,
            self.hazard_encoder.mean, self.hazard_encoder.scale,
            self.hazard_encoder.prototypes,
            self.hazard_encoder.prototype_count,
            self.hazard_encoder.output_count,
            history_input, len(history),
            row_actions, row_supported, row_tie_break, support_threshold,
            self.support_mean, self.support_scale, self.support_prototypes,
            self.support_prototype_count, self.support_offsets,
            self.support_action_count,
            self.state_indices_array, len(self.state_indices),
            self.action_indices_array, len(self.action_indices),
            self.state_mean_array, self.state_scale_array,
            self.action_mean_array, self.action_scale_array,
            self.model_count, self.hidden, self.rank,
            *self.arrays,
            ctypes.byref(proposal_row), ctypes.byref(supported_count),
        )
        if status != 0:
            raise RuntimeError(
                f"native Generation-6 support/actor failed with {status}"
            )
        if (
            not 0 <= proposal_row.value < len(actions)
            or not 0 <= supported_count.value < len(actions)
        ):
            raise RuntimeError("native Generation-6 proposal result is invalid")
        return proposal_row.value, supported_count.value


class AutonomousIqlActorPolicy:
    api_version = POLICY_API_VERSION
    name = POLICY_NAME + "-uninitialized"

    def __init__(self) -> None:
        self.loaded = False
        self.mode = "shadow"
        self.random = random.Random(0)
        self.policy_seed = 0
        self.scorer = None
        self.support = None
        self.support_threshold = 0.0
        self.factual_supported_actions: frozenset[str] = frozenset()
        self.hazard_encoder = None
        self.active_id: str | None = None
        self.active_intent: str | None = None
        self.active_boundary_probability = 1.0
        self.active_start_frame = 0
        self.active_last_frame = -1
        self.active_scope: tuple[int, int, int, int] | None = None
        self.option_counter = 0
        self.decisions = 0
        self.boundaries = 0
        self.continuations = 0
        self.proposals = 0
        self.interventions = 0
        self.intervention_budget: int | None = None
        self.budget_abstentions = 0
        self.support_abstentions = 0
        self.selected: Counter[str] = Counter()
        self.proposed: Counter[str] = Counter()
        self.terminations: Counter[str] = Counter()
        self.timing_ms = deque(maxlen=4096)
        self.over_four_ms = 0
        self.deadline_misses = 0

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("Generation-6 policy state schema mismatch")
        mode = state.get("mode")
        authorization = state.get("authorization")
        intervention = state.get("intervention")
        if (
            mode not in ("shadow", "active")
            or not isinstance(authorization, dict)
            or authorization.get("offline_qualification_passed") is not True
            or authorization.get("deployable_target_audit_passed") is not True
            or authorization.get("qualification_result_sha256")
            != EXPECTED_QUALIFICATION_SHA256
            or authorization.get("deployable_target_audit_sha256")
            != EXPECTED_DEPLOYABLE_AUDIT_SHA256
            or (mode == "active" and not isinstance(
                authorization.get("frozen_wine_canary"), dict
            ))
            or not isinstance(intervention, dict)
            or float(intervention.get("probability_cap", math.nan))
            != INTERVENTION_CAP
            or float(intervention.get("minimum_uniform_mass", math.nan))
            != MINIMUM_UNIFORM_MASS
            or float(intervention.get("density_ratio_cap", math.nan))
            != DENSITY_RATIO_CAP
            or int(state.get("option_horizon_frames", -1))
            != OPTION_HORIZON_FRAMES
        ):
            raise ValueError("Generation-6 authorization or target drifted")
        payload = state.get("candidate_payload")
        if state.get("candidate_codec") != MODEL_CODEC or not isinstance(payload, str):
            raise ValueError("Generation-6 candidate payload is absent")
        decoded = zlib.decompress(base64.b64decode(payload, validate=True))
        if (
            state.get("candidate_sha256") != EXPECTED_CANDIDATE_SHA256
            or hashlib.sha256(decoded).hexdigest() != EXPECTED_CANDIDATE_SHA256
        ):
            raise ValueError("Generation-6 candidate payload hash differs")
        candidate = json.loads(decoded.decode("utf-8"))
        if (
            candidate.get("schema") != CANDIDATE_SCHEMA
            or candidate.get("passed") is not True
            or candidate.get("qualification_samples_loaded") is not False
            or tuple(candidate.get("feature_names", ())) != rich_feature_names()
            or candidate.get("selection", {}).get("physical_safety")
            != "native-safe-set-only"
            or candidate.get("selection", {}).get("bomb") != "forbidden"
        ):
            raise ValueError("Generation-6 candidate contract is invalid")
        path_value = os.environ.get(NATIVE_SCORER_ENV)
        if not path_value:
            raise RuntimeError("Generation-6 policy requires the native scorer")
        native = state.get("native_scorer")
        path = Path(path_value)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            not isinstance(native, dict)
            or actual != native.get("sha256")
            or actual not in ALLOWED_NATIVE_SCORER_SHA256
        ):
            raise ValueError("Generation-6 native scorer hash differs")
        support_artifact = candidate.get("support")
        representation = candidate.get("representation")
        actors = candidate.get("actors")
        if (
            not isinstance(support_artifact, dict)
            or not isinstance(representation, dict)
            or not isinstance(actors, list)
        ):
            raise ValueError("Generation-6 fitted artifacts are absent")
        portable_support = PortablePrototypeSupport(
            support_artifact, feature_count=len(rich_feature_names())
        )
        self.hazard_encoder = NativeHazardCodebookEncoder(
            path,
            expected_sha256=actual,
            artifact=representation,
            output_count=len(hazard_codebook_feature_names()),
        )
        self.scorer = _NativeActorPopulation(
            path, actors, portable_support, self.hazard_encoder
        )
        supported = frozenset(map(
            str, support_artifact.get("factual_supported_actions", ())
        ))
        threshold = float(support_artifact.get("threshold", math.nan))
        if (
            not supported or not supported <= set(ACTION_NAMES)
            or not math.isfinite(threshold) or threshold < 0.0
        ):
            raise ValueError("Generation-6 support contract is invalid")
        seed = int(state.get("policy_seed", -1))
        if not 0 <= seed < 2**64:
            raise ValueError("Generation-6 policy seed is invalid")
        if mode == "active":
            canary = authorization["frozen_wine_canary"]
            if (
                canary.get("schema")
                != "autonomous-generation-6-wine-canary-authorization-v1"
                or canary.get("contract_sha256")
                not in ALLOWED_CANARY_CONTRACT_SHA256
                or canary.get("normal_speed") is not True
                or canary.get("natural_rng") is not True
                or canary.get("complete_stage_hit_continuation") is not True
                or canary.get("bomb") != "forbidden"
            ):
                raise ValueError("Generation-6 active canary contract is invalid")
            budget = int(canary.get("maximum_interventions", -1))
            if not 1 <= budget <= 64:
                raise ValueError("Generation-6 intervention budget is invalid")
            self.intervention_budget = budget
        else:
            self.intervention_budget = None
        self.mode = str(mode)
        self.policy_seed = seed
        self.random = random.Random(seed)
        self.support_threshold = threshold
        self.factual_supported_actions = supported
        self.loaded = True
        self.name = f"{POLICY_NAME}-{mode}"

    def _history(self, context) -> tuple[float, ...]:
        names = tuple(name for name, _value in context.history_features)
        values = tuple(float(value) for _name, value in context.history_features)
        if names != HISTORY_FEATURE_NAMES:
            raise ValueError("Generation-6 history feature schema differs")
        return values

    def _candidate_rows(
        self,
        context,
        legal: tuple[str, ...],
        baseline: str,
        hazard: tuple[float, ...],
        history: tuple[float, ...],
    ) -> list[list[float]]:
        """Build shared-state rows once instead of reparsing per action."""
        observation = tuple(context.observation_features)
        if tuple(name for name, _value in observation) != OBSERVATION_FEATURE_NAMES:
            raise ValueError("Generation-6 observation feature schema differs")
        observation_values = tuple(float(value) for _name, value in observation)
        raw_actions = dict(context.action_features)
        if len(raw_actions) != len(context.action_features) or baseline not in raw_actions:
            raise ValueError("Generation-6 action feature set is invalid")

        def values(action: str) -> tuple[float, ...]:
            raw = tuple(raw_actions.get(action, ()))
            if tuple(name for name, _value in raw) != ACTION_FEATURE_NAMES:
                raise ValueError("Generation-6 action feature schema differs")
            return tuple(float(value) for _name, value in raw)

        baseline_values = values(baseline)
        suffix = (*hazard, *history)
        rows = []
        for action in legal:
            selected = values(action)
            rows.append(list((
                *observation_values,
                *selected,
                *(left - right for left, right in zip(
                    selected, baseline_values, strict=True
                )),
                float(action == baseline),
                float(action == context.current_action),
                *suffix,
            )))
        if any(len(row) != len(rich_feature_names()) for row in rows):
            raise RuntimeError("Generation-6 online feature width differs")
        return rows

    def _adapter_arrays(
        self, context, legal: tuple[str, ...]
    ) -> tuple[tuple[float, ...], list[tuple[float, ...]]]:
        observation = tuple(context.observation_features)
        if tuple(name for name, _value in observation) != OBSERVATION_FEATURE_NAMES:
            raise ValueError("Generation-6 observation feature schema differs")
        raw_actions = dict(context.action_features)
        if len(raw_actions) != len(context.action_features):
            raise ValueError("Generation-6 action feature set has duplicates")
        actions = []
        for action in legal:
            raw = tuple(raw_actions.get(action, ()))
            if tuple(name for name, _value in raw) != ACTION_FEATURE_NAMES:
                raise ValueError("Generation-6 action feature schema differs")
            actions.append(tuple(float(value) for _name, value in raw))
        return (
            tuple(float(value) for _name, value in observation), actions
        )

    def _proposal(self, context, legal: tuple[str, ...], baseline: str) -> str:
        started = time.perf_counter()
        history = self._history(context)
        observation, actions = self._adapter_arrays(context, legal)
        proposal_row, supported_count = self.scorer.choose_context(
            observation,
            actions,
            tuple(context.hazard_primitives),
            history,
            baseline_row=legal.index(baseline),
            current_row=(
                legal.index(context.current_action)
                if context.current_action in legal else -1
            ),
            action_indices=[_ACTION_INDEX[action] for action in legal],
            supported=[
                int(action in self.factual_supported_actions)
                for action in legal
            ],
            tie_break_ranks=[_LEXICAL_RANK[action] for action in legal],
            support_threshold=self.support_threshold,
        )
        self.support_abstentions += len(legal) - 1 - supported_count
        proposal = legal[proposal_row]
        elapsed = (time.perf_counter() - started) * 1000.0
        self.timing_ms.append(elapsed)
        self.over_four_ms += elapsed > 4.0
        self.deadline_misses += elapsed > (1000.0 / 60.0)
        return proposal

    def _end_active(self, reason: str) -> str:
        if self.active_id is None:
            raise RuntimeError("cannot terminate an absent Generation-6 option")
        self.terminations[reason] += 1
        self.active_id = None
        self.active_intent = None
        self.active_scope = None
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
            raise RuntimeError("Generation-6 continuation has no safe option")
        preceding = self._preceding(context, legal)
        if self.active_id is None:
            return PolicyDecision(context.baseline_action, self.name, 1.0)
        if preceding is not None:
            raise RuntimeError("Generation-6 option survived invalid continuation")
        return self.decide(context)

    def reject_publication(self, decision: PolicyDecision) -> None:
        if decision.option is None or self.active_id is None:
            return
        if decision.option.option_id != self.active_id:
            raise RuntimeError("Generation-6 rejection named stale option")
        self._end_active("publication-rejected")

    def decide(self, context) -> PolicyDecision:
        if not self.loaded:
            raise RuntimeError("Generation-6 policy state is not loaded")
        legal = tuple(sorted(
            set(context.locally_admissible_actions), key=_ACTION_INDEX.__getitem__
        ))
        baseline = str(context.baseline_action)
        if not legal or baseline not in legal:
            raise ValueError("Generation-6 received an invalid native-safe set")
        preceding = self._preceding(context, legal)
        boundary = self.active_id is None
        if boundary:
            proposal = self._proposal(context, legal, baseline)
            rho = (
                min(INTERVENTION_CAP, DENSITY_RATIO_CAP * MINIMUM_UNIFORM_MASS / len(legal))
                if proposal != baseline else 0.0
            )
            self.proposals += proposal != baseline
            self.proposed[proposal] += proposal != baseline
            budget_available = (
                self.intervention_budget is None
                or self.interventions < self.intervention_budget
            )
            if self.mode == "active" and budget_available:
                intervene = proposal != baseline and self.random.random() < rho
                chosen = proposal if intervene else baseline
                probability = (
                    rho if intervene
                    else 1.0 - rho if proposal != baseline else 1.0
                )
            else:
                chosen = baseline
                probability = 1.0
                self.budget_abstentions += (
                    self.mode == "active"
                    and proposal != baseline
                    and not budget_available
                )
            self.interventions += chosen != baseline
            self.option_counter += 1
            self.active_id = f"{self.policy_seed:016x}:{self.option_counter:08d}"
            self.active_intent = chosen
            self.active_boundary_probability = probability
            self.active_start_frame = int(context.frame)
            self.active_scope = tuple(context.scope)
            self.boundaries += 1
        else:
            chosen = str(self.active_intent)
            probability = 1.0
            self.continuations += 1
        elapsed = int(context.frame) - self.active_start_frame + 1
        if not 1 <= elapsed <= OPTION_HORIZON_FRAMES:
            raise RuntimeError("Generation-6 option horizon invariant failed")
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
        return PolicyDecision(chosen, self.name, probability, trace)

    def metrics(self) -> dict[str, object]:
        return {
            "schema": STATE_SCHEMA,
            "mode": self.mode,
            "policy_seed": self.policy_seed,
            "target": {
                "probability_cap": INTERVENTION_CAP,
                "minimum_uniform_mass": MINIMUM_UNIFORM_MASS,
                "density_ratio_cap": DENSITY_RATIO_CAP,
            },
            "decisions": self.decisions,
            "option_boundaries": self.boundaries,
            "option_continuations": self.continuations,
            "proposals": self.proposals,
            "interventions": self.interventions,
            "intervention_budget": self.intervention_budget,
            "budget_abstentions": self.budget_abstentions,
            "support_abstentions": self.support_abstentions,
            "selected": dict(sorted(self.selected.items())),
            "proposed": dict(sorted(self.proposed.items())),
            "terminations": dict(sorted(self.terminations.items())),
            "latency_p95_ms": _p95(self.timing_ms),
            "over_four_ms": self.over_four_ms,
            "deadline_misses": self.deadline_misses,
        }


def create_policy() -> AutonomousIqlActorPolicy:
    return AutonomousIqlActorPolicy()
