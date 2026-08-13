"""Learner-neutral proposal-level ITT options from immutable Wine rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Iterable

from ..actions import ACTION_NAMES
from ..learning_features import tree_candidate_vector
from ..hazard_representation import (
    HAZARD_PRIMITIVE_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    summarize_hazard_primitives,
)
from ..th06.learning_adapter import ACTION_FEATURE_NAMES, OBSERVATION_FEATURE_NAMES
from ..wine_corpus_registry import WineCorpusEntry
from ..wine_transitions import iter_transition_rows, validate_wine_run
from .feature_contract import (
    compact_actor_feature_names,
    richer_causal_context_feature_names,
)
from .outcomes import OptionOutcome, assert_hit_conservation


RECONSTRUCTED_V9_POLICY = "safe-option-exploration-v1"
RECONSTRUCTED_V9_EPSILON = 0.10
NONEXECUTED_TERMINATIONS = frozenset({
    "publication-rejected",
    "hard-empty",
    "authority-loss",
    "stage-transition",
    "bomb",
    "physical-hit",
})


@dataclass(frozen=True)
class FactualOption:
    episode_id: str
    source_id: str
    cohort_id: str
    transition_schema: str
    stage: int
    option_id: str
    option_index: int
    sequence: int
    frame: int
    proposal_action: str
    boundary_executed_action: str
    complied: bool
    baseline_action: str
    legal_actions: tuple[str, ...]
    behavior_probabilities: tuple[float, ...]
    factual_probability: float
    candidate_features: tuple[tuple[float, ...], ...]
    causal_context_features: tuple[float, ...]
    hit_cost: int
    duration_frames: int
    terminal: bool = False

    def __post_init__(self) -> None:
        count = len(self.legal_actions)
        if (
            not self.episode_id
            or not self.source_id
            or not self.cohort_id
            or not self.option_id
            or self.option_index < 0
            or self.sequence < 0
            or self.frame < 0
            or self.proposal_action not in self.legal_actions
            or self.boundary_executed_action not in ACTION_NAMES
            or self.complied
            != (self.proposal_action == self.boundary_executed_action)
            or self.baseline_action not in self.legal_actions
            or len(self.behavior_probabilities) != count
            or len(self.candidate_features) != count
            or any(
                len(row) != len(compact_actor_feature_names())
                for row in self.candidate_features
            )
            or len(self.causal_context_features)
            != len(richer_causal_context_feature_names())
            or any(not math.isfinite(value) for value in self.causal_context_features)
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.behavior_probabilities
            )
            or not math.isclose(
                sum(self.behavior_probabilities),
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.factual_probability,
                self.behavior_probabilities[
                    self.legal_actions.index(self.proposal_action)
                ],
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or self.hit_cost < 0
            or self.duration_frames <= 0
        ):
            raise ValueError("factual option contract is invalid")

    @property
    def factual_index(self) -> int:
        """Index of the randomized proposal, not post-revalidation movement."""
        return self.legal_actions.index(self.proposal_action)

    @property
    def baseline_index(self) -> int:
        return self.legal_actions.index(self.baseline_action)


@dataclass(frozen=True)
class FactualEpisode:
    episode_id: str
    source_id: str
    cohort_id: str
    transition_schema: str
    stage: int
    options: tuple[FactualOption, ...]
    pre_option_hits: int
    manifest_hits: int
    transition_rows: int

    def __post_init__(self) -> None:
        if (
            not self.options
            or any(option.episode_id != self.episode_id for option in self.options)
            or any(option.source_id != self.source_id for option in self.options)
            or any(option.stage != self.stage for option in self.options)
            or tuple(option.option_index for option in self.options)
            != tuple(range(len(self.options)))
            or any(option.terminal for option in self.options[:-1])
            or not self.options[-1].terminal
        ):
            raise ValueError("factual episode grouping is invalid")
        assert_hit_conservation(
            tuple(
                OptionOutcome(
                    option.hit_cost,
                    option.duration_frames,
                    option.terminal,
                )
                for option in self.options
            ),
            pre_option_hits=self.pre_option_hits,
            manifest_hits=self.manifest_hits,
        )


@dataclass
class _PendingOption:
    option_id: str
    sequence: int
    frame: int
    proposal_action: str
    boundary_executed_action: str
    complied: bool
    baseline: str
    legal: tuple[str, ...]
    probabilities: tuple[float, ...]
    factual_probability: float
    candidate_features: tuple[tuple[float, ...], ...]
    causal_context_features: tuple[float, ...]
    hit_cost: int = 0
    physical_elapsed: int = 0
    termination: str | None = None


def _physical_frame(snapshot_ref: object) -> int:
    marker = str(snapshot_ref).rsplit(":f", 1)
    if len(marker) != 2:
        raise ValueError("snapshot reference has no physical frame")
    return int(marker[1])


def _feature_rows(
    row: dict[str, object],
    legal: tuple[str, ...],
    *,
    option_index: int,
) -> tuple[tuple[float, ...], ...]:
    context = row.get("policy_context")
    if not isinstance(context, dict):
        raise TypeError("factual option boundary has no policy context")
    result = []
    for action in legal:
        base = tree_candidate_vector(
            observation_features=context.get("observation_features"),
            action_features=context.get("action_features"),
            action=action,
            baseline_action=str(row.get("baseline_action", "")),
            current_action=str(context.get("current_action", "")),
            observation_names=OBSERVATION_FEATURE_NAMES,
            action_names=ACTION_FEATURE_NAMES,
        )
        result.append((*base, math.log1p(option_index)))
    if any(len(values) != len(compact_actor_feature_names()) for values in result):
        raise RuntimeError("causal actor feature width drifted")
    return tuple(result)


def _named_values(raw: object, names: tuple[str, ...], *, label: str) -> tuple[float, ...]:
    if not isinstance(raw, list):
        raise TypeError(f"{label} is not a named feature list")
    values = {}
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{label} row is invalid")
        name, value = str(item[0]), float(item[1])
        if name in values or not math.isfinite(value):
            raise ValueError(f"{label} contains duplicate/non-finite values")
        values[name] = value
    if set(values) != set(names):
        raise ValueError(f"{label} feature contract is incomplete")
    return tuple(values[name] for name in names)


def _causal_context_features(row: dict[str, object]) -> tuple[float, ...]:
    context = row.get("policy_context")
    if not isinstance(context, dict):
        raise TypeError("option boundary has no causal context")
    history = _named_values(
        context.get("history_features"), HISTORY_FEATURE_NAMES, label="history"
    )
    raw_hazards = context.get("hazard_primitives")
    if not isinstance(raw_hazards, list):
        raise TypeError("hazard primitives are not a list")
    hazards = tuple(tuple(float(value) for value in item) for item in raw_hazards)
    if any(len(item) != len(HAZARD_PRIMITIVE_FEATURE_NAMES) for item in hazards):
        raise ValueError("hazard primitive width drifted")
    result = (*history, *summarize_hazard_primitives(hazards))
    if len(result) != len(richer_causal_context_feature_names()):
        raise RuntimeError("richer causal context width drifted")
    return tuple(result)


def _reconstructed_probabilities(
    legal: tuple[str, ...], baseline: str
) -> tuple[float, ...]:
    if len(legal) == 1:
        return (1.0,)
    exploratory = RECONSTRUCTED_V9_EPSILON / len(legal)
    return tuple(
        exploratory + (
            1.0 - RECONSTRUCTED_V9_EPSILON if action == baseline else 0.0
        )
        for action in legal
    )


def _behavior_probabilities(
    row: dict[str, object],
    option: dict[str, object],
    legal: tuple[str, ...],
    *,
    reconstructible: bool,
) -> tuple[float, ...]:
    raw = option.get("behavior_probabilities")
    if isinstance(raw, list) and raw:
        values: dict[str, float] = {}
        for item in raw:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("recorded propensity vector row is invalid")
            name, probability = str(item[0]), float(item[1])
            if (
                name in values
                or not math.isfinite(probability)
                or probability <= 0.0
            ):
                raise ValueError("recorded propensity vector is invalid")
            values[name] = probability
        if set(values) != set(legal):
            raise ValueError("recorded propensity vector is incomplete")
        probabilities = tuple(values[action] for action in legal)
    elif reconstructible:
        if row.get("policy_id") != RECONSTRUCTED_V9_POLICY:
            raise ValueError("reconstructible propensity behavior policy drifted")
        probabilities = _reconstructed_probabilities(
            legal, str(row.get("baseline_action", ""))
        )
    else:
        raise ValueError("factual option boundary lacks complete propensities")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("behavior probabilities do not sum to one")
    return probabilities


def aggregate_factual_episode(
    rows: Iterable[dict[str, object]],
    *,
    episode_id: str,
    source_id: str,
    transition_schema: str,
    stage: int,
    manifest_hits: int,
    reconstructible_propensity: bool,
) -> FactualEpisode:
    """Aggregate every randomized proposal and its factual downstream interval.

    The treatment is assignment (intention-to-treat), not post-assignment
    compliance. Native revalidation/fallback is part of the factual outcome.
    Filtering on ``executed_action == intent`` would condition on an
    action-dependent post-treatment event and invalidate the logged propensity.
    """
    completed: list[_PendingOption] = []
    current: _PendingOption | None = None
    pre_option_hits = 0
    physical_hit_rows = 0
    transition_rows = 0
    policy_ids: set[str] = set()
    for row in rows:
        transition_rows += 1
        option = row.get("option")
        if option is not None and not isinstance(option, dict):
            raise TypeError("option trace is not an object")
        outcome = row.get("outcome_terms")
        if not isinstance(outcome, dict):
            raise TypeError("transition row has no factual outcome")
        if outcome.get("bomb_used") is True or outcome.get("authority_lost") is True:
            raise ValueError("Bomb/authority-loss row cannot enter learner data")
        hit_cost = int(outcome.get("life_lost") is True)
        physical_hit_rows += hit_cost
        option_id = str(option.get("option_id", "")) if option else ""
        boundary = bool(option and option.get("boundary") is True)
        proposal_action = str(option.get("intent", "")) if option else ""
        executed_action = str(row.get("executed_action", ""))
        if option is not None and (not option_id or not proposal_action):
            raise ValueError("option trace has no identity/intent")

        if option is not None and executed_action != proposal_action:
            if (
                option.get("termination_reason") not in NONEXECUTED_TERMINATIONS
                or row.get("learning_eligible") is not False
            ):
                raise ValueError("unexecuted option is not explicitly rejected")
            if not boundary:
                if current is None or option_id != current.option_id:
                    raise ValueError(
                        "unexecuted continuation escaped its proposal boundary"
                    )
                current.hit_cost += hit_cost
                current.physical_elapsed += int(outcome.get("elapsed_frames", 0))
                if current.termination is None:
                    current.termination = str(option.get("termination_reason"))
                continue

        if boundary:
            if current is not None:
                if current.termination is None:
                    current.termination = str(
                        option.get("preceding_termination_reason")
                        or "next-option-boundary"
                    )
                completed.append(current)
            legal_raw = row.get("legal_actions")
            baseline = str(row.get("baseline_action", ""))
            if not isinstance(legal_raw, list):
                raise TypeError("option boundary has no native-safe action set")
            legal = tuple(str(value) for value in legal_raw)
            if (
                not legal
                or len(set(legal)) != len(legal)
                or proposal_action not in legal
                or baseline not in legal
                or executed_action not in ACTION_NAMES
            ):
                raise ValueError("proposal/fallback action contract is invalid")
            probabilities = _behavior_probabilities(
                row,
                option,
                legal,
                reconstructible=reconstructible_propensity,
            )
            factual_probability = probabilities[legal.index(proposal_action)]
            if (
                not math.isclose(
                    float(option.get("boundary_probability", 0.0)),
                    factual_probability,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row.get("behavior_probability", 0.0)),
                    factual_probability,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("factual propensity disagrees with assignment")
            policy_id = str(row.get("policy_id", ""))
            if not policy_id:
                raise ValueError("option boundary has no behavior policy identity")
            policy_ids.add(policy_id)
            current = _PendingOption(
                option_id=option_id,
                sequence=int(row.get("sequence", -1)),
                frame=_physical_frame(row.get("snapshot_ref")),
                proposal_action=proposal_action,
                boundary_executed_action=executed_action,
                complied=executed_action == proposal_action,
                baseline=baseline,
                legal=legal,
                probabilities=probabilities,
                factual_probability=factual_probability,
                candidate_features=_feature_rows(
                    row, legal, option_index=len(completed)
                ),
                causal_context_features=_causal_context_features(row),
            )
        elif option is not None:
            if current is None or option_id != current.option_id:
                raise ValueError("option continuation escaped its factual boundary")
            if proposal_action != current.proposal_action:
                raise ValueError("proposal intent changed during treatment")

        if current is None:
            pre_option_hits += hit_cost
            continue
        current.hit_cost += hit_cost
        current.physical_elapsed += int(outcome.get("elapsed_frames", 0))
        termination = option.get("termination_reason") if option else None
        if termination is not None:
            current.termination = str(termination)

    if current is not None:
        if current.termination is None:
            current.termination = "complete-stage-tail"
        completed.append(current)
    if not completed or not policy_ids:
        raise ValueError("complete Stage contains no factual option assignments")
    if physical_hit_rows != manifest_hits:
        raise ValueError("transition rows and manifest disagree on physical HITs")
    cohort_id = "+".join(sorted(policy_ids))
    options = []
    for index, item in enumerate(completed):
        next_frame = completed[index + 1].frame if index + 1 < len(completed) else None
        duration = (
            next_frame - item.frame
            if next_frame is not None
            else max(1, item.physical_elapsed)
        )
        if duration <= 0:
            raise ValueError("factual option frames are not increasing")
        options.append(FactualOption(
            episode_id=episode_id,
            source_id=source_id,
            cohort_id=cohort_id,
            transition_schema=transition_schema,
            stage=stage,
            option_id=item.option_id,
            option_index=index,
            sequence=item.sequence,
            frame=item.frame,
            proposal_action=item.proposal_action,
            boundary_executed_action=item.boundary_executed_action,
            complied=item.complied,
            baseline_action=item.baseline,
            legal_actions=item.legal,
            behavior_probabilities=item.probabilities,
            factual_probability=item.factual_probability,
            candidate_features=item.candidate_features,
            causal_context_features=item.causal_context_features,
            hit_cost=item.hit_cost,
            duration_frames=duration,
        ))
    options[-1] = replace(options[-1], terminal=True)
    return FactualEpisode(
        episode_id=episode_id,
        source_id=source_id,
        cohort_id=cohort_id,
        transition_schema=transition_schema,
        stage=stage,
        options=tuple(options),
        pre_option_hits=pre_option_hits,
        manifest_hits=manifest_hits,
        transition_rows=transition_rows,
    )


def load_factual_episode(entry: WineCorpusEntry) -> FactualEpisode:
    run, manifest, schema = validate_wine_run(
        entry.path,
        expected_transition_schema=entry.transition_schema,
        require_stage_complete=True,
    )
    reconstructible = (
        "reconstructible_complete_behavior_propensity" in entry.capabilities
    )
    if not reconstructible and (
        "recorded_complete_behavior_propensity" not in entry.capabilities
    ):
        raise ValueError("registry entry lacks complete propensity capability")
    return aggregate_factual_episode(
        iter_transition_rows(
            entry.path,
            manifest,
            expected_transition_schema=schema,
        ),
        episode_id=str(run.get("run_id", entry.run_id)),
        source_id=entry.source,
        transition_schema=schema,
        stage=entry.stage,
        manifest_hits=entry.physical_hits,
        reconstructible_propensity=reconstructible,
    )
