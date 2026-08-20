"""Causal option-level dataset over explicitly admitted Wine episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

from th06_rl.feature_contract import validate_actor_feature_rows
from th06_rl.th06.source_dataset import (
    SourceFrameBundle,
    _rows,
    _stream_paths,
    iter_source_frames,
)


OPTION_DATASET_SCHEMA = "th06-rl-causal-options-v1"
DISCOUNT = 1.0


class OfflineOptionError(RuntimeError):
    """Logged transitions do not define a valid causal option sequence."""


@dataclass(frozen=True)
class ActorState:
    observation_features: tuple[tuple[str, float], ...]
    action_features: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    history_features: tuple[tuple[str, float], ...]
    legal_actions: tuple[str, ...]
    baseline_action: str
    current_action: str


@dataclass(frozen=True)
class OfflineOptionTransition:
    schema: str
    episode_id: str
    episode_unit: str
    option_id: str
    start_sequence: int
    end_sequence: int
    start_stage: int
    diagnostic_scope: str
    action: str
    behavior_probability: float
    behavior_probabilities: tuple[tuple[str, float], ...]
    state: ActorState
    next_state: ActorState | None
    physical_hit_cost: int
    elapsed_frames: int
    terminal: bool
    eligible: bool
    exclusion_reasons: tuple[str, ...]


@dataclass
class _ActiveOption:
    episode_id: str
    episode_unit: str
    option_id: str
    start_sequence: int
    end_sequence: int
    start_stage: int
    diagnostic_scope: str
    action: str
    behavior_probability: float
    behavior_probabilities: tuple[tuple[str, float], ...]
    state: ActorState
    physical_hit_cost: int = 0
    elapsed_frames: int = 0
    last_declared_elapsed: int = 0
    exclusions: set[str] = field(default_factory=set)


def _named_values(rows, *, label: str) -> tuple[tuple[str, float], ...]:
    result = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise OfflineOptionError(f"{label} is not a name/value sequence")
        try:
            name, value = str(row[0]), float(row[1])
        except (TypeError, ValueError) as error:
            raise OfflineOptionError(f"{label} contains a non-numeric value") from error
        if not name or not math.isfinite(value):
            raise OfflineOptionError(f"{label} contains invalid numeric facts")
        result.append((name, value))
    return tuple(result)


def _numeric(value, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise OfflineOptionError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise OfflineOptionError(f"{label} is non-finite")
    return result


def _actor_state(
    bundle: SourceFrameBundle,
    transition: dict[str, object],
) -> ActorState:
    context = transition.get("policy_context") or {}
    observation = _named_values(
        context.get("observation_features", ()),
        label="observation features",
    )
    actions = tuple(
        (
            str(row[0]),
            _named_values(row[1], label=f"action {row[0]} features"),
        )
        for row in context.get("action_features", ())
        if isinstance(row, (list, tuple)) and len(row) == 2
    )
    history = _named_values(
        context.get("history_features", ()),
        label="causal history features",
    )
    validate_actor_feature_rows(observation, actions, history)
    legal = tuple(str(action) for action in transition.get("legal_actions", ()))
    baseline = str(transition.get("baseline_action"))
    current = str(context.get("current_action"))
    action_names = {action for action, _values in actions}
    scope = transition.get("scope")
    if (
        not legal
        or len(set(legal)) != len(legal)
        or baseline not in legal
        or current not in action_names
        or not set(legal).issubset(action_names)
        or not isinstance(scope, dict)
        or bundle.control.stage != int(scope.get("stage", -1))
    ):
        raise OfflineOptionError("option decision state/action scope is incomplete")
    return ActorState(observation, actions, history, legal, baseline, current)


def _probabilities(
    option: dict[str, object],
    state: ActorState,
    transition: dict[str, object],
) -> tuple[tuple[str, float], ...]:
    rows = _named_values(
        option.get("behavior_probabilities", ()),
        label="behavior probabilities",
    )
    values = dict(rows)
    intent = str(option.get("intent"))
    factual = _numeric(
        option.get("boundary_probability"),
        label="option boundary probability",
    )
    conditional = _numeric(
        option.get("conditional_probability"),
        label="option conditional probability",
    )
    logged = _numeric(
        transition.get("behavior_probability"),
        label="transition behavior probability",
    )
    if (
        len(values) != len(rows)
        or intent not in values
        or set(values) != set(state.legal_actions)
        or any(value < 0.0 for value in values.values())
        or values[intent] <= 0.0
        or not math.isclose(sum(values.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9)
        or not math.isclose(values[intent], factual, rel_tol=1e-12, abs_tol=1e-12)
        or not math.isclose(conditional, factual, rel_tol=1e-12, abs_tol=1e-12)
        or not math.isclose(logged, factual, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise OfflineOptionError("option boundary propensity vector is invalid")
    return rows


def _finish(
    active: _ActiveOption,
    *,
    next_state: ActorState | None,
    terminal: bool,
) -> OfflineOptionTransition:
    if active.elapsed_frames <= 0:
        raise OfflineOptionError("option has no factual physical transition")
    exclusions = tuple(sorted(active.exclusions))
    return OfflineOptionTransition(
        OPTION_DATASET_SCHEMA,
        active.episode_id,
        active.episode_unit,
        active.option_id,
        active.start_sequence,
        active.end_sequence,
        active.start_stage,
        active.diagnostic_scope,
        active.action,
        active.behavior_probability,
        active.behavior_probabilities,
        active.state,
        next_state,
        active.physical_hit_cost,
        active.elapsed_frames,
        terminal,
        not exclusions,
        exclusions,
    )


def _joined_steps(
    run_dir: Path,
) -> Iterator[tuple[SourceFrameBundle, SourceFrameBundle, dict[str, object]]]:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    transitions = iter(_rows(_stream_paths(run_dir, manifest, "transitions")))
    frames = iter(iter_source_frames(run_dir))
    try:
        before = next(frames)
    except StopIteration:
        return
    for after in frames:
        try:
            transition = next(transitions)
        except StopIteration as error:
            raise OfflineOptionError("corpus has fewer transitions than frame links") from error
        sequence = int(transition.get("sequence", -1))
        if (
            sequence != before.sequence
            or transition.get("snapshot_ref") != before.snapshot_id
            or transition.get("next_snapshot_ref") != after.snapshot_id
            or after.sequence != before.sequence + 1
        ):
            raise OfflineOptionError("frame/transition linkage is not contiguous")
        yield before, after, transition
        before = after
    try:
        extra = next(transitions)
    except StopIteration:
        return
    raise OfflineOptionError(f"orphan transition {extra.get('sequence')}")


def iter_offline_options(run_dir: Path) -> Iterator[OfflineOptionTransition]:
    """Aggregate one admitted physical episode into choice-to-choice options."""
    active: _ActiveOption | None = None
    pending: _ActiveOption | None = None
    for before, _after, transition in _joined_steps(run_dir):
        option = transition.get("option")
        if option is not None and not isinstance(option, dict):
            raise OfflineOptionError("option metadata is not an object")
        boundary = bool(option and option.get("boundary"))
        state = _actor_state(before, transition) if boundary else None
        if boundary:
            assert option is not None and state is not None
            if active is not None:
                raise OfflineOptionError(
                    "new option boundary preceded an unterminated option"
                )
            if pending is not None:
                yield _finish(pending, next_state=state, terminal=False)
                pending = None
            probabilities = _probabilities(option, state, transition)
            action = str(option.get("intent"))
            probability = _numeric(
                option.get("boundary_probability"),
                label="option boundary probability",
            )
            episode = transition.get("episode")
            scope = transition.get("scope")
            if not isinstance(episode, dict) or not isinstance(scope, dict):
                raise OfflineOptionError("option boundary lacks episode/scope identity")
            episode_id = str(episode.get("id", ""))
            episode_unit = str(episode.get("unit", ""))
            option_id = str(option.get("option_id", ""))
            diagnostic_scope = str(scope.get("key", ""))
            if not episode_id or not episode_unit or not option_id or not diagnostic_scope:
                raise OfflineOptionError("option boundary identity is incomplete")
            active = _ActiveOption(
                episode_id,
                episode_unit,
                option_id,
                int(transition["sequence"]),
                int(transition["sequence"]),
                before.control.stage,
                diagnostic_scope,
                action,
                probability,
                probabilities,
                state,
            )
        elif option is not None:
            if active is None:
                raise OfflineOptionError("option continuation has no active boundary")
            if (
                option.get("option_id") != active.option_id
                or option.get("intent") != active.action
            ):
                raise OfflineOptionError("option continuation identity changed")
            rows = _named_values(
                option.get("behavior_probabilities", ()),
                label="continuation behavior probabilities",
            )
            boundary_probability = _numeric(
                option.get("boundary_probability"),
                label="continuation boundary probability",
            )
            conditional = _numeric(
                option.get("conditional_probability"),
                label="continuation conditional probability",
            )
            logged = _numeric(
                transition.get("behavior_probability"),
                label="continuation behavior probability",
            )
            if (
                rows != active.behavior_probabilities
                or not math.isclose(
                    boundary_probability,
                    active.behavior_probability,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(conditional, 1.0, rel_tol=0.0, abs_tol=0.0)
                or not math.isclose(logged, 1.0, rel_tol=0.0, abs_tol=0.0)
            ):
                raise OfflineOptionError("option continuation propensity changed")
        elif active is not None:
            raise OfflineOptionError(
                "active option disappeared without a recorded termination"
            )

        if active is None:
            continue
        assert option is not None
        episode = transition.get("episode")
        if (
            not isinstance(episode, dict)
            or episode.get("id") != active.episode_id
            or episode.get("unit") != active.episode_unit
        ):
            raise OfflineOptionError("option crossed a physical episode identity")
        declared_elapsed = int(option.get("elapsed_frames_at_decision", -1))
        if declared_elapsed != active.last_declared_elapsed + 1:
            raise OfflineOptionError("option decision elapsed sequence is not contiguous")
        active.last_declared_elapsed = declared_elapsed
        active.end_sequence = int(transition["sequence"])
        outcome = transition.get("outcome_terms") or {}
        elapsed = int(outcome.get("elapsed_frames", -1))
        hit = outcome.get("life_lost")
        if (
            elapsed < 0
            or not isinstance(hit, bool)
            or int(option.get("physical_elapsed_frames", -1)) != elapsed
        ):
            raise OfflineOptionError("option outcome lacks factual time/HIT state")
        active.elapsed_frames += elapsed
        active.physical_hit_cost += int(hit)
        if transition.get("executed_action") != active.action:
            active.exclusions.add("option-action-not-executed")
        if transition.get("learning_eligible") is not True:
            reasons = transition.get("learning_exclusion_reasons") or (
                "transition-not-learning-eligible",
            )
            active.exclusions.update(str(reason) for reason in reasons)
        if outcome.get("bomb_used"):
            active.exclusions.add("bomb")
        if outcome.get("authority_lost"):
            active.exclusions.add("authority-loss")
        termination = option.get("termination_reason")
        if termination is not None:
            if not isinstance(termination, str) or not termination:
                raise OfflineOptionError("option termination reason is invalid")
            if pending is not None:
                raise OfflineOptionError("multiple options await one decision boundary")
            pending = active
            active = None

    if active is not None:
        yield _finish(active, next_state=None, terminal=True)
    elif pending is not None:
        yield _finish(pending, next_state=None, terminal=True)


def whole_episode_split(
    episode_ids: tuple[str, ...],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[frozenset[str], frozenset[str]]:
    """Deterministic split whose indivisible unit is one physical episode."""
    unique = tuple(sorted(set(episode_ids)))
    if (
        len(unique) < 2
        or not 0.0 < validation_fraction < 1.0
        or not 0 <= seed < 2**64
    ):
        raise ValueError("whole-episode split requires >=2 episodes and valid bounds")
    ordered = sorted(
        unique,
        key=lambda episode: hashlib.sha256(
            f"{seed}:{episode}".encode("utf-8")
        ).digest(),
    )
    validation_count = min(
        len(ordered) - 1,
        max(1, round(len(ordered) * validation_fraction)),
    )
    validation = frozenset(ordered[:validation_count])
    training = frozenset(ordered[validation_count:])
    if training & validation or training | validation != set(unique):
        raise RuntimeError("whole-episode split lost or duplicated an episode")
    return training, validation
