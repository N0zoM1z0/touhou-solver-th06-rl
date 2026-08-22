"""Algorithm- and game-independent access to complete factual episodes."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

from .actions import ACTION_NAMES
from .corpus import (
    FRAME_SCHEMA,
    LEGACY_FRAME_SCHEMAS,
    LEGACY_TRANSITION_SCHEMAS,
    MANIFEST_SCHEMA,
    RUN_SCHEMA,
    TRANSITION_SCHEMA,
)
from .policy_api import ActionExposure


DECISION_EPOCH_SCHEMA = "th06-rl-decision-epoch-v1"
PORTABLE_ROOT_SCHEMA = "th06-rl-portable-policy-root-v1"
POLICY_INVOCATION_REASONS = frozenset({
    "ok",
    "stale-retain-observed-shield-current",
})
NON_POLICY_INVOCATION_REASONS = frozenset({
    "input-lease",
    "passive",
    "physical-hit",
    "player-not-active",
})


class EpisodeDatasetError(RuntimeError):
    """Stored rows do not form one intact physical episode."""


@dataclass(frozen=True)
class EpisodeFrame:
    sequence: int
    snapshot_id: str
    stage: int
    player_state: int
    snapshot: dict[str, object]
    scope: dict[str, object]
    decision: dict[str, object]


@dataclass(frozen=True)
class EpisodeTransition:
    sequence: int
    snapshot_id: str
    next_snapshot_id: str
    legal_actions: tuple[str, ...]
    baseline_action: str | None
    proposed_action: str | None
    published_action: str | None
    commanded_action: str | None
    sampled_action: str | None
    executed_action: str | None
    behavior_probability: float
    behavior_probabilities: tuple[tuple[str, float], ...]
    policy_id: str | None
    action_exposure: ActionExposure | None
    policy_context: dict[str, object]
    outcome: dict[str, object]
    learning_eligible: bool
    learning_exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PortableDecisionRoot:
    """Versioned actor-visible facts at one actual policy invocation."""

    schema: str
    player_x: float
    player_y: float
    power: int
    bullet_count: int
    laser_count: int
    current_action: str
    locally_admissible_actions: tuple[str, ...]
    shield_action_evaluations: tuple[
        tuple[str, float | None, float, float], ...
    ]

    def __post_init__(self) -> None:
        if self.schema != PORTABLE_ROOT_SCHEMA:
            raise EpisodeDatasetError("portable root schema mismatch")
        if not math.isfinite(self.player_x) or not math.isfinite(self.player_y):
            raise EpisodeDatasetError("portable player position is not finite")
        if min(self.power, self.bullet_count, self.laser_count) < 0:
            raise EpisodeDatasetError("portable counters must be nonnegative")
        legal = self.locally_admissible_actions
        if not legal or len(set(legal)) != len(legal):
            raise EpisodeDatasetError("portable shield action set is empty or duplicated")
        if self.current_action not in ACTION_NAMES or any(
            action not in ACTION_NAMES for action in legal
        ):
            raise EpisodeDatasetError("portable root contains an unknown action")
        evaluations = self.shield_action_evaluations
        if tuple(row[0] for row in evaluations) != legal:
            raise EpisodeDatasetError("portable shield evaluations disagree with action set")
        for _action, clearance, final_x, final_y in evaluations:
            if (
                (clearance is not None and not math.isfinite(clearance))
                or not math.isfinite(final_x)
                or not math.isfinite(final_y)
            ):
                raise EpisodeDatasetError("portable shield evaluation is not finite")


@dataclass(frozen=True)
class DecisionEpoch:
    """One factual policy intervention through the next intervention/terminal."""

    schema: str
    episode_id: str
    index: int
    start_sequence: int
    next_sequence: int
    snapshot_id: str
    next_snapshot_id: str
    observation: PortableDecisionRoot
    next_observation: PortableDecisionRoot | None
    policy_id: str
    baseline_action: str | None
    proposed_action: str | None
    published_action: str | None
    behavior_probability: float
    behavior_probabilities: tuple[tuple[str, float], ...]
    action_exposure: ActionExposure | None
    transition_sequences: tuple[int, ...]
    commanded_actions: tuple[str | None, ...]
    sampled_actions: tuple[str | None, ...]
    executed_actions: tuple[str | None, ...]
    elapsed_game_frames: int
    hit_cost: int
    terminal: bool
    learning_eligible: bool
    exclusion_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != DECISION_EPOCH_SCHEMA:
            raise EpisodeDatasetError("decision epoch schema mismatch")
        length = len(self.transition_sequences)
        if not all(
            len(actions) == length
            for actions in (
                self.commanded_actions,
                self.sampled_actions,
                self.executed_actions,
            )
        ):
            raise EpisodeDatasetError("decision action evidence is not interval-aligned")
        if self.elapsed_game_frames < 0 or self.hit_cost < 0:
            raise EpisodeDatasetError("decision interval totals must be nonnegative")
        if self.terminal != (self.next_observation is None):
            raise EpisodeDatasetError("decision terminal and successor disagree")
        if self.learning_eligible and (
            self.published_action is None or not self.transition_sequences
        ):
            raise EpisodeDatasetError("eligible decision lacks an executed interval")


def _stream_paths(
    run_dir: Path,
    manifest: dict[str, object],
    stream: str,
) -> tuple[Path, ...]:
    declarations = [
        row
        for row in manifest.get("shards", ())
        if isinstance(row, dict) and row.get("stream") == stream
    ]
    declarations.sort(key=lambda row: int(row.get("first_sequence", -1)))
    paths = []
    for declaration in declarations:
        path = (run_dir / str(declaration.get("path", ""))).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as error:
            raise EpisodeDatasetError(f"{stream} shard escapes its run") from error
        if not path.is_file():
            raise EpisodeDatasetError(f"missing {stream} shard {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != declaration.get("sha256"):
            raise EpisodeDatasetError(f"{stream} shard digest differs")
        if path.stat().st_size != int(declaration.get("compressed_bytes", -1)):
            raise EpisodeDatasetError(f"{stream} shard byte count differs")
        paths.append(path)
    return tuple(paths)


def _rows(paths: tuple[Path, ...]) -> Iterator[dict[str, object]]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise EpisodeDatasetError(f"non-object row in {path.name}")
                yield row


def _episode_contract(
    run_dir: Path,
) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
    run_dir = run_dir.resolve()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(run, dict) or not isinstance(manifest, dict):
        raise EpisodeDatasetError("episode metadata is not an object")
    metadata = run.get("metadata")
    schemas = run.get("schemas")
    episode = manifest.get("episode")
    outcome = manifest.get("run_outcome")
    if not all(
        isinstance(value, dict)
        for value in (metadata, schemas, episode, outcome)
    ):
        raise EpisodeDatasetError("episode contract is incomplete")
    assert isinstance(metadata, dict)
    assert isinstance(schemas, dict)
    assert isinstance(episode, dict)
    assert isinstance(outcome, dict)
    unit = metadata.get("episode_unit")
    expected_reason = {
        "route": "route-complete",
        "practice-stage": "practice-stage-complete",
    }.get(unit)
    if (
        run.get("schema_version") != RUN_SCHEMA
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or schemas.get("frame") not in ({FRAME_SCHEMA} | LEGACY_FRAME_SCHEMAS)
        or schemas.get("transition")
        not in ({TRANSITION_SCHEMA} | LEGACY_TRANSITION_SCHEMAS)
        or (
            schemas.get("frame") == FRAME_SCHEMA
            and schemas.get("transition") != TRANSITION_SCHEMA
        )
        or (
            schemas.get("frame") in LEGACY_FRAME_SCHEMAS
            and schemas.get("transition") not in LEGACY_TRANSITION_SCHEMAS
        )
        or run.get("run_id") != manifest.get("run_id")
        or episode.get("id") != run.get("run_id")
        or episode.get("unit") != unit
        or manifest.get("complete") is not True
        or manifest.get("dropped_records") != 0
        or manifest.get("stage_trajectory_complete") is not True
        or episode.get("complete") is not True
        or outcome.get("stage_completed") is not True
        or outcome.get("termination_reason") != expected_reason
    ):
        raise EpisodeDatasetError("episode is not one complete factual trajectory")
    return run_dir, manifest, metadata, schemas


def iter_episode_frames(run_dir: Path) -> Iterator[EpisodeFrame]:
    """Yield learner-facing facts without decoding game-specific raw state."""
    run_dir, manifest, metadata, schemas = _episode_contract(run_dir)
    expected_stages = {int(stage) for stage in metadata.get("expected_stages", ())}
    observed_stages: set[int] = set()
    expected_sequence = 0
    for row in _rows(_stream_paths(run_dir, manifest, "frames")):
        sequence = int(row.get("sequence", -1))
        snapshot_id = row.get("snapshot_id")
        snapshot = row.get("snapshot")
        scope = row.get("scope")
        decision = row.get("decision")
        if (
            row.get("schema_version") != schemas["frame"]
            or sequence != expected_sequence
            or not isinstance(snapshot_id, str)
            or not isinstance(snapshot, dict)
            or not isinstance(scope, dict)
            or not isinstance(decision, dict)
        ):
            raise EpisodeDatasetError("frame stream is malformed or non-contiguous")
        stage = int(snapshot.get("stage", -1))
        player_state = int(snapshot.get("player_state", -1))
        if stage < 1 or player_state < 0 or int(scope.get("stage", -1)) != stage:
            raise EpisodeDatasetError("frame scope disagrees with factual scalars")
        observed_stages.add(stage)
        yield EpisodeFrame(
            sequence,
            snapshot_id,
            stage,
            player_state,
            dict(snapshot),
            dict(scope),
            dict(decision),
        )
        expected_sequence += 1
    declared = int((manifest.get("records") or {}).get("frames", -1))
    if expected_sequence == 0 or declared != expected_sequence:
        raise EpisodeDatasetError("frame count is empty or disagrees with manifest")
    if expected_stages and observed_stages != expected_stages:
        raise EpisodeDatasetError("frames do not cover the declared episode stages")


def _action_exposure(value: object) -> ActionExposure | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise EpisodeDatasetError("action exposure is malformed")
    raw_probabilities = value.get("assignment_probabilities")
    if not isinstance(raw_probabilities, list):
        raise EpisodeDatasetError("action exposure assignment distribution is malformed")
    try:
        probabilities = tuple(
            (str(row[0]), float(row[1]))
            for row in raw_probabilities
            if isinstance(row, list) and len(row) == 2
        )
        if len(probabilities) != len(raw_probabilities):
            raise ValueError("incomplete assignment rows")
        return ActionExposure(
            str(value.get("schema", "")),
            int(value["group_id"]),
            int(value["step"]),
            int(value["horizon"]),
            str(value["intended_action"]),
            float(value["assignment_probability"]),
            probabilities,
            str(value["override_reason"])
            if value.get("override_reason") is not None else None,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EpisodeDatasetError("action exposure is invalid") from error


def _iter_episode_links(
    run_dir: Path,
) -> Iterator[tuple[EpisodeFrame, EpisodeTransition, EpisodeFrame]]:
    """Yield each validated before/transition/after triple in one frame pass."""
    run_dir, manifest, _metadata, schemas = _episode_contract(run_dir)
    frames = iter(iter_episode_frames(run_dir))
    transitions = iter(_rows(_stream_paths(run_dir, manifest, "transitions")))
    before = next(frames)
    count = 0
    for after in frames:
        try:
            row = next(transitions)
        except StopIteration as error:
            raise EpisodeDatasetError("transition stream ended before frame links") from error
        outcome = row.get("outcome_terms")
        legal = tuple(str(action) for action in row.get("legal_actions", ()))
        baseline = row.get("baseline_action")
        proposed = row.get("proposed_action")
        published = row.get("published_action")
        policy_id = row.get("policy_id")
        action_exposure = _action_exposure(row.get("action_exposure"))
        policy_context = row.get("policy_context")
        raw_probabilities = row.get("behavior_probabilities", ())
        probabilities = tuple(
            (str(item[0]), float(item[1]))
            for item in raw_probabilities
            if isinstance(item, (list, tuple)) and len(item) == 2
        )
        probability = float(row.get("behavior_probability", float("nan")))
        probability_map = dict(probabilities)
        if (
            row.get("schema_version") != schemas["transition"]
            or int(row.get("sequence", -1)) != before.sequence
            or row.get("snapshot_ref") != before.snapshot_id
            or row.get("next_snapshot_ref") != after.snapshot_id
            or after.sequence != before.sequence + 1
            or not isinstance(outcome, dict)
            or not isinstance(policy_context, dict)
            or len(set(legal)) != len(legal)
            or not math.isfinite(probability)
            or not 0.0 < probability <= 1.0
        ):
            raise EpisodeDatasetError("transition linkage or scalar contract is invalid")
        if published is not None and (
            set(probability_map) != set(legal)
            or len(probability_map) != len(probabilities)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in probability_map.values()
            )
            or not math.isclose(
                sum(probability_map.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9
            )
            or not math.isclose(
                probability_map.get(str(published), -1.0),
                probability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise EpisodeDatasetError("published action propensity is incomplete")
        transition = EpisodeTransition(
            count,
            before.snapshot_id,
            after.snapshot_id,
            legal,
            str(baseline) if baseline is not None else None,
            str(proposed) if proposed is not None else None,
            str(published) if published is not None else None,
            str(row["commanded_action"])
            if row.get("commanded_action") is not None else None,
            str(row["sampled_action"])
            if row.get("sampled_action") is not None else None,
            str(row["executed_action"])
            if row.get("executed_action") is not None else None,
            probability,
            probabilities,
            str(policy_id) if policy_id is not None else None,
            action_exposure,
            dict(policy_context),
            dict(outcome),
            bool(row.get("learning_eligible")),
            tuple(str(reason) for reason in row.get("learning_exclusion_reasons", ())),
        )
        yield before, transition, after
        before = after
        count += 1
    try:
        extra = next(transitions)
    except StopIteration:
        extra = None
    if extra is not None:
        raise EpisodeDatasetError("transition stream has an orphan row")
    declared = int((manifest.get("records") or {}).get("transitions", -1))
    if declared != count:
        raise EpisodeDatasetError("transition count disagrees with manifest")


def iter_episode_transitions(run_dir: Path) -> Iterator[EpisodeTransition]:
    """Yield exactly one factual transition for each adjacent frame pair."""
    for _before, transition, _after in _iter_episode_links(run_dir):
        yield transition


def _is_policy_invocation(frame: EpisodeFrame) -> bool:
    reason = str(frame.decision.get("reason", ""))
    if reason in POLICY_INVOCATION_REASONS:
        return True
    if (
        reason in NON_POLICY_INVOCATION_REASONS
        or reason.startswith("control-dead-end:")
        or reason.startswith("infrastructure-stop:")
    ):
        return False
    raise EpisodeDatasetError(f"unknown decision reason {reason!r}")


def _portable_root(
    frame: EpisodeFrame,
    transition: EpisodeTransition | None,
) -> PortableDecisionRoot:
    if not _is_policy_invocation(frame):
        raise EpisodeDatasetError("portable root was not a policy invocation")
    decision = frame.decision
    snapshot = frame.snapshot
    raw_legal = decision.get("locally_admissible_actions")
    raw_evaluations = decision.get("shield_actions")
    if not isinstance(raw_legal, list) or not isinstance(raw_evaluations, list):
        raise EpisodeDatasetError("policy invocation lacks shield action evidence")
    legal = tuple(str(action) for action in raw_legal)
    evaluations = []
    for row in raw_evaluations:
        if not isinstance(row, list) or len(row) != 4:
            raise EpisodeDatasetError("shield action evaluation is malformed")
        clearance = None if row[1] is None else float(row[1])
        evaluations.append((str(row[0]), clearance, float(row[2]), float(row[3])))
    current_action = decision.get("current_action")
    try:
        root = PortableDecisionRoot(
            PORTABLE_ROOT_SCHEMA,
            float(snapshot["x"]),
            float(snapshot["y"]),
            int(snapshot["current_power"]),
            int(snapshot["live_bullet_count"]),
            int(snapshot["laser_count"]),
            str(current_action),
            legal,
            tuple(evaluations),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EpisodeDatasetError("portable root scalar facts are malformed") from error

    if transition is None:
        return root
    context = transition.policy_context
    expected_context = {
        "current_action": root.current_action,
        "shield_admissible_actions": list(root.locally_admissible_actions),
        "player_x": root.player_x,
        "player_y": root.player_y,
        "power": root.power,
        "bullet_count": root.bullet_count,
        "laser_count": root.laser_count,
        "shield_action_count": len(root.locally_admissible_actions),
        "shield_collision_margin": decision.get("shield_collision_margin"),
    }
    if (
        transition.snapshot_id != frame.snapshot_id
        or transition.legal_actions != root.locally_admissible_actions
        or transition.baseline_action != decision.get("baseline_action")
        or transition.proposed_action != decision.get("proposed_action")
        or transition.published_action != decision.get("published_action")
        or transition.policy_id != decision.get("policy_id")
        or context != expected_context
    ):
        raise EpisodeDatasetError("raw frame and compact policy context disagree")
    return root


def _behavior_evidence(
    frame: EpisodeFrame,
    root: PortableDecisionRoot,
) -> tuple[
    str | None,
    str | None,
    float,
    tuple[tuple[str, float], ...],
    ActionExposure | None,
]:
    decision = frame.decision
    proposed = decision.get("proposed_action")
    published = decision.get("published_action")
    exposure = _action_exposure(decision.get("action_exposure"))
    raw_probabilities = decision.get("behavior_probabilities")
    if not isinstance(raw_probabilities, list):
        raise EpisodeDatasetError("decision behavior distribution is malformed")
    try:
        probability = float(decision["behavior_probability"])
        probabilities = tuple(
            (str(row[0]), float(row[1]))
            for row in raw_probabilities
            if isinstance(row, list) and len(row) == 2
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EpisodeDatasetError("decision propensity is malformed") from error
    if len(probabilities) != len(raw_probabilities) or not (
        math.isfinite(probability) and 0.0 < probability <= 1.0
    ):
        raise EpisodeDatasetError("decision propensity is invalid")
    if proposed is not None and str(proposed) not in ACTION_NAMES:
        raise EpisodeDatasetError("proposed action is outside the canonical vocabulary")
    if published is not None:
        published = str(published)
        probability_map = dict(probabilities)
        if (
            published not in root.locally_admissible_actions
            or set(probability_map) != set(root.locally_admissible_actions)
            or len(probability_map) != len(probabilities)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in probability_map.values()
            )
            or not math.isclose(sum(probability_map.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(
                probability_map.get(published, -1.0),
                probability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise EpisodeDatasetError("published decision propensity is incomplete")
    return (
        str(proposed) if proposed is not None else None,
        published,
        probability,
        probabilities,
        exposure,
    )


def _decision_epoch(
    *,
    episode_id: str,
    index: int,
    start: EpisodeFrame,
    successor: EpisodeFrame,
    transitions: tuple[EpisodeTransition, ...],
    terminal: bool,
) -> DecisionEpoch:
    start_transition = transitions[0] if transitions else None
    root = _portable_root(start, start_transition)
    proposed, published, probability, probabilities, exposure = _behavior_evidence(
        start, root
    )
    if start_transition is not None and (
        start_transition.behavior_probability != probability
        or start_transition.behavior_probabilities != probabilities
        or start_transition.action_exposure != exposure
    ):
        raise EpisodeDatasetError("raw frame and transition propensity disagree")
    policy_id = start.decision.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise EpisodeDatasetError("policy invocation lacks an identity")
    commanded = tuple(item.commanded_action for item in transitions)
    sampled = tuple(item.sampled_action for item in transitions)
    executed = tuple(item.executed_action for item in transitions)
    raw_exclusions = tuple(
        f"{item.sequence}:{reason}"
        for item in transitions
        for reason in item.learning_exclusion_reasons
    )
    hard_exclusions = []
    if published is None:
        hard_exclusions.append("decision-action-not-published")
    if not transitions:
        hard_exclusions.append("empty-decision-interval")
    elif published is not None:
        if commanded[0] != published:
            raise EpisodeDatasetError("published and first commanded action disagree")
        if published not in executed:
            hard_exclusions.append("published-action-not-executed")
    if any(
        reason in ("observation-gap", "bomb", "infrastructure-failure")
        for item in transitions
        for reason in item.learning_exclusion_reasons
    ):
        hard_exclusions.append("invalid-factual-link")
    if any(item.action_exposure != exposure for item in transitions):
        hard_exclusions.append("exposure-metadata-discontinuous")
    next_root = None if terminal else _portable_root(successor, None)
    return DecisionEpoch(
        DECISION_EPOCH_SCHEMA,
        episode_id,
        index,
        start.sequence,
        successor.sequence,
        start.snapshot_id,
        successor.snapshot_id,
        root,
        next_root,
        policy_id,
        start_transition.baseline_action if start_transition is not None else None,
        proposed,
        published,
        probability,
        probabilities,
        exposure,
        tuple(item.sequence for item in transitions),
        commanded,
        sampled,
        executed,
        sum(int(item.outcome.get("elapsed_frames", -1)) for item in transitions),
        sum(bool(item.outcome.get("life_lost")) for item in transitions),
        terminal,
        not hard_exclusions,
        tuple(hard_exclusions) + raw_exclusions,
    )


def iter_decision_epochs(run_dir: Path) -> Iterator[DecisionEpoch]:
    """Derive factual policy-decision intervals and audit all episode HITs."""
    _run_dir, manifest, _metadata, _schemas = _episode_contract(run_dir)
    episode = manifest["episode"]
    outcome = manifest["run_outcome"]
    assert isinstance(episode, dict)
    assert isinstance(outcome, dict)
    episode_id = str(episode["id"])
    links = iter(_iter_episode_links(run_dir))
    try:
        first_before, first_transition, first_after = next(links)
    except StopIteration as error:
        raise EpisodeDatasetError("decision view requires at least one transition") from error

    start: EpisodeFrame | None = (
        first_before if _is_policy_invocation(first_before) else None
    )
    interval: list[EpisodeTransition] = []
    prefix_hits = 0
    raw_hits = 0
    decision_hits = 0
    bombs = 0
    infrastructure_failures = 0
    decision_index = 0
    last_frame = first_after

    def consume(
        before: EpisodeFrame,
        transition: EpisodeTransition,
        after: EpisodeFrame,
    ) -> Iterator[DecisionEpoch]:
        nonlocal start, interval, prefix_hits, raw_hits, decision_hits
        nonlocal bombs, infrastructure_failures, decision_index, last_frame
        last_frame = after
        hit = int(bool(transition.outcome.get("life_lost")))
        raw_hits += hit
        bombs += int(bool(transition.outcome.get("bomb_used")))
        infrastructure_failures += int(
            bool(transition.outcome.get("infrastructure_failed"))
        )
        if start is None:
            prefix_hits += hit
        else:
            interval.append(transition)
        if _is_policy_invocation(after):
            if start is not None:
                epoch = _decision_epoch(
                    episode_id=episode_id,
                    index=decision_index,
                    start=start,
                    successor=after,
                    transitions=tuple(interval),
                    terminal=False,
                )
                decision_hits += epoch.hit_cost
                decision_index += 1
                yield epoch
            start = after
            interval = []

    yield from consume(first_before, first_transition, first_after)
    for before, transition, after in links:
        yield from consume(before, transition, after)

    if start is None:
        raise EpisodeDatasetError("episode contains no policy invocation")
    terminal_epoch = _decision_epoch(
        episode_id=episode_id,
        index=decision_index,
        start=start,
        successor=last_frame,
        transitions=tuple(interval),
        terminal=True,
    )
    decision_hits += terminal_epoch.hit_cost
    yield terminal_epoch

    try:
        declared_hits = int(outcome["physical_hits"])
    except (KeyError, TypeError, ValueError) as error:
        raise EpisodeDatasetError("episode outcome lacks physical HIT total") from error
    if bombs:
        raise EpisodeDatasetError("Bomb-bearing episode cannot enter a learner view")
    if infrastructure_failures:
        raise EpisodeDatasetError("infrastructure failure cannot enter a learner view")
    if prefix_hits:
        raise EpisodeDatasetError("physical HIT occurred before the first policy invocation")
    if raw_hits != declared_hits or decision_hits != raw_hits:
        raise EpisodeDatasetError(
            "decision epochs do not conserve the complete-episode HIT total"
        )


def validate_decision_epochs(run_dir: Path) -> dict[str, int]:
    """Exhaust and summarize the E2/L0 causal learner view."""
    decisions = 0
    eligible = 0
    hits = 0
    eligible_hits = 0
    for epoch in iter_decision_epochs(run_dir):
        decisions += 1
        eligible += int(epoch.learning_eligible)
        hits += epoch.hit_cost
        eligible_hits += epoch.hit_cost if epoch.learning_eligible else 0
    return {
        "decision_epochs": decisions,
        "learning_eligible_decision_epochs": eligible,
        "excluded_decision_epochs": decisions - eligible,
        "physical_hits": hits,
        "learning_eligible_physical_hits": eligible_hits,
        "excluded_physical_hits": hits - eligible_hits,
    }


def validate_episode(run_dir: Path) -> dict[str, int]:
    """Validate one complete algorithm-independent physical episode."""
    frames = sum(1 for _frame in iter_episode_frames(run_dir))
    transitions = sum(1 for _transition in iter_episode_transitions(run_dir))
    if transitions != frames - 1:
        raise EpisodeDatasetError("complete episode does not have N-1 transitions")
    return {"frames": frames, "transitions": transitions}
