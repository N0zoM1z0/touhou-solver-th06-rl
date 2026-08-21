"""Algorithm- and game-independent access to complete factual episodes."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

from .corpus import FRAME_SCHEMA, MANIFEST_SCHEMA, RUN_SCHEMA, TRANSITION_SCHEMA


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
    published_action: str | None
    commanded_action: str | None
    sampled_action: str | None
    executed_action: str | None
    behavior_probability: float
    behavior_probabilities: tuple[tuple[str, float], ...]
    outcome: dict[str, object]
    learning_eligible: bool
    learning_exclusion_reasons: tuple[str, ...]


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
) -> tuple[Path, dict[str, object], dict[str, object]]:
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
        or schemas.get("frame") != FRAME_SCHEMA
        or schemas.get("transition") != TRANSITION_SCHEMA
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
    return run_dir, manifest, metadata


def iter_episode_frames(run_dir: Path) -> Iterator[EpisodeFrame]:
    """Yield learner-facing facts without decoding game-specific raw state."""
    run_dir, manifest, metadata = _episode_contract(run_dir)
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
            sequence != expected_sequence
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


def iter_episode_transitions(run_dir: Path) -> Iterator[EpisodeTransition]:
    """Yield exactly one factual transition for each adjacent frame pair."""
    run_dir, manifest, _metadata = _episode_contract(run_dir)
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
        published = row.get("published_action")
        raw_probabilities = row.get("behavior_probabilities", ())
        probabilities = tuple(
            (str(item[0]), float(item[1]))
            for item in raw_probabilities
            if isinstance(item, (list, tuple)) and len(item) == 2
        )
        probability = float(row.get("behavior_probability", float("nan")))
        probability_map = dict(probabilities)
        if (
            row.get("schema_version") != TRANSITION_SCHEMA
            or int(row.get("sequence", -1)) != before.sequence
            or row.get("snapshot_ref") != before.snapshot_id
            or row.get("next_snapshot_ref") != after.snapshot_id
            or after.sequence != before.sequence + 1
            or not isinstance(outcome, dict)
            or len(set(legal)) != len(legal)
            or not math.isfinite(probability)
            or not 0.0 < probability <= 1.0
        ):
            raise EpisodeDatasetError("transition linkage or scalar contract is invalid")
        if published is not None and (
            set(probability_map) != set(legal)
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
        yield EpisodeTransition(
            count,
            before.snapshot_id,
            after.snapshot_id,
            legal,
            str(published) if published is not None else None,
            str(row["commanded_action"])
            if row.get("commanded_action") is not None else None,
            str(row["sampled_action"])
            if row.get("sampled_action") is not None else None,
            str(row["executed_action"])
            if row.get("executed_action") is not None else None,
            probability,
            probabilities,
            dict(outcome),
            bool(row.get("learning_eligible")),
            tuple(str(reason) for reason in row.get("learning_exclusion_reasons", ())),
        )
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


def validate_episode(run_dir: Path) -> dict[str, int]:
    """Validate one complete algorithm-independent physical episode."""
    frames = sum(1 for _frame in iter_episode_frames(run_dir))
    transitions = sum(1 for _transition in iter_episode_transitions(run_dir))
    if transitions != frames - 1:
        raise EpisodeDatasetError("complete episode does not have N-1 transitions")
    return {"frames": frames, "transitions": transitions}
