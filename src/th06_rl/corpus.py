"""Lossless, compressed physical evidence independent of any learner."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
import queue
import tempfile
import threading
import time

from .th06.donor import enable_donor_imports

enable_donor_imports()
from th06.model import BUTTON_BOMB  # noqa: E402


RUN_SCHEMA = "th06-rl-run-v1"
MANIFEST_SCHEMA = "th06-rl-manifest-v1"
OBJECT_SCHEMA = "th06-rl-source-object-v1"
FRAME_SCHEMA = "th06-rl-authoritative-frame-v1"
TRANSITION_SCHEMA = "th06-rl-transition-v2"
EVENT_SCHEMA = "th06-rl-event-v1"
DEFAULT_SHARD_RECORDS = 128
DEFAULT_QUEUE_RECORDS = 512
DEFAULT_MAX_RUN_BYTES = 512 * 1024 * 1024

SOURCE_OBJECT_FIELDS = (
    "timeline_instructions",
    "timeline_ecl_program",
    "ecl_subroutines",
    "bullet_sizes",
    "timeline_emitter_subs",
    "timeline_boss_subs",
    "timeline_message_delays",
)
SPAWNER_SOURCE_OBJECT_FIELDS = ("ecl_program", "ecl_subroutines")


class CorpusError(RuntimeError):
    pass


class CorpusBackpressure(CorpusError):
    pass


class CorpusStorageLimit(CorpusError):
    pass


@dataclass(frozen=True)
class RunMetadata:
    code_commit: str
    executable_sha256: str
    native_kernel_sha256: str
    input_backend: str
    difficulty: int
    character: int
    shot_type: int
    stage: int
    planner: dict[str, int | float]


@dataclass(frozen=True)
class FrameEvidence:
    phase_id: str
    current_action: str | None
    # None means source-known unbounded clearance (native +infinity).
    hard_actions: tuple[tuple[str, float | None, float, float], ...]
    baseline_action: str | None
    locally_admissible_actions: tuple[str, ...]
    proposed_action: str | None
    published_action: str | None
    behavior_probability: float
    policy_id: str | None
    policy_generation: int
    policy_sha256: str | None
    effort_horizon: int
    plan_min_clearance: float | None
    cumulative_risk: float | None
    terminal_x: float | None
    terminal_y: float | None
    endpoint_count: int
    continuation_action_count: int
    solve_ms: float
    reason: str

    def __post_init__(self) -> None:
        if not self.phase_id:
            raise ValueError("phase_id cannot be empty")
        if not 0.0 < self.behavior_probability <= 1.0:
            raise ValueError("behavior probability must be in (0, 1]")
        if self.published_action is not None and (
            self.published_action not in self.locally_admissible_actions
        ):
            raise ValueError("published action is outside the recorded local set")


@dataclass(frozen=True)
class _Envelope:
    sequence: int
    snapshot_id: str
    snapshot: object
    evidence: FrameEvidence
    scope: dict[str, object]


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_canonical(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _ShardWriter:
    def __init__(self, run_dir: Path, stream: str, records: int, on_shard) -> None:
        self.run_dir = run_dir
        self.stream = stream
        self.records = records
        self.on_shard = on_shard
        self.index = 0
        self.count = 0
        self.first_sequence: int | None = None
        self.last_sequence: int | None = None
        self.uncompressed_bytes = 0
        self.raw = None
        self.gzip = None
        self.temporary: Path | None = None

    def _open(self) -> None:
        if self.raw is not None:
            return
        self.temporary = self.run_dir / f".{self.stream}-{self.index:06d}.partial"
        self.raw = self.temporary.open("wb")
        self.gzip = gzip.GzipFile(
            fileobj=self.raw,
            mode="wb",
            compresslevel=3,
            mtime=0,
            filename="",
        )

    def write(self, record: dict[str, object], sequence: int) -> None:
        self._open()
        row = _canonical(record) + b"\n"
        self.gzip.write(row)
        self.uncompressed_bytes += len(row)
        self.count += 1
        self.first_sequence = sequence if self.first_sequence is None else self.first_sequence
        self.last_sequence = sequence
        if self.count >= self.records:
            self.close_shard()

    def close_shard(self) -> None:
        if self.raw is None:
            return
        self.gzip.close()
        self.raw.close()
        digest = hashlib.sha256(self.temporary.read_bytes()).hexdigest()
        final = self.run_dir / f"{self.stream}-{self.index:06d}-{digest[:16]}.jsonl.gz"
        self.temporary.replace(final)
        self.on_shard({
            "stream": self.stream,
            "path": final.name,
            "sha256": digest,
            "records": self.count,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": final.stat().st_size,
        })
        self.index += 1
        self.count = 0
        self.first_sequence = None
        self.last_sequence = None
        self.uncompressed_bytes = 0
        self.raw = None
        self.gzip = None
        self.temporary = None

    def close(self) -> None:
        self.close_shard()


class _ObjectStore:
    def __init__(self, writer: _ShardWriter) -> None:
        self.writer = writer
        self.seen: set[str] = set()
        self.cache: dict[tuple[str, int, int], tuple[object, str]] = {}
        self.sequence = 0

    def reference(self, kind: str, value: object) -> dict[str, str]:
        length = len(value) if isinstance(value, tuple) else -1
        key = (kind, id(value), length)
        cached = self.cache.get(key)
        if cached is not None and cached[0] is value:
            return {"kind": kind, "object_ref": cached[1]}
        payload = _jsonable(value)
        object_id = _content_id({"kind": kind, "payload": payload})
        self.cache[key] = (value, object_id)
        if object_id not in self.seen:
            self.seen.add(object_id)
            self.writer.write({
                "schema_version": OBJECT_SCHEMA,
                "object_id": object_id,
                "kind": kind,
                "payload": payload,
            }, self.sequence)
            self.sequence += 1
        return {"kind": kind, "object_ref": object_id}


def _serialize_spawner(spawner, objects: _ObjectStore) -> dict[str, object]:
    result = {}
    for field in fields(spawner):
        value = getattr(spawner, field.name)
        result[field.name] = (
            objects.reference(f"enemy-spawner.{field.name}", value)
            if field.name in SPAWNER_SOURCE_OBJECT_FIELDS
            else _jsonable(value)
        )
    return result


def _serialize_snapshot(snapshot, objects: _ObjectStore) -> dict[str, object]:
    result = {}
    for field in fields(snapshot):
        value = getattr(snapshot, field.name)
        if field.name in SOURCE_OBJECT_FIELDS:
            result[field.name] = objects.reference(f"snapshot.{field.name}", value)
        elif field.name == "spawners":
            result[field.name] = [_serialize_spawner(item, objects) for item in value]
        else:
            result[field.name] = _jsonable(value)
    return result


def _scope(snapshot, phase_id: str) -> dict[str, object]:
    shot_type = snapshot.player_attack.shot_type if snapshot.player_attack else -1
    values = {
        "difficulty": snapshot.difficulty,
        "character": snapshot.character,
        "shot_type": shot_type,
        "stage": snapshot.stage,
        "phase_id": phase_id,
    }
    return {"key": "/".join(str(values[key]) for key in values), **values}


def _boss_life(snapshot) -> int | None:
    bosses = sorted(
        (item for item in snapshot.spawners if item.is_boss),
        key=lambda item: (item.boss_id, item.slot),
    )
    return bosses[0].life if bosses else None


def _transition(before: _Envelope, after: _Envelope) -> dict[str, object]:
    enable_donor_imports()
    from th06.trial import physical_hit

    hit = physical_hit(before.snapshot.player_state, after.snapshot.player_state)
    bomb = bool(
        (before.snapshot.input_mask | after.snapshot.input_mask) & BUTTON_BOMB
        or (
            after.snapshot.player_attack is not None
            and after.snapshot.player_attack.bomb_active
        )
    )
    control_dead_end = after.evidence.reason in (
        "authority-stop:Hard safe set empty",
        "authority-stop:local forecast has no safe continuation",
    )
    authority = (
        after.evidence.reason.startswith("authority-stop:")
        and not control_dead_end
        and after.evidence.reason
        not in (
            "authority-stop:physical HIT",
            "authority-stop:physical Bomb state/input",
        )
    )
    before_life = _boss_life(before.snapshot)
    after_life = _boss_life(after.snapshot)
    outcome = {
        "life_lost": hit,
        "bomb_used": bomb,
        "control_dead_end": control_dead_end,
        "authority_lost": authority,
        "elapsed_frames": max(0, after.snapshot.frame - before.snapshot.frame),
        "lives_delta": after.snapshot.lives_remaining - before.snapshot.lives_remaining,
        "power_delta": after.snapshot.current_power - before.snapshot.current_power,
        "rank_delta": after.snapshot.rank - before.snapshot.rank,
        "boss_life_before": before_life,
        "boss_life_after": after_life,
        "player_x_before": before.snapshot.x,
        "player_y_before": before.snapshot.y,
        "player_x_after": after.snapshot.x,
        "player_y_after": after.snapshot.y,
        "hard_count_before": len(before.evidence.hard_actions),
        "hard_count_after": len(after.evidence.hard_actions),
        "phase_changed": before.scope["key"] != after.scope["key"],
    }
    terminal_reason = (
        "life-lost"
        if hit
        else "bomb-used"
        if bomb
        else "control-dead-end"
        if control_dead_end
        else "authority-lost"
        if authority
        else None
    )
    return {
        "schema_version": TRANSITION_SCHEMA,
        "sequence": before.sequence,
        "snapshot_ref": before.snapshot_id,
        "next_snapshot_ref": after.snapshot_id,
        "scope": before.scope,
        "next_scope": after.scope,
        "legal_actions": list(before.evidence.locally_admissible_actions),
        "baseline_action": before.evidence.baseline_action,
        "proposed_action": before.evidence.proposed_action,
        "published_action": before.evidence.published_action,
        "behavior_probability": before.evidence.behavior_probability,
        "outcome_terms": outcome,
        "learning_eligible": bool(
            before.evidence.published_action is not None
            and before.evidence.reason in ("ok", "input-lease")
            and not bomb
            and not control_dead_end
            and not authority
        ),
        "terminal": {
            "done": terminal_reason is not None,
            "reason": terminal_reason,
            "boundary_reason": (
                "phase-transition"
                if outcome["phase_changed"] and terminal_reason is None
                else None
            ),
        },
    }


class CorpusRecorder:
    """Asynchronous gzip shards; loss or storage overflow fails the run."""

    def __init__(
        self,
        root: Path,
        metadata: RunMetadata,
        *,
        run_id: str | None = None,
        shard_records: int = DEFAULT_SHARD_RECORDS,
        queue_records: int = DEFAULT_QUEUE_RECORDS,
        max_run_bytes: int = DEFAULT_MAX_RUN_BYTES,
    ) -> None:
        if min(shard_records, queue_records, max_run_bytes) <= 0:
            raise ValueError("corpus bounds must be positive")
        created_ns = time.time_ns()
        self.run_id = run_id or (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + f"-{created_ns % 1_000_000_000:09d}"
        )
        self.run_dir = Path(root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.shard_records = shard_records
        self.max_run_bytes = max_run_bytes
        self.sequence = 0
        self.enqueued = 0
        self.written = 0
        self.dropped = 0
        self.closed = False
        self.error: BaseException | None = None
        self.queue: queue.Queue[_Envelope | None] = queue.Queue(queue_records)
        self.manifest_lock = threading.Lock()
        self.manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA,
            "run_id": self.run_id,
            "complete": False,
            "shards": [],
            "records": {},
            "compressed_bytes": 0,
            "uncompressed_bytes": 0,
            "dropped_records": 0,
            "max_run_bytes": max_run_bytes,
        }
        _atomic_json(self.run_dir / "run.json", {
            "schema_version": RUN_SCHEMA,
            "run_id": self.run_id,
            "created_unix_ns": created_ns,
            "metadata": asdict(metadata),
            "schemas": {
                "object": OBJECT_SCHEMA,
                "frame": FRAME_SCHEMA,
                "transition": TRANSITION_SCHEMA,
                "event": EVENT_SCHEMA,
            },
            "storage": {
                "compression": "gzip-3",
                "source_objects": "sha256-content-addressed",
                "frame_policy": "lossless-no-drop",
            },
        })
        _atomic_json(self.run_dir / "manifest.json", self.manifest)
        self.thread = threading.Thread(target=self._worker, name=f"corpus-{self.run_id}")
        self.thread.start()

    def _on_shard(self, shard: dict[str, object]) -> None:
        with self.manifest_lock:
            projected = int(self.manifest["compressed_bytes"]) + int(
                shard["compressed_bytes"]
            )
            if projected > self.max_run_bytes:
                raise CorpusStorageLimit(
                    f"corpus exceeded {self.max_run_bytes} compressed bytes"
                )
            self.manifest["compressed_bytes"] = projected
            self.manifest["uncompressed_bytes"] = int(
                self.manifest["uncompressed_bytes"]
            ) + int(shard["uncompressed_bytes"])
            self.manifest["shards"].append(shard)
            records = self.manifest["records"]
            records[shard["stream"]] = int(records.get(shard["stream"], 0)) + int(
                shard["records"]
            )
            _atomic_json(self.run_dir / "manifest.json", self.manifest)

    def _raise_error(self) -> None:
        if self.error is not None:
            raise CorpusError(f"corpus worker failed: {self.error}") from self.error

    def record(self, snapshot, evidence: FrameEvidence) -> str:
        if self.closed:
            raise CorpusError("corpus recorder is closed")
        self._raise_error()
        if snapshot.input_mask & BUTTON_BOMB:
            raise CorpusError("Bomb bit observed in corpus root")
        snapshot_id = f"{self.run_id}:{self.sequence:08d}:f{snapshot.frame}"
        envelope = _Envelope(
            self.sequence,
            snapshot_id,
            snapshot,
            evidence,
            _scope(snapshot, evidence.phase_id),
        )
        try:
            self.queue.put_nowait(envelope)
        except queue.Full as error:
            self.dropped += 1
            raise CorpusBackpressure("corpus queue full; refusing partial evidence") from error
        self.sequence += 1
        self.enqueued += 1
        return snapshot_id

    def _worker(self) -> None:
        writers = {
            stream: _ShardWriter(self.run_dir, stream, self.shard_records, self._on_shard)
            for stream in ("objects", "frames", "transitions", "events")
        }
        objects = _ObjectStore(writers["objects"])
        previous: _Envelope | None = None
        try:
            while True:
                envelope = self.queue.get()
                if envelope is None:
                    self.queue.task_done()
                    break
                try:
                    writers["frames"].write({
                        "schema_version": FRAME_SCHEMA,
                        "sequence": envelope.sequence,
                        "snapshot_id": envelope.snapshot_id,
                        "scope": envelope.scope,
                        "snapshot": _serialize_snapshot(envelope.snapshot, objects),
                        "decision": _jsonable(envelope.evidence),
                    }, envelope.sequence)
                    if previous is not None:
                        transition = _transition(previous, envelope)
                        writers["transitions"].write(transition, previous.sequence)
                        if transition["terminal"]["done"]:
                            writers["events"].write({
                                "schema_version": EVENT_SCHEMA,
                                "event": transition["terminal"]["reason"],
                                "sequence": previous.sequence,
                                "snapshot_ref": envelope.snapshot_id,
                                "scope": previous.scope,
                            }, previous.sequence)
                    previous = envelope
                    self.written += 1
                finally:
                    self.queue.task_done()
            if previous is not None:
                writers["events"].write({
                    "schema_version": EVENT_SCHEMA,
                    "event": "run-end",
                    "sequence": previous.sequence,
                    "snapshot_ref": previous.snapshot_id,
                    "scope": previous.scope,
                }, previous.sequence)
            for writer in writers.values():
                writer.close()
        except BaseException as error:
            self.error = error

    def close(self) -> Path:
        if self.closed:
            self._raise_error()
            return self.run_dir
        self.closed = True
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        self._raise_error()
        with self.manifest_lock:
            self.manifest.update({
                "complete": self.dropped == 0,
                "dropped_records": self.dropped,
                "enqueued_frames": self.enqueued,
                "written_frames": self.written,
                "closed_unix_ns": time.time_ns(),
            })
            _atomic_json(self.run_dir / "manifest.json", self.manifest)
        return self.run_dir
