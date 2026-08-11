"""Lossless, compressed physical evidence independent of any learner."""

from __future__ import annotations

import gzip
import base64
import hashlib
import heapq
import json
import math
import os
from collections import Counter, defaultdict
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
MANIFEST_SCHEMA = "th06-rl-manifest-v2"
OBJECT_SCHEMA = "th06-rl-source-object-v1"
FRAME_SCHEMA = "th06-rl-authoritative-frame-v5"
TRANSITION_SCHEMA = "th06-rl-transition-v5"
EVENT_SCHEMA = "th06-rl-event-v1"
ANCHOR_SCHEMA = "th06-rl-authoritative-anchor-v1"
FRAME_BUDGET_MS = 1000.0 / 60.0
# Match one shard of burst tolerance.  A dense control root is bounded by 640
# raw 122-byte bullet-tail records, so 512 queued roots remain tens of MiB,
# while tolerating one transient UNC shard close/fsync without dropping a
# physical trajectory.
DEFAULT_SHARD_RECORDS = 512
DEFAULT_QUEUE_RECORDS = 512
DEFAULT_MAX_RUN_BYTES = 512 * 1024 * 1024
DEFAULT_SPOOL_MAX_RUN_BYTES = 4 * 1024 * 1024 * 1024

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
DATACLASS_ROWS_CODEC = "dataclass-rows-v1"
DATACLASS_RECORD_CODEC = "dataclass-record-v1"
BYTES_CODEC = "bytes-base64-v1"
_LAYOUTS: dict[type, tuple[str, ...]] = {}


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
    planner: dict[str, object]


@dataclass(frozen=True)
class DialogueDeliverySample:
    """Tiny retail input evidence retained while battle capture is paused."""

    game_frame: int
    current_input_mask: int
    previous_input_mask: int
    published_input_mask: int
    held_repeat: int
    held_frames: int
    active: bool
    skippable: bool
    pulsed_shoot: bool

    def __post_init__(self) -> None:
        allowed = 0x01 | 0x04 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100
        if self.game_frame < 0:
            raise ValueError("dialogue delivery frame must be nonnegative")
        for name in (
            "current_input_mask",
            "previous_input_mask",
            "published_input_mask",
        ):
            mask = int(getattr(self, name))
            if mask & BUTTON_BOMB or mask & ~allowed:
                raise ValueError(f"invalid or Bomb-bearing dialogue {name}")
        if not 0 <= self.held_repeat <= 0xFFFF or not 0 <= self.held_frames <= 0xFFFF:
            raise ValueError("dialogue delivery held counters must fit u16")


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
    capture_ms: float
    solve_ms: float
    reason: str
    capture_attempts: int = 1
    observation_gap: int = 1
    snapshot_tier: str = "authoritative-full"
    phase_elapsed_frames: int = 0
    dialogue_delivery: tuple[DialogueDeliverySample, ...] = ()

    def __post_init__(self) -> None:
        if not self.phase_id:
            raise ValueError("phase_id cannot be empty")
        if not 0.0 < self.behavior_probability <= 1.0:
            raise ValueError("behavior probability must be in (0, 1]")
        if self.published_action is not None and (
            self.published_action not in self.locally_admissible_actions
        ):
            raise ValueError("published action is outside the recorded local set")
        if any(
            right.game_frame < left.game_frame
            for left, right in zip(self.dialogue_delivery, self.dialogue_delivery[1:])
        ):
            raise ValueError("dialogue delivery samples must be frame ordered")


@dataclass(frozen=True)
class _Envelope:
    sequence: int
    snapshot_id: str
    snapshot: object
    evidence: FrameEvidence
    scope: dict[str, object]


@dataclass(frozen=True)
class _AnchorEnvelope:
    sequence: int
    snapshot: object
    phase_id: str
    reason: str
    control_snapshot_ref: str | None


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _layout(value: object) -> tuple[str, ...]:
    value_type = type(value)
    layout = _LAYOUTS.get(value_type)
    if layout is None:
        layout = tuple(field.name for field in fields(value))
        _LAYOUTS[value_type] = layout
    return layout


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


class _HashingWriter:
    """Hash compressed bytes as gzip writes them to the UNC-backed file."""

    def __init__(self, raw) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()

    def write(self, payload: bytes) -> int:
        self.digest.update(payload)
        return self.raw.write(payload)

    def flush(self) -> None:
        self.raw.flush()

    def tell(self) -> int:
        return self.raw.tell()


class _ShardWriter:
    def __init__(
        self,
        run_dir: Path,
        stream: str,
        records: int,
        on_shard,
        *,
        compresslevel: int,
    ) -> None:
        self.run_dir = run_dir
        self.stream = stream
        self.records = records
        self.on_shard = on_shard
        self.compresslevel = compresslevel
        self.index = 0
        self.count = 0
        self.first_sequence: int | None = None
        self.last_sequence: int | None = None
        self.uncompressed_bytes = 0
        self.raw = None
        self.hashing = None
        self.gzip = None
        self.temporary: Path | None = None

    def _open(self) -> None:
        if self.raw is not None:
            return
        self.temporary = self.run_dir / f".{self.stream}-{self.index:06d}.partial"
        self.raw = self.temporary.open("wb")
        self.hashing = _HashingWriter(self.raw)
        self.gzip = gzip.GzipFile(
            fileobj=self.hashing,
            mode="wb",
            compresslevel=self.compresslevel,
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
        # The digest was accumulated over the exact compressed stream. Avoid
        # rereading a dense shard through WSL UNC merely to name it.
        digest = self.hashing.digest.hexdigest()
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
        self.hashing = None
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


def _encode_dataclass(
    value,
    objects: _ObjectStore,
    kind: str,
    *,
    object_fields: tuple[str, ...] = (),
) -> dict[str, object]:
    layout = _layout(value)
    encoded = []
    for name in layout:
        field_value = getattr(value, name)
        encoded.append(
            objects.reference(f"{kind}.{name}", field_value)
            if name in object_fields
            else _encode_value(field_value, objects, f"{kind}.{name}")
        )
    return {
        "codec": DATACLASS_RECORD_CODEC,
        "layout": objects.reference(f"layout.{kind}", layout),
        "values": encoded,
    }


def _encode_value(value: object, objects: _ObjectStore, kind: str) -> object:
    if isinstance(value, bytes):
        return {
            "codec": BYTES_CODEC,
            "data": base64.b64encode(value).decode("ascii"),
        }
    if is_dataclass(value):
        return _encode_dataclass(value, objects, kind)
    if isinstance(value, (tuple, list)):
        if value and all(
            is_dataclass(item) and type(item) is type(value[0])
            for item in value
        ):
            layout = _layout(value[0])
            return {
                "codec": DATACLASS_ROWS_CODEC,
                "layout": objects.reference(f"layout.{kind}", layout),
                "rows": [
                    [
                        _encode_value(
                            getattr(item, name),
                            objects,
                            f"{kind}.{name}",
                        )
                        for name in layout
                    ]
                    for item in value
                ],
            }
        return [
            _encode_value(item, objects, f"{kind}[]") for item in value
        ]
    if isinstance(value, dict):
        return {
            str(key): _encode_value(item, objects, f"{kind}.{key}")
            for key, item in value.items()
        }
    return value


def expand_compact(value: object, objects: dict[str, object]) -> object:
    """Hydrate content references and lossless dataclass row codecs."""
    if isinstance(value, dict):
        if set(value) == {"kind", "object_ref"}:
            return expand_compact(objects[str(value["object_ref"])], objects)
        codec = value.get("codec")
        if codec == BYTES_CODEC:
            data = value.get("data")
            if not isinstance(data, str):
                raise CorpusError("invalid compact bytes payload")
            try:
                return base64.b64decode(data, validate=True)
            except ValueError as error:
                raise CorpusError("invalid compact bytes encoding") from error
        if codec in (DATACLASS_RECORD_CODEC, DATACLASS_ROWS_CODEC):
            layout = expand_compact(value["layout"], objects)
            if not isinstance(layout, list) or not all(
                isinstance(name, str) for name in layout
            ):
                raise CorpusError("invalid compact dataclass layout")
            if codec == DATACLASS_RECORD_CODEC:
                values = expand_compact(value["values"], objects)
                if not isinstance(values, list) or len(values) != len(layout):
                    raise CorpusError("invalid compact dataclass record")
                return dict(zip(layout, values, strict=True))
            rows = expand_compact(value["rows"], objects)
            if not isinstance(rows, list):
                raise CorpusError("invalid compact dataclass rows")
            result = []
            for row in rows:
                if not isinstance(row, list) or len(row) != len(layout):
                    raise CorpusError("invalid compact dataclass row")
                result.append(dict(zip(layout, row, strict=True)))
            return result
        return {
            key: expand_compact(item, objects) for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand_compact(item, objects) for item in value]
    return value


def _serialize_snapshot(snapshot, objects: _ObjectStore) -> dict[str, object]:
    result = {}
    for field in fields(snapshot):
        value = getattr(snapshot, field.name)
        if (
            field.name == "bullets"
            and getattr(snapshot, "capture_tier", None) == "control-v2"
            and getattr(snapshot, "raw_bullet_tails", b"")
        ):
            # The packed source tail for every occupied slot plus the compact
            # pointer->visual-size table reconstructs the complete Bullet
            # tuple.  Re-encoding the resident reachable subset as hundreds
            # of JSON dataclass rows duplicates information and made the
            # asynchronous writer slower than the 60 Hz producer.
            result[field.name] = []
        elif field.name in SOURCE_OBJECT_FIELDS:
            result[field.name] = objects.reference(f"snapshot.{field.name}", value)
        elif field.name == "spawners":
            if value:
                layout = _layout(value[0])
                result[field.name] = {
                    "codec": DATACLASS_ROWS_CODEC,
                    "layout": objects.reference("layout.enemy-spawner", layout),
                    "rows": [
                        [
                            (
                                objects.reference(
                                    f"enemy-spawner.{name}",
                                    getattr(item, name),
                                )
                                if name in SPAWNER_SOURCE_OBJECT_FIELDS
                                else _encode_value(
                                    getattr(item, name),
                                    objects,
                                    f"enemy-spawner.{name}",
                                )
                            )
                            for name in layout
                        ]
                        for item in value
                    ],
                }
            else:
                result[field.name] = []
        else:
            result[field.name] = _encode_value(
                value,
                objects,
                f"snapshot.{field.name}",
            )
    return result


def _scope(snapshot, phase_id: str) -> dict[str, object]:
    shot_type = getattr(snapshot, "shot_type", None)
    if shot_type is None:
        attack = getattr(snapshot, "player_attack", None)
        shot_type = attack.shot_type if attack else -1
    values = {
        "difficulty": snapshot.difficulty,
        "character": snapshot.character,
        "shot_type": shot_type,
        "stage": snapshot.stage,
        "phase_id": phase_id,
    }
    return {"key": "/".join(str(values[key]) for key in values), **values}


def _boss_life(snapshot) -> int | None:
    direct = getattr(snapshot, "boss_life", None)
    if direct is not None:
        return int(direct)
    bosses = sorted(
        (
            item
            for item in getattr(snapshot, "spawners", ())
            if item.is_boss
        ),
        key=lambda item: (item.boss_id, item.slot),
    )
    return bosses[0].life if bosses else None


def _frame_from_snapshot_ref(value: object) -> int | None:
    marker = str(value).rsplit(":f", 1)
    if len(marker) != 2:
        return None
    try:
        return int(marker[1])
    except ValueError:
        return None


def _transition(before: _Envelope, after: _Envelope) -> dict[str, object]:
    enable_donor_imports()
    from th06.trial import physical_hit

    hit = physical_hit(before.snapshot.player_state, after.snapshot.player_state)
    bomb = bool(
        (before.snapshot.input_mask | after.snapshot.input_mask) & BUTTON_BOMB
        or (
            getattr(after.snapshot, "bomb_active", False)
            or (
                getattr(after.snapshot, "player_attack", None) is not None
                and after.snapshot.player_attack.bomb_active
            )
        )
    )
    control_dead_end = after.evidence.reason in (
        "control-dead-end:Hard safe set empty",
        "control-dead-end:local forecast has no safe continuation",
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
        "capture_ms_before": before.evidence.capture_ms,
        "capture_ms_after": after.evidence.capture_ms,
        "capture_attempts_before": before.evidence.capture_attempts,
        "capture_attempts_after": after.evidence.capture_attempts,
        "observation_gap": max(0, after.snapshot.frame - before.snapshot.frame),
    }
    failure = (
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
    learning_exclusions = []
    if before.evidence.published_action is None:
        learning_exclusions.append("action-not-published")
    if before.evidence.reason not in ("ok", "input-lease"):
        learning_exclusions.append(f"decision:{before.evidence.reason}")
    if outcome["elapsed_frames"] != 1:
        learning_exclusions.append("observation-gap")
    if before.evidence.capture_ms > FRAME_BUDGET_MS:
        learning_exclusions.append("capture-over-frame-budget")
    if bomb:
        learning_exclusions.append("bomb")
    if authority:
        learning_exclusions.append("authority-loss")
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
        # This compact projection is sufficient to reconstruct the exact
        # online UCB key without decoding the large raw hazard root. Raw
        # snapshots remain the learner-independent authority evidence.
        "policy_context": {
            "current_action": before.evidence.current_action,
            "hard_admissible_actions": [
                str(item[0]) for item in before.evidence.hard_actions
            ],
            "phase_elapsed_frames": before.evidence.phase_elapsed_frames,
            "player_x": before.snapshot.x,
            "player_y": before.snapshot.y,
            "power": before.snapshot.current_power,
            "bullet_count": before.snapshot.live_bullet_count,
            "laser_count": before.snapshot.laser_count,
            "hard_action_count": len(before.evidence.hard_actions),
        },
        "outcome_terms": outcome,
        "learning_eligible": not learning_exclusions,
        "learning_exclusion_reasons": learning_exclusions,
        # A patched HIT is a failure observation, not the end of the physical
        # Practice Stage.  Keep the stage episode, source option boundary and
        # failure signal independent so an offline trainer may choose its own
        # bootstrapping semantics without guessing what legacy `done` meant.
        "episode": {
            "id": before.snapshot_id.split(":", 1)[0],
            "unit": "practice-stage",
            "step": before.sequence,
            "done": False,
        },
        "boundary": {
            "source_context_changed": outcome["phase_changed"],
            "source_context": before.scope["key"],
            "next_source_context": after.scope["key"],
            "failure": failure,
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
        deferred_compression: bool = False,
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
        self.archive_max_run_bytes = max_run_bytes
        self.max_run_bytes = (
            DEFAULT_SPOOL_MAX_RUN_BYTES if deferred_compression else max_run_bytes
        )
        self.compresslevel = 0 if deferred_compression else 3
        self.sequence = 0
        self.anchor_sequence = 0
        self.enqueued = 0
        self.written = 0
        self.dropped = 0
        self.queue_high_watermark = 0
        self.closed = False
        self.error: BaseException | None = None
        self.capture_timings: list[float] = []
        self.solve_timings: list[float] = []
        self.reason_counts: Counter[str] = Counter()
        self.phase_metrics = defaultdict(lambda: {
            "frames": 0,
            "elapsed_frames": 0,
            "hits": 0,
            "control_dead_ends": 0,
            "learning_eligible_transitions": 0,
            "learning_eligible_elapsed_frames": 0,
            "hard_sum": 0,
            "published_actions": Counter(),
            "legal_opportunities": Counter(),
        })
        self.first_frame: int | None = None
        self.last_frame: int | None = None
        self.hit_frames: list[int] = []
        self.dense_frames: list[tuple[int, int, int]] = []
        self.observation_gap_frames = 0
        self.over_budget_capture_frames = 0
        self.queue: queue.Queue[_Envelope | _AnchorEnvelope | None] = queue.Queue(
            queue_records
        )
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
            "max_run_bytes": self.max_run_bytes,
            "archive_max_run_bytes": self.archive_max_run_bytes,
            "storage_compression": f"gzip-{self.compresslevel}",
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
                "anchor": ANCHOR_SCHEMA,
            },
            "storage": {
                "compression": f"gzip-{self.compresslevel}",
                "source_objects": "sha256-content-addressed",
                "repeated_dataclasses": DATACLASS_ROWS_CODEC,
                "control_bullets": "raw-tail-plus-visual-map-v1",
                "frame_policy": "lossless-no-drop",
                "queue_records": queue_records,
            },
        })
        _atomic_json(self.run_dir / "manifest.json", self.manifest)
        self.thread = threading.Thread(target=self._worker, name=f"corpus-{self.run_id}")
        self.thread.start()

    @staticmethod
    def _timing_summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {name: None for name in ("p50_ms", "p95_ms", "p99_ms", "max_ms")}
        ordered = sorted(values)

        def percentile(fraction: float) -> float:
            return ordered[
                min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
            ]

        return {
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
            "max_ms": ordered[-1],
        }

    def _observe_frame_metrics(self, envelope: _Envelope) -> None:
        evidence = envelope.evidence
        self.capture_timings.append(float(evidence.capture_ms))
        self.solve_timings.append(float(evidence.solve_ms))
        self.reason_counts[evidence.reason] += 1
        self.observation_gap_frames += evidence.observation_gap != 1
        self.over_budget_capture_frames += evidence.capture_ms > FRAME_BUDGET_MS
        frame = int(envelope.snapshot.frame)
        bullet_count = int(
            getattr(
                envelope.snapshot,
                "live_bullet_count",
                len(envelope.snapshot.bullets),
            )
        )
        dense_entry = (bullet_count, envelope.sequence, frame)
        if len(self.dense_frames) < 64:
            heapq.heappush(self.dense_frames, dense_entry)
        else:
            heapq.heappushpop(self.dense_frames, dense_entry)
        self.first_frame = frame if self.first_frame is None else min(self.first_frame, frame)
        self.last_frame = frame if self.last_frame is None else max(self.last_frame, frame)
        phase = self.phase_metrics[str(envelope.scope["key"])]
        phase["frames"] += 1
        phase["hard_sum"] += len(evidence.hard_actions)
        if evidence.published_action is not None:
            phase["published_actions"][evidence.published_action] += 1
        phase["legal_opportunities"].update(evidence.locally_admissible_actions)

    def _observe_transition_metrics(self, transition: dict[str, object]) -> None:
        phase = self.phase_metrics[str(transition["scope"]["key"])]
        outcome = transition["outcome_terms"]
        phase["elapsed_frames"] += max(0, int(outcome["elapsed_frames"]))
        phase["learning_eligible_transitions"] += bool(
            transition["learning_eligible"]
        )
        if transition["learning_eligible"]:
            phase["learning_eligible_elapsed_frames"] += max(
                0, int(outcome["elapsed_frames"])
            )
        if outcome["life_lost"]:
            phase["hits"] += 1
            frame = _frame_from_snapshot_ref(transition["next_snapshot_ref"])
            if frame is not None:
                self.hit_frames.append(frame)
        if outcome["control_dead_end"]:
            phase["control_dead_ends"] += 1

    def _metrics_summary(self) -> dict[str, object]:
        frames = self.written
        compressed = int(self.manifest["compressed_bytes"])
        longest = None
        if self.first_frame is not None and self.last_frame is not None:
            boundaries = [
                self.first_frame,
                *sorted(self.hit_frames),
                self.last_frame,
            ]
            longest = max(
                (right - left for left, right in zip(boundaries, boundaries[1:])),
                default=0,
            )
        phases = []
        for key, raw in self.phase_metrics.items():
            count = int(raw["frames"])
            phases.append({
                "scope": key,
                "frames": count,
                "elapsed_frames": int(raw["elapsed_frames"]),
                "hits": int(raw["hits"]),
                "control_dead_ends": int(raw["control_dead_ends"]),
                "learning_eligible_transitions": int(
                    raw["learning_eligible_transitions"]
                ),
                "learning_eligible_elapsed_frames": int(
                    raw["learning_eligible_elapsed_frames"]
                ),
                "mean_hard_actions": raw["hard_sum"] / count if count else None,
                "published_actions": dict(raw["published_actions"].most_common()),
                "legal_opportunities": dict(
                    raw["legal_opportunities"].most_common()
                ),
            })
        phases.sort(key=lambda item: (-item["hits"], -item["elapsed_frames"], item["scope"]))
        return {
            "frames": frames,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "hit_frames": self.hit_frames,
            "longest_observed_no_hit_frames": longest,
            "capture_timing": self._timing_summary(self.capture_timings),
            "solve_timing": self._timing_summary(self.solve_timings),
            "reason_counts": dict(self.reason_counts.most_common()),
            "stale_retry_rate": (
                self.reason_counts["stale-retry"] / frames if frames else None
            ),
            "observation_gap_rate": (
                self.observation_gap_frames / frames if frames else None
            ),
            "capture_over_frame_budget_rate": (
                self.over_budget_capture_frames / frames if frames else None
            ),
            "learning_eligible_elapsed_frames": sum(
                int(raw["learning_eligible_elapsed_frames"])
                for raw in self.phase_metrics.values()
            ),
            "compressed_bytes_per_frame": compressed / frames if frames else None,
            "dense_frame_samples": [
                {"bullets": bullets, "sequence": sequence, "frame": frame}
                for bullets, sequence, frame in sorted(
                    self.dense_frames, reverse=True
                )
            ],
            "phases": phases,
        }

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
        self.queue_high_watermark = max(
            self.queue_high_watermark,
            self.queue.qsize(),
        )
        return snapshot_id

    def record_anchor(
        self,
        snapshot,
        *,
        phase_id: str,
        reason: str,
        control_snapshot_ref: str | None,
    ) -> None:
        """Queue one exhaustive source root outside the decision hot path."""
        if self.closed:
            raise CorpusError("corpus recorder is closed")
        self._raise_error()
        if snapshot.input_mask & BUTTON_BOMB:
            raise CorpusError("Bomb bit observed in corpus anchor")
        envelope = _AnchorEnvelope(
            self.anchor_sequence,
            snapshot,
            phase_id,
            reason,
            control_snapshot_ref,
        )
        try:
            self.queue.put_nowait(envelope)
        except queue.Full as error:
            self.dropped += 1
            raise CorpusBackpressure(
                "corpus queue full while retaining source anchor"
            ) from error
        self.anchor_sequence += 1
        self.queue_high_watermark = max(
            self.queue_high_watermark,
            self.queue.qsize(),
        )

    def _worker(self) -> None:
        writers = {
            stream: _ShardWriter(
                self.run_dir,
                stream,
                self.shard_records,
                self._on_shard,
                compresslevel=self.compresslevel,
            )
            for stream in ("objects", "frames", "transitions", "events", "anchors")
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
                    if isinstance(envelope, _AnchorEnvelope):
                        writers["anchors"].write({
                            "schema_version": ANCHOR_SCHEMA,
                            "sequence": envelope.sequence,
                            "frame": envelope.snapshot.frame,
                            "scope": _scope(envelope.snapshot, envelope.phase_id),
                            "reason": envelope.reason,
                            "control_snapshot_ref": envelope.control_snapshot_ref,
                            "snapshot": _serialize_snapshot(
                                envelope.snapshot, objects
                            ),
                        }, envelope.sequence)
                        continue
                    writers["frames"].write({
                        "schema_version": FRAME_SCHEMA,
                        "sequence": envelope.sequence,
                        "snapshot_id": envelope.snapshot_id,
                        "scope": envelope.scope,
                        "snapshot": _serialize_snapshot(envelope.snapshot, objects),
                        "decision": _jsonable(envelope.evidence),
                    }, envelope.sequence)
                    self._observe_frame_metrics(envelope)
                    if previous is not None:
                        transition = _transition(previous, envelope)
                        writers["transitions"].write(transition, previous.sequence)
                        self._observe_transition_metrics(transition)
                        if transition["boundary"]["failure"] is not None:
                            writers["events"].write({
                                "schema_version": EVENT_SCHEMA,
                                "event": transition["boundary"]["failure"],
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

    def close(self, run_outcome: dict[str, object] | None = None) -> Path:
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
            stage_complete = bool(
                self.dropped == 0
                and run_outcome is not None
                and run_outcome.get("stage_completed") is True
            )
            self.manifest.update({
                "complete": self.dropped == 0,
                "stage_trajectory_complete": stage_complete,
                "episode": {
                    "id": self.run_id,
                    "unit": "practice-stage",
                    "complete": stage_complete,
                    "termination_reason": (
                        run_outcome.get("termination_reason")
                        if run_outcome is not None
                        else None
                    ),
                },
                "dropped_records": self.dropped,
                "enqueued_frames": self.enqueued,
                "written_frames": self.written,
                "queue_high_watermark": self.queue_high_watermark,
                "queue_capacity": self.queue.maxsize,
                "closed_unix_ns": time.time_ns(),
                "run_outcome": run_outcome,
                "summary": self._metrics_summary(),
            })
            _atomic_json(self.run_dir / "manifest.json", self.manifest)
        return self.run_dir
