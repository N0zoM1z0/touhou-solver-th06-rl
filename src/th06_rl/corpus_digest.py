"""Exact run-ID-independent digest for fixed-seed Wine differentials."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


DIGEST_SCHEMA = "th06-rl-normalized-factual-digest-v1"
STREAMS = ("objects", "anchors", "frames", "transitions", "events")
_RUN_REFERENCE_KEYS = frozenset({
    "id", "run_id", "snapshot_ref", "next_snapshot_ref", "control_snapshot_ref",
})
_DROP = object()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: Any, *, run_id: str, key: str | None = None) -> Any:
    if key is not None and (
        key == "capture_ms"
        or key.startswith("capture_ms_")
        or key == "solve_ms"
        or key.startswith("solve_ms_")
    ):
        return _DROP
    if isinstance(value, dict):
        result = {}
        for child_key in sorted(value):
            normalized = _normalize(value[child_key], run_id=run_id, key=child_key)
            if normalized is not _DROP:
                result[child_key] = normalized
        return result
    if isinstance(value, list):
        return [_normalize(item, run_id=run_id) for item in value]
    if isinstance(value, str) and key in _RUN_REFERENCE_KEYS:
        if value == run_id:
            return "<run>"
        prefix = f"{run_id}:"
        if value.startswith(prefix):
            return "<run>:" + value[len(prefix):]
    return value


def normalized_factual_digest(run_dir: Path) -> dict[str, object]:
    """Validate every declared shard and hash canonical factual record streams."""
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_id = str(run.get("run_id", ""))
    shards = manifest.get("shards")
    if (
        not run_id
        or not isinstance(shards, list)
        or manifest.get("complete") is not True
        or manifest.get("stage_trajectory_complete") is not True
        or manifest.get("dropped_records") != 0
    ):
        raise ValueError("factual digest requires a complete no-drop episode")
    by_stream: dict[str, list[dict[str, object]]] = {name: [] for name in STREAMS}
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("stream") not in by_stream:
            raise ValueError("manifest contains an unknown factual shard")
        relative = str(shard.get("path", ""))
        path = run_dir / relative
        if Path(relative).name != relative or not path.is_file() or path.is_symlink():
            raise ValueError("manifest factual shard path is invalid")
        if _sha256(path) != shard.get("sha256"):
            raise ValueError(f"factual shard hash differs: {relative}")
        by_stream[str(shard["stream"])].append(shard)

    stream_digests: dict[str, str] = {}
    stream_records: dict[str, int] = {}
    combined = hashlib.sha256()
    for stream in STREAMS:
        digest = hashlib.sha256()
        records = 0
        rows = sorted(
            by_stream[stream],
            key=lambda row: (int(row.get("first_sequence", -1)), str(row["path"])),
        )
        for shard in rows:
            count = 0
            with gzip.open(run_dir / str(shard["path"]), "rt", encoding="utf-8") as source:
                for line in source:
                    value = json.loads(line)
                    normalized = _normalize(value, run_id=run_id)
                    payload = json.dumps(
                        normalized,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    digest.update(payload)
                    digest.update(b"\n")
                    count += 1
            if count != shard.get("records"):
                raise ValueError(f"factual shard record count differs: {shard['path']}")
            records += count
        stream_digests[stream] = digest.hexdigest()
        stream_records[stream] = records
        combined.update(stream.encode("ascii"))
        combined.update(b"\0")
        combined.update(stream_digests[stream].encode("ascii"))
        combined.update(b"\n")
    return {
        "schema": DIGEST_SCHEMA,
        "sha256": combined.hexdigest(),
        "stream_sha256": stream_digests,
        "records": stream_records,
    }
