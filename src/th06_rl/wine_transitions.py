"""Learner-neutral validation and streaming for immutable Wine transitions."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterator


_CLEAN_OUTCOME_FIELDS = (
    "background_reactivations",
    "capture_failures",
    "corpus_failures",
    "infrastructure_failures",
    "trace_failures",
)


def load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_wine_run(
    run_dir: Path,
    *,
    expected_transition_schema: str | None = None,
    require_stage_complete: bool = False,
) -> tuple[dict[str, object], dict[str, object], str]:
    """Validate physical provenance without imposing a learner generation."""
    run_dir = run_dir.resolve()
    run = load_json_object(run_dir / "run.json")
    manifest = load_json_object(run_dir / "manifest.json")
    if (
        manifest.get("complete") is not True
        or int(manifest.get("dropped_records", -1)) != 0
    ):
        raise ValueError(f"incomplete physical corpus: {run_dir}")
    if require_stage_complete and (
        manifest.get("stage_trajectory_complete") is not True
    ):
        raise ValueError(f"physical Stage trajectory is incomplete: {run_dir}")
    schemas = run.get("schemas")
    transition_schema = (
        str(schemas.get("transition", "")) if isinstance(schemas, dict) else ""
    )
    if not transition_schema:
        raise ValueError("physical corpus transition schema is absent")
    if (
        expected_transition_schema is not None
        and transition_schema != expected_transition_schema
    ):
        raise ValueError(
            "physical corpus transition schema mismatch: "
            f"{transition_schema!r} != {expected_transition_schema!r}"
        )
    outcome = manifest.get("run_outcome")
    if not isinstance(outcome, dict):
        raise TypeError("physical run outcome is absent")
    for field in _CLEAN_OUTCOME_FIELDS:
        if int(outcome.get(field, -1)) != 0:
            raise ValueError(f"physical corpus has infrastructure failure: {field}")
    if outcome.get("corpus_failure") is not None:
        raise ValueError("physical corpus writer failed")
    if not isinstance(run.get("metadata"), dict):
        raise TypeError("physical run metadata is absent")
    return run, manifest, transition_schema


def iter_transition_rows(
    run_dir: Path,
    manifest: dict[str, object],
    *,
    expected_transition_schema: str,
) -> Iterator[dict[str, object]]:
    """Yield losslessly verified factual rows in contiguous sequence order."""
    if not expected_transition_schema:
        raise ValueError("expected transition schema cannot be empty")
    run_dir = run_dir.resolve()
    expected_sequence = 0
    expected_records = 0
    observed_records = 0
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise TypeError("corpus manifest shard list is invalid")
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("stream") != "transitions":
            continue
        name = str(shard.get("path", ""))
        if not name or Path(name).name != name:
            raise ValueError("unsafe transition shard path")
        path = run_dir / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(shard.get("compressed_bytes", -1)):
            raise ValueError(f"transition shard size mismatch: {path}")
        if sha256_file(path) != shard.get("sha256"):
            raise ValueError(f"transition shard digest mismatch: {path}")
        expected_records += int(shard.get("records", 0))
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("transition row is not an object")
                if row.get("schema_version") != expected_transition_schema:
                    raise ValueError("transition row schema mismatch")
                if int(row.get("sequence", -1)) != expected_sequence:
                    raise ValueError("transition sequence is not contiguous")
                expected_sequence += 1
                observed_records += 1
                yield row
    manifest_records = manifest.get("records")
    recorded = (
        int(manifest_records.get("transitions", -1))
        if isinstance(manifest_records, dict)
        else -1
    )
    if observed_records != expected_records or observed_records != recorded:
        raise ValueError("transition record count mismatch")
