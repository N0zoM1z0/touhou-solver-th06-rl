from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from th06_rl.wine_transitions import iter_transition_rows, validate_wine_run


SCHEMA = "test-transition-v1"


def _write_run(tmp_path, *, sequences=(0, 1)):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = "".join(
        json.dumps({"schema_version": SCHEMA, "sequence": sequence}) + "\n"
        for sequence in sequences
    )
    shard = run_dir / "transitions-000000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as output:
        output.write(payload)
    (run_dir / "run.json").write_text(json.dumps({
        "schemas": {"transition": SCHEMA},
        "metadata": {"stage": 6},
    }), encoding="utf-8")
    outcome = {
        "corpus_failure": None,
        "background_reactivations": 0,
        "capture_failures": 0,
        "corpus_failures": 0,
        "infrastructure_failures": 0,
        "trace_failures": 0,
    }
    (run_dir / "manifest.json").write_text(json.dumps({
        "complete": True,
        "stage_trajectory_complete": True,
        "dropped_records": 0,
        "run_outcome": outcome,
        "records": {"transitions": len(sequences)},
        "shards": [{
            "stream": "transitions",
            "path": shard.name,
            "records": len(sequences),
            "compressed_bytes": shard.stat().st_size,
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        }],
    }), encoding="utf-8")
    return run_dir


def test_validation_and_streaming_are_learner_neutral(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    _run, manifest, schema = validate_wine_run(
        run_dir,
        expected_transition_schema=SCHEMA,
        require_stage_complete=True,
    )
    assert [row["sequence"] for row in iter_transition_rows(
        run_dir, manifest, expected_transition_schema=schema
    )] == [0, 1]


def test_stream_rejects_noncontiguous_factual_rows(tmp_path) -> None:
    run_dir = _write_run(tmp_path, sequences=(0, 2))
    _run, manifest, schema = validate_wine_run(run_dir)
    with pytest.raises(ValueError, match="not contiguous"):
        list(iter_transition_rows(
            run_dir, manifest, expected_transition_schema=schema
        ))
