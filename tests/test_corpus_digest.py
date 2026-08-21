from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from th06_rl.corpus_digest import normalized_factual_digest


def _write_run(root: Path, run_id: str, *, x: float, capture_ms: float) -> Path:
    root.mkdir(parents=True)
    rows = {
        "objects": [{"object_id": "same", "payload": ["x"]}],
        "frames": [{
            "sequence": 0,
            "snapshot_ref": f"{run_id}:00000000:f1",
            "snapshot": {"x": x},
            "decision": {"capture_ms": capture_ms, "solve_ms": 1.0},
        }],
        "transitions": [{
            "sequence": 0,
            "episode": {"id": run_id, "done": True},
            "snapshot_ref": f"{run_id}:00000000:f1",
            "next_snapshot_ref": f"{run_id}:00000001:f2",
            "outcome_terms": {"capture_ms_before": capture_ms},
        }],
        "events": [{
            "sequence": 0,
            "snapshot_ref": f"{run_id}:00000000:f1",
            "event": "hit",
        }],
    }
    shards = []
    for stream, values in rows.items():
        path = root / f"{stream}-000000.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as output:
            for value in values:
                output.write(json.dumps(value) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        shards.append({
            "stream": stream,
            "path": path.name,
            "sha256": digest,
            "records": len(values),
            "first_sequence": 0,
        })
    (root / "run.json").write_text(json.dumps({"run_id": run_id}))
    (root / "manifest.json").write_text(json.dumps({
        "complete": True,
        "stage_trajectory_complete": True,
        "dropped_records": 0,
        "shards": shards,
    }))
    return root


def test_digest_ignores_only_run_identity_and_diagnostic_timing(tmp_path: Path) -> None:
    left = _write_run(tmp_path / "left", "run-left", x=12.0, capture_ms=1.0)
    right = _write_run(tmp_path / "right", "run-right", x=12.0, capture_ms=99.0)

    assert normalized_factual_digest(left)["sha256"] == (
        normalized_factual_digest(right)["sha256"]
    )


def test_digest_detects_a_factual_difference(tmp_path: Path) -> None:
    left = _write_run(tmp_path / "left", "run-left", x=12.0, capture_ms=1.0)
    right = _write_run(tmp_path / "right", "run-right", x=13.0, capture_ms=1.0)

    assert normalized_factual_digest(left)["sha256"] != (
        normalized_factual_digest(right)["sha256"]
    )


def test_digest_rejects_corrupt_declared_shard(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "run", "run", x=12.0, capture_ms=1.0)
    with (run / "frames-000000.jsonl.gz").open("ab") as output:
        output.write(b"corrupt")

    with pytest.raises(ValueError, match="hash differs"):
        normalized_factual_digest(run)
