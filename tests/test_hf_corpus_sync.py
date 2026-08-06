from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.sync_hf_corpus import (
    build_checkpoint_snapshot,
    discover_runs,
    validate_run,
)


def _closed_run(root: Path, run_id: str, *, complete: bool = True) -> Path:
    run = root / run_id
    run.mkdir(parents=True)
    shard = run / "frames-000000-placeholder.jsonl.gz"
    shard.write_bytes(b"evidence")
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    renamed = shard.with_name(f"frames-000000-{digest[:16]}.jsonl.gz")
    shard.rename(renamed)
    (run / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "th06-rl-run-v1",
                "run_id": run_id,
                "metadata": {
                    "difficulty": 3,
                    "character": 0,
                    "shot_type": 0,
                    "stage": 6,
                    "code_commit": "abc",
                },
                "schemas": {"transition": "v4"},
            }
        )
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "th06-rl-manifest-v2",
                "run_id": run_id,
                "complete": complete,
                "compressed_bytes": len(b"evidence"),
                "dropped_records": 0,
                "episode": {
                    "complete": True,
                    "termination_reason": "practice-stage-complete",
                },
                "run_outcome": {
                    "stage_completed": True,
                    "infrastructure_failures": 0,
                    "physical_hits": 7,
                },
                "records": {"frames": 1},
                "shards": [
                    {
                        "path": renamed.name,
                        "sha256": digest,
                        "compressed_bytes": len(b"evidence"),
                    }
                ],
            }
        )
    )
    return run


def test_discovers_only_closed_runs_and_labels_training_stratum(tmp_path) -> None:
    closed = _closed_run(tmp_path, "closed")
    _closed_run(tmp_path, "open", complete=False)

    runs = discover_runs(tmp_path, verify_content=True)

    assert [path for path, _row in runs] == [closed]
    assert runs[0][1]["training_eligible_complete_stage"] is True


def test_rejects_manifest_path_traversal(tmp_path) -> None:
    run = _closed_run(tmp_path, "unsafe")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["shards"][0]["path"] = "../outside"
    (run / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unsafe or duplicate"):
        validate_run(run)


def test_checkpoint_snapshot_uses_stage_start_during_live_transaction(tmp_path) -> None:
    policies = tmp_path / "policies"
    policies.mkdir()
    state = policies / "lunatic_reimu_a_stage6.json"
    state.write_text(
        json.dumps(
            {
                "schema": "th06-rl-online-ucb-v1",
                "reward_version": "survival-reserve-v1",
                "decisions": 2,
                "trials": {"new": 2},
            }
        )
    )
    backup = policies / ".lunatic_reimu_a_stage6.json.stage-start"
    backup.write_text(
        json.dumps(
            {
                "schema": "th06-rl-online-ucb-v1",
                "reward_version": "survival-reserve-v1",
                "decisions": 1,
                "trials": {"committed": 1},
            }
        )
    )
    (policies / ".lunatic_reimu_a_stage6.json.stage-transaction.json").write_text(
        "{}"
    )

    manifest = build_checkpoint_snapshot(policies, tmp_path / "snapshot")

    row = manifest["policies"][state.name]
    assert row["selection"] == "stage-start-committed"
    assert row["decisions"] == 1
    copied = json.loads((tmp_path / "snapshot" / state.name).read_text())
    assert copied["trials"] == {"committed": 1}
