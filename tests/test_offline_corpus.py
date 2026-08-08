from __future__ import annotations

import gzip
import hashlib
import json

from th06_rl.offline import audit_dataset, iter_run_transitions, load_dataset_index
from th06_rl.offline_learning import HIT_CREDIT_DISCOUNT, label_transitions


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    run_id = "20260808T000000Z-test"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    rows = [
        {
            "schema_version": "th06-rl-transition-v5",
            "sequence": 0,
            "snapshot_ref": f"{run_id}:00000000:f100",
            "next_snapshot_ref": f"{run_id}:00000001:f101",
            "scope": {"key": "3/0/0/6/phase:a"},
            "legal_actions": ["stay", "left"],
            "proposed_action": "left",
            "published_action": "left",
            "behavior_probability": 0.25,
            "learning_eligible": True,
            "learning_exclusion_reasons": [],
            "policy_context": {"bullet_count": 12},
            "outcome_terms": {"elapsed_frames": 1, "life_lost": False, "bomb_used": False},
        },
        {
            "schema_version": "th06-rl-transition-v5",
            "sequence": 1,
            "snapshot_ref": f"{run_id}:00000001:f101",
            "next_snapshot_ref": f"{run_id}:00000002:f102",
            "scope": {"key": "3/0/0/6/phase:a"},
            "legal_actions": ["stay"],
            "proposed_action": None,
            "published_action": None,
            "behavior_probability": 1.0,
            "learning_eligible": False,
            "learning_exclusion_reasons": ["action-not-published"],
            "policy_context": {"bullet_count": 13},
            "outcome_terms": {"elapsed_frames": 1, "life_lost": True, "bomb_used": False},
        },
    ]
    shard = run_dir / "transitions-000000-test.jsonl.gz"
    with shard.open("wb") as raw_output, gzip.GzipFile(
        fileobj=raw_output,
        mode="wb",
        mtime=0,
    ) as output:
        for row in rows:
            output.write(_canonical(row))
    run = {"run_id": run_id, "schemas": {"transition": "th06-rl-transition-v5"}}
    (run_dir / "run.json").write_bytes(_canonical(run))
    local_manifest = {
        "run_id": run_id,
        "shards": [{
            "stream": "transitions",
            "path": shard.name,
            "sha256": _sha256(shard),
            "compressed_bytes": shard.stat().st_size,
            "records": len(rows),
        }],
    }
    (run_dir / "manifest.json").write_bytes(_canonical(local_manifest))
    dataset = {
        "schema": "th06-rl-hf-dataset-v1",
        "runs": [{
            "run_id": run_id,
            "remote_path": f"runs/{run_id}",
            "difficulty": 3,
            "character": 0,
            "shot_type": 0,
            "stage": 6,
            "schemas": {"transition": "th06-rl-transition-v5"},
            "records": {"transitions": len(rows)},
            "storage_complete": True,
            "stage_trajectory_complete": True,
            "training_eligible_complete_stage": True,
            "code_commit": "abc",
            "physical_hits": 1,
            "manifest_sha256": _sha256(run_dir / "manifest.json"),
            "run_sha256": _sha256(run_dir / "run.json"),
        }],
    }
    (tmp_path / "dataset_manifest.json").write_bytes(_canonical(dataset))
    return tmp_path


def test_fixed_snapshot_is_hash_and_sequence_validated(tmp_path) -> None:
    root = _fixture(tmp_path)
    _, runs = load_dataset_index(root)

    rows = list(iter_run_transitions(root, runs[0]))

    assert [row["sequence"] for row in rows] == [0, 1]


def test_audit_separates_complete_run_and_transition_eligibility(tmp_path) -> None:
    report = audit_dataset(_fixture(tmp_path))
    overall = report["overall"]

    assert overall["counts"]["rows"] == 2
    assert overall["counts"]["training_eligible_runs"] == 1
    assert overall["counts"]["trainable_rows"] == 1
    assert overall["counts"]["physical_hit_events"] == 1
    assert overall["ratios"]["factual_trainable_per_processed"] == 0.5
    assert overall["hit_window_exposures"]["120"]["positive_rows"] == 1
    assert overall["action_coverage"]["left"]["clipped_ipw_ess"] == 1.0


def test_label_reconstructs_delayed_physical_hit_credit(tmp_path) -> None:
    root = _fixture(tmp_path)
    _, runs = load_dataset_index(root)
    raw = list(iter_run_transitions(root, runs[0]))

    labeled = label_transitions(raw, runs[0], exact_context_only=True)

    assert len(labeled) == 1
    assert labeled[0].hit_within_30 is True
    immediate = 1.0
    assert labeled[0].reward == immediate - 100.0 * HIT_CREDIT_DISCOUNT ** 2
    assert labeled[0].features["context_quality"] == "exact-v5"
