from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import th06_rl.g7_dataset as g7_dataset
from th06_rl.g7_dataset import (
    COLLECTION_SCHEMA,
    DATASET_SCHEMA,
    _verify_shards,
    build_dataset_index,
    load_admitted_episodes,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _collection(path: Path) -> dict[str, object]:
    value = {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "repository_commit": "a" * 40,
        "game_clock": "original-retail-normal-speed",
        "natural_rng": True,
        "episode_unit": "complete-route",
        "episodes": [{"episode": 0}, {"episode": 1}],
    }
    _json(path, value)
    return value


def _admitted(index: int) -> dict[str, object]:
    return {
        "episode_id": f"episode-{index}",
        "run_dir": f"corpus/episode-{index}/run",
        "report_path": f"artifacts/episode-{index}/report.json",
        "audit_path": f"artifacts/episode-{index}/audit.json",
        "run_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "report_sha256": "3" * 64,
        "audit_sha256": "4" * 64,
        "factual_digest": f"{index + 5:x}" * 64,
        "code_commit": "a" * 40,
        "policy_seed": index + 10,
        "exploration_probability": 0.2,
        "behavior_policy_id": "safe-option-exploration-v2",
        "options": 1,
        "eligible_options": 1,
        "physical_hits": index,
        "controlled_hits": index,
        "interstitial_hits": 0,
        "controlled_elapsed_frames": 8,
        "interstitial_elapsed_frames": 0,
    }


def _option(index: int):
    return SimpleNamespace(
        episode_id=f"episode-{index}",
        option_id=f"{index + 10:016x}:00000001",
    )


def test_dataset_index_contains_only_provenance_and_algorithm_free_totals(
    tmp_path: Path, monkeypatch,
) -> None:
    collection_path = tmp_path / "artifacts/collection.json"
    _collection(collection_path)

    def admit(_repository, row, *, collection_commit):
        index = int(row["episode"])
        return _admitted(index), (_option(index),)

    monkeypatch.setattr(g7_dataset, "_admit_episode", admit)
    result = build_dataset_index((collection_path,), repository=tmp_path)

    assert result["schema"] == DATASET_SCHEMA
    assert result["totals"] == {
        "episodes": 2,
        "options": 2,
        "eligible_options": 2,
        "physical_hits": 1,
        "controlled_hits": 1,
        "interstitial_hits": 0,
    }
    assert "algorithm" not in result
    assert result["collections"][0]["path"] == "artifacts/collection.json"


def test_dataset_load_replays_bound_collection_admission(
    tmp_path: Path, monkeypatch,
) -> None:
    collection_path = tmp_path / "artifacts/collection.json"
    _collection(collection_path)

    def admit(_repository, row, *, collection_commit):
        index = int(row["episode"])
        return _admitted(index), (_option(index),)

    monkeypatch.setattr(g7_dataset, "_admit_episode", admit)
    index = build_dataset_index((collection_path,), repository=tmp_path)
    dataset_path = tmp_path / "datasets/d0.json"
    _json(dataset_path, index)

    episodes = load_admitted_episodes(dataset_path, repository=tmp_path)

    assert [episode[0].episode_id for episode in episodes] == [
        "episode-0", "episode-1"
    ]
    collection_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_admitted_episodes(dataset_path, repository=tmp_path)


def test_shard_verification_rejects_mutation_and_path_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = []
    for stream in ("frames", "transitions", "objects"):
        path = run_dir / f"{stream}.jsonl.gz"
        path.write_bytes(stream.encode())
        rows.append({
            "stream": stream,
            "path": path.name,
            "compressed_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest = {"shards": rows}
    _verify_shards(run_dir, manifest)

    (run_dir / "frames.jsonl.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        _verify_shards(run_dir, manifest)
    manifest["shards"][0]["path"] = "../escape"
    with pytest.raises(ValueError, match="escapes"):
        _verify_shards(run_dir, manifest)
