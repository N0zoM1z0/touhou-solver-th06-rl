from __future__ import annotations

import hashlib
import json
from pathlib import Path

from th06_rl.wine_corpus_registry import (
    build_wine_corpus_source,
    load_wine_corpus_registry,
    select_wine_corpora,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_run(
    path: Path, *, executable: str, stage: int, hits: int,
    transition: str = "th06-rl-transition-v10",
) -> None:
    _write(path / "run.json", {
        "run_id": path.name,
        "schemas": {"transition": transition},
        "metadata": {"stage": stage, "executable_sha256": executable},
    })
    _write(path / "manifest.json", {
        "complete": True,
        "stage_trajectory_complete": True,
        "dropped_records": 0,
        "run_outcome": {
            "stage_completed": True,
            "physical_hits": hits,
            "corpus_failure": None,
            "background_reactivations": 0,
            "capture_failures": 0,
            "corpus_failures": 0,
            "infrastructure_failures": 0,
            "trace_failures": 0,
        },
    })


def test_registry_selects_by_capability_not_source_generation(tmp_path: Path) -> None:
    executable = "a" * 64
    source = tmp_path / "artifacts/source"
    run_dir = source / "run-1"
    _write(run_dir / "run.json", {
        "run_id": "run-1",
        "schemas": {"transition": "transition-v1"},
        "metadata": {"stage": 4, "executable_sha256": executable},
    })
    _write(run_dir / "manifest.json", {
        "complete": True,
        "stage_trajectory_complete": True,
        "dropped_records": 0,
        "run_outcome": {
            "stage_completed": True,
            "physical_hits": 3,
            "corpus_failure": None,
            "background_reactivations": 0,
            "capture_failures": 0,
            "corpus_failures": 0,
            "infrastructure_failures": 0,
            "trace_failures": 0,
        },
    })
    manifest_sha = hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
    run_sha = hashlib.sha256((run_dir / "run.json").read_bytes()).hexdigest()
    inventory = hashlib.sha256()
    inventory.update(b"artifacts/source/run-1\0")
    inventory.update(manifest_sha.encode())
    inventory.update(b"\0")
    inventory.update(run_sha.encode())
    inventory.update(b"\n")
    config = tmp_path / "registry.json"
    _write(config, {
        "schema": "immutable-wine-corpus-registry-v1",
        "original_retail_executable_sha256": executable,
        "sources": [{
            "id": "historical-origin-name",
            "access": "training",
            "capabilities": ["state-value", "sequential-rl"],
            "root": "artifacts/source",
            "transition_schema": "transition-v1",
            "expected_clean_complete_runs": 1,
            "inventory_sha256": inventory.hexdigest(),
        }],
    })

    _registry, entries = load_wine_corpus_registry(config, repository=tmp_path)

    assert len(select_wine_corpora(
        entries, required_capabilities=frozenset({"sequential-rl"})
    )) == 1
    assert not select_wine_corpora(
        entries, required_capabilities=frozenset({"counterfactual-successor"})
    )


def test_build_source_round_trips_through_registry(tmp_path: Path) -> None:
    executable = "e" * 64
    root = tmp_path / "artifacts" / "round"
    _write_run(root / "run-0", executable=executable, stage=5, hits=3)
    source = build_wine_corpus_source(
        repository=tmp_path,
        root=root,
        source_id="round-v1",
        access="training",
        capabilities=("sequential_offline_rl", "natural_rng"),
        transition_schema="th06-rl-transition-v10",
        executable_sha256=executable,
    )
    config = tmp_path / "registry.json"
    _write(config, {
        "schema": "immutable-wine-corpus-registry-v1",
        "original_retail_executable_sha256": executable,
        "sources": [source],
    })
    _registry, entries = load_wine_corpus_registry(config, repository=tmp_path)
    assert len(entries) == 1
    assert entries[0].source == "round-v1"
    assert entries[0].physical_hits == 3


def test_build_source_rejects_any_incomplete_manifest(tmp_path: Path) -> None:
    executable = "e" * 64
    root = tmp_path / "artifacts" / "round"
    _write_run(root / "run-0", executable=executable, stage=6, hits=2)
    manifest = root / "run-0" / "manifest.json"
    value = json.loads(manifest.read_text())
    value["complete"] = False
    manifest.write_text(json.dumps(value))
    try:
        build_wine_corpus_source(
            repository=tmp_path,
            root=root,
            source_id="round-v1",
            access="training",
            capabilities=("sequential_offline_rl",),
            transition_schema="th06-rl-transition-v10",
            executable_sha256=executable,
        )
    except ValueError as error:
        assert "rejected run" in str(error)
    else:
        raise AssertionError("incomplete corpus source was accepted")
