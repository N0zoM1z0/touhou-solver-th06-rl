import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts.gate_parallel_wine import (
    _gate_rows,
    _pool,
    build_runner_command,
    run_batch,
    validate_gate_run,
)
from th06_rl.wine_workers import (
    allocate_worker_specifications,
    attest_retail_template,
    prepare_wine_workers,
)


def _documents(stage: int = 4):
    hits = 3
    report = {
        "controller_returncode": 0,
        "controller_completion": {
            "practice_stage_completed": True,
            "practice_stage": stage,
            "physical_hits": hits,
        },
        "retail_sha256": (
            "9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245"
        ),
        "leftover_prefix_processes": [],
        "native_sha256": "n",
        "policy_plugin_sha256": "p",
        "policy_state_sha256_before": "s",
    }
    run = {
        "schemas": {"frame": "th06-rl-authoritative-frame-v13"},
        "metadata": {"online_contract": {
            "algorithm": "observed-shield4-paused-publication-v1",
            "shield_contract": "observed-hazard-kinematics-v1",
            "publication_epoch": "coherent-root-process-suspended-v1",
            "shield_horizon": 4,
            "minimum_collision_margin": 0.35,
            "predicts_future_births": False,
        }},
    }
    outcome = {
        "stage_completed": True,
        "controller_exit_code": 0,
        "corpus_failure": None,
        "physical_hits": hits,
        "background_reactivations": 0,
        "capture_failures": 0,
        "corpus_failures": 0,
        "infrastructure_failures": 0,
        "policy_failures": 0,
        "policy_last_error": None,
        "trace_failures": 0,
    }
    manifest = {
        "complete": True,
        "stage_trajectory_complete": True,
        "dropped_records": 0,
        "run_outcome": outcome,
    }
    audit = {
        "physical_hits": hits,
        "bomb_events": 0,
        "integrity_errors": [],
        "player_successor_parity": {
            "method": "contiguous-player-center-successor-v1",
            "arithmetic_comparison": "float32-bit-exact",
            "input_semantics": "next-completed-root-sampled-input",
            "checked_links": 90,
            "mismatches": 0,
        },
        "dense_shield_parity": {
            "checked": 64,
            "unsafe_divergences": [],
            "conservative_divergences": [],
        },
        "latency": {
            "capture": {"p99_ms": 20.0},
            "solve": {"p99_ms": 8.0},
            "observation_gap_rate": 0.0,
            "stale_retry_rate": 0.0,
        },
    }
    return report, run, manifest, audit


def test_gate_requires_current_observed_shield_clean_episode() -> None:
    report, run, manifest, audit = _documents()
    result = validate_gate_run(
        report=report, run=run, manifest=manifest, audit=audit, stage=4,
    )
    assert result["physical_hits"] == 3
    assert all(result["checks"].values())

    audit["dense_shield_parity"]["unsafe_divergences"] = ["extra-action"]
    with pytest.raises(ValueError, match="native_parity"):
        validate_gate_run(
            report=report, run=run, manifest=manifest, audit=audit, stage=4,
        )


def test_gate_rejects_online_latency_regression() -> None:
    report, run, manifest, audit = _documents()
    audit["latency"]["solve"]["p99_ms"] = 17.0
    with pytest.raises(ValueError, match="online_latency"):
        validate_gate_run(
            report=report, run=run, manifest=manifest, audit=audit, stage=4,
        )


def test_gate_rejects_player_successor_mismatch() -> None:
    report, run, manifest, audit = _documents()
    audit["player_successor_parity"]["mismatches"] = 1
    with pytest.raises(ValueError, match="player_successor"):
        validate_gate_run(
            report=report, run=run, manifest=manifest, audit=audit, stage=4,
        )


def test_gate_runner_command_is_complete_normal_speed_hit_continuation(
    tmp_path: Path,
) -> None:
    command = build_runner_command(
        worker={
            "game_dir": tmp_path / "game",
            "wine_prefix": tmp_path / "prefix",
            "display": ":107",
            "game_cpu_list": "0,1,2,3",
            "controller_cpu_list": "4,5,6,7",
        },
        score_template=tmp_path / "score.dat",
        policy_plugin=tmp_path / "policy.py",
        policy_state=tmp_path / "policy.json",
        stage=4,
        rng_seed=0x1234,
        artifact_dir=tmp_path / "artifacts",
        corpus_root=tmp_path / "corpus",
    )
    assert "--complete-stage-training-corpus-root" in command
    assert command[command.index("--diagnostic-rng-seed") + 1] == "0x1234"
    assert "--stop-on-hit" not in command
    assert "--seconds" in command
    assert command[command.index("--seconds") + 1] == "0"


def test_pool_validates_real_worker_markers_and_filesystem_ownership(
    tmp_path: Path,
) -> None:
    source = tmp_path / "template"
    source.mkdir()
    executable = source / "東方紅魔郷.exe"
    executable.write_bytes(b"retail")
    attest_retail_template(
        source,
        archive_sha256="a" * 64,
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
    )
    specifications = allocate_worker_specifications(
        available_cpus=tuple(range(32)), workers=4, cpus_per_worker=8,
    )
    workers = prepare_wine_workers(
        root=tmp_path / "workers",
        source_game_dir=source,
        specifications=specifications,
    )
    rows = [
        {
            **worker,
            "game_cpu_list": specification["game_cpu_list"],
            "controller_cpu_list": specification["controller_cpu_list"],
        }
        for worker, specification in zip(workers, specifications, strict=True)
    ]
    pool_path = tmp_path / "pool.json"
    resource_contract = {
        "worker_count": 4,
        "assigned_total_cpus": 32,
        "cpus_per_worker": 8,
        "minimum_memory_gib_per_worker": 8.0,
        "minimum_disk_gib_per_worker": 8.0,
        "game_clock": "original-retail-normal-speed",
    }
    pool_path.write_text(json.dumps({
        "schema": "th06-rl-normal-speed-wine-pool-v2",
        "workers": rows,
        "resource_contract": resource_contract,
    }))
    assert len(_pool(pool_path)["workers"]) == 4
    assert [name for name, _worker in _gate_rows(rows)] == [
        "serial-worker-0",
        "concurrent-worker-0",
        "concurrent-worker-1",
        "concurrent-worker-2",
        "concurrent-worker-3",
    ]

    resource_contract["assigned_total_cpus"] = 31
    pool_path.write_text(json.dumps({
        "schema": "th06-rl-normal-speed-wine-pool-v2",
        "workers": rows,
        "resource_contract": resource_contract,
    }))
    with pytest.raises(ValueError, match="resource contract"):
        _pool(pool_path)
    resource_contract["assigned_total_cpus"] = 32

    rows[1]["game_dir"] = rows[0]["game_dir"]
    pool_path.write_text(json.dumps({
        "schema": "th06-rl-normal-speed-wine-pool-v2",
        "workers": rows,
        "resource_contract": resource_contract,
    }))
    with pytest.raises(ValueError, match="integrity differs"):
        _pool(pool_path)


def test_failed_worker_interrupts_and_reaps_batch(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    failing = [sys.executable, "-c", "raise SystemExit(7)"]
    sibling = [
        sys.executable,
        "-c",
        (
            "import pathlib,time; time.sleep(10); "
            f"pathlib.Path({str(marker)!r}).write_text('late')"
        ),
    ]
    with pytest.raises(RuntimeError, match="batch failed"):
        run_batch([
            ("failure", failing, tmp_path / "failure.log"),
            ("sibling", sibling, tmp_path / "sibling.log"),
        ])
    assert not marker.exists()
