from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

import scripts.run_generation6_autonomous_round as autonomous_round
from scripts.run_generation6_autonomous_round import (
    _materialize_reused_collection,
    _paired_verdict,
    _priority_passed,
    _startup_smoke_passed,
    _validate_contract_shape,
)


def _row(block: int, role: str, hits: int, interventions: int = 0):
    return {
        "block": block,
        "role": role,
        "physical_hits": hits,
        "interventions": interventions,
        "passed": True,
    }


def test_paired_round_requires_aggregate_block_and_exposure_gates() -> None:
    rows = []
    for block, (incumbent, candidate) in enumerate(
        ((10, 8), (9, 8), (11, 7), (7, 8), (10, 9), (8, 8))
    ):
        rows.extend((
            _row(block, "incumbent", incumbent),
            _row(block, "candidate", candidate, 1),
        ))
    result = _paired_verdict(rows)
    assert result["verdict"] == "effective-learning-signal"
    assert result["effect_hits"] == 7
    assert result["candidate_no_worse_blocks"] == 5
    assert result["promotion_eligible"] is False


def test_paired_round_rejects_positive_total_without_block_consistency() -> None:
    rows = []
    for block, (incumbent, candidate) in enumerate(
        ((20, 1), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2))
    ):
        rows.extend((
            _row(block, "incumbent", incumbent),
            _row(block, "candidate", candidate, 1),
        ))
    result = _paired_verdict(rows)
    assert result["effect_hits"] > 0
    assert result["verdict"] == "no-effective-learning-signal"


def _contract():
    cpus = sorted(os.sched_getaffinity(0))[:4]
    return {
        "maximum_interventions": 64,
        "retail_executable_sha256": "e" * 64,
        "frozen_inputs": {"input": "f" * 64},
        "collection": {
            "episodes": 12,
            "stages": [4, 5, 6],
            "schedule": [
                {"episode": index + 1, "stage": (4, 5, 6)[index % 3],
                 "policy_seed": 1_000 + index}
                for index in range(12)
            ],
        },
        "canary": {"schedule": [
            {"trial": index + 1, "stage": 4, "policy_seed": 2_000 + index}
            for index in range(2)
        ]},
        "evaluation": {"schedule": [
            {"trial": block * 2 + role_index + 1, "block": block,
             "role": role, "policy_seed": 3_000 + block * 2 + role_index}
            for block in range(6)
            for role_index, role in enumerate(
                ("incumbent", "candidate")
                if block % 2 == 0 else ("candidate", "incumbent")
            )
        ]},
        "environment": {
            "game_cpu_list": ",".join(map(str, cpus[:2])),
            "controller_cpu_list": ",".join(map(str, cpus[2:])),
            "source_game_inventory_sha256": "g" * 64,
        },
        "offline": {
            "crossfit_seed": 4_000,
            "seed_offset": 5_000,
            "preflight_policy_seed": 6_000,
        },
        "registry_append": {
            "source_id": "round",
            "base_source_count": 5,
            "capabilities": [
                "complete_stage_observation", "physical_hit_outcome",
                "representation_pretraining", "behavior_state_value",
                "factual_semi_markov_options",
                "recorded_complete_behavior_propensity",
                "native_safe_candidates", "sequential_offline_rl",
                "natural_rng", "generation6_actor_ess_behavior",
            ],
        },
    }


def test_round_contract_requires_balanced_outcome_blind_collection() -> None:
    contract = _contract()
    _validate_contract_shape(contract)
    drifted = deepcopy(contract)
    drifted["collection"]["schedule"][0]["rng_seed"] = 99
    with pytest.raises(ValueError, match="balanced/frozen"):
        _validate_contract_shape(drifted)


def _successor_contract() -> dict[str, object]:
    contract = _contract()
    old_schedule = contract["collection"]["schedule"]
    reused = [
        {
            **row,
            "source": f"artifacts/old/episode-{row['episode']:02d}/run",
            "report": f"artifacts/old/episode-{row['episode']:02d}/report.json",
            "report_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "run_sha256": "c" * 64,
        }
        for row in old_schedule[:10]
    ]
    contract["collection"].update({
        "reused_episodes": 10,
        "new_episodes": 2,
        "prior_ledger": "artifacts/old/generation.json",
        "prior_ledger_sha256": "d" * 64,
        "reused_schedule": reused,
        "schedule": old_schedule[10:],
    })
    contract["environment"].update({
        "scheduler": "SCHED_OTHER",
        "game_nice": -10,
        "controller_nice": -10,
    })
    return contract


def test_successor_contract_reuses_only_a_bound_balanced_prefix() -> None:
    contract = _successor_contract()
    _validate_contract_shape(contract)

    missing = deepcopy(contract)
    missing["collection"]["reused_schedule"].pop()
    with pytest.raises(ValueError, match="balanced/frozen"):
        _validate_contract_shape(missing)

    unbound = deepcopy(contract)
    del unbound["collection"]["reused_schedule"][0]["run_sha256"]
    with pytest.raises(ValueError, match="binding"):
        _validate_contract_shape(unbound)

    unbounded_priority = deepcopy(contract)
    unbounded_priority["environment"]["scheduler"] = "SCHED_FIFO"
    with pytest.raises(ValueError, match="process priority"):
        _validate_contract_shape(unbounded_priority)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reuse_materializes_complete_machine_passing_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    source = repository / "artifacts/old/accepted/episode-01/run"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("{}\n", encoding="utf-8")
    (source / "run.json").write_text("{}\n", encoding="utf-8")
    report = repository / "artifacts/old/episode-01/report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}\n", encoding="utf-8")
    report_hash = _digest(report)
    passing = {
        "episode": 1,
        "stage": 4,
        "policy_seed": 1000,
        "passed": True,
        "corpus_run_dir": str(source),
        "report_sha256": report_hash,
        "manifest_sha256": _digest(source / "manifest.json"),
        "run_sha256": _digest(source / "run.json"),
    }
    failed = {**passing, "episode": 2, "stage": 5, "passed": False}
    ledger_path = repository / "artifacts/old/generation.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps({
        "schema": autonomous_round.SCHEMA,
        "status": "invalid",
        "decision": "collection-audit-failed",
        "collection": [passing, failed],
    }), encoding="utf-8")
    contract = {"collection": {
        "prior_ledger": str(ledger_path.relative_to(repository)),
        "prior_ledger_sha256": _digest(ledger_path),
        "reused_schedule": [{
            "episode": 1,
            "stage": 4,
            "policy_seed": 1000,
            "source": str(source.relative_to(repository)),
            "report": str(report.relative_to(repository)),
            "report_sha256": report_hash,
            "manifest_sha256": passing["manifest_sha256"],
            "run_sha256": passing["run_sha256"],
        }],
    }}
    monkeypatch.setattr(autonomous_round, "REPOSITORY", repository)
    monkeypatch.setattr(autonomous_round, "_validate_run", lambda *args, **kwargs: None)

    accepted = repository / "artifacts/new/accepted-corpus"
    rows = _materialize_reused_collection(
        contract=contract, accepted_root=accepted
    )
    destination = accepted / "episode-01/run"
    assert rows[0]["reused_under_successor_contract"] is True
    assert rows[0]["corpus_run_dir"] == str(destination)
    assert destination.is_dir()
    assert os.stat(source / "run.json").st_ino == os.stat(
        destination / "run.json"
    ).st_ino

    # A contract may not cherry-pick around a failed row or omit a passing row.
    incomplete = deepcopy(contract)
    incomplete["collection"]["reused_schedule"] = []
    assert _materialize_reused_collection(
        contract=incomplete, accepted_root=accepted
    ) == []
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["collection"].insert(1, {**passing, "episode": 2})
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    incomplete["collection"]["prior_ledger_sha256"] = _digest(ledger_path)
    incomplete["collection"]["reused_schedule"] = contract["collection"][
        "reused_schedule"
    ]
    with pytest.raises(ValueError, match="complete passing prefix"):
        _materialize_reused_collection(
            contract=incomplete, accepted_root=accepted
        )


def test_priority_gate_requires_exact_dropped_identity_attestation() -> None:
    environment = {
        "game_cpu_list": "0-1",
        "controller_cpu_list": "2-3",
        "game_nice": -10,
        "controller_nice": -10,
    }
    report = {}
    for role, cpus in (("game", [0, 1]), ("controller", [2, 3])):
        report[f"{role}_nice"] = -10
        report[f"{role}_priority_attestation"] = {
            "schema": "bounded-wine-process-priority-v1",
            "authority": "linux-setpriority-and-sched-setaffinity",
            "scheduler": "SCHED_OTHER",
            "nice": -10,
            "effective_nice": -10,
            "cpus": cpus,
            "effective_cpus": cpus,
            "uid": 1000,
            "target_uid": 1000,
            "gid": 1000,
            "target_gid": 1000,
        }
    assert _priority_passed(report, environment)
    report["controller_priority_attestation"]["effective_nice"] = -9
    assert not _priority_passed(report, environment)


def test_startup_smoke_is_non_gameplay_and_binds_exec_child_identity() -> None:
    environment = {
        "game_cpu_list": "0-1",
        "controller_cpu_list": "2-3",
        "game_nice": -10,
        "controller_nice": -10,
    }
    report = {
        "error": None,
        "gdb_normalized": True,
        "controller_returncode": 0,
        "immutable_policy_state_equal": True,
        "evaluation_mode": "hit-continuation-benchmark",
        "complete_stage_training_corpus_root": None,
        "first_failure_corpus_root": None,
        "option_smoke_corpus_root": None,
        "seconds": 0.25,
        "trace": {"corpus_run_ids": [], "decisions": 0},
        "leftover_prefix_processes": [],
        "game_host_pid": 100,
        "game_process_pid": 101,
    }
    for role, cpus in (("game", [0, 1]), ("controller", [2, 3])):
        report[f"{role}_nice"] = -10
        report[f"{role}_priority_attestation"] = {
            "schema": "bounded-wine-process-priority-v1",
            "authority": "linux-setpriority-and-sched-setaffinity",
            "scheduler": "SCHED_OTHER",
            "nice": -10,
            "effective_nice": -10,
            "cpus": cpus,
            "effective_cpus": cpus,
            "uid": 1000,
            "target_uid": 1000,
            "gid": 1000,
            "target_gid": 1000,
            "pid": 101 if role == "game" else 102,
        }
    assert _startup_smoke_passed(report, environment)
    report["trace"]["corpus_run_ids"] = ["forbidden"]
    assert not _startup_smoke_passed(report, environment)


def test_paired_round_reports_invalid_shape_without_partial_verdict() -> None:
    result = _paired_verdict([_row(0, "candidate", 1, 1)])
    assert result["verdict"] == "invalid"
    assert result["incumbent_hits"] == []
    assert result["candidate_hits"] == []
