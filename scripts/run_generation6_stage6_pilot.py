#!/usr/bin/env python3
"""Run the frozen alternating Generation-6 natural-RNG Stage-6 pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.export_generation6_policy import export_state  # noqa: E402
from scripts.run_generation5_wine import complete_run  # noqa: E402
from scripts.run_generation6_wine_canary import (  # noqa: E402
    CANDIDATE,
    DEPLOYABLE_AUDIT,
    INFRA_EVENTS,
    NATIVE_SCORER,
    POLICY_PLUGIN,
    QUALIFICATION,
    _atomic_json,
    _bind_resources,
    _last_policy_status,
    _object,
    _sha256,
)
from th06_rl.actions import ACTION_NAMES  # noqa: E402
from th06_rl.policies.autonomous_iql_actor import (  # noqa: E402
    ALLOWED_CANARY_CONTRACT_SHA256,
    POLICY_NAME,
)
from th06_rl.wine_workers import prepare_wine_worker  # noqa: E402


def pilot_verdict(
    rows: list[dict[str, object]], *, expected_runs: int,
    required_exercised_candidate_stages: int,
) -> dict[str, object]:
    incumbent = [
        int(row["physical_hits"]) for row in rows
        if row.get("role") == "incumbent"
    ]
    candidate = [
        int(row["physical_hits"]) for row in rows
        if row.get("role") == "candidate"
    ]
    runtime_valid = (
        len(rows) == expected_runs
        and len(incumbent) == len(candidate) == expected_runs // 2
        and all(row.get("passed") is True for row in rows)
    )
    exercised = sum(
        int(row.get("interventions", 0)) > 0 for row in rows
        if row.get("role") == "candidate"
    )
    exposure_valid = exercised >= required_exercised_candidate_stages
    incumbent_total = sum(incumbent)
    candidate_total = sum(candidate)
    strict_reduction = candidate_total < incumbent_total
    if not runtime_valid:
        verdict = "invalid"
    elif not exposure_valid:
        verdict = "inconclusive"
    elif strict_reduction:
        verdict = "effective-pilot-signal"
    else:
        verdict = "no-effective-pilot-signal"
    return {
        "incumbent_hits": incumbent,
        "candidate_hits": candidate,
        "incumbent_total_hits": incumbent_total,
        "candidate_total_hits": candidate_total,
        "effect_hits": incumbent_total - candidate_total,
        "candidate_exercised_stages": exercised,
        "gates": {
            "all_runs_complete_and_clean": runtime_valid,
            "candidate_exercised_in_required_stages": exposure_valid,
            "strictly_fewer_candidate_hits": strict_reduction,
        },
        "verdict": verdict,
    }


def _state(
    *, path: Path, contract_path: Path, role: str, policy_seed: int,
) -> dict[str, object]:
    expected = export_state(
        candidate_path=CANDIDATE,
        qualification_path=QUALIFICATION,
        deployable_audit_path=DEPLOYABLE_AUDIT,
        native_scorer_path=NATIVE_SCORER,
        canary_contract_path=contract_path,
        mode="active" if role == "candidate" else "shadow",
        policy_seed=policy_seed,
    )
    if path.is_file():
        if _object(path) != expected:
            raise ValueError(f"Generation-6 pilot state drifted: {path}")
    else:
        _atomic_json(path, expected)
    return expected


def _audit_run(
    *, report: dict[str, object], artifact_dir: Path, state_path: Path,
    schedule: dict[str, object], environment: dict[str, object],
) -> dict[str, object]:
    trace = report.get("trace")
    completion = report.get("controller_completion")
    if not isinstance(trace, dict) or not isinstance(completion, dict):
        raise ValueError("Generation-6 pilot report lacks completion evidence")
    status = _last_policy_status(artifact_dir / "trace.jsonl")
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Generation-6 pilot report lacks policy metrics")
    role = str(schedule["role"])
    mode = "active" if role == "candidate" else "shadow"
    selected = metrics.get("selected")
    selected_actions = set(selected) if isinstance(selected, dict) else set()
    events = trace.get("event_counts")
    gates = {
        "expected_evaluation_mode": report.get("evaluation_mode")
        == "hit-continuation-benchmark",
        "natural_rng": report.get("diagnostic_rng_seed") is None,
        "expected_stage": completion.get("practice_stage") == 6,
        "complete_stage": completion.get("practice_stage_completed") is True,
        "controller_success": report.get("controller_returncode") == 0,
        "immutable_policy_state": (
            report.get("immutable_policy_state_equal") is True
            and report.get("policy_state_sha256_before") == _sha256(state_path)
        ),
        "optimized_native_scorer": report.get("policy_scorer_library_sha256")
        == _sha256(NATIVE_SCORER),
        "frozen_cpu_partitions": (
            report.get("game_cpu_list") == environment.get("game_cpu_list")
            and report.get("controller_cpu_list")
            == environment.get("controller_cpu_list")
        ),
        "zero_infrastructure_events": isinstance(events, dict) and not any(
            int(events.get(name, 0)) for name in INFRA_EVENTS
        ),
        "policy_loaded_once": (
            status.get("policy_id") == f"{POLICY_NAME}-{mode}"
            and status.get("reload_failures") == 0
            and status.get("last_error") is None
        ),
        "expected_policy_mode_and_seed": (
            metrics.get("mode") == mode
            and metrics.get("policy_seed") == int(schedule["policy_seed"])
        ),
        "native_safe_actions_only": bool(selected_actions)
        and selected_actions <= set(ACTION_NAMES),
        "zero_bomb": "bomb" not in selected_actions,
        "shadow_zero_interventions": (
            role != "incumbent" or metrics.get("interventions") == 0
        ),
        "candidate_budget_respected": (
            role != "candidate"
            or (
                int(metrics.get("interventions", -1))
                <= int(metrics.get("intervention_budget", -2))
                <= 64
                and metrics.get("budget_abstentions") == 0
            )
        ),
        "online_p95_below_4_ms": (
            isinstance(metrics.get("latency_p95_ms"), (int, float))
            and float(metrics["latency_p95_ms"]) < 4.0
        ),
        "zero_policy_deadline_misses": metrics.get("deadline_misses") == 0,
        "no_corpus_created": not trace.get("corpus_run_ids"),
        "zero_leftover_prefix_processes": not report.get(
            "leftover_prefix_processes"
        ),
    }
    return {
        "trial": int(schedule["trial"]),
        "block": int(schedule["block"]),
        "role": role,
        "policy_seed": int(schedule["policy_seed"]),
        "artifact_dir": str(artifact_dir.resolve()),
        "report_sha256": _sha256(artifact_dir / "report.json"),
        "policy_state_sha256": _sha256(state_path),
        "physical_hits": int(completion["physical_hits"]),
        "proposals": int(metrics.get("proposals", 0)),
        "interventions": int(metrics.get("interventions", 0)),
        "latency_p95_ms": float(metrics["latency_p95_ms"]),
        "over_four_ms": int(metrics.get("over_four_ms", 0)),
        "deadline_misses": int(metrics.get("deadline_misses", 0)),
        "gates": gates,
        "passed": all(gates.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args(argv)
    contract_path = args.contract.resolve()
    contract_sha256 = _sha256(contract_path)
    if contract_sha256 not in ALLOWED_CANARY_CONTRACT_SHA256:
        raise ValueError("Generation-6 pilot contract drifted")
    contract = _object(contract_path)
    schedule = contract.get("schedule")
    if (
        not isinstance(schedule, list)
        or len(schedule) != 6
        or any(not isinstance(row, dict) for row in schedule)
        or [int(row.get("trial", -1)) for row in schedule]
        != list(range(len(schedule)))
        or any(row.get("role") not in ("incumbent", "candidate")
               for row in schedule)
        or any(int(row.get("stage", -1)) != 6 for row in schedule)
        or sum(row.get("role") == "incumbent" for row in schedule) != 3
        or sum(row.get("role") == "candidate" for row in schedule) != 3
    ):
        raise ValueError("Generation-6 pilot schedule is invalid")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY, text=True,
    )
    if dirty.strip():
        raise RuntimeError("Generation-6 pilot requires a clean tracked checkout")
    maximum = int(contract["resource_contract"]["maximum_logical_cpus"])
    effective = sorted(os.sched_getaffinity(0))[:maximum]
    os.sched_setaffinity(0, effective)
    if not effective or len(os.sched_getaffinity(0)) > 32:
        raise RuntimeError("Generation-6 pilot CPU affinity contract failed")

    environment = _bind_resources(contract)
    output_root = (REPOSITORY / str(environment["worker_root"])).parent
    worker = prepare_wine_worker(
        root=REPOSITORY / str(environment["worker_root"]),
        source_game_dir=REPOSITORY / str(environment["source_game_dir"]),
        worker=0,
        directory=str(environment["worker_directory"]),
        display=str(environment["display"]),
    )
    ledger_path = output_root / "pilot-ledger-v1.json"
    ledger = _object(ledger_path) if ledger_path.is_file() else {
        "schema": "autonomous-generation-6-stage6-pilot-ledger-v1",
        "contract_sha256": contract_sha256,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "runs": [],
        "status": "running",
    }
    if ledger.get("contract_sha256") != contract_sha256:
        raise ValueError("Generation-6 pilot ledger contract drifted")
    stored_rows = ledger.get("runs")
    if not isinstance(stored_rows, list) or len(stored_rows) > len(schedule):
        raise ValueError("Generation-6 pilot ledger rows are invalid")
    rows = list(stored_rows)
    for index, row in enumerate(rows):
        scheduled = schedule[index]
        if any(row.get(name) != scheduled.get(name) for name in (
            "trial", "block", "role", "policy_seed",
        )):
            raise ValueError("Generation-6 pilot ledger schedule drifted")
    remaining = [] if any(row.get("passed") is not True for row in rows) else (
        schedule[len(rows):]
    )
    for row in remaining:
        trial = int(row["trial"])
        role = str(row["role"])
        state_path = output_root / "states" / f"trial-{trial:02d}-{role}.json"
        _state(
            path=state_path, contract_path=contract_path, role=role,
            policy_seed=int(row["policy_seed"]),
        )
        artifact_dir = output_root / f"trial-{trial:02d}-{role}"
        report, run_dir = complete_run(
            artifact_dir=artifact_dir,
            worker=worker,
            stage=6,
            policy_plugin=POLICY_PLUGIN,
            policy_state=state_path,
            scorer=NATIVE_SCORER,
            rng_seed=None,
            corpus_root=None,
            game_cpu_list=environment.get("game_cpu_list"),
            controller_cpu_list=environment.get("controller_cpu_list"),
        )
        if run_dir is not None:
            raise RuntimeError("Generation-6 efficacy pilot created a corpus")
        audited = _audit_run(
            report=report, artifact_dir=artifact_dir, state_path=state_path,
            schedule=row, environment=environment,
        )
        rows.append(audited)
        ledger["runs"] = rows
        _atomic_json(ledger_path, ledger)
        print(json.dumps({"completed": audited}, sort_keys=True), flush=True)
        if audited["passed"] is not True:
            break

    decision = pilot_verdict(
        rows,
        expected_runs=len(schedule),
        required_exercised_candidate_stages=int(
            contract["decision_rule"]["required_exercised_candidate_stages"]
        ),
    )
    complete = len(rows) == len(schedule)
    result = {
        "schema": "autonomous-generation-6-stage6-pilot-result-v1",
        "evidence_eligible": complete and decision["verdict"] not in (
            "invalid", "inconclusive"
        ),
        "authorization_eligible": False,
        "contract_sha256": contract_sha256,
        "evaluation_mode": "normal-speed-natural-complete-stage-hit-continuation",
        "fixed_rng": False,
        "runs": rows,
        **decision,
    }
    if complete:
        result_path = output_root / "pilot-result-v1.json"
        if result_path.is_file() and _object(result_path) != result:
            raise ValueError("Generation-6 pilot result drifted")
        if not result_path.exists():
            _atomic_json(result_path, result)
        ledger["status"] = "complete"
        ledger["result_sha256"] = _sha256(result_path)
        _atomic_json(ledger_path, ledger)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if complete and decision["verdict"] != "invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
