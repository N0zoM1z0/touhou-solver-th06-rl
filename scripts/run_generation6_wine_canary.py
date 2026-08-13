#!/usr/bin/env python3
"""Run and audit the one frozen Generation-6 original-Wine wiring canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.export_generation6_policy import export_state  # noqa: E402
from scripts.run_generation5_wine import complete_run  # noqa: E402
from th06_rl.offline import ACTION_NAMES  # noqa: E402
from th06_rl.policies.autonomous_iql_actor import (  # noqa: E402
    ALLOWED_CANARY_CONTRACT_SHA256,
    POLICY_NAME,
)
from th06_rl.wine_workers import prepare_wine_worker  # noqa: E402


CANDIDATE = REPOSITORY / "artifacts/autonomous-generation-6-candidate/candidate-v1.json"
QUALIFICATION = REPOSITORY / "artifacts/autonomous-generation-6-qualification/qualification-v1.json"
DEPLOYABLE_AUDIT = REPOSITORY / "artifacts/autonomous-generation-6-qualification/deployable-target-audit-v1.json"
POLICY_PLUGIN = REPOSITORY / "src/th06_rl/policies/autonomous_iql_actor.py"
NATIVE_SCORER = REPOSITORY / "build/native-win32-fully-static/libth06_rl_ranker.dll"
INFRA_EVENTS = frozenset({
    "background-reactivated", "capture-gap-fail-close", "capture-incoherent",
    "continuous-fail-close", "system-memory-stop", "system-memory-unavailable",
})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _bind_resources(contract: dict[str, object]) -> dict[str, object]:
    environment = contract.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("Generation-6 canary environment is absent")
    paths = {
        "wine_sha256": Path(str(environment["wine"])),
        "retail_executable_sha256": (
            REPOSITORY / str(environment["source_game_dir"]) / "東方紅魔郷.exe"
        ),
        "full_unlock_score_sha256": (
            REPOSITORY / "reference/th06-game-original/full-unlock-score.dat"
        ),
        "windows_python_sha256": (
            REPOSITORY / "reference/tools/windows-python-3.11.9-embed-win32/python.exe"
        ),
        "native_controller_sha256": (
            REPOSITORY / "build/native-win32-fully-static/libth06_rl_native.dll"
        ),
        "native_policy_scorer_sha256": NATIVE_SCORER,
    }
    for field, path in paths.items():
        if not path.is_file() or _sha256(path) != environment[field]:
            raise ValueError(f"Generation-6 canary input drifted: {field}")
    version = subprocess.check_output(
        [str(paths["wine_sha256"]), "--version"], text=True
    ).strip()
    if version != environment["wine_version"]:
        raise ValueError("Generation-6 Wine version drifted")
    return environment


def _last_policy_status(trace_path: Path) -> dict[str, object]:
    last = None
    with trace_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value = json.loads(line)
            policy = value.get("policy") if isinstance(value, dict) else None
            if isinstance(policy, dict) and isinstance(
                policy.get("metrics"), dict
            ):
                last = policy
    if not isinstance(last, dict):
        raise ValueError("Generation-6 canary has no policy status")
    return last


def _audit(
    *, report: dict[str, object], artifact_dir: Path,
    state_path: Path, schedule: dict[str, object], contract_path: Path,
    environment: dict[str, object],
) -> dict[str, object]:
    trace = report.get("trace")
    completion = report.get("controller_completion")
    if not isinstance(trace, dict) or not isinstance(completion, dict):
        raise ValueError("Generation-6 Wine report lacks completion evidence")
    status = _last_policy_status(artifact_dir / "trace.jsonl")
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Generation-6 Wine report lacks policy metrics")
    selected = metrics.get("selected")
    selected_actions = set(selected) if isinstance(selected, dict) else set()
    events = trace.get("event_counts")
    gates = {
        "expected_evaluation_mode": report.get("evaluation_mode")
        == "hit-continuation-benchmark",
        "natural_rng": report.get("diagnostic_rng_seed") is None,
        "expected_stage": completion.get("practice_stage") == schedule["stage"],
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
            status.get("policy_id") == f"{POLICY_NAME}-active"
            and status.get("reload_failures") == 0
            and status.get("last_error") is None
        ),
        "native_safe_actions_only": bool(selected_actions)
        and selected_actions <= set(ACTION_NAMES),
        "zero_bomb": "bomb" not in selected_actions,
        "proposal_exercised": int(metrics.get("proposals", 0)) > 0,
        "intervention_exercised": int(metrics.get("interventions", 0)) > 0,
        "intervention_budget_respected": (
            int(metrics.get("interventions", -1))
            <= int(metrics.get("intervention_budget", -2))
            <= 64
        ),
        "zero_budget_abstentions": metrics.get("budget_abstentions") == 0,
        "online_p95_below_4_ms": (
            isinstance(metrics.get("latency_p95_ms"), (int, float))
            and float(metrics["latency_p95_ms"]) < 4.0
        ),
        "zero_policy_deadline_misses": metrics.get("deadline_misses") == 0,
        "no_corpus_created": not trace.get("corpus_run_ids"),
        "zero_leftover_prefix_processes": not report.get("leftover_prefix_processes"),
    }
    return {
        "schema": "autonomous-generation-6-wine-canary-result-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "contract_sha256": _sha256(contract_path),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "schedule": schedule,
        "report": str((artifact_dir / "report.json").resolve()),
        "report_sha256": _sha256(artifact_dir / "report.json"),
        "policy_plugin_sha256": _sha256(POLICY_PLUGIN),
        "policy_state_sha256": _sha256(state_path),
        "native_policy_scorer_sha256": _sha256(NATIVE_SCORER),
        "physical_hits": completion.get("physical_hits"),
        "policy_metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "interpretation": (
            "wiring canary only; pass permits freezing a separate Stage-6 efficacy pilot"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path,
        default=REPOSITORY / "config/autonomous_generation6_wine_canary_v3.json",
    )
    args = parser.parse_args(argv)
    contract_path = args.contract.resolve()
    if _sha256(contract_path) not in ALLOWED_CANARY_CONTRACT_SHA256:
        raise ValueError("Generation-6 canary contract drifted")
    contract = _object(contract_path)
    environment = _bind_resources(contract)
    output_root = (REPOSITORY / str(environment["worker_root"])).parent
    schedule_rows = contract.get("schedule")
    if not isinstance(schedule_rows, list) or len(schedule_rows) != 1:
        raise ValueError("Generation-6 canary schedule is not singular")
    schedule = schedule_rows[0]
    if not isinstance(schedule, dict) or schedule.get("role") != "candidate":
        raise ValueError("Generation-6 canary schedule is invalid")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY, text=True,
    )
    if dirty.strip():
        raise RuntimeError("Generation-6 canary requires a clean tracked checkout")
    available = sorted(os.sched_getaffinity(0))
    maximum = int(contract["resource_contract"]["maximum_logical_cpus"])
    effective = available[:maximum]
    os.sched_setaffinity(0, effective)
    if not effective or len(os.sched_getaffinity(0)) > 32:
        raise RuntimeError("Generation-6 CPU affinity contract failed")

    worker = prepare_wine_worker(
        root=REPOSITORY / str(environment["worker_root"]),
        source_game_dir=REPOSITORY / str(environment["source_game_dir"]),
        worker=0,
        directory=str(environment["worker_directory"]),
        display=str(environment["display"]),
    )
    state_path = output_root / "policy-active.json"
    expected_state = export_state(
        candidate_path=CANDIDATE,
        qualification_path=QUALIFICATION,
        deployable_audit_path=DEPLOYABLE_AUDIT,
        native_scorer_path=NATIVE_SCORER,
        canary_contract_path=contract_path,
        mode="active",
        policy_seed=int(schedule["policy_seed"]),
    )
    if state_path.is_file():
        if _object(state_path) != expected_state:
            raise ValueError("Generation-6 canary state drifted")
    else:
        _atomic_json(state_path, expected_state)

    artifact_dir = output_root / str(schedule["id"])
    report, _run_dir = complete_run(
        artifact_dir=artifact_dir,
        worker=worker,
        stage=int(schedule["stage"]),
        policy_plugin=POLICY_PLUGIN,
        policy_state=state_path,
        scorer=NATIVE_SCORER,
        rng_seed=None,
        corpus_root=None,
        game_cpu_list=environment.get("game_cpu_list"),
        controller_cpu_list=environment.get("controller_cpu_list"),
    )
    result = _audit(
        report=report, artifact_dir=artifact_dir,
        state_path=state_path, schedule=schedule,
        contract_path=contract_path, environment=environment,
    )
    result_path = output_root / "canary-result-v1.json"
    if result_path.is_file() and _object(result_path) != result:
        raise ValueError("Generation-6 canary result drifted")
    if not result_path.exists():
        _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
