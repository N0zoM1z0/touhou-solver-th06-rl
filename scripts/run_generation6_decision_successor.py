#!/usr/bin/env python3
"""Run the evidence-bound G6 decision successor canary and HIT evaluation."""

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

from scripts.export_generation6_round_policy import export_round_state  # noqa: E402
from scripts.run_generation5_wine import complete_run  # noqa: E402
from scripts.run_generation6_autonomous_round import (  # noqa: E402
    _audit_evaluation,
    _paired_verdict,
    _priority_arguments,
)
from scripts.run_generation6_wine_canary import (  # noqa: E402
    NATIVE_SCORER as WINE_SCORER,
    POLICY_PLUGIN,
)
from th06_rl.resource_control import enforce_training_cpu_affinity  # noqa: E402
from th06_rl.wine_workers import prepare_wine_worker  # noqa: E402


SCHEMA = "autonomous-generation-6-decision-gameplay-ledger-v1"
CONTRACT_SCHEMA = "autonomous-generation-6-decision-gameplay-contract-v1"


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
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _path(raw: object) -> Path:
    path = (REPOSITORY / str(raw)).resolve()
    if not path.is_relative_to(REPOSITORY):
        raise ValueError("decision gameplay path escapes the repository")
    return path


def validate_gameplay_contract(
    contract: dict[str, object], *, contract_sha256: str,
) -> dict[str, Path]:
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("authorization_eligible") is not False
        or contract.get("normal_speed") is not True
        or contract.get("natural_rng") is not True
        or contract.get("complete_stage_hit_continuation") is not True
        or contract.get("bomb") != "forbidden"
        or len(contract_sha256) != 64
    ):
        raise ValueError("decision gameplay contract semantics differ")
    implementation = str(contract.get("implementation_commit", ""))
    if len(implementation) != 40 or subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation, "HEAD"],
        cwd=REPOSITORY, check=False,
    ).returncode:
        raise ValueError("decision gameplay implementation is not an ancestor")
    bindings = contract.get("bindings")
    frozen = contract.get("frozen_inputs")
    canary = contract.get("canary")
    evaluation = contract.get("evaluation")
    environment = contract.get("environment")
    if not all(isinstance(value, dict) for value in (
        bindings, frozen, canary, evaluation, environment
    )):
        raise ValueError("decision gameplay contract sections are absent")
    paths = {}
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"decision gameplay binding is invalid: {name}")
        path = _path(binding["path"])
        if not path.is_file() or _sha256(path) != binding["sha256"]:
            raise ValueError(f"decision gameplay binding drifted: {name}")
        paths[str(name)] = path
    for name, expected in frozen.items():
        path = _path(name)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"decision gameplay runtime input drifted: {name}")
    successor = _object(paths["successor_contract"])
    candidate = _object(paths["candidate"])
    offline = _object(paths["offline_audit"])
    successor_sha = _sha256(paths["successor_contract"])
    candidate_sha = _sha256(paths["candidate"])
    registry_sha = _sha256(paths["registry"])
    if (
        successor.get("schema")
        != "autonomous-generation-6-decision-successor-contract-v2"
        or candidate.get("passed") is not True
        or candidate.get("autonomous_round_contract_sha256") != successor_sha
        or candidate.get("training_identity", {}).get("sha256") != registry_sha
        or offline.get("schema")
        != "autonomous-generation-6-decision-offline-smoke-v2"
        or offline.get("passed") is not True
        or offline.get("contract_sha256") != successor_sha
        or offline.get("candidate_sha256") != candidate_sha
        or offline.get("training_registry_sha256") != registry_sha
        or not isinstance(offline.get("gates"), dict)
        or not offline["gates"]
        or not all(value is True for value in offline["gates"].values())
    ):
        raise ValueError("decision gameplay offline authorization differs")
    canary_schedule = canary.get("schedule")
    evaluation_schedule = evaluation.get("schedule")
    if (
        not isinstance(canary_schedule, list) or len(canary_schedule) != 2
        or [row.get("trial") for row in canary_schedule] != [1, 2]
        or any(set(row) != {"trial", "stage", "policy_seed"}
               or row["stage"] != 4 for row in canary_schedule)
    ):
        raise ValueError("decision gameplay canary schedule differs")
    expected_pairs = {
        (block, role) for block in range(6)
        for role in ("incumbent", "candidate")
    }
    if (
        not isinstance(evaluation_schedule, list)
        or len(evaluation_schedule) != 12
        or [row.get("trial") for row in evaluation_schedule]
        != list(range(1, 13))
        or {(row.get("block"), row.get("role"))
            for row in evaluation_schedule} != expected_pairs
        or any(set(row) != {"trial", "block", "role", "policy_seed"}
               for row in evaluation_schedule)
    ):
        raise ValueError("decision gameplay evaluation schedule differs")
    seeds = [
        int(row["policy_seed"])
        for rows in (canary_schedule, evaluation_schedule) for row in rows
    ]
    if len(set(seeds)) != 14 or any(not 0 <= seed < 2**64 for seed in seeds):
        raise ValueError("decision gameplay policy seeds differ")
    game_cpus = set(range(0, 8))
    controller_cpus = set(range(8, 32))
    if (
        environment.get("game_cpu_list") != "0-7"
        or environment.get("controller_cpu_list") != "8-31"
        or environment.get("scheduler") != "SCHED_OTHER"
        or environment.get("game_nice") != -10
        or environment.get("controller_nice") != -10
        or not game_cpus | controller_cpus <= set(os.sched_getaffinity(0))
    ):
        raise ValueError("decision gameplay resource contract differs")
    return paths


def run(args: argparse.Namespace) -> int:
    contract_path = args.contract.resolve()
    contract = _object(contract_path)
    contract_sha = _sha256(contract_path)
    paths = validate_gameplay_contract(contract, contract_sha256=contract_sha)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY, text=True,
    )
    if dirty.strip():
        raise RuntimeError("decision gameplay requires a clean tracked checkout")
    enforce_training_cpu_affinity(32)
    root = args.output_root.resolve()
    ledger_path = root / "generation.json"
    if ledger_path.is_file():
        ledger = _object(ledger_path)
        if (
            ledger.get("schema") != SCHEMA
            or ledger.get("contract_sha256") != contract_sha
        ):
            raise ValueError("decision gameplay ledger drifted")
        if ledger.get("status") == "complete":
            print(json.dumps(ledger["decision"], sort_keys=True))
            return 0
    else:
        ledger = {
            "schema": SCHEMA,
            "contract_sha256": contract_sha,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
            ).strip(),
            "status": "canary",
            "offline_audit_sha256": _sha256(paths["offline_audit"]),
            "canary": [],
            "evaluation": [],
            "decision": None,
        }
        _atomic_json(ledger_path, ledger)
    environment = contract["environment"]
    worker = prepare_wine_worker(
        root=root / "workers",
        source_game_dir=REPOSITORY / str(environment["source_game_dir"]),
        worker=0,
        directory=str(environment["worker_directory"]),
        display=str(environment["display"]),
        source_inventory_sha256=str(environment["source_game_inventory_sha256"]),
    )
    states = root / "policy-states"
    for row in contract["canary"]["schedule"][len(ledger["canary"]):]:
        trial = int(row["trial"])
        state_path = states / f"canary-{trial:02d}.json"
        if not state_path.is_file():
            _atomic_json(state_path, export_round_state(
                candidate_path=paths["candidate"],
                smoke_path=paths["offline_audit"],
                registry_path=paths["registry"], scorer_path=WINE_SCORER,
                contract_path=paths["successor_contract"], mode="active",
                policy_seed=int(row["policy_seed"]),
            ))
        artifact = root / "canary" / f"trial-{trial:02d}"
        report, run_dir = complete_run(
            artifact_dir=artifact, worker=worker, stage=4,
            policy_plugin=POLICY_PLUGIN, policy_state=state_path,
            scorer=WINE_SCORER, rng_seed=None, corpus_root=None,
            game_cpu_list=str(environment["game_cpu_list"]),
            controller_cpu_list=str(environment["controller_cpu_list"]),
            **_priority_arguments(environment),
        )
        assert run_dir is None
        audited = _audit_evaluation(
            report=report, artifact=artifact, state_path=state_path,
            role="candidate", environment=environment,
        )
        audited["trial"] = trial
        ledger["canary"].append(audited)
        _atomic_json(ledger_path, ledger)
        print(json.dumps({"canary_completed": audited}, sort_keys=True), flush=True)
        if not audited["passed"]:
            break
    canary_valid = (
        len(ledger["canary"]) == 2
        and all(row["passed"] for row in ledger["canary"])
        and any(int(row["interventions"]) > 0 for row in ledger["canary"])
    )
    if not canary_valid:
        ledger.update({"status": "complete", "decision": {
            "verdict": "canary-rejected",
            "reason": "runtime gate or treatment exposure failed",
            "promotion_eligible": False,
        }})
        _atomic_json(ledger_path, ledger)
        print(json.dumps(ledger["decision"], sort_keys=True))
        return 0
    ledger["status"] = "evaluation"
    _atomic_json(ledger_path, ledger)
    for row in contract["evaluation"]["schedule"][len(ledger["evaluation"]):]:
        trial = int(row["trial"])
        role = str(row["role"])
        state_path = states / f"evaluation-{trial:02d}-{role}.json"
        if not state_path.is_file():
            _atomic_json(state_path, export_round_state(
                candidate_path=paths["candidate"],
                smoke_path=paths["offline_audit"],
                registry_path=paths["registry"], scorer_path=WINE_SCORER,
                contract_path=paths["successor_contract"],
                mode="active" if role == "candidate" else "shadow",
                policy_seed=int(row["policy_seed"]),
            ))
        artifact = root / "evaluation" / f"trial-{trial:02d}-{role}"
        report, run_dir = complete_run(
            artifact_dir=artifact, worker=worker, stage=6,
            policy_plugin=POLICY_PLUGIN, policy_state=state_path,
            scorer=WINE_SCORER, rng_seed=None, corpus_root=None,
            game_cpu_list=str(environment["game_cpu_list"]),
            controller_cpu_list=str(environment["controller_cpu_list"]),
            **_priority_arguments(environment),
        )
        assert run_dir is None
        audited = _audit_evaluation(
            report=report, artifact=artifact, state_path=state_path,
            role=role, environment=environment,
        )
        audited.update({"trial": trial, "block": int(row["block"])})
        ledger["evaluation"].append(audited)
        _atomic_json(ledger_path, ledger)
        print(json.dumps({"evaluation_completed": audited}, sort_keys=True), flush=True)
        if not audited["passed"]:
            break
    decision = _paired_verdict(ledger["evaluation"])
    ledger.update({"status": "complete", "decision": decision})
    _atomic_json(ledger_path, ledger)
    _atomic_json(root / "result.json", {
        "schema": "autonomous-generation-6-decision-gameplay-result-v1",
        "contract_sha256": contract_sha,
        "offline_audit_sha256": _sha256(paths["offline_audit"]),
        "canary": ledger["canary"],
        "evaluation": ledger["evaluation"],
        "decision": decision,
    })
    print(json.dumps(decision, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY / "artifacts/autonomous-generation-6-decision-gameplay",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
