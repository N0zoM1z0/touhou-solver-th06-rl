#!/usr/bin/env python3
"""Run the offline-only Stage 4 L1c optimization-timebox experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from scripts.gate_parallel_wine import (  # noqa: E402
    _atomic_json,
    _object,
    _repository_commit,
    _sha256,
    run_batch,
)
from scripts import run_l1b_stage4 as l1b  # noqa: E402
from scripts.run_l1_stage4 import (  # noqa: E402
    _relative,
    _repository_path,
    _require_clean_worktree,
    _work_event,
)


PREREG_SCHEMA = "th06-rl-l1c-stage4-bc-timebox-prereg-v1"
PLAN_SCHEMA = "th06-rl-l1c-stage4-bc-timebox-plan-v1"
RESULT_SCHEMA = "th06-rl-l1c-stage4-bc-timebox-result-v1"
PRIOR_MAXIMUM_UPDATES = 2000
MAXIMUM_UPDATES = 10000


def _without(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("Stage 4 L1c preregistration schema mismatch")
    for key in ("data", "fit", "gate", "paths", "sha256_bindings", "online_canary"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"Stage 4 L1c preregistration lacks {key}")

    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    prior_path = _repository_path(paths["source_l1b_preregistration"])
    if _sha256(prior_path) != bindings.get("source_l1b_preregistration"):
        raise ValueError("frozen L1b preregistration binding differs")
    prior = _object(prior_path)
    if prior.get("schema") != l1b.PREREG_SCHEMA:
        raise ValueError("frozen L1b preregistration schema differs")

    for key in ("auxiliary_targets", "comparators", "data", "online_canary"):
        if prereg.get(key) != prior.get(key):
            raise ValueError(f"L1c changed frozen L1b field: {key}")
    if prereg["fit"].get("maximum_updates") != MAXIMUM_UPDATES:
        raise ValueError("L1c maximum update timebox differs")
    if prior["fit"].get("maximum_updates") != PRIOR_MAXIMUM_UPDATES:
        raise ValueError("frozen L1b maximum update timebox differs")
    if _without(prereg["fit"], "maximum_updates") != _without(
        prior["fit"], "maximum_updates"
    ):
        raise ValueError("L1c changed more than the maximum update timebox")
    if _without(prereg["gate"], "timebox") != _without(prior["gate"], "timebox"):
        raise ValueError("L1c changed a frozen acceptance gate")
    if "10000" not in str(prereg["gate"].get("timebox")):
        raise ValueError("L1c timebox declaration is incomplete")

    shared_paths = (
        "bc_policy_plugin",
        "bc_training_module",
        "fit_cli",
        "source_collection_ledger",
        "source_corpus_root",
        "source_l1_model",
        "source_l1_result",
        "work_log_root",
    )
    for key in shared_paths:
        if paths.get(key) != prior["paths"].get(key):
            raise ValueError(f"L1c changed frozen L1b path: {key}")
    shared_bindings = (
        "bc_policy_plugin",
        "bc_training_module",
        "fit_cli",
        "source_collection_ledger",
        "source_l1_model",
        "source_l1_result",
    )
    for key in shared_bindings:
        if bindings.get(key) != prior["sha256_bindings"].get(key):
            raise ValueError(f"L1c changed frozen L1b binding: {key}")

    for key in (
        "artifact_root",
        "fit_artifact",
        "experiment_plan",
        "experiment_result",
        "work_log_root",
        "source_collection_ledger",
        "source_corpus_root",
        "source_l1_result",
        "source_l1_model",
        "source_l1b_preregistration",
        "source_l1b_result",
        "source_l1b_model",
        "bc_policy_plugin",
        "bc_training_module",
        "fit_cli",
        "l1b_runner",
    ):
        _repository_path(paths[key])
    for key in (
        "bc_policy_plugin",
        "bc_training_module",
        "fit_cli",
        "l1b_runner",
        "source_l1b_preregistration",
        "source_l1b_result",
        "source_l1b_model",
    ):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L1c input hash differs: {key}")

    prior_result = _object(_repository_path(paths["source_l1b_result"]))
    if (
        prior_result.get("complete") is not True
        or prior_result.get("decision")
        != "inconclusive-l1b-optimization-not-converged"
        or prior_result.get("optimization_converged") is not False
        or prior_result.get("online_canary") is not None
        or prior_result.get("fit_artifact_sha256") != bindings["source_l1b_model"]
    ):
        raise ValueError("L1c source is not the frozen inconclusive L1b result")
    return prereg


def fit_command(
    prereg: dict[str, Any],
    inventory: dict[int, dict[str, Any]],
    output: Path,
) -> list[str]:
    return l1b.fit_command(prereg, inventory, output)


def result_decision(model: dict[str, Any]) -> str:
    recorded = model.get("fit") or {}
    optimization = recorded.get("optimization") or {}
    if optimization.get("converged") is not True:
        return "inconclusive-l1c-optimization-not-converged"
    if recorded.get("learnability_gate_passed") is True:
        return "admit-stage4-bc-integration-canary"
    return "stop-l1c-linear-current-observation"


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    inventory = l1b.load_source_inventory(prereg)
    paths = prereg["paths"]
    artifact_root = _repository_path(paths["artifact_root"])
    model_path = _repository_path(paths["fit_artifact"])
    plan_path = _repository_path(paths["experiment_plan"])
    result_path = _repository_path(paths["experiment_result"])
    if result_path.is_file():
        result = _object(result_path)
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("preregistration_sha256") != _sha256(prereg_path)
        ):
            raise ValueError("completed Stage 4 L1c result differs")
        return result

    commit = _repository_commit()
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("Stage 4 L1c artifact root lacks its immutable plan")
        started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        work_log = _repository_path(paths["work_log_root"]) / (
            f"{started}-{prereg['experiment_id']}"
        )
        if work_log.exists():
            raise ValueError(f"work log already exists: {work_log}")
        _atomic_json(work_log / "session.json", {
            "schema": "th06-rl-work-log-session-v1",
            "experiment_id": prereg["experiment_id"],
            "repository_commit": commit,
            "preregistration_path": _relative(prereg_path),
            "preregistration_sha256": _sha256(prereg_path),
        })
        plan = {
            "schema": PLAN_SCHEMA,
            "experiment_id": prereg["experiment_id"],
            "repository_commit": commit,
            "preregistration_path": _relative(prereg_path),
            "preregistration_sha256": _sha256(prereg_path),
            "source_collection_ledger_sha256": prereg["sha256_bindings"][
                "source_collection_ledger"
            ],
            "source_l1b_result_sha256": prereg["sha256_bindings"][
                "source_l1b_result"
            ],
            "work_log_path": _relative(work_log),
            "fit_command": fit_command(prereg, inventory, model_path),
            "online_wine": False,
        }
        _atomic_json(plan_path, plan)
    expected_plan = {
        "schema": PLAN_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "source_collection_ledger_sha256": prereg["sha256_bindings"][
            "source_collection_ledger"
        ],
        "source_l1b_result_sha256": prereg["sha256_bindings"]["source_l1b_result"],
        "work_log_path": _relative(work_log),
        "fit_command": fit_command(prereg, inventory, model_path),
        "online_wine": False,
    }
    if plan != expected_plan:
        raise ValueError("Stage 4 L1c immutable plan differs")

    _work_event(
        work_log,
        "experiment-started-or-resumed",
        repository_commit=commit,
        plan_path=_relative(plan_path),
        plan_sha256=_sha256(plan_path),
    )
    if not model_path.is_file():
        fit_log = artifact_root / "logs" / "fit.log"
        if fit_log.exists():
            raise ValueError("partial L1c fit requires manual triage")
        _work_event(
            work_log,
            "fit-started",
            command=plan["fit_command"],
            launcher_log=_relative(fit_log),
        )
        run_batch([("fit", plan["fit_command"], fit_log)])
    model = l1b.validate_model(prereg, inventory, model_path, commit=commit)
    decision = result_decision(model)
    recorded_fit = model["fit"]
    _work_event(
        work_log,
        "fit-complete",
        fit_artifact=_relative(model_path),
        fit_artifact_sha256=_sha256(model_path),
        policy_id=model.get("policy_id"),
        decision=decision,
        fit_metrics=recorded_fit,
    )
    result = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "decision": decision,
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "experiment_plan_sha256": _sha256(plan_path),
        "source_collection_ledger_sha256": prereg["sha256_bindings"][
            "source_collection_ledger"
        ],
        "source_l1b_result_sha256": prereg["sha256_bindings"]["source_l1b_result"],
        "fit_artifact_path": _relative(model_path),
        "fit_artifact_sha256": _sha256(model_path),
        "policy_id": model.get("policy_id"),
        "optimization_converged": recorded_fit["optimization"]["converged"],
        "learnability_gate_passed": recorded_fit["learnability_gate_passed"],
        "fit_metrics": recorded_fit,
        "online_canary": None,
        "claim": (
            "optimization convergence and behavior learnability only; no online, "
            "HIT-reduction, or NMNB-improvement claim"
        ),
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "experiment-complete",
        result_path=_relative(result_path),
        result_sha256=_sha256(result_path),
        decision=decision,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=REPOSITORY / "experiments/l1c-stage4-bc-timebox-v1.json",
    )
    args = parser.parse_args(argv)
    result = run(args.preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
