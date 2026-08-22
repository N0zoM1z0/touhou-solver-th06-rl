#!/usr/bin/env python3
"""Run the offline-only Stage 4 L1e shared-action BC ablation."""

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
from th06_rl.policy_loader import ImmutablePolicy  # noqa: E402
from th06_rl.shared_action_bc_training import (  # noqa: E402
    INITIALIZATION_KIND,
    MODEL_KIND,
    OPTIMIZER_KIND,
)
from th06_rl.shared_action_features import (  # noqa: E402
    ACTION_FEATURE_NAMES,
    ACTION_FEATURE_SCHEMA,
)


PREREG_SCHEMA = "th06-rl-l1e-stage4-bc-shared-action-prereg-v1"
PLAN_SCHEMA = "th06-rl-l1e-stage4-bc-shared-action-plan-v1"
RESULT_SCHEMA = "th06-rl-l1e-stage4-bc-shared-action-result-v1"
DIAGNOSIS_SCHEMA = "th06-rl-l1d-target-contract-diagnosis-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("Stage 4 L1e preregistration schema mismatch")
    for key in ("data", "fit", "gate", "paths", "sha256_bindings", "online_canary"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"Stage 4 L1e preregistration lacks {key}")
    data = prereg["data"]
    fit = prereg["fit"]
    gate = prereg["gate"]
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    canary = prereg["online_canary"]
    if (
        data.get("reuse_without_mutation") is not True
        or data.get("source_experiment_id") != "l1-stage4-bc-v1"
        or data.get("train_episode_indices") != [0, 1, 3, 4, 6, 7, 9, 10]
        or data.get("validation_episode_indices") != [2, 5, 8, 11]
        or prereg.get("auxiliary_targets") != []
    ):
        raise ValueError("Stage 4 L1e data scope changed")
    expected_fit = {
        "decision_epoch_schema": "th06-rl-decision-epoch-v1",
        "feature_schema": "th06-rl-current-observation-features-v1",
        "action_feature_schema": ACTION_FEATURE_SCHEMA,
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "target_schema": "th06-rl-published-executed-action-target-v1",
        "model": MODEL_KIND,
        "optimizer": OPTIMIZER_KIND,
        "initialization": INITIALIZATION_KIND,
        "maximum_updates": 10_000,
        "minimum_updates": 100,
        "relative_gradient_l2_tolerance": 0.01,
        "learning_rate": 0.05,
        "l2": 0.0001,
        "seed": 0,
        "bootstrap_samples": 2_000,
        "exploration_probability": 0.2,
        "max_rows_per_split": 400_000,
    }
    if any(fit.get(key) != value for key, value in expected_fit.items()):
        raise ValueError("Stage 4 L1e fit contract changed")
    if (
        gate.get("primary_comparator") != "frozen-l1d-exact-propensity-score"
        or gate.get("maximum_validation_kl") != 0.1
        or gate.get("minimum_reactive_agreement") != 0.95
        or gate.get("minimum_final_tie_agreement") != 0.5
        or gate.get("minimum_validation_episodes") != 4
        or gate.get("optimization_convergence_required") is not True
        or canary.get("run_in_this_experiment") is not False
        or canary.get("run_only_after_joint_gate") is not True
    ):
        raise ValueError("Stage 4 L1e gate or online boundary changed")

    tracked = (
        "action_feature_module",
        "bc_policy_plugin",
        "bc_training_module",
        "fit_cli",
        "l1e_runner",
        "source_diagnosis_preregistration",
    )
    sources = (
        "source_collection_ledger",
        "source_diagnosis_result",
        "source_l1_model",
        "source_l1_result",
        "source_l1d_model",
        "source_l1d_result",
    )
    for key in (
        "artifact_root",
        "fit_artifact",
        "experiment_plan",
        "experiment_result",
        "work_log_root",
        "source_corpus_root",
        *tracked,
        *sources,
    ):
        _repository_path(paths[key])
    for key in (*tracked, *sources):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L1e input hash differs: {key}")

    diagnosis = _object(_repository_path(paths["source_diagnosis_result"]))
    attribution = diagnosis.get("attribution")
    if (
        diagnosis.get("schema") != DIAGNOSIS_SCHEMA
        or diagnosis.get("complete") is not True
        or not isinstance(attribution, dict)
        or attribution.get("propensity_pipeline_exact") is not True
        or attribution.get("premature_relative_gradient_stop_material") is not False
        or attribution.get("discarded_full_propensity_target_material") is not False
        or attribution.get("next_experiment")
        != "preregister-structured-current-observation-scorer"
        or diagnosis.get("source_l1d_model_sha256") != bindings["source_l1d_model"]
    ):
        raise ValueError("L1e source diagnosis did not select this ablation")
    return prereg


def fit_command(
    prereg: dict[str, Any],
    inventory: dict[int, dict[str, Any]],
    output: Path,
) -> list[str]:
    fit = prereg["fit"]
    gate = prereg["gate"]
    command = [sys.executable, str(_repository_path(prereg["paths"]["fit_cli"]))]
    for key, option in (
        ("train_episode_indices", "--train-run"),
        ("validation_episode_indices", "--validation-run"),
    ):
        for index in prereg["data"][key]:
            command.extend((option, str(REPOSITORY / inventory[int(index)]["run_dir"])))
    command.extend((
        "--l1d-comparator-state",
        str(_repository_path(prereg["paths"]["source_l1d_model"])),
        "--output", str(output),
        "--epochs", str(fit["maximum_updates"]),
        "--minimum-updates", str(fit["minimum_updates"]),
        "--relative-gradient-l2-tolerance", str(fit["relative_gradient_l2_tolerance"]),
        "--learning-rate", str(fit["learning_rate"]),
        "--l2", str(fit["l2"]),
        "--seed", str(fit["seed"]),
        "--bootstrap-samples", str(fit["bootstrap_samples"]),
        "--exploration-probability", str(fit["exploration_probability"]),
        "--maximum-validation-kl", str(gate["maximum_validation_kl"]),
        "--minimum-reactive-agreement", str(gate["minimum_reactive_agreement"]),
        "--minimum-final-tie-agreement", str(gate["minimum_final_tie_agreement"]),
        "--max-rows", str(fit["max_rows_per_split"]),
    ))
    return command


def validate_model(
    prereg: dict[str, Any],
    inventory: dict[int, dict[str, Any]],
    model_path: Path,
    *,
    commit: str,
) -> dict[str, Any]:
    model = _object(model_path)
    fit = prereg["fit"]
    gate = prereg["gate"]
    plugin = _repository_path(prereg["paths"]["bc_policy_plugin"])
    ImmutablePolicy(plugin, state_path=model_path)
    provenance = model.get("provenance")
    recorded = model.get("fit")
    model_inventory = model.get("inventory")
    architecture = model.get("model")
    initialization = model.get("initialization")
    if not all(isinstance(value, dict) for value in (
        provenance, recorded, model_inventory, architecture, initialization
    )):
        raise ValueError("L1e model lacks provenance, fit, or architecture")
    optimization = recorded.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("L1e model lacks optimization evidence")
    if (
        provenance.get("code_commit") != commit
        or provenance.get("policy_plugin_sha256") != _sha256(plugin)
        or provenance.get("frozen_l1d_comparator_sha256")
        != prereg["sha256_bindings"]["source_l1d_model"]
        or architecture.get("kind") != fit["model"]
        or initialization.get("kind") != fit["initialization"]
        or recorded.get("epochs") != fit["maximum_updates"]
        or recorded.get("learning_rate") != fit["learning_rate"]
        or recorded.get("l2") != fit["l2"]
        or recorded.get("seed") != fit["seed"]
        or recorded.get("bootstrap_samples") != fit["bootstrap_samples"]
        or recorded.get("exploration_probability") != fit["exploration_probability"]
        or recorded.get("maximum_validation_kl") != gate["maximum_validation_kl"]
        or recorded.get("minimum_reactive_agreement")
        != gate["minimum_reactive_agreement"]
        or recorded.get("minimum_final_tie_agreement")
        != gate["minimum_final_tie_agreement"]
        or optimization.get("kind") != fit["optimizer"]
        or optimization.get("maximum_updates") != fit["maximum_updates"]
        or optimization.get("minimum_updates") != fit["minimum_updates"]
        or optimization.get("relative_gradient_l2_tolerance")
        != fit["relative_gradient_l2_tolerance"]
    ):
        raise ValueError("L1e model differs from the preregistered fit")
    expected = {
        split: [
            {
                "episode_id": inventory[index]["run_id"],
                "run_sha256": inventory[index]["run_sha256"],
                "manifest_sha256": inventory[index]["manifest_sha256"],
            }
            for index in prereg["data"][f"{split}_episode_indices"]
        ]
        for split in ("train", "validation")
    }
    observed = {
        split: [
            {
                "episode_id": row.get("episode_id"),
                "run_sha256": row.get("run_sha256"),
                "manifest_sha256": row.get("manifest_sha256"),
            }
            for row in model_inventory.get(split, ())
        ]
        for split in ("train", "validation")
    }
    if observed != expected:
        raise ValueError("L1e whole-episode inventory differs")
    return model


def result_decision(model: dict[str, Any]) -> str:
    recorded = model.get("fit") or {}
    optimization = recorded.get("optimization") or {}
    if optimization.get("converged") is not True:
        return "inconclusive-l1e-shared-action-optimization-not-converged"
    if recorded.get("learnability_gate_passed") is True:
        return "admit-stage4-shared-action-bc-integration-canary"
    return "stop-l1e-shared-action-current-observation"


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
            raise ValueError("completed Stage 4 L1e result differs")
        return result

    commit = _repository_commit()
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("Stage 4 L1e artifact root lacks its immutable plan")
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
            "source_collection_ledger_sha256": prereg["sha256_bindings"]["source_collection_ledger"],
            "source_diagnosis_result_sha256": prereg["sha256_bindings"]["source_diagnosis_result"],
            "source_l1d_result_sha256": prereg["sha256_bindings"]["source_l1d_result"],
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
        "source_collection_ledger_sha256": prereg["sha256_bindings"]["source_collection_ledger"],
        "source_diagnosis_result_sha256": prereg["sha256_bindings"]["source_diagnosis_result"],
        "source_l1d_result_sha256": prereg["sha256_bindings"]["source_l1d_result"],
        "work_log_path": _relative(work_log),
        "fit_command": fit_command(prereg, inventory, model_path),
        "online_wine": False,
    }
    if plan != expected_plan:
        raise ValueError("Stage 4 L1e immutable plan differs")

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
            raise ValueError("partial L1e fit requires manual triage")
        _work_event(
            work_log,
            "fit-started",
            command=plan["fit_command"],
            launcher_log=_relative(fit_log),
        )
        run_batch([("fit", plan["fit_command"], fit_log)])
    model = validate_model(prereg, inventory, model_path, commit=commit)
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
        "source_collection_ledger_sha256": prereg["sha256_bindings"]["source_collection_ledger"],
        "source_diagnosis_result_sha256": prereg["sha256_bindings"]["source_diagnosis_result"],
        "source_l1d_result_sha256": prereg["sha256_bindings"]["source_l1d_result"],
        "fit_artifact_path": _relative(model_path),
        "fit_artifact_sha256": _sha256(model_path),
        "policy_id": model.get("policy_id"),
        "optimization_converged": recorded_fit["optimization"]["converged"],
        "learnability_gate_passed": recorded_fit["learnability_gate_passed"],
        "fit_metrics": recorded_fit,
        "online_canary": None,
        "claim": (
            "shared current-observation representation learnability only; no "
            "online, HIT-reduction, value-learning, or NMNB-improvement claim"
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
        default=REPOSITORY / "experiments/l1e-stage4-bc-shared-action-v1.json",
    )
    args = parser.parse_args(argv)
    result = run(args.preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
