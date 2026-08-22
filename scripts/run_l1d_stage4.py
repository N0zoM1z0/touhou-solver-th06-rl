#!/usr/bin/env python3
"""Run the offline-only Stage 4 L1d small-MLP representation ablation."""

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
from th06_rl.bc_training import CALIBRATION_SCHEMA  # noqa: E402
from th06_rl.mlp_bc_training import INITIALIZATION_KIND, RNG_KIND  # noqa: E402
from th06_rl.policies.small_mlp_behavior_clone import HIDDEN_WIDTH  # noqa: E402
from th06_rl.policy_loader import ImmutablePolicy  # noqa: E402


PREREG_SCHEMA = "th06-rl-l1d-stage4-bc-mlp-prereg-v1"
PLAN_SCHEMA = "th06-rl-l1d-stage4-bc-mlp-plan-v1"
RESULT_SCHEMA = "th06-rl-l1d-stage4-bc-mlp-result-v1"
DIAGNOSIS_SCHEMA = "th06-rl-l1c-residual-diagnosis-result-v1"
L1C_RESULT_SCHEMA = "th06-rl-l1c-stage4-bc-timebox-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("Stage 4 L1d preregistration schema mismatch")
    for key in (
        "data",
        "fit",
        "gate",
        "paths",
        "sha256_bindings",
        "online_canary",
    ):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"Stage 4 L1d preregistration lacks {key}")
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
        raise ValueError("Stage 4 L1d data or target scope changed")
    if (
        fit.get("decision_epoch_schema") != "th06-rl-decision-epoch-v1"
        or fit.get("feature_schema")
        != "th06-rl-current-observation-features-v1"
        or fit.get("target_schema")
        != "th06-rl-published-executed-action-target-v1"
        or fit.get("model") != "masked-one-hidden-relu-softmax"
        or fit.get("hidden_width") != HIDDEN_WIDTH
        or fit.get("optimizer") != "full-batch-gradient-descent"
        or fit.get("initialization") != INITIALIZATION_KIND
        or fit.get("initialization_rng") != RNG_KIND
        or fit.get("calibration_schema") != CALIBRATION_SCHEMA
        or fit.get("maximum_updates") != 10_000
        or fit.get("minimum_updates") != 100
        or fit.get("relative_gradient_l2_tolerance") != 0.01
        or fit.get("learning_rate") != 0.05
        or fit.get("l2") != 0.0001
        or fit.get("seed") != 0
        or fit.get("bootstrap_samples") != 2_000
        or fit.get("calibration_tolerance") != 0.02
        or fit.get("max_rows_per_split") != 400_000
    ):
        raise ValueError("Stage 4 L1d fit contract changed")
    if (
        gate.get("primary_comparator") != "frozen-converged-l1c-linear-bc"
        or gate.get("optimization_convergence_required") is not True
        or gate.get("minimum_validation_episodes") != 4
        or canary.get("run_in_this_experiment") is not False
        or canary.get("run_only_after_joint_gate") is not True
    ):
        raise ValueError("Stage 4 L1d gate or online boundary changed")

    tracked_bindings = (
        "bc_policy_plugin",
        "bc_training_module",
        "fit_cli",
        "l1d_runner",
        "source_diagnosis_preregistration",
        "source_l1c_preregistration",
    )
    source_bindings = (
        "source_collection_ledger",
        "source_diagnosis_result",
        "source_l1_model",
        "source_l1_result",
        "source_l1c_model",
        "source_l1c_result",
    )
    for key in (
        "artifact_root",
        "fit_artifact",
        "experiment_plan",
        "experiment_result",
        "work_log_root",
        "source_corpus_root",
        *(key for key in tracked_bindings),
        *(key for key in source_bindings),
    ):
        _repository_path(paths[key])
    for key in (*tracked_bindings, *source_bindings):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L1d input hash differs: {key}")

    diagnosis = _object(_repository_path(paths["source_diagnosis_result"]))
    selection = diagnosis.get("selection")
    if (
        diagnosis.get("schema") != DIAGNOSIS_SCHEMA
        or diagnosis.get("complete") is not True
        or not isinstance(selection, dict)
        or selection.get("ablation") != "small-current-observation-mlp"
        or selection.get("uses_transformed_validation_metrics") is not False
        or diagnosis.get("scaled_validation") is not None
        or diagnosis.get("source_l1c_model_sha256")
        != bindings["source_l1c_model"]
    ):
        raise ValueError("L1d source diagnosis did not select the frozen MLP branch")
    source_l1c = _object(_repository_path(paths["source_l1c_result"]))
    if (
        source_l1c.get("schema") != L1C_RESULT_SCHEMA
        or source_l1c.get("complete") is not True
        or source_l1c.get("decision") != "stop-l1c-linear-current-observation"
        or source_l1c.get("optimization_converged") is not True
        or source_l1c.get("learnability_gate_passed") is not False
        or source_l1c.get("fit_artifact_sha256")
        != bindings["source_l1c_model"]
        or source_l1c.get("online_canary") is not None
    ):
        raise ValueError("L1d source is not the frozen negative converged L1c fit")

    l1c_prereg = _object(_repository_path(paths["source_l1c_preregistration"]))
    if data != l1c_prereg.get("data"):
        raise ValueError("L1d changed the frozen L1c episode split")
    for key in (
        "bootstrap_samples",
        "calibration_schema",
        "calibration_tolerance",
        "decision_epoch_schema",
        "feature_schema",
        "l2",
        "learning_rate",
        "max_rows_per_split",
        "maximum_updates",
        "minimum_updates",
        "optimizer",
        "relative_gradient_l2_tolerance",
        "seed",
        "target_schema",
        "validation_use",
    ):
        if fit.get(key) != l1c_prereg.get("fit", {}).get(key):
            raise ValueError(f"L1d changed frozen L1c fit field: {key}")
    return prereg


def fit_command(
    prereg: dict[str, Any],
    inventory: dict[int, dict[str, Any]],
    output: Path,
) -> list[str]:
    fit = prereg["fit"]
    command = [sys.executable, str(_repository_path(prereg["paths"]["fit_cli"]))]
    for key, option in (
        ("train_episode_indices", "--train-run"),
        ("validation_episode_indices", "--validation-run"),
    ):
        for index in prereg["data"][key]:
            command.extend((
                option,
                str(REPOSITORY / str(inventory[int(index)]["run_dir"])),
            ))
    command.extend((
        "--linear-comparator-state",
        str(_repository_path(prereg["paths"]["source_l1c_model"])),
        "--output", str(output),
        "--epochs", str(fit["maximum_updates"]),
        "--minimum-updates", str(fit["minimum_updates"]),
        "--relative-gradient-l2-tolerance",
        str(fit["relative_gradient_l2_tolerance"]),
        "--learning-rate", str(fit["learning_rate"]),
        "--l2", str(fit["l2"]),
        "--seed", str(fit["seed"]),
        "--bootstrap-samples", str(fit["bootstrap_samples"]),
        "--calibration-tolerance", str(fit["calibration_tolerance"]),
        "--hidden-width", str(fit["hidden_width"]),
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
    plugin = _repository_path(prereg["paths"]["bc_policy_plugin"])
    ImmutablePolicy(plugin, state_path=model_path)
    provenance = model.get("provenance")
    recorded = model.get("fit")
    model_inventory = model.get("inventory")
    initialization = model.get("initialization")
    architecture = model.get("model")
    if not all(isinstance(value, dict) for value in (
        provenance,
        recorded,
        model_inventory,
        initialization,
        architecture,
    )):
        raise ValueError("L1d model lacks provenance, fit, or model evidence")
    optimization = recorded.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("L1d model lacks optimization evidence")
    if (
        provenance.get("code_commit") != commit
        or provenance.get("policy_plugin_sha256") != _sha256(plugin)
        or provenance.get("frozen_linear_comparator_sha256")
        != prereg["sha256_bindings"]["source_l1c_model"]
        or architecture.get("kind") != fit["model"]
        or architecture.get("hidden_width") != fit["hidden_width"]
        or initialization.get("kind") != fit["initialization"]
        or initialization.get("rng") != fit["initialization_rng"]
        or initialization.get("seed") != fit["seed"]
        or recorded.get("epochs") != fit["maximum_updates"]
        or recorded.get("learning_rate") != fit["learning_rate"]
        or recorded.get("l2") != fit["l2"]
        or recorded.get("seed") != fit["seed"]
        or recorded.get("bootstrap_samples") != fit["bootstrap_samples"]
        or recorded.get("calibration_tolerance") != fit["calibration_tolerance"]
        or recorded.get("calibration_schema") != fit["calibration_schema"]
        or optimization.get("kind") != fit["optimizer"]
        or optimization.get("maximum_updates") != fit["maximum_updates"]
        or optimization.get("minimum_updates") != fit["minimum_updates"]
        or optimization.get("relative_gradient_l2_tolerance")
        != fit["relative_gradient_l2_tolerance"]
    ):
        raise ValueError("L1d model differs from the preregistered fit")

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
        raise ValueError("L1d model whole-episode inventory differs")
    return model


def result_decision(model: dict[str, Any]) -> str:
    recorded = model.get("fit") or {}
    optimization = recorded.get("optimization") or {}
    if optimization.get("converged") is not True:
        return "inconclusive-l1d-mlp-optimization-not-converged"
    if recorded.get("learnability_gate_passed") is True:
        return "admit-stage4-mlp-bc-integration-canary"
    return "stop-l1d-small-current-observation-mlp"


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
            raise ValueError("completed Stage 4 L1d result differs")
        return result

    commit = _repository_commit()
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("Stage 4 L1d artifact root lacks its immutable plan")
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
            "source_diagnosis_result_sha256": prereg["sha256_bindings"][
                "source_diagnosis_result"
            ],
            "source_l1c_result_sha256": prereg["sha256_bindings"][
                "source_l1c_result"
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
        "source_diagnosis_result_sha256": prereg["sha256_bindings"][
            "source_diagnosis_result"
        ],
        "source_l1c_result_sha256": prereg["sha256_bindings"]["source_l1c_result"],
        "work_log_path": _relative(work_log),
        "fit_command": fit_command(prereg, inventory, model_path),
        "online_wine": False,
    }
    if plan != expected_plan:
        raise ValueError("Stage 4 L1d immutable plan differs")

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
            raise ValueError("partial L1d fit requires manual triage")
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
        "source_collection_ledger_sha256": prereg["sha256_bindings"][
            "source_collection_ledger"
        ],
        "source_diagnosis_result_sha256": prereg["sha256_bindings"][
            "source_diagnosis_result"
        ],
        "source_l1c_result_sha256": prereg["sha256_bindings"]["source_l1c_result"],
        "fit_artifact_path": _relative(model_path),
        "fit_artifact_sha256": _sha256(model_path),
        "policy_id": model.get("policy_id"),
        "optimization_converged": recorded_fit["optimization"]["converged"],
        "direct_l1c_nll_gate_passed": recorded_fit[
            "direct_l1c_nll_gate_passed"
        ],
        "calibration_gate_passed": recorded_fit["calibration_gate_passed"],
        "learnability_gate_passed": recorded_fit["learnability_gate_passed"],
        "fit_metrics": recorded_fit,
        "online_canary": None,
        "claim": (
            "small-current-observation representation learnability only; no "
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
        default=REPOSITORY / "experiments/l1d-stage4-bc-mlp-v1.json",
    )
    args = parser.parse_args(argv)
    result = run(args.preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
