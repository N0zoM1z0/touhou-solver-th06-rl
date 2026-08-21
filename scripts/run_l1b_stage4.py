#!/usr/bin/env python3
"""Run the offline-only Stage 4 L1b convergence follow-up on frozen L1 data."""

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
from scripts.run_l1_stage4 import (  # noqa: E402
    _relative,
    _repository_path,
    _require_clean_worktree,
    _work_event,
)
from th06_rl.bc_training import CALIBRATION_SCHEMA  # noqa: E402
from th06_rl.policy_loader import ImmutablePolicy  # noqa: E402


PREREG_SCHEMA = "th06-rl-l1b-stage4-bc-convergence-prereg-v1"
PLAN_SCHEMA = "th06-rl-l1b-stage4-bc-convergence-plan-v1"
RESULT_SCHEMA = "th06-rl-l1b-stage4-bc-convergence-result-v1"
SOURCE_COLLECTION_SCHEMA = "th06-rl-l1-stage4-collection-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("Stage 4 L1b preregistration schema mismatch")
    data = prereg.get("data")
    fit = prereg.get("fit")
    gate = prereg.get("gate")
    paths = prereg.get("paths")
    bindings = prereg.get("sha256_bindings")
    canary = prereg.get("online_canary")
    if not all(
        isinstance(value, dict)
        for value in (data, fit, gate, paths, bindings, canary)
    ):
        raise ValueError("Stage 4 L1b preregistration is incomplete")
    assert isinstance(data, dict)
    assert isinstance(fit, dict)
    assert isinstance(gate, dict)
    assert isinstance(paths, dict)
    assert isinstance(bindings, dict)
    assert isinstance(canary, dict)

    train = data.get("train_episode_indices")
    validation = data.get("validation_episode_indices")
    if train != [0, 1, 3, 4, 6, 7, 9, 10] or validation != [2, 5, 8, 11]:
        raise ValueError("Stage 4 L1b whole-episode split changed")
    if (
        data.get("reuse_without_mutation") is not True
        or data.get("source_experiment_id") != "l1-stage4-bc-v1"
        or prereg.get("auxiliary_targets") != []
    ):
        raise ValueError("Stage 4 L1b data or target scope changed")

    maximum_updates = fit.get("maximum_updates")
    minimum_updates = fit.get("minimum_updates")
    tolerance = fit.get("relative_gradient_l2_tolerance")
    if (
        fit.get("decision_epoch_schema") != "th06-rl-decision-epoch-v1"
        or fit.get("feature_schema")
        != "th06-rl-current-observation-features-v1"
        or fit.get("target_schema")
        != "th06-rl-published-executed-action-target-v1"
        or fit.get("model") != "masked-linear-softmax"
        or fit.get("optimizer") != "full-batch-gradient-descent"
        or fit.get("initialization") != "all-zero-weights-and-biases"
        or fit.get("calibration_schema") != CALIBRATION_SCHEMA
        or not isinstance(maximum_updates, int)
        or not isinstance(minimum_updates, int)
        or not 0 < minimum_updates <= maximum_updates
        or not isinstance(tolerance, (int, float))
        or not 0.0 < float(tolerance) < 1.0
    ):
        raise ValueError("Stage 4 L1b fit contract changed")
    if (
        gate.get("optimization_convergence_required") is not True
        or gate.get("minimum_validation_episodes") != 4
        or canary.get("run_in_this_experiment") is not False
        or canary.get("run_only_after_joint_gate") is not True
    ):
        raise ValueError("Stage 4 L1b gate or online boundary changed")

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
        "bc_policy_plugin",
        "bc_training_module",
        "fit_cli",
    ):
        _repository_path(paths[key])
    for key in ("bc_policy_plugin", "bc_training_module", "fit_cli"):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered tracked input hash differs: {key}")
    return prereg


def load_source_inventory(prereg: dict[str, Any]) -> dict[int, dict[str, Any]]:
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    for path_key, binding_key in (
        ("source_collection_ledger", "source_collection_ledger"),
        ("source_l1_result", "source_l1_result"),
        ("source_l1_model", "source_l1_model"),
    ):
        source = _repository_path(paths[path_key])
        if not source.is_file() or _sha256(source) != bindings[binding_key]:
            raise ValueError(f"frozen L1 source artifact differs: {path_key}")

    source_result = _object(_repository_path(paths["source_l1_result"]))
    if (
        source_result.get("decision") != "stop-l1-bc-learnability"
        or source_result.get("learnability_gate_passed") is not False
    ):
        raise ValueError("L1b source result is not the frozen negative L1 result")

    ledger = _object(_repository_path(paths["source_collection_ledger"]))
    rows = ledger.get("episodes")
    if (
        ledger.get("schema") != SOURCE_COLLECTION_SCHEMA
        or ledger.get("complete") is not True
        or not isinstance(rows, list)
        or len(rows) != 12
    ):
        raise ValueError("L1b source collection ledger is incomplete")
    by_index: dict[int, dict[str, Any]] = {}
    corpus_root = _repository_path(paths["source_corpus_root"])
    train = set(prereg["data"]["train_episode_indices"])
    validation = set(prereg["data"]["validation_episode_indices"])
    for unresolved in rows:
        if not isinstance(unresolved, dict):
            raise ValueError("L1b source episode row is malformed")
        row = unresolved
        index = int(row.get("index", -1))
        expected_split = "train" if index in train else "validation"
        if index in by_index or index not in train | validation:
            raise ValueError("L1b source episode indices changed")
        if row.get("split") != expected_split:
            raise ValueError("L1b source episode split differs")
        run_dir = _repository_path(row.get("run_dir"))
        try:
            run_dir.relative_to(corpus_root)
        except ValueError as error:
            raise ValueError("L1b source run escaped its frozen corpus root") from error
        for name, digest_key in (
            ("run.json", "run_sha256"),
            ("manifest.json", "manifest_sha256"),
        ):
            artifact = run_dir / name
            if not artifact.is_file() or _sha256(artifact) != row.get(digest_key):
                raise ValueError(f"L1b source episode binding differs: {artifact}")
        by_index[index] = row
    if sorted(by_index) != list(range(12)):
        raise ValueError("L1b source collection inventory changed")
    return by_index


def fit_command(
    prereg: dict[str, Any],
    inventory: dict[int, dict[str, Any]],
    output: Path,
) -> list[str]:
    fit = prereg["fit"]
    command = [sys.executable, str(REPOSITORY / "scripts/fit_behavior_clone.py")]
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
    if not all(
        isinstance(value, dict)
        for value in (provenance, recorded, model_inventory)
    ):
        raise ValueError("L1b model lacks provenance, fit, or inventory")
    assert isinstance(provenance, dict)
    assert isinstance(recorded, dict)
    assert isinstance(model_inventory, dict)
    optimization = recorded.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("L1b model lacks optimization evidence")
    if (
        provenance.get("code_commit") != commit
        or provenance.get("policy_plugin_sha256") != _sha256(plugin)
        or recorded.get("epochs") != fit["maximum_updates"]
        or recorded.get("learning_rate") != fit["learning_rate"]
        or recorded.get("l2") != fit["l2"]
        or recorded.get("seed") != fit["seed"]
        or recorded.get("bootstrap_samples") != fit["bootstrap_samples"]
        or recorded.get("calibration_tolerance") != fit["calibration_tolerance"]
        or recorded.get("calibration_schema") != fit["calibration_schema"]
        or optimization.get("maximum_updates") != fit["maximum_updates"]
        or optimization.get("minimum_updates") != fit["minimum_updates"]
        or optimization.get("relative_gradient_l2_tolerance")
        != fit["relative_gradient_l2_tolerance"]
    ):
        raise ValueError("L1b model differs from the preregistered fit")

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
        raise ValueError("L1b model whole-episode inventory differs")
    return model


def result_decision(model: dict[str, Any]) -> str:
    recorded = model.get("fit") or {}
    optimization = recorded.get("optimization") or {}
    if optimization.get("converged") is not True:
        return "inconclusive-l1b-optimization-not-converged"
    if recorded.get("learnability_gate_passed") is True:
        return "admit-stage4-bc-integration-canary"
    return "stop-l1b-linear-current-observation"


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    inventory = load_source_inventory(prereg)
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
            raise ValueError("completed Stage 4 L1b result differs")
        return result

    commit = _repository_commit()
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("Stage 4 L1b artifact root lacks its immutable plan")
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
        command = fit_command(prereg, inventory, model_path)
        plan = {
            "schema": PLAN_SCHEMA,
            "experiment_id": prereg["experiment_id"],
            "repository_commit": commit,
            "preregistration_path": _relative(prereg_path),
            "preregistration_sha256": _sha256(prereg_path),
            "source_collection_ledger_sha256": prereg["sha256_bindings"][
                "source_collection_ledger"
            ],
            "work_log_path": _relative(work_log),
            "fit_command": command,
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
        "work_log_path": _relative(work_log),
        "fit_command": fit_command(prereg, inventory, model_path),
        "online_wine": False,
    }
    if plan != expected_plan:
        raise ValueError("Stage 4 L1b immutable plan differs")

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
            raise ValueError("partial L1b fit requires manual triage")
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
        default=REPOSITORY / "experiments/l1b-stage4-bc-convergence-v1.json",
    )
    args = parser.parse_args(argv)
    result = run(args.preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
