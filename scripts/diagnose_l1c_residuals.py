#!/usr/bin/env python3
"""Run the read-only Stage 4 L1c residual decomposition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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
)
from scripts import run_l1b_stage4 as l1b  # noqa: E402
from scripts import run_l1c_stage4 as l1c  # noqa: E402
from scripts.run_l1_stage4 import (  # noqa: E402
    _relative,
    _repository_path,
    _require_clean_worktree,
    _work_event,
)
from th06_rl.bc_residual_diagnostics import (  # noqa: E402
    DIAGNOSIS_SCHEMA,
    diagnose_l1c_residuals,
)
from th06_rl.bc_training import load_behavior_dataset  # noqa: E402


PREREG_SCHEMA = "th06-rl-l1c-residual-diagnosis-prereg-v1"
PLAN_SCHEMA = "th06-rl-l1c-residual-diagnosis-plan-v1"
RESULT_SCHEMA = "th06-rl-l1c-residual-diagnosis-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("L1c residual-diagnosis preregistration schema mismatch")
    for key in ("data", "diagnosis", "paths", "sha256_bindings", "selection_rule"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"L1c residual diagnosis lacks {key}")
    data = prereg["data"]
    diagnosis = prereg["diagnosis"]
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    selection = prereg["selection_rule"]
    if (
        data.get("train_episode_indices") != [0, 1, 3, 4, 6, 7, 9, 10]
        or data.get("validation_episode_indices") != [2, 5, 8, 11]
        or data.get("reuse_without_mutation") is not True
    ):
        raise ValueError("L1c residual diagnosis changed the frozen episode split")
    if (
        diagnosis.get("fit_deployable_model") is not False
        or diagnosis.get("evaluate_scaled_validation") is not False
        or diagnosis.get("exploration_probability") != 0.2
        or diagnosis.get("bootstrap_samples") != 2000
        or diagnosis.get("bootstrap_seed") != 0
        or not math.isclose(
            float(diagnosis.get("calibration_limit", math.nan)),
            0.02873176278051103,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("L1c residual diagnosis changed its read-only boundary")
    if (
        selection.get("selection_uses_transformed_validation") is not False
        or selection.get("scale_branch") != "train-only-scalar-calibration"
        or selection.get("nonlinear_branch") != "small-current-observation-mlp"
        or selection.get("history_for_behavior_target") is not False
    ):
        raise ValueError("L1c residual diagnosis selection rule changed")

    for key in (
        "artifact_root",
        "diagnosis_result",
        "experiment_plan",
        "work_log_root",
        "source_collection_ledger",
        "source_corpus_root",
        "source_l1c_model",
        "source_l1c_preregistration",
        "source_l1c_result",
        "bc_diagnostics_module",
        "bc_features_module",
        "bc_training_module",
        "diagnosis_runner",
    ):
        _repository_path(paths[key])
    for key in (
        "source_collection_ledger",
        "source_l1c_model",
        "source_l1c_preregistration",
        "source_l1c_result",
        "bc_diagnostics_module",
        "bc_features_module",
        "bc_training_module",
        "diagnosis_runner",
    ):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"frozen L1c residual-diagnosis input differs: {key}")
    source_result = _object(_repository_path(paths["source_l1c_result"]))
    if (
        source_result.get("complete") is not True
        or source_result.get("decision") != "stop-l1c-linear-current-observation"
        or source_result.get("optimization_converged") is not True
        or source_result.get("learnability_gate_passed") is not False
        or source_result.get("fit_artifact_sha256") != bindings["source_l1c_model"]
    ):
        raise ValueError("diagnosis source is not the frozen negative L1c result")
    return prereg


def _validate_reproduction(
    diagnosis: dict[str, object],
    model: dict[str, Any],
) -> None:
    reproduced = diagnosis.get("frozen_model_reproduction")
    if not isinstance(reproduced, dict):
        raise ValueError("L1c diagnosis lacks frozen-model reproduction")
    for split in ("train", "validation"):
        observed = reproduced.get(split)
        expected = model["fit"][split]
        if not isinstance(observed, dict):
            raise ValueError(f"L1c diagnosis lacks {split} metrics")
        for metric in (
            "negative_log_likelihood",
            "accuracy",
            "brier_score",
            "expected_calibration_error_10_bin",
            "reactive_action_accuracy",
        ):
            if not math.isclose(
                float(observed[metric]),
                float(expected[metric]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"L1c diagnosis failed to reproduce {split} {metric}")


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    paths = prereg["paths"]
    artifact_root = _repository_path(paths["artifact_root"])
    plan_path = _repository_path(paths["experiment_plan"])
    result_path = _repository_path(paths["diagnosis_result"])
    if result_path.is_file():
        result = _object(result_path)
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("preregistration_sha256") != _sha256(prereg_path)
        ):
            raise ValueError("completed L1c residual diagnosis differs")
        return result

    commit = _repository_commit()
    l1c_prereg_path = _repository_path(paths["source_l1c_preregistration"])
    l1c_prereg = l1c.load_prereg(l1c_prereg_path)
    inventory = l1b.load_source_inventory(l1c_prereg)
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("L1c residual artifact root lacks its immutable plan")
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
            "source_l1c_result_sha256": prereg["sha256_bindings"][
                "source_l1c_result"
            ],
            "work_log_path": _relative(work_log),
            "online_wine": False,
            "mutates_source_model_or_corpus": False,
            "evaluates_scaled_validation": False,
        }
        _atomic_json(plan_path, plan)
    expected_plan = {
        "schema": PLAN_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "source_l1c_result_sha256": prereg["sha256_bindings"]["source_l1c_result"],
        "work_log_path": _relative(work_log),
        "online_wine": False,
        "mutates_source_model_or_corpus": False,
        "evaluates_scaled_validation": False,
    }
    if plan != expected_plan:
        raise ValueError("L1c residual immutable plan differs")

    _work_event(
        work_log,
        "diagnosis-started",
        repository_commit=commit,
        plan_path=_relative(plan_path),
        plan_sha256=_sha256(plan_path),
    )
    train_runs = tuple(
        _repository_path(inventory[index]["run_dir"])
        for index in prereg["data"]["train_episode_indices"]
    )
    validation_runs = tuple(
        _repository_path(inventory[index]["run_dir"])
        for index in prereg["data"]["validation_episode_indices"]
    )
    train = load_behavior_dataset(train_runs, max_rows=400000)
    validation = load_behavior_dataset(validation_runs, max_rows=400000)
    model_path = _repository_path(paths["source_l1c_model"])
    model = l1b.validate_model(
        l1c_prereg,
        inventory,
        model_path,
        commit=_object(_repository_path(paths["source_l1c_result"]))[
            "repository_commit"
        ],
    )
    diagnosis = diagnose_l1c_residuals(
        train,
        validation,
        model,
        calibration_limit=float(prereg["diagnosis"]["calibration_limit"]),
        bootstrap_samples=int(prereg["diagnosis"]["bootstrap_samples"]),
        bootstrap_seed=int(prereg["diagnosis"]["bootstrap_seed"]),
    )
    if diagnosis.get("schema") != DIAGNOSIS_SCHEMA:
        raise ValueError("L1c residual diagnostic schema differs")
    _validate_reproduction(diagnosis, model)
    result = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "experiment_plan_sha256": _sha256(plan_path),
        "source_l1c_result_sha256": prereg["sha256_bindings"]["source_l1c_result"],
        "source_l1c_model_sha256": prereg["sha256_bindings"]["source_l1c_model"],
        "selection": diagnosis["selection"],
        "diagnosis": diagnosis,
        "online_canary": None,
        "claim": "read-only residual attribution and next-ablation selection only",
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "diagnosis-complete",
        result_path=_relative(result_path),
        result_sha256=_sha256(result_path),
        selection=result["selection"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=REPOSITORY / "experiments/l1c-stage4-residual-diagnosis-v1.json",
    )
    args = parser.parse_args(argv)
    result = run(args.preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
