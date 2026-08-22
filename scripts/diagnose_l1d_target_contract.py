#!/usr/bin/env python3
"""Run the frozen L1d behavior-target and optimization root-cause diagnosis."""

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

from scripts.gate_parallel_wine import _atomic_json, _object, _repository_commit, _sha256  # noqa: E402
from scripts import run_l1b_stage4 as l1b  # noqa: E402
from scripts import run_l1d_stage4 as l1d  # noqa: E402
from scripts.run_l1_stage4 import (  # noqa: E402
    _relative,
    _repository_path,
    _require_clean_worktree,
    _work_event,
)
from th06_rl.bc_target_diagnostics import (  # noqa: E402
    DIAGNOSIS_SCHEMA,
    diagnose_l1d_target_contract,
    load_propensity_dataset,
)


PREREG_SCHEMA = "th06-rl-l1d-target-contract-diagnosis-prereg-v1"
PLAN_SCHEMA = "th06-rl-l1d-target-contract-diagnosis-plan-v1"
RESULT_SCHEMA = "th06-rl-l1d-target-contract-diagnosis-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("L1d target-diagnosis preregistration schema mismatch")
    for key in ("data", "diagnosis", "paths", "sha256_bindings"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"L1d target diagnosis lacks {key}")
    data = prereg["data"]
    diagnosis = prereg["diagnosis"]
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    if (
        data.get("train_episode_indices") != [0, 1, 3, 4, 6, 7, 9, 10]
        or data.get("validation_episode_indices") != [2, 5, 8, 11]
        or data.get("reuse_without_mutation") is not True
    ):
        raise ValueError("L1d target diagnosis changed the frozen episode split")
    if (
        diagnosis.get("exploration_probability") != 0.2
        or diagnosis.get("continuation_updates") != 500
        or diagnosis.get("continuation_checkpoints") != [0, 100, 250, 500]
        or diagnosis.get("learning_rate") != 0.05
        or diagnosis.get("l2") != 0.0001
        or diagnosis.get("mixture_tolerance") != 1e-15
        or diagnosis.get("material_kl_reduction") != 0.01
        or diagnosis.get("material_soft_target_advantage") != 0.005
        or diagnosis.get("evaluate_continuation_validation") is not False
        or diagnosis.get("serialize_continuation_parameters") is not False
        or diagnosis.get("fit_deployable_model") is not False
    ):
        raise ValueError("L1d target diagnosis changed its attribution contract")
    for key in (
        "artifact_root",
        "diagnosis_result",
        "experiment_plan",
        "work_log_root",
        "source_collection_ledger",
        "source_corpus_root",
        "source_l1d_model",
        "source_l1d_preregistration",
        "source_l1d_result",
        "diagnostics_module",
        "diagnosis_runner",
    ):
        _repository_path(paths[key])
    for key in (
        "source_collection_ledger",
        "source_l1d_model",
        "source_l1d_preregistration",
        "source_l1d_result",
        "diagnostics_module",
        "diagnosis_runner",
    ):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"frozen L1d target-diagnosis input differs: {key}")
    source_result = _object(_repository_path(paths["source_l1d_result"]))
    if (
        source_result.get("complete") is not True
        or source_result.get("decision") != "stop-l1d-small-current-observation-mlp"
        or source_result.get("optimization_converged") is not True
        or source_result.get("direct_l1c_nll_gate_passed") is not True
        or source_result.get("calibration_gate_passed") is not False
        or source_result.get("learnability_gate_passed") is not False
        or source_result.get("fit_artifact_sha256") != bindings["source_l1d_model"]
        or source_result.get("online_canary") is not None
    ):
        raise ValueError("target diagnosis source is not the frozen negative L1d result")
    return prereg


def _validate_reproduction(
    diagnosis: dict[str, object],
    model: dict[str, Any],
) -> None:
    frozen = diagnosis.get("frozen_l1d")
    if not isinstance(frozen, dict):
        raise ValueError("target diagnosis lacks frozen L1d reproduction")
    mapping = {
        "negative_log_likelihood": "hard_sample_nll",
        "accuracy": "hard_sample_accuracy",
        "brier_score": "hard_sample_brier",
        "expected_calibration_error_10_bin": "hard_sample_ece_10",
    }
    for split in ("train", "validation"):
        observed = frozen.get(split)
        expected = model["fit"][split]
        if not isinstance(observed, dict):
            raise ValueError(f"target diagnosis lacks {split} metrics")
        for source_key, observed_key in mapping.items():
            if not math.isclose(
                float(observed[observed_key]),
                float(expected[source_key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"target diagnosis failed to reproduce {split} {source_key}"
                )


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
            raise ValueError("completed L1d target diagnosis differs")
        return result

    commit = _repository_commit()
    source_prereg_path = _repository_path(paths["source_l1d_preregistration"])
    source_prereg = l1d.load_prereg(source_prereg_path)
    inventory = l1b.load_source_inventory(source_prereg)
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("L1d target-diagnosis artifact root lacks its plan")
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
            "source_l1d_result_sha256": prereg["sha256_bindings"][
                "source_l1d_result"
            ],
            "work_log_path": _relative(work_log),
            "online_wine": False,
            "mutates_source_model_or_corpus": False,
            "evaluates_continuation_validation": False,
            "serializes_continuation_parameters": False,
        }
        _atomic_json(plan_path, plan)
    expected_plan = {
        "schema": PLAN_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "source_l1d_result_sha256": prereg["sha256_bindings"]["source_l1d_result"],
        "work_log_path": _relative(work_log),
        "online_wine": False,
        "mutates_source_model_or_corpus": False,
        "evaluates_continuation_validation": False,
        "serializes_continuation_parameters": False,
    }
    if plan != expected_plan:
        raise ValueError("L1d target-diagnosis immutable plan differs")

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
    settings = prereg["diagnosis"]
    train = load_propensity_dataset(
        train_runs,
        exploration_probability=float(settings["exploration_probability"]),
        max_rows=400000,
    )
    validation = load_propensity_dataset(
        validation_runs,
        exploration_probability=float(settings["exploration_probability"]),
        max_rows=400000,
    )
    model_path = _repository_path(paths["source_l1d_model"])
    source_result = _object(_repository_path(paths["source_l1d_result"]))
    model = l1d.validate_model(
        source_prereg,
        inventory,
        model_path,
        commit=str(source_result["repository_commit"]),
    )
    diagnosis = diagnose_l1d_target_contract(
        train,
        validation,
        model,
        exploration_probability=float(settings["exploration_probability"]),
        continuation_updates=int(settings["continuation_updates"]),
        continuation_checkpoints=tuple(settings["continuation_checkpoints"]),
        learning_rate=float(settings["learning_rate"]),
        l2=float(settings["l2"]),
        mixture_tolerance=float(settings["mixture_tolerance"]),
        material_kl_reduction=float(settings["material_kl_reduction"]),
        material_soft_target_advantage=float(
            settings["material_soft_target_advantage"]
        ),
    )
    if diagnosis.get("schema") != DIAGNOSIS_SCHEMA:
        raise ValueError("L1d target diagnostic schema differs")
    _validate_reproduction(diagnosis, model)
    result = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "experiment_plan_sha256": _sha256(plan_path),
        "source_l1d_result_sha256": prereg["sha256_bindings"]["source_l1d_result"],
        "source_l1d_model_sha256": prereg["sha256_bindings"]["source_l1d_model"],
        "attribution": diagnosis["attribution"],
        "diagnosis": diagnosis,
        "online_canary": None,
        "claim": (
            "offline target/optimization attribution only; continuation branches "
            "are train-only, non-serialized, and non-deployable"
        ),
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "diagnosis-complete",
        result_path=_relative(result_path),
        result_sha256=_sha256(result_path),
        attribution=result["attribution"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=REPOSITORY / "experiments/l1d-stage4-target-diagnosis-v1.json",
    )
    args = parser.parse_args(argv)
    result = run(args.preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
