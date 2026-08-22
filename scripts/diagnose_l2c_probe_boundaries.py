#!/usr/bin/env python3
"""Run the preregistered L2c support/lifecycle/calibration diagnosis once."""

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
)
from scripts import run_l1b_stage4 as l1b  # noqa: E402
from scripts import run_l2_stage4_factual_probes as l2  # noqa: E402
from scripts.run_l1_stage4 import (  # noqa: E402
    _relative,
    _repository_path,
    _require_clean_worktree,
    _work_event,
)
from th06_rl.factual_probe_boundary_diagnostics import (  # noqa: E402
    BOUNDARY_DIAGNOSIS_SCHEMA,
    LIFECYCLE_STRATA,
    PROPENSITY_STRATA,
    diagnose_probe_boundaries,
    load_boundary_probe_dataset,
)


PREREG_SCHEMA = "th06-rl-l2c-boundary-prereg-v1"
PLAN_SCHEMA = "th06-rl-l2c-boundary-plan-v1"
RESULT_SCHEMA = "th06-rl-l2c-boundary-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("L2c boundary preregistration schema mismatch")
    for key in ("data", "diagnosis", "gate", "paths", "sha256_bindings"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"L2c boundary preregistration lacks {key}")
    data = prereg["data"]
    diagnosis = prereg["diagnosis"]
    gate = prereg["gate"]
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    if (
        data.get("reuse_without_mutation") is not True
        or data.get("reuses_previously_evaluated_validation") is not True
        or data.get("independent_confirmation") is not False
        or data.get("source_experiment_id") != "l2b-incremental-action-v1"
        or data.get("split_unit") != "complete-physical-episode"
        or data.get("train_episode_indices") != [0, 1, 3, 4, 6, 7, 9, 10]
        or data.get("validation_episode_indices") != [2, 5, 8, 11]
    ):
        raise ValueError("L2c data scope changed")
    expected_diagnosis = {
        "horizons_game_frames": [16, 64],
        "primary_horizon_game_frames": 16,
        "calibration_bins": 10,
        "max_rows_per_horizon": 400000,
        "lifecycle_strata": list(LIFECYCLE_STRATA),
        "lifecycle_boundaries_game_frames": [64, 256],
        "propensity_strata": list(PROPENSITY_STRATA),
        "support_views": [
            "published-equals-baseline",
            "published-differs-from-baseline",
            "logged-propensity-strata",
            "published-action",
        ],
        "source_brier_reproduction_tolerance": 1e-12,
    }
    if diagnosis != expected_diagnosis:
        raise ValueError("L2c boundary diagnosis changed")
    expected_gate = {
        "descriptive_only": True,
        "fresh_confirmation_required": True,
        "history_admitted": False,
        "value_learning_admitted": False,
        "source_l2b_reproduction_required": True,
    }
    if gate != expected_gate:
        raise ValueError("L2c boundary gate changed")
    if prereg.get("online_wine") is not False or prereg.get("fits_policy") is not False:
        raise ValueError("L2c may not run Wine or fit a policy")
    for key in (
        "artifact_root",
        "experiment_plan",
        "experiment_result",
        "work_log_root",
        "source_corpus_root",
        "source_collection_ledger",
        "source_l1_result",
        "source_l1_model",
        "source_l2_preregistration",
        "source_l2_fit",
        "source_l2_result",
        "source_l2b_preregistration",
        "source_l2b_fit",
        "source_l2b_result",
        "boundary_module",
        "diagnosis_runner",
    ):
        _repository_path(paths[key])
    for key in (
        "source_collection_ledger",
        "source_l1_result",
        "source_l1_model",
        "source_l2_preregistration",
        "source_l2_fit",
        "source_l2_result",
        "source_l2b_preregistration",
        "source_l2b_fit",
        "source_l2b_result",
        "boundary_module",
        "diagnosis_runner",
    ):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L2c input differs: {key}")
    source_l2b = _object(_repository_path(paths["source_l2b_result"]))
    if (
        source_l2b.get("complete") is not True
        or source_l2b.get("decision") != "proceed-action-relative-current-root-signal"
        or source_l2b.get("independent_confirmation") is not False
    ):
        raise ValueError("L2c source is not the frozen reused-heldout L2b result")
    return prereg


def _reproduction_errors(
    diagnosis: dict[str, object],
    source_l2b: dict[str, object],
    horizons: tuple[int, ...],
) -> dict[str, object]:
    errors = {}
    maximum = 0.0
    source = source_l2b["diagnosis"]
    for horizon in horizons:
        observed = diagnosis["horizons"][str(horizon)]["overall"]
        expected = source["horizons"][str(horizon)]["targets"]["hit"]
        full_error = abs(
            float(observed["full"]["brier"])
            - float(expected["full_current_root_action"]["brier"])
        )
        state_error = abs(
            float(observed["state_only"]["brier"])
            - float(expected["state_only"]["brier"])
        )
        errors[str(horizon)] = {
            "full_brier_absolute_error": full_error,
            "state_only_brier_absolute_error": state_error,
        }
        maximum = max(maximum, full_error, state_error)
    return {"horizons": errors, "maximum_absolute_error": maximum}


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    inventory = l1b.load_source_inventory(prereg)
    paths = prereg["paths"]
    artifact_root = _repository_path(paths["artifact_root"])
    plan_path = _repository_path(paths["experiment_plan"])
    result_path = _repository_path(paths["experiment_result"])
    if result_path.is_file():
        result = _object(result_path)
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("preregistration_sha256") != _sha256(prereg_path)
        ):
            raise ValueError("completed L2c result differs")
        return result

    commit = _repository_commit()
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("L2c artifact root lacks its immutable plan")
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
            "source_l2b_result_sha256": prereg["sha256_bindings"][
                "source_l2b_result"
            ],
            "work_log_path": _relative(work_log),
            "online_wine": False,
            "independent_confirmation": False,
        }
        _atomic_json(plan_path, plan)
    if plan != {
        "schema": PLAN_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "source_l2b_result_sha256": prereg["sha256_bindings"]["source_l2b_result"],
        "work_log_path": _relative(work_log),
        "online_wine": False,
        "independent_confirmation": False,
    }:
        raise ValueError("L2c immutable plan differs")

    settings = prereg["diagnosis"]
    horizons = tuple(int(value) for value in settings["horizons_game_frames"])
    _work_event(work_log, "reused-validation-boundary-load-started")
    dataset = load_boundary_probe_dataset(
        l2._run_paths(prereg, inventory, "validation"),
        horizons=horizons,
        max_rows=int(settings["max_rows_per_horizon"]),
    )
    l2._validate_inventory(prereg, inventory, "validation", dataset.factual.inventory)
    full_fit = _object(_repository_path(paths["source_l2_fit"]))
    state_fit = _object(_repository_path(paths["source_l2b_fit"]))
    diagnosis = diagnose_probe_boundaries(
        full_fit["fit"],
        state_fit["fit"],
        dataset,
        calibration_bins=int(settings["calibration_bins"]),
    )
    source_l2b = _object(_repository_path(paths["source_l2b_result"]))
    reproduction = _reproduction_errors(diagnosis, source_l2b, horizons)
    tolerance = float(settings["source_brier_reproduction_tolerance"])
    reproduced = float(reproduction["maximum_absolute_error"]) <= tolerance
    diagnosis["summary"]["source_l2b_reproduced"] = reproduced
    diagnosis["summary"]["maximum_source_brier_reproduction_error"] = reproduction[
        "maximum_absolute_error"
    ]
    if not reproduced:
        diagnosis["summary"]["decision"] = "stop-source-l2b-reproduction-failed"

    result = {
        "schema": RESULT_SCHEMA,
        "diagnosis_schema": BOUNDARY_DIAGNOSIS_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "plan_path": _relative(plan_path),
        "plan_sha256": _sha256(plan_path),
        "source_l2_fit_sha256": prereg["sha256_bindings"]["source_l2_fit"],
        "source_l2b_fit_sha256": prereg["sha256_bindings"]["source_l2b_fit"],
        "source_l2b_result_sha256": prereg["sha256_bindings"]["source_l2b_result"],
        "validation_inventory": list(dataset.factual.inventory),
        "source_reproduction": reproduction,
        "diagnosis": diagnosis,
        "decision": diagnosis["summary"]["decision"],
        "online_wine": None,
        "independent_confirmation": False,
        "complete": True,
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "boundary-diagnosis-completed",
        result_path=_relative(result_path),
        result_sha256=_sha256(result_path),
        decision=result["decision"],
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=REPOSITORY / "experiments/l2c-stage4-boundary-diagnosis-v1.json",
    )
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
