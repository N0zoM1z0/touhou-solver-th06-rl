#!/usr/bin/env python3
"""Run the preregistered L2b state-only factual-risk attribution once."""

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

from scripts.gate_parallel_wine import _atomic_json, _object, _repository_commit, _sha256  # noqa: E402
from scripts import run_l1b_stage4 as l1b  # noqa: E402
from scripts import run_l2_stage4_factual_probes as l2  # noqa: E402
from scripts.run_l1_stage4 import (  # noqa: E402
    _relative,
    _repository_path,
    _require_clean_worktree,
    _work_event,
)
from th06_rl.factual_probe_diagnostics import (  # noqa: E402
    DIAGNOSIS_SCHEMA,
    STATE_ONLY_FEATURE_NAMES,
    diagnose_incremental_action_signal,
    fit_state_only_probe_models,
)
from th06_rl.factual_probes import load_factual_probe_dataset  # noqa: E402


PREREG_SCHEMA = "th06-rl-l2b-incremental-action-prereg-v1"
PLAN_SCHEMA = "th06-rl-l2b-incremental-action-plan-v1"
FIT_SCHEMA = "th06-rl-l2b-state-only-fit-v1"
RESULT_SCHEMA = "th06-rl-l2b-incremental-action-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("L2b incremental-action preregistration schema mismatch")
    for key in ("data", "diagnosis", "gate", "paths", "sha256_bindings"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"L2b incremental-action preregistration lacks {key}")
    data = prereg["data"]
    diagnosis = prereg["diagnosis"]
    gate = prereg["gate"]
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    if (
        data.get("reuse_without_mutation") is not True
        or data.get("reuses_previously_evaluated_validation") is not True
        or data.get("independent_confirmation") is not False
        or data.get("source_experiment_id") != "l2-stage4-factual-probes-v1"
        or data.get("train_episode_indices") != [0, 1, 3, 4, 6, 7, 9, 10]
        or data.get("validation_episode_indices") != [2, 5, 8, 11]
    ):
        raise ValueError("L2b data or reused-heldout scope changed")
    expected_diagnosis = {
        "feature_ablation": "remove every action-relative feature",
        "state_only_feature_names": list(STATE_ONLY_FEATURE_NAMES),
        "horizons_game_frames": [1, 4, 16, 64],
        "supported_hit_horizons": [16, 64],
        "model": "standardized-ridge-linear-probability",
        "ridge_l2": 0.001,
        "calibration_bins": 10,
        "bootstrap_samples": 2000,
        "bootstrap_seed": 20260823,
        "source_brier_reproduction_tolerance": 1e-12,
        "max_rows_per_split_per_horizon": 400000,
        "validation_use": (
            "load reused validation exactly once after serializing the train-only "
            "state-only fit; do not tune or claim independent confirmation"
        ),
    }
    if diagnosis != expected_diagnosis:
        raise ValueError("L2b incremental-action diagnosis changed")
    expected_gate = {
        "primary": (
            "at least one frozen supported HIT horizon has a complete-episode "
            "bootstrap upper endpoint below zero for full minus state-only Brier"
        ),
        "shield_comparison": "secondary attribution reported at every horizon",
        "source_full_probe_reproduction_required": True,
        "history_admitted": False,
        "value_learning_admitted": False,
    }
    if gate != expected_gate:
        raise ValueError("L2b incremental-action gate changed")
    if prereg.get("online_wine") is not False or prereg.get("fits_policy") is not False:
        raise ValueError("L2b may not run Wine or fit a policy")
    for key in (
        "artifact_root",
        "experiment_plan",
        "experiment_result",
        "state_only_fit_artifact",
        "work_log_root",
        "source_corpus_root",
        "source_collection_ledger",
        "source_l1_result",
        "source_l1_model",
        "source_l2_preregistration",
        "source_l2_fit",
        "source_l2_result",
        "diagnostics_module",
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
        "diagnostics_module",
        "diagnosis_runner",
    ):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L2b input differs: {key}")
    source_result = _object(_repository_path(paths["source_l2_result"]))
    if (
        source_result.get("complete") is not True
        or source_result.get("decision")
        != "proceed-current-observation-factual-signal"
        or source_result.get("fit_artifact_sha256") != bindings["source_l2_fit"]
        or source_result.get("online_wine") is not None
    ):
        raise ValueError("L2b source is not the frozen positive factual probe")
    return prereg


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    source_l2_prereg = l2.load_prereg(
        _repository_path(prereg["paths"]["source_l2_preregistration"])
    )
    inventory = l1b.load_source_inventory(prereg)
    paths = prereg["paths"]
    artifact_root = _repository_path(paths["artifact_root"])
    fit_path = _repository_path(paths["state_only_fit_artifact"])
    plan_path = _repository_path(paths["experiment_plan"])
    result_path = _repository_path(paths["experiment_result"])
    if result_path.is_file():
        result = _object(result_path)
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("preregistration_sha256") != _sha256(prereg_path)
        ):
            raise ValueError("completed L2b incremental-action result differs")
        return result

    commit = _repository_commit()
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("L2b artifact root lacks its immutable plan")
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
            "source_l2_result_sha256": prereg["sha256_bindings"][
                "source_l2_result"
            ],
            "work_log_path": _relative(work_log),
            "online_wine": False,
            "independent_confirmation": False,
            "loads_validation_after_train_fit": True,
        }
        _atomic_json(plan_path, plan)
    expected_plan = {
        "schema": PLAN_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "source_l2_result_sha256": prereg["sha256_bindings"]["source_l2_result"],
        "work_log_path": _relative(work_log),
        "online_wine": False,
        "independent_confirmation": False,
        "loads_validation_after_train_fit": True,
    }
    if plan != expected_plan:
        raise ValueError("L2b immutable plan differs")

    settings = prereg["diagnosis"]
    horizons = tuple(int(value) for value in settings["horizons_game_frames"])
    _work_event(work_log, "train-load-started", repository_commit=commit)
    train = load_factual_probe_dataset(
        l2._run_paths(prereg, inventory, "train"),
        horizons=horizons,
        max_rows=int(settings["max_rows_per_split_per_horizon"]),
    )
    l2._validate_inventory(prereg, inventory, "train", train.inventory)
    state_only = fit_state_only_probe_models(
        train,
        ridge_l2=float(settings["ridge_l2"]),
    )
    fit_artifact = {
        "schema": FIT_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_sha256": _sha256(prereg_path),
        "train_inventory": list(train.inventory),
        "fit": state_only,
        "validation_loaded": False,
        "deployable_policy": False,
    }
    _atomic_json(fit_path, fit_artifact)
    _work_event(
        work_log,
        "state-only-train-fit-frozen",
        fit_artifact_path=_relative(fit_path),
        fit_artifact_sha256=_sha256(fit_path),
    )

    _work_event(work_log, "reused-validation-load-started")
    validation = load_factual_probe_dataset(
        l2._run_paths(prereg, inventory, "validation"),
        horizons=horizons,
        max_rows=int(settings["max_rows_per_split_per_horizon"]),
    )
    l2._validate_inventory(prereg, inventory, "validation", validation.inventory)
    source_fit = _object(_repository_path(paths["source_l2_fit"]))
    source_result = _object(_repository_path(paths["source_l2_result"]))
    if (
        source_fit.get("schema") != l2.FIT_SCHEMA
        or source_fit.get("preregistration_sha256")
        != _sha256(_repository_path(paths["source_l2_preregistration"]))
    ):
        raise ValueError("L2b source full fit contract differs")
    diagnosis = diagnose_incremental_action_signal(
        source_fit["fit"],
        state_only,
        validation,
        source_result["evaluation"],
        supported_hit_horizons=tuple(
            int(value) for value in settings["supported_hit_horizons"]
        ),
        bootstrap_samples=int(settings["bootstrap_samples"]),
        bootstrap_seed=int(settings["bootstrap_seed"]),
        calibration_bins=int(settings["calibration_bins"]),
        reproduction_tolerance=float(
            settings["source_brier_reproduction_tolerance"]
        ),
    )
    result = {
        "schema": RESULT_SCHEMA,
        "diagnosis_schema": DIAGNOSIS_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "plan_path": _relative(plan_path),
        "plan_sha256": _sha256(plan_path),
        "fit_artifact_path": _relative(fit_path),
        "fit_artifact_sha256": _sha256(fit_path),
        "source_l2_result_sha256": prereg["sha256_bindings"]["source_l2_result"],
        "source_l2_fit_sha256": prereg["sha256_bindings"]["source_l2_fit"],
        "train_inventory": list(train.inventory),
        "validation_inventory": list(validation.inventory),
        "diagnosis": diagnosis,
        "decision": diagnosis["summary"]["decision"],
        "online_wine": None,
        "independent_confirmation": False,
        "complete": True,
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "diagnosis-completed",
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
        default=REPOSITORY / "experiments/l2b-incremental-action-v1.json",
    )
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
