#!/usr/bin/env python3
"""Run the preregistered L2e train-only Platt calibration experiment once."""

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
    load_boundary_probe_dataset,
)
from th06_rl.factual_probe_calibration import (  # noqa: E402
    CALIBRATION_EVALUATION_SCHEMA,
    CALIBRATOR_SCHEMA,
    evaluate_train_only_platt_calibrator,
    fit_train_only_platt_calibrator,
)
from th06_rl.factual_probes import load_factual_probe_dataset  # noqa: E402


PREREG_SCHEMA = "th06-rl-l2e-train-only-calibration-prereg-v1"
PLAN_SCHEMA = "th06-rl-l2e-train-only-calibration-plan-v1"
FIT_SCHEMA = "th06-rl-l2e-train-only-calibration-fit-v1"
RESULT_SCHEMA = "th06-rl-l2e-train-only-calibration-result-v1"
SOURCE_COLLECTION_SCHEMA = "th06-rl-l2d-stage4-fresh-confirmation-collection-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("L2e calibration preregistration schema mismatch")
    for key in ("data", "fit", "evaluation", "gate", "paths", "sha256_bindings"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"L2e calibration preregistration lacks {key}")
    data = prereg["data"]
    fit = prereg["fit"]
    evaluation = prereg["evaluation"]
    gate = prereg["gate"]
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    if (
        data.get("source_train_experiment_id") != "l2-stage4-factual-probes-v1"
        or data.get("evaluation_experiment_id")
        != "l2d-stage4-fresh-confirmation-v1"
        or data.get("train_episode_indices") != [0, 1, 3, 4, 6, 7, 9, 10]
        or data.get("validation_episode_indices") != [2, 5, 8, 11]
        or data.get("evaluation_episode_indices") != list(range(8))
        or data.get("split_unit") != "complete-physical-episode"
        or data.get("reuses_previously_evaluated_l2d") is not True
        or data.get("independent_confirmation") is not False
        or data.get("reuse_without_mutation") is not True
    ):
        raise ValueError("L2e data scope changed")
    expected_fit = {
        "horizon_game_frames": 16,
        "target": "physical-hit-within-fixed-horizon",
        "source_model": "frozen-l2-standardized-ridge-linear-probability",
        "source_ridge_l2": 0.001,
        "source_score": "unclipped-full-current-root-action-score",
        "probability_surface": "two-parameter-affine-logistic-platt",
        "calibrator_regularization": "none",
        "optimizer": "damped-newton-with-backtracking",
        "maximum_updates": 50,
        "minimum_updates": 1,
        "gradient_inf_tolerance": 1e-10,
        "maximum_line_search_steps": 20,
        "max_rows": 400000,
        "validation_loaded_after_fit_serialized": True,
    }
    if fit != expected_fit:
        raise ValueError("L2e train-only calibration fit changed")
    expected_evaluation = {
        "horizon_game_frames": 16,
        "calibration_bins": 10,
        "bootstrap_samples": 2000,
        "bootstrap_seed": 20260825,
        "max_rows": 400000,
        "source_brier_reproduction_tolerance": 1e-12,
        "use": (
            "single post-selection evaluation on already observed L2d episodes; "
            "may select fresh confirmation but is not independent evidence"
        ),
    }
    if evaluation != expected_evaluation:
        raise ValueError("L2e evaluation changed")
    expected_gate = {
        "minimum_overall_hit_positives": 800,
        "minimum_overall_hit_negatives": 100000,
        "minimum_nonbaseline_hit_positives": 100,
        "minimum_prefirst_hit_positives": 64,
        "minimum_episodes_favoring_calibrated": 6,
        "positive_monotone_slope_required": True,
        "proper_score_repair": (
            "complete-episode bootstrap upper endpoint for calibrated minus "
            "uncalibrated full Brier is below zero"
        ),
        "signal_preservation": (
            "complete-episode bootstrap upper endpoint for calibrated full minus "
            "state-only Brier is below zero"
        ),
        "nonbaseline_support": (
            "known-nonbaseline rows occur in all eight episodes, meet positive "
            "support, and have negative calibrated-full minus state-only point Brier"
        ),
        "prefirst_hit_lifecycle": (
            "pre-first-HIT rows occur in all eight episodes, meet positive support, "
            "and have negative calibrated-full minus state-only point Brier"
        ),
        "calibration_in_the_large_absolute_max": 0.002,
        "calibrated_ece_over_state_only_max": 0.001,
        "fresh_confirmation_required_if_selected": True,
        "history_admitted": False,
        "value_learning_admitted": False,
        "online_policy_admitted": False,
    }
    if gate != expected_gate:
        raise ValueError("L2e calibration gate changed")
    if prereg.get("online_wine") is not False or prereg.get("fits_policy") is not False:
        raise ValueError("L2e may not run Wine or fit a policy")
    for key in (
        "artifact_root",
        "experiment_plan",
        "fit_artifact",
        "experiment_result",
        "work_log_root",
        "source_corpus_root",
        "source_collection_ledger",
        "source_l1_result",
        "source_l1_model",
        "source_l2_preregistration",
        "source_l2_fit",
        "source_l2_result",
        "source_l2b_fit",
        "source_l2b_result",
        "source_l2d_corpus_root",
        "source_l2d_preregistration",
        "source_l2d_collection_ledger",
        "source_l2d_result",
        "calibration_module",
        "calibration_runner",
    ):
        _repository_path(paths[key])
    for key in (
        "source_collection_ledger",
        "source_l1_result",
        "source_l1_model",
        "source_l2_preregistration",
        "source_l2_fit",
        "source_l2_result",
        "source_l2b_fit",
        "source_l2b_result",
        "source_l2d_preregistration",
        "source_l2d_collection_ledger",
        "source_l2d_result",
        "calibration_module",
        "calibration_runner",
    ):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L2e input differs: {key}")
    source_l2d = _object(_repository_path(paths["source_l2d_result"]))
    if (
        source_l2d.get("complete") is not True
        or source_l2d.get("decision")
        != "confirm-predictive-signal-calibration-not-ready"
        or source_l2d.get("independent_confirmation") is not True
    ):
        raise ValueError("L2e source is not the frozen L2d calibration failure")
    return prereg


def load_l2d_inventory(prereg: dict[str, Any]) -> dict[int, dict[str, Any]]:
    paths = prereg["paths"]
    ledger = _object(_repository_path(paths["source_l2d_collection_ledger"]))
    rows = ledger.get("episodes")
    if (
        ledger.get("schema") != SOURCE_COLLECTION_SCHEMA
        or ledger.get("complete") is not True
        or not isinstance(rows, list)
        or len(rows) != 8
    ):
        raise ValueError("L2e source L2d collection ledger is incomplete")
    corpus_root = _repository_path(paths["source_l2d_corpus_root"])
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("L2e source L2d episode row is malformed")
        index = int(row.get("index", -1))
        if (
            index in by_index
            or index not in prereg["data"]["evaluation_episode_indices"]
            or row.get("split") != "independent-confirmation"
        ):
            raise ValueError("L2e source L2d episode index or split changed")
        run_dir = _repository_path(row.get("run_dir"))
        try:
            run_dir.relative_to(corpus_root)
        except ValueError as error:
            raise ValueError("L2e source L2d run escaped its corpus") from error
        for name, digest_key in (
            ("run.json", "run_sha256"),
            ("manifest.json", "manifest_sha256"),
        ):
            artifact = run_dir / name
            if not artifact.is_file() or _sha256(artifact) != row.get(digest_key):
                raise ValueError(f"L2e source L2d binding differs: {artifact}")
        by_index[index] = row
    if sorted(by_index) != list(range(8)):
        raise ValueError("L2e source L2d inventory changed")
    return by_index


def _source_reproduction(
    evaluation: dict[str, object],
    source_l2d: dict[str, object],
) -> dict[str, float]:
    expected = source_l2d["confirmation"]["boundaries"]["horizons"]["16"][
        "overall"
    ]
    observed = evaluation["metrics"]
    return {
        "uncalibrated_full_brier_absolute_error": abs(
            float(observed["uncalibrated_full"]["brier"])
            - float(expected["full"]["brier"])
        ),
        "state_only_brier_absolute_error": abs(
            float(observed["state_only"]["brier"])
            - float(expected["state_only"]["brier"])
        ),
    }


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    source_inventory = l1b.load_source_inventory(prereg)
    l2d_inventory = load_l2d_inventory(prereg)
    paths = prereg["paths"]
    artifact_root = _repository_path(paths["artifact_root"])
    plan_path = _repository_path(paths["experiment_plan"])
    fit_path = _repository_path(paths["fit_artifact"])
    result_path = _repository_path(paths["experiment_result"])
    if result_path.is_file():
        result = _object(result_path)
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("preregistration_sha256") != _sha256(prereg_path)
        ):
            raise ValueError("completed L2e result differs")
        return result

    commit = _repository_commit()
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("L2e artifact root lacks its immutable plan")
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
            "source_l2d_result_sha256": prereg["sha256_bindings"][
                "source_l2d_result"
            ],
            "work_log_path": _relative(work_log),
            "online_wine": False,
            "independent_confirmation": False,
            "loads_reused_l2d_after_train_fit": True,
        }
        _atomic_json(plan_path, plan)
    expected_plan = {
        "schema": PLAN_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "source_l2d_result_sha256": prereg["sha256_bindings"]["source_l2d_result"],
        "work_log_path": _relative(work_log),
        "online_wine": False,
        "independent_confirmation": False,
        "loads_reused_l2d_after_train_fit": True,
    }
    if plan != expected_plan:
        raise ValueError("L2e immutable plan differs")

    fit_settings = prereg["fit"]
    horizon = int(fit_settings["horizon_game_frames"])
    _work_event(work_log, "train-only-calibration-fit-started")
    train = load_factual_probe_dataset(
        l2._run_paths(prereg, source_inventory, "train"),
        horizons=(horizon,),
        max_rows=int(fit_settings["max_rows"]),
    )
    l2._validate_inventory(prereg, source_inventory, "train", train.inventory)
    full_fit = _object(_repository_path(paths["source_l2_fit"]))
    calibrator = fit_train_only_platt_calibrator(
        full_fit["fit"],
        train,
        horizon=horizon,
        maximum_updates=int(fit_settings["maximum_updates"]),
        minimum_updates=int(fit_settings["minimum_updates"]),
        gradient_inf_tolerance=float(fit_settings["gradient_inf_tolerance"]),
        maximum_line_search_steps=int(fit_settings["maximum_line_search_steps"]),
    )
    fit_artifact = {
        "schema": FIT_SCHEMA,
        "calibrator_schema": CALIBRATOR_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_sha256": _sha256(prereg_path),
        "source_l2_fit_sha256": prereg["sha256_bindings"]["source_l2_fit"],
        "train_inventory": list(train.inventory),
        "calibrator": calibrator,
        "evaluation_loaded": False,
        "deployable_policy": False,
    }
    _atomic_json(fit_path, fit_artifact)
    _work_event(
        work_log,
        "train-only-calibration-fit-frozen",
        fit_artifact_path=_relative(fit_path),
        fit_artifact_sha256=_sha256(fit_path),
        optimization=calibrator["optimization"],
    )

    _work_event(work_log, "reused-l2d-evaluation-started")
    evaluation_settings = prereg["evaluation"]
    run_dirs = tuple(
        _repository_path(l2d_inventory[index]["run_dir"])
        for index in prereg["data"]["evaluation_episode_indices"]
    )
    evaluation_dataset = load_boundary_probe_dataset(
        run_dirs,
        horizons=(horizon,),
        max_rows=int(evaluation_settings["max_rows"]),
    )
    expected_inventory = [
        {
            "episode_id": l2d_inventory[index]["run_id"],
            "run_sha256": l2d_inventory[index]["run_sha256"],
            "manifest_sha256": l2d_inventory[index]["manifest_sha256"],
        }
        for index in prereg["data"]["evaluation_episode_indices"]
    ]
    observed_inventory = [
        {
            "episode_id": row.get("episode_id"),
            "run_sha256": row.get("run_sha256"),
            "manifest_sha256": row.get("manifest_sha256"),
        }
        for row in evaluation_dataset.factual.inventory
    ]
    if observed_inventory != expected_inventory:
        raise ValueError("L2e L2d whole-episode inventory differs")
    state_fit = _object(_repository_path(paths["source_l2b_fit"]))
    evaluation = evaluate_train_only_platt_calibrator(
        full_fit["fit"],
        state_fit["fit"],
        calibrator,
        evaluation_dataset,
        bootstrap_samples=int(evaluation_settings["bootstrap_samples"]),
        bootstrap_seed=int(evaluation_settings["bootstrap_seed"]),
        calibration_bins=int(evaluation_settings["calibration_bins"]),
        minimum_overall_positives=int(prereg["gate"]["minimum_overall_hit_positives"]),
        minimum_overall_negatives=int(prereg["gate"]["minimum_overall_hit_negatives"]),
        minimum_nonbaseline_positives=int(
            prereg["gate"]["minimum_nonbaseline_hit_positives"]
        ),
        minimum_prefirst_hit_positives=int(
            prereg["gate"]["minimum_prefirst_hit_positives"]
        ),
        minimum_episodes_favoring_calibrated=int(
            prereg["gate"]["minimum_episodes_favoring_calibrated"]
        ),
        calibration_in_the_large_absolute_max=float(
            prereg["gate"]["calibration_in_the_large_absolute_max"]
        ),
        calibrated_ece_over_state_only_max=float(
            prereg["gate"]["calibrated_ece_over_state_only_max"]
        ),
    )
    source_l2d = _object(_repository_path(paths["source_l2d_result"]))
    reproduction = _source_reproduction(evaluation, source_l2d)
    max_error = max(reproduction.values())
    reproduced = max_error <= float(
        evaluation_settings["source_brier_reproduction_tolerance"]
    )
    if not reproduced:
        evaluation["summary"]["decision"] = "stop-source-l2d-reproduction-failed"
        evaluation["gates"]["selected_for_fresh_confirmation"] = False
    result = {
        "schema": RESULT_SCHEMA,
        "evaluation_schema": CALIBRATION_EVALUATION_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "plan_path": _relative(plan_path),
        "plan_sha256": _sha256(plan_path),
        "fit_artifact_path": _relative(fit_path),
        "fit_artifact_sha256": _sha256(fit_path),
        "source_l2d_result_sha256": prereg["sha256_bindings"]["source_l2d_result"],
        "source_reproduction": {
            **reproduction,
            "maximum_absolute_error": max_error,
            "passed": reproduced,
        },
        "train_inventory": list(train.inventory),
        "evaluation_inventory": list(evaluation_dataset.factual.inventory),
        "evaluation": evaluation,
        "decision": evaluation["summary"]["decision"],
        "online_wine": None,
        "independent_confirmation": False,
        "complete": True,
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "calibration-evaluation-completed",
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
        default=REPOSITORY / "experiments/l2e-train-only-calibration-v1.json",
    )
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
