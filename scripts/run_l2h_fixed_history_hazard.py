#!/usr/bin/env python3
"""Run the frozen L2h fixed causal-history h16 hazard experiment once."""

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
from scripts.run_l2e_train_only_calibration import load_l2d_inventory  # noqa: E402
from th06_rl.factual_history_dataset import load_history_probe_dataset  # noqa: E402
from th06_rl.factual_history_hazard_model import (  # noqa: E402
    HISTORY_HAZARD_EVALUATION_SCHEMA,
    HISTORY_HAZARD_FIT_SCHEMA,
    MODEL_KIND,
    evaluate_history_hazard_models,
    fit_history_hazard_models,
)


PREREG_SCHEMA = "th06-rl-l2h-fixed-history-hazard-prereg-v1"
PLAN_SCHEMA = "th06-rl-l2h-fixed-history-hazard-plan-v1"
FIT_ARTIFACT_SCHEMA = "th06-rl-l2h-fixed-history-hazard-artifact-v1"
RESULT_SCHEMA = "th06-rl-l2h-fixed-history-hazard-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("L2h history preregistration schema mismatch")
    for key in ("data", "fit", "evaluation", "gate", "paths", "sha256_bindings"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"L2h history preregistration lacks {key}")
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
        raise ValueError("L2h data scope changed")
    expected_fit = {
        "horizon_game_frames": 16,
        "target": "physical-hit-within-fixed-horizon-under-behavior-continuation",
        "history_length_decision_roots": 16,
        "history_fields_per_root": 15,
        "history_total_fields": 255,
        "history_order": "oldest-to-newest-then-current",
        "history_padding": "none-drop-row-without-complete-prefix",
        "history_requires_consecutive_learning_eligible_roots": True,
        "history_requires_unit_game_frame_intervals": True,
        "training_proper_score": "mean-unweighted-row-brier",
        "model": "shared-depth3-gradient-boosted-brier-regressor",
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "grow_policy": "depthwise",
        "boosted_rounds": 64,
        "maximum_depth": 3,
        "learning_rate": 0.05,
        "minimum_child_weight": 64.0,
        "l2_leaf_regularization": 1.0,
        "maximum_histogram_bins": 256,
        "subsample": 1.0,
        "column_subsample": 1.0,
        "seed": 20260836,
        "threads": 16,
        "device": "cpu",
        "xgboost_version": "3.2.0",
        "max_rows": 400000,
        "validation_loaded_after_fit_serialized": True,
    }
    if fit != expected_fit:
        raise ValueError("L2h history fit changed")
    expected_evaluation = {
        "horizon_game_frames": 16,
        "calibration_bins": 10,
        "bootstrap_samples": 2000,
        "bootstrap_seed": 20260837,
        "max_rows": 400000,
        "source_brier_reproduction_tolerance": 1e-12,
        "use": (
            "single post-selection evaluation on already observed L2d episodes; "
            "may select fresh confirmation but is not independent evidence"
        ),
    }
    if evaluation != expected_evaluation:
        raise ValueError("L2h evaluation changed")
    expected_gate = {
        "minimum_overall_hit_positives": 800,
        "minimum_overall_hit_negatives": 100000,
        "minimum_nonbaseline_hit_positives": 100,
        "minimum_low_propensity_hit_positives": 150,
        "minimum_prefirst_hit_positives": 64,
        "minimum_temporal_gain_episodes": 6,
        "minimum_overall_episodes_favoring_full": 6,
        "minimum_nonbaseline_episodes_favoring_full": 6,
        "minimum_low_propensity_episodes_favoring_full": 6,
        "minimum_prefirst_episodes_favoring_full": 6,
        "maximum_raw_clipped_fraction": 0.001,
        "history_temporal_gain": (
            "complete-episode bootstrap upper endpoint for history-full minus "
            "same-row current-only Brier is below zero with at least six of "
            "eight favorable episode directions"
        ),
        "current_action_signal": (
            "complete-episode bootstrap upper endpoint for history-full minus "
            "current-action-ablated-history Brier is below zero with at least "
            "six of eight favorable episode directions"
        ),
        "nonbaseline_action_signal": (
            "known-nonbaseline rows occur in all eight episodes, meet positive "
            "support, have history-full minus action-ablated bootstrap upper "
            "endpoint below zero, and at least six favorable episode directions"
        ),
        "low_propensity_action_signal": (
            "behavior-propensity-below-0.025 rows occur in all eight episodes, "
            "meet positive support, have history-full minus action-ablated "
            "bootstrap upper endpoint below zero, and at least six favorable "
            "episode directions"
        ),
        "prefirst_hit_lifecycle": (
            "pre-first-HIT rows occur in all eight episodes, meet positive support, "
            "have history-full minus action-ablated bootstrap upper endpoint below "
            "zero, and at least six favorable episode directions"
        ),
        "candidate_probability_surface": (
            "history-full raw clipped fraction is at most 0.001; comparator "
            "clipping is reported but is not a candidate rejection"
        ),
        "calibration_in_the_large_absolute_max": 0.002,
        "full_ece_over_action_ablated_max": 0.001,
        "fresh_confirmation_required_if_selected": True,
        "history_admitted": True,
        "value_learning_admitted": False,
        "online_policy_admitted": False,
    }
    if gate != expected_gate:
        raise ValueError("L2h history gate changed")
    if prereg.get("online_wine") is not False or prereg.get("fits_policy") is not False:
        raise ValueError("L2h may not run Wine or fit a deployable policy")
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
        "source_l2d_corpus_root",
        "source_l2d_preregistration",
        "source_l2d_collection_ledger",
        "source_l2d_result",
        "source_l2f_preregistration",
        "source_l2f_fit",
        "source_l2f_result",
        "source_l2f_hazard_module",
        "source_l2g_result",
        "history_dataset_module",
        "history_hazard_module",
        "history_hazard_runner",
    ):
        _repository_path(paths[key])
    for key in (
        "source_collection_ledger",
        "source_l1_result",
        "source_l1_model",
        "source_l2d_preregistration",
        "source_l2d_collection_ledger",
        "source_l2d_result",
        "source_l2f_preregistration",
        "source_l2f_fit",
        "source_l2f_result",
        "source_l2f_hazard_module",
        "source_l2g_result",
        "history_dataset_module",
        "history_hazard_module",
        "history_hazard_runner",
    ):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L2h input differs: {key}")
    source_l2g = _object(_repository_path(paths["source_l2g_result"]))
    if (
        source_l2g.get("complete") is not True
        or source_l2g.get("decision") != "reject-weighted-h16-hazard"
    ):
        raise ValueError("L2h source is not the frozen L2g rejection")
    return prereg


def _source_reproduction(
    evaluation: dict[str, object],
    source_l2f: dict[str, object],
) -> dict[str, float]:
    observed = evaluation["metrics"]["frozen_l2f_full_all_current_rows"]
    expected = source_l2f["evaluation"]["metrics"]["full_current_root_action"]
    return {
        "l2f_full_brier_absolute_error": abs(
            float(observed["brier"]) - float(expected["brier"])
        ),
        "l2f_full_nll_absolute_error": abs(
            float(observed["negative_log_likelihood"])
            - float(expected["negative_log_likelihood"])
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
            raise ValueError("completed L2h result differs")
        return result

    commit = _repository_commit()
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("L2h artifact root lacks its immutable plan")
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
            "source_l2g_result_sha256": prereg["sha256_bindings"][
                "source_l2g_result"
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
        "source_l2g_result_sha256": prereg["sha256_bindings"]["source_l2g_result"],
        "work_log_path": _relative(work_log),
        "online_wine": False,
        "independent_confirmation": False,
        "loads_reused_l2d_after_train_fit": True,
    }
    if plan != expected_plan:
        raise ValueError("L2h immutable plan differs")

    fit_settings = prereg["fit"]
    horizon = int(fit_settings["horizon_game_frames"])
    history_length = int(fit_settings["history_length_decision_roots"])
    _work_event(work_log, "train-only-history-hazard-fit-started")
    train = load_history_probe_dataset(
        l2._run_paths(prereg, source_inventory, "train"),
        horizons=(horizon,),
        history_length=history_length,
        max_rows=int(fit_settings["max_rows"]),
    )
    l2._validate_inventory(prereg, source_inventory, "train", train.inventory)
    hazard = fit_history_hazard_models(
        train,
        horizon=horizon,
        history_length=history_length,
        boosted_rounds=int(fit_settings["boosted_rounds"]),
        maximum_depth=int(fit_settings["maximum_depth"]),
        learning_rate=float(fit_settings["learning_rate"]),
        minimum_child_weight=float(fit_settings["minimum_child_weight"]),
        l2_leaf_regularization=float(fit_settings["l2_leaf_regularization"]),
        maximum_histogram_bins=int(fit_settings["maximum_histogram_bins"]),
        seed=int(fit_settings["seed"]),
        threads=int(fit_settings["threads"]),
        expected_xgboost_version=str(fit_settings["xgboost_version"]),
    )
    if hazard.get("model") != MODEL_KIND or hazard.get("schema") != HISTORY_HAZARD_FIT_SCHEMA:
        raise ValueError("L2h history hazard fit identity changed")
    fit_artifact = {
        "schema": FIT_ARTIFACT_SCHEMA,
        "history_hazard_fit_schema": HISTORY_HAZARD_FIT_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_sha256": _sha256(prereg_path),
        "train_inventory": list(train.inventory),
        "hazard": hazard,
        "evaluation_loaded": False,
        "deployable_policy": False,
    }
    _atomic_json(fit_path, fit_artifact)
    _work_event(
        work_log,
        "train-only-history-hazard-fit-frozen",
        fit_artifact_path=_relative(fit_path),
        fit_artifact_sha256=_sha256(fit_path),
        train=hazard["train"],
    )

    _work_event(work_log, "reused-l2d-history-hazard-evaluation-started")
    evaluation_settings = prereg["evaluation"]
    run_dirs = tuple(
        _repository_path(l2d_inventory[index]["run_dir"])
        for index in prereg["data"]["evaluation_episode_indices"]
    )
    evaluation_dataset = load_history_probe_dataset(
        run_dirs,
        horizons=(horizon,),
        history_length=history_length,
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
        for row in evaluation_dataset.inventory
    ]
    if observed_inventory != expected_inventory:
        raise ValueError("L2h L2d whole-episode inventory differs")
    source_l2f_fit = _object(_repository_path(paths["source_l2f_fit"]))
    gate = prereg["gate"]
    evaluation = evaluate_history_hazard_models(
        hazard,
        source_l2f_fit["hazard"],
        evaluation_dataset,
        bootstrap_samples=int(evaluation_settings["bootstrap_samples"]),
        bootstrap_seed=int(evaluation_settings["bootstrap_seed"]),
        calibration_bins=int(evaluation_settings["calibration_bins"]),
        minimum_overall_positives=int(gate["minimum_overall_hit_positives"]),
        minimum_overall_negatives=int(gate["minimum_overall_hit_negatives"]),
        minimum_nonbaseline_positives=int(
            gate["minimum_nonbaseline_hit_positives"]
        ),
        minimum_low_propensity_positives=int(
            gate["minimum_low_propensity_hit_positives"]
        ),
        minimum_prefirst_hit_positives=int(gate["minimum_prefirst_hit_positives"]),
        minimum_temporal_gain_episodes=int(gate["minimum_temporal_gain_episodes"]),
        minimum_overall_episodes_favoring_full=int(
            gate["minimum_overall_episodes_favoring_full"]
        ),
        minimum_nonbaseline_episodes_favoring_full=int(
            gate["minimum_nonbaseline_episodes_favoring_full"]
        ),
        minimum_low_propensity_episodes_favoring_full=int(
            gate["minimum_low_propensity_episodes_favoring_full"]
        ),
        minimum_prefirst_episodes_favoring_full=int(
            gate["minimum_prefirst_episodes_favoring_full"]
        ),
        calibration_in_the_large_absolute_max=float(
            gate["calibration_in_the_large_absolute_max"]
        ),
        full_ece_over_action_ablated_max=float(
            gate["full_ece_over_action_ablated_max"]
        ),
        maximum_raw_clipped_fraction=float(gate["maximum_raw_clipped_fraction"]),
    )
    source_l2f = _object(_repository_path(paths["source_l2f_result"]))
    reproduction = _source_reproduction(evaluation, source_l2f)
    max_error = max(reproduction.values())
    reproduced = max_error <= float(
        evaluation_settings["source_brier_reproduction_tolerance"]
    )
    if not reproduced:
        evaluation["summary"]["decision"] = "stop-source-l2f-reproduction-failed"
        evaluation["gates"]["selected_for_fresh_confirmation"] = False
    result = {
        "schema": RESULT_SCHEMA,
        "evaluation_schema": HISTORY_HAZARD_EVALUATION_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "plan_path": _relative(plan_path),
        "plan_sha256": _sha256(plan_path),
        "fit_artifact_path": _relative(fit_path),
        "fit_artifact_sha256": _sha256(fit_path),
        "source_l2g_result_sha256": prereg["sha256_bindings"]["source_l2g_result"],
        "source_reproduction": {
            **reproduction,
            "maximum_absolute_error": max_error,
            "passed": reproduced,
        },
        "train_inventory": list(train.inventory),
        "evaluation_inventory": list(evaluation_dataset.inventory),
        "evaluation": evaluation,
        "decision": evaluation["summary"]["decision"],
        "online_wine": None,
        "independent_confirmation": False,
        "complete": True,
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "history-hazard-evaluation-completed",
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
        default=REPOSITORY / "experiments/l2h-fixed-history-hazard-v1.json",
    )
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
