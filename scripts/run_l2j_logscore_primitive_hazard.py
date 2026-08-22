#!/usr/bin/env python3
"""Run the frozen L2j BCE-with-logits primitive-set experiment once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
from th06_rl.factual_primitive_dataset import (  # noqa: E402
    load_primitive_probe_dataset,
)
from th06_rl.factual_primitive_log_hazard_model import (  # noqa: E402
    MODEL_KIND,
    PRIMITIVE_LOG_HAZARD_EVALUATION_SCHEMA,
    PRIMITIVE_LOG_HAZARD_FIT_SCHEMA,
    TRAINING_PROPER_SCORE,
    benchmark_primitive_log_hazard,
    evaluate_primitive_log_hazard_models,
    fit_primitive_log_hazard_models,
)


PREREG_SCHEMA = "th06-rl-l2j-logscore-primitive-hazard-prereg-v1"
PLAN_SCHEMA = "th06-rl-l2j-logscore-primitive-hazard-plan-v1"
FIT_ARTIFACT_SCHEMA = "th06-rl-l2j-logscore-primitive-hazard-artifact-v1"
RESULT_SCHEMA = "th06-rl-l2j-logscore-primitive-hazard-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("L2j logscore preregistration schema mismatch")
    for key in ("data", "fit", "evaluation", "gate", "paths", "sha256_bindings"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"L2j logscore preregistration lacks {key}")
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    source_prereg_path = _repository_path(paths["source_l2i_preregistration"])
    if (
        not source_prereg_path.is_file()
        or _sha256(source_prereg_path)
        != bindings.get("source_l2i_preregistration")
    ):
        raise ValueError("frozen L2i preregistration differs")
    source = _object(source_prereg_path)
    if prereg["data"] != source["data"]:
        raise ValueError("L2j data changed from L2i")

    expected_fit = dict(source["fit"])
    expected_fit.update({
        "model": MODEL_KIND,
        "training_proper_score": TRAINING_PROPER_SCORE,
        "loss_implementation": "torch.nn.BCEWithLogitsLoss",
        "positive_class_weight": 1.0,
        "negative_class_weight": 1.0,
    })
    if prereg["fit"] != expected_fit:
        raise ValueError("L2j changed more than the frozen training loss")
    expected_evaluation = dict(source["evaluation"])
    expected_evaluation.update({
        "source_l2i_reproduction_tolerance": 1e-12,
        "loss_correction_bootstrap_seed": 20260853,
    })
    if prereg["evaluation"] != expected_evaluation:
        raise ValueError("L2j evaluation changed")
    expected_gate = dict(source["gate"])
    expected_gate.update({
        "minimum_loss_correction_episodes": 6,
        "loss_only_improves_frozen_l2i": (
            "complete-episode bootstrap upper endpoint for logscore object-full "
            "minus frozen L2i object-full Brier is below zero with at least "
            "six of eight favorable episode directions"
        ),
        "logscore_probability_model_admitted": True,
    })
    if prereg["gate"] != expected_gate:
        raise ValueError("L2j gate changed")
    if prereg.get("online_wine") is not False or prereg.get("fits_policy") is not False:
        raise ValueError("L2j may not run Wine or fit a deployable policy")

    required_paths = (
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
        "source_l2f_fit",
        "source_l2f_result",
        "source_l2i_preregistration",
        "source_l2i_fit",
        "source_l2i_result",
        "primitive_dataset_module",
        "primitive_brier_module",
        "primitive_log_hazard_module",
        "primitive_log_hazard_runner",
    )
    for key in required_paths:
        _repository_path(paths[key])
    bound_files = (
        "source_collection_ledger",
        "source_l1_result",
        "source_l1_model",
        "source_l2d_preregistration",
        "source_l2d_collection_ledger",
        "source_l2d_result",
        "source_l2f_fit",
        "source_l2f_result",
        "source_l2i_preregistration",
        "source_l2i_fit",
        "source_l2i_result",
        "primitive_dataset_module",
        "primitive_brier_module",
        "primitive_log_hazard_module",
        "primitive_log_hazard_runner",
    )
    for key in bound_files:
        source_path = _repository_path(paths[key])
        if not source_path.is_file() or _sha256(source_path) != bindings.get(key):
            raise ValueError(f"preregistered L2j input differs: {key}")
    source_result = _object(_repository_path(paths["source_l2i_result"]))
    if (
        source_result.get("complete") is not True
        or source_result.get("decision")
        != "reject-observed-primitive-set-h16-hazard"
    ):
        raise ValueError("L2j source is not the frozen L2i rejection")
    return prereg


def _source_reproduction(
    evaluation: dict[str, object],
    source_l2f: dict[str, object],
    source_l2i: dict[str, object],
) -> dict[str, float]:
    observed_l2f = evaluation["metrics"]["frozen_l2f_full_same_rows"]
    expected_l2f = source_l2f["evaluation"]["metrics"][
        "full_current_root_action"
    ]
    observed_l2i = evaluation["metrics"]["frozen_l2i_object_full_same_rows"]
    expected_l2i = source_l2i["evaluation"]["metrics"]["object_full"]
    return {
        "l2f_full_brier_absolute_error": abs(
            float(observed_l2f["brier"]) - float(expected_l2f["brier"])
        ),
        "l2f_full_nll_absolute_error": abs(
            float(observed_l2f["negative_log_likelihood"])
            - float(expected_l2f["negative_log_likelihood"])
        ),
        "l2i_object_full_brier_absolute_error": abs(
            float(observed_l2i["brier"]) - float(expected_l2i["brier"])
        ),
        "l2i_object_full_nll_absolute_error": abs(
            float(observed_l2i["negative_log_likelihood"])
            - float(expected_l2i["negative_log_likelihood"])
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
            raise ValueError("completed L2j result differs")
        return result

    commit = _repository_commit()
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("L2j artifact root lacks its immutable plan")
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
            "source_l2i_result_sha256": prereg["sha256_bindings"][
                "source_l2i_result"
            ],
            "work_log_path": _relative(work_log),
            "online_wine": False,
            "independent_confirmation": False,
            "loads_reused_l2d_after_train_fit": True,
            "single_changed_variable": "training-proper-score",
        }
        _atomic_json(plan_path, plan)
    expected_plan = {
        "schema": PLAN_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "source_l2i_result_sha256": prereg["sha256_bindings"][
            "source_l2i_result"
        ],
        "work_log_path": _relative(work_log),
        "online_wine": False,
        "independent_confirmation": False,
        "loads_reused_l2d_after_train_fit": True,
        "single_changed_variable": "training-proper-score",
    }
    if plan != expected_plan:
        raise ValueError("L2j immutable plan differs")

    fit_settings = prereg["fit"]
    horizon = int(fit_settings["horizon_game_frames"])
    token_cap = int(fit_settings["primitive_token_cap"])
    _work_event(work_log, "train-only-logscore-primitive-hazard-fit-started")
    train = load_primitive_probe_dataset(
        l2._run_paths(prereg, source_inventory, "train"),
        horizons=(horizon,),
        token_cap=token_cap,
        max_rows=int(fit_settings["max_rows"]),
        workers=int(fit_settings["loader_workers"]),
    )
    l2._validate_inventory(prereg, source_inventory, "train", train.inventory)
    hazard = fit_primitive_log_hazard_models(
        train,
        horizon=horizon,
        token_cap=token_cap,
        token_hidden=int(fit_settings["token_hidden"]),
        head_hidden_1=int(fit_settings["head_hidden_1"]),
        head_hidden_2=int(fit_settings["head_hidden_2"]),
        epochs=int(fit_settings["epochs"]),
        batch_size=int(fit_settings["batch_size"]),
        learning_rate=float(fit_settings["learning_rate"]),
        weight_decay=float(fit_settings["weight_decay"]),
        gradient_clip_norm=float(fit_settings["gradient_clip_norm"]),
        seed=int(fit_settings["seed"]),
        threads=int(fit_settings["threads"]),
        expected_torch_version=str(fit_settings["torch_version"]),
    )
    if hazard.get("model") != MODEL_KIND or hazard.get("schema") != PRIMITIVE_LOG_HAZARD_FIT_SCHEMA:
        raise ValueError("L2j logscore hazard fit identity changed")
    evaluation_settings = prereg["evaluation"]
    hazard["train"]["inference_benchmark"] = benchmark_primitive_log_hazard(
        hazard,
        train,
        batch_rows=int(evaluation_settings["inference_benchmark_batch_rows"]),
        warmup_repetitions=int(
            evaluation_settings["inference_benchmark_warmup_repetitions"]
        ),
        measured_repetitions=int(
            evaluation_settings["inference_benchmark_measured_repetitions"]
        ),
        threads=int(evaluation_settings["inference_benchmark_threads"]),
    )
    train_inventory = list(train.inventory)
    fit_artifact = {
        "schema": FIT_ARTIFACT_SCHEMA,
        "primitive_log_hazard_fit_schema": PRIMITIVE_LOG_HAZARD_FIT_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_sha256": _sha256(prereg_path),
        "train_inventory": train_inventory,
        "hazard": hazard,
        "evaluation_loaded": False,
        "deployable_policy": False,
    }
    _atomic_json(fit_path, fit_artifact)
    _work_event(
        work_log,
        "train-only-logscore-primitive-hazard-fit-frozen",
        fit_artifact_path=_relative(fit_path),
        fit_artifact_sha256=_sha256(fit_path),
        train=hazard["train"],
    )
    del train

    _work_event(work_log, "reused-l2d-logscore-primitive-evaluation-started")
    run_dirs = tuple(
        _repository_path(l2d_inventory[index]["run_dir"])
        for index in prereg["data"]["evaluation_episode_indices"]
    )
    evaluation_dataset = load_primitive_probe_dataset(
        run_dirs,
        horizons=(horizon,),
        token_cap=token_cap,
        max_rows=int(evaluation_settings["max_rows"]),
        workers=int(fit_settings["loader_workers"]),
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
        raise ValueError("L2j L2d whole-episode inventory differs")
    source_l2f_fit = _object(_repository_path(paths["source_l2f_fit"]))
    source_l2i_fit = _object(_repository_path(paths["source_l2i_fit"]))
    gate = prereg["gate"]
    evaluation = evaluate_primitive_log_hazard_models(
        hazard,
        source_l2f_fit["hazard"],
        source_l2i_fit["hazard"],
        evaluation_dataset,
        prediction_batch_size=int(evaluation_settings["prediction_batch_size"]),
        bootstrap_samples=int(evaluation_settings["bootstrap_samples"]),
        bootstrap_seed=int(evaluation_settings["bootstrap_seed"]),
        calibration_bins=int(evaluation_settings["calibration_bins"]),
        minimum_overall_positives=int(gate["minimum_overall_hit_positives"]),
        minimum_overall_negatives=int(gate["minimum_overall_hit_negatives"]),
        minimum_nonbaseline_positives=int(gate["minimum_nonbaseline_hit_positives"]),
        minimum_low_propensity_positives=int(gate["minimum_low_propensity_hit_positives"]),
        minimum_prefirst_hit_positives=int(gate["minimum_prefirst_hit_positives"]),
        minimum_object_gain_episodes=int(gate["minimum_object_gain_episodes"]),
        minimum_overall_episodes_favoring_full=int(gate["minimum_overall_episodes_favoring_full"]),
        minimum_nonbaseline_episodes_favoring_full=int(gate["minimum_nonbaseline_episodes_favoring_full"]),
        minimum_low_propensity_episodes_favoring_full=int(gate["minimum_low_propensity_episodes_favoring_full"]),
        minimum_prefirst_episodes_favoring_full=int(gate["minimum_prefirst_episodes_favoring_full"]),
        calibration_in_the_large_absolute_max=float(gate["calibration_in_the_large_absolute_max"]),
        full_ece_over_action_ablated_max=float(gate["full_ece_over_action_ablated_max"]),
        maximum_saturated_fraction=float(gate["maximum_saturated_fraction"]),
        maximum_parameter_count=int(gate["maximum_parameter_count"]),
        maximum_batch18_p99_ms=float(gate["maximum_batch18_one_thread_p99_ms"]),
        minimum_loss_correction_episodes=int(gate["minimum_loss_correction_episodes"]),
    )
    source_l2f = _object(_repository_path(paths["source_l2f_result"]))
    source_l2i = _object(_repository_path(paths["source_l2i_result"]))
    reproduction = _source_reproduction(evaluation, source_l2f, source_l2i)
    max_error = max(reproduction.values())
    reproduced = max_error <= min(
        float(evaluation_settings["source_brier_reproduction_tolerance"]),
        float(evaluation_settings["source_l2i_reproduction_tolerance"]),
    )
    if not reproduced:
        evaluation["summary"]["decision"] = "stop-source-reproduction-failed"
        evaluation["gates"]["selected_for_fresh_confirmation"] = False
    result = {
        "schema": RESULT_SCHEMA,
        "evaluation_schema": PRIMITIVE_LOG_HAZARD_EVALUATION_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "plan_path": _relative(plan_path),
        "plan_sha256": _sha256(plan_path),
        "fit_artifact_path": _relative(fit_path),
        "fit_artifact_sha256": _sha256(fit_path),
        "source_l2i_result_sha256": prereg["sha256_bindings"]["source_l2i_result"],
        "source_reproduction": {
            **reproduction,
            "maximum_absolute_error": max_error,
            "passed": reproduced,
        },
        "train_inventory": train_inventory,
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
        "logscore-primitive-hazard-evaluation-completed",
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
        default=REPOSITORY / "experiments/l2j-logscore-primitive-hazard-v1.json",
    )
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
