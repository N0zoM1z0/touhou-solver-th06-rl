#!/usr/bin/env python3
"""Collect, fit, and evaluate the frozen L2m h12 intention hazard once."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
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
    _pool,
    _repository_commit,
    _sha256,
)
from scripts import run_l1_stage4 as l1  # noqa: E402
from scripts import run_l2l_stage4_action_exposure as l2l  # noqa: E402
from scripts.run_l1_stage4 import (  # noqa: E402
    _relative,
    _repository_path,
    _require_clean_worktree,
    _work_event,
)
from th06_rl.action_exposure_audit_v2 import (  # noqa: E402
    audit_episode,
    audit_hit_target_episode,
    summarize_audits,
)
from th06_rl.action_intention_dataset import (  # noqa: E402
    DATASET_SCHEMA,
    load_action_intention_dataset,
)
from th06_rl.action_intention_hazard import (  # noqa: E402
    EVALUATION_SCHEMA,
    FIT_SCHEMA,
    MODEL_KIND,
    evaluate_action_intention_hazard_models,
    fit_action_intention_hazard_models,
)
from th06_rl.actions import ACTION_NAMES  # noqa: E402
from th06_rl.policies.fixed_shield_action_exposure import (  # noqa: E402
    STATE_SCHEMA as EXPOSURE_STATE_SCHEMA,
)


PREREG_SCHEMA = "th06-rl-l2m-stage4-intention-hazard-prereg-v1"
SCHEDULE_SCHEMA = "th06-rl-l2m-stage4-intention-hazard-schedule-v1"
COLLECTION_SCHEMA = "th06-rl-l2m-stage4-intention-hazard-collection-v1"
FIT_ARTIFACT_SCHEMA = "th06-rl-l2m-stage4-intention-hazard-artifact-v1"
RESULT_SCHEMA = "th06-rl-l2m-stage4-intention-hazard-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("L2m preregistration schema mismatch")
    for key in (
        "data",
        "collection",
        "fit",
        "evaluation",
        "gate",
        "paths",
        "sha256_bindings",
    ):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"L2m preregistration lacks {key}")
    data = prereg["data"]
    collection = prereg["collection"]
    fit = prereg["fit"]
    evaluation = prereg["evaluation"]
    gate = prereg["gate"]
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    episodes = collection.get("episodes")
    if (
        data != {
            "source_pilot_experiment_id": "l2l-stage4-action-exposure-v1",
            "source_pilot_episode_indices": [0, 1],
            "new_train_episode_indices": list(range(10)),
            "new_validation_episode_indices": list(range(10, 16)),
            "train_episode_count": 12,
            "validation_episode_count": 6,
            "split_unit": "complete-physical-episode",
            "pilot_use": "train-only",
            "validation_loaded_after_fit_serialized": True,
            "reuse_without_mutation": True,
        }
        or not isinstance(episodes, list)
        or len(episodes) != 16
        or collection.get("stage") != 4
        or collection.get("difficulty") != "lunatic"
        or collection.get("episode_unit") != "complete-practice-stage"
        or collection.get("assignment")
        != "uniform-over-current-observed-shield-set-at-group-start"
        or collection.get("exposure_roots") != 12
        or collection.get("target_horizon_unit_frames") != 12
        or collection.get("continuation")
        != "retain-intended-action-when-currently-shield-admissible"
        or collection.get("override")
        != "deterministic-current-reactive-baseline-when-intention-is-not-currently-shield-admissible"
        or collection.get("serial_wine_workers") != 1
        or collection.get("worker_index") != 0
        or collection.get("natural_retail_rng") is not True
        or collection.get("game_clock") != "original-retail-normal-speed"
        or collection.get("complete_episode_required") is not True
        or collection.get("physical_hit_retry") is not False
        or collection.get("max_attempts_per_slot") != 3
        or collection.get("peek_or_sequential_stop") is not False
    ):
        raise ValueError("L2m data or collection contract changed")
    if collection.get("lifecycle_interruptions") != [
        "physical-hit",
        "passive",
        "player-not-active",
        "control-dead-end",
    ]:
        raise ValueError("L2m lifecycle contract changed")
    if collection.get("infrastructure_admission") != {
        "capture_p99_ms_max": 40.0,
        "dense_shield_parity_required": True,
        "observation_gap_rate_max": 0.005,
        "player_successor_bit_exact_required": True,
        "solve_p99_ms_max": 16.7,
        "stale_retry_rate_max": 0.0,
        "validator": "scripts.gate_parallel_wine.validate_gate_run",
    }:
        raise ValueError("L2m infrastructure admission changed")
    domain = str(collection.get("policy_seed_derivation_domain"))
    base_seed = int(collection.get("base_policy_seed", -1))
    for index, row in enumerate(episodes):
        expected_split = "train" if index < 10 else "validation"
        if (
            not isinstance(row, dict)
            or int(row.get("index", -1)) != index
            or row.get("split") != expected_split
            or int(row.get("policy_seed", -1))
            != l1._derived_policy_seed(domain, base_seed, index)
        ):
            raise ValueError("L2m episode seed or split changed")
    expected_fit = {
        "target": "physical-hit-during-randomized-h12-intention-before-next-assignment",
        "training_proper_score": "mean-unweighted-group-brier",
        "model": "paired-depth3-gradient-boosted-direct-brier-regressor",
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
        "seed": 20260824,
        "threads": 1,
        "device": "cpu",
        "xgboost_version": "3.2.0",
        "dataset_workers": 4,
        "max_rows": 100000,
        "single_fit_no_sweep": True,
    }
    expected_evaluation = {
        "calibration_bins": 10,
        "bootstrap_samples": 5000,
        "bootstrap_seed": 20260825,
        "dataset_workers": 4,
        "max_rows": 100000,
        "use": "single evaluation on six untouched complete physical episodes",
    }
    expected_gate = {
        "minimum_complete_groups_per_episode": 1800,
        "minimum_assignments_per_action_new_collection": 1000,
        "minimum_no_override_fraction": 0.7,
        "minimum_full_execution_fraction": 0.7,
        "maximum_control_dead_end_rate": 0.005,
        "minimum_train_positives_including_pilot": 128,
        "minimum_validation_positives": 64,
        "minimum_train_accepted_rows_per_action": 800,
        "minimum_validation_accepted_rows_per_action": 400,
        "minimum_validation_negatives": 10000,
        "minimum_validation_positive_episodes": 6,
        "minimum_validation_episodes": 6,
        "minimum_validation_episodes_favoring_full": 5,
        "maximum_calibration_in_the_large_absolute": 0.0025,
        "maximum_expected_calibration_error": 0.005,
        "maximum_full_ece_over_state_only": 0.001,
        "maximum_raw_clipped_fraction": 0.001,
        "zero_bomb_required": True,
        "zero_infrastructure_failure_required": True,
        "exact_exposure_metadata_required": True,
        "complete_episode_required": True,
        "full_minus_state_only_bootstrap_upper_below_zero": True,
        "full_minus_train_prevalence_constant_bootstrap_upper_below_zero": True,
        "parallel_collection_admitted": False,
        "history_admitted": False,
        "value_learning_admitted": False,
        "online_learned_policy_admitted": False,
    }
    if fit != expected_fit or evaluation != expected_evaluation or gate != expected_gate:
        raise ValueError("L2m fit, evaluation, or gate changed")
    if prereg.get("runs_learned_policy") is not False:
        raise ValueError("L2m may not run a learned online policy")
    bound_files = (
        "base_policy_state",
        "exposure_policy_plugin",
        "policy_api",
        "policy_loader",
        "controller",
        "corpus_module",
        "episode_dataset_module",
        "exposure_audit_module",
        "intention_dataset_module",
        "intention_hazard_module",
        "factual_hazard_module",
        "experiment_runner",
        "source_l2l_result",
        "source_l2l_collection_ledger",
        "wine_pool",
    )
    for value in paths.values():
        _repository_path(value)
    for key in bound_files:
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L2m input differs: {key}")
    state = _object(_repository_path(paths["base_policy_state"]))
    if (
        state.get("schema") != EXPOSURE_STATE_SCHEMA
        or state.get("exposure_roots") != 12
    ):
        raise ValueError("L2m base exposure policy differs")
    source_result = _object(_repository_path(paths["source_l2l_result"]))
    if (
        source_result.get("decision")
        != "proceed-serial-action-exposure-training-collection"
        or source_result.get("pilot_train_data_admitted") is not True
        or source_result.get("fits_model") is not False
    ):
        raise ValueError("L2m source is not the frozen successful L2l pilot")
    return prereg


def _pilot_evidence(prereg: dict[str, Any]) -> list[dict[str, object]]:
    paths = prereg["paths"]
    result = _object(_repository_path(paths["source_l2l_result"]))
    ledger = _object(_repository_path(paths["source_l2l_collection_ledger"]))
    if ledger.get("complete") is not True or len(ledger.get("episodes", ())) != 2:
        raise ValueError("L2m pilot collection ledger is incomplete")
    result_inventory = result.get("inventory")
    if not isinstance(result_inventory, list) or len(result_inventory) != 2:
        raise ValueError("L2m pilot result inventory is malformed")
    evidence = list(ledger["episodes"])
    for source, inventory in zip(evidence, result_inventory, strict=True):
        if (
            source.get("run_id") != inventory.get("episode_id")
            or source.get("run_sha256") != inventory.get("run_sha256")
            or source.get("manifest_sha256") != inventory.get("manifest_sha256")
            or inventory.get("split") != "pilot-train-only"
        ):
            raise ValueError("L2m pilot result and ledger disagree")
        l1._verify_recorded_evidence(source)
    return evidence


def _audit_run(path: Path, exposure_roots: int) -> tuple[dict[str, object], dict[str, object]]:
    return (
        audit_episode(path, exposure_roots=exposure_roots),
        audit_hit_target_episode(path, exposure_roots=exposure_roots),
    )


def _split_support(
    prereg: dict[str, Any],
    audited: list[tuple[dict[str, object], dict[str, object]]],
    pilot_result: dict[str, object],
) -> dict[str, object]:
    train_indices = set(prereg["data"]["new_train_episode_indices"])
    validation_indices = set(prereg["data"]["new_validation_episode_indices"])
    pilot_targets = pilot_result["action_exposure_audit"]["target_episodes"]

    def summarize(targets: list[dict[str, object]]) -> dict[str, object]:
        status: Counter[str] = Counter()
        actions: Counter[str] = Counter()
        positives: Counter[str] = Counter()
        for target in targets:
            status.update(target["status"])
            actions.update(target["accepted_actions"])
            positives.update(target["positive_actions"])
        return {
            "episodes": len(targets),
            "accepted_rows": int(status["accepted-label-0"] + status["accepted-label-1"]),
            "positives": int(status["accepted-label-1"]),
            "negatives": int(status["accepted-label-0"]),
            "accepted_actions": dict(sorted(actions.items())),
            "positive_actions": dict(sorted(positives.items())),
            "positive_episodes": sum(
                int(target["status"].get("accepted-label-1", 0)) > 0
                for target in targets
            ),
        }

    train_targets = list(pilot_targets) + [
        audited[index][1] for index in sorted(train_indices)
    ]
    validation_targets = [
        audited[index][1] for index in sorted(validation_indices)
    ]
    train = summarize(train_targets)
    validation = summarize(validation_targets)
    gate = prereg["gate"]
    gates = {
        "train_positive_support": train["positives"]
        >= int(gate["minimum_train_positives_including_pilot"]),
        "validation_positive_support": validation["positives"]
        >= int(gate["minimum_validation_positives"]),
        "validation_negative_support": validation["negatives"]
        >= int(gate["minimum_validation_negatives"]),
        "validation_positive_episode_support": validation["positive_episodes"]
        >= int(gate["minimum_validation_positive_episodes"]),
        "train_action_support": all(
            int(train["accepted_actions"].get(action, 0))
            >= int(gate["minimum_train_accepted_rows_per_action"])
            for action in ACTION_NAMES
        ),
        "validation_action_support": all(
            int(validation["accepted_actions"].get(action, 0))
            >= int(gate["minimum_validation_accepted_rows_per_action"])
            for action in ACTION_NAMES
        ),
    }
    return {"train": train, "validation": validation, "gates": gates}


def _expected_inventory(evidence: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{
        "episode_id": row["run_id"],
        "run_sha256": row["run_sha256"],
        "manifest_sha256": row["manifest_sha256"],
    } for row in evidence]


def _check_dataset_inventory(
    observed: tuple[dict[str, object], ...],
    expected: list[dict[str, object]],
) -> None:
    projected = [{
        "episode_id": row.get("episode_id"),
        "run_sha256": row.get("run_sha256"),
        "manifest_sha256": row.get("manifest_sha256"),
    } for row in observed]
    if projected != expected:
        raise ValueError("L2m learner dataset inventory differs")


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    paths = prereg["paths"]
    artifact_root = _repository_path(paths["artifact_root"])
    corpus_root = _repository_path(paths["corpus_root"])
    collection_path = _repository_path(paths["collection_ledger"])
    fit_path = _repository_path(paths["fit_artifact"])
    result_path = _repository_path(paths["experiment_result"])
    if result_path.is_file():
        result = _object(result_path)
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("preregistration_sha256") != _sha256(prereg_path)
        ):
            raise ValueError("completed L2m result differs")
        return result

    pool_path = _repository_path(paths["wine_pool"])
    pool = _pool(pool_path)
    worker = next(
        row for row in pool["workers"]
        if int(row["worker"]) == int(prereg["collection"]["worker_index"])
    )
    commit = _repository_commit()
    schedule_path = artifact_root / "schedule.json"
    if schedule_path.is_file():
        existing = _object(schedule_path)
        work_log = _repository_path(existing.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("L2m artifact root lacks its immutable schedule")
        started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        work_log = _repository_path(paths["work_log_root"]) / (
            f"{started}-{prereg['experiment_id']}"
        )
        _atomic_json(work_log / "session.json", {
            "schema": "th06-rl-work-log-session-v1",
            "experiment_id": prereg["experiment_id"],
            "repository_commit": commit,
            "preregistration_path": _relative(prereg_path),
            "preregistration_sha256": _sha256(prereg_path),
        })
        existing = l2l.build_schedule(
            prereg,
            prereg_path=prereg_path,
            pool_path=pool_path,
            commit=commit,
            work_log_path=work_log,
        )
        existing["schema"] = SCHEDULE_SCHEMA
        _atomic_json(schedule_path, existing)
    expected_schedule = l2l.build_schedule(
        prereg,
        prereg_path=prereg_path,
        pool_path=pool_path,
        commit=commit,
        work_log_path=work_log,
    )
    expected_schedule["schema"] = SCHEDULE_SCHEMA
    if existing != expected_schedule:
        raise ValueError("L2m immutable schedule differs")
    schedule = existing
    _work_event(
        work_log,
        "experiment-started-or-resumed",
        schedule_path=_relative(schedule_path),
        schedule_sha256=_sha256(schedule_path),
    )

    policy_plugin = _repository_path(paths["exposure_policy_plugin"])
    for row in schedule["episodes"]:
        index = int(row["index"])
        state = l2l.derive_episode_policy_state(prereg, episode=index)
        state_path = artifact_root / str(row["policy_state_path"])
        if state_path.is_file():
            if _object(state_path) != state:
                raise ValueError(f"L2m episode policy state differs: {state_path}")
        else:
            _atomic_json(state_path, state)
        if _sha256(state_path) != row["policy_state_sha256"]:
            raise ValueError(f"L2m policy state hash differs: {state_path}")

    score_template = Path(pool["score_template"]).resolve()
    evidence: dict[int, dict[str, object]] = {}
    for row in schedule["episodes"]:
        index = int(row["index"])
        evidence[index] = l1._run_attempts(
            prereg=prereg,
            row=row,
            worker=worker,
            score_template=score_template,
            artifact_root=artifact_root,
            corpus_root=corpus_root,
            policy_plugin=policy_plugin,
            policy_state=artifact_root / str(row["policy_state_path"]),
            group="collection",
            max_attempts=int(prereg["collection"]["max_attempts_per_slot"]),
            work_log=work_log,
        )

    collection = {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "repository_commit": commit,
        "preregistration_sha256": _sha256(prereg_path),
        "schedule_sha256": _sha256(schedule_path),
        "pool_sha256": _sha256(pool_path),
        "game_clock": "original-retail-normal-speed",
        "natural_retail_rng": True,
        "serial_wine_workers": 1,
        "episodes": [evidence[index] for index in sorted(evidence)],
    }
    if collection_path.is_file():
        if _object(collection_path) != collection:
            raise ValueError("L2m collection ledger differs")
    else:
        _atomic_json(collection_path, collection)
    _work_event(
        work_log,
        "collection-complete",
        collection_ledger=_relative(collection_path),
        collection_ledger_sha256=_sha256(collection_path),
        episodes=len(evidence),
    )

    run_dirs = [_repository_path(evidence[index]["run_dir"]) for index in sorted(evidence)]
    exposure_roots = int(prereg["collection"]["exposure_roots"])
    audit_workers = min(4, len(run_dirs))
    with ProcessPoolExecutor(
        max_workers=audit_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        audited = list(executor.map(
            _audit_run,
            run_dirs,
            [exposure_roots] * len(run_dirs),
        ))
    gate = prereg["gate"]
    pilot_result = _object(_repository_path(paths["source_l2l_result"]))
    pilot_positives = int(
        pilot_result["action_exposure_audit"]["aggregate"]["target_status"]
        ["accepted-label-1"]
    )
    exposure_audit = summarize_audits(
        [row[0] for row in audited],
        [row[1] for row in audited],
        exposure_roots=exposure_roots,
        minimum_complete_groups_per_episode=int(
            gate["minimum_complete_groups_per_episode"]
        ),
        minimum_assignments_per_action=int(
            gate["minimum_assignments_per_action_new_collection"]
        ),
        minimum_no_override_fraction=float(gate["minimum_no_override_fraction"]),
        minimum_full_execution_fraction=float(
            gate["minimum_full_execution_fraction"]
        ),
        maximum_control_dead_end_rate=float(gate["maximum_control_dead_end_rate"]),
        hit_support_diagnostic_minimum=(
            max(
                0,
                int(gate["minimum_train_positives_including_pilot"])
                - pilot_positives,
            )
            + int(gate["minimum_validation_positives"])
        ),
    )
    split_support = _split_support(prereg, audited, pilot_result)
    collection_admitted = (
        all(exposure_audit["gates"].values())
        and all(split_support["gates"].values())
    )
    base_result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "schedule_sha256": _sha256(schedule_path),
        "collection_ledger_path": _relative(collection_path),
        "collection_ledger_sha256": _sha256(collection_path),
        "source_l2l_result_sha256": prereg["sha256_bindings"]["source_l2l_result"],
        "action_exposure_audit": exposure_audit,
        "split_support": split_support,
        "collection_admitted": collection_admitted,
        "parallel_collection_admitted": False,
        "runs_learned_policy": False,
    }
    if not collection_admitted:
        result = {
            **base_result,
            "fit_artifact_path": None,
            "fit_artifact_sha256": None,
            "evaluation": None,
            "decision": "reject-l2m-collection-or-split-support",
            "complete": True,
        }
        _atomic_json(result_path, result)
        _work_event(
            work_log,
            "l2m-stopped-before-fit",
            result_path=_relative(result_path),
            result_sha256=_sha256(result_path),
            decision=result["decision"],
        )
        return result

    pilot_evidence = _pilot_evidence(prereg)
    train_indices = prereg["data"]["new_train_episode_indices"]
    validation_indices = prereg["data"]["new_validation_episode_indices"]
    train_evidence = pilot_evidence + [evidence[index] for index in train_indices]
    validation_evidence = [evidence[index] for index in validation_indices]
    train_dirs = [_repository_path(row["run_dir"]) for row in train_evidence]
    validation_dirs = [_repository_path(row["run_dir"]) for row in validation_evidence]
    fit_settings = prereg["fit"]
    if fit_path.is_file():
        fit_artifact = _object(fit_path)
        if (
            fit_artifact.get("schema") != FIT_ARTIFACT_SCHEMA
            or fit_artifact.get("preregistration_sha256") != _sha256(prereg_path)
            or fit_artifact.get("repository_commit") != commit
        ):
            raise ValueError("L2m frozen fit artifact differs")
        fitted = fit_artifact["fit"]
    else:
        _work_event(work_log, "train-dataset-load-started", episodes=len(train_dirs))
        train = load_action_intention_dataset(
            train_dirs,
            exposure_roots=exposure_roots,
            max_rows=int(fit_settings["max_rows"]),
            workers=min(int(fit_settings["dataset_workers"]), len(train_dirs)),
        )
        _check_dataset_inventory(train.inventory, _expected_inventory(train_evidence))
        if train.schema != DATASET_SCHEMA:
            raise ValueError("L2m train dataset schema changed")
        fitted = fit_action_intention_hazard_models(
            train,
            boosted_rounds=int(fit_settings["boosted_rounds"]),
            maximum_depth=int(fit_settings["maximum_depth"]),
            learning_rate=float(fit_settings["learning_rate"]),
            minimum_child_weight=float(fit_settings["minimum_child_weight"]),
            l2_leaf_regularization=float(fit_settings["l2_leaf_regularization"]),
            maximum_histogram_bins=int(fit_settings["maximum_histogram_bins"]),
            seed=int(fit_settings["seed"]),
            expected_xgboost_version=str(fit_settings["xgboost_version"]),
        )
        if fitted.get("schema") != FIT_SCHEMA or fitted.get("model") != MODEL_KIND:
            raise ValueError("L2m fit identity changed")
        fit_artifact = {
            "schema": FIT_ARTIFACT_SCHEMA,
            "experiment_id": prereg["experiment_id"],
            "repository_commit": commit,
            "preregistration_sha256": _sha256(prereg_path),
            "train_inventory": list(train.inventory),
            "fit": fitted,
            "validation_loaded": False,
            "deployable_policy": False,
        }
        _atomic_json(fit_path, fit_artifact)
        _work_event(
            work_log,
            "train-only-intention-hazard-fit-frozen",
            fit_artifact_path=_relative(fit_path),
            fit_artifact_sha256=_sha256(fit_path),
            train=fitted["train"],
        )

    evaluation_settings = prereg["evaluation"]
    _work_event(
        work_log,
        "untouched-validation-dataset-load-started",
        episodes=len(validation_dirs),
    )
    validation = load_action_intention_dataset(
        validation_dirs,
        exposure_roots=exposure_roots,
        max_rows=int(evaluation_settings["max_rows"]),
        workers=min(int(evaluation_settings["dataset_workers"]), len(validation_dirs)),
    )
    _check_dataset_inventory(
        validation.inventory,
        _expected_inventory(validation_evidence),
    )
    evaluation = evaluate_action_intention_hazard_models(
        fitted,
        validation,
        bootstrap_samples=int(evaluation_settings["bootstrap_samples"]),
        bootstrap_seed=int(evaluation_settings["bootstrap_seed"]),
        calibration_bins=int(evaluation_settings["calibration_bins"]),
        minimum_validation_episodes=int(gate["minimum_validation_episodes"]),
        minimum_validation_positives=int(gate["minimum_validation_positives"]),
        minimum_validation_negatives=int(gate["minimum_validation_negatives"]),
        minimum_positive_episodes=int(gate["minimum_validation_positive_episodes"]),
        minimum_episodes_favoring_full=int(
            gate["minimum_validation_episodes_favoring_full"]
        ),
        maximum_calibration_in_the_large_absolute=float(
            gate["maximum_calibration_in_the_large_absolute"]
        ),
        maximum_expected_calibration_error=float(
            gate["maximum_expected_calibration_error"]
        ),
        maximum_full_ece_over_state_only=float(
            gate["maximum_full_ece_over_state_only"]
        ),
        maximum_raw_clipped_fraction=float(
            gate["maximum_raw_clipped_fraction"]
        ),
    )
    result = {
        **base_result,
        "fit_artifact_path": _relative(fit_path),
        "fit_artifact_sha256": _sha256(fit_path),
        "fit_schema": FIT_SCHEMA,
        "evaluation_schema": EVALUATION_SCHEMA,
        "train_inventory": list(fit_artifact["train_inventory"]),
        "validation_inventory": list(validation.inventory),
        "evaluation": evaluation,
        "decision": evaluation["summary"]["decision"],
        "complete": True,
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "intention-hazard-evaluation-completed",
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
        default=REPOSITORY / "experiments/l2m-stage4-intention-hazard-v1.json",
    )
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args().preregistration), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
