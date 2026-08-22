#!/usr/bin/env python3
"""Run the preregistered offline Stage 4 factual-probe pilot once."""

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
from scripts.run_l1_stage4 import (  # noqa: E402
    _relative,
    _repository_path,
    _require_clean_worktree,
    _work_event,
)
from th06_rl.factual_probes import (  # noqa: E402
    PROBE_FEATURE_NAMES,
    PROBE_FEATURE_SCHEMA,
    evaluate_factual_probe_models,
    fit_factual_probe_models,
    load_factual_probe_dataset,
)


PREREG_SCHEMA = "th06-rl-l2-stage4-factual-probes-prereg-v1"
PLAN_SCHEMA = "th06-rl-l2-stage4-factual-probes-plan-v1"
FIT_SCHEMA = "th06-rl-l2-stage4-factual-probes-fit-v1"
RESULT_SCHEMA = "th06-rl-l2-stage4-factual-probes-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("Stage 4 L2 factual-probe preregistration schema mismatch")
    for key in ("data", "probes", "gate", "paths", "sha256_bindings"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"Stage 4 L2 factual probes lack {key}")
    data = prereg["data"]
    probes = prereg["probes"]
    gate = prereg["gate"]
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    if (
        data.get("reuse_without_mutation") is not True
        or data.get("source_experiment_id") != "l1-stage4-bc-v1"
        or data.get("train_episode_indices") != [0, 1, 3, 4, 6, 7, 9, 10]
        or data.get("validation_episode_indices") != [2, 5, 8, 11]
        or data.get("split_unit") != "complete-physical-episode"
    ):
        raise ValueError("Stage 4 L2 factual-probe data scope changed")
    expected_probes = {
        "decision_epoch_schema": "th06-rl-decision-epoch-v1",
        "feature_schema": PROBE_FEATURE_SCHEMA,
        "feature_names": list(PROBE_FEATURE_NAMES),
        "horizons_game_frames": [1, 4, 16, 64],
        "dynamics_target": "first contiguous raw transition executed delta-x/delta-y",
        "shield_collapse_target": (
            "within horizon: explicit observed-shield control dead end or strict "
            "contraction to a positive shield-action count below the start count; "
            "zero passive/HIT counts alone are excluded"
        ),
        "hit_target": "any exact physical life_lost within the contiguous horizon",
        "risk_model": "standardized-ridge-linear-probability",
        "ridge_l2": 0.001,
        "max_rows_per_split_per_horizon": 400000,
        "validation_use": (
            "load and evaluate validation exactly once after serializing the "
            "train-only fit; never select a feature, horizon, threshold, or model"
        ),
    }
    if probes != expected_probes:
        raise ValueError("Stage 4 L2 factual-probe definition changed")
    expected_gate = {
        "bootstrap_samples": 2000,
        "bootstrap_seed": 20260822,
        "dynamics_executed_to_global_mse_ratio_max": 0.25,
        "execution_match_rate_min": 0.999,
        "mismatch_rows_min": 100,
        "mismatch_executed_to_published_mse_ratio_max": 0.8,
        "minimum_train_positives": 20,
        "minimum_validation_positives": 10,
        "minimum_validation_negatives": 100,
        "risk_proper_score": (
            "upper endpoint of 2000-draw complete-episode bootstrap 95 percent "
            "interval for candidate minus train-prevalence validation Brier is below zero"
        ),
        "proceed_rule": (
            "dynamics gate passes and at least one sufficient HIT horizon and one "
            "sufficient shield-collapse horizon pass the frozen proper-score gate"
        ),
        "history_admitted_by_this_experiment": False,
    }
    if gate != expected_gate:
        raise ValueError("Stage 4 L2 factual-probe gate changed")
    if prereg.get("online_wine") is not False or prereg.get("fits_policy") is not False:
        raise ValueError("Stage 4 L2 pilot may not run Wine or fit a policy")
    for key in (
        "artifact_root",
        "fit_artifact",
        "experiment_plan",
        "experiment_result",
        "work_log_root",
        "source_corpus_root",
        "source_collection_ledger",
        "source_l1_result",
        "source_l1_model",
        "probe_module",
        "probe_runner",
    ):
        _repository_path(paths[key])
    for key in (
        "source_collection_ledger",
        "source_l1_result",
        "source_l1_model",
        "probe_module",
        "probe_runner",
    ):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L2 factual-probe input differs: {key}")
    return prereg


def _run_paths(
    prereg: dict[str, Any],
    inventory: dict[int, dict[str, Any]],
    split: str,
) -> tuple[Path, ...]:
    return tuple(
        _repository_path(inventory[index]["run_dir"])
        for index in prereg["data"][f"{split}_episode_indices"]
    )


def _validate_inventory(
    prereg: dict[str, Any],
    inventory: dict[int, dict[str, Any]],
    split: str,
    observed: tuple[dict[str, object], ...],
) -> None:
    indices = prereg["data"][f"{split}_episode_indices"]
    expected = [
        {
            "episode_id": inventory[index]["run_id"],
            "run_sha256": inventory[index]["run_sha256"],
            "manifest_sha256": inventory[index]["manifest_sha256"],
        }
        for index in indices
    ]
    compact = [
        {
            "episode_id": row.get("episode_id"),
            "run_sha256": row.get("run_sha256"),
            "manifest_sha256": row.get("manifest_sha256"),
        }
        for row in observed
    ]
    if compact != expected:
        raise ValueError(f"Stage 4 L2 {split} whole-episode inventory differs")


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    inventory = l1b.load_source_inventory(prereg)
    paths = prereg["paths"]
    artifact_root = _repository_path(paths["artifact_root"])
    fit_path = _repository_path(paths["fit_artifact"])
    plan_path = _repository_path(paths["experiment_plan"])
    result_path = _repository_path(paths["experiment_result"])
    if result_path.is_file():
        result = _object(result_path)
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("preregistration_sha256") != _sha256(prereg_path)
        ):
            raise ValueError("completed Stage 4 L2 factual-probe result differs")
        return result

    commit = _repository_commit()
    if plan_path.is_file():
        plan = _object(plan_path)
        work_log = _repository_path(plan.get("work_log_path"))
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("Stage 4 L2 artifact root lacks its immutable plan")
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
            "work_log_path": _relative(work_log),
            "online_wine": False,
            "mutates_source_corpus": False,
            "loads_validation_after_train_fit": True,
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
        "online_wine": False,
        "mutates_source_corpus": False,
        "loads_validation_after_train_fit": True,
    }
    if plan != expected_plan:
        raise ValueError("Stage 4 L2 immutable plan differs")

    horizons = tuple(int(value) for value in prereg["probes"]["horizons_game_frames"])
    _work_event(
        work_log,
        "train-load-started",
        repository_commit=commit,
        plan_path=_relative(plan_path),
        plan_sha256=_sha256(plan_path),
    )
    train = load_factual_probe_dataset(
        _run_paths(prereg, inventory, "train"),
        horizons=horizons,
        max_rows=int(prereg["probes"]["max_rows_per_split_per_horizon"]),
    )
    _validate_inventory(prereg, inventory, "train", train.inventory)
    fitted = fit_factual_probe_models(
        train,
        ridge_l2=float(prereg["probes"]["ridge_l2"]),
    )
    fit_artifact = {
        "schema": FIT_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_sha256": _sha256(prereg_path),
        "source_collection_ledger_sha256": prereg["sha256_bindings"][
            "source_collection_ledger"
        ],
        "train_inventory": list(train.inventory),
        "fit": fitted,
        "validation_loaded": False,
        "deployable_policy": False,
    }
    _atomic_json(fit_path, fit_artifact)
    _work_event(
        work_log,
        "train-fit-frozen",
        fit_artifact_path=_relative(fit_path),
        fit_artifact_sha256=_sha256(fit_path),
        dynamics_rows=train.dynamics.rows,
        horizon_rows={str(view.horizon): view.rows for view in train.horizons},
    )

    _work_event(work_log, "validation-load-started")
    validation = load_factual_probe_dataset(
        _run_paths(prereg, inventory, "validation"),
        horizons=horizons,
        max_rows=int(prereg["probes"]["max_rows_per_split_per_horizon"]),
    )
    _validate_inventory(prereg, inventory, "validation", validation.inventory)
    gate = prereg["gate"]
    evaluation = evaluate_factual_probe_models(
        fitted,
        validation,
        dynamics_mse_ratio_max=float(
            gate["dynamics_executed_to_global_mse_ratio_max"]
        ),
        execution_match_rate_min=float(gate["execution_match_rate_min"]),
        mismatch_rows_min=int(gate["mismatch_rows_min"]),
        mismatch_mse_ratio_max=float(
            gate["mismatch_executed_to_published_mse_ratio_max"]
        ),
        minimum_train_positives=int(gate["minimum_train_positives"]),
        minimum_validation_positives=int(gate["minimum_validation_positives"]),
        minimum_validation_negatives=int(gate["minimum_validation_negatives"]),
        bootstrap_samples=int(gate["bootstrap_samples"]),
        bootstrap_seed=int(gate["bootstrap_seed"]),
    )
    result = {
        "schema": RESULT_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "plan_path": _relative(plan_path),
        "plan_sha256": _sha256(plan_path),
        "fit_artifact_path": _relative(fit_path),
        "fit_artifact_sha256": _sha256(fit_path),
        "train_inventory": list(train.inventory),
        "validation_inventory": list(validation.inventory),
        "evaluation": evaluation,
        "decision": evaluation["summary"]["decision"],
        "online_wine": None,
        "complete": True,
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "experiment-completed",
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
        default=REPOSITORY / "experiments/l2-stage4-factual-probes-v1.json",
    )
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
