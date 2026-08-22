#!/usr/bin/env python3
"""Collect and audit the frozen L2k four-root action-exposure pilot."""

from __future__ import annotations

import argparse
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
from scripts.run_l1_stage4 import (  # noqa: E402
    _relative,
    _repository_path,
    _require_clean_worktree,
    _work_event,
)
from th06_rl.action_exposure_audit import (  # noqa: E402
    audit_episode,
    summarize_action_exposure_audits,
)
from th06_rl.policies.fixed_shield_action_exposure import (  # noqa: E402
    STATE_SCHEMA as EXPOSURE_STATE_SCHEMA,
)


PREREG_SCHEMA = "th06-rl-l2k-stage4-action-exposure-prereg-v1"
SCHEDULE_SCHEMA = "th06-rl-l2k-stage4-action-exposure-schedule-v1"
COLLECTION_SCHEMA = "th06-rl-l2k-stage4-action-exposure-collection-v1"
RESULT_SCHEMA = "th06-rl-l2k-stage4-action-exposure-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("L2k preregistration schema mismatch")
    for key in ("collection", "gate", "paths", "sha256_bindings"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"L2k preregistration lacks {key}")
    collection = prereg["collection"]
    gate = prereg["gate"]
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    episodes = collection.get("episodes")
    if (
        not isinstance(episodes, list)
        or len(episodes) != 2
        or collection.get("stage") != 4
        or collection.get("difficulty") != "lunatic"
        or collection.get("episode_unit") != "complete-practice-stage"
        or collection.get("assignment")
        != "uniform-over-current-observed-shield-set-at-group-start"
        or collection.get("exposure_roots") != 4
        or collection.get("continuation")
        != "retain-intended-action-when-currently-shield-admissible"
        or collection.get("override")
        != "deterministic-current-reactive-baseline-when-intention-is-not-currently-shield-admissible"
        or collection.get("inflight_shield_loss")
        != "existing-fail-closed-control-dead-end-and-input-release"
        or collection.get("serial_wine_workers") != 1
        or collection.get("worker_index") != 0
        or collection.get("natural_retail_rng") is not True
        or collection.get("game_clock") != "original-retail-normal-speed"
        or collection.get("complete_episode_required") is not True
        or collection.get("physical_hit_retry") is not False
        or collection.get("max_attempts_per_slot") != 3
        or collection.get("peek_or_sequential_stop") is not False
    ):
        raise ValueError("L2k collection contract changed")
    if collection.get("infrastructure_admission") != {
        "capture_p99_ms_max": 40.0,
        "dense_shield_parity_required": True,
        "observation_gap_rate_max": 0.005,
        "player_successor_bit_exact_required": True,
        "solve_p99_ms_max": 16.7,
        "stale_retry_rate_max": 0.0,
        "validator": "scripts.gate_parallel_wine.validate_gate_run",
    }:
        raise ValueError("L2k infrastructure admission changed")
    domain = str(collection.get("policy_seed_derivation_domain"))
    base_seed = int(collection.get("base_policy_seed", -1))
    for index, row in enumerate(episodes):
        if (
            not isinstance(row, dict)
            or int(row.get("index", -1)) != index
            or row.get("split") != "pilot-train-only"
            or int(row.get("policy_seed", -1))
            != l1._derived_policy_seed(domain, base_seed, index)
        ):
            raise ValueError("L2k episode seed or split changed")
    expected_gate = {
        "minimum_complete_groups_per_episode": 4000,
        "minimum_assignments_per_action": 100,
        "minimum_no_override_fraction": 0.8,
        "minimum_four_execution_fraction": 0.8,
        "maximum_control_dead_end_rate": 0.005,
        "h16_support_diagnostic_minimum": 64,
        "h16_support_is_retry_or_contract_gate": False,
        "zero_bomb_required": True,
        "zero_infrastructure_failure_required": True,
        "exact_exposure_metadata_required": True,
        "complete_episode_required": True,
        "parallel_collection_admitted": False,
        "hazard_model_admitted": False,
        "value_learning_admitted": False,
        "online_learned_policy_admitted": False,
    }
    if gate != expected_gate:
        raise ValueError("L2k gate changed")
    if prereg.get("fits_model") is not False or prereg.get("runs_learned_policy") is not False:
        raise ValueError("L2k may not fit or run a learned policy")
    bound_files = (
        "base_policy_state",
        "exposure_policy_plugin",
        "policy_api",
        "policy_loader",
        "controller",
        "corpus_module",
        "episode_dataset_module",
        "exposure_audit_module",
        "experiment_runner",
        "source_l2j_preregistration",
        "source_l2j_result",
        "wine_pool",
    )
    for key in paths:
        _repository_path(paths[key])
    for key in bound_files:
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L2k input differs: {key}")
    state = _object(_repository_path(paths["base_policy_state"]))
    if (
        state.get("schema") != EXPOSURE_STATE_SCHEMA
        or state.get("exposure_roots") != 4
    ):
        raise ValueError("L2k base exposure policy differs")
    source = _object(_repository_path(paths["source_l2j_result"]))
    if (
        source.get("complete") is not True
        or source.get("decision")
        != "reject-logscore-observed-primitive-h16-hazard"
    ):
        raise ValueError("L2k source is not the immutable L2j rejection")
    return prereg


def derive_episode_policy_state(
    prereg: dict[str, Any], *, episode: int
) -> dict[str, object]:
    state = _object(_repository_path(prereg["paths"]["base_policy_state"]))
    state["policy_seed"] = int(prereg["collection"]["episodes"][episode]["policy_seed"])
    return state


def build_schedule(
    prereg: dict[str, Any],
    *,
    prereg_path: Path,
    pool_path: Path,
    commit: str,
    work_log_path: Path,
) -> dict[str, object]:
    rows = []
    for raw in prereg["collection"]["episodes"]:
        index = int(raw["index"])
        state = derive_episode_policy_state(prereg, episode=index)
        name = f"stage4-action-exposure-{index:03d}"
        rows.append({
            **raw,
            "name": name,
            "policy_state_path": f"policy-states/{name}.json",
            "policy_state_sha256": hashlib.sha256(l1._canonical(state)).hexdigest(),
        })
    return {
        "schema": SCHEDULE_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "pool_sha256": _sha256(pool_path),
        "work_log_path": _relative(work_log_path),
        "game_clock": "original-retail-normal-speed",
        "natural_retail_rng": True,
        "serial_wine_workers": 1,
        "episodes": rows,
    }


def _audit_kwargs(prereg: dict[str, Any]) -> dict[str, object]:
    gate = prereg["gate"]
    return {
        "exposure_roots": int(prereg["collection"]["exposure_roots"]),
        "minimum_complete_groups_per_episode": int(
            gate["minimum_complete_groups_per_episode"]
        ),
        "minimum_assignments_per_action": int(gate["minimum_assignments_per_action"]),
        "minimum_no_override_fraction": float(gate["minimum_no_override_fraction"]),
        "minimum_four_execution_fraction": float(
            gate["minimum_four_execution_fraction"]
        ),
        "maximum_control_dead_end_rate": float(gate["maximum_control_dead_end_rate"]),
        "h16_support_diagnostic_minimum": int(
            gate["h16_support_diagnostic_minimum"]
        ),
    }


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    paths = prereg["paths"]
    pool_path = _repository_path(paths["wine_pool"])
    pool = _pool(pool_path)
    workers = sorted(pool["workers"], key=lambda item: int(item["worker"]))
    worker = next(
        item for item in workers
        if int(item["worker"]) == int(prereg["collection"]["worker_index"])
    )
    artifact_root = _repository_path(paths["artifact_root"])
    corpus_root = _repository_path(paths["corpus_root"])
    collection_path = _repository_path(paths["collection_ledger"])
    result_path = _repository_path(paths["experiment_result"])
    if result_path.is_file():
        result = _object(result_path)
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("preregistration_sha256") != _sha256(prereg_path)
        ):
            raise ValueError("completed L2k result differs")
        return result

    commit = _repository_commit()
    schedule_path = artifact_root / "schedule.json"
    if schedule_path.is_file():
        existing = _object(schedule_path)
        work_log = _repository_path(existing.get("work_log_path"))
        schedule = build_schedule(
            prereg,
            prereg_path=prereg_path,
            pool_path=pool_path,
            commit=commit,
            work_log_path=work_log,
        )
        if existing != schedule:
            raise ValueError("L2k resume schedule differs")
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("L2k artifact root lacks its immutable schedule")
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
        schedule = build_schedule(
            prereg,
            prereg_path=prereg_path,
            pool_path=pool_path,
            commit=commit,
            work_log_path=work_log,
        )
        _atomic_json(schedule_path, schedule)
    _work_event(
        work_log,
        "experiment-started-or-resumed",
        schedule_path=_relative(schedule_path),
        schedule_sha256=_sha256(schedule_path),
    )

    policy_plugin = _repository_path(paths["exposure_policy_plugin"])
    for row in schedule["episodes"]:
        index = int(row["index"])
        state = derive_episode_policy_state(prereg, episode=index)
        state_path = artifact_root / str(row["policy_state_path"])
        if state_path.is_file():
            if _object(state_path) != state:
                raise ValueError(f"L2k episode policy state differs: {state_path}")
        else:
            _atomic_json(state_path, state)
        if _sha256(state_path) != row["policy_state_sha256"]:
            raise ValueError(f"L2k episode policy state hash differs: {state_path}")

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
            raise ValueError("L2k collection ledger differs")
    else:
        _atomic_json(collection_path, collection)
    _work_event(
        work_log,
        "collection-complete",
        collection_ledger=_relative(collection_path),
        collection_ledger_sha256=_sha256(collection_path),
        episodes=len(evidence),
    )

    run_dirs = [
        _repository_path(evidence[index]["run_dir"]) for index in sorted(evidence)
    ]
    exposure_roots = int(prereg["collection"]["exposure_roots"])
    with ProcessPoolExecutor(
        max_workers=len(run_dirs),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(audit_episode, run_dir, exposure_roots=exposure_roots)
            for run_dir in run_dirs
        ]
        episode_audits = [future.result() for future in futures]
    audit = summarize_action_exposure_audits(
        episode_audits,
        **_audit_kwargs(prereg),
    )
    result = {
        "schema": RESULT_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "schedule_sha256": _sha256(schedule_path),
        "collection_ledger_path": _relative(collection_path),
        "collection_ledger_sha256": _sha256(collection_path),
        "inventory": [
            {
                "episode_id": evidence[index]["run_id"],
                "run_sha256": evidence[index]["run_sha256"],
                "manifest_sha256": evidence[index]["manifest_sha256"],
                "split": "pilot-train-only",
            }
            for index in sorted(evidence)
        ],
        "action_exposure_audit": audit,
        "decision": audit["decision"],
        "pilot_train_data_admitted": all(audit["gates"].values()),
        "parallel_collection_admitted": False,
        "fits_model": False,
        "runs_learned_policy": False,
        "online_wine": "collector-only fixed randomized action exposure",
        "independent_confirmation": False,
        "complete": True,
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "action-exposure-audit-completed",
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
        default=REPOSITORY / "experiments/l2k-stage4-action-exposure-v1.json",
    )
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
