#!/usr/bin/env python3
"""Collect and evaluate the frozen Stage 4 L2d confirmation inventory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from th06_rl.factual_probe_boundary_diagnostics import (  # noqa: E402
    load_boundary_probe_dataset,
)
from th06_rl.factual_probe_confirmation import (  # noqa: E402
    CONFIRMATION_SCHEMA,
    evaluate_frozen_probe_confirmation,
)
from th06_rl.policies.uniform_shield_exploration import (  # noqa: E402
    STATE_SCHEMA as UNIFORM_STATE_SCHEMA,
)


PREREG_SCHEMA = "th06-rl-l2d-stage4-fresh-confirmation-prereg-v1"
SCHEDULE_SCHEMA = "th06-rl-l2d-stage4-fresh-confirmation-schedule-v1"
COLLECTION_SCHEMA = "th06-rl-l2d-stage4-fresh-confirmation-collection-v1"
RESULT_SCHEMA = "th06-rl-l2d-stage4-fresh-confirmation-result-v1"


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("L2d fresh-confirmation preregistration schema mismatch")
    for key in ("collection", "evaluation", "gate", "paths", "sha256_bindings"):
        if not isinstance(prereg.get(key), dict):
            raise ValueError(f"L2d preregistration lacks {key}")
    collection = prereg["collection"]
    evaluation = prereg["evaluation"]
    gate = prereg["gate"]
    paths = prereg["paths"]
    bindings = prereg["sha256_bindings"]
    episodes = collection.get("episodes")
    if (
        not isinstance(episodes, list)
        or len(episodes) != 8
        or collection.get("stage") != 4
        or collection.get("difficulty") != "lunatic"
        or collection.get("serial_wine_workers") != 1
        or collection.get("natural_retail_rng") is not True
        or collection.get("game_clock") != "original-retail-normal-speed"
        or collection.get("exploration_probability") != 0.2
        or collection.get("complete_episode_required") is not True
        or collection.get("physical_hit_retry") is not False
        or collection.get("max_attempts_per_slot") != 3
        or collection.get("peek_or_sequential_stop") is not False
    ):
        raise ValueError("L2d collection contract changed")
    if collection.get("infrastructure_admission") != {
        "capture_p99_ms_max": 40.0,
        "dense_shield_parity_required": True,
        "observation_gap_rate_max": 0.005,
        "player_successor_bit_exact_required": True,
        "solve_p99_ms_max": 16.7,
        "stale_retry_rate_max": 0.0,
        "validator": "scripts.gate_parallel_wine.validate_gate_run",
    }:
        raise ValueError("L2d infrastructure admission changed")
    domain = collection.get("policy_seed_derivation_domain")
    base_seed = int(collection.get("base_policy_seed", -1))
    indices = []
    for row in episodes:
        if not isinstance(row, dict):
            raise ValueError("L2d episode row is malformed")
        index = int(row.get("index", -1))
        if (
            row.get("split") != "independent-confirmation"
            or int(row.get("policy_seed", -1))
            != l1._derived_policy_seed(str(domain), base_seed, index)
        ):
            raise ValueError("L2d episode seed or split changed")
        indices.append(index)
    if indices != list(range(8)):
        raise ValueError("L2d episode indices changed")
    expected_evaluation = {
        "horizons_game_frames": [16, 64],
        "primary_horizon_game_frames": 16,
        "calibration_bins": 10,
        "bootstrap_samples": 2000,
        "bootstrap_seed": 20260824,
        "max_rows_per_horizon": 400000,
        "frozen_full_model": "l2-stage4-factual-probes-v1 train-only fit",
        "frozen_state_only_model": "l2b-incremental-action-v1 train-only fit",
    }
    if evaluation != expected_evaluation:
        raise ValueError("L2d evaluation changed")
    expected_gate = {
        "minimum_overall_hit_positives": 800,
        "minimum_overall_hit_negatives": 100000,
        "minimum_nonbaseline_hit_positives": 100,
        "minimum_prefirst_hit_positives": 64,
        "minimum_episodes_favoring_full": 6,
        "overall_signal": (
            "16-frame complete-episode bootstrap upper endpoint for full minus "
            "state-only Brier is below zero"
        ),
        "nonbaseline_support": (
            "published-differs-from-baseline rows occur in all eight episodes, "
            "meet positive support, and have negative point Brier delta"
        ),
        "prefirst_hit_lifecycle": (
            "pre-first-HIT rows occur in all eight episodes, meet positive support, "
            "and have negative point Brier delta"
        ),
        "calibration_in_the_large_absolute_max": 0.002,
        "full_ece_over_state_only_max": 0.001,
        "history_admitted": False,
        "value_learning_admitted": False,
        "online_policy_admitted": False,
    }
    if gate != expected_gate:
        raise ValueError("L2d confirmation gate changed")
    if prereg.get("fits_model") is not False or prereg.get("runs_learned_policy") is not False:
        raise ValueError("L2d may not fit or run a learned policy")
    for key in (
        "artifact_root",
        "corpus_root",
        "collection_ledger",
        "experiment_result",
        "base_policy_state",
        "uniform_policy_plugin",
        "wine_pool",
        "work_log_root",
        "source_l2_fit",
        "source_l2_result",
        "source_l2b_fit",
        "source_l2b_result",
        "source_l2c_preregistration",
        "source_l2c_result",
        "boundary_module",
        "confirmation_module",
        "confirmation_runner",
    ):
        _repository_path(paths[key])
    for key in (
        "base_policy_state",
        "uniform_policy_plugin",
        "wine_pool",
        "source_l2_fit",
        "source_l2_result",
        "source_l2b_fit",
        "source_l2b_result",
        "source_l2c_preregistration",
        "source_l2c_result",
        "boundary_module",
        "confirmation_module",
        "confirmation_runner",
    ):
        source = _repository_path(paths[key])
        if not source.is_file() or _sha256(source) != bindings.get(key):
            raise ValueError(f"preregistered L2d input differs: {key}")
    base_state = _object(_repository_path(paths["base_policy_state"]))
    if (
        base_state.get("schema") != UNIFORM_STATE_SCHEMA
        or base_state.get("exploration_probability") != 0.2
    ):
        raise ValueError("L2d base exploration policy differs")
    source_l2c = _object(_repository_path(paths["source_l2c_result"]))
    if (
        source_l2c.get("complete") is not True
        or source_l2c.get("decision") != "freeze-fresh-independent-confirmation"
        or source_l2c.get("independent_confirmation") is not False
    ):
        raise ValueError("L2d source is not the frozen L2c boundary result")
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
        name = f"stage4-confirmation-{index:03d}"
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


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    paths = prereg["paths"]
    pool_path = _repository_path(paths["wine_pool"])
    pool = _pool(pool_path)
    workers = sorted(pool["workers"], key=lambda item: int(item["worker"]))
    worker_index = int(prereg["collection"]["worker_index"])
    worker = next(item for item in workers if int(item["worker"]) == worker_index)
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
            raise ValueError("completed L2d result differs")
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
            raise ValueError("L2d resume schedule differs")
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("L2d artifact root lacks its immutable schedule")
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

    policy_plugin = _repository_path(paths["uniform_policy_plugin"])
    for row in schedule["episodes"]:
        index = int(row["index"])
        state = derive_episode_policy_state(prereg, episode=index)
        state_path = artifact_root / str(row["policy_state_path"])
        if state_path.is_file():
            if _object(state_path) != state:
                raise ValueError(f"L2d episode policy state differs: {state_path}")
        else:
            _atomic_json(state_path, state)
        if _sha256(state_path) != row["policy_state_sha256"]:
            raise ValueError(f"L2d episode policy state hash differs: {state_path}")

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
            raise ValueError("L2d collection ledger differs")
    else:
        _atomic_json(collection_path, collection)
    _work_event(
        work_log,
        "collection-complete",
        collection_ledger=_relative(collection_path),
        collection_ledger_sha256=_sha256(collection_path),
        episodes=len(evidence),
    )

    settings = prereg["evaluation"]
    horizons = tuple(int(value) for value in settings["horizons_game_frames"])
    _work_event(work_log, "fresh-confirmation-evaluation-started")
    dataset = load_boundary_probe_dataset(
        tuple(_repository_path(evidence[index]["run_dir"]) for index in sorted(evidence)),
        horizons=horizons,
        max_rows=int(settings["max_rows_per_horizon"]),
    )
    expected_inventory = [
        {
            "episode_id": evidence[index]["run_id"],
            "run_sha256": evidence[index]["run_sha256"],
            "manifest_sha256": evidence[index]["manifest_sha256"],
        }
        for index in sorted(evidence)
    ]
    observed_inventory = [
        {
            "episode_id": row.get("episode_id"),
            "run_sha256": row.get("run_sha256"),
            "manifest_sha256": row.get("manifest_sha256"),
        }
        for row in dataset.factual.inventory
    ]
    if observed_inventory != expected_inventory:
        raise ValueError("L2d evaluation inventory differs from collection")
    full_fit = _object(_repository_path(paths["source_l2_fit"]))
    state_fit = _object(_repository_path(paths["source_l2b_fit"]))
    confirmation = evaluate_frozen_probe_confirmation(
        full_fit["fit"],
        state_fit["fit"],
        dataset,
        primary_horizon=int(settings["primary_horizon_game_frames"]),
        bootstrap_samples=int(settings["bootstrap_samples"]),
        bootstrap_seed=int(settings["bootstrap_seed"]),
        calibration_bins=int(settings["calibration_bins"]),
        minimum_overall_positives=int(prereg["gate"]["minimum_overall_hit_positives"]),
        minimum_overall_negatives=int(prereg["gate"]["minimum_overall_hit_negatives"]),
        minimum_nonbaseline_positives=int(
            prereg["gate"]["minimum_nonbaseline_hit_positives"]
        ),
        minimum_prefirst_hit_positives=int(
            prereg["gate"]["minimum_prefirst_hit_positives"]
        ),
        minimum_episodes_favoring_full=int(
            prereg["gate"]["minimum_episodes_favoring_full"]
        ),
        calibration_in_the_large_absolute_max=float(
            prereg["gate"]["calibration_in_the_large_absolute_max"]
        ),
        full_ece_over_state_only_max=float(
            prereg["gate"]["full_ece_over_state_only_max"]
        ),
    )
    result = {
        "schema": RESULT_SCHEMA,
        "confirmation_schema": CONFIRMATION_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "schedule_sha256": _sha256(schedule_path),
        "collection_ledger_path": _relative(collection_path),
        "collection_ledger_sha256": _sha256(collection_path),
        "source_l2_fit_sha256": prereg["sha256_bindings"]["source_l2_fit"],
        "source_l2b_fit_sha256": prereg["sha256_bindings"]["source_l2b_fit"],
        "inventory": list(dataset.factual.inventory),
        "confirmation": confirmation,
        "decision": confirmation["summary"]["decision"],
        "online_wine": "collector-only; frozen learned probes were evaluated offline",
        "independent_confirmation": True,
        "complete": True,
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "fresh-confirmation-completed",
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
        default=REPOSITORY / "experiments/l2d-stage4-fresh-confirmation-v1.json",
    )
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
