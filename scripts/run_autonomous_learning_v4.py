#!/usr/bin/env python3
"""Run the fixed, resumable Generation-4 Wine-only learning experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.authorize_sequential_r_canary import authorize  # noqa: E402
from scripts.run_autonomous_learning_v3 import (  # noqa: E402
    _archive_incomplete,
    _atomic_json,
    _baseline_state,
    _candidate_metrics,
    _candidate_runtime_clean,
    _complete_run,
    _sha256,
)
from scripts.run_generation4_preflight import (  # noqa: E402
    HISTORICAL,
    SEEDS,
    _contract_sha256,
    _validate_historical_contract,
    _validate_seed_contract,
    run as run_preflight,
)
from scripts.shadow_sequential_r_critic import shadow  # noqa: E402
from th06_rl.advantage_learning import _object, _validate_run  # noqa: E402
from th06_rl.policies.propensity_aware_option_exploration import (  # noqa: E402
    INCUMBENT_MASS,
    INFORMATION_MASS,
    OPTION_HORIZON_FRAMES,
    STATE_SCHEMA as EXPLORATION_STATE_SCHEMA,
    UNIFORM_MASS,
)
from th06_rl.sequential_learning import (  # noqa: E402
    STATE_SCHEMA as CANDIDATE_STATE_SCHEMA,
    TRANSITION_SCHEMA,
)


SCHEMA = "autonomous-wine-learning-generation-v4"
INFRA_MIGRATIONS = REPOSITORY / "config/autonomous_generation4_infra_migrations.json"
DIFFICULTY = "lunatic"
STAGE = 6
GENERATION_SEED = 260812
NEW_COLLECTION_EPISODES = 16
FIT_BOUNDARIES = (8, 12, 16)
CANARY_PAIRS = 3
FINAL_PAIRS = 12
SHADOW_EPISODES = 3
MAXIMUM_P95_MS = 4.0
EXPLORATION_PLUGIN = (
    REPOSITORY / "src/th06_rl/policies/propensity_aware_option_exploration.py"
)
BASELINE_PLUGIN = REPOSITORY / "src/th06_rl/policies/uniform_safe_exploration.py"
CANDIDATE_PLUGIN = (
    REPOSITORY / "src/th06_rl/policies/autonomous_sequential_r_critic.py"
)


def _reconcile_resume_contract(
    state: dict[str, Any],
    config: dict[str, object],
) -> bool:
    if state.get("schema") != SCHEMA:
        raise RuntimeError("refusing Generation-4 resume with changed schema")
    previous = state.get("config")
    if previous == config:
        return False
    if not isinstance(previous, dict) or state.get("status") != "infra_failure":
        raise RuntimeError("refusing Generation-4 resume with changed contract")
    changed = {
        key for key in set(previous) | set(config)
        if previous.get(key) != config.get(key)
    }
    migrations = _object(INFRA_MIGRATIONS).get("migrations")
    if changed != {"preflight_contract_sha256"} or not isinstance(migrations, list):
        raise RuntimeError("refusing Generation-4 resume with outcome contract drift")
    migration = next((
        row for row in migrations
        if isinstance(row, dict)
        and row.get("from_preflight_contract_sha256")
        == previous.get("preflight_contract_sha256")
        and row.get("to_preflight_contract_sha256")
        == config.get("preflight_contract_sha256")
    ), None)
    if migration is None:
        raise RuntimeError("Generation-4 infra migration is not predeclared")
    log = state.setdefault("infra_migrations", [])
    if not isinstance(log, list):
        raise TypeError("Generation-4 infra migration log is invalid")
    log.append({
        "schema": "autonomous-generation-4-infra-migration-v1",
        "id": str(migration["id"]),
        "reason": str(migration["reason"]),
        "from_preflight_contract_sha256": previous["preflight_contract_sha256"],
        "to_preflight_contract_sha256": config["preflight_contract_sha256"],
        "preserved_new_collection_episodes": len(state.get("episodes", ())),
        "preserved_transition_schema": TRANSITION_SCHEMA,
        "outcome_or_schedule_fields_changed": False,
        "triggering_failure": state.get("infra_failure"),
        "migration_manifest_sha256": _sha256(INFRA_MIGRATIONS),
    })
    state["config"] = config
    return True


def _option_state(
    path: Path,
    *,
    policy_seed: int,
    information_policy: Path | None,
) -> None:
    information = None
    information_sha = None
    if information_policy is not None:
        information = _object(information_policy)
        authorization = information.get("authorization")
        if (
            information.get("schema") != CANDIDATE_STATE_SCHEMA
            or information.get("mode") != "shadow"
            or not isinstance(authorization, dict)
            or authorization.get("fit_eligible") is not True
        ):
            raise ValueError("exploration information critic is not fit-authorized")
        information_sha = _sha256(information_policy)
    value = {
        "schema": EXPLORATION_STATE_SCHEMA,
        "policy_seed": policy_seed,
        "option_horizon_frames": OPTION_HORIZON_FRAMES,
        "mixture": {
            "incumbent": INCUMBENT_MASS,
            "uniform": UNIFORM_MASS,
            "information": INFORMATION_MASS,
        },
        "information_policy": information,
        "information_policy_sha256": information_sha,
    }
    if path.is_file() and _object(path) != value:
        raise ValueError("existing Generation-4 behavior state differs")
    if not path.exists():
        _atomic_json(path, value)


def _config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "difficulty": DIFFICULTY,
        "stage": STAGE,
        "generation_seed": GENERATION_SEED,
        "frozen_historical_episodes": 13,
        "new_collection_episodes": NEW_COLLECTION_EPISODES,
        "fit_boundaries": list(FIT_BOUNDARIES),
        "shadow_episodes": SHADOW_EPISODES,
        "canary_pairs": CANARY_PAIRS,
        "final_pairs": FINAL_PAIRS,
        "maximum_p95_ms": MAXIMUM_P95_MS,
        "seed_schedule_sha256": _sha256(SEEDS),
        "historical_contract_sha256": _sha256(HISTORICAL),
        "preflight_contract_sha256": _contract_sha256(),
        "wine_native_scorer_sha256": _sha256(args.wine_native_scorer),
        "host_native_scorer_sha256": _sha256(args.host_native_scorer),
    }


def _fit(
    args: argparse.Namespace,
    run_dirs: list[Path],
    round_dir: Path,
) -> tuple[Path, dict[str, object]]:
    state_path = round_dir / "policy-shadow.json"
    report_path = round_dir / "report.json"
    if state_path.is_file() and report_path.is_file():
        state = _object(state_path)
        report = _object(report_path)
        if (
            state.get("schema") != CANDIDATE_STATE_SCHEMA
            or report.get("authorization") != state.get("authorization")
        ):
            raise ValueError("cached Generation-4 fit is inconsistent")
        return state_path, report
    if round_dir.exists():
        _archive_incomplete(round_dir)
    command = [
        sys.executable,
        str(REPOSITORY / "scripts/fit_sequential_r_critic.py"),
        *(str(path) for path in run_dirs),
        "--output-dir", str(round_dir),
        "--native-scorer", str(args.wine_native_scorer),
        "--compatible-native-scorer", str(args.host_native_scorer),
        "--seed", str(GENERATION_SEED),
        "--threads", str(args.threads),
    ]
    completed = subprocess.run(command, cwd=REPOSITORY, check=False)
    if completed.returncode:
        raise RuntimeError(f"Generation-4 fit failed with {completed.returncode}")
    return state_path, _object(report_path)


def _round_seeds(schedule: dict[str, object], round_index: int) -> list[int]:
    rows = [
        row for row in schedule["canary"]
        if isinstance(row, dict) and int(row["round"]) == round_index
    ]
    rows.sort(key=lambda row: int(row["pair"]))
    if len(rows) != CANARY_PAIRS:
        raise ValueError("Generation-4 canary seed round is incomplete")
    return [int(row["game_rng_seed"]) for row in rows]


def _paired_canary(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    output_root: Path,
    row: dict[str, Any],
    candidate_state: Path,
    schedule: dict[str, object],
) -> Path | None:
    round_index = int(row["round"])
    root = output_root / "canary" / f"round-{round_index:02d}"
    audit_path = root / "audit.json"
    evaluation_path = root / "policy-full-evaluation.json"
    if audit_path.is_file():
        audit = _object(audit_path)
        if audit.get("candidate_state_sha256") != _sha256(candidate_state):
            raise ValueError("cached Generation-4 canary audit is stale")
        if audit.get("canary_eligible") is True:
            if not evaluation_path.is_file():
                raise FileNotFoundError(evaluation_path)
            row.update({"status": "canary-passed", "canary_audit": str(audit_path)})
            _atomic_json(state_path, state)
            return evaluation_path
        row.update({"status": "canary-rejected", "canary_audit": str(audit_path)})
        _atomic_json(state_path, state)
        return None
    baseline_state = root / "baseline-state.json"
    _baseline_state(baseline_state)
    corpus = root / "non-training-corpus"
    runs = []
    for pair, seed in enumerate(_round_seeds(schedule, round_index)):
        for arm in ("baseline", "candidate"):
            artifact = root / f"pair-{pair:02d}-{arm}"
            report, run_dir = _complete_run(
                artifact_dir=artifact,
                policy_plugin=(BASELINE_PLUGIN if arm == "baseline" else CANDIDATE_PLUGIN),
                policy_state=(baseline_state if arm == "baseline" else candidate_state),
                scorer=(None if arm == "baseline" else args.wine_native_scorer),
                rng_seed=seed,
                corpus_root=corpus,
                transition_schema=TRANSITION_SCHEMA,
            )
            assert run_dir is not None
            completion = report["controller_completion"]
            metrics = _candidate_metrics(report) if arm == "candidate" else {}
            runs.append({
                "pair": pair,
                "arm": arm,
                "game_rng_seed": seed,
                "artifact_dir": str(artifact),
                "corpus_run_dir": str(run_dir),
                "physical_hits": int(completion["physical_hits"]),
                "active_overrides": int(metrics.get("active_overrides", 0)),
                "runtime_clean": (
                    _candidate_runtime_clean(metrics) if arm == "candidate" else True
                ),
                "decision_latency_p95_ms": metrics.get("decision_latency_p95_ms"),
            })
            row["canary_runs"] = runs
            _atomic_json(state_path, state)
    baseline = {
        int(item["pair"]): int(item["physical_hits"])
        for item in runs if item["arm"] == "baseline"
    }
    candidate = {
        int(item["pair"]): int(item["physical_hits"])
        for item in runs if item["arm"] == "candidate"
    }
    exercised = sum(
        int(item["active_overrides"]) > 0
        for item in runs if item["arm"] == "candidate"
    )
    no_worse = sum(candidate[pair] <= baseline[pair] for pair in baseline)
    gates = {
        "six_clean_complete_stages": len(runs) == CANARY_PAIRS * 2,
        "candidate_exercised_in_two_pairs": exercised >= 2,
        "strictly_fewer_aggregate_hits": sum(candidate.values()) < sum(baseline.values()),
        "candidate_no_worse_in_two_pairs": no_worse >= 2,
        "candidate_runtime_clean": all(
            bool(item["runtime_clean"]) for item in runs if item["arm"] == "candidate"
        ),
    }
    audit = {
        "schema": "autonomous-generation-4-paired-canary-v1",
        "candidate_state_sha256": _sha256(candidate_state),
        "seed_schedule_sha256": _sha256(SEEDS),
        "runs": runs,
        "baseline_total_hits": sum(baseline.values()),
        "candidate_total_hits": sum(candidate.values()),
        "effect": sum(baseline.values()) - sum(candidate.values()),
        "exercised_pairs": exercised,
        "no_worse_pairs": no_worse,
        "gates": gates,
        "canary_eligible": all(gates.values()),
    }
    _atomic_json(audit_path, audit)
    row.update({
        "status": "canary-passed" if audit["canary_eligible"] else "canary-rejected",
        "canary_audit": str(audit_path),
        "canary_eligible": audit["canary_eligible"],
    })
    _atomic_json(state_path, state)
    if not audit["canary_eligible"]:
        return None
    evaluation = _object(candidate_state)
    evaluation["authorization"]["full_evaluation"] = {
        "schema": "autonomous-generation-4-full-evaluation-authorization-v1",
        "canary_audit_sha256": _sha256(audit_path),
        "candidate_canary_state_sha256": _sha256(candidate_state),
        "fixed_rng_effect": audit["effect"],
    }
    _atomic_json(evaluation_path, evaluation)
    return evaluation_path


def _process_round(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    output_root: Path,
    all_run_dirs: list[Path],
    new_run_dirs: list[Path],
    round_index: int,
    schedule: dict[str, object],
) -> Path | None:
    round_dir = output_root / "rounds" / f"round-{round_index:02d}"
    shadow_state, fit_report = _fit(args, all_run_dirs, round_dir)
    while len(state["rounds"]) < round_index:
        state["rounds"].append({"round": len(state["rounds"]) + 1})
    row = state["rounds"][round_index - 1]
    authorization = fit_report.get("authorization")
    fit_eligible = (
        isinstance(authorization, dict)
        and authorization.get("fit_eligible") is True
    )
    row.update({
        "round": round_index,
        "historical_episodes": len(all_run_dirs) - len(new_run_dirs),
        "new_episodes": len(new_run_dirs),
        "total_episodes": len(all_run_dirs),
        "fit_dir": str(round_dir),
        "fit_eligible": fit_eligible,
        "status": "fit-ineligible",
    })
    _atomic_json(state_path, state)
    if not fit_eligible:
        return None
    audit_runs = new_run_dirs[-SHADOW_EPISODES:]
    shadow_path = round_dir / "shadow-audit.json"
    if shadow_path.is_file():
        shadow_report = _object(shadow_path)
    else:
        shadow_report = shadow(
            shadow_state,
            audit_runs,
            native_scorer=args.host_native_scorer,
            maximum_p95_ms=MAXIMUM_P95_MS,
        )
        _atomic_json(shadow_path, shadow_report)
    row.update({
        "shadow_audit": str(shadow_path),
        "shadow_eligible": bool(shadow_report["shadow_eligible"]),
        "status": "shadow-ineligible",
    })
    _atomic_json(state_path, state)
    if not row["shadow_eligible"]:
        return None
    # This fit-authorized shadow may guide only information allocation in later
    # collection; it never publishes an action through the exploration policy.
    state["latest_information_policy"] = str(shadow_state)
    _atomic_json(state_path, state)
    active_path = round_dir / "policy-canary.json"
    if not active_path.is_file():
        _atomic_json(active_path, authorize(shadow_state, shadow_path))
    row.update({"status": "canary-running", "canary_state": str(active_path)})
    _atomic_json(state_path, state)
    return _paired_canary(
        args, state, state_path, output_root, row, active_path, schedule
    )


def _rate_ratio(candidate: int, baseline: int) -> dict[str, float]:
    adjusted_candidate = candidate + 0.5
    adjusted_baseline = baseline + 0.5
    ratio = adjusted_candidate / adjusted_baseline
    error = math.sqrt(1.0 / adjusted_candidate + 1.0 / adjusted_baseline)
    return {
        "estimate": ratio,
        "approximate_95_percent_lower": math.exp(math.log(ratio) - 1.96 * error),
        "approximate_95_percent_upper": math.exp(math.log(ratio) + 1.96 * error),
    }


def _full_stage_ab(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    output_root: Path,
    candidate_state: Path,
) -> dict[str, object]:
    root = output_root / "full-stage"
    report_path = root / "report.json"
    if report_path.is_file():
        return _object(report_path)
    baseline_state = root / "baseline-state.json"
    _baseline_state(baseline_state)
    runs = []
    arms = ("baseline", "candidate") * FINAL_PAIRS
    for trial, arm in enumerate(arms):
        artifact = root / f"trial-{trial:02d}-{arm}"
        report, run_dir = _complete_run(
            artifact_dir=artifact,
            policy_plugin=(BASELINE_PLUGIN if arm == "baseline" else CANDIDATE_PLUGIN),
            policy_state=(baseline_state if arm == "baseline" else candidate_state),
            scorer=(None if arm == "baseline" else args.wine_native_scorer),
            rng_seed=None,
            corpus_root=None,
        )
        assert run_dir is None
        completion = report["controller_completion"]
        metrics = _candidate_metrics(report) if arm == "candidate" else {}
        runs.append({
            "trial": trial,
            "arm": arm,
            "artifact_dir": str(artifact),
            "physical_hits": int(completion["physical_hits"]),
            "active_overrides": int(metrics.get("active_overrides", 0)),
            "runtime_clean": (
                _candidate_runtime_clean(metrics) if arm == "candidate" else True
            ),
            "decision_latency_p95_ms": metrics.get("decision_latency_p95_ms"),
        })
        state["full_stage"] = {"status": "running", "runs": runs}
        _atomic_json(state_path, state)
    baseline_total = sum(
        int(row["physical_hits"]) for row in runs if row["arm"] == "baseline"
    )
    candidate_total = sum(
        int(row["physical_hits"]) for row in runs if row["arm"] == "candidate"
    )
    exercised_stages = sum(
        int(row["active_overrides"]) > 0
        for row in runs if row["arm"] == "candidate"
    )
    gates = {
        "all_24_stages_complete": len(runs) == FINAL_PAIRS * 2,
        "strictly_fewer_candidate_hits": candidate_total < baseline_total,
        "candidate_exercised_in_six_stages": exercised_stages >= 6,
        "candidate_runtime_clean": all(
            bool(row["runtime_clean"]) for row in runs if row["arm"] == "candidate"
        ),
    }
    result = {
        "schema": "autonomous-generation-4-natural-full-stage-ab-v1",
        "evaluation_mode": "normal-speed-natural-complete-stage-hit-continuation",
        "fixed_rng": False,
        "trial_order": list(arms),
        "runs": runs,
        "baseline_total_hits": baseline_total,
        "candidate_total_hits": candidate_total,
        "effect": baseline_total - candidate_total,
        "candidate_exercised_stages": exercised_stages,
        "candidate_total_overrides": sum(
            int(row["active_overrides"])
            for row in runs if row["arm"] == "candidate"
        ),
        "hit_rate_ratio": _rate_ratio(candidate_total, baseline_total),
        "gates": gates,
        "verdict": "effective" if all(gates.values()) else "ineffective",
        "rule": (
            "strict aggregate HIT reduction, >=6 exercised candidate Stages, "
            "and clean runtime across all predeclared natural-RNG trials"
        ),
    }
    _atomic_json(report_path, result)
    state["full_stage"] = {"status": "complete", "report": str(report_path), "runs": runs}
    _atomic_json(state_path, state)
    return result


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    preflight = run_preflight(
        output_root / "preflight",
        threads=args.threads,
        seconds=45.0,
        wine_scorer=args.wine_native_scorer,
        host_scorer=args.host_native_scorer,
    )
    if preflight.get("passed") is not True:
        raise RuntimeError("Generation-4 preflight did not pass")
    schedule = _validate_seed_contract()
    historical = _validate_historical_contract()
    state_path = output_root / "generation.json"
    config = _config(args)
    if state_path.is_file():
        state = _object(state_path)
        if _reconcile_resume_contract(state, config):
            _atomic_json(state_path, state)
        if state.get("status") == "complete":
            print(json.dumps(state["decision"], sort_keys=True))
            return 0
    else:
        state = {
            "schema": SCHEMA,
            "status": "collecting",
            "config": config,
            "preflight": str(output_root / "preflight/preflight.json"),
            "historical_runs": [str(path) for path in historical],
            "episodes": [],
            "rounds": [],
            "latest_information_policy": None,
            "full_stage": None,
            "decision": None,
        }
        _atomic_json(state_path, state)
    try:
        state.pop("infra_failure", None)
        new_runs = [Path(row["corpus_run_dir"]) for row in state["episodes"]]
        for run_dir in new_runs:
            _validate_run(run_dir, transition_schema=TRANSITION_SCHEMA)
        collection = schedule["collection"]
        candidate = None
        for round_index, boundary in enumerate(FIT_BOUNDARIES, start=1):
            while len(state["episodes"]) < boundary:
                index = len(state["episodes"])
                seed_row = collection[index]
                information_raw = state.get("latest_information_policy")
                information = Path(information_raw) if isinstance(information_raw, str) else None
                behavior_state = output_root / "behavior-states" / f"episode-{index:03d}.json"
                _option_state(
                    behavior_state,
                    policy_seed=int(seed_row["policy_seed"]),
                    information_policy=information,
                )
                artifact = output_root / "collection" / f"episode-{index:03d}"
                report, run_dir = _complete_run(
                    artifact_dir=artifact,
                    policy_plugin=EXPLORATION_PLUGIN,
                    policy_state=behavior_state,
                    scorer=(args.wine_native_scorer if information is not None else None),
                    rng_seed=int(seed_row["game_rng_seed"]),
                    corpus_root=output_root / "collection-corpus",
                    transition_schema=TRANSITION_SCHEMA,
                )
                assert run_dir is not None
                completion = report["controller_completion"]
                state["episodes"].append({
                    "episode": index,
                    "artifact_dir": str(artifact),
                    "corpus_run_dir": str(run_dir),
                    "game_rng_seed": int(seed_row["game_rng_seed"]),
                    "policy_seed": int(seed_row["policy_seed"]),
                    "information_policy_sha256": (
                        _sha256(information) if information is not None else None
                    ),
                    "physical_hits": int(completion["physical_hits"]),
                })
                state["status"] = "collecting"
                _atomic_json(state_path, state)
                new_runs.append(run_dir)
            candidate = _process_round(
                args,
                state,
                state_path,
                output_root,
                [*historical, *new_runs],
                new_runs,
                round_index,
                schedule,
            )
            if candidate is not None:
                break
        if candidate is None:
            decision = {
                "verdict": "ineffective",
                "reason": "16-new-Stage evidence budget ended without canary authorization",
                "historical_episodes": len(historical),
                "new_collection_episodes": len(state["episodes"]),
                "rounds": len(state["rounds"]),
            }
        else:
            state["status"] = "full-stage-evaluation"
            _atomic_json(state_path, state)
            final = _full_stage_ab(args, state, state_path, output_root, candidate)
            decision = {
                "verdict": final["verdict"],
                "reason": "predeclared natural complete-Stage physical HIT gates",
                "baseline_total_hits": final["baseline_total_hits"],
                "candidate_total_hits": final["candidate_total_hits"],
                "effect": final["effect"],
                "candidate_exercised_stages": final["candidate_exercised_stages"],
                "hit_rate_ratio": final["hit_rate_ratio"],
            }
        state.update({"status": "complete", "decision": decision})
        _atomic_json(state_path, state)
        print(json.dumps(decision, sort_keys=True))
        return 0
    except BaseException as error:
        state["status"] = "infra_failure"
        state["infra_failure"] = f"{type(error).__name__}: {error}"
        _atomic_json(state_path, state)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY / "artifacts/autonomous-wine-generation-4",
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--wine-native-scorer",
        type=Path,
        default=REPOSITORY / "build/native-win32-fully-static/libth06_rl_ranker.dll",
    )
    parser.add_argument(
        "--host-native-scorer",
        type=Path,
        default=REPOSITORY / "build/native/libth06_rl_ranker.so",
    )
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("thread count must be positive")
    if not args.wine_native_scorer.is_file() or not args.host_native_scorer.is_file():
        parser.error("native scorer library is absent")
    args.wine_native_scorer = args.wine_native_scorer.resolve()
    args.host_native_scorer = args.host_native_scorer.resolve()
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
