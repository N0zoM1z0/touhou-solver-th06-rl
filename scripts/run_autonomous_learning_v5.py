#!/usr/bin/env python3
"""Run the resumable Generation-5 Stage-4 -> 5 -> 6 Wine curriculum."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.authorize_supported_implicit_q_canary import authorize  # noqa: E402
from scripts.run_autonomous_learning_v3 import (  # noqa: E402
    _archive_incomplete,
    _atomic_json,
    _candidate_metrics,
    _candidate_runtime_clean,
)
from scripts.run_generation5_wine import (  # noqa: E402
    complete_run,
    normalized_option_sha256,
)
from scripts.shadow_supported_implicit_q import shadow  # noqa: E402
from th06_rl.advantage_learning import _object, _validate_run  # noqa: E402
from th06_rl.curriculum_contract import load_curriculum_schedule  # noqa: E402
from th06_rl.implicit_learning import STATE_SCHEMA as CANDIDATE_SCHEMA  # noqa: E402
from th06_rl.policies.propensity_aware_option_exploration import (  # noqa: E402
    INCUMBENT_MASS,
    INFORMATION_MASS,
    OPTION_HORIZON_FRAMES,
    STATE_SCHEMA as BEHAVIOR_SCHEMA,
    UNIFORM_MASS,
)
from th06_rl.policies.uniform_safe_exploration import (  # noqa: E402
    STATE_SCHEMA as BASELINE_SCHEMA,
)
from th06_rl.sequential_learning import TRANSITION_SCHEMA  # noqa: E402
from th06_rl.wine_workers import prepare_wine_workers  # noqa: E402


SCHEMA = "autonomous-wine-learning-generation-v5-curriculum-v1"
SCHEDULE = REPOSITORY / "config/autonomous_generation5_curriculum_seeds.json"
CONTRACT = REPOSITORY / "config/autonomous_generation5_curriculum_contract.json"
EXPLORATION_PLUGIN = (
    REPOSITORY / "src/th06_rl/policies/propensity_aware_option_exploration.py"
)
BASELINE_PLUGIN = REPOSITORY / "src/th06_rl/policies/uniform_safe_exploration.py"
CANDIDATE_PLUGIN = (
    REPOSITORY / "src/th06_rl/policies/autonomous_supported_implicit_q.py"
)
MAXIMUM_P95_MS = 4.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contract() -> dict[str, object]:
    contract = _object(CONTRACT)
    files = contract.get("files")
    external = contract.get("external_inputs")
    if (
        contract.get("schema") != "autonomous-generation-5-curriculum-contract-v1"
        or contract.get("schedule_sha256") != _sha256(SCHEDULE)
        or not isinstance(files, list)
        or not isinstance(external, list)
    ):
        raise ValueError("Generation-5 curriculum contract header differs")
    for row in [*files, *external]:
        if not isinstance(row, dict):
            raise TypeError("Generation-5 contract row is not an object")
        raw = Path(str(row["path"]))
        path = raw if raw.is_absolute() else REPOSITORY / raw
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            raise ValueError(f"Generation-5 contract input differs: {path}")
    return contract


def _behavior_state(
    path: Path, *, policy_seed: int, information_policy: Path | None
) -> None:
    information = None
    information_sha = None
    if information_policy is not None:
        information = _object(information_policy)
        authorization = information.get("authorization")
        if (
            information.get("schema") != CANDIDATE_SCHEMA
            or information.get("mode") != "shadow"
            or not isinstance(authorization, dict)
            or authorization.get("fit_eligible") is not True
        ):
            raise ValueError("exploration information population is not fit-eligible")
        information_sha = _sha256(information_policy)
    value = {
        "schema": BEHAVIOR_SCHEMA,
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
        raise ValueError("existing Generation-5 behavior state differs")
    if not path.exists():
        _atomic_json(path, value)


def _baseline_state(path: Path) -> None:
    value = {
        "schema": BASELINE_SCHEMA,
        "policy_seed": 260_813,
        "exploration_probability": 0.0,
    }
    if path.is_file() and _object(path) != value:
        raise ValueError("existing Generation-5 baseline state differs")
    if not path.exists():
        _atomic_json(path, value)


def _parallelism_differential(
    *,
    root: Path,
    schedule: dict[str, object],
    workers: list[dict[str, object]],
) -> dict[str, object]:
    audit_path = root / "parallelism-differential" / "audit.json"
    if audit_path.is_file():
        audit = _object(audit_path)
        if audit.get("passed") is not True or audit.get("evidence_eligible") is not False:
            raise RuntimeError("cached parallelism differential did not pass")
        return audit
    spec = schedule["parallelism_differential"]
    policy_state = root / "parallelism-differential" / "behavior-state.json"
    _behavior_state(
        policy_state,
        policy_seed=int(spec["policy_seed"]),
        information_policy=None,
    )

    def execute(label: str, worker_index: int):
        base = root / "parallelism-differential" / label
        return complete_run(
            artifact_dir=base / "artifact",
            worker=workers[worker_index],
            stage=int(spec["stage"]),
            policy_plugin=EXPLORATION_PLUGIN,
            policy_state=policy_state,
            scorer=None,
            rng_seed=int(spec["game_rng_seed"]),
            corpus_root=base / "corpus",
        )

    reference_report, reference_run = execute("serial-worker-0", 0)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(execute, f"concurrent-worker-{index}", index)
            for index in (0, 1)
        ]
        concurrent = [future.result() for future in futures]
    assert reference_run is not None and all(run is not None for _report, run in concurrent)
    reports = [reference_report, *(report for report, _run in concurrent)]
    runs = [reference_run, *(run for _report, run in concurrent)]
    hits = [int(report["controller_completion"]["physical_hits"]) for report in reports]
    digests = [normalized_option_sha256(run) for run in runs]
    gates = {
        "three_complete_stage_runs": len(runs) == 3,
        "identical_physical_hits": len(set(hits)) == 1,
        "identical_factual_option_semantics": len(set(digests)) == 1,
        "isolated_worker_paths": len({
            (report["display"], report["wine_prefix"], Path(report["retail_executable"]).parent)
            for report in reports[1:]
        }) == 2,
    }
    audit = {
        "schema": "autonomous-generation-5-wine-parallelism-differential-v1",
        "evidence_eligible": False,
        "game_rng_seed": int(spec["game_rng_seed"]),
        "policy_seed": int(spec["policy_seed"]),
        "physical_hits": hits,
        "normalized_option_sha256": digests,
        "runs": [str(path) for path in runs],
        "gates": gates,
        "passed": all(gates.values()),
    }
    _atomic_json(audit_path, audit)
    if not audit["passed"]:
        raise RuntimeError("isolated Wine parallelism differential failed")
    return audit


def _collect_wave(
    *,
    root: Path,
    stage: int,
    rows: list[dict[str, object]],
    workers: list[dict[str, object]],
    information_policy: Path | None,
    scorer: Path,
) -> list[dict[str, object]]:
    if len({int(row["worker"]) for row in rows}) != len(rows):
        raise ValueError("one collection wave assigned a Wine worker twice")

    def execute(row: dict[str, object]):
        episode = int(row["episode"])
        worker_index = int(row["worker"])
        behavior = root / f"stage-{stage}" / "behavior" / f"episode-{episode:03d}.json"
        _behavior_state(
            behavior,
            policy_seed=int(row["policy_seed"]),
            information_policy=information_policy,
        )
        artifact = root / f"stage-{stage}" / "collection" / f"episode-{episode:03d}"
        report, run_dir = complete_run(
            artifact_dir=artifact,
            worker=workers[worker_index],
            stage=stage,
            policy_plugin=EXPLORATION_PLUGIN,
            policy_state=behavior,
            scorer=scorer if information_policy is not None else None,
            rng_seed=int(row["game_rng_seed"]),
            corpus_root=(
                root / f"stage-{stage}" / "worker-corpora" / f"wine-{worker_index}"
            ),
        )
        assert run_dir is not None
        return {
            "episode": episode,
            "worker": worker_index,
            "artifact_dir": str(artifact),
            "corpus_run_dir": str(run_dir),
            "game_rng_seed": int(row["game_rng_seed"]),
            "policy_seed": int(row["policy_seed"]),
            "information_policy_sha256": (
                _sha256(information_policy) if information_policy is not None else None
            ),
            "physical_hits": int(report["controller_completion"]["physical_hits"]),
        }

    with ThreadPoolExecutor(max_workers=len(rows)) as executor:
        return sorted(
            (future.result() for future in [executor.submit(execute, row) for row in rows]),
            key=lambda row: int(row["episode"]),
        )


def _smoke_fit(
    *, root: Path, stage: int, boundary: int, run_dirs: list[Path], threads: int
) -> dict[str, object]:
    output = root / f"stage-{stage}" / "fits" / f"boundary-{boundary:02d}-smoke.json"
    if not output.is_file():
        command = [
            sys.executable,
            str(REPOSITORY / "scripts/smoke_supported_implicit_q_wine.py"),
            *(str(path) for path in run_dirs),
            "--output", str(output),
            "--iterations", "2",
            "--n-step-options", "8",
            "--q-trees", "8",
            "--value-trees", "8",
            "--threads", str(threads),
            "--fold-workers", "5",
        ]
        completed = subprocess.run(command, cwd=REPOSITORY, check=False)
        if completed.returncode:
            raise RuntimeError(f"Stage-{stage} learner smoke failed")
    report = _object(output)
    if report.get("evidence_eligible") is not False:
        raise ValueError("learner smoke unexpectedly claims evidence eligibility")
    return report


def _production_fit(
    *,
    root: Path,
    stage: int,
    boundary: int,
    all_runs: list[Path],
    new_runs: list[Path],
    threads: int,
    wine_scorer: Path,
    host_scorer: Path,
) -> tuple[Path, dict[str, object]]:
    output = root / f"stage-{stage}" / "fits" / f"boundary-{boundary:02d}-production"
    state_path = output / "policy-shadow.json"
    report_path = output / "report.json"
    if not (state_path.is_file() and report_path.is_file()):
        if output.exists():
            _archive_incomplete(output)
        command = [
            sys.executable,
            str(REPOSITORY / "scripts/fit_supported_implicit_q.py"),
            *(str(path) for path in all_runs),
            *(item for path in new_runs for item in ("--new-run", str(path))),
            "--output-dir", str(output),
            "--native-scorer", str(wine_scorer),
            "--compatible-native-scorer", str(host_scorer),
            "--threads", str(threads),
            "--fold-workers", "5",
        ]
        completed = subprocess.run(command, cwd=REPOSITORY, check=False)
        if completed.returncode:
            raise RuntimeError(f"Stage-{stage} production fit failed")
    state = _object(state_path)
    report = _object(report_path)
    if state.get("schema") != CANDIDATE_SCHEMA or report.get("authorization") != state.get("authorization"):
        raise ValueError("cached Generation-5 production fit is inconsistent")
    return state_path, report


def _shadow_authorize(
    *, root: Path, stage: int, boundary: int, state_path: Path,
    audit_runs: list[Path], host_scorer: Path
) -> tuple[Path, dict[str, object]]:
    fit_root = root / f"stage-{stage}" / "fits" / f"boundary-{boundary:02d}-production"
    shadow_path = fit_root / "shadow-audit.json"
    if shadow_path.is_file():
        shadow_report = _object(shadow_path)
    else:
        shadow_report = shadow(
            state_path, audit_runs, native_scorer=host_scorer,
            maximum_p95_ms=MAXIMUM_P95_MS,
        )
        _atomic_json(shadow_path, shadow_report)
    if shadow_report.get("shadow_eligible") is not True:
        raise RuntimeError("Generation-5 native shadow did not pass")
    active_path = fit_root / "policy-canary.json"
    if not active_path.is_file():
        _atomic_json(active_path, authorize(state_path, shadow_path))
    return active_path, shadow_report


def _canary(
    *,
    root: Path,
    stage: int,
    boundary: int,
    seeds: list[int],
    worker: dict[str, object],
    candidate_state: Path,
    wine_scorer: Path,
) -> tuple[Path | None, dict[str, object]]:
    canary_root = root / f"stage-{stage}" / "canary" / f"boundary-{boundary:02d}"
    audit_path = canary_root / "audit.json"
    evaluation_path = canary_root / "policy-evaluation.json"
    if audit_path.is_file():
        audit = _object(audit_path)
        if audit.get("candidate_state_sha256") != _sha256(candidate_state):
            raise ValueError("cached Generation-5 canary is stale")
        if audit.get("canary_eligible") is True and not evaluation_path.is_file():
            raise FileNotFoundError(evaluation_path)
        return (evaluation_path if audit.get("canary_eligible") is True else None), audit
    baseline_state = canary_root / "baseline-state.json"
    _baseline_state(baseline_state)
    runs = []
    for pair, seed in enumerate(seeds):
        for arm in ("baseline", "candidate"):
            artifact = canary_root / f"pair-{pair:02d}-{arm}"
            report, _run = complete_run(
                artifact_dir=artifact,
                worker=worker,
                stage=stage,
                policy_plugin=BASELINE_PLUGIN if arm == "baseline" else CANDIDATE_PLUGIN,
                policy_state=baseline_state if arm == "baseline" else candidate_state,
                scorer=None if arm == "baseline" else wine_scorer,
                rng_seed=seed,
                corpus_root=canary_root / "non-training-corpus",
            )
            metrics = _candidate_metrics(report) if arm == "candidate" else {}
            runs.append({
                "pair": pair,
                "arm": arm,
                "game_rng_seed": seed,
                "artifact_dir": str(artifact),
                "physical_hits": int(report["controller_completion"]["physical_hits"]),
                "active_overrides": int(metrics.get("active_overrides", 0)),
                "runtime_clean": _candidate_runtime_clean(metrics) if arm == "candidate" else True,
                "decision_latency_p95_ms": metrics.get("decision_latency_p95_ms"),
            })
    baseline = {int(row["pair"]): int(row["physical_hits"]) for row in runs if row["arm"] == "baseline"}
    candidate = {int(row["pair"]): int(row["physical_hits"]) for row in runs if row["arm"] == "candidate"}
    exercised = sum(int(row["active_overrides"]) > 0 for row in runs if row["arm"] == "candidate")
    no_worse = sum(candidate[pair] <= baseline[pair] for pair in baseline)
    gates = {
        "six_clean_complete_stages": len(runs) == 6,
        "candidate_exercised_in_two_pairs": exercised >= 2,
        "strictly_fewer_aggregate_hits": sum(candidate.values()) < sum(baseline.values()),
        "candidate_no_worse_in_two_pairs": no_worse >= 2,
        "candidate_runtime_clean": all(row["runtime_clean"] for row in runs if row["arm"] == "candidate"),
    }
    audit = {
        "schema": "autonomous-generation-5-paired-canary-v1",
        "stage": stage,
        "boundary": boundary,
        "candidate_state_sha256": _sha256(candidate_state),
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
    if not audit["canary_eligible"]:
        return None, audit
    evaluation = _object(candidate_state)
    evaluation["authorization"]["full_evaluation"] = {
        "schema": "autonomous-generation-5-full-evaluation-authorization-v1",
        "canary_audit_sha256": _sha256(audit_path),
        "candidate_canary_state_sha256": _sha256(candidate_state),
        "stage": stage,
        "fixed_rng_effect": audit["effect"],
    }
    _atomic_json(evaluation_path, evaluation)
    return evaluation_path, audit


def _final_stage6(
    *, root: Path, worker: dict[str, object], candidate_state: Path,
    wine_scorer: Path, trial_order: list[str]
) -> dict[str, object]:
    final_root = root / "final-stage6"
    report_path = final_root / "report.json"
    if report_path.is_file():
        return _object(report_path)
    baseline_state = final_root / "baseline-state.json"
    _baseline_state(baseline_state)
    runs = []
    for trial, arm in enumerate(trial_order):
        artifact = final_root / f"trial-{trial:02d}-{arm}"
        report, run_dir = complete_run(
            artifact_dir=artifact,
            worker=worker,
            stage=6,
            policy_plugin=BASELINE_PLUGIN if arm == "baseline" else CANDIDATE_PLUGIN,
            policy_state=baseline_state if arm == "baseline" else candidate_state,
            scorer=None if arm == "baseline" else wine_scorer,
            rng_seed=None,
            corpus_root=None,
        )
        assert run_dir is None
        metrics = _candidate_metrics(report) if arm == "candidate" else {}
        runs.append({
            "trial": trial,
            "arm": arm,
            "artifact_dir": str(artifact),
            "physical_hits": int(report["controller_completion"]["physical_hits"]),
            "active_overrides": int(metrics.get("active_overrides", 0)),
            "runtime_clean": _candidate_runtime_clean(metrics) if arm == "candidate" else True,
            "decision_latency_p95_ms": metrics.get("decision_latency_p95_ms"),
        })
    baseline_total = sum(int(row["physical_hits"]) for row in runs if row["arm"] == "baseline")
    candidate_total = sum(int(row["physical_hits"]) for row in runs if row["arm"] == "candidate")
    exercised = sum(int(row["active_overrides"]) > 0 for row in runs if row["arm"] == "candidate")
    gates = {
        "all_24_stages_complete": len(runs) == 24,
        "strictly_fewer_candidate_hits": candidate_total < baseline_total,
        "candidate_exercised_in_six_stages": exercised >= 6,
        "candidate_runtime_clean": all(row["runtime_clean"] for row in runs if row["arm"] == "candidate"),
    }
    result = {
        "schema": "autonomous-generation-5-natural-stage6-ab-v1",
        "evaluation_mode": "normal-speed-natural-complete-stage-hit-continuation",
        "fixed_rng": False,
        "trial_order": trial_order,
        "runs": runs,
        "baseline_total_hits": baseline_total,
        "candidate_total_hits": candidate_total,
        "effect": baseline_total - candidate_total,
        "candidate_exercised_stages": exercised,
        "gates": gates,
        "verdict": "effective" if all(gates.values()) else "ineffective",
    }
    _atomic_json(report_path, result)
    return result


def run(args: argparse.Namespace) -> int:
    schedule = load_curriculum_schedule(SCHEDULE)
    contract = _validate_contract()
    root = args.output_root.resolve()
    workers = prepare_wine_workers(
        root=root / "workers",
        source_game_dir=args.source_game_dir,
        specifications=schedule["resource_contract"]["workers"],
    )
    if any(
        row["source_inventory_sha256"]
        != contract.get("source_game_inventory_sha256")
        for row in workers
    ):
        raise ValueError("original game directory inventory differs from contract")
    config = {
        "schedule_sha256": _sha256(SCHEDULE),
        "contract_sha256": _sha256(CONTRACT),
        "source_contract": contract,
        "threads": args.threads,
        "wine_scorer_sha256": _sha256(args.wine_native_scorer),
        "host_scorer_sha256": _sha256(args.host_native_scorer),
        "workers": workers,
    }
    state_path = root / "generation.json"
    if state_path.is_file():
        state = _object(state_path)
        if state.get("schema") != SCHEMA or state.get("config") != config:
            raise RuntimeError("refusing Generation-5 resume with contract drift")
        if state.get("status") == "complete":
            print(json.dumps(state["decision"], sort_keys=True))
            return 0
    else:
        state = {
            "schema": SCHEMA,
            "status": "preflight",
            "config": config,
            "parallelism_differential": None,
            "stages": [],
            "latest_information_policy": None,
            "decision": None,
        }
        _atomic_json(state_path, state)
    try:
        state.pop("infra_failure", None)
        differential = _parallelism_differential(
            root=root, schedule=schedule, workers=workers
        )
        state["parallelism_differential"] = differential
        state["status"] = "curriculum"
        _atomic_json(state_path, state)
        all_runs: list[Path] = []
        final_candidate = None
        for stage_spec in schedule["stages"]:
            stage = int(stage_spec["stage"])
            while len(state["stages"]) < stage - 3:
                next_stage = 4 + len(state["stages"])
                state["stages"].append({
                    "stage": next_stage, "status": "collecting", "episodes": [],
                    "fits": [], "canary": None,
                })
            stage_state = state["stages"][stage - 4]
            stage_runs = [Path(row["corpus_run_dir"]) for row in stage_state["episodes"]]
            for run_dir in stage_runs:
                _validate_run(run_dir, transition_schema=TRANSITION_SCHEMA)
            stage_candidate = None
            for fit_spec in stage_spec["fits"]:
                boundary = int(fit_spec["boundary"])
                while len(stage_state["episodes"]) < boundary:
                    start = len(stage_state["episodes"])
                    stop = min(boundary, start + 4)
                    wave = stage_spec["collection"][start:stop]
                    information_raw = state.get("latest_information_policy")
                    information = Path(information_raw) if isinstance(information_raw, str) else None
                    completed = _collect_wave(
                        root=root,
                        stage=stage,
                        rows=wave,
                        workers=workers,
                        information_policy=information,
                        scorer=args.wine_native_scorer,
                    )
                    stage_state["episodes"].extend(completed)
                    stage_state["episodes"].sort(key=lambda row: int(row["episode"]))
                    stage_state["status"] = "collecting"
                    _atomic_json(state_path, state)
                    stage_runs.extend(Path(row["corpus_run_dir"]) for row in completed)
                fit_row: dict[str, Any] = {
                    "boundary": boundary, "mode": fit_spec["mode"], "status": "running"
                }
                existing = next((row for row in stage_state["fits"] if row["boundary"] == boundary), None)
                if existing is None:
                    stage_state["fits"].append(fit_row)
                else:
                    fit_row = existing
                _atomic_json(state_path, state)
                if fit_spec["mode"] == "non-authorizing-smoke":
                    smoke = _smoke_fit(
                        root=root, stage=stage, boundary=boundary,
                        run_dirs=[*all_runs, *stage_runs], threads=args.threads,
                    )
                    fit_row.update({
                        "status": "smoke-complete",
                        "relative_q_loss": smoke["report"]["overall"]["relative_q_loss"],
                        "episodes_beating_zero": smoke["report"]["overall"]["episodes_beating_zero"],
                        "authorization_eligible": False,
                    })
                    _atomic_json(state_path, state)
                    continue
                state_path_fit, fit_report = _production_fit(
                    root=root, stage=stage, boundary=boundary,
                    all_runs=[*all_runs, *stage_runs], new_runs=stage_runs,
                    threads=args.threads, wine_scorer=args.wine_native_scorer,
                    host_scorer=args.host_native_scorer,
                )
                eligible = fit_report["authorization"]["fit_eligible"] is True
                fit_row.update({
                    "status": "fit-eligible" if eligible else "fit-ineligible",
                    "fit_dir": str(state_path_fit.parent), "fit_eligible": eligible,
                })
                _atomic_json(state_path, state)
                if not eligible:
                    continue
                active_path, shadow_report = _shadow_authorize(
                    root=root, stage=stage, boundary=boundary,
                    state_path=state_path_fit, audit_runs=stage_runs[-3:],
                    host_scorer=args.host_native_scorer,
                )
                state["latest_information_policy"] = str(state_path_fit)
                fit_row["shadow_audit"] = str(state_path_fit.parent / "shadow-audit.json")
                fit_row["shadow_p95_ms"] = shadow_report["latency"]["p95_ms"]
                _atomic_json(state_path, state)
                seeds = [
                    int(row["game_rng_seed"]) for row in stage_spec["canary"]
                    if int(row["boundary"]) == boundary
                ]
                if not seeds:
                    raise RuntimeError("fit-eligible boundary lacks frozen canary seeds")
                stage_candidate, canary = _canary(
                    root=root, stage=stage, boundary=boundary, seeds=seeds,
                    worker=workers[0], candidate_state=active_path,
                    wine_scorer=args.wine_native_scorer,
                )
                stage_state["canary"] = canary
                fit_row["status"] = "canary-passed" if stage_candidate else "canary-rejected"
                _atomic_json(state_path, state)
                if stage_candidate is not None:
                    break
            all_runs.extend(stage_runs)
            if stage_candidate is None:
                decision = {
                    "verdict": "ineffective",
                    "reason": f"Stage-{stage} exhausted its frozen evidence budget",
                    "failed_stage": stage,
                    "episodes": len(stage_state["episodes"]),
                }
                state.update({"status": "complete", "decision": decision})
                _atomic_json(state_path, state)
                print(json.dumps(decision, sort_keys=True))
                return 0
            stage_state["status"] = "passed"
            stage_state["evaluation_policy"] = str(stage_candidate)
            _atomic_json(state_path, state)
            final_candidate = stage_candidate
        assert final_candidate is not None
        final = _final_stage6(
            root=root, worker=workers[0], candidate_state=final_candidate,
            wine_scorer=args.wine_native_scorer,
            trial_order=list(schedule["final_evaluation"]["trial_order"]),
        )
        decision = {
            "verdict": final["verdict"],
            "reason": "predeclared natural complete-Stage-6 physical HIT gates",
            "baseline_total_hits": final["baseline_total_hits"],
            "candidate_total_hits": final["candidate_total_hits"],
            "effect": final["effect"],
            "candidate_exercised_stages": final["candidate_exercised_stages"],
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
        "--output-root", type=Path,
        default=REPOSITORY / "artifacts/autonomous-wine-generation-5-curriculum",
    )
    parser.add_argument(
        "--source-game-dir", type=Path,
        default=REPOSITORY / "reference/th06-game-original/th06",
    )
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--wine-native-scorer", type=Path,
        default=REPOSITORY / "build/native-win32-fully-static/libth06_rl_ranker.dll",
    )
    parser.add_argument(
        "--host-native-scorer", type=Path,
        default=REPOSITORY / "build/native/libth06_rl_ranker.so",
    )
    args = parser.parse_args(argv)
    if args.threads != 32:
        parser.error("canonical Generation-5 curriculum uses exactly the frozen 32-thread cap")
    for name in ("source_game_dir", "wine_native_scorer", "host_native_scorer"):
        path = getattr(args, name).resolve()
        if not path.exists():
            parser.error(f"required input is absent: {path}")
        setattr(args, name, path)
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
