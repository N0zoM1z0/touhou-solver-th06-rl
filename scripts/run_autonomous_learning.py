#!/usr/bin/env python3
"""Run a resumable Wine collect-fit-shadow-canary-full-Stage RL generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from scripts.authorize_autonomous_canary import authorize
from scripts.shadow_autonomous_q import shadow
from th06_rl.autonomous_learning import _object, _validate_run


SCHEMA = "autonomous-wine-learning-generation-v1"
EXPLORATION_SCHEMA = "th06-rl-uniform-safe-exploration-v1"
EXPLORATION_PLUGIN = REPOSITORY / "src/th06_rl/policies/uniform_safe_exploration.py"
CANDIDATE_PLUGIN = REPOSITORY / "src/th06_rl/policies/autonomous_linear_q.py"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_schedule(seed: int, count: int) -> list[dict[str, int]]:
    generator = random.Random(seed)
    game_seeds = generator.sample(range(0x10000), count)
    return [
        {
            "game_rng_seed": game_seed,
            "policy_seed": generator.randrange(2**64),
        }
        for game_seed in game_seeds
    ]


def _exploration_state(
    path: Path, *, policy_seed: int, exploration_probability: float
) -> None:
    _atomic_json(path, {
        "schema": EXPLORATION_SCHEMA,
        "policy_seed": policy_seed,
        "exploration_probability": exploration_probability,
    })


def _corpus_runs(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        path.name for path in root.iterdir()
        if path.is_dir() and (path / "run.json").is_file()
    }


def _validate_retail_report(
    report: dict[str, object],
    *,
    mode: str,
    diagnostic_rng_seed: int | None,
    full_stage: int | None = None,
) -> None:
    if report.get("error") is not None:
        raise RuntimeError(f"Wine runner error: {report['error']}")
    if report.get("evaluation_mode") != mode:
        raise RuntimeError("Wine runner evaluation mode mismatch")
    if report.get("diagnostic_rng_seed") != diagnostic_rng_seed:
        raise RuntimeError("Wine runner RNG provenance mismatch")
    if report.get("immutable_policy_state_equal") is not True:
        raise RuntimeError("Wine runner mutated an immutable policy state")
    if report.get("leftover_prefix_processes") != []:
        raise RuntimeError("Wine runner left prefix processes alive")
    if full_stage is None:
        if int(report.get("controller_returncode", -1)) not in (0, 10, 12):
            raise RuntimeError("Wine training/canary controller failed")
    else:
        completion = report.get("controller_completion")
        trace = report.get("trace")
        if (
            int(report.get("controller_returncode", -1)) != 0
            or not isinstance(completion, dict)
            or completion.get("practice_stage_completed") is not True
            or int(completion.get("practice_stage", -1)) != full_stage
            or not isinstance(trace, dict)
            or int(completion.get("physical_hits", -1))
            != int(trace.get("physical_hits_in_run", -2))
        ):
            raise RuntimeError("Wine full-Stage completion/HIT accounting failed")


def _run_retail(
    *,
    artifact_dir: Path,
    policy_plugin: Path,
    policy_state: Path,
    difficulty: str,
    stage: int,
    diagnostic_rng_seed: int | None,
    corpus_root: Path | None,
) -> tuple[dict[str, object], Path | None]:
    before = _corpus_runs(corpus_root) if corpus_root is not None else set()
    command = [
        sys.executable,
        str(REPOSITORY / "scripts/run_wine_retail.py"),
        "--practice-stage",
        str(stage),
        "--difficulty",
        difficulty,
        "--artifact-dir",
        str(artifact_dir),
        "--policy-plugin",
        str(policy_plugin),
        "--policy-state",
        str(policy_state),
        "--immutable-policy",
        "--exploration-rate",
        "0",
    ]
    if corpus_root is not None:
        command.extend(("--first-failure-corpus-root", str(corpus_root)))
    if diagnostic_rng_seed is not None:
        command.extend(("--diagnostic-rng-seed", hex(diagnostic_rng_seed)))
    completed = subprocess.run(command, cwd=REPOSITORY, check=False)
    report = _object(artifact_dir / "report.json")
    mode = (
        "fixed-rng-first-failure-training"
        if diagnostic_rng_seed is not None
        else "first-failure-corpus"
        if corpus_root is not None
        else "hit-continuation-benchmark"
    )
    _validate_retail_report(
        report,
        mode=mode,
        diagnostic_rng_seed=diagnostic_rng_seed,
        full_stage=stage if corpus_root is None else None,
    )
    if completed.returncode != int(report["controller_returncode"]):
        raise RuntimeError("outer and recorded controller return codes differ")
    run_dir = None
    if corpus_root is not None:
        after = _corpus_runs(corpus_root)
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError(f"Wine trial created {len(created)} corpus runs")
        run_dir = corpus_root / created[0]
        _validate_run(run_dir)
    return report, run_dir


def _config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "difficulty": args.difficulty,
        "stage": args.stage,
        "generation_seed": args.generation_seed,
        "collection_episodes": args.collection_episodes,
        "round_size": args.round_size,
        "minimum_rounds_before_canary": args.minimum_rounds_before_canary,
        "canary_episodes": args.canary_episodes,
        "full_stage_pairs": args.full_stage_pairs,
        "exploration_probability": args.exploration_probability,
        "validation_episodes": args.validation_episodes,
        "return_horizon": args.return_horizon,
        "gamma": args.gamma,
        "ridge_alpha": args.ridge_alpha,
        "propensity_clip": args.propensity_clip,
        "minimum_train_groups": args.minimum_train_groups,
        "minimum_validation_groups": args.minimum_validation_groups,
        "minimum_train_rows": args.minimum_train_rows,
        "minimum_non_baseline_rows": args.minimum_non_baseline_rows,
        "minimum_action_samples": args.minimum_action_samples,
        "minimum_action_ess": args.minimum_action_ess,
        "required_rmse_ratio": args.required_rmse_ratio,
        "margin_rmse_fraction": args.margin_rmse_fraction,
        "minimum_shadow_rows": args.minimum_shadow_rows,
        "minimum_shadow_proposals": args.minimum_shadow_proposals,
        "maximum_shadow_p95_ms": args.maximum_shadow_p95_ms,
    }


def _fit_command(
    args: argparse.Namespace, run_dirs: list[Path], output_dir: Path
) -> list[str]:
    return [
        sys.executable,
        str(REPOSITORY / "scripts/fit_autonomous_q.py"),
        *(str(path) for path in run_dirs),
        "--output-dir",
        str(output_dir),
        "--validation-episodes",
        str(args.validation_episodes),
        "--exploration-probability",
        str(args.exploration_probability),
        "--return-horizon",
        str(args.return_horizon),
        "--gamma",
        str(args.gamma),
        "--ridge-alpha",
        str(args.ridge_alpha),
        "--propensity-clip",
        str(args.propensity_clip),
        "--minimum-train-groups",
        str(args.minimum_train_groups),
        "--minimum-validation-groups",
        str(args.minimum_validation_groups),
        "--minimum-train-rows",
        str(args.minimum_train_rows),
        "--minimum-non-baseline-rows",
        str(args.minimum_non_baseline_rows),
        "--minimum-action-samples",
        str(args.minimum_action_samples),
        "--minimum-action-ess",
        str(args.minimum_action_ess),
        "--required-rmse-ratio",
        str(args.required_rmse_ratio),
        "--margin-rmse-fraction",
        str(args.margin_rmse_fraction),
    ]


def _fit_round(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    output_root: Path,
) -> Path | None:
    round_index = len(state["rounds"]) + 1
    run_dirs = [Path(row["corpus_run_dir"]) for row in state["episodes"]]
    round_dir = output_root / "rounds" / f"round-{round_index:02d}"
    if not round_dir.exists():
        fit = subprocess.run(
            _fit_command(args, run_dirs, round_dir),
            cwd=REPOSITORY,
            check=False,
        )
        if fit.returncode:
            raise RuntimeError(f"offline grouped fit failed with {fit.returncode}")
    elif not (
        (round_dir / "policy-shadow.json").is_file()
        and (round_dir / "report.json").is_file()
    ):
        raise RuntimeError(f"incomplete offline round output: {round_dir}")
    shadow_state = round_dir / "policy-shadow.json"
    fit_report = _object(round_dir / "report.json")
    row: dict[str, Any] = {
        "round": round_index,
        "episodes": len(run_dirs),
        "fit_dir": str(round_dir),
        "fit_eligible": bool(fit_report["authorization"]["fit_eligible"]),
        "status": "fit-ineligible",
    }
    state["rounds"].append(row)
    _atomic_json(state_path, state)
    return _continue_fit_round(
        args, state, state_path, row, run_dirs, shadow_state
    )


def _continue_fit_round(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    row: dict[str, Any],
    run_dirs: list[Path],
    shadow_state: Path,
) -> Path | None:
    if not row["fit_eligible"]:
        return None

    validation_dirs = run_dirs[-args.validation_episodes:]
    round_dir = Path(row["fit_dir"])
    shadow_path = round_dir / "shadow-audit.json"
    if shadow_path.is_file():
        shadow_report = _object(shadow_path)
    else:
        shadow_report = shadow(
            shadow_state,
            validation_dirs,
            minimum_rows=args.minimum_shadow_rows,
            minimum_proposals=args.minimum_shadow_proposals,
            maximum_p95_ms=args.maximum_shadow_p95_ms,
        )
        _atomic_json(shadow_path, shadow_report)
    row["shadow_eligible"] = bool(shadow_report["shadow_eligible"])
    row["shadow_audit"] = str(shadow_path)
    if (
        int(row["round"]) < args.minimum_rounds_before_canary
        or not row["shadow_eligible"]
    ):
        row["status"] = (
            "minimum-rounds-not-reached"
            if int(row["round"]) < args.minimum_rounds_before_canary
            else "shadow-ineligible"
        )
        _atomic_json(state_path, state)
        return None
    active = authorize(shadow_state, shadow_path)
    active_path = round_dir / "policy-canary.json"
    if not active_path.is_file():
        _atomic_json(active_path, active)
    row.update({"status": "canary-ready", "canary_state": str(active_path)})
    _atomic_json(state_path, state)
    return active_path


def _canary(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    output_root: Path,
    candidate_state: Path,
    seeds: list[dict[str, int]],
) -> Path | None:
    canary_root = output_root / "canary"
    corpus_root = output_root / "canary-corpus"
    prior = state.get("canary")
    reports = (
        list(prior.get("runs", ()))
        if isinstance(prior, dict) and prior.get("status") == "running"
        else []
    )
    total_overrides = sum(int(row["active_overrides"]) for row in reports)
    clean = all(bool(row["clean"]) for row in reports)
    for index in range(len(reports), args.canary_episodes):
        artifact = canary_root / f"episode-{index:03d}"
        report, run_dir = _run_retail(
            artifact_dir=artifact,
            policy_plugin=CANDIDATE_PLUGIN,
            policy_state=candidate_state,
            difficulty=args.difficulty,
            stage=args.stage,
            diagnostic_rng_seed=seeds[index]["game_rng_seed"],
            corpus_root=corpus_root,
        )
        assert run_dir is not None
        metrics = (report.get("trace") or {}).get("last_policy_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        overrides = int(metrics.get("active_overrides", 0))
        budget = int(metrics.get("active_override_budget", 0))
        total_overrides += overrides
        manifest = _object(run_dir / "manifest.json")
        solve = (manifest.get("summary") or {}).get("solve_timing")
        solve_p95 = (
            float(solve.get("p95_ms")) if isinstance(solve, dict)
            and solve.get("p95_ms") is not None else None
        )
        trial_clean = bool(
            0 <= overrides <= budget
            and budget > 0
            and solve_p95 is not None
            and solve_p95 <= 1000.0 / 60.0
        )
        clean = clean and trial_clean
        reports.append({
            "episode": index,
            "artifact_dir": str(artifact),
            "corpus_run_dir": str(run_dir),
            "game_rng_seed": seeds[index]["game_rng_seed"],
            "active_overrides": overrides,
            "override_budget": budget,
            "solve_p95_ms": solve_p95,
            "clean": trial_clean,
        })
        state["canary"] = {"status": "running", "runs": reports}
        _atomic_json(state_path, state)
    gates = {
        "all_runs_clean": clean,
        "candidate_exercised": total_overrides > 0,
        "expected_run_count": len(reports) == args.canary_episodes,
    }
    audit = {
        "schema": "autonomous-wine-active-canary-audit-v1",
        "candidate_state": str(candidate_state),
        "candidate_state_sha256": _sha256(candidate_state),
        "runs": reports,
        "total_active_overrides": total_overrides,
        "gates": gates,
        "canary_eligible": all(gates.values()),
    }
    audit_path = canary_root / "audit.json"
    _atomic_json(audit_path, audit)
    state["canary"] = {
        "status": "passed" if audit["canary_eligible"] else "rejected",
        "audit": str(audit_path),
        "runs": reports,
    }
    _atomic_json(state_path, state)
    if not audit["canary_eligible"]:
        return None
    evaluation = _object(candidate_state)
    evaluation["selection"]["active_override_budget"] = None
    evaluation["authorization"]["full_evaluation"] = {
        "schema": "autonomous-full-evaluation-authorization-v1",
        "canary_audit_sha256": _sha256(audit_path),
        "candidate_canary_state_sha256": _sha256(candidate_state),
        "canary_runs": len(reports),
        "active_overrides": total_overrides,
    }
    evaluation_path = canary_root / "policy-full-evaluation.json"
    _atomic_json(evaluation_path, evaluation)
    return evaluation_path


def _verdict(rows: list[dict[str, object]]) -> dict[str, object]:
    baseline = [int(row["physical_hits"]) for row in rows if row["arm"] == "baseline"]
    candidate = [int(row["physical_hits"]) for row in rows if row["arm"] == "candidate"]
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("full-Stage A/B rows are unbalanced")
    return {
        "baseline_hits": baseline,
        "candidate_hits": candidate,
        "baseline_total_hits": sum(baseline),
        "candidate_total_hits": sum(candidate),
        "effect": sum(baseline) - sum(candidate),
        "verdict": "effective" if sum(candidate) < sum(baseline) else "ineffective",
        "rule": "candidate aggregate physical HITs must be strictly lower",
    }


def _full_stage_ab(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    output_root: Path,
    candidate_state: Path,
) -> dict[str, object]:
    baseline_state = output_root / "full-stage" / "baseline-state.json"
    _exploration_state(
        baseline_state, policy_seed=args.generation_seed, exploration_probability=0.0
    )
    prior = state.get("full_stage")
    rows = (
        list(prior.get("runs", ()))
        if isinstance(prior, dict) and prior.get("status") == "running"
        else []
    )
    arms = ["baseline", "candidate"] * args.full_stage_pairs
    for index in range(len(rows), len(arms)):
        arm = arms[index]
        artifact = output_root / "full-stage" / f"trial-{index:02d}-{arm}"
        report, run_dir = _run_retail(
            artifact_dir=artifact,
            policy_plugin=(EXPLORATION_PLUGIN if arm == "baseline" else CANDIDATE_PLUGIN),
            policy_state=(baseline_state if arm == "baseline" else candidate_state),
            difficulty=args.difficulty,
            stage=args.stage,
            diagnostic_rng_seed=None,
            corpus_root=None,
        )
        assert run_dir is None
        completion = report["controller_completion"]
        assert isinstance(completion, dict)
        metrics = (report.get("trace") or {}).get("last_policy_metrics")
        row = {
            "trial": index,
            "arm": arm,
            "artifact_dir": str(artifact),
            "physical_hits": int(completion["physical_hits"]),
            "active_overrides": (
                int(metrics.get("active_overrides", 0))
                if isinstance(metrics, dict) else 0
            ),
        }
        rows.append(row)
        state["full_stage"] = {"status": "running", "runs": rows}
        _atomic_json(state_path, state)
    result = _verdict(rows)
    report = {
        "schema": "autonomous-wine-full-stage-ab-v1",
        "evaluation_mode": "normal-speed-natural-complete-stage-hit-continuation",
        "fixed_rng": False,
        "alternation": arms,
        "runs": rows,
        **result,
    }
    report_path = output_root / "full-stage" / "report.json"
    _atomic_json(report_path, report)
    state["full_stage"] = {
        "status": "complete",
        "report": str(report_path),
        "runs": rows,
    }
    return report


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    state_path = output_root / "generation.json"
    config = _config(args)
    if state_path.exists():
        state = _object(state_path)
        if state.get("schema") != SCHEMA or state.get("config") != config:
            raise RuntimeError("refusing to resume a generation with different config")
        if state.get("status") == "complete":
            print(json.dumps(state.get("decision"), sort_keys=True))
            return 0
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        state = {
            "schema": SCHEMA,
            "status": "collecting",
            "config": config,
            "episodes": [],
            "rounds": [],
            "canary": None,
            "full_stage": None,
            "decision": None,
        }
        _atomic_json(state_path, state)

    total_seeds = args.collection_episodes + args.canary_episodes
    seeds = _seed_schedule(args.generation_seed, total_seeds)
    corpus_root = output_root / "collection-corpus"
    states_root = output_root / "behavior-states"
    candidate_state: Path | None = None
    try:
        while len(state["episodes"]) < args.collection_episodes:
            index = len(state["episodes"])
            state_file = states_root / f"episode-{index:03d}.json"
            _exploration_state(
                state_file,
                policy_seed=seeds[index]["policy_seed"],
                exploration_probability=args.exploration_probability,
            )
            artifact = output_root / "collection" / f"episode-{index:03d}"
            if (artifact / "report.json").is_file():
                report = _object(artifact / "report.json")
                _validate_retail_report(
                    report,
                    mode="fixed-rng-first-failure-training",
                    diagnostic_rng_seed=seeds[index]["game_rng_seed"],
                )
                run_ids = (report.get("trace") or {}).get("corpus_run_ids")
                if not isinstance(run_ids, list) or len(run_ids) != 1:
                    raise RuntimeError("completed collection artifact lacks one corpus ID")
                run_dir = corpus_root / str(run_ids[0])
                _validate_run(run_dir)
            else:
                report, run_dir = _run_retail(
                    artifact_dir=artifact,
                    policy_plugin=EXPLORATION_PLUGIN,
                    policy_state=state_file,
                    difficulty=args.difficulty,
                    stage=args.stage,
                    diagnostic_rng_seed=seeds[index]["game_rng_seed"],
                    corpus_root=corpus_root,
                )
            assert run_dir is not None
            state["episodes"].append({
                "episode": index,
                "status": "complete",
                "artifact_dir": str(artifact),
                "corpus_run_dir": str(run_dir),
                "game_rng_seed": seeds[index]["game_rng_seed"],
                "policy_seed": seeds[index]["policy_seed"],
                "termination_reason": _object(run_dir / "manifest.json")[
                    "run_outcome"
                ]["termination_reason"],
                "physical_hits": report["trace"]["physical_hits_in_run"],
            })
            _atomic_json(state_path, state)
            completed = len(state["episodes"])
            minimum = args.minimum_train_groups + args.validation_episodes
            if completed >= minimum and (
                completed % args.round_size == 0
                or completed == args.collection_episodes
            ):
                candidate_state = _fit_round(
                    args, state, state_path, output_root
                )
                if candidate_state is not None:
                    break

        if candidate_state is None:
            for row in reversed(state["rounds"]):
                path = row.get("canary_state")
                if row.get("status") == "canary-ready" and isinstance(path, str):
                    candidate_state = Path(path)
                    break
                if row.get("fit_eligible") is True:
                    round_runs = [
                        Path(item["corpus_run_dir"])
                        for item in state["episodes"][: int(row["episodes"])]
                    ]
                    resumed = _continue_fit_round(
                        args,
                        state,
                        state_path,
                        row,
                        round_runs,
                        Path(row["fit_dir"]) / "policy-shadow.json",
                    )
                    if resumed is not None:
                        candidate_state = resumed
                        break
        if candidate_state is None:
            decision = {
                "verdict": "ineffective",
                "reason": "collection budget ended without fit+shadow authorization",
                "episodes": len(state["episodes"]),
                "rounds": len(state["rounds"]),
            }
            state.update({"status": "complete", "decision": decision})
            _atomic_json(state_path, state)
            print(json.dumps(decision, sort_keys=True))
            return 0

        state["status"] = "canary"
        _atomic_json(state_path, state)
        saved_evaluation = output_root / "canary" / "policy-full-evaluation.json"
        evaluation_state = (
            saved_evaluation
            if saved_evaluation.is_file()
            else _canary(
                args,
                state,
                state_path,
                output_root,
                candidate_state,
                seeds[args.collection_episodes:],
            )
        )
        if evaluation_state is None:
            decision = {
                "verdict": "ineffective",
                "reason": "bounded active Wine canary failed",
            }
            state.update({"status": "complete", "decision": decision})
            _atomic_json(state_path, state)
            print(json.dumps(decision, sort_keys=True))
            return 0

        state["status"] = "full-stage-evaluation"
        _atomic_json(state_path, state)
        saved_final = output_root / "full-stage" / "report.json"
        final = (
            _object(saved_final)
            if saved_final.is_file()
            else _full_stage_ab(
                args, state, state_path, output_root, evaluation_state
            )
        )
        decision = {
            "verdict": final["verdict"],
            "reason": "normal-speed complete-Stage physical HIT aggregate",
            "baseline_total_hits": final["baseline_total_hits"],
            "candidate_total_hits": final["candidate_total_hits"],
            "effect": final["effect"],
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--difficulty", choices=("normal", "hard", "lunatic"), default="lunatic")
    parser.add_argument("--stage", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--generation-seed", type=int, default=6006)
    parser.add_argument("--collection-episodes", type=int, default=10)
    parser.add_argument("--round-size", type=int, default=5)
    parser.add_argument("--minimum-rounds-before-canary", type=int, default=2)
    parser.add_argument("--canary-episodes", type=int, default=2)
    parser.add_argument("--full-stage-pairs", type=int, default=2)
    parser.add_argument("--exploration-probability", type=float, default=0.10)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--return-horizon", type=int, default=120)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--propensity-clip", type=float, default=20.0)
    parser.add_argument("--minimum-train-groups", type=int, default=3)
    parser.add_argument("--minimum-validation-groups", type=int, default=2)
    parser.add_argument("--minimum-train-rows", type=int, default=1000)
    parser.add_argument("--minimum-non-baseline-rows", type=int, default=64)
    parser.add_argument("--minimum-action-samples", type=int, default=16)
    parser.add_argument("--minimum-action-ess", type=float, default=8.0)
    parser.add_argument("--required-rmse-ratio", type=float, default=0.995)
    parser.add_argument("--margin-rmse-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-shadow-rows", type=int, default=500)
    parser.add_argument("--minimum-shadow-proposals", type=int, default=10)
    parser.add_argument("--maximum-shadow-p95-ms", type=float, default=4.0)
    args = parser.parse_args(argv)
    integers = (
        args.collection_episodes,
        args.round_size,
        args.minimum_rounds_before_canary,
        args.canary_episodes,
        args.full_stage_pairs,
        args.validation_episodes,
        args.minimum_train_groups,
        args.minimum_validation_groups,
    )
    if min(integers) <= 0:
        parser.error("episode, round, pair, and group counts must be positive")
    if not 0.0 < args.exploration_probability <= 1.0:
        parser.error("exploration probability must be in (0, 1]")
    if args.collection_episodes < args.minimum_train_groups + args.validation_episodes:
        parser.error("collection budget cannot satisfy grouped train/validation split")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
