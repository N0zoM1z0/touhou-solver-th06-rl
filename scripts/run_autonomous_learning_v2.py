#!/usr/bin/env python3
"""Run resumable complete-Stage Wine conservative-RL generation 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.authorize_conservative_canary import authorize  # noqa: E402
from scripts.run_autonomous_learning import _validate_retail_report, _verdict  # noqa: E402
from scripts.shadow_conservative_q import shadow  # noqa: E402
from th06_rl.autonomous_learning import _object, _validate_run  # noqa: E402


SCHEMA = "autonomous-wine-learning-generation-v2"
EXPLORATION_SCHEMA = "th06-rl-uniform-safe-exploration-v1"
EXPLORATION_PLUGIN = REPOSITORY / "src/th06_rl/policies/uniform_safe_exploration.py"
CANDIDATE_PLUGIN = REPOSITORY / "src/th06_rl/policies/autonomous_conservative_q.py"


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_schedule(seed: int, count: int) -> list[dict[str, int]]:
    generator = random.Random(seed)
    game_seeds = generator.sample(range(0x10000), count)
    return [{
        "game_rng_seed": game_seed,
        "policy_seed": generator.randrange(2**64),
    } for game_seed in game_seeds]


def _policy_state(
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


def _complete_run(
    *,
    artifact_dir: Path,
    policy_plugin: Path,
    policy_state: Path,
    scorer: Path | None,
    difficulty: str,
    stage: int,
    rng_seed: int | None,
    corpus_root: Path | None,
) -> tuple[dict[str, object], Path | None]:
    before = _corpus_runs(corpus_root) if corpus_root is not None else set()
    command = [
        sys.executable,
        str(REPOSITORY / "scripts/run_wine_retail.py"),
        "--practice-stage", str(stage),
        "--difficulty", difficulty,
        "--artifact-dir", str(artifact_dir),
        "--policy-plugin", str(policy_plugin),
        "--policy-state", str(policy_state),
        "--immutable-policy",
        "--exploration-rate", "0",
    ]
    if scorer is not None:
        command.extend(("--policy-scorer-library", str(scorer)))
    if corpus_root is not None:
        if rng_seed is None:
            raise ValueError("training corpus requires fixed original-RNG seed")
        command.extend((
            "--complete-stage-training-corpus-root", str(corpus_root),
            "--diagnostic-rng-seed", hex(rng_seed),
        ))
    completed = subprocess.run(command, cwd=REPOSITORY, check=False)
    report = _object(artifact_dir / "report.json")
    mode = (
        "fixed-rng-complete-stage-training"
        if corpus_root is not None else "hit-continuation-benchmark"
    )
    _validate_retail_report(
        report,
        mode=mode,
        diagnostic_rng_seed=rng_seed,
        full_stage=stage,
    )
    if completed.returncode != int(report["controller_returncode"]):
        raise RuntimeError("outer and recorded controller return codes differ")
    run_dir = None
    if corpus_root is not None:
        created = sorted(_corpus_runs(corpus_root) - before)
        if len(created) != 1:
            raise RuntimeError(f"complete Stage created {len(created)} corpus runs")
        run_dir = corpus_root / created[0]
        _run, manifest = _validate_run(run_dir)
        outcome = manifest.get("run_outcome")
        completion = report.get("controller_completion")
        if (
            manifest.get("stage_trajectory_complete") is not True
            or not isinstance(outcome, dict)
            or outcome.get("stage_completed") is not True
            or not isinstance(completion, dict)
            or int(outcome.get("physical_hits", -1))
            != int(completion.get("physical_hits", -2))
        ):
            raise RuntimeError("complete training corpus HIT/stage binding failed")
    return report, run_dir


def _config(args: argparse.Namespace) -> dict[str, object]:
    names = (
        "difficulty", "stage", "generation_seed", "collection_episodes",
        "initial_fit_episodes", "round_size", "validation_episodes",
        "exploration_probability", "n_step_frames", "ensemble_members",
        "bellman_iterations", "trees_per_iteration", "propensity_clip",
        "prototypes_per_action", "support_quantile", "uncertainty_scale",
        "minimum_shadow_rows", "minimum_shadow_proposals",
        "maximum_shadow_p95_ms", "canary_pairs", "full_stage_pairs",
    )
    return {
        **{name: getattr(args, name) for name in names},
        "wine_native_scorer_sha256": _sha256(args.wine_native_scorer),
        "host_native_scorer_sha256": _sha256(args.host_native_scorer),
    }


def _fit(
    args: argparse.Namespace,
    run_dirs: list[Path],
    round_dir: Path,
) -> tuple[Path, dict[str, object]]:
    command = [
        sys.executable,
        str(REPOSITORY / "scripts/fit_conservative_q.py"),
        *(str(path) for path in run_dirs),
        "--output-dir", str(round_dir),
        "--native-scorer", str(args.wine_native_scorer),
        "--shadow-native-scorer", str(args.host_native_scorer),
        "--validation-episodes", str(args.validation_episodes),
        "--exploration-probability", str(args.exploration_probability),
        "--n-step-frames", str(args.n_step_frames),
        "--ensemble-members", str(args.ensemble_members),
        "--bellman-iterations", str(args.bellman_iterations),
        "--trees-per-iteration", str(args.trees_per_iteration),
        "--propensity-clip", str(args.propensity_clip),
        "--prototypes-per-action", str(args.prototypes_per_action),
        "--support-quantile", str(args.support_quantile),
        "--uncertainty-scale", str(args.uncertainty_scale),
        "--seed", str(args.generation_seed),
        "--threads", str(args.threads),
    ]
    fit = subprocess.run(command, cwd=REPOSITORY, check=False)
    if fit.returncode:
        raise RuntimeError(f"conservative offline fit failed with {fit.returncode}")
    state_path = round_dir / "policy-shadow.json"
    report = _object(round_dir / "report.json")
    return state_path, report


def _candidate_round(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    output_root: Path,
    collection_dirs: list[Path],
    seeds: list[dict[str, int]],
) -> Path | None:
    round_index = len(state["rounds"]) + 1
    round_dir = output_root / "rounds" / f"round-{round_index:02d}"
    shadow_state, fit_report = _fit(args, collection_dirs, round_dir)
    row: dict[str, Any] = {
        "round": round_index,
        "episodes": len(collection_dirs),
        "fit_dir": str(round_dir),
        "fit_eligible": bool(fit_report["authorization"]["fit_eligible"]),
        "status": "fit-ineligible",
    }
    state["rounds"].append(row)
    _atomic_json(state_path, state)
    if not row["fit_eligible"]:
        return None
    validation = collection_dirs[-args.validation_episodes:]
    shadow_path = round_dir / "shadow-audit.json"
    audit = shadow(
        shadow_state,
        validation,
        minimum_rows=args.minimum_shadow_rows,
        minimum_proposals=args.minimum_shadow_proposals,
        maximum_p95_ms=args.maximum_shadow_p95_ms,
        native_scorer=args.host_native_scorer,
    )
    _atomic_json(shadow_path, audit)
    row.update({
        "shadow_audit": str(shadow_path),
        "shadow_eligible": bool(audit["shadow_eligible"]),
        "status": "shadow-ineligible",
    })
    _atomic_json(state_path, state)
    if not audit["shadow_eligible"]:
        return None
    active_path = round_dir / "policy-canary.json"
    _atomic_json(active_path, authorize(shadow_state, shadow_path))
    row.update({"status": "canary-ready", "canary_state": str(active_path)})
    _atomic_json(state_path, state)
    return _paired_canary(
        args, state, state_path, output_root, row, active_path, seeds
    )


def _paired_canary(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    output_root: Path,
    round_row: dict[str, Any],
    candidate_state: Path,
    seeds: list[dict[str, int]],
) -> Path | None:
    round_index = int(round_row["round"])
    root = output_root / "canary" / f"round-{round_index:02d}"
    corpus = root / "corpus"
    baseline_state = root / "baseline-state.json"
    _policy_state(
        baseline_state,
        policy_seed=args.generation_seed + round_index,
        exploration_probability=0.0,
    )
    runs = []
    for pair in range(args.canary_pairs):
        seed = seeds[pair]["game_rng_seed"]
        for arm in ("baseline", "candidate"):
            artifact = root / f"pair-{pair:02d}-{arm}"
            report, run_dir = _complete_run(
                artifact_dir=artifact,
                policy_plugin=(
                    EXPLORATION_PLUGIN if arm == "baseline" else CANDIDATE_PLUGIN
                ),
                policy_state=(
                    baseline_state if arm == "baseline" else candidate_state
                ),
                scorer=(None if arm == "baseline" else args.wine_native_scorer),
                difficulty=args.difficulty,
                stage=args.stage,
                rng_seed=seed,
                corpus_root=corpus,
            )
            assert run_dir is not None
            metrics = (report.get("trace") or {}).get("last_policy_metrics")
            completion = report["controller_completion"]
            runs.append({
                "pair": pair,
                "arm": arm,
                "rng_seed": seed,
                "artifact_dir": str(artifact),
                "corpus_run_dir": str(run_dir),
                "physical_hits": int(completion["physical_hits"]),
                "active_overrides": (
                    int(metrics.get("active_overrides", 0))
                    if isinstance(metrics, dict) else 0
                ),
            })
            round_row["canary_runs"] = runs
            _atomic_json(state_path, state)
    result = _verdict(runs)
    candidate_overrides = sum(
        int(row["active_overrides"])
        for row in runs if row["arm"] == "candidate"
    )
    gates = {
        "paired_complete_stages": len(runs) == args.canary_pairs * 2,
        "candidate_exercised": candidate_overrides > 0,
        "strictly_lower_fixed_rng_hits": result["verdict"] == "effective",
    }
    audit = {
        "schema": "autonomous-wine-paired-canary-v2",
        "candidate_state_sha256": _sha256(candidate_state),
        "runs": runs,
        "result": result,
        "gates": gates,
        "canary_eligible": all(gates.values()),
    }
    audit_path = root / "audit.json"
    _atomic_json(audit_path, audit)
    round_row.update({
        "canary_audit": str(audit_path),
        "canary_eligible": audit["canary_eligible"],
        "status": "canary-passed" if audit["canary_eligible"] else "canary-rejected",
    })
    _atomic_json(state_path, state)
    if not audit["canary_eligible"]:
        return None
    evaluation = _object(candidate_state)
    evaluation["selection"]["active_override_budget"] = None
    evaluation["authorization"]["full_evaluation"] = {
        "schema": "autonomous-conservative-full-evaluation-v2",
        "canary_audit_sha256": _sha256(audit_path),
        "candidate_canary_state_sha256": _sha256(candidate_state),
        "fixed_rng_effect": result["effect"],
    }
    path = root / "policy-full-evaluation.json"
    _atomic_json(path, evaluation)
    return path


def _full_stage_ab(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    output_root: Path,
    candidate_state: Path,
) -> dict[str, object]:
    root = output_root / "full-stage"
    baseline_state = root / "baseline-state.json"
    _policy_state(
        baseline_state,
        policy_seed=args.generation_seed,
        exploration_probability=0.0,
    )
    runs = []
    for trial, arm in enumerate(["baseline", "candidate"] * args.full_stage_pairs):
        artifact = root / f"trial-{trial:02d}-{arm}"
        report, _run_dir = _complete_run(
            artifact_dir=artifact,
            policy_plugin=(
                EXPLORATION_PLUGIN if arm == "baseline" else CANDIDATE_PLUGIN
            ),
            policy_state=(baseline_state if arm == "baseline" else candidate_state),
            scorer=(None if arm == "baseline" else args.wine_native_scorer),
            difficulty=args.difficulty,
            stage=args.stage,
            rng_seed=None,
            corpus_root=None,
        )
        metrics = (report.get("trace") or {}).get("last_policy_metrics")
        completion = report["controller_completion"]
        runs.append({
            "trial": trial,
            "arm": arm,
            "artifact_dir": str(artifact),
            "physical_hits": int(completion["physical_hits"]),
            "active_overrides": (
                int(metrics.get("active_overrides", 0))
                if isinstance(metrics, dict) else 0
            ),
        })
        state["full_stage"] = {"status": "running", "runs": runs}
        _atomic_json(state_path, state)
    result = _verdict(runs)
    report = {
        "schema": "autonomous-wine-full-stage-ab-v2",
        "evaluation_mode": "normal-speed-natural-complete-stage-hit-continuation",
        "fixed_rng": False,
        "runs": runs,
        **result,
    }
    report_path = root / "report.json"
    _atomic_json(report_path, report)
    state["full_stage"] = {"status": "complete", "report": str(report_path), "runs": runs}
    return report


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    state_path = output_root / "generation.json"
    config = _config(args)
    if state_path.exists():
        state = _object(state_path)
        if state.get("schema") != SCHEMA or state.get("config") != config:
            raise RuntimeError("refusing generation-2 resume with different config")
        if state.get("status") == "complete":
            print(json.dumps(state["decision"], sort_keys=True))
            return 0
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        state = {
            "schema": SCHEMA,
            "status": "collecting",
            "config": config,
            "episodes": [],
            "rounds": [],
            "full_stage": None,
            "decision": None,
        }
        _atomic_json(state_path, state)
    seeds = _seed_schedule(
        args.generation_seed,
        args.collection_episodes + args.canary_pairs * 4,
    )
    corpus_root = output_root / "collection-corpus"
    try:
        candidate = None
        while len(state["episodes"]) < args.collection_episodes:
            index = len(state["episodes"])
            policy_state = output_root / "behavior-states" / f"episode-{index:03d}.json"
            _policy_state(
                policy_state,
                policy_seed=seeds[index]["policy_seed"],
                exploration_probability=args.exploration_probability,
            )
            artifact = output_root / "collection" / f"episode-{index:03d}"
            report, run_dir = _complete_run(
                artifact_dir=artifact,
                policy_plugin=EXPLORATION_PLUGIN,
                policy_state=policy_state,
                scorer=None,
                difficulty=args.difficulty,
                stage=args.stage,
                rng_seed=seeds[index]["game_rng_seed"],
                corpus_root=corpus_root,
            )
            assert run_dir is not None
            completion = report["controller_completion"]
            state["episodes"].append({
                "episode": index,
                "artifact_dir": str(artifact),
                "corpus_run_dir": str(run_dir),
                "game_rng_seed": seeds[index]["game_rng_seed"],
                "policy_seed": seeds[index]["policy_seed"],
                "physical_hits": int(completion["physical_hits"]),
            })
            _atomic_json(state_path, state)
            completed = len(state["episodes"])
            should_fit = (
                completed >= args.initial_fit_episodes
                and (
                    (completed - args.initial_fit_episodes) % args.round_size == 0
                    or completed == args.collection_episodes
                )
            )
            if should_fit:
                dirs = [Path(row["corpus_run_dir"]) for row in state["episodes"]]
                offset = args.collection_episodes + (len(state["rounds"]) * args.canary_pairs)
                candidate = _candidate_round(
                    args,
                    state,
                    state_path,
                    output_root,
                    dirs,
                    seeds[offset:offset + args.canary_pairs],
                )
                if candidate is not None:
                    break
        if candidate is None:
            decision = {
                "verdict": "ineffective",
                "reason": "evidence budget ended without paired-canary authorization",
                "episodes": len(state["episodes"]),
                "rounds": len(state["rounds"]),
            }
        else:
            state["status"] = "full-stage-evaluation"
            _atomic_json(state_path, state)
            final = _full_stage_ab(
                args, state, state_path, output_root, candidate
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
    parser.add_argument("--generation-seed", type=int, default=260811)
    parser.add_argument("--collection-episodes", type=int, default=8)
    parser.add_argument("--initial-fit-episodes", type=int, default=6)
    parser.add_argument("--round-size", type=int, default=2)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--exploration-probability", type=float, default=0.10)
    parser.add_argument("--n-step-frames", type=int, default=60)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--bellman-iterations", type=int, default=6)
    parser.add_argument("--trees-per-iteration", type=int, default=96)
    parser.add_argument("--propensity-clip", type=float, default=20.0)
    parser.add_argument("--prototypes-per-action", type=int, default=12)
    parser.add_argument("--support-quantile", type=float, default=0.99)
    parser.add_argument("--uncertainty-scale", type=float, default=1.0)
    parser.add_argument("--minimum-shadow-rows", type=int, default=500)
    parser.add_argument("--minimum-shadow-proposals", type=int, default=10)
    parser.add_argument("--maximum-shadow-p95-ms", type=float, default=4.0)
    parser.add_argument("--canary-pairs", type=int, default=1)
    parser.add_argument("--full-stage-pairs", type=int, default=2)
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
    integer_values = (
        args.collection_episodes, args.initial_fit_episodes, args.round_size,
        args.validation_episodes, args.ensemble_members, args.bellman_iterations,
        args.trees_per_iteration, args.prototypes_per_action, args.canary_pairs,
        args.full_stage_pairs, args.threads,
    )
    if min(integer_values) <= 0 or args.ensemble_members < 3:
        parser.error("generation-2 integer bounds are invalid")
    if args.initial_fit_episodes < args.validation_episodes + 3:
        parser.error("initial fit needs three train and two validation Stages")
    if args.collection_episodes < args.initial_fit_episodes:
        parser.error("collection budget is below the initial fit boundary")
    if not 0.0 < args.exploration_probability <= 1.0:
        parser.error("exploration probability must be in (0, 1]")
    if not args.wine_native_scorer.is_file() or not args.host_native_scorer.is_file():
        parser.error("generation-2 native scorer builds are absent")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
