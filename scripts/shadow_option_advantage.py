#!/usr/bin/env python3
"""Replay a Generation-3 native population on exact held-out Wine options."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.advantage_learning import (  # noqa: E402
    POPULATION_MEMBERS,
    STATE_SCHEMA,
    _object,
    _rows,
    _validate_run,
    load_option_episode,
)
from th06_rl.policies.autonomous_dr_option_advantage import (  # noqa: E402
    AutonomousDROptionAdvantagePolicy,
)
from th06_rl.policies.offline_ranker import NATIVE_SCORER_ENV  # noqa: E402
from th06_rl.policy_api import PolicyContext  # noqa: E402


SCHEMA = "autonomous-generation-3-native-shadow-audit-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _p95(values: list[float]) -> float:
    if not values:
        return float("inf")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _pairs(raw: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(raw, list):
        raise TypeError("shadow feature vector is not a list")
    return tuple((str(name), float(value)) for name, value in raw)


def _action_pairs(
    raw: object,
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    if not isinstance(raw, list):
        raise TypeError("shadow action feature vector is not a list")
    return tuple((str(action), _pairs(features)) for action, features in raw)


def _hazards(raw: object) -> tuple[tuple[float, ...], ...]:
    if not isinstance(raw, list):
        raise TypeError("shadow hazard primitive set is not a list")
    return tuple(tuple(float(value) for value in primitive) for primitive in raw)


def _context(row: dict[str, object]) -> PolicyContext:
    replay = row.get("policy_context")
    legal_raw = row.get("legal_actions")
    if not isinstance(replay, dict) or not isinstance(legal_raw, list):
        raise TypeError("shadow option boundary lacks adapter context")
    legal = tuple(str(action) for action in legal_raw)
    baseline = str(row.get("baseline_action", ""))
    if not legal or baseline not in legal:
        raise ValueError("shadow option boundary has an invalid native-safe set")
    return PolicyContext(
        frame=int(row["sequence"]),
        scope=(0, 0, 0, 0),
        source_context="adapter-hidden-from-learner",
        baseline_action=baseline,
        locally_admissible_actions=legal,
        player_x=float(replay.get("player_x", 0.0)),
        player_y=float(replay.get("player_y", 0.0)),
        power=int(replay.get("power", 0)),
        bullet_count=int(replay.get("bullet_count", 0)),
        laser_count=int(replay.get("laser_count", 0)),
        hard_action_count=int(replay.get("hard_action_count", len(legal))),
        exploration_rate=0.0,
        current_action=str(replay.get("current_action", "stay")),
        hard_admissible_actions=tuple(map(
            str, replay.get("hard_admissible_actions", ())
        )),
        effort_horizon=int(replay.get("effort_horizon", 0)),
        observation_features=_pairs(replay.get("observation_features")),
        action_features=_action_pairs(replay.get("action_features")),
        hazard_primitives=_hazards(replay.get("hazard_primitives")),
        history_features=_pairs(replay.get("history_features")),
    )


def shadow(
    state_path: Path,
    run_dirs: list[Path],
    *,
    native_scorer: Path,
    exploration_probability: float = 0.10,
    maximum_p95_ms: float = 4.0,
) -> dict[str, object]:
    state_path = state_path.resolve()
    native_scorer = native_scorer.resolve()
    state = _object(state_path)
    if state.get("schema") != STATE_SCHEMA or state.get("mode") != "shadow":
        raise ValueError("Generation-3 shadow audit requires a shadow state")
    prior = os.environ.get(NATIVE_SCORER_ENV)
    try:
        os.environ[NATIVE_SCORER_ENV] = str(native_scorer)
        policy = AutonomousDROptionAdvantagePolicy()
        policy.import_state(state)
    finally:
        if prior is None:
            os.environ.pop(NATIVE_SCORER_ENV, None)
        else:
            os.environ[NATIVE_SCORER_ENV] = prior

    fit_report = state.get("fit_report")
    if not isinstance(fit_report, dict):
        raise TypeError("Generation-3 fit report is absent")
    expected_groups = set(map(str, fit_report.get("validation_groups", ())))
    expected_options = int(fit_report.get("validation_options", -1))
    observed_groups = set()
    decisions = 0
    invalid_publications = 0
    timings = []
    runs = []
    for raw_dir in run_dirs:
        run_dir = raw_dir.resolve()
        run, manifest = _validate_run(run_dir)
        episode_id = str(run.get("run_id", run_dir.name))
        observed_groups.add(episode_id)
        loaded, _report = load_option_episode(
            run_dir,
            exploration_probability=exploration_probability,
        )
        expected_run_options = len(loaded)
        before = policy.metrics()
        replayed = 0
        for row in _rows(run_dir, manifest):
            option = row.get("option")
            if (
                not isinstance(option, dict)
                or option.get("boundary") is not True
                or row.get("executed_action") != option.get("intent")
            ):
                continue
            context = _context(row)
            started = time.perf_counter()
            decision = policy.decide(context)
            timings.append((time.perf_counter() - started) * 1000.0)
            decisions += 1
            replayed += 1
            invalid_publications += (
                decision.action != context.baseline_action
                or decision.action not in context.locally_admissible_actions
            )
        if replayed != expected_run_options:
            raise ValueError("shadow did not replay every factual option boundary")
        after = policy.metrics()
        runs.append({
            "episode_id": episode_id,
            "run_dir": str(run_dir),
            "factual_options": replayed,
            "shadow_proposals": (
                int(after["shadow_proposals"])
                - int(before["shadow_proposals"])
            ),
        })
    if observed_groups != expected_groups:
        raise ValueError("shadow runs do not exactly match held-out groups")
    metrics = policy.metrics()
    p95 = _p95(timings)
    gates = {
        "exact_heldout_episode_groups": observed_groups == expected_groups,
        "exact_heldout_option_count": decisions == expected_options,
        "baseline_only_publication": invalid_publications == 0,
        "native_batch_scorer": metrics["scorer_backend"] == "native-batch",
        "complete_population": metrics["population_members"] == POPULATION_MEMBERS,
        "maximum_p95_latency": p95 <= maximum_p95_ms,
        "zero_controller_deadline_misses": (
            int(metrics["controller_deadline_misses"]) == 0
        ),
    }
    return {
        "schema": SCHEMA,
        "policy_state": str(state_path),
        "policy_state_sha256": _sha256(state_path),
        "native_scorer": str(native_scorer),
        "native_scorer_sha256": _sha256(native_scorer),
        "heldout_episode_groups": sorted(observed_groups),
        "runs": runs,
        "decisions": decisions,
        "invalid_publications": invalid_publications,
        "policy_metrics": metrics,
        "latency": {
            "mean_ms": statistics.fmean(timings) if timings else None,
            "p95_ms": p95 if timings else None,
            "max_ms": max(timings) if timings else None,
        },
        "thresholds": {"maximum_p95_ms": maximum_p95_ms},
        "gates": gates,
        "shadow_eligible": all(gates.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--native-scorer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-p95-ms", type=float, default=4.0)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace shadow audit: {args.output}")
    report = shadow(
        args.state,
        args.runs,
        native_scorer=args.native_scorer,
        maximum_p95_ms=args.maximum_p95_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "shadow_eligible": report["shadow_eligible"],
        "decisions": report["decisions"],
        "proposals": report["policy_metrics"]["shadow_proposals"],
        "p95_ms": report["latency"]["p95_ms"],
    }, sort_keys=True))
    return 0 if report["shadow_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
