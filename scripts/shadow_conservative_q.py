#!/usr/bin/env python3
"""Replay generation-2 immutable Q committee on held-out physical states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.autonomous_learning import _object, _transition_rows, _validate_run  # noqa: E402
from th06_rl.policies.autonomous_conservative_q import (  # noqa: E402
    AutonomousConservativeQPolicy,
)
from th06_rl.policies.offline_ranker import NATIVE_SCORER_ENV  # noqa: E402
from th06_rl.policy_api import PolicyContext  # noqa: E402


SCHEMA = "autonomous-conservative-q-shadow-audit-v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))]


def shadow(
    state_path: Path,
    run_dirs: list[Path],
    *,
    minimum_rows: int,
    minimum_proposals: int,
    maximum_p95_ms: float,
    native_scorer: Path | None = None,
) -> dict[str, object]:
    state = _object(state_path)
    if state.get("mode") != "shadow":
        raise ValueError("shadow audit requires a shadow state")
    prior = os.environ.get(NATIVE_SCORER_ENV)
    try:
        if native_scorer is not None:
            os.environ[NATIVE_SCORER_ENV] = str(native_scorer.resolve())
        policy = AutonomousConservativeQPolicy()
        policy.import_state(state)
    finally:
        if native_scorer is not None:
            if prior is None:
                os.environ.pop(NATIVE_SCORER_ENV, None)
            else:
                os.environ[NATIVE_SCORER_ENV] = prior
    expected_groups = set(state["fit_report"]["validation_groups"])
    observed_groups = set()
    decisions = 0
    invalid_publications = 0
    timings = []
    runs = []
    for raw_dir in run_dirs:
        run_dir = raw_dir.resolve()
        run, manifest = _validate_run(run_dir)
        episode_id = str(run["run_id"])
        observed_groups.add(episode_id)
        before = policy.metrics()
        eligible = 0
        for row in _transition_rows(run_dir, manifest):
            replay = row.get("policy_context")
            legal_raw = row.get("legal_actions")
            if (
                row.get("learning_eligible") is not True
                or not isinstance(replay, dict)
                or not isinstance(legal_raw, list)
                or not isinstance(replay.get("observation_features"), list)
                or not isinstance(replay.get("action_features"), list)
            ):
                continue
            legal = tuple(str(action) for action in legal_raw)
            baseline = str(row.get("baseline_action", ""))
            if not legal or baseline not in legal:
                raise ValueError("shadow row has an invalid native-safe set")
            context = PolicyContext(
                frame=int(row["sequence"]),
                scope=(0, 0, 0, 0),
                source_context="adapter-hidden-from-learner",
                baseline_action=baseline,
                locally_admissible_actions=legal,
                player_x=0.0,
                player_y=0.0,
                power=0,
                bullet_count=0,
                laser_count=0,
                hard_action_count=len(legal),
                exploration_rate=0.0,
                current_action=str(replay.get("current_action", "stay")),
                observation_features=tuple(
                    (str(name), float(value))
                    for name, value in replay["observation_features"]
                ),
                action_features=tuple(
                    (
                        str(action),
                        tuple(
                            (str(name), float(value))
                            for name, value in features
                        ),
                    )
                    for action, features in replay["action_features"]
                ),
            )
            started = time.perf_counter()
            decision = policy.decide(context)
            timings.append((time.perf_counter() - started) * 1000.0)
            decisions += 1
            eligible += 1
            invalid_publications += decision.action not in legal or decision.action != baseline
        after = policy.metrics()
        runs.append({
            "episode_id": episode_id,
            "run_dir": str(run_dir),
            "eligible_rows": eligible,
            "shadow_proposals": (
                int(after["shadow_proposals"]) - int(before["shadow_proposals"])
            ),
        })
    if observed_groups != expected_groups:
        raise ValueError("shadow runs do not exactly match held-out groups")
    metrics = policy.metrics()
    p95 = _percentile(timings, 0.95) if timings else float("inf")
    gates = {
        "exact_heldout_groups": observed_groups == expected_groups,
        "minimum_rows": decisions >= minimum_rows,
        "minimum_proposals": int(metrics["shadow_proposals"]) >= minimum_proposals,
        "baseline_only_publication": invalid_publications == 0,
        "native_batch_scorer": metrics["scorer_backend"] == "native-batch",
        "maximum_p95_latency": p95 <= maximum_p95_ms,
    }
    return {
        "schema": SCHEMA,
        "policy_state": str(state_path.resolve()),
        "policy_state_sha256": _sha256(state_path),
        "heldout_episode_groups": sorted(observed_groups),
        "runs": runs,
        "decisions": decisions,
        "policy_metrics": metrics,
        "invalid_publications": invalid_publications,
        "latency": {
            "mean_ms": statistics.fmean(timings) if timings else None,
            "p95_ms": p95 if timings else None,
            "max_ms": max(timings) if timings else None,
        },
        "thresholds": {
            "minimum_rows": minimum_rows,
            "minimum_proposals": minimum_proposals,
            "maximum_p95_ms": maximum_p95_ms,
        },
        "gates": gates,
        "shadow_eligible": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--native-scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=500)
    parser.add_argument("--minimum-proposals", type=int, default=10)
    parser.add_argument("--maximum-p95-ms", type=float, default=4.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace shadow audit: {args.output}")
    report = shadow(
        args.state,
        args.runs,
        minimum_rows=args.minimum_rows,
        minimum_proposals=args.minimum_proposals,
        maximum_p95_ms=args.maximum_p95_ms,
        native_scorer=args.native_scorer,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "shadow_eligible": report["shadow_eligible"],
        "decisions": report["decisions"],
        "proposals": report["policy_metrics"]["shadow_proposals"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
