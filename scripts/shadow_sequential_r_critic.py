#!/usr/bin/env python3
"""Replay a Generation-4 native population on exact factual Wine options."""

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
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.shadow_option_advantage import _context  # noqa: E402
from th06_rl.advantage_learning import _object, _rows  # noqa: E402
from th06_rl.audited_option_loader import load_audited_option_episode  # noqa: E402
from th06_rl.policies.autonomous_sequential_r_critic import (  # noqa: E402
    AutonomousSequentialRCriticPolicy,
)
from th06_rl.policies.offline_ranker import NATIVE_SCORER_ENV  # noqa: E402
from th06_rl.sequential_learning import (  # noqa: E402
    CRITIC_TREES,
    POPULATION_MEMBERS,
    STATE_SCHEMA,
)


SCHEMA = "autonomous-generation-4-native-shadow-audit-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _p95(values: list[float]) -> float:
    if not values:
        return float("inf")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def shadow(
    state_path: Path,
    run_dirs: list[Path],
    *,
    native_scorer: Path,
    maximum_p95_ms: float = 4.0,
    policy_type=AutonomousSequentialRCriticPolicy,
    state_schema: str = STATE_SCHEMA,
    population_members: int = POPULATION_MEMBERS,
    full_trees: int = CRITIC_TREES,
    report_schema: str = SCHEMA,
    generation_label: str = "Generation-4",
) -> dict[str, object]:
    state_path = state_path.resolve()
    native_scorer = native_scorer.resolve()
    state = _object(state_path)
    authorization = state.get("authorization")
    if (
        state.get("schema") != state_schema
        or state.get("mode") != "shadow"
        or not isinstance(authorization, dict)
        or authorization.get("fit_eligible") is not True
    ):
        raise ValueError(
            f"{generation_label} shadow requires a fit-eligible shadow state"
        )
    prior = os.environ.get(NATIVE_SCORER_ENV)
    try:
        os.environ[NATIVE_SCORER_ENV] = str(native_scorer)
        policy = policy_type()
        policy.import_state(state)
    finally:
        if prior is None:
            os.environ.pop(NATIVE_SCORER_ENV, None)
        else:
            os.environ[NATIVE_SCORER_ENV] = prior

    decisions = invalid_publications = 0
    timings: list[float] = []
    runs = []
    observed_groups = set()
    for raw_dir in run_dirs:
        run_dir = raw_dir.resolve()
        run = _object(run_dir / "run.json")
        schemas = run.get("schemas")
        transition = schemas.get("transition") if isinstance(schemas, dict) else None
        loaded, report = load_audited_option_episode(run_dir)
        episode_id = str(report["episode_id"])
        if episode_id in observed_groups:
            raise ValueError(f"{generation_label} shadow received a duplicate episode")
        observed_groups.add(episode_id)
        before = policy.metrics()
        replayed = 0
        manifest = _object(run_dir / "manifest.json")
        for row in _rows(run_dir, manifest, transition_schema=str(transition)):
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
            timings.append((time.perf_counter() - started) * 1_000.0)
            replayed += 1
            decisions += 1
            invalid_publications += (
                decision.action != context.baseline_action
                or decision.action not in context.locally_admissible_actions
            )
        if replayed != len(loaded):
            raise ValueError(f"{generation_label} shadow missed factual option boundaries")
        after = policy.metrics()
        runs.append({
            "episode_id": episode_id,
            "run_dir": str(run_dir),
            "factual_options": replayed,
            "shadow_proposals": (
                int(after["shadow_proposals"]) - int(before["shadow_proposals"])
            ),
        })
    metrics = policy.metrics()
    p95 = _p95(timings)
    gates = {
        "nonempty_exact_episode_set": len(observed_groups) == len(run_dirs) > 0,
        "exact_factual_option_count": decisions == sum(
            int(row["factual_options"]) for row in runs
        ),
        "baseline_only_publication": invalid_publications == 0,
        "proposal_witnessed": int(metrics["shadow_proposals"]) > 0,
        "native_batch_scorer": metrics["scorer_backend"] == "native-batch",
        "complete_population": metrics["population_members"] == population_members,
        "full_tree_population": metrics["trees_per_member"] == full_trees,
        "maximum_p95_latency": p95 <= maximum_p95_ms,
        "zero_controller_deadline_misses": metrics["controller_deadline_misses"] == 0,
    }
    return {
        "schema": report_schema,
        "policy_state": str(state_path),
        "policy_state_sha256": _sha256(state_path),
        "native_scorer": str(native_scorer),
        "native_scorer_sha256": _sha256(native_scorer),
        "audit_episode_groups": sorted(observed_groups),
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
