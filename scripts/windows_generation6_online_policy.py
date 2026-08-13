#!/usr/bin/env python3
"""Stdlib-only Wine runner for the complete Generation-6 online policy path."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

from th06_rl.policy_api import PolicyContext
from th06_rl.policies.autonomous_iql_actor import AutonomousIqlActorPolicy
from th06_rl.policies.offline_ranker import NATIVE_SCORER_ENV


def _context(value: dict[str, object]) -> PolicyContext:
    value = dict(value)
    for name in (
        "scope", "locally_admissible_actions", "hard_admissible_actions",
    ):
        value[name] = tuple(value[name])
    value["hard_action_evaluations"] = tuple(
        tuple(row) for row in value["hard_action_evaluations"]
    )
    value["observation_features"] = tuple(
        tuple(row) for row in value["observation_features"]
    )
    value["action_features"] = tuple(
        (row[0], tuple(tuple(feature) for feature in row[1]))
        for row in value["action_features"]
    )
    value["hazard_primitives"] = tuple(
        tuple(row) for row in value["hazard_primitives"]
    )
    value["history_features"] = tuple(
        tuple(row) for row in value["history_features"]
    )
    return PolicyContext(**value)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--repetitions", type=int, default=1200)
    args = parser.parse_args()
    os.environ[NATIVE_SCORER_ENV] = args.library
    fixture = json.load(open(args.fixture, encoding="utf-8"))
    contexts = [_context(row) for row in fixture["contexts"]]
    state = json.load(open(args.state, encoding="utf-8"))
    loaded_at = time.perf_counter()
    policy = AutonomousIqlActorPolicy()
    policy.import_state(state)
    load_seconds = time.perf_counter() - loaded_at
    choices = [policy._proposal(
        context,
        tuple(context.locally_admissible_actions),
        context.baseline_action,
    ) for context in contexts]
    latencies = []
    for index in range(args.repetitions):
        context = contexts[index % len(contexts)]
        started = time.perf_counter()
        policy._proposal(
            context,
            tuple(context.locally_admissible_actions),
            context.baseline_action,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
    print(json.dumps({
        "choices": choices,
        "load_seconds": load_seconds,
        "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
        "latency_p95_ms": _p95(latencies),
        "latency_max_ms": max(latencies),
        "deadline_misses": sum(value > 1000.0 / 60.0 for value in latencies),
        "over_four_ms": sum(value > 4.0 for value in latencies),
        "samples": len(latencies),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
