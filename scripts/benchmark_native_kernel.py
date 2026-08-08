#!/usr/bin/env python3
"""Fuzz source-grounded geometry and measure bounded native planning latency."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import time

from th06_rl.core import Kinematics, LocalPlannerConfig
from th06_rl.native import ACTIONS, Aabb, LaserRect, NativeKernel, PackedHazards


KINEMATICS = Kinematics(4.0, 2.0, 2.8284270763397217, 1.4142135381698608)
BY_NAME = {action.name: action for action in ACTIONS}
SOURCE_ANCHORS = {
    "src/GameManager.hpp": (
        "#define GAME_REGION_WIDTH 384.0",
        "#define GAME_REGION_HEIGHT 448.0",
    ),
    "src/Player.cpp": (
        "p->hitboxSize.x = 1.25;",
        "p->hitboxSize.y = 1.25;",
        "this->hitboxTopLeft.x > bulletRight",
    ),
    "src/BulletManager.cpp": (
        "curBullet->exFlags & 0x400",
        "curBullet->exFlags & 0x800",
        "g_Player.CalcKillBoxCollision",
    ),
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def _timings(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "p50_us": _percentile(values, 0.50),
        "p95_us": _percentile(values, 0.95),
        "p99_us": _percentile(values, 0.99),
        "max_us": max(values),
    }


def _collides(
    player_x: float,
    player_y: float,
    half_width: float,
    half_height: float,
    hazard: Aabb,
    margin: float,
) -> bool:
    """Independent scalar transcription of source AABB overlap semantics."""
    gap_x = max(
        hazard.left - (player_x + half_width),
        (player_x - half_width) - hazard.right,
    )
    gap_y = max(
        hazard.top - (player_y + half_height),
        (player_y - half_height) - hazard.bottom,
    )
    if gap_x <= 0.0 and gap_y <= 0.0:
        return max(gap_x, gap_y) <= margin
    return math.hypot(max(gap_x, 0.0), max(gap_y, 0.0)) <= margin


def _source_provenance(root: Path) -> dict[str, object]:
    checked = {}
    for relative, anchors in SOURCE_ANCHORS.items():
        path = root / relative
        text = path.read_text(encoding="utf-8")
        missing = [anchor for anchor in anchors if anchor not in text]
        if missing:
            raise ValueError(f"authoritative source anchors missing from {path}: {missing}")
        checked[relative] = list(anchors)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"path": str(root), "commit": commit, "clean": not dirty, "anchors": checked}


def _random_aabb(rng: random.Random, x: float, y: float, span: float = 28.0) -> Aabb:
    center_x = x + rng.uniform(-span, span)
    center_y = y + rng.uniform(-span, span)
    half_x = rng.uniform(0.5, 8.0)
    half_y = rng.uniform(0.5, 8.0)
    return Aabb(center_x - half_x, center_y - half_y, center_x + half_x, center_y + half_y)


def _random_hazards(rng: random.Random, x: float, y: float, horizon: int) -> PackedHazards:
    aabbs = []
    lasers = []
    for frame in range(horizon):
        count = rng.randrange(0, 8)
        drift = frame * rng.uniform(-1.5, 1.5)
        aabbs.append(tuple(
            _random_aabb(rng, x + drift, y + drift, span=48.0)
            for _ in range(count)
        ))
        lasers.append(tuple(
            LaserRect(
                x + rng.uniform(-80.0, 80.0),
                y + rng.uniform(-80.0, 80.0),
                rng.uniform(-math.pi, math.pi),
                rng.uniform(10.0, 100.0),
                rng.uniform(20.0, 160.0),
                rng.uniform(2.0, 12.0),
            )
            for _ in range(rng.randrange(0, 2))
        ))
    return PackedHazards(tuple(aabbs), tuple(lasers))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reference/GensokyoClub-th06"))
    parser.add_argument("--oracle-cases", type=int, default=4000)
    parser.add_argument("--gate-cases", type=int, default=800)
    parser.add_argument("--planner-cases", type=int, default=160)
    parser.add_argument("--seed", type=int, default=6006)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.oracle_cases, args.gate_cases, args.planner_cases) <= 0:
        parser.error("benchmark case counts must be positive")
    rng = random.Random(args.seed)
    kernel = NativeKernel()
    provenance = _source_provenance(args.source)

    oracle_mismatches = 0
    oracle_us = []
    for _ in range(args.oracle_cases):
        x = rng.uniform(8.0, 376.0)
        y = rng.uniform(16.0, 432.0)
        half_width = rng.uniform(0.5, 3.0)
        half_height = rng.uniform(0.5, 3.0)
        margin = rng.uniform(0.0, 1.0)
        hazard = _random_aabb(rng, x, y)
        packed = NativeKernel.prepare_hazards(PackedHazards(((hazard,),), ((),)))
        started = time.perf_counter_ns()
        certified = kernel.certify_actions(
            x=x,
            y=y,
            half_width=half_width,
            half_height=half_height,
            kinematics=KINEMATICS,
            current_action=BY_NAME["stay"],
            hazards=packed,
            candidates=(BY_NAME["stay"],),
            delivery_delays=(0,),
            collision_margin=margin,
        )
        oracle_us.append((time.perf_counter_ns() - started) / 1000.0)
        if bool(certified) == _collides(x, y, half_width, half_height, hazard, margin):
            oracle_mismatches += 1

    gate_us = []
    margin_monotonicity_failures = 0
    deterministic_failures = 0
    endpoint_bound_failures = 0
    planner_us = []
    planner_results = 0
    planner_authority_failures = 0
    planner_bound_failures = 0
    for case in range(max(args.gate_cases, args.planner_cases)):
        x = rng.uniform(16.0, 368.0)
        y = rng.uniform(24.0, 424.0)
        hazards = _random_hazards(rng, x, y, 12)
        current = rng.choice(ACTIONS)
        prepared = NativeKernel.prepare_hazards(hazards)
        if case < args.gate_cases:
            started = time.perf_counter_ns()
            normal = kernel.certify_actions(
                x=x,
                y=y,
                half_width=1.25,
                half_height=1.25,
                kinematics=KINEMATICS,
                current_action=current,
                hazards=prepared.prefix(4),
            )
            gate_us.append((time.perf_counter_ns() - started) / 1000.0)
            strict = kernel.certify_actions(
                x=x,
                y=y,
                half_width=1.25,
                half_height=1.25,
                kinematics=KINEMATICS,
                current_action=current,
                hazards=prepared.prefix(4),
                collision_margin=0.75,
            )
            repeated = kernel.certify_actions(
                x=x,
                y=y,
                half_width=1.25,
                half_height=1.25,
                kinematics=KINEMATICS,
                current_action=current,
                hazards=prepared.prefix(4),
            )
            normal_actions = {item.action for item in normal}
            strict_actions = {item.action for item in strict}
            margin_monotonicity_failures += not strict_actions.issubset(normal_actions)
            deterministic_failures += repeated != normal
            endpoint_bound_failures += sum(
                not (8.0 <= item.final_x <= 376.0 and 16.0 <= item.final_y <= 432.0)
                for item in normal
            )
        if case < args.planner_cases:
            hard = kernel.certify_actions(
                x=x,
                y=y,
                half_width=1.25,
                half_height=1.25,
                kinematics=KINEMATICS,
                current_action=current,
                hazards=prepared.prefix(4),
            )
            if not hard:
                continue
            started = time.perf_counter_ns()
            plan = kernel.plan(
                x=x,
                y=y,
                half_width=1.25,
                half_height=1.25,
                kinematics=KINEMATICS,
                current_action=current,
                hazards=hazards,
                hard=hard,
                config=LocalPlannerConfig(horizon=12),
            )
            planner_us.append((time.perf_counter_ns() - started) / 1000.0)
            if plan is None:
                continue
            planner_results += 1
            hard_actions = {item.action for item in hard}
            planner_authority_failures += plan.action not in hard_actions
            planner_bound_failures += not (
                8.0 <= plan.terminal_x <= 376.0
                and 16.0 <= plan.terminal_y <= 432.0
                and 1 <= plan.effort_horizon <= 12
                and plan.endpoint_count > 0
                and plan.continuation_action_count > 0
            )

    result = {
        "schema": "th06-rl-native-benchmark-v1",
        "source": provenance,
        "native_library": str(kernel.path),
        "seed": args.seed,
        "geometry": {
            "oracle_cases": args.oracle_cases,
            "oracle_mismatches": oracle_mismatches,
            "certify_one_action_prepared": _timings(oracle_us),
        },
        "gate": {
            "cases": args.gate_cases,
            "margin_monotonicity_failures": margin_monotonicity_failures,
            "deterministic_failures": deterministic_failures,
            "endpoint_bound_failures": endpoint_bound_failures,
            "certify_18_actions_h4_prepared": _timings(gate_us),
        },
        "planner": {
            "attempted_cases": args.planner_cases,
            "timed_nonempty_hard_cases": len(planner_us),
            "available_results": planner_results,
            "selected_outside_hard_failures": planner_authority_failures,
            "result_bound_failures": planner_bound_failures,
            "plan_h12": _timings(planner_us),
        },
        "passes": not any((
            oracle_mismatches,
            margin_monotonicity_failures,
            deterministic_failures,
            endpoint_bound_failures,
            planner_authority_failures,
            planner_bound_failures,
        )),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
