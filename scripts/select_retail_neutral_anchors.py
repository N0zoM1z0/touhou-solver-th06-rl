#!/usr/bin/env python3
"""Select fixed neutral-family anchors from two disjoint Wine prefixes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from th06_rl.wine_risk import FROZEN_INCUMBENT_POLICY_ID, load_first_failure_prefix


SCHEMA = "th06-rl-retail-neutral-anchor-panel-v1"
TARGET_CONTEXT = "boss:0:sub31:life_cb31:timer_cb19:spell"
INCUMBENT_ACTION = "down_right"
BASELINE_ACTION = "down_left"
NEUTRAL_ACTIONS = ("stay", "stay_fast")


def _eligible(example: Any, *, target_context: str = TARGET_CONTEXT) -> bool:
    features = example.features
    transition = example.transition
    return bool(
        example.failure_within_120
        and transition.source_context == target_context
        and features["action"] == INCUMBENT_ACTION
        and features["baseline_action"] == BASELINE_ACTION
        and float(features["edge_reserve"]) > 16.0
        and float(features["laser_count"]) > 0.0
        and float(features["hard_action_count"]) >= 5.0
        and all(
            float(features[f"hard_{action}"]) == 1.0
            for action in (INCUMBENT_ACTION, *NEUTRAL_ACTIONS)
        )
        and all(
            action in transition.legal_actions
            for action in (INCUMBENT_ACTION, *NEUTRAL_ACTIONS)
        )
    )


def select_anchor(
    examples: Sequence[Any],
    *,
    target_context: str = TARGET_CONTEXT,
) -> tuple[dict[str, Any] | None, int]:
    eligible = [
        example
        for example in examples
        if _eligible(example, target_context=target_context)
    ]
    if not eligible:
        return None, 0
    selected = min(
        eligible,
        key=lambda example: (
            int(example.frames_to_failure),
            -int(example.transition.sequence),
        ),
    )
    features: Mapping[str, str | float] = selected.features
    return (
        {
            "sequence": int(selected.transition.sequence),
            "frame": int(selected.transition.frame),
            "frames_to_failure": int(selected.frames_to_failure),
            "source_context": str(selected.transition.source_context),
            "incumbent_action": str(features["action"]),
            "baseline_action": str(features["baseline_action"]),
            "hard_action_count": int(float(features["hard_action_count"])),
            "legal_action_count": len(selected.transition.legal_actions),
            "edge_reserve": float(features["edge_reserve"]),
            "bullet_count": int(float(features["bullet_count"])),
            "laser_count": int(float(features["laser_count"])),
        },
        len(eligible),
    )


def select_panel(
    run_directories: Sequence[Path],
    *,
    expected_scope: tuple[int, int, int, int],
    expected_executable_sha256: str,
    expected_native_kernel_sha256: str,
) -> dict[str, Any]:
    if len(run_directories) != 2:
        raise ValueError("neutral anchor panel requires exactly two Wine episodes")
    rows = []
    run_ids = set()
    run_hashes = set()
    for raw_path in run_directories:
        prefix = load_first_failure_prefix(
            raw_path.resolve(),
            expected_scope=expected_scope,
            expected_executable_sha256=expected_executable_sha256,
            expected_native_kernel_sha256=expected_native_kernel_sha256,
            expected_policy_id=FROZEN_INCUMBENT_POLICY_ID,
        )
        if prefix.run_id in run_ids or prefix.run_sha256 in run_hashes:
            raise ValueError("neutral anchor panel episodes are not independent")
        run_ids.add(prefix.run_id)
        run_hashes.add(prefix.run_sha256)
        selected, eligible_count = select_anchor(prefix.examples)
        rows.append(
            {
                "run_id": prefix.run_id,
                "run": str(prefix.run_dir),
                "manifest_sha256": prefix.manifest_sha256,
                "run_sha256": prefix.run_sha256,
                "code_commit": prefix.code_commit,
                "failure_kind": prefix.failure_kind,
                "failure_frame": prefix.failure_frame,
                "failure_context": prefix.failure_context,
                "eligible_anchor_rows": eligible_count,
                "anchor": selected,
            }
        )
    rows.sort(key=lambda row: str(row["run_id"]))
    passed = all(
        row["failure_kind"] == "control-dead-end"
        and row["failure_context"] == TARGET_CONTEXT
        and row["anchor"] is not None
        for row in rows
    )
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": list(expected_scope),
        "selection_contract": {
            "failure_context": TARGET_CONTEXT,
            "failure_window_frames": 120,
            "boundary_regime": "edge_reserve > 16",
            "hazard_regime": "laser_count > 0",
            "safe_set_regime": "hard_action_count >= 5",
            "incumbent_action": INCUMBENT_ACTION,
            "baseline_action": BASELINE_ACTION,
            "required_native_safe_actions": [INCUMBENT_ACTION, *NEUTRAL_ACTIONS],
            "tie_break": "minimum frames_to_failure, then maximum sequence",
        },
        "episodes": rows,
        "gate": {
            "passed": passed,
            "anchors": sum(row["anchor"] is not None for row in rows),
            "headless_cow_allowed": passed,
            "active_candidates": 0,
        },
        "evidence_boundary": {
            "episode_grouped": True,
            "training_corpus": False,
            "promotion_authority": False,
            "headless_cow_reject_only": True,
            "wine_shadow_required": True,
            "complete_wine_stage_hit_count_required": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--scope", default="3/0/0/6")
    parser.add_argument("--retail-sha256", required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    scope = tuple(int(value) for value in args.scope.split("/"))
    if len(scope) != 4:
        parser.error("scope must contain four integers")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = select_panel(
            args.runs,
            expected_scope=scope,  # type: ignore[arg-type]
            expected_executable_sha256=args.retail_sha256,
            expected_native_kernel_sha256=args.native_sha256,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "passed": report["gate"]["passed"],
                "anchors": report["gate"]["anchors"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
