#!/usr/bin/env python3
"""Audit conservative sequential tuples for CPU offline RL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from th06_rl.offline import iter_run_transitions, load_dataset_index
from th06_rl.offline_rl import sequential_transitions, summarize_sequential


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--scope", required=True, help="difficulty/character/shot/stage")
    parser.add_argument("--view", choices=("exact-v5", "common"), default="exact-v5")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scope = tuple(int(item) for item in args.scope.split("/"))
    if len(scope) != 4:
        parser.error("scope must contain exactly four integers")
    _, indexed = load_dataset_index(args.dataset)
    runs = [
        run for run in indexed
        if run.scope == scope
        and run.training_eligible
        and (args.view == "common" or run.transition_schema == "th06-rl-transition-v5")
    ]
    if not runs:
        raise SystemExit("no eligible complete Stages in scope/view")
    combined = []
    by_run = {}
    action_counts: Counter[str] = Counter()
    for run in sorted(runs, key=lambda item: item.run_id):
        transitions = sequential_transitions(
            iter_run_transitions(args.dataset, run, verify_sha256=False),
            run,
            exact_context_only=args.view == "exact-v5",
        )
        by_run[run.run_id] = summarize_sequential(transitions)
        combined.extend(transitions)
        action_counts.update(row.state.action for row in transitions)
    result = {
        "schema": "th06-rl-offline-sequence-audit-v1",
        "dataset_revision": args.revision,
        "scope": list(scope),
        "view": args.view,
        "runs": len(runs),
        "overall": summarize_sequential(combined),
        "selected_action_counts": dict(sorted(action_counts.items())),
        "by_run": by_run,
        "contract": {
            "hit_penalty_attached_once_to_latest_eligible_action": True,
            "bootstraps_across_gap": False,
            "bootstraps_across_source_context": False,
            "bootstraps_across_physical_hit": False,
            "next_action_set_is_recorded_native_legal_set": True,
            "bomb_representable": False,
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
