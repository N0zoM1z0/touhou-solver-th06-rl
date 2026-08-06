#!/usr/bin/env python3
"""Measure coarse and hierarchical UCB aliasing in one completed corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

from th06_rl.policies.adaptive import AdaptivePolicy
from th06_rl.policy_api import PolicyContext


def _rows(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)


def _stream_paths(run_dir: Path, manifest: dict, stream: str) -> list[Path]:
    return [
        run_dir / item["path"]
        for item in manifest.get("shards", ())
        if item.get("stream") == stream
    ]


def _clearance_signature(hard_actions) -> tuple[tuple[str, object], ...]:
    return tuple(
        (
            str(item[0]),
            "inf" if item[1] is None else math.floor(float(item[1]) / 2.0),
        )
        for item in hard_actions
    )


def _group_summary(groups, minimum_support: int) -> dict[str, object]:
    supported = [item for item in groups.values() if item["records"] >= minimum_support]
    mixed = [
        item for item in supported
        if 0 < item["next_hard_empty"] < item["records"]
    ]
    total = sum(item["records"] for item in groups.values())
    multi_signature_records = sum(
        item["records"]
        for item in groups.values()
        if len(item["physical_signatures"]) > 1
    )
    return {
        "context_actions": len(groups),
        "minimum_support": minimum_support,
        "supported_context_actions": len(supported),
        "mixed_safe_and_next_hard_empty": len(mixed),
        "records_in_mixed_groups": sum(item["records"] for item in mixed),
        "next_hard_empty_in_mixed_groups": sum(
            item["next_hard_empty"] for item in mixed
        ),
        "multi_physical_signature_context_actions": sum(
            len(item["physical_signatures"]) > 1 for item in supported
        ),
        "records_in_multi_physical_signature_groups": multi_signature_records,
        "multi_physical_signature_record_rate": (
            multi_signature_records / total if total else None
        ),
    }


def analyze(run_dir: Path, minimum_support: int) -> dict[str, object]:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    policy = AdaptivePolicy()
    frame_evidence = {}
    active_context = None
    context_start_frame = None
    for row in _rows(_stream_paths(run_dir, manifest, "frames")):
        snapshot = row["snapshot"]
        decision = row["decision"]
        source_context = str(row["scope"]["phase_id"])
        frame = int(snapshot["frame"])
        if source_context != active_context:
            active_context = source_context
            context_start_frame = frame
        assert context_start_frame is not None
        hard_actions = tuple(decision.get("hard_actions", ()))
        hard_names = tuple(str(item[0]) for item in hard_actions)
        legal = tuple(str(item) for item in decision.get(
            "locally_admissible_actions", ()
        ))
        current_action = decision.get("current_action")
        baseline_action = decision.get("baseline_action")
        published = decision.get("published_action")
        if not all(isinstance(value, str) for value in (
            current_action,
            baseline_action,
            published,
        )):
            continue
        context = PolicyContext(
            frame=frame,
            scope=tuple(int(row["scope"][name]) for name in (
                "difficulty", "character", "shot_type", "stage"
            )),
            source_context=source_context,
            baseline_action=baseline_action,
            locally_admissible_actions=legal,
            player_x=float(snapshot["x"]),
            player_y=float(snapshot["y"]),
            power=int(snapshot.get("current_power", 0)),
            bullet_count=int(snapshot.get("live_bullet_count", 0)),
            laser_count=int(snapshot.get("laser_count", 0)),
            hard_action_count=len(hard_names),
            exploration_rate=0.0,
            current_action=current_action,
            hard_admissible_actions=hard_names,
            phase_elapsed_frames=max(0, frame - context_start_frame),
        )
        physical_signature = (
            current_action,
            baseline_action,
            policy._action_mask(hard_names),
            policy._action_mask(legal),
            _clearance_signature(hard_actions),
        )
        frame_evidence[int(row["sequence"])] = (
            policy._context_key(context),
            policy._fine_context_key(context),
            published,
            physical_signature,
        )

    coarse = defaultdict(lambda: {
        "records": 0,
        "next_hard_empty": 0,
        "physical_signatures": Counter(),
    })
    fine = defaultdict(lambda: {
        "records": 0,
        "next_hard_empty": 0,
        "physical_signatures": Counter(),
    })
    eligible = 0
    for row in _rows(_stream_paths(run_dir, manifest, "transitions")):
        evidence = frame_evidence.get(int(row["sequence"]))
        outcome = row.get("outcome_terms", {})
        if (
            evidence is None
            or not row.get("learning_eligible")
            or int(outcome.get("elapsed_frames", 0)) != 1
        ):
            continue
        coarse_key, fine_key, action, signature = evidence
        if action != row.get("published_action"):
            continue
        next_hard = int(outcome.get("hard_count_after", -1))
        if next_hard < 0:
            continue
        eligible += 1
        for groups, context_key in ((coarse, coarse_key), (fine, fine_key)):
            item = groups[(context_key, action)]
            item["records"] += 1
            item["next_hard_empty"] += next_hard == 0
            item["physical_signatures"][signature] += 1

    return {
        "schema": "th06-rl-policy-alias-audit-v1",
        "run_id": manifest.get("run_id", run_dir.name),
        "stage_complete": manifest.get("stage_trajectory_complete"),
        "eligible_transitions": eligible,
        "next_hard_empty_transitions": sum(
            item["next_hard_empty"] for item in coarse.values()
        ),
        "coarse_ucb": _group_summary(coarse, minimum_support),
        "hierarchical_fine_counterfactual": _group_summary(
            fine,
            minimum_support,
        ),
        "interpretation": (
            "The fine result is a counterfactual partition of recorded behavior, "
            "not an off-policy reward estimate. It measures whether cheap current "
            "action, phase clock, baseline, and Hard/legal masks reduce state "
            "aliasing while the coarse statistics remain the hot-start backoff."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--minimum-support", type=int, default=2)
    args = parser.parse_args()
    if args.minimum_support < 1:
        parser.error("minimum support must be positive")
    print(json.dumps(
        analyze(args.run_dir, args.minimum_support),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
