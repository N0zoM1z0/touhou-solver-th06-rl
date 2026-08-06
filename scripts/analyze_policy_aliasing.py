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
    failure_support = sorted(
        item["records"]
        for item in groups.values()
        for _ in range(item["next_hard_empty"])
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
        "next_hard_empty_group_support": {
            "events": len(failure_support),
            "singleton_events": sum(value == 1 for value in failure_support),
            "p50_records": (
                failure_support[len(failure_support) // 2]
                if failure_support else None
            ),
            "max_records": max(failure_support, default=None),
        },
    }


def _collect_groups(run_dir: Path):
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
            policy._middle_context_key(context),
            policy._fine_context_key(context),
            published,
            physical_signature,
        )

    coarse = defaultdict(lambda: {
        "records": 0,
        "next_hard_empty": 0,
        "physical_signatures": Counter(),
    })
    middle = defaultdict(lambda: {
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
        coarse_key, middle_key, fine_key, action, signature = evidence
        if action != row.get("published_action"):
            continue
        next_hard = int(outcome.get("hard_count_after", -1))
        if next_hard < 0:
            continue
        eligible += 1
        for groups, context_key in (
            (coarse, coarse_key),
            (middle, middle_key),
            (fine, fine_key),
        ):
            item = groups[(context_key, action)]
            item["records"] += 1
            item["next_hard_empty"] += next_hard == 0
            item["physical_signatures"][signature] += 1

    return manifest, eligible, {
        "coarse_ucb": coarse,
        "hierarchical_middle_counterfactual": middle,
        "hierarchical_fine_counterfactual": fine,
    }


def _prior_support(current, prior) -> dict[str, object]:
    total = sum(item["records"] for item in current.values())
    failures = sum(item["next_hard_empty"] for item in current.values())
    shared = set(current).intersection(prior)
    shared_records = sum(current[key]["records"] for key in shared)
    shared_failures = sum(
        current[key]["next_hard_empty"] for key in shared
    )
    union = set(current).union(prior)
    return {
        "prior_context_actions": len(prior),
        "current_context_actions": len(current),
        "shared_context_actions": len(shared),
        "key_jaccard": len(shared) / len(union) if union else None,
        "current_record_prior_support_rate": (
            shared_records / total if total else None
        ),
        "current_next_hard_empty_prior_support_rate": (
            shared_failures / failures if failures else None
        ),
    }


def analyze(
    run_dir: Path,
    minimum_support: int,
    prior_run_dirs: tuple[Path, ...] = (),
) -> dict[str, object]:
    manifest, eligible, groups = _collect_groups(run_dir)
    result = {
        "schema": "th06-rl-policy-alias-audit-v2",
        "run_id": manifest.get("run_id", run_dir.name),
        "stage_complete": manifest.get("stage_trajectory_complete"),
        "eligible_transitions": eligible,
        "next_hard_empty_transitions": sum(
            item["next_hard_empty"]
            for item in groups["coarse_ucb"].values()
        ),
        **{
            name: _group_summary(value, minimum_support)
            for name, value in groups.items()
        },
        "interpretation": (
            "The middle and fine results are counterfactual partitions of "
            "recorded behavior, not off-policy reward estimates. They measure "
            "the alias/reuse tradeoff of the phase-clock/control backoff and "
            "exact Hard/legal frontier while coarse statistics remain the "
            "broadest hot-start backoff."
        ),
    }
    if prior_run_dirs:
        prior_run_ids = []
        prior_keys = {name: set() for name in groups}
        for prior_run_dir in prior_run_dirs:
            prior_manifest, _prior_eligible, prior_groups = _collect_groups(
                prior_run_dir
            )
            prior_run_ids.append(prior_manifest.get(
                "run_id",
                prior_run_dir.name,
            ))
            for name, values in prior_groups.items():
                prior_keys[name].update(values)
        result["prior_run_support"] = {
            "prior_run_ids": prior_run_ids,
            **{
                name: _prior_support(value, prior_keys[name])
                for name, value in groups.items()
            },
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--prior-run-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "measure reuse from a prior run; repeat to match the union of "
            "all runs loaded by the policy"
        ),
    )
    parser.add_argument("--minimum-support", type=int, default=2)
    args = parser.parse_args()
    if args.minimum_support < 1:
        parser.error("minimum support must be positive")
    print(json.dumps(
        analyze(
            args.run_dir,
            args.minimum_support,
            tuple(args.prior_run_dir),
        ),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
