#!/usr/bin/env python3
"""Gate two disjoint policy-faithful COWs for the fixed neutral family."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from audit_retail_first_action_scan import select_discovery_candidate
    from audit_retail_policy_continuation import _sha256
    from audit_retail_policy_cow import _validate_document
    from export_wine_action_stream import _object
    from select_retail_neutral_anchors import (
        INCUMBENT_ACTION,
        NEUTRAL_ACTIONS,
        SCHEMA as ANCHOR_SCHEMA,
    )
except ModuleNotFoundError:  # Imported as scripts.audit_retail_neutral_policy_cow.
    from scripts.audit_retail_first_action_scan import select_discovery_candidate
    from scripts.audit_retail_policy_continuation import _sha256
    from scripts.audit_retail_policy_cow import _validate_document
    from scripts.export_wine_action_stream import _object
    from scripts.select_retail_neutral_anchors import (
        INCUMBENT_ACTION,
        NEUTRAL_ACTIONS,
        SCHEMA as ANCHOR_SCHEMA,
    )


SCHEMA = "th06-rl-retail-neutral-policy-cow-audit-v1"


def unanimous_neutral_candidate(
    selections: Sequence[Mapping[str, Any]],
) -> str | None:
    if len(selections) != 2:
        return None
    winners = [selection.get("robust_winners") for selection in selections]
    if not all(isinstance(value, list) and len(value) == 1 for value in winners):
        return None
    winner = str(winners[0][0])
    if winner not in NEUTRAL_ACTIONS or any(value != [winner] for value in winners):
        return None
    return winner


def audit_panel(
    anchor_report_path: Path,
    cow_paths: Sequence[Path],
    *,
    expected_policy_state_sha256: str,
    expected_source_commit: str,
    expected_source_binary_sha256: str,
    expected_branch_frames: int = 600,
) -> dict[str, Any]:
    if len(cow_paths) != 2:
        raise ValueError("neutral COW gate requires exactly two documents")
    anchor_report_path = anchor_report_path.resolve()
    anchors = _object(anchor_report_path)
    episodes = anchors.get("episodes")
    if (
        anchors.get("schema") != ANCHOR_SCHEMA
        or anchors.get("gate", {}).get("passed") is not True
        or not isinstance(episodes, list)
        or len(episodes) != 2
    ):
        raise ValueError("neutral anchor report did not pass")
    by_run = {}
    for episode in episodes:
        if not isinstance(episode, Mapping) or not isinstance(
            episode.get("anchor"), Mapping
        ):
            raise TypeError("neutral anchor row is malformed")
        by_run[str(episode["run_id"])] = episode
    if len(by_run) != 2:
        raise ValueError("neutral anchor run identities are not unique")

    actions = (INCUMBENT_ACTION, *NEUTRAL_ACTIONS)
    selections = []
    documents = []
    seen = set()
    for raw_path in cow_paths:
        path = raw_path.resolve()
        raw = _object(path)
        input_row = raw.get("input")
        if not isinstance(input_row, Mapping):
            raise TypeError("neutral COW input row is malformed")
        run_id = str(input_row.get("run_id", ""))
        episode = by_run.get(run_id)
        if episode is None or run_id in seen:
            raise ValueError("neutral COW run is absent or duplicated")
        seen.add(run_id)
        sequence = int(episode["anchor"]["sequence"])
        document, _checkpoint = _validate_document(
            path,
            expected_run_id=run_id,
            expected_sequence=sequence,
            expected_actions=actions,
            expected_policy_state_sha256=expected_policy_state_sha256,
            expected_source_commit=expected_source_commit,
            expected_source_binary_sha256=expected_source_binary_sha256,
            expected_branch_frames=expected_branch_frames,
        )
        selection = select_discovery_candidate(
            {"outcomes": document["outcomes"]},
            incumbent_action=INCUMBENT_ACTION,
            excluded_actions=(),
        )
        selections.append({"run_id": run_id, **selection})
        documents.append(
            {
                "run_id": run_id,
                "sequence": sequence,
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    if seen != set(by_run):
        raise ValueError("neutral COW documents do not cover both anchors")
    selections.sort(key=lambda row: str(row["run_id"]))
    documents.sort(key=lambda row: str(row["run_id"]))
    candidate = unanimous_neutral_candidate(selections)
    conclusion = (
        "unanimous-neutral-headless-hypothesis-only"
        if candidate is not None
        else "neutral-family-rejected"
    )
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anchor_report": {
            "path": str(anchor_report_path),
            "sha256": _sha256(anchor_report_path),
        },
        "protocol": {
            "actions": list(actions),
            "branch_frames": expected_branch_frames,
            "selection_rank": "robust-outcome-rank-v1",
            "same_unique_neutral_winner_required_in_both_episodes": True,
        },
        "source": {
            "commit": expected_source_commit,
            "binary_sha256": expected_source_binary_sha256,
        },
        "policy_state_sha256": expected_policy_state_sha256,
        "documents": documents,
        "episode_selections": selections,
        "gate": {
            "conclusion": conclusion,
            "candidate": candidate,
            "headless_hypothesis_candidates": int(candidate is not None),
            "active_candidates": 0,
        },
        "evidence_boundary": {
            "training_corpus": False,
            "promotion_authority": False,
            "still_new_wine_shadow_required": True,
            "complete_wine_stage_hit_count_required": True,
            "headless_winner_cannot_activate": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-report", required=True, type=Path)
    parser.add_argument("cows", nargs=2, type=Path)
    parser.add_argument("--expected-policy-state-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-binary-sha256", required=True)
    parser.add_argument("--expected-branch-frames", type=int, default=600)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = audit_panel(
            args.anchor_report,
            args.cows,
            expected_policy_state_sha256=args.expected_policy_state_sha256,
            expected_source_commit=args.expected_source_commit,
            expected_source_binary_sha256=args.expected_source_binary_sha256,
            expected_branch_frames=args.expected_branch_frames,
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
                "conclusion": report["gate"]["conclusion"],
                "candidate": report["gate"]["candidate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
