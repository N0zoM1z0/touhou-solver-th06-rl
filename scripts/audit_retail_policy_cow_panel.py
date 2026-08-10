#!/usr/bin/env python3
"""Audit a fixed multi-episode policy-faithful Wine-anchored COW panel."""

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
except ModuleNotFoundError:  # Imported as scripts.audit_retail_policy_cow_panel.
    from scripts.audit_retail_first_action_scan import select_discovery_candidate
    from scripts.audit_retail_policy_continuation import _sha256
    from scripts.audit_retail_policy_cow import _validate_document


SCHEMA = "th06-rl-retail-policy-cow-panel-audit-v1"


def panel_gate(
    selections: Sequence[Mapping[str, Any]],
    *,
    minimum_unique_nonincumbent: int,
) -> dict[str, Any]:
    unique = [row for row in selections if len(row.get("robust_winners", ())) == 1]
    nonincumbent = [
        row for row in unique if row["robust_winners"][0] != row["incumbent_action"]
    ]
    passed = len(nonincumbent) >= minimum_unique_nonincumbent
    return {
        "episodes": len(selections),
        "unique_winner_episodes": len(unique),
        "unique_nonincumbent_winner_episodes": len(nonincumbent),
        "minimum_unique_nonincumbent_required": minimum_unique_nonincumbent,
        "targeted_action_relative_fit_allowed": passed,
        "candidate_population_cap": 3 if passed else 0,
        "active_candidates": 0,
    }


def parse_episode(raw: str) -> dict[str, Any]:
    fields = raw.split("::")
    if len(fields) != 5:
        raise ValueError(
            "episode must be PATH::RUN_ID::SEQUENCE::INCUMBENT::ACTION,ACTION"
        )
    path, run_id, sequence, incumbent, actions_raw = fields
    actions = actions_raw.split(",")
    if (
        not path
        or not run_id
        or not incumbent
        or not actions
        or incumbent not in actions
        or len(actions) != len(set(actions))
    ):
        raise ValueError("episode identity or action set is invalid")
    return {
        "path": Path(path),
        "run_id": run_id,
        "sequence": int(sequence),
        "incumbent_action": incumbent,
        "actions": actions,
    }


def audit_panel(
    episodes: Sequence[Mapping[str, Any]],
    *,
    expected_policy_state_sha256: str,
    expected_source_commit: str,
    expected_source_binary_sha256: str,
    expected_branch_frames: int,
    minimum_unique_nonincumbent: int,
) -> dict[str, Any]:
    if len(episodes) < minimum_unique_nonincumbent:
        raise ValueError("panel has fewer episodes than its support threshold")
    run_ids = [str(row["run_id"]) for row in episodes]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("panel run identities are not independent")
    selections = []
    documents = []
    for row in episodes:
        path = Path(row["path"]).resolve()
        incumbent = str(row["incumbent_action"])
        actions = tuple(str(action) for action in row["actions"])
        document, _checkpoint = _validate_document(
            path,
            expected_run_id=str(row["run_id"]),
            expected_sequence=int(row["sequence"]),
            expected_actions=actions,
            expected_policy_state_sha256=expected_policy_state_sha256,
            expected_source_commit=expected_source_commit,
            expected_source_binary_sha256=expected_source_binary_sha256,
            expected_branch_frames=expected_branch_frames,
        )
        selection = select_discovery_candidate(
            {"outcomes": document["outcomes"]},
            incumbent_action=incumbent,
            excluded_actions=(),
        )
        selections.append(
            {
                "run_id": str(row["run_id"]),
                "sequence": int(row["sequence"]),
                "incumbent_action": incumbent,
                **selection,
            }
        )
        documents.append(
            {
                "run_id": str(row["run_id"]),
                "sequence": int(row["sequence"]),
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    gate = panel_gate(
        selections,
        minimum_unique_nonincumbent=minimum_unique_nonincumbent,
    )
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "branch_frames": expected_branch_frames,
            "selection_rank": "robust-outcome-rank-v1",
            "episode_is_support_unit": True,
            "minimum_unique_nonincumbent_winners": minimum_unique_nonincumbent,
        },
        "source": {
            "commit": expected_source_commit,
            "binary_sha256": expected_source_binary_sha256,
        },
        "policy_state_sha256": expected_policy_state_sha256,
        "documents": documents,
        "episode_selections": selections,
        "gate": gate,
        "evidence_boundary": {
            "direct_policy_candidate": False,
            "promotion_authority": False,
            "fit_requires_separate_predeclared_protocol": True,
            "wine_shadow_required": True,
            "complete_wine_stage_hit_count_required": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        action="append",
        required=True,
        help="PATH::RUN_ID::SEQUENCE::INCUMBENT::ACTION,ACTION",
    )
    parser.add_argument("--expected-policy-state-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-binary-sha256", required=True)
    parser.add_argument("--expected-branch-frames", type=int, default=600)
    parser.add_argument("--minimum-unique-nonincumbent", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        episodes = [parse_episode(raw) for raw in args.episode]
        report = audit_panel(
            episodes,
            expected_policy_state_sha256=args.expected_policy_state_sha256,
            expected_source_commit=args.expected_source_commit,
            expected_source_binary_sha256=args.expected_source_binary_sha256,
            expected_branch_frames=args.expected_branch_frames,
            minimum_unique_nonincumbent=args.minimum_unique_nonincumbent,
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
                "fit_allowed": report["gate"]["targeted_action_relative_fit_allowed"],
                "unique_nonincumbent": report["gate"][
                    "unique_nonincumbent_winner_episodes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
