#!/usr/bin/env python3
"""Apply the predeclared robust gate to policy-faithful retail COW branches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from audit_retail_first_action_scan import select_discovery_candidate
    from audit_retail_policy_continuation import _sha256
    from export_wine_action_stream import _object
    from label_retail_policy_cow import SCHEMA as INPUT_SCHEMA
except ModuleNotFoundError:  # Imported as scripts.audit_retail_policy_cow.
    from scripts.audit_retail_first_action_scan import select_discovery_candidate
    from scripts.audit_retail_policy_continuation import _sha256
    from scripts.export_wine_action_stream import _object
    from scripts.label_retail_policy_cow import SCHEMA as INPUT_SCHEMA


REPORT_SCHEMA = "th06-rl-retail-policy-cow-audit-v1"


def _validate_document(
    path: Path,
    *,
    expected_run_id: str,
    expected_sequence: int,
    expected_actions: Sequence[str],
    expected_policy_state_sha256: str,
    expected_source_commit: str,
    expected_source_binary_sha256: str,
    expected_branch_frames: int,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    path = path.resolve()
    document = _object(path)
    if document.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"unsupported policy COW document: {path}")
    input_row = document.get("input")
    source = document.get("source")
    policy = document.get("policy")
    checkpoint = document.get("checkpoint")
    boundary = document.get("evidence_boundary")
    if not isinstance(input_row, Mapping) or input_row.get("run_id") != expected_run_id:
        raise ValueError(f"policy COW run identity mismatch: {path}")
    if (
        not isinstance(source, Mapping)
        or source.get("clean") is not True
        or source.get("commit") != expected_source_commit
        or source.get("binary_sha256") != expected_source_binary_sha256
    ):
        raise ValueError(f"policy COW source mismatch: {path}")
    if (
        not isinstance(policy, Mapping)
        or policy.get("state_sha256") != expected_policy_state_sha256
        or policy.get("immutable_observe_suppressed") is not True
    ):
        raise ValueError(f"policy COW frozen state mismatch: {path}")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("training_corpus") is not False
        or boundary.get("promotion_authority") is not False
        or boundary.get("native_gate_unchanged") is not True
        or boundary.get("bomb_forbidden") is not True
    ):
        raise ValueError(f"policy COW evidence boundary mismatch: {path}")
    actions = tuple(str(action) for action in expected_actions)
    if (
        document.get("branch_frames") != expected_branch_frames
        or tuple(document.get("requested_first_actions", ())) != actions
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("sequence") != expected_sequence
        or checkpoint.get("retail_source_state_match_at_1e_6") is not True
        or checkpoint.get("restored_policy_action_match") is not True
        or document.get("factual_regression", {}).get("passed") is not True
    ):
        raise ValueError(f"policy COW checkpoint contract mismatch: {path}")
    outcomes = document.get("outcomes")
    if not isinstance(outcomes, list) or tuple(
        str(outcome.get("first_action"))
        for outcome in outcomes
        if isinstance(outcome, Mapping)
    ) != actions:
        raise ValueError(f"policy COW outcome order mismatch: {path}")
    return document, checkpoint


def audit_policy_cow(
    discovery_path: Path,
    *,
    expected_discovery_run_id: str,
    expected_discovery_sequence: int,
    expected_discovery_actions: Sequence[str],
    incumbent_action: str,
    expected_policy_state_sha256: str,
    expected_source_commit: str,
    expected_source_binary_sha256: str,
    expected_branch_frames: int = 600,
    confirmation_path: Path | None = None,
    expected_confirmation_run_id: str | None = None,
    expected_confirmation_sequence: int | None = None,
) -> dict[str, Any]:
    discovery_document, _checkpoint = _validate_document(
        discovery_path,
        expected_run_id=expected_discovery_run_id,
        expected_sequence=expected_discovery_sequence,
        expected_actions=expected_discovery_actions,
        expected_policy_state_sha256=expected_policy_state_sha256,
        expected_source_commit=expected_source_commit,
        expected_source_binary_sha256=expected_source_binary_sha256,
        expected_branch_frames=expected_branch_frames,
    )
    discovery = select_discovery_candidate(
        {"outcomes": discovery_document["outcomes"]},
        incumbent_action=incumbent_action,
        excluded_actions=(),
    )
    candidate = discovery["candidate"]
    conclusion = str(discovery["conclusion"])
    confirmation = None
    hypothesis_candidates = 0
    documents = [{
        "role": "discovery",
        "path": str(discovery_path.resolve()),
        "sha256": _sha256(discovery_path.resolve()),
    }]
    if confirmation_path is not None:
        if candidate is None:
            raise ValueError("confirmation supplied without discovery candidate")
        if expected_confirmation_run_id is None or expected_confirmation_sequence is None:
            raise ValueError("confirmation identity is required")
        confirmation_document, _checkpoint = _validate_document(
            confirmation_path,
            expected_run_id=expected_confirmation_run_id,
            expected_sequence=expected_confirmation_sequence,
            expected_actions=(incumbent_action, candidate),
            expected_policy_state_sha256=expected_policy_state_sha256,
            expected_source_commit=expected_source_commit,
            expected_source_binary_sha256=expected_source_binary_sha256,
            expected_branch_frames=expected_branch_frames,
        )
        confirmation_selection = select_discovery_candidate(
            {"outcomes": confirmation_document["outcomes"]},
            incumbent_action=incumbent_action,
            excluded_actions=(),
        )
        candidate_wins = confirmation_selection["robust_winners"] == [candidate]
        confirmation = {
            "candidate": candidate,
            "robust_winners": confirmation_selection["robust_winners"],
            "candidate_strictly_better": candidate_wins,
            "ranked_outcomes": confirmation_selection["ranked_outcomes"],
        }
        if candidate_wins:
            conclusion = "confirmed-policy-cow-hypothesis-only"
            hypothesis_candidates = 1
        else:
            conclusion = "confirmation-rejected"
        documents.append({
            "role": "confirmation",
            "path": str(confirmation_path.resolve()),
            "sha256": _sha256(confirmation_path.resolve()),
        })
    return {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
        "protocol": {
            "incumbent_action": incumbent_action,
            "branch_frames": expected_branch_frames,
            "selection_rank": "robust-outcome-rank-v1",
            "unique_discovery_winner_required": True,
            "strict_confirmation_win_required": True,
        },
        "source": {
            "commit": expected_source_commit,
            "binary_sha256": expected_source_binary_sha256,
        },
        "policy_state_sha256": expected_policy_state_sha256,
        "discovery": discovery,
        "confirmation": confirmation,
        "gate": {
            "conclusion": conclusion,
            "candidate": candidate if hypothesis_candidates else None,
            "headless_hypothesis_candidates": hypothesis_candidates,
            "active_candidates": 0,
        },
        "evidence_boundary": {
            "training_corpus": False,
            "promotion_authority": False,
            "wine_shadow_required": True,
            "complete_wine_stage_hit_count_required": True,
            "headless_winner_cannot_activate": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("discovery", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--expected-discovery-run-id", required=True)
    parser.add_argument("--expected-discovery-sequence", type=int, required=True)
    parser.add_argument("--expected-discovery-action", action="append", required=True)
    parser.add_argument("--expected-confirmation-run-id")
    parser.add_argument("--expected-confirmation-sequence", type=int)
    parser.add_argument("--incumbent-action", required=True)
    parser.add_argument("--expected-policy-state-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-binary-sha256", required=True)
    parser.add_argument("--expected-branch-frames", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = audit_policy_cow(
            args.discovery,
            expected_discovery_run_id=args.expected_discovery_run_id,
            expected_discovery_sequence=args.expected_discovery_sequence,
            expected_discovery_actions=args.expected_discovery_action,
            incumbent_action=args.incumbent_action,
            expected_policy_state_sha256=args.expected_policy_state_sha256,
            expected_source_commit=args.expected_source_commit,
            expected_source_binary_sha256=args.expected_source_binary_sha256,
            expected_branch_frames=args.expected_branch_frames,
            confirmation_path=args.confirmation,
            expected_confirmation_run_id=args.expected_confirmation_run_id,
            expected_confirmation_sequence=args.expected_confirmation_sequence,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": report["schema"],
        "conclusion": report["gate"]["conclusion"],
        "candidate": report["gate"]["candidate"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
