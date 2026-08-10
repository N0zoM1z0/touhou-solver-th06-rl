#!/usr/bin/env python3
"""Audit one predeclared Wine-anchored exhaustive first-action COW scan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from audit_targeted_headless_cow import robust_outcome_rank
    from export_wine_action_stream import _object
    from label_retail_replay_cow import (
        RETAIL_NATIVE_DELIVERY_DELAYS,
        SCHEMA as INPUT_SCHEMA,
    )
except ModuleNotFoundError:  # Imported as scripts.audit_retail_first_action_scan.
    from scripts.audit_targeted_headless_cow import robust_outcome_rank
    from scripts.export_wine_action_stream import _object
    from scripts.label_retail_replay_cow import (
        RETAIL_NATIVE_DELIVERY_DELAYS,
        SCHEMA as INPUT_SCHEMA,
    )

from th06_rl.headless_geometry import HEADLESS_DELIVERY_DELAYS


REPORT_SCHEMA = "th06-rl-retail-first-action-scan-audit-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_document(
    path: Path,
    *,
    expected_run_id: str,
    expected_sequence: int,
    expected_actions: Sequence[str],
    incumbent_action: str,
    expected_source_commit: str,
    expected_binary_sha256: str,
    expected_branch_frames: int,
    expected_teacher_horizon: int,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    path = path.resolve()
    document = _object(path)
    if document.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"unsupported COW document: {path}")
    if document.get("input_run_id") != expected_run_id:
        raise ValueError(f"COW run identity mismatch: {path}")
    if document.get("scope") != {
        "difficulty": 3,
        "character": 0,
        "shot_type": 0,
        "stage": 6,
    }:
        raise ValueError(f"COW scope mismatch: {path}")
    if document.get("branch_frames") != expected_branch_frames:
        raise ValueError(f"COW branch horizon mismatch: {path}")
    if document.get("teacher_horizon") != expected_teacher_horizon:
        raise ValueError(f"COW teacher horizon mismatch: {path}")
    if document.get("delivery_contracts") != {
        "retail_native_gate": list(RETAIL_NATIVE_DELIVERY_DELAYS),
        "source_step_branch": list(HEADLESS_DELIVERY_DELAYS),
    }:
        raise ValueError(f"COW delivery contract mismatch: {path}")
    source = document.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("clean") is not True
        or source.get("commit") != expected_source_commit
        or source.get("binary_sha256") != expected_binary_sha256
    ):
        raise ValueError(f"COW source identity mismatch: {path}")
    boundary = document.get("evidence_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("training_corpus") is not False
        or boundary.get("promotion_authority") is not False
        or boundary.get("native_gate_unchanged") is not True
        or boundary.get("bomb_forbidden") is not True
    ):
        raise ValueError(f"COW evidence boundary mismatch: {path}")

    run_directory = Path(str(document.get("input_run", ""))).resolve()
    manifest = run_directory / "manifest.json"
    run = run_directory / "run.json"
    if (
        not manifest.is_file()
        or not run.is_file()
        or _sha256(manifest) != document.get("input_manifest_sha256")
        or _sha256(run) != document.get("input_run_sha256")
    ):
        raise ValueError(f"COW retail input identity mismatch: {path}")

    actions = tuple(dict.fromkeys(str(action) for action in expected_actions))
    if not actions or len(actions) != len(expected_actions):
        raise ValueError("expected actions must be non-empty and unique")
    if tuple(document.get("requested_first_actions", ())) != actions:
        raise ValueError(f"COW requested action set mismatch: {path}")
    checkpoints = document.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1:
        raise ValueError(f"COW must contain exactly one checkpoint: {path}")
    checkpoint = checkpoints[0]
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"COW checkpoint is malformed: {path}")
    if (
        checkpoint.get("sequence") != expected_sequence
        or checkpoint.get("factual_action") != incumbent_action
        or checkpoint.get("retail_source_state_match_at_1e_6") is not True
        or checkpoint.get("retail_source_native_hard_set_match") is not True
        or checkpoint.get("retail_native_delivery_delays")
        != list(RETAIL_NATIVE_DELIVERY_DELAYS)
        or checkpoint.get("source_branch_delivery_delays")
        != list(HEADLESS_DELIVERY_DELAYS)
        or tuple(checkpoint.get("evaluated_first_actions", ())) != actions
    ):
        raise ValueError(f"COW checkpoint contract mismatch: {path}")
    outcomes = checkpoint.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(actions):
        raise ValueError(f"COW outcome count mismatch: {path}")
    outcome_actions = tuple(
        str(outcome.get("first_action"))
        for outcome in outcomes
        if isinstance(outcome, Mapping)
    )
    if outcome_actions != actions:
        raise ValueError(f"COW outcome action order mismatch: {path}")
    return document, checkpoint


def select_discovery_candidate(
    checkpoint: Mapping[str, Any],
    *,
    incumbent_action: str,
    excluded_actions: Sequence[str],
) -> dict[str, Any]:
    outcomes = checkpoint["outcomes"]
    ranked = [
        {
            "action": str(outcome["first_action"]),
            "robust_rank": list(robust_outcome_rank(outcome)),
            "outcome": outcome,
        }
        for outcome in outcomes
    ]
    best_rank = max(tuple(row["robust_rank"]) for row in ranked)
    winners = sorted(
        row["action"] for row in ranked if tuple(row["robust_rank"]) == best_rank
    )
    excluded = set(excluded_actions)
    candidate: str | None = None
    if len(winners) != 1:
        conclusion = "discovery-robust-tie-rejected"
    elif winners[0] == incumbent_action:
        conclusion = "discovery-incumbent-wins"
    elif winners[0] in excluded:
        conclusion = "discovery-previously-rejected-action-wins"
    else:
        candidate = winners[0]
        conclusion = "confirmation-required"
    return {
        "robust_winners": winners,
        "candidate": candidate,
        "conclusion": conclusion,
        "ranked_outcomes": ranked,
    }


def audit_scan(
    discovery_path: Path,
    *,
    expected_discovery_run_id: str,
    expected_discovery_sequence: int,
    expected_discovery_actions: Sequence[str],
    incumbent_action: str,
    excluded_actions: Sequence[str],
    expected_source_commit: str,
    expected_binary_sha256: str,
    expected_branch_frames: int = 600,
    expected_teacher_horizon: int = 12,
    confirmation_path: Path | None = None,
    expected_confirmation_run_id: str | None = None,
    expected_confirmation_sequence: int | None = None,
) -> dict[str, Any]:
    discovery_document, discovery_checkpoint = _validate_document(
        discovery_path,
        expected_run_id=expected_discovery_run_id,
        expected_sequence=expected_discovery_sequence,
        expected_actions=expected_discovery_actions,
        incumbent_action=incumbent_action,
        expected_source_commit=expected_source_commit,
        expected_binary_sha256=expected_binary_sha256,
        expected_branch_frames=expected_branch_frames,
        expected_teacher_horizon=expected_teacher_horizon,
    )
    discovery = select_discovery_candidate(
        discovery_checkpoint,
        incumbent_action=incumbent_action,
        excluded_actions=excluded_actions,
    )
    candidate = discovery["candidate"]
    confirmation: dict[str, Any] | None = None
    conclusion = str(discovery["conclusion"])
    hypothesis_candidates = 0
    documents = [{
        "role": "discovery",
        "path": str(discovery_path.resolve()),
        "sha256": _sha256(discovery_path.resolve()),
        "run_id": discovery_document["input_run_id"],
        "sequence": expected_discovery_sequence,
    }]

    if confirmation_path is not None:
        if candidate is None:
            raise ValueError("confirmation supplied without a discovery candidate")
        if expected_confirmation_run_id is None or expected_confirmation_sequence is None:
            raise ValueError("confirmation identity is required")
        confirmation_document, confirmation_checkpoint = _validate_document(
            confirmation_path,
            expected_run_id=expected_confirmation_run_id,
            expected_sequence=expected_confirmation_sequence,
            expected_actions=(incumbent_action, candidate),
            incumbent_action=incumbent_action,
            expected_source_commit=expected_source_commit,
            expected_binary_sha256=expected_binary_sha256,
            expected_branch_frames=expected_branch_frames,
            expected_teacher_horizon=expected_teacher_horizon,
        )
        confirmation_ranked = {
            str(outcome["first_action"]): robust_outcome_rank(outcome)
            for outcome in confirmation_checkpoint["outcomes"]
        }
        candidate_better = (
            confirmation_ranked[candidate] > confirmation_ranked[incumbent_action]
        )
        confirmation = {
            "candidate": candidate,
            "candidate_robust_rank": list(confirmation_ranked[candidate]),
            "incumbent_robust_rank": list(confirmation_ranked[incumbent_action]),
            "candidate_strictly_better": candidate_better,
        }
        if candidate_better:
            conclusion = "confirmed-headless-hypothesis-only"
            hypothesis_candidates = 1
        else:
            conclusion = "confirmation-rejected"
        documents.append({
            "role": "confirmation",
            "path": str(confirmation_path.resolve()),
            "sha256": _sha256(confirmation_path.resolve()),
            "run_id": confirmation_document["input_run_id"],
            "sequence": expected_confirmation_sequence,
        })

    return {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "incumbent_action": incumbent_action,
            "excluded_actions": list(dict.fromkeys(excluded_actions)),
            "branch_frames": expected_branch_frames,
            "teacher_horizon": expected_teacher_horizon,
            "selection_rank": "robust-outcome-rank-v1",
            "unique_discovery_winner_required": True,
            "strict_confirmation_win_required": True,
        },
        "source": {
            "commit": expected_source_commit,
            "binary_sha256": expected_binary_sha256,
        },
        "documents": documents,
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
    parser.add_argument("--excluded-action", action="append", default=[])
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--expected-branch-frames", type=int, default=600)
    parser.add_argument("--expected-teacher-horizon", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = audit_scan(
            args.discovery,
            expected_discovery_run_id=args.expected_discovery_run_id,
            expected_discovery_sequence=args.expected_discovery_sequence,
            expected_discovery_actions=args.expected_discovery_action,
            incumbent_action=args.incumbent_action,
            excluded_actions=args.excluded_action,
            expected_source_commit=args.expected_source_commit,
            expected_binary_sha256=args.expected_binary_sha256,
            expected_branch_frames=args.expected_branch_frames,
            expected_teacher_horizon=args.expected_teacher_horizon,
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
