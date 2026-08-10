#!/usr/bin/env python3
"""Audit targeted retail-replay COW without promoting headless winners."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from audit_targeted_headless_cow import compare_actions
    from export_wine_action_stream import _object, _verified_stream_rows
    from label_retail_replay_cow import (
        RETAIL_NATIVE_DELIVERY_DELAYS,
        SCHEMA as INPUT_SCHEMA,
    )
except ModuleNotFoundError:  # Imported as scripts.audit_retail_replay_cow.
    from scripts.audit_targeted_headless_cow import compare_actions
    from scripts.export_wine_action_stream import _object, _verified_stream_rows
    from scripts.label_retail_replay_cow import (
        RETAIL_NATIVE_DELIVERY_DELAYS,
        SCHEMA as INPUT_SCHEMA,
    )

from th06_rl.headless_geometry import HEADLESS_DELIVERY_DELAYS


REPORT_SCHEMA = "th06-rl-retail-replay-cow-audit-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_pair_results(
    results: Sequence[str],
    *,
    minimum_independent_prefixes: int = 2,
) -> dict[str, Any]:
    counts = Counter(results)
    left_gate = (
        len(results) >= minimum_independent_prefixes
        and counts["left-better"] == len(results)
    )
    return {
        "independent_prefixes": len(results),
        "minimum_independent_prefixes": minimum_independent_prefixes,
        "comparisons": dict(sorted(counts.items())),
        "left_alternative_unanimous": left_gate,
        "residual_candidates": 1 if left_gate else 0,
        "conclusion": (
            "left-alternative-headless-hypothesis-only"
            if left_gate
            else "left-alternative-rejected"
        ),
    }


def audit_retail_replay_cow(
    paths: Sequence[Path],
    *,
    left_action: str,
    right_action: str,
    expected_source_commit: str,
    expected_binary_sha256: str,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one retail replay COW document is required")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    documents: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        document = _object(path)
        if document.get("schema") != INPUT_SCHEMA:
            raise ValueError(f"unsupported retail replay COW document: {path}")
        if document.get("delivery_contracts") != {
            "retail_native_gate": list(RETAIL_NATIVE_DELIVERY_DELAYS),
            "source_step_branch": list(HEADLESS_DELIVERY_DELAYS),
        }:
            raise ValueError(f"retail replay COW delivery contract mismatch: {path}")
        source = document.get("source")
        boundary = document.get("evidence_boundary")
        if (
            not isinstance(source, Mapping)
            or source.get("commit") != expected_source_commit
            or source.get("clean") is not True
            or source.get("binary_sha256") != expected_binary_sha256
        ):
            raise ValueError(f"retail replay COW source mismatch: {path}")
        if (
            not isinstance(boundary, Mapping)
            or boundary.get("training_corpus") is not False
            or boundary.get("promotion_authority") is not False
            or boundary.get("native_gate_unchanged") is not True
            or boundary.get("bomb_forbidden") is not True
        ):
            raise ValueError(f"retail replay COW evidence boundary mismatch: {path}")
        run_directory = Path(str(document.get("input_run", ""))).resolve()
        manifest_path = run_directory / "manifest.json"
        run_path = run_directory / "run.json"
        if (
            not manifest_path.is_file()
            or not run_path.is_file()
            or _sha256(manifest_path) != document.get("input_manifest_sha256")
            or _sha256(run_path) != document.get("input_run_sha256")
        ):
            raise ValueError(f"retail replay COW input identity mismatch: {path}")
        manifest = _object(manifest_path)
        frames, frame_evidence = _verified_stream_rows(
            run_directory, manifest, "frames"
        )
        checkpoints = document.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError(f"retail replay COW has no checkpoints: {path}")
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping):
                raise TypeError("retail replay COW checkpoint is not an object")
            sequence = int(checkpoint.get("sequence", -1))
            if not 0 <= sequence < len(frames):
                raise ValueError("retail replay COW checkpoint sequence is invalid")
            key = (str(document.get("input_run_id", "")), sequence)
            if not key[0] or key in seen:
                raise ValueError("duplicate retail replay COW checkpoint")
            seen.add(key)
            if (
                checkpoint.get("retail_source_state_match_at_1e_6") is not True
                or checkpoint.get("retail_source_native_hard_set_match") is not True
                or checkpoint.get("retail_native_delivery_delays")
                != list(RETAIL_NATIVE_DELIVERY_DELAYS)
                or checkpoint.get("source_branch_delivery_delays")
                != list(HEADLESS_DELIVERY_DELAYS)
                or checkpoint.get("factual_action") != right_action
                or checkpoint.get("local_teacher_action") != left_action
                or set(checkpoint.get("evaluated_first_actions", ()))
                != {left_action, right_action}
            ):
                raise ValueError("retail replay COW checkpoint contract mismatch")
            comparison = compare_actions(checkpoint, left_action, right_action)
            if comparison is None:
                raise ValueError("retail replay COW lacks one targeted pair outcome")
            decision = frames[sequence].get("decision")
            if not isinstance(decision, Mapping):
                raise TypeError("retail replay COW source decision is absent")
            records.append(
                {
                    "run_id": key[0],
                    "sequence": sequence,
                    "checkpoint_tick": checkpoint.get("checkpoint_tick"),
                    "policy_id": decision.get("policy_id"),
                    "comparison": comparison,
                    "best_actions": checkpoint.get("best_actions"),
                    "outcomes": checkpoint.get("outcomes"),
                    "document": str(path),
                    "document_sha256": _sha256(path),
                }
            )
        documents.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "run_id": document.get("input_run_id"),
                "frame_shards": frame_evidence,
            }
        )
    summary = summarize_pair_results(
        [record["comparison"] for record in records]
    )
    return {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pair": {"left_alternative": left_action, "incumbent_action": right_action},
        "source": {
            "commit": expected_source_commit,
            "binary_sha256": expected_binary_sha256,
        },
        "delivery_contracts": {
            "retail_native_gate": list(RETAIL_NATIVE_DELIVERY_DELAYS),
            "source_step_branch": list(HEADLESS_DELIVERY_DELAYS),
        },
        "policy_support": dict(sorted(Counter(
            str(record["policy_id"]) for record in records
        ).items())),
        "records": records,
        "documents": documents,
        "gate": summary,
        "evidence_boundary": {
            "promotion_authority": False,
            "training_corpus": False,
            "wine_shadow_required": True,
            "headless_winner_cannot_activate": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--left-action", required=True)
    parser.add_argument("--right-action", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = audit_retail_replay_cow(
            args.paths,
            left_action=args.left_action,
            right_action=args.right_action,
            expected_source_commit=args.expected_source_commit,
            expected_binary_sha256=args.expected_binary_sha256,
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
        "residual_candidates": report["gate"]["residual_candidates"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
