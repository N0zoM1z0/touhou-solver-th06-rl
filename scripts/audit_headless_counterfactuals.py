#!/usr/bin/env python3
"""Audit dynamic COW counterfactual labels and summarize useful yield."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from label_headless_cow_counterfactuals import outcome_rank
except ModuleNotFoundError:
    from scripts.label_headless_cow_counterfactuals import outcome_rank


SCHEMA = "th06-rl-headless-cow-counterfactual-v1"
AUTHORITY = "first-action-native-legal-and-dynamic-continuation-revalidated"
TERMINATIONS = {
    "authority-failure",
    "physical-hit",
    "tick-limit",
    "chain-exit-success",
    "stage-clear-success",
    "chain-exit-error",
}


def _files(paths: Iterable[Path]) -> tuple[Path, ...]:
    result = []
    for path in paths:
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(path.rglob("*.json")))
    return tuple(dict.fromkeys(item.resolve() for item in result))


def audit_file(path: Path) -> dict[str, Any]:
    errors = []
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        errors.append("unsupported schema")
    if document.get("authority") != AUTHORITY:
        errors.append("invalid authority contract")
    if document.get("runtime_source", {}).get("clean") is not True:
        errors.append("counterfactual runtime source was dirty")
    delivery_contract = document.get(
        "runtime_delivery_contract",
        "legacy-unspecified-v0",
    )
    delivery_delays = document.get("runtime_delivery_delays")
    if delivery_contract == "synchronous-step-v1":
        if delivery_delays != [0]:
            errors.append("synchronous COW delivery must be exactly [0]")
    elif delivery_contract != "legacy-unspecified-v0":
        errors.append("unsupported COW delivery contract")
    branch_frames = int(document.get("branch_frames", 0))
    checkpoints = document.get("checkpoints")
    if branch_frames <= 0 or not isinstance(checkpoints, list) or not checkpoints:
        errors.append("missing checkpoint or branch bound")
        checkpoints = []
    outcomes_count = 0
    unique_best = 0
    local_best = 0
    factual_best = 0
    terminations: Counter[str] = Counter()
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        prefix = f"checkpoint {checkpoint_index}"
        legal = checkpoint.get("native_legal_actions")
        outcomes = checkpoint.get("outcomes")
        if not isinstance(legal, list) or not legal or "bomb" in legal or len(legal) != len(set(legal)):
            errors.append(f"{prefix}: invalid native legal set")
            continue
        if not isinstance(outcomes, list):
            errors.append(f"{prefix}: outcomes missing")
            continue
        actions = [outcome.get("first_action") for outcome in outcomes]
        if len(outcomes) != len(legal) or set(actions) != set(legal):
            errors.append(f"{prefix}: outcomes do not cover native legal set")
            continue
        outcomes_count += len(outcomes)
        for outcome in outcomes:
            terminal = str(outcome.get("termination_reason"))
            terminations[terminal] += 1
            if terminal not in TERMINATIONS:
                errors.append(f"{prefix}: invalid termination {terminal}")
            survival = int(outcome.get("survival_ticks", -1))
            issued = int(outcome.get("actions_issued", -1))
            if not 0 <= survival <= branch_frames or not 0 <= issued <= branch_frames:
                errors.append(f"{prefix}: outcome exceeds branch bound")
        expected_rank = max(outcome_rank(outcome) for outcome in outcomes)
        expected_best = sorted(
            str(outcome["first_action"])
            for outcome in outcomes
            if outcome_rank(outcome) == expected_rank
        )
        if checkpoint.get("best_actions") != expected_best:
            errors.append(f"{prefix}: best action summary mismatch")
        unique_best += len(expected_best) == 1
        expected_local = checkpoint.get("local_teacher_action") in expected_best
        expected_factual = checkpoint.get("factual_action") in expected_best
        if checkpoint.get("local_teacher_action_is_best") != expected_local:
            errors.append(f"{prefix}: local teacher flag mismatch")
        if checkpoint.get("factual_action_is_best") != expected_factual:
            errors.append(f"{prefix}: factual action flag mismatch")
        local_best += expected_local
        factual_best += expected_factual
    return {
        "path": str(path),
        "valid": not errors,
        "errors": errors,
        "scope": document.get("scope"),
        "initial_seed": document.get("initial_seed"),
        "runtime_delivery_contract": delivery_contract,
        "runtime_delivery_delays": delivery_delays,
        "checkpoints": len(checkpoints),
        "outcomes": outcomes_count,
        "unique_best_checkpoints": unique_best,
        "local_teacher_best_checkpoints": local_best,
        "factual_best_checkpoints": factual_best,
        "terminations": dict(sorted(terminations.items())),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        scope = result.get("scope")
        if isinstance(scope, dict) and isinstance(scope.get("stage"), int):
            by_stage[scope["stage"]].append(result)

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        checkpoints = sum(int(row["checkpoints"]) for row in rows)
        outcomes = sum(int(row["outcomes"]) for row in rows)
        unique = sum(int(row["unique_best_checkpoints"]) for row in rows)
        local = sum(int(row["local_teacher_best_checkpoints"]) for row in rows)
        factual = sum(int(row["factual_best_checkpoints"]) for row in rows)
        return {
            "files": len(rows),
            "valid_files": sum(row["valid"] is True for row in rows),
            "checkpoints": checkpoints,
            "outcomes": outcomes,
            "unique_best_checkpoints": unique,
            "unique_best_ratio": unique / checkpoints if checkpoints else 0.0,
            "local_teacher_best_ratio": local / checkpoints if checkpoints else 0.0,
            "factual_action_best_ratio": factual / checkpoints if checkpoints else 0.0,
        }

    delivery_contracts = Counter(
        str(result.get("runtime_delivery_contract")) for result in results
    )
    return {
        "schema": "th06-rl-headless-cow-counterfactual-audit-v1",
        **aggregate(results),
        "runtime_delivery_contracts": dict(sorted(delivery_contracts.items())),
        "mixed_runtime_delivery_contracts": len(delivery_contracts) > 1,
        "by_stage": [
            {"stage": stage, **aggregate(rows)}
            for stage, rows in sorted(by_stage.items())
        ],
        "file_results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = _files(args.paths)
    if not files:
        parser.error("no counterfactual JSON files found")
    result = summarize([audit_file(path) for path in files])
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if (
        result["valid_files"] == result["files"]
        and not result["mixed_runtime_delivery_contracts"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
