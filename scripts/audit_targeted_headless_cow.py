#!/usr/bin/env python3
"""Audit Wine-targeted headless COW labels without treating rows as episodes.

The three regions in this module are hypotheses admitted by the immutable
Stage 6 Wine failure audit.  They are offline selection/audit keys only; none
of them is an online movement rule.  Every retained checkpoint must still
branch its complete native-safe first-action set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REPORT_SCHEMA = "th06-rl-wine-targeted-headless-cow-audit-v1"
WINE_AUDIT_SCHEMA = "th06-rl-wine-episode-failure-region-audit-v1"
COW_SCHEMA = "th06-rl-headless-cow-counterfactual-v1"

FAMILY_CONTRACTS = {
    "sub10-dense-boundary-broad": {
        "source_context": "boss:0/10",
        "physical_region": "boundary/dense-bullets/broad-5-plus",
        "wine_hypothesis": "right_fast versus left_fast",
    },
    "sub31-interior-lasers-broad": {
        "source_context": "boss:0/31",
        "physical_region": "interior/lasers-present/broad-5-plus",
        "wine_hypothesis": "exhaustive native-safe first action",
    },
    "sub18-medium-boundary-broad": {
        "source_context": "boss:0/18",
        "physical_region": "boundary/medium-bullets/broad-5-plus",
        "wine_hypothesis": "exhaustive native-safe first action",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def targeted_family(row: Mapping[str, Any]) -> str | None:
    """Return the admitted generic family for a compact headless row."""
    if row.get("benchmark_forced_action") is True:
        return None
    legal = row.get("legal_actions")
    state = row.get("state")
    if not isinstance(legal, list) or len(legal) < 5 or not isinstance(state, dict):
        return None
    context = str(row.get("source_context", ""))
    reserve = float(state["boundary_reserve"])
    bullets = int(state["bullet_count"])
    lasers = int(state["laser_count"])
    if (
        context == "boss:0/10"
        and reserve <= 16.0
        and bullets >= 384
        and lasers == 0
    ):
        return "sub10-dense-boundary-broad"
    if context == "boss:0/31" and reserve > 16.0 and lasers > 0:
        return "sub31-interior-lasers-broad"
    if (
        context == "boss:0/18"
        and reserve <= 16.0
        and 128 <= bullets < 384
        and lasers == 0
    ):
        return "sub18-medium-boundary-broad"
    return None


def robust_outcome_rank(outcome: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    """Rank outcome tiers while discarding pixel-level accidental argmaxes."""
    terminal = str(outcome["termination_reason"])
    no_death = int(outcome.get("physical_deaths_delta", 0)) == 0
    completed = no_death and terminal in {
        "tick-limit",
        "chain-exit-success",
        "stage-clear-success",
    }
    width = max(int(outcome["minimum_native_legal_actions"]), 0)
    width_bucket = min(width.bit_length() - 1, 4) if width else 0
    reserve = max(float(outcome["terminal_boundary_reserve"]), 0.0)
    reserve_bucket = sum(
        reserve >= threshold for threshold in (1e-6, 8.0, 16.0, 32.0, 64.0)
    )
    return (
        int(completed),
        int(no_death and terminal != "physical-hit"),
        int(outcome["survival_ticks"]),
        width_bucket if completed else 0,
        reserve_bucket if completed else 0,
    )


def compare_actions(
    checkpoint: Mapping[str, Any], left: str, right: str
) -> str | None:
    outcomes = {
        str(outcome["first_action"]): outcome
        for outcome in checkpoint["outcomes"]
    }
    if left not in outcomes or right not in outcomes:
        return None
    left_rank = robust_outcome_rank(outcomes[left])
    right_rank = robust_outcome_rank(outcomes[right])
    if left_rank > right_rank:
        return "left-better"
    if left_rank < right_rank:
        return "right-better"
    return "tie"


def _paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    result = []
    for path in paths:
        if path.is_file():
            result.append(path.resolve())
        elif path.is_dir():
            result.extend(item.resolve() for item in path.rglob("*.json"))
    return tuple(sorted(set(result)))


def _load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _object(path / "manifest.json")
    if manifest.get("transaction_complete") is not True:
        raise ValueError(f"headless input is not transaction complete: {path}")
    rows = []
    with gzip.open(path / "transitions.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows.extend(json.loads(line) for line in stream)
    if len(rows) != int(manifest["transition_count"]):
        raise ValueError(f"headless transition count mismatch: {path}")
    return manifest, rows


def _split(seed: int) -> str:
    # Declared before candidate construction.  A complete seed stays in one split.
    return "development-odd-seeds" if seed % 2 else "confirmation-even-seeds"


def _seed_comparison(values: Iterable[str]) -> str:
    observed = set(values)
    if "left-better" in observed and "right-better" in observed:
        return "mixed"
    if "left-better" in observed:
        return "left-better"
    if "right-better" in observed:
        return "right-better"
    return "tie"


def audit(
    *,
    wine_audit_path: Path,
    cow_paths: Iterable[Path],
    expected_branch_frames: int,
    runtime_binary_sha256: str,
) -> dict[str, Any]:
    wine_audit = _object(wine_audit_path)
    if wine_audit.get("schema") != WINE_AUDIT_SCHEMA:
        raise ValueError("targeted COW requires the episode-grouped Wine audit")
    if not str(wine_audit.get("next_gate", "")).startswith(
        "targeted multi-seed headless COW"
    ):
        raise ValueError("Wine audit does not admit targeted COW")

    run_cache: dict[Path, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    checkpoints: dict[tuple[Path, int], dict[str, Any]] = {}
    evidence_files: dict[tuple[Path, int], set[str]] = defaultdict(set)
    for cow_path in _paths(cow_paths):
        document = _object(cow_path)
        if document.get("schema") != COW_SCHEMA:
            continue
        if int(document["branch_frames"]) != expected_branch_frames:
            continue
        runtime = document.get("runtime_source", {})
        if runtime.get("binary_sha256") != runtime_binary_sha256:
            raise ValueError(f"COW runtime binary mismatch: {cow_path}")
        if document.get("runtime_delivery_contract") != "synchronous-step-v1":
            raise ValueError(f"unsupported COW delivery contract: {cow_path}")
        if document.get("runtime_delivery_delays") != [0]:
            raise ValueError(f"unsupported COW delivery delays: {cow_path}")
        run = Path(str(document["input_run"])).resolve()
        if run not in run_cache:
            run_cache[run] = _load_run(run)
        manifest, rows = run_cache[run]
        if document.get("scope") != manifest.get("scope"):
            raise ValueError(f"COW/input scope mismatch: {cow_path}")
        if document.get("input_source") != manifest.get("source"):
            raise ValueError(f"COW/input source mismatch: {cow_path}")
        for checkpoint in document["checkpoints"]:
            sequence = int(checkpoint["sequence"])
            row = rows[sequence]
            if checkpoint["observation_sha256"] != row["observation_sha256"]:
                raise ValueError(f"COW observation mismatch: {cow_path}")
            if checkpoint["source_context"] != row["source_context"]:
                raise ValueError(f"COW source context mismatch: {cow_path}")
            if checkpoint["native_legal_actions"] != row["legal_actions"]:
                raise ValueError(f"COW native legal set mismatch: {cow_path}")
            family = targeted_family(row)
            if family is None:
                continue
            key = (run, sequence)
            record = {
                "family": family,
                "seed": int(manifest["initial_seed"]),
                "split": _split(int(manifest["initial_seed"])),
                "run": str(run),
                "sequence": sequence,
                "tick": int(row["tick"]),
                "state": {
                    "boundary_reserve": float(row["state"]["boundary_reserve"]),
                    "bullet_count": int(row["state"]["bullet_count"]),
                    "laser_count": int(row["state"]["laser_count"]),
                    "native_legal_action_count": len(row["legal_actions"]),
                },
                "checkpoint": checkpoint,
            }
            if key in checkpoints and checkpoints[key] != record:
                raise ValueError(f"conflicting duplicate COW checkpoint: {key}")
            checkpoints[key] = record
            evidence_files[key].add(str(cow_path))

    family_rows = []
    for family, contract in FAMILY_CONTRACTS.items():
        records = sorted(
            (record for record in checkpoints.values() if record["family"] == family),
            key=lambda record: (record["seed"], record["sequence"]),
        )
        seeds = sorted({int(record["seed"]) for record in records})
        split_seeds = {
            split: sorted({
                int(record["seed"])
                for record in records
                if record["split"] == split
            })
            for split in ("development-odd-seeds", "confirmation-even-seeds")
        }
        comparisons: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            checkpoint = record["checkpoint"]
            factual = str(checkpoint["factual_action"])
            for action in checkpoint["native_legal_actions"]:
                result = compare_actions(checkpoint, str(action), factual)
                if result is not None:
                    comparisons[str(action)].append({
                        "seed": record["seed"],
                        "result": result,
                    })
        action_summary = {}
        for action, values in sorted(comparisons.items()):
            seed_values: dict[int, list[str]] = defaultdict(list)
            for value in values:
                seed_values[int(value["seed"])].append(str(value["result"]))
            action_summary[action] = {
                "checkpoint_comparisons": dict(sorted(Counter(
                    str(value["result"]) for value in values
                ).items())),
                "seed_comparisons": dict(sorted(Counter(
                    _seed_comparison(items) for items in seed_values.values()
                ).items())),
                "seeds": {
                    str(seed): _seed_comparison(items)
                    for seed, items in sorted(seed_values.items())
                },
            }
        specific = None
        if family == "sub10-dense-boundary-broad":
            pair_records = [
                record for record in records
                if record["checkpoint"]["factual_action"] == "right_fast"
                and "left_fast" in record["checkpoint"]["native_legal_actions"]
            ]
            by_seed: dict[int, list[str]] = defaultdict(list)
            for record in pair_records:
                result = compare_actions(
                    record["checkpoint"], "left_fast", "right_fast"
                )
                if result is not None:
                    by_seed[int(record["seed"])].append(result)
            specific = {
                "comparison": "left_fast versus factual right_fast",
                "checkpoints": len(pair_records),
                "seeds": {
                    str(seed): _seed_comparison(values)
                    for seed, values in sorted(by_seed.items())
                },
                "seed_comparisons": dict(sorted(Counter(
                    _seed_comparison(values) for values in by_seed.values()
                ).items())),
            }
        family_rows.append({
            "family": family,
            "contract": contract,
            "checkpoints": len(records),
            "seeds": seeds,
            "checkpoints_per_seed": dict(sorted(Counter(
                int(record["seed"]) for record in records
            ).items())),
            "split_seeds": split_seeds,
            "candidate_construction_support_gate": (
                len(split_seeds["development-odd-seeds"]) >= 3
                and len(split_seeds["confirmation-even-seeds"]) >= 2
            ),
            "specific_hypothesis": specific,
            "pairwise_actions_vs_factual": action_summary,
        })

    return {
        "schema": REPORT_SCHEMA,
        "authority": (
            "hypothesis evidence only; original-retail Wine remains promotion authority"
        ),
        "wine_audit": {
            "path": str(wine_audit_path.resolve()),
            "sha256": _sha256(wine_audit_path),
        },
        "runtime_binary_sha256": runtime_binary_sha256,
        "branch_frames": expected_branch_frames,
        "split_contract": {
            "development": "complete odd-numbered seeds",
            "confirmation": "complete even-numbered seeds",
            "no_checkpoint_crosses_splits": True,
        },
        "unique_targeted_checkpoints": len(checkpoints),
        "evidence_files": len({path for paths in evidence_files.values() for path in paths}),
        "families": family_rows,
        "next_gate": (
            "construct at most three low-activation residual hypotheses only from "
            "families passing the seed support gate; disagreement returns to incumbent"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wine-audit", type=Path, required=True)
    parser.add_argument("--cow-root", type=Path, action="append", required=True)
    parser.add_argument("--branch-frames", type=int, default=600)
    parser.add_argument("--runtime-binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.branch_frames <= 0:
        parser.error("branch frames must be positive")
    result = audit(
        wine_audit_path=args.wine_audit.resolve(),
        cow_paths=args.cow_root,
        expected_branch_frames=args.branch_frames,
        runtime_binary_sha256=args.runtime_binary_sha256,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
