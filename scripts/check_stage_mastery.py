#!/usr/bin/env python3
"""Report whether the latest trustworthy runs are consecutive no-HIT clears."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DIFFICULTIES = {
    "easy": 0,
    "normal": 1,
    "hard": 2,
    "lunatic": 3,
}


def trustworthy_result(run_dir: Path, difficulty: int, stage: int) -> int | None:
    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        audit = json.loads(
            (run_dir / "infra-audit.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    metadata = run.get("metadata", {})
    if (
        metadata.get("difficulty") != difficulty
        or metadata.get("character") != 0
        or metadata.get("shot_type") != 0
        or metadata.get("stage") != stage
    ):
        return None

    outcome = manifest.get("run_outcome", {})
    if not (
        manifest.get("complete") is True
        and manifest.get("stage_trajectory_complete") is True
        and outcome.get("stage_completed") is True
        and outcome.get("controller_exit_code") == 0
        and outcome.get("capture_failures") == 0
        and outcome.get("infrastructure_failures") == 0
        and outcome.get("corpus_failures") == 0
        and outcome.get("policy_state_committed") is True
        and audit.get("stage_completed") is True
        and audit.get("infra_stable_for_learning") is True
        and not audit.get("integrity_errors")
        and not audit.get("scope_pollution")
    ):
        return None
    hits = outcome.get("physical_hits")
    return hits if isinstance(hits, int) and hits >= 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--difficulty", choices=DIFFICULTIES, required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--consecutive-clears", type=int, default=3)
    args = parser.parse_args()
    if args.consecutive_clears <= 0:
        parser.error("--consecutive-clears must be positive")

    results: list[tuple[str, int]] = []
    for manifest_path in sorted(args.corpus_root.glob("*/manifest.json")):
        hits = trustworthy_result(
            manifest_path.parent,
            DIFFICULTIES[args.difficulty],
            args.stage,
        )
        if hits is not None:
            results.append((manifest_path.parent.name, hits))
    recent = results[-args.consecutive_clears :]
    mastered = (
        len(recent) == args.consecutive_clears
        and all(hits == 0 for _run_id, hits in recent)
    )
    print(json.dumps({
        "difficulty": args.difficulty,
        "stage": args.stage,
        "required_consecutive_clears": args.consecutive_clears,
        "trustworthy_complete_runs": len(results),
        "recent": [
            {"run_id": run_id, "physical_hits": hits}
            for run_id, hits in recent
        ],
        "mastered": mastered,
    }, separators=(",", ":")))
    return 0 if mastered else 1


if __name__ == "__main__":
    raise SystemExit(main())
