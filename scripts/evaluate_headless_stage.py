#!/usr/bin/env python3
"""Audit and classify same-scope headless rollouts for one TH06 stage."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from audit_headless_corpus import audit_run
except ModuleNotFoundError:  # Imported as scripts.evaluate_headless_stage in tests.
    from scripts.audit_headless_corpus import audit_run


def _run_directories(paths: Iterable[Path]) -> tuple[Path, ...]:
    result = []
    for path in paths:
        if (path / "manifest.json").is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(item.parent for item in path.rglob("manifest.json")))
    return tuple(dict.fromkeys(item.resolve() for item in result))


def classify_stage_runs(
    runs: list[Mapping[str, Any]],
    *,
    required_seeds: int,
    minimum_ticks: int,
) -> dict[str, Any]:
    if required_seeds < 1 or minimum_ticks < 1:
        raise ValueError("stage evaluation bounds must be positive")
    if not runs:
        raise ValueError("no stage rollouts supplied")
    scopes = {json.dumps(run.get("scope"), sort_keys=True) for run in runs}
    if len(scopes) != 1:
        raise ValueError("headless stage evaluation refuses to mix scopes")
    stages = {int(run["scope"]["stage"]) for run in runs}
    if len(stages) != 1:
        raise ValueError("headless stage evaluation refuses to mix stages")
    seeds = [int(run["initial_seed"]) for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("headless stage evaluation requires unique seeds")

    enough_seeds = len(seeds) >= required_seeds
    all_valid = all(run.get("valid") is True for run in runs)
    safety_complete = all(
        run.get("termination_reason") not in {"physical-hit", "authority-failure"}
        and run.get("physical_hit") is not True
        and run.get("authority_failure") in {None, ""}
        and int(run.get("benchmark_forced_rows", 0)) == 0
        for run in runs
    )
    bounded_survival = (
        enough_seeds
        and all_valid
        and safety_complete
        and all(int(run.get("rows", 0)) >= minimum_ticks - 1 for run in runs)
    )
    nmnb_stage_clear = (
        enough_seeds
        and all_valid
        and safety_complete
        and all(run.get("nmnb_stage_clear") is True for run in runs)
    )
    if nmnb_stage_clear:
        status = "headless-nmnb-stage-clear-candidate"
    elif bounded_survival:
        status = "bounded-headless-survival-candidate"
    else:
        status = "rejected"
    return {
        "scope": runs[0]["scope"],
        "stage": next(iter(stages)),
        "seeds": sorted(seeds),
        "required_seeds": required_seeds,
        "minimum_ticks": minimum_ticks,
        "runs": len(runs),
        "all_runs_valid": all_valid,
        "safety_complete": safety_complete,
        "minimum_observed_ticks": min(int(run.get("rows", 0)) + 1 for run in runs),
        "terminations": dict(sorted(Counter(
            str(run.get("termination_reason")) for run in runs
        ).items())),
        "bounded_survival_qualified": bounded_survival,
        "headless_nmnb_stage_clear_qualified": nmnb_stage_clear,
        "headless_status": status,
        "windows_promotion_allowed": False,
        "windows_promotion_blocker": (
            "headless evidence accelerates learning but does not replace physical Windows validation"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--required-seeds", type=int, default=2)
    parser.add_argument("--minimum-ticks", type=int, default=3000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directories = _run_directories(args.paths)
    if not directories:
        parser.error("no compact headless corpus manifests found")
    runs = []
    source_commits = set()
    ranker_hashes = set()
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        audited = audit_run(directory)
        runs.append({
            **audited,
            "authority_failure": manifest.get("authority_failure"),
            "physical_hit": manifest.get("physical_hit"),
            "nmnb_stage_clear": manifest.get("nmnb_stage_clear"),
        })
        source_commits.add(manifest.get("source", {}).get("commit"))
        ranker_hashes.add(manifest.get("ranker", {}).get("sha256"))
    if len(source_commits) != 1:
        parser.error("headless stage evaluation refuses mixed source revisions")
    if len(ranker_hashes) != 1:
        parser.error("headless stage evaluation refuses mixed ranker artifacts")
    try:
        result = classify_stage_runs(
            runs,
            required_seeds=args.required_seeds,
            minimum_ticks=args.minimum_ticks,
        )
    except ValueError as error:
        parser.error(str(error))
    result["source_commit"] = next(iter(source_commits))
    result["ranker_sha256"] = next(iter(ranker_hashes))
    result["run_results"] = runs
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["headless_status"] != "rejected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
