#!/usr/bin/env python3
"""Fit Generation 3 cross-fitted option-advantage population from Wine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.advantage_learning import (  # noqa: E402
    CROSSFIT_FOLDS,
    NUISANCE_MEMBERS,
    NUISANCE_TREES,
    POPULATION_MEMBERS,
    POPULATION_TREES,
    fit_dr_option_advantage,
    load_option_episode,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-scorer", required=True, type=Path)
    parser.add_argument("--validation-episodes", type=int, default=3)
    parser.add_argument("--exploration-probability", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=260812)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args(argv)
    if args.validation_episodes != 3:
        parser.error("Generation 3 requires three held-out episodes")
    if not 0.0 < args.exploration_probability <= 1.0:
        parser.error("exploration probability must be in (0, 1]")
    if args.seed != 260812:
        parser.error("Generation 3 learner seed is fixed at 260812")
    if args.threads <= 0:
        parser.error("thread count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace output: {args.output_dir}")
    if not args.native_scorer.is_file():
        parser.error("native scorer library is absent")
    runs = []
    seen = set()
    for path in args.runs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            runs.append(resolved)
    if len(runs) < 12:
        parser.error("Generation 3 first fit requires 12 complete Wine Stages")
    loaded = [
        load_option_episode(
            run,
            exploration_probability=args.exploration_probability,
        )
        for run in runs
    ]
    split = len(loaded) - args.validation_episodes
    train = [sample for samples, _report in loaded[:split] for sample in samples]
    validation = [sample for samples, _report in loaded[split:] for sample in samples]
    state = fit_dr_option_advantage(
        train,
        validation,
        crossfit_folds=CROSSFIT_FOLDS,
        nuisance_members=NUISANCE_MEMBERS,
        population_members=POPULATION_MEMBERS,
        nuisance_trees=NUISANCE_TREES,
        population_trees=POPULATION_TREES,
        seed=args.seed,
        threads=args.threads,
        native_scorer_sha256=_sha256(args.native_scorer),
    )
    args.output_dir.mkdir(parents=True)
    _write(args.output_dir / "policy-shadow.json", state)
    _write(args.output_dir / "report.json", {
        "schema": "autonomous-wine-generation-3-fit-report-v1",
        "algorithm": "cross-fitted-multi-action-aipw-option-advantage",
        "inputs": [report for _samples, report in loaded],
        "split": {
            "kind": "whole-complete-physical-stage-holdout",
            "training": [report["episode_id"] for _samples, report in loaded[:split]],
            "validation": [report["episode_id"] for _samples, report in loaded[split:]],
        },
        "parameters": {
            "exploration_probability": args.exploration_probability,
            "crossfit_folds": CROSSFIT_FOLDS,
            "nuisance_members": NUISANCE_MEMBERS,
            "population_members": POPULATION_MEMBERS,
            "nuisance_trees": NUISANCE_TREES,
            "population_trees": POPULATION_TREES,
            "seed": args.seed,
        },
        "fit": state["fit_report"],
        "authorization": state["authorization"],
    })
    print(json.dumps({
        "output": str(args.output_dir),
        "fit_eligible": state["authorization"]["fit_eligible"],
        "train_options": len(train),
        "validation_options": len(validation),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
