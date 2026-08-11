#!/usr/bin/env python3
"""Fit one episode-grouped autonomous Q residual from factual Wine corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.autonomous_learning import fit_grouped_ridge, load_episode
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--exploration-probability", type=float, default=0.10)
    parser.add_argument("--return-horizon", type=int, default=120)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--propensity-clip", type=float, default=20.0)
    parser.add_argument("--minimum-train-groups", type=int, default=3)
    parser.add_argument("--minimum-validation-groups", type=int, default=2)
    parser.add_argument("--minimum-train-rows", type=int, default=1000)
    parser.add_argument("--minimum-non-baseline-rows", type=int, default=64)
    parser.add_argument("--minimum-action-samples", type=int, default=16)
    parser.add_argument("--minimum-action-ess", type=float, default=8.0)
    parser.add_argument("--required-rmse-ratio", type=float, default=0.995)
    parser.add_argument("--margin-rmse-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace output: {args.output_dir}")
    runs = sorted({path.resolve() for path in args.runs})
    if args.validation_episodes <= 0 or len(runs) <= args.validation_episodes:
        parser.error("need training episodes plus at least one validation episode")
    loaded = [
        load_episode(
            run,
            exploration_probability=args.exploration_probability,
            return_horizon=args.return_horizon,
            gamma=args.gamma,
            observation_names=OBSERVATION_FEATURE_NAMES,
            action_names=ACTION_FEATURE_NAMES,
        )
        for run in runs
    ]
    split = len(runs) - args.validation_episodes
    train = [sample for samples, _report in loaded[:split] for sample in samples]
    validation = [
        sample for samples, _report in loaded[split:] for sample in samples
    ]
    state = fit_grouped_ridge(
        train,
        validation,
        observation_names=OBSERVATION_FEATURE_NAMES,
        action_names=ACTION_FEATURE_NAMES,
        alpha=args.ridge_alpha,
        propensity_clip=args.propensity_clip,
        minimum_train_groups=args.minimum_train_groups,
        minimum_validation_groups=args.minimum_validation_groups,
        minimum_train_rows=args.minimum_train_rows,
        minimum_non_baseline_rows=args.minimum_non_baseline_rows,
        minimum_action_samples=args.minimum_action_samples,
        minimum_action_ess=args.minimum_action_ess,
        required_rmse_ratio=args.required_rmse_ratio,
        margin_rmse_fraction=args.margin_rmse_fraction,
    )
    args.output_dir.mkdir(parents=True)
    _write(args.output_dir / "policy-shadow.json", state)
    report = {
        "schema": "autonomous-wine-q-fit-report-v1",
        "algorithm": "grouped-propensity-weighted-ridge-monte-carlo",
        "inputs": [report for _samples, report in loaded],
        "split": {
            "kind": "whole-episode-chronological-holdout",
            "training": [report["episode_id"] for _samples, report in loaded[:split]],
            "validation": [report["episode_id"] for _samples, report in loaded[split:]],
        },
        "generation_parameters": {
            "exploration_probability": args.exploration_probability,
            "return_horizon": args.return_horizon,
            "gamma": args.gamma,
            "ridge_alpha": args.ridge_alpha,
            "propensity_clip": args.propensity_clip,
        },
        "fit": state["fit_report"],
        "authorization": state["authorization"],
        "support": state["support"],
    }
    _write(args.output_dir / "report.json", report)
    print(json.dumps({
        "output": str(args.output_dir),
        "fit_eligible": state["authorization"]["fit_eligible"],
        "train_rows": len(train),
        "validation_rows": len(validation),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
