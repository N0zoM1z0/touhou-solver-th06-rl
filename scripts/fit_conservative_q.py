#!/usr/bin/env python3
"""Fit generation-2 conservative fitted-Q trees from complete Wine Stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.conservative_learning import (  # noqa: E402
    fit_conservative_fqi,
    load_complete_episode,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-scorer", type=Path, required=True)
    parser.add_argument("--shadow-native-scorer", type=Path)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--exploration-probability", type=float, default=0.10)
    parser.add_argument("--n-step-frames", type=int, default=60)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--bellman-iterations", type=int, default=6)
    parser.add_argument("--trees-per-iteration", type=int, default=96)
    parser.add_argument("--propensity-clip", type=float, default=20.0)
    parser.add_argument("--prototypes-per-action", type=int, default=12)
    parser.add_argument("--support-quantile", type=float, default=0.99)
    parser.add_argument("--uncertainty-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260811)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args(argv)
    integers = (
        args.validation_episodes,
        args.n_step_frames,
        args.ensemble_members,
        args.bellman_iterations,
        args.trees_per_iteration,
        args.prototypes_per_action,
        args.threads,
    )
    if min(integers) <= 0 or args.ensemble_members < 3:
        parser.error("positive bounds and at least three ensemble members required")
    if not 0.0 < args.exploration_probability <= 1.0:
        parser.error("exploration probability must be in (0, 1]")
    if not 0.5 <= args.support_quantile < 1.0:
        parser.error("support quantile must be in [0.5, 1)")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace output: {args.output_dir}")
    if not args.native_scorer.is_file():
        parser.error("native scorer library is absent")
    if (
        args.shadow_native_scorer is not None
        and not args.shadow_native_scorer.is_file()
    ):
        parser.error("shadow native scorer library is absent")
    runs = sorted({path.resolve() for path in args.runs})
    if len(runs) < args.validation_episodes + 3:
        parser.error("need three training and two held-out physical episodes")
    loaded = [
        load_complete_episode(
            run,
            exploration_probability=args.exploration_probability,
            n_step_frames=args.n_step_frames,
        )
        for run in runs
    ]
    split = len(loaded) - args.validation_episodes
    train = [sample for samples, _report in loaded[:split] for sample in samples]
    validation = [
        sample for samples, _report in loaded[split:] for sample in samples
    ]
    state = fit_conservative_fqi(
        train,
        validation,
        ensemble_members=args.ensemble_members,
        bellman_iterations=args.bellman_iterations,
        trees_per_iteration=args.trees_per_iteration,
        propensity_clip=args.propensity_clip,
        prototypes_per_action=args.prototypes_per_action,
        support_quantile=args.support_quantile,
        uncertainty_scale=args.uncertainty_scale,
        seed=args.seed,
        threads=args.threads,
        native_scorer_sha256=_sha256(args.native_scorer),
        compatible_native_scorer_sha256=(
            (_sha256(args.shadow_native_scorer),)
            if args.shadow_native_scorer is not None else ()
        ),
    )
    args.output_dir.mkdir(parents=True)
    _write(args.output_dir / "policy-shadow.json", state)
    report = {
        "schema": "autonomous-wine-conservative-fit-report-v2",
        "algorithm": "episode-bootstrap-conservative-n-step-fitted-q",
        "inputs": [report for _samples, report in loaded],
        "split": {
            "kind": "whole-complete-physical-stage-holdout",
            "training": [report["episode_id"] for _samples, report in loaded[:split]],
            "validation": [report["episode_id"] for _samples, report in loaded[split:]],
        },
        "parameters": {
            "exploration_probability": args.exploration_probability,
            "n_step_frames": args.n_step_frames,
            "ensemble_members": args.ensemble_members,
            "bellman_iterations": args.bellman_iterations,
            "trees_per_iteration": args.trees_per_iteration,
            "propensity_clip": args.propensity_clip,
            "prototypes_per_action": args.prototypes_per_action,
            "support_quantile": args.support_quantile,
            "uncertainty_scale": args.uncertainty_scale,
            "seed": args.seed,
        },
        "fit": state["fit_report"],
        "authorization": state["authorization"],
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
