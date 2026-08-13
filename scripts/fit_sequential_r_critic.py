#!/usr/bin/env python3
"""Fit the Generation-4 sequential action-centered population from Wine."""

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
    BEHAVIOR_POLICY as GENERATION3_POLICY,
    TRANSITION_SCHEMA as GENERATION3_TRANSITION,
    _object,
    load_option_episode,
)
from th06_rl.sequential_learning import (  # noqa: E402
    BEHAVIOR_POLICY as GENERATION4_POLICY,
    CRITIC_TREES,
    NUISANCE_TREES,
    TRANSITION_SCHEMA as GENERATION4_TRANSITION,
    fit_sequential_r_critic,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load(run_dir: Path):
    run = _object(run_dir / "run.json")
    schemas = run.get("schemas")
    transition = schemas.get("transition") if isinstance(schemas, dict) else None
    if transition == GENERATION3_TRANSITION:
        return load_option_episode(
            run_dir,
            exploration_probability=0.10,
            behavior_policy=GENERATION3_POLICY,
            transition_schema=GENERATION3_TRANSITION,
        )
    if transition == GENERATION4_TRANSITION:
        return load_option_episode(
            run_dir,
            exploration_probability=None,
            behavior_policy=GENERATION4_POLICY,
            transition_schema=GENERATION4_TRANSITION,
        )
    raise ValueError(f"unsupported sequential Wine corpus schema: {transition}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-scorer", required=True, type=Path)
    parser.add_argument("--compatible-native-scorer", action="append", type=Path)
    parser.add_argument("--seed", type=int, default=260812)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args(argv)
    if args.seed != 260812:
        parser.error("Generation 4 learner seed is fixed at 260812")
    if args.threads <= 0:
        parser.error("thread count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace output: {args.output_dir}")
    scorer_paths = [args.native_scorer, *(args.compatible_native_scorer or ())]
    if any(not path.is_file() for path in scorer_paths):
        parser.error("native scorer library is absent")
    runs = []
    seen = set()
    for raw in args.runs:
        path = raw.resolve()
        if path not in seen:
            seen.add(path)
            runs.append(path)
    if len(runs) < 10:
        parser.error("Generation 4 sequential fit needs ten complete episodes")
    loaded = [_load(run) for run in runs]
    samples = [sample for rows, _report in loaded for sample in rows]
    state = fit_sequential_r_critic(
        samples,
        nuisance_trees=NUISANCE_TREES,
        critic_trees=CRITIC_TREES,
        seed=args.seed,
        threads=args.threads,
        native_scorer_sha256=_sha256(args.native_scorer),
        compatible_native_scorer_sha256=tuple(
            _sha256(path) for path in args.compatible_native_scorer or ()
        ),
    )
    args.output_dir.mkdir(parents=True)
    _write(args.output_dir / "policy-shadow.json", state)
    _write(args.output_dir / "report.json", {
        "schema": "autonomous-wine-generation-4-fit-report-v1",
        "algorithm": "cross-fitted-n-step-generalized-r-option-critic",
        "inputs": [report for _rows, report in loaded],
        "parameters": {
            "nuisance_trees": NUISANCE_TREES,
            "critic_trees": CRITIC_TREES,
            "seed": args.seed,
            "native_scorer_sha256": _sha256(args.native_scorer),
            "compatible_native_scorer_sha256": [
                _sha256(path) for path in args.compatible_native_scorer or ()
            ],
        },
        "fit": state["fit_report"],
        "authorization": state["authorization"],
    })
    print(json.dumps({
        "output": str(args.output_dir),
        "episode_groups": len(runs),
        "options": len(samples),
        "fit_eligible": state["authorization"]["fit_eligible"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
