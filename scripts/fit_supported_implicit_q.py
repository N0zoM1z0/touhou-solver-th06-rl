#!/usr/bin/env python3
"""Fit the Generation-5 supported implicit-Q population from factual Wine."""

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
from th06_rl.implicit_learning import (  # noqa: E402
    BELLMAN_ITERATIONS,
    CALIBRATION_Q_TREES,
    CALIBRATION_VALUE_TREES,
    N_STEP_OPTIONS,
    Q_TREES,
    VALUE_TREES,
    fit_supported_implicit_q,
)
from th06_rl.option_cache import load_cached_option_episode  # noqa: E402
from th06_rl.sequential_learning import (  # noqa: E402
    BEHAVIOR_POLICY as GENERATION4_POLICY,
    TRANSITION_SCHEMA as GENERATION4_TRANSITION,
)


OPTION_CACHE_CONTRACT = (
    REPOSITORY / "src/th06_rl/advantage_learning.py",
    REPOSITORY / "src/th06_rl/autonomous_learning.py",
    REPOSITORY / "src/th06_rl/corpus.py",
    Path(__file__).resolve(),
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
    raise ValueError(f"unsupported implicit-Q Wine corpus schema: {transition}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--new-run", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-scorer", required=True, type=Path)
    parser.add_argument("--compatible-native-scorer", action="append", type=Path)
    parser.add_argument("--seed", type=int, default=260_813)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPOSITORY / "artifacts/cache/audited-option-episodes",
    )
    args = parser.parse_args(argv)
    if args.seed != 260_813:
        parser.error("Generation 5 learner seed is fixed at 260813")
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
    new_runs = {path.resolve() for path in args.new_run}
    if not new_runs <= set(runs):
        parser.error("every --new-run must also be present in positional runs")
    if len(runs) < 10:
        parser.error("Generation 5 fit needs ten complete episodes")
    if len(new_runs) < 8:
        parser.error("Generation 5 authorization fit needs eight --new-run episodes")
    cached = [load_cached_option_episode(
        run,
        loader=_load,
        cache_root=args.cache_dir,
        contract_files=OPTION_CACHE_CONTRACT,
    ) for run in runs]
    loaded = [(rows, report) for rows, report, _hit in cached]
    samples = [sample for rows, _report in loaded for sample in rows]
    run_to_episode = {
        run: rows[0].episode_id for run, (rows, _report) in zip(
            runs, loaded, strict=True
        )
    }
    state = fit_supported_implicit_q(
        samples,
        new_episode_ids=frozenset(run_to_episode[run] for run in new_runs),
        iterations=BELLMAN_ITERATIONS,
        n_step_options=N_STEP_OPTIONS,
        q_trees=Q_TREES,
        value_trees=VALUE_TREES,
        calibration_q_trees=CALIBRATION_Q_TREES,
        calibration_value_trees=CALIBRATION_VALUE_TREES,
        seed=args.seed,
        total_threads=args.threads,
        native_scorer_sha256=_sha256(args.native_scorer),
        compatible_native_scorer_sha256=tuple(
            _sha256(path) for path in args.compatible_native_scorer or ()
        ),
    )
    args.output_dir.mkdir(parents=True)
    _write(args.output_dir / "policy-shadow.json", state)
    _write(args.output_dir / "report.json", {
        "schema": "autonomous-wine-generation-5-fit-report-v1",
        "algorithm": "action-centered-in-sample-implicit-fitted-q",
        "inputs": [report for _rows, report in loaded],
        "parameters": {
            "bellman_iterations": BELLMAN_ITERATIONS,
            "n_step_options": N_STEP_OPTIONS,
            "q_trees": Q_TREES,
            "value_trees": VALUE_TREES,
            "calibration_q_trees": CALIBRATION_Q_TREES,
            "calibration_value_trees": CALIBRATION_VALUE_TREES,
            "seed": args.seed,
            "total_threads": args.threads,
            "option_cache_hits": sum(
                hit for _rows, _report, hit in cached
            ),
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
        "new_episode_groups": len(new_runs),
        "options": len(samples),
        "fit_eligible": state["authorization"]["fit_eligible"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
