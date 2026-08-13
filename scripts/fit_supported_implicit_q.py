#!/usr/bin/env python3
"""Fit the Generation-5 supported implicit-Q population from factual Wine."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.audited_option_loader import (  # noqa: E402
    AUDITED_OPTION_LOADER_CONTRACT,
    load_audited_option_episode,
)
from th06_rl.implicit_learning import (  # noqa: E402
    BELLMAN_ITERATIONS,
    CALIBRATION_Q_TREES,
    CALIBRATION_VALUE_TREES,
    CROSSFIT_FOLDS,
    DEFAULT_CROSSFIT_WORKERS,
    MAX_TRAINING_THREADS,
    N_STEP_OPTIONS,
    Q_TREES,
    VALUE_TREES,
    fit_supported_implicit_q,
)
from th06_rl.option_cache import (  # noqa: E402
    load_cached_option_episode,
    prime_option_episode_cache,
)
OPTION_CACHE_CONTRACT = AUDITED_OPTION_LOADER_CONTRACT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


_load = load_audited_option_episode


def _prime_cache(arguments: tuple[Path, Path]) -> tuple[bool, int]:
    run, cache_root = arguments
    return prime_option_episode_cache(
        run,
        loader=_load,
        cache_root=cache_root,
        contract_files=OPTION_CACHE_CONTRACT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--new-run", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-scorer", required=True, type=Path)
    parser.add_argument("--compatible-native-scorer", action="append", type=Path)
    parser.add_argument("--seed", type=int, default=260_813)
    parser.add_argument("--threads", type=int, default=MAX_TRAINING_THREADS)
    parser.add_argument(
        "--fold-workers", type=int, default=DEFAULT_CROSSFIT_WORKERS
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPOSITORY / "artifacts/cache/audited-option-episodes",
    )
    parser.add_argument("--load-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.seed != 260_813:
        parser.error("Generation 5 learner seed is fixed at 260813")
    if not 1 <= args.threads <= MAX_TRAINING_THREADS:
        parser.error(
            f"thread budget must be between 1 and {MAX_TRAINING_THREADS}"
        )
    if not 1 <= args.fold_workers <= min(CROSSFIT_FOLDS, args.threads):
        parser.error("fold workers must fit both the fold count and thread budget")
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
    if args.load_workers < 1:
        parser.error("load worker count must be positive")

    def load(run):
        return load_cached_option_episode(
            run,
            loader=_load,
            cache_root=args.cache_dir,
            contract_files=OPTION_CACHE_CONTRACT,
        )

    with ProcessPoolExecutor(
        max_workers=min(args.load_workers, len(runs))
    ) as executor:
        prime_results = list(executor.map(
            _prime_cache, ((run, args.cache_dir) for run in runs)
        ))
    cached = [load(run) for run in runs]
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
        crossfit_workers=args.fold_workers,
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
            "crossfit_workers": args.fold_workers,
            "option_cache_hits": sum(
                hit for hit, _options in prime_results
            ),
            "load_workers": args.load_workers,
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
