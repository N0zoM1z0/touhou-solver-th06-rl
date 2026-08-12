#!/usr/bin/env python3
"""Run a low-cost non-authorizing Generation-5 fit on frozen Wine episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fit_supported_implicit_q import _load  # noqa: E402
from th06_rl.advantage_learning import (  # noqa: E402
    _augment_steps,
    fit_hazard_codebook,
)
from th06_rl.implicit_learning import (  # noqa: E402
    CROSSFIT_FOLDS,
    DEFAULT_CROSSFIT_WORKERS,
    MAX_TRAINING_THREADS,
    crossfit_implicit_q_report,
)
from th06_rl.option_cache import (  # noqa: E402
    load_cached_option_episode,
    prime_option_episode_cache,
)


OPTION_CACHE_CONTRACT = (
    REPOSITORY / "src/th06_rl/advantage_learning.py",
    REPOSITORY / "src/th06_rl/autonomous_learning.py",
    REPOSITORY / "src/th06_rl/corpus.py",
    REPOSITORY / "scripts/fit_supported_implicit_q.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--n-step-options", type=int, default=8)
    parser.add_argument("--q-trees", type=int, default=8)
    parser.add_argument("--value-trees", type=int, default=8)
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
    if args.output.exists():
        raise FileExistsError(f"refusing to replace smoke report: {args.output}")
    runs = tuple(dict.fromkeys(path.resolve() for path in args.runs))
    if len(runs) < 10:
        parser.error("Wine smoke needs at least ten complete episodes")
    started = time.perf_counter()
    if args.load_workers < 1:
        parser.error("load worker count must be positive")
    if not 1 <= args.threads <= MAX_TRAINING_THREADS:
        parser.error(
            f"thread budget must be between 1 and {MAX_TRAINING_THREADS}"
        )
    if not 1 <= args.fold_workers <= min(CROSSFIT_FOLDS, args.threads):
        parser.error("fold workers must fit both the fold count and thread budget")

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
    loaded_at = time.perf_counter()
    samples = [sample for rows, _report in loaded for sample in rows]
    new_episode_ids = frozenset(
        rows[0].episode_id for rows, _report in loaded[-16:]
    )
    representation = fit_hazard_codebook(samples, seed=290_813)
    augmented = _augment_steps(samples, representation)
    augmented_at = time.perf_counter()
    report = crossfit_implicit_q_report(
        augmented,
        new_episode_ids=new_episode_ids,
        iterations=args.iterations,
        n_step_options=args.n_step_options,
        q_trees=args.q_trees,
        value_trees=args.value_trees,
        seed=260_813,
        total_threads=args.threads,
        fold_workers=args.fold_workers,
    )
    completed_at = time.perf_counter()
    result = {
        "schema": "autonomous-generation-5-frozen-wine-smoke-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "runs": [{
            "path": str(run),
            "manifest_sha256": _sha256(run / "manifest.json"),
        } for run in runs],
        "episode_groups": len(runs),
        "new_development_episode_groups": len(new_episode_ids),
        "options": len(samples),
        "option_cache": {
            "hits": sum(hit for hit, _options in prime_results),
            "misses": sum(not hit for hit, _options in prime_results),
            "root": str(args.cache_dir.resolve()),
        },
        "parameters": {
            "iterations": args.iterations,
            "n_step_options": args.n_step_options,
            "q_trees": args.q_trees,
            "value_trees": args.value_trees,
            "seed": 260_813,
            "threads": args.threads,
            "fold_workers": args.fold_workers,
            "load_workers": args.load_workers,
        },
        "timing_seconds": {
            "load": loaded_at - started,
            "representation_and_augmentation": augmented_at - loaded_at,
            "crossfit": completed_at - augmented_at,
            "total": completed_at - started,
        },
        "report": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "episode_groups": len(runs),
        "options": len(samples),
        "relative_q_loss": report["overall"]["relative_q_loss"],
        "episodes_beating_zero": report["overall"]["episodes_beating_zero"],
        "proposals": report["proposals"],
        "proposal_rate": report["proposal_rate"],
        "conditional_half_agreement": report["conditional_half_agreement"],
        "seconds": completed_at - started,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
