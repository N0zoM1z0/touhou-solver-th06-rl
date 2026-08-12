#!/usr/bin/env python3
"""Run a low-cost non-authorizing Generation-5 fit on frozen Wine episodes."""

from __future__ import annotations

import argparse
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
from th06_rl.implicit_learning import crossfit_implicit_q_report  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--n-step-options", type=int, default=8)
    parser.add_argument("--q-trees", type=int, default=8)
    parser.add_argument("--value-trees", type=int, default=8)
    parser.add_argument("--threads", type=int, default=48)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace smoke report: {args.output}")
    runs = tuple(dict.fromkeys(path.resolve() for path in args.runs))
    if len(runs) < 10:
        parser.error("Wine smoke needs at least ten complete episodes")
    started = time.perf_counter()
    loaded = [_load(run) for run in runs]
    loaded_at = time.perf_counter()
    samples = [sample for rows, _report in loaded for sample in rows]
    representation = fit_hazard_codebook(samples, seed=290_813)
    augmented = _augment_steps(samples, representation)
    augmented_at = time.perf_counter()
    report = crossfit_implicit_q_report(
        augmented,
        iterations=args.iterations,
        n_step_options=args.n_step_options,
        q_trees=args.q_trees,
        value_trees=args.value_trees,
        seed=260_813,
        total_threads=args.threads,
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
        "options": len(samples),
        "parameters": {
            "iterations": args.iterations,
            "n_step_options": args.n_step_options,
            "q_trees": args.q_trees,
            "value_trees": args.value_trees,
            "seed": 260_813,
            "threads": args.threads,
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
