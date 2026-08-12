#!/usr/bin/env python3
"""Measure the old tree critic's actual full policy on development data."""

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

from th06_rl.advantage_learning import _augment_steps, fit_hazard_codebook  # noqa: E402
from th06_rl.audited_option_loader import (  # noqa: E402
    AUDITED_OPTION_LOADER_CONTRACT,
    load_audited_option_episode,
)
from th06_rl.implicit_learning import crossfit_implicit_q_report  # noqa: E402
from th06_rl.option_cache import load_cached_option_episode  # noqa: E402
from th06_rl.qualification_corpus import load_qualification_partition  # noqa: E402
from th06_rl.resource_control import enforce_training_cpu_affinity  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--q-trees", type=int, default=8)
    parser.add_argument("--value-trees", type=int, default=8)
    parser.add_argument(
        "--contract", type=Path,
        default=REPOSITORY / "config/autonomous_generation6_qualification.json",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=REPOSITORY / "artifacts/cache/audited-option-episodes",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace diagnostic report: {args.output}")
    started = time.perf_counter()
    affinity = enforce_training_cpu_affinity(args.threads)
    _contract, partition = load_qualification_partition(
        args.contract, repository=REPOSITORY
    )
    development = tuple(row for row in partition if row.role == "development")
    loaded = [
        load_cached_option_episode(
            row.path,
            loader=load_audited_option_episode,
            cache_root=args.cache_dir,
            contract_files=AUDITED_OPTION_LOADER_CONTRACT,
        )
        for row in development
    ]
    samples = [sample for rows, _report, _hit in loaded for sample in rows]
    loaded_at = time.perf_counter()
    representation = fit_hazard_codebook(samples, seed=390_813)
    augmented = _augment_steps(samples, representation)
    augmented_at = time.perf_counter()
    report = crossfit_implicit_q_report(
        augmented,
        new_episode_ids=frozenset(sample.episode_id for sample in augmented),
        iterations=args.iterations,
        n_step_options=8,
        q_trees=args.q_trees,
        value_trees=args.value_trees,
        seed=360_813,
        total_threads=args.threads,
        fold_workers=5,
    )
    completed = time.perf_counter()
    result = {
        "schema": "autonomous-generation-6-tree-full-policy-diagnostic-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "qualification_samples_loaded": False,
        "representation_fit_scope": (
            "all-development-unsupervised-diagnostic-not-qualification"
        ),
        "contract_sha256": _sha256(args.contract),
        "learner_sha256": _sha256(
            REPOSITORY / "src/th06_rl/implicit_learning.py"
        ),
        "development_episode_groups": len(development),
        "options": len(samples),
        "parameters": {
            "iterations": args.iterations,
            "n_step_options": 8,
            "q_trees": args.q_trees,
            "value_trees": args.value_trees,
            "seed": 360813,
            "folds": 5,
            "members": 7,
            "threads": args.threads,
        },
        "resource_contract": affinity.as_dict(),
        "cache_hits": sum(hit for _rows, _report, hit in loaded),
        "timing_seconds": {
            "load": loaded_at - started,
            "representation": augmented_at - loaded_at,
            "crossfit": completed - augmented_at,
            "total": completed - started,
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
        "seconds": completed - started,
        "relative_q_loss": report["overall"]["relative_q_loss"],
        "episodes_beating_zero": report["overall"]["episodes_beating_zero"],
        "deployed_population_proposal_rate": (
            report["deployed_population_proposal_rate"]
        ),
        "deployed_proposal_loo_exact_rate": (
            report["deployed_proposal_loo_exact_rate"]
        ),
        "deployed_population_loo_union_stability": (
            report["deployed_population_loo_union_stability"]
        ),
        "split_conditional_agreement": report["conditional_half_agreement"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
