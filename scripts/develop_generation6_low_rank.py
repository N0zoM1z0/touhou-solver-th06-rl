#!/usr/bin/env python3
"""Evaluate Generation-6 low-rank critic on frozen development episodes only."""

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

from th06_rl.advantage_learning import rich_feature_names  # noqa: E402
from th06_rl.audited_option_loader import (  # noqa: E402
    AUDITED_OPTION_LOADER_CONTRACT,
    load_audited_option_episode,
)
from th06_rl.low_rank_learning import (  # noqa: E402
    crossfit_low_rank_report,
    named_feature_roles,
)
from th06_rl.option_cache import load_cached_option_episode  # noqa: E402
from th06_rl.qualification_corpus import load_qualification_partition  # noqa: E402
from th06_rl.resource_control import enforce_training_cpu_affinity  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--members", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--q-trees", type=int, default=16)
    parser.add_argument("--value-trees", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
        raise FileExistsError(f"refusing to replace development report: {args.output}")
    started = time.perf_counter()
    affinity = enforce_training_cpu_affinity(args.threads)
    contract, partition = load_qualification_partition(
        args.contract, repository=REPOSITORY
    )
    development = tuple(row for row in partition if row.role == "development")
    if len(development) != 31:
        raise ValueError("Generation-6 development partition drifted")
    loaded = [
        load_cached_option_episode(
            row.path,
            loader=load_audited_option_episode,
            cache_root=args.cache_dir,
            contract_files=AUDITED_OPTION_LOADER_CONTRACT,
        )
        for row in development
    ]
    loaded_at = time.perf_counter()
    samples = [sample for rows, _report, _hit in loaded for sample in rows]
    cohorts = {
        rows[0].episode_id: f"stage-{spec.stage}"
        for spec, (rows, _report, _hit) in zip(development, loaded, strict=True)
    }
    report = crossfit_low_rank_report(
        samples,
        layout=named_feature_roles(rich_feature_names()),
        episode_cohorts=cohorts,
        folds=args.folds,
        members=args.members,
        iterations=args.iterations,
        q_trees=args.q_trees,
        value_trees=args.value_trees,
        threads=args.threads,
        hidden=args.hidden,
        rank=args.rank,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    completed = time.perf_counter()
    result = {
        "schema": "autonomous-generation-6-low-rank-development-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "qualification_samples_loaded": False,
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "source_sha256": _sha256(Path(__file__)),
        "learner_sha256": _sha256(
            REPOSITORY / "src/th06_rl/low_rank_learning.py"
        ),
        "development_episode_groups": len(development),
        "options": len(samples),
        "parameters": {
            "threads": args.threads,
            "folds": args.folds,
            "members": args.members,
            "iterations": args.iterations,
            "n_step_options": 8,
            "q_trees": args.q_trees,
            "value_trees": args.value_trees,
            "hidden": args.hidden,
            "rank": args.rank,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": 360813,
        },
        "resource_contract": affinity.as_dict(),
        "cache_hits": sum(hit for _rows, _report, hit in loaded),
        "timing_seconds": {
            "load": loaded_at - started,
            "crossfit": completed - loaded_at,
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
        "cohorts": report["cohorts"],
        "full_proposal_rate": report["full_proposal_rate"],
        "full_proposal_loo_exact_rate": report["full_proposal_loo_exact_rate"],
        "loo_union_stability": report["loo_union_stability"],
        "split_conditional_agreement": report["split_conditional_agreement"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
