#!/usr/bin/env python3
"""Evaluate nested cross-fit policy-level IQL actor on development Wine."""

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
from th06_rl.iql_actor_learning import crossfit_iql_actor_report  # noqa: E402
from th06_rl.low_rank_learning import named_feature_roles  # noqa: E402
from th06_rl.option_cache import load_cached_option_episode  # noqa: E402
from th06_rl.qualification_corpus import load_qualification_partition  # noqa: E402
from th06_rl.resource_control import enforce_training_cpu_affinity  # noqa: E402
from th06_rl.wine_corpus_registry import (  # noqa: E402
    load_wine_corpus_registry,
    select_wine_corpora,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--actor-advantage-folds", type=int, default=3)
    parser.add_argument("--critic-iterations", type=int, default=2)
    parser.add_argument("--q-trees", type=int, default=8)
    parser.add_argument("--value-trees", type=int, default=8)
    parser.add_argument("--actor-hidden", type=int, default=64)
    parser.add_argument("--actor-rank", type=int, default=24)
    parser.add_argument("--actor-epochs", type=int, default=8)
    parser.add_argument("--actor-batch-size", type=int, default=1024)
    parser.add_argument(
        "--contract", type=Path,
        default=REPOSITORY / "config/autonomous_generation6_qualification.json",
    )
    parser.add_argument(
        "--registry", type=Path,
        help="use every registered sequential training episode instead",
    )
    parser.add_argument("--seed", type=int, default=460_813)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=REPOSITORY / "artifacts/cache/audited-option-episodes",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace development report: {args.output}")
    started = time.perf_counter()
    affinity = enforce_training_cpu_affinity(args.threads)
    if args.registry is None:
        _contract, partition = load_qualification_partition(
            args.contract, repository=REPOSITORY
        )
        development = tuple(row for row in partition if row.role == "development")
        input_identity = {
            "kind": "historical-development-partition",
            "sha256": _sha256(args.contract),
        }
    else:
        _registry, entries = load_wine_corpus_registry(
            args.registry, repository=REPOSITORY
        )
        development = select_wine_corpora(
            entries,
            required_capabilities=frozenset({"sequential_offline_rl"}),
        )
        if not development:
            raise ValueError("Generation-6 smoke selected no sequential corpus")
        input_identity = {
            "kind": "immutable-capability-registry",
            "sha256": _sha256(args.registry),
            "sources": sorted({row.source for row in development}),
            "manifest_sha256": [row.manifest_sha256 for row in development],
        }
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
    cohorts = {
        rows[0].episode_id: f"stage-{spec.stage}"
        for spec, (rows, _report, _hit) in zip(development, loaded, strict=True)
    }
    report = crossfit_iql_actor_report(
        samples,
        layout=named_feature_roles(rich_feature_names()),
        episode_cohorts=cohorts,
        folds=args.folds,
        critic_iterations=args.critic_iterations,
        q_trees=args.q_trees,
        value_trees=args.value_trees,
        total_threads=args.threads,
        actor_hidden=args.actor_hidden,
        actor_rank=args.actor_rank,
        actor_epochs=args.actor_epochs,
        actor_batch_size=args.actor_batch_size,
        actor_advantage_folds=args.actor_advantage_folds,
        seed=args.seed,
        intervention_probability_cap=0.10,
        intervention_density_ratio_cap=2.0,
        fit_representation_on_train=True,
    )
    completed = time.perf_counter()
    result = {
        "schema": "autonomous-generation-6-crossfit-actor-development-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "qualification_samples_loaded": args.registry is not None,
        "representation_fit_scope": "outer-training-episodes-only",
        "input_identity": input_identity,
        "learner_sha256": _sha256(
            REPOSITORY / "src/th06_rl/iql_actor_learning.py"
        ),
        "development_episode_groups": len(development),
        "options": len(samples),
        "parameters": {
            "threads": args.threads,
            "folds": args.folds,
            "actor_advantage_folds": args.actor_advantage_folds,
            "critic_iterations": args.critic_iterations,
            "n_step_options": 8,
            "q_trees": args.q_trees,
            "value_trees": args.value_trees,
            "actor_hidden": args.actor_hidden,
            "actor_rank": args.actor_rank,
            "actor_epochs": args.actor_epochs,
            "actor_batch_size": args.actor_batch_size,
            "actor_learning_rate": 1e-3,
            "actor_log_weight_clip": 4.0,
            "intervention_probability_cap": 0.10,
            "intervention_density_ratio_cap": 2.0,
            "seed": args.seed,
        },
        "resource_contract": affinity.as_dict(),
        "cache_hits": sum(hit for _rows, _report, hit in loaded),
        "timing_seconds": {
            "load": loaded_at - started,
            "crossfit_with_fold_representation": completed - loaded_at,
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
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
