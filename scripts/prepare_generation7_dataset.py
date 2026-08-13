#!/usr/bin/env python3
"""Materialize learner-independent Generation-7 arrays from registry facts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.generation7.offline_dataset import (  # noqa: E402
    load_episode_arrays,
    prepare_episode_arrays,
)
from th06_rl.wine_corpus_registry import (  # noqa: E402
    load_wine_corpus_registry,
    select_wine_corpora,
)


MAX_WORKERS = 16


def _prepare(payload):
    entry, cache_root = payload
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    paths, hit = prepare_episode_arrays(
        entry,
        repository=REPOSITORY,
        cache_root=cache_root,
    )
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    return str(paths.arrays), hit, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPOSITORY / "config/wine_corpus_registry.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY / "config/generation7_offline_contract.json",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPOSITORY / "artifacts/cache/generation7-factual-arrays",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "artifacts/generation7-offline/dataset.json",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error("workers must be in 1..16")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("wine_outcome_facing_authorized") is not False:
        raise ValueError("dataset preparation cannot be outcome-facing")
    _registry, all_entries = load_wine_corpus_registry(
        args.registry, repository=REPOSITORY
    )
    entries = select_wine_corpora(
        all_entries,
        required_capabilities=frozenset(contract["required_corpus_capabilities"]),
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = tuple(executor.map(
            _prepare,
            ((entry, args.cache_root) for entry in entries),
        ))
    report = {
        "schema": "generation7-factual-dataset-report-v2",
        "evidence_eligible": False,
        "episodes": len(rows),
        "options": sum(int(row[2]["options"]) for row in rows),
        "candidate_rows": sum(int(row[2]["candidate_rows"]) for row in rows),
        "feature_count": len(load_episode_arrays(
            prepare_episode_arrays(
                entries[0], repository=REPOSITORY, cache_root=args.cache_root
            )[0]
        )["feature_names"]),
        "causal_context_feature_count": len(load_episode_arrays(
            prepare_episode_arrays(
                entries[0], repository=REPOSITORY, cache_root=args.cache_root
            )[0]
        )["causal_context_feature_names"]),
        "cache_hits": sum(bool(row[1]) for row in rows),
        "arrays": [row[0] for row in rows],
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
