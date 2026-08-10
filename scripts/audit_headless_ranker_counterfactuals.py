#!/usr/bin/env python3
"""Evaluate one frozen ranker against exact-state COW action-value labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from collect_headless_dagger import DistilledRanker, source_compatible
    from train_headless_cow_value import load_value_groups
    from train_headless_teacher import load_decisions
except ModuleNotFoundError:
    from scripts.collect_headless_dagger import DistilledRanker, source_compatible
    from scripts.train_headless_cow_value import load_value_groups
    from scripts.train_headless_teacher import load_decisions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_groups(ranker: DistilledRanker, groups) -> dict[str, Any]:
    results = []
    top1 = 0
    completed_or_best_top1 = 0
    reciprocal_ranks = []
    completed_or_best_reciprocal_ranks = []
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for group in groups:
        ranked = ranker.rank_decision(group.decision)
        best_rank = min(ranked.index(action) + 1 for action in group.best_actions)
        matched = ranked[0] in group.best_actions
        completed_or_best_matched = ranked[0] in group.completed_or_best_actions
        top1 += matched
        reciprocal_ranks.append(1.0 / best_rank)
        completed_or_best_rank = min(
            ranked.index(action) + 1 for action in group.completed_or_best_actions
        )
        completed_or_best_top1 += completed_or_best_matched
        completed_or_best_reciprocal_ranks.append(1.0 / completed_or_best_rank)
        result = {
            "seed": group.seed,
            "sequence": group.decision.sequence,
            "observation_sha256": group.observation_sha256,
            "predicted_action": ranked[0],
            "best_actions": list(group.best_actions),
            "predicted_is_best": matched,
            "best_action_rank": best_rank,
            "completed_or_best_actions": list(group.completed_or_best_actions),
            "predicted_is_completed_or_best": completed_or_best_matched,
            "completed_or_best_action_rank": completed_or_best_rank,
            "ranked_actions": list(ranked),
        }
        results.append(result)
        by_seed.setdefault(group.seed, []).append(result)
    return {
        "groups": len(groups),
        "counterfactual_best_top1_accuracy": top1 / len(groups),
        "counterfactual_best_mean_reciprocal_rank": (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
        ),
        "counterfactual_completed_or_best_top1_accuracy": (
            completed_or_best_top1 / len(groups)
        ),
        "counterfactual_completed_or_best_mean_reciprocal_rank": (
            sum(completed_or_best_reciprocal_ranks) / len(groups)
        ),
        "by_seed": [
            {
                "seed": seed,
                "groups": len(seed_results),
                "counterfactual_best_top1_accuracy": (
                    sum(result["predicted_is_best"] for result in seed_results)
                    / len(seed_results)
                ),
                "counterfactual_completed_or_best_top1_accuracy": (
                    sum(
                        result["predicted_is_completed_or_best"]
                        for result in seed_results
                    ) / len(seed_results)
                ),
            }
            for seed, seed_results in sorted(by_seed.items())
        ],
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.threads <= 12:
        parser.error("threads must be in 1..12 on the shared VPS")

    decisions, provenance = load_decisions(args.corpus)
    groups, label_report = load_value_groups(decisions, provenance, args.labels)
    ranker = DistilledRanker(args.model, threads=args.threads)
    if ranker.scope != provenance["scope"]:
        raise ValueError("ranker and COW corpus scopes differ")
    if not source_compatible(ranker.compatible_headless_sources, provenance["source"]):
        raise ValueError("ranker and COW corpus source builds differ")
    if (
        ranker.native_delivery_contract != provenance["native_delivery_contract"]
        or ranker.native_delivery_delays != provenance["native_delivery_delays"]
    ):
        raise ValueError("ranker and COW corpus delivery contracts differ")
    if ranker.observation_digest_contract != provenance["observation_digest_contract"]:
        raise ValueError("ranker and COW corpus observation digest contracts differ")

    report = {
        "schema": "th06-rl-headless-ranker-counterfactual-audit-v1",
        "model": {
            "path": str(args.model.resolve()),
            "sha256": _sha256(args.model),
        },
        "scope": provenance["scope"],
        "source": provenance["source"],
        "native_delivery_contract": provenance["native_delivery_contract"],
        "native_delivery_delays": provenance["native_delivery_delays"],
        "observation_digest_contract": provenance["observation_digest_contract"],
        "label_report": label_report,
        "evaluation": evaluate_groups(ranker, groups),
        "promotion_allowed": False,
        "promotion_blocker": "exact-state ranking is diagnostic until unseen-seed rollout",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
