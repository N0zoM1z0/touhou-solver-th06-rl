#!/usr/bin/env python3
"""Train a CPU LambdaMART ranker on dynamic COW counterfactual outcomes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import resource
import time
from typing import Any, Iterable, Mapping

try:
    from label_headless_cow_counterfactuals import learning_outcome_rank
    from train_headless_teacher import (
        CATEGORICAL_FEATURES,
        Decision,
        Encoder,
        FEATURE_NAMES,
        candidate_features,
        load_decisions,
        repository_commit,
    )
except ModuleNotFoundError:
    from scripts.label_headless_cow_counterfactuals import learning_outcome_rank
    from scripts.train_headless_teacher import (
        CATEGORICAL_FEATURES,
        Decision,
        Encoder,
        FEATURE_NAMES,
        candidate_features,
        load_decisions,
        repository_commit,
    )


@dataclass(frozen=True)
class ValueGroup:
    seed: int
    observation_sha256: str
    decision: Decision
    actions: tuple[str, ...]
    labels: tuple[int, ...]
    best_actions: tuple[str, ...]


def behavior_value_groups(
    decisions: Iterable[Decision],
    *,
    excluded_observations: frozenset[str] = frozenset(),
    stride: int = 1,
) -> list[ValueGroup]:
    """Build conservative behavior targets away from causal COW overrides."""
    if stride <= 0:
        raise ValueError("behavior stride must be positive")
    groups = []
    for decision in decisions:
        if (
            decision.sequence % stride != 0
            or decision.observation_sha256 in excluded_observations
            or decision.selected_action not in decision.legal_actions
        ):
            continue
        actions = tuple(decision.legal_actions)
        groups.append(ValueGroup(
            seed=decision.seed,
            observation_sha256=decision.observation_sha256,
            decision=decision,
            actions=actions,
            labels=tuple(int(action == decision.selected_action) for action in actions),
            best_actions=(decision.selected_action,),
        ))
    return groups


def require_compatible_provenance(
    factual: Mapping[str, Any],
    behavior: Mapping[str, Any],
) -> None:
    for key in (
        "scope",
        "source",
        "native_delivery_contract",
        "native_delivery_delays",
        "observation_digest_contract",
    ):
        if behavior.get(key) != factual.get(key):
            raise ValueError(f"behavior corpus uses incompatible {key}")


def _files(paths: Iterable[Path]) -> tuple[Path, ...]:
    result = []
    for path in paths:
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(path.rglob("*.json")))
    return tuple(dict.fromkeys(item.resolve() for item in result))


def _file_records(files: Iterable[Path]) -> tuple[list[dict[str, str]], str]:
    records = []
    for path in files:
        records.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return records, hashlib.sha256(payload).hexdigest()


def ordinal_outcome_labels(outcomes: Iterable[Mapping[str, Any]]) -> tuple[int, ...]:
    ranks = tuple(learning_outcome_rank(outcome) for outcome in outcomes)
    ordered = {rank: index for index, rank in enumerate(sorted(set(ranks)))}
    return tuple(ordered[rank] for rank in ranks)


def delivery_contract(document: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    name = str(document.get(
        "runtime_delivery_contract",
        "legacy-unspecified-v0",
    ))
    raw_delays = document.get("runtime_delivery_delays")
    delays = (
        tuple(int(value) for value in raw_delays)
        if isinstance(raw_delays, list)
        else ()
    )
    return name, delays


def load_value_groups(
    decisions: list[Decision],
    provenance: Mapping[str, Any],
    paths: Iterable[Path],
) -> tuple[list[ValueGroup], dict[str, Any]]:
    files = _files(paths)
    if not files:
        raise ValueError("no COW value label files found")
    file_records, file_set_sha256 = _file_records(files)
    decision_by_digest = {decision.observation_sha256: decision for decision in decisions}
    groups_by_checkpoint: dict[tuple[int, str], tuple[int, ValueGroup]] = {}
    runtime_sources = set()
    expected_delivery = (
        str(provenance.get("native_delivery_contract", "legacy-unspecified-v0")),
        tuple(int(value) for value in provenance.get("native_delivery_delays", ())),
    )
    expected_digest_contract = str(provenance.get(
        "observation_digest_contract",
        "legacy-full-observation-v0",
    ))
    unmatched = 0
    uninformative = 0
    duplicate_checkpoints = 0
    longer_horizon_replacements = 0
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != "th06-rl-headless-cow-counterfactual-v1":
            raise ValueError(f"unsupported COW value schema: {path}")
        if document.get("scope") != provenance["scope"]:
            raise ValueError("COW value labels silently mix scopes")
        if document.get("input_source", {}).get("commit") != provenance["source"].get("commit"):
            raise ValueError("COW value labels use a different factual source revision")
        current_delivery = delivery_contract(document)
        if current_delivery != expected_delivery:
            raise ValueError("COW value labels use a different delivery contract")
        if str(document.get(
            "observation_digest_contract",
            "legacy-full-observation-v0",
        )) != expected_digest_contract:
            raise ValueError("COW value labels use a different observation digest contract")
        runtime_sources.add(json.dumps(document["runtime_source"], sort_keys=True))
        seed = int(document["initial_seed"])
        for checkpoint in document["checkpoints"]:
            digest = str(checkpoint["observation_sha256"])
            decision = decision_by_digest.get(digest)
            if decision is None:
                unmatched += 1
                continue
            outcomes = tuple(checkpoint["outcomes"])
            by_action = {str(outcome["first_action"]): outcome for outcome in outcomes}
            actions = tuple(decision.legal_actions)
            if set(by_action) != set(actions):
                raise ValueError("COW outcomes do not equal the native legal action set")
            ordered_outcomes = tuple(by_action[action] for action in actions)
            labels = ordinal_outcome_labels(ordered_outcomes)
            if len(set(labels)) == 1:
                uninformative += 1
                continue
            best_label = max(labels)
            group = ValueGroup(
                seed=seed,
                observation_sha256=digest,
                decision=decision,
                actions=actions,
                labels=labels,
                best_actions=tuple(
                    action for action, label in zip(actions, labels, strict=True)
                    if label == best_label
                ),
            )
            key = (seed, digest)
            branch_frames = int(checkpoint["branch_frames"])
            previous = groups_by_checkpoint.get(key)
            if previous is not None:
                duplicate_checkpoints += 1
                previous_horizon, previous_group = previous
                if branch_frames < previous_horizon:
                    continue
                if branch_frames == previous_horizon:
                    if group != previous_group:
                        raise ValueError(
                            "duplicate COW checkpoints disagree at the same horizon"
                        )
                    continue
                longer_horizon_replacements += 1
            groups_by_checkpoint[key] = (branch_frames, group)
    groups = [entry[1] for entry in groups_by_checkpoint.values()]
    if not groups:
        raise ValueError("no COW value groups matched compact corpus observations")
    return groups, {
        "files": len(files),
        "files_used": file_records,
        "file_set_sha256": file_set_sha256,
        "groups": len(groups),
        "candidate_outcomes": sum(len(group.actions) for group in groups),
        "unmatched_checkpoints": unmatched,
        "uninformative_checkpoints": uninformative,
        "duplicate_checkpoints": duplicate_checkpoints,
        "longer_horizon_replacements": longer_horizon_replacements,
        "value_target": "completed-quality-buckets-or-failed-survival-v3",
        "runtime_sources": [json.loads(item) for item in sorted(runtime_sources)],
        "native_delivery_contract": expected_delivery[0],
        "native_delivery_delays": list(expected_delivery[1]),
        "observation_digest_contract": expected_digest_contract,
    }


def _matrix(groups: list[ValueGroup]):
    features = []
    labels = []
    sizes = []
    for group in groups:
        candidates = {str(candidate["action"]): candidate for candidate in group.decision.candidates}
        sizes.append(len(group.actions))
        for action, label in zip(group.actions, group.labels, strict=True):
            features.append(candidate_features(group.decision, candidates[action]))
            labels.append(label)
    return features, labels, sizes


def evaluate(model, encoder: Encoder, groups: list[ValueGroup], *, threads: int) -> dict[str, Any]:
    features, _, sizes = _matrix(groups)
    scores = model.booster_.predict(encoder.encode(features), num_threads=threads)
    offset = 0
    top1 = 0
    reciprocal_ranks = []
    for group, size in zip(groups, sizes, strict=True):
        ranked = sorted(
            zip(group.actions, scores[offset:offset + size], strict=True),
            key=lambda item: (float(item[1]), item[0]),
            reverse=True,
        )
        offset += size
        top1 += ranked[0][0] in group.best_actions
        best_rank = next(
            index for index, (action, _) in enumerate(ranked, 1)
            if action in group.best_actions
        )
        reciprocal_ranks.append(1.0 / best_rank)
    return {
        "groups": len(groups),
        "candidate_outcomes": sum(len(group.actions) for group in groups),
        "counterfactual_best_top1_accuracy": top1 / len(groups),
        "counterfactual_best_mean_reciprocal_rank": sum(reciprocal_ranks) / len(groups),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--behavior-corpus",
        nargs="+",
        type=Path,
        help="optional exact-contract behavior support for conservative policy improvement",
    )
    parser.add_argument("--behavior-stride", type=int, default=1)
    parser.add_argument("--behavior-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--holdout-seed", type=int, action="append")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=6006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 12:
        parser.error("threads must be in 1..12")
    if min(args.iterations, args.num_leaves, args.max_depth, args.min_child_samples) <= 0:
        parser.error("model capacity bounds must be positive")
    if args.behavior_stride <= 0 or min(args.behavior_weight, args.value_weight) <= 0:
        parser.error("behavior stride and training weights must be positive")
    # Freeze implementation provenance before decoding or fitting. Other
    # benchmark commits may advance this branch while the CPU job is active.
    code_commit = repository_commit()
    # Freeze a live generator directory before the potentially long corpus
    # decode. The report below records the exact immutable snapshot.
    label_files = _files(args.labels)
    decisions, provenance = load_decisions(args.corpus)
    groups, label_report = load_value_groups(decisions, provenance, label_files)
    seeds = sorted({group.seed for group in groups})
    holdout = set(args.holdout_seed or seeds[-1:])
    train = [group for group in groups if group.seed not in holdout]
    test = [group for group in groups if group.seed in holdout]
    if not train or not test:
        parser.error("COW value train and holdout must contain complete seed groups")
    behavior_groups: list[ValueGroup] = []
    behavior_provenance: Mapping[str, Any] | None = None
    if args.behavior_corpus:
        behavior_decisions, behavior_provenance = load_decisions(args.behavior_corpus)
        require_compatible_provenance(provenance, behavior_provenance)
        behavior_groups = behavior_value_groups(
            behavior_decisions,
            excluded_observations=frozenset(group.observation_sha256 for group in groups),
            stride=args.behavior_stride,
        )
        if not behavior_groups:
            parser.error("behavior corpus supplied no conservative support groups")
    behavior_train = [group for group in behavior_groups if group.seed not in holdout]
    behavior_test = [group for group in behavior_groups if group.seed in holdout]
    combined_train = behavior_train + train
    encoder = Encoder(group.decision for group in combined_train)
    train_features, train_labels, train_sizes = _matrix(combined_train)
    train_weights = [
        weight
        for groups_part, weight in (
            (behavior_train, args.behavior_weight),
            (train, args.value_weight),
        )
        for group in groups_part
        for _ in group.actions
    ]

    from lightgbm import LGBMRanker
    import joblib
    import numpy as np

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.iterations,
        learning_rate=0.04,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=args.threads,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(
        encoder.encode(train_features),
        np.asarray(train_labels, dtype=np.int16),
        group=train_sizes,
        sample_weight=np.asarray(train_weights, dtype=np.float32),
        categorical_feature=list(range(len(CATEGORICAL_FEATURES))),
    )
    elapsed = time.perf_counter() - started
    compatible_sources = [provenance["source"]]
    for source in label_report["runtime_sources"]:
        if source not in compatible_sources:
            compatible_sources.append(source)
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "categories": encoder.manifest(),
        "scope": provenance["scope"],
        "headless_source": provenance["source"],
        "compatible_headless_sources": compatible_sources,
        "native_delivery_contract": provenance["native_delivery_contract"],
        "native_delivery_delays": provenance["native_delivery_delays"],
        "observation_digest_contract": provenance["observation_digest_contract"],
        "value_contract": "dynamic-cow-quality-tier-conservative-improvement-v3",
    }
    joblib.dump(artifact, args.output / "cow-value-ranker.joblib", compress=3)
    report = {
        "schema": "th06-rl-headless-cow-value-v1",
        "algorithm": (
            "lightgbm-lambdarank-counterfactual-value-with-behavior-regularization"
            if behavior_groups
            else "lightgbm-lambdarank-counterfactual-action-value"
        ),
        "authority": "rank-native-legal-set-only",
        "scope": provenance["scope"],
        "code_commit": code_commit,
        "factual_source": provenance["source"],
        "compatible_headless_sources": compatible_sources,
        "native_delivery_contract": provenance["native_delivery_contract"],
        "native_delivery_delays": provenance["native_delivery_delays"],
        "observation_digest_contract": provenance["observation_digest_contract"],
        "label_report": label_report,
        "behavior_regularization": {
            "enabled": bool(behavior_groups),
            "stride": args.behavior_stride,
            "behavior_weight": args.behavior_weight,
            "value_weight": args.value_weight,
            "train_groups": len(behavior_train),
            "holdout_groups": len(behavior_test),
            "factual_corpus": (
                behavior_provenance.get("factual_corpus")
                if behavior_provenance is not None
                else None
            ),
        },
        "train_seeds": sorted(set(seeds) - holdout),
        "holdout_seeds": sorted(holdout),
        "iterations": args.iterations,
        "threads": args.threads,
        "fit_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "train": evaluate(model, encoder, train, threads=args.threads),
        "holdout": evaluate(model, encoder, test, threads=args.threads),
        "behavior_train": (
            evaluate(model, encoder, behavior_train, threads=args.threads)
            if behavior_train else None
        ),
        "behavior_holdout": (
            evaluate(model, encoder, behavior_test, threads=args.threads)
            if behavior_test else None
        ),
        "promotion_allowed": False,
        "promotion_blocker": (
            "counterfactual ranking is an offline diagnostic until full unseen-seed stage rollout"
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
