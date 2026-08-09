#!/usr/bin/env python3
"""Gate an exact-value correction behind a conservative incumbent-risk model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
from typing import Any, Iterable, Mapping

try:
    from collect_headless_dagger import DistilledRanker, source_compatible
    from train_headless_cow_value import ValueGroup, load_value_groups
    from train_headless_teacher import (
        CATEGORICAL_FEATURES,
        Encoder,
        FEATURE_NAMES,
        candidate_features,
        load_decisions,
        repository_commit,
    )
except ModuleNotFoundError:
    from scripts.collect_headless_dagger import DistilledRanker, source_compatible
    from scripts.train_headless_cow_value import ValueGroup, load_value_groups
    from scripts.train_headless_teacher import (
        CATEGORICAL_FEATURES,
        Encoder,
        FEATURE_NAMES,
        candidate_features,
        load_decisions,
        repository_commit,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if artifact.get("ensemble_members") is not None or artifact.get("supported_residual"):
        raise ValueError("nested ensemble/residual rankers are not supported")
    return {
        "model": artifact["model"],
        "feature_names": artifact.get("feature_names"),
        "categories": artifact["categories"],
    }


def _candidate_feature(decision, action: str):
    candidates = {
        str(candidate["action"]): candidate for candidate in decision.candidates
    }
    if action not in candidates:
        raise ValueError("residual gate action escaped the native-safe candidates")
    return candidate_features(decision, candidates[action])


def risk_label(group: ValueGroup) -> int:
    """Mark only incumbent actions without a completed/best COW witness."""
    return int(group.decision.selected_action not in group.completed_or_best_actions)


def _metrics(labels: list[int], probabilities, threshold: float) -> dict[str, Any]:
    predictions = [float(value) >= threshold for value in probabilities]
    if len(labels) != len(predictions):
        raise ValueError("gate labels and probabilities differ in length")
    true_positive = sum(
        label == 1 and prediction
        for label, prediction in zip(labels, predictions, strict=True)
    )
    false_positive = sum(
        label == 0 and prediction
        for label, prediction in zip(labels, predictions, strict=True)
    )
    positive = sum(labels)
    negative = len(labels) - positive
    predicted_positive = true_positive + false_positive
    return {
        "groups": len(labels),
        "risk_groups": positive,
        "predicted_risk_groups": predicted_positive,
        "precision": true_positive / predicted_positive if predicted_positive else 1.0,
        "recall": true_positive / positive if positive else 1.0,
        "false_positive_rate": false_positive / negative if negative else 0.0,
    }


def choose_threshold(
    cow_labels: list[int],
    cow_probabilities,
    behavior_probabilities,
    *,
    min_precision: float,
    max_behavior_activation: float,
) -> tuple[float, dict[str, Any]]:
    candidates = sorted(
        {float(value) for value in cow_probabilities}
        | {float(value) for value in behavior_probabilities},
        reverse=True,
    )
    accepted = []
    for threshold in candidates:
        metrics = _metrics(cow_labels, cow_probabilities, threshold)
        activation = sum(
            float(value) >= threshold for value in behavior_probabilities
        ) / max(len(behavior_probabilities), 1)
        if (
            metrics["precision"] >= min_precision
            and activation <= max_behavior_activation
        ):
            accepted.append((metrics["recall"], threshold, activation, metrics))
    if not accepted:
        return 1.000001, {
            "status": "no-safe-training-threshold",
            "behavior_activation_ratio": 0.0,
            **_metrics(cow_labels, cow_probabilities, 1.000001),
        }
    recall, threshold, activation, metrics = max(
        accepted,
        key=lambda item: (item[0], item[1]),
    )
    return threshold, {
        "status": "selected-on-training-only",
        "behavior_activation_ratio": activation,
        **metrics,
    }


def _gate_probabilities(model, encoder: Encoder, features, *, threads: int):
    return model.booster_.predict(encoder.encode(features), num_threads=threads)


def residual_value_metrics(
    groups: Iterable[ValueGroup],
    probabilities,
    *,
    threshold: float,
    correction: DistilledRanker,
) -> dict[str, Any]:
    groups = list(groups)
    strict_best = 0
    completed_or_best = 0
    overrides = 0
    rows = []
    for group, probability in zip(groups, probabilities, strict=True):
        base_action = group.decision.selected_action
        correction_action = correction.rank_decision(group.decision)[0]
        override = float(probability) >= threshold and correction_action != base_action
        selected = correction_action if override else base_action
        overrides += override
        strict_best += selected in group.best_actions
        completed_or_best += selected in group.completed_or_best_actions
        rows.append({
            "seed": group.seed,
            "sequence": group.decision.sequence,
            "risk_probability": float(probability),
            "base_action": base_action,
            "correction_action": correction_action,
            "selected_action": selected,
            "override": override,
            "selected_is_best": selected in group.best_actions,
            "selected_is_completed_or_best": selected in group.completed_or_best_actions,
        })
    return {
        "groups": len(groups),
        "overrides": overrides,
        "override_ratio": overrides / len(groups),
        "counterfactual_best_top1_accuracy": strict_best / len(groups),
        "counterfactual_completed_or_best_top1_accuracy": completed_or_best / len(groups),
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--correction-model", type=Path, required=True)
    parser.add_argument("--holdout-seed", type=int, action="append", required=True)
    parser.add_argument("--behavior-stride", type=int, default=8)
    parser.add_argument("--cow-weight", type=float, default=50.0)
    parser.add_argument("--min-train-precision", type=float, default=0.9)
    parser.add_argument("--max-behavior-activation", type=float, default=0.005)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 12:
        parser.error("threads must be in 1..12 on the shared VPS")
    if (
        args.behavior_stride <= 0
        or args.cow_weight <= 0
        or min(
            args.iterations,
            args.num_leaves,
            args.max_depth,
            args.min_child_samples,
        ) <= 0
    ):
        parser.error("stride, COW weight, and iterations must be positive")
    if not 0 < args.min_train_precision <= 1:
        parser.error("minimum precision must be in (0, 1]")
    if not 0 <= args.max_behavior_activation <= 1:
        parser.error("maximum behavior activation must be in [0, 1]")

    code_commit = repository_commit()
    decisions, provenance = load_decisions(args.corpus)
    groups, label_report = load_value_groups(decisions, provenance, args.labels)
    holdout = set(args.holdout_seed)
    train_groups = [group for group in groups if group.seed not in holdout]
    test_groups = [group for group in groups if group.seed in holdout]
    if not train_groups or not test_groups:
        parser.error("risk gate train and holdout must contain complete seed groups")
    if not any(risk_label(group) for group in train_groups):
        parser.error("risk gate training split has no incumbent-risk group")

    import joblib
    import numpy as np
    from lightgbm import LGBMClassifier

    base_path = args.base_model.resolve()
    correction_path = args.correction_model.resolve()
    base_sha256 = _sha256(base_path)
    for run in provenance["factual_corpus"]["runs_used"]:
        manifest = json.loads(
            (Path(run["run"]) / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("ranker", {}).get("sha256") != base_sha256:
            parser.error("factual corpus was not generated by the declared base ranker")
    base_artifact = joblib.load(base_path)
    correction_artifact = joblib.load(correction_path)
    for name, artifact in (("base", base_artifact), ("correction", correction_artifact)):
        if artifact.get("scope") != provenance["scope"]:
            parser.error(f"{name} ranker scope differs from the factual corpus")
        allowed = artifact.get("compatible_headless_sources", [artifact.get("headless_source")])
        if not source_compatible(allowed, provenance["source"]):
            parser.error(f"{name} ranker source differs from the factual corpus")
    if (
        correction_artifact.get("native_delivery_contract")
        != provenance["native_delivery_contract"]
        or correction_artifact.get("native_delivery_delays")
        != provenance["native_delivery_delays"]
        or correction_artifact.get("observation_digest_contract")
        != provenance["observation_digest_contract"]
    ):
        parser.error("correction ranker uses incompatible runtime contracts")

    cow_digests = {group.observation_sha256 for group in groups}
    behavior_decisions = [
        decision for decision in decisions
        if decision.sequence % args.behavior_stride == 0
        and decision.observation_sha256 not in cow_digests
        and decision.selected_action in decision.legal_actions
    ]
    behavior_train = [item for item in behavior_decisions if item.seed not in holdout]
    behavior_test = [item for item in behavior_decisions if item.seed in holdout]
    encoder = Encoder([group.decision for group in train_groups] + behavior_train)
    cow_train_features = [
        _candidate_feature(group.decision, group.decision.selected_action)
        for group in train_groups
    ]
    cow_test_features = [
        _candidate_feature(group.decision, group.decision.selected_action)
        for group in test_groups
    ]
    behavior_train_features = [
        _candidate_feature(decision, decision.selected_action)
        for decision in behavior_train
    ]
    behavior_test_features = [
        _candidate_feature(decision, decision.selected_action)
        for decision in behavior_test
    ]
    train_features = behavior_train_features + cow_train_features
    train_labels = [0] * len(behavior_train_features) + [
        risk_label(group) for group in train_groups
    ]
    train_weights = [1.0] * len(behavior_train_features) + [
        args.cow_weight
    ] * len(cow_train_features)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=args.iterations,
        learning_rate=0.04,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=6006,
        n_jobs=args.threads,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(
        encoder.encode(train_features),
        np.asarray(train_labels, dtype=np.int8),
        sample_weight=np.asarray(train_weights, dtype=np.float32),
        categorical_feature=list(range(len(CATEGORICAL_FEATURES))),
    )
    elapsed = time.perf_counter() - started
    cow_train_probabilities = _gate_probabilities(
        model, encoder, cow_train_features, threads=args.threads
    )
    cow_test_probabilities = _gate_probabilities(
        model, encoder, cow_test_features, threads=args.threads
    )
    behavior_train_probabilities = _gate_probabilities(
        model, encoder, behavior_train_features, threads=args.threads
    )
    behavior_test_probabilities = _gate_probabilities(
        model, encoder, behavior_test_features, threads=args.threads
    )
    threshold, threshold_report = choose_threshold(
        [risk_label(group) for group in train_groups],
        cow_train_probabilities,
        behavior_train_probabilities,
        min_precision=args.min_train_precision,
        max_behavior_activation=args.max_behavior_activation,
    )
    correction = DistilledRanker(correction_path, threads=args.threads)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "supported_residual": {
            "base": _member(base_artifact),
            "correction": _member(correction_artifact),
            "gate": {
                "model": model,
                "feature_names": FEATURE_NAMES,
                "categories": encoder.manifest(),
                "threshold": threshold,
            },
        },
        "scope": provenance["scope"],
        "headless_source": provenance["source"],
        "compatible_headless_sources": [provenance["source"]],
        "native_delivery_contract": provenance["native_delivery_contract"],
        "native_delivery_delays": provenance["native_delivery_delays"],
        "observation_digest_contract": provenance["observation_digest_contract"],
        "residual_contract": "incumbent-default-cow-risk-gated-v1",
    }
    artifact_path = output / "residual-ranker.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    report = {
        "schema": "th06-rl-headless-cow-supported-residual-v1",
        "algorithm": "incumbent-default-cow-risk-gated-residual",
        "authority": "rank-native-legal-set-only",
        "code_commit": code_commit,
        "scope": provenance["scope"],
        "source": provenance["source"],
        "native_delivery_contract": provenance["native_delivery_contract"],
        "native_delivery_delays": provenance["native_delivery_delays"],
        "observation_digest_contract": provenance["observation_digest_contract"],
        "base_model": {"path": str(base_path), "sha256": base_sha256},
        "correction_model": {
            "path": str(correction_path), "sha256": _sha256(correction_path)
        },
        "output_model": {
            "path": artifact_path.name, "sha256": _sha256(artifact_path)
        },
        "factual_corpus": provenance["factual_corpus"],
        "label_report": label_report,
        "train_seeds": sorted({group.seed for group in train_groups}),
        "holdout_seeds": sorted({group.seed for group in test_groups}),
        "behavior_stride": args.behavior_stride,
        "cow_weight": args.cow_weight,
        "threshold": threshold,
        "threshold_selection": threshold_report,
        "train_gate": _metrics(
            [risk_label(group) for group in train_groups],
            cow_train_probabilities,
            threshold,
        ),
        "holdout_gate": _metrics(
            [risk_label(group) for group in test_groups],
            cow_test_probabilities,
            threshold,
        ),
        "behavior_train_activation_ratio": sum(
            float(value) >= threshold for value in behavior_train_probabilities
        ) / max(len(behavior_train_probabilities), 1),
        "behavior_holdout_activation_ratio": sum(
            float(value) >= threshold for value in behavior_test_probabilities
        ) / max(len(behavior_test_probabilities), 1),
        "train_residual_value": residual_value_metrics(
            train_groups,
            cow_train_probabilities,
            threshold=threshold,
            correction=correction,
        ),
        "holdout_residual_value": residual_value_metrics(
            test_groups,
            cow_test_probabilities,
            threshold=threshold,
            correction=correction,
        ),
        "iterations": args.iterations,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "threads": args.threads,
        "fit_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "promotion_allowed": False,
        "promotion_blocker": "risk-gated residual requires unseen-seed natural rollout",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "report.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
