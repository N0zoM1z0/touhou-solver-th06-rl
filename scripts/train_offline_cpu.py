#!/usr/bin/env python3
"""Train and evaluate CPU policy-ranking candidates on complete physical Stages."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import time
import warnings

from th06_rl.offline import ACTION_NAMES, load_dataset_index
from th06_rl.offline_learning import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    LABEL_SCHEMA,
    LabeledTransition,
    features_for_candidate,
    load_labeled_run,
    regression_metrics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Encoder:
    def __init__(self, rows: list[LabeledTransition]) -> None:
        self.categories: dict[str, dict[str, int]] = {}
        for name in CATEGORICAL_FEATURES:
            values = {str(row.features[name]) for row in rows}
            if name in ("action", "baseline_action", "current_action"):
                values.update(ACTION_NAMES)
                values.add("unknown")
            self.categories[name] = {
                value: index for index, value in enumerate(sorted(values))
            }

    def encode(self, features: list[dict[str, str | float]]):
        import numpy as np

        output = np.empty((len(features), len(FEATURE_NAMES)), dtype=np.float32)
        categorical = set(CATEGORICAL_FEATURES)
        for column, name in enumerate(FEATURE_NAMES):
            if name in categorical:
                mapping = self.categories[name]
                output[:, column] = [mapping.get(str(row[name]), -1) for row in features]
            else:
                output[:, column] = [float(row[name]) for row in features]
        return output

    def manifest(self) -> dict[str, list[str]]:
        return {
            name: [value for value, _ in sorted(mapping.items(), key=lambda item: item[1])]
            for name, mapping in self.categories.items()
        }


def _model(algorithm: str, *, threads: int, iterations: int, seed: int):
    categorical_indices = [FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURES]
    if algorithm == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=iterations,
            depth=8,
            learning_rate=0.06,
            loss_function="RMSE",
            random_seed=seed,
            thread_count=threads,
            allow_writing_files=False,
            verbose=False,
        ), {"categorical_indices": categorical_indices}
    if algorithm == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=iterations,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=10,
            min_child_samples=80,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
        ), {"categorical_feature": categorical_indices}
    if algorithm == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=iterations,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=threads,
        ), {}
    if algorithm == "extra-trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=max(128, iterations // 2),
            max_depth=24,
            min_samples_leaf=4,
            max_features=0.85,
            random_state=seed,
            n_jobs=threads,
        ), {}
    raise ValueError(f"unknown algorithm: {algorithm}")


def _fit(model, algorithm: str, x, y, weights, fit_options: dict[str, object]) -> None:
    if algorithm == "catboost":
        categorical_indices = fit_options["categorical_indices"]
        catboost_x = x.astype(object)
        for index in categorical_indices:
            catboost_x[:, index] = x[:, index].astype(int).astype(str)
        model.fit(
            catboost_x,
            y,
            sample_weight=weights,
            cat_features=categorical_indices,
        )
    elif algorithm == "lightgbm":
        model.fit(x, y, sample_weight=weights, categorical_feature=fit_options["categorical_feature"])
    else:
        model.fit(x, y, sample_weight=weights)


def _predict(model, encoder: Encoder, features: list[dict[str, str | float]], *, chunk: int = 65536):
    import numpy as np

    values = []
    for start in range(0, len(features), chunk):
        matrix = encoder.encode(features[start:start + chunk])
        if model.__class__.__module__.startswith("catboost"):
            catboost_matrix = matrix.astype(object)
            for index in (FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURES):
                catboost_matrix[:, index] = matrix[:, index].astype(int).astype(str)
            matrix = catboost_matrix
        with warnings.catch_warnings():
            # LightGBM records synthetic feature names even when fitted from a
            # NumPy matrix.  Candidate scoring deliberately uses the same
            # ordered matrix contract, so this sklearn warning is inapplicable.
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            values.append(np.asarray(model.predict(matrix), dtype=float))
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _ope(rows: list[LabeledTransition], chosen: list[str], q_target, q_logged) -> dict[str, float | int | None]:
    weights = []
    corrections = []
    rewards = []
    for row, target, target_q, logged_q in zip(rows, chosen, q_target, q_logged, strict=True):
        weight = min(20.0, (1.0 / row.behavior_probability) if target == row.action else 0.0)
        weights.append(weight)
        rewards.append(row.reward)
        corrections.append(float(target_q) + weight * (row.reward - float(logged_q)))
    weight_sum = sum(weights)
    weight_square_sum = sum(value * value for value in weights)
    return {
        "rows": len(rows),
        "logged_action_matches": sum(target == row.action for row, target in zip(rows, chosen, strict=True)),
        "match_rate": sum(target == row.action for row, target in zip(rows, chosen, strict=True)) / len(rows),
        "clipped_ips": sum(weight * reward for weight, reward in zip(weights, rewards, strict=True)) / len(rows),
        "clipped_wis": (
            sum(weight * reward for weight, reward in zip(weights, rewards, strict=True)) / weight_sum
            if weight_sum else None
        ),
        "clipped_dr": sum(corrections) / len(corrections),
        "clipped_weight_sum": weight_sum,
        "clipped_ess": weight_sum * weight_sum / weight_square_sum if weight_square_sum else 0.0,
    }


def _policy_metrics(
    model,
    encoder: Encoder,
    rows: list[LabeledTransition],
    training_support: Counter[tuple[str, str]],
) -> dict[str, object]:
    q_logged = _predict(model, encoder, [row.features for row in rows])
    chosen: dict[str, list[str]] = {
        "unconstrained": [],
        "baseline_prior_0_18": [],
        "support32_margin_0_5": [],
        "support32_margin_1_0": [],
    }
    target_q: dict[str, list[float]] = {name: [] for name in chosen}
    baseline_q = []
    # Bound peak memory while scoring every native-safe counterfactual action.
    for row_start in range(0, len(rows), 2048):
        batch = rows[row_start:row_start + 2048]
        candidate_features = []
        slices = []
        for row in batch:
            start = len(candidate_features)
            candidate_features.extend(
                features_for_candidate(row, action) for action in row.legal_actions
            )
            slices.append((start, len(candidate_features)))
        candidate_q = _predict(model, encoder, candidate_features)
        for row, (start, stop) in zip(batch, slices, strict=True):
            scores = candidate_q[start:stop]
            index = max(
                range(len(scores)),
                key=lambda item: (float(scores[item]), row.legal_actions[item]),
            )
            baseline_index = row.legal_actions.index(row.baseline_action)
            baseline_value = float(scores[baseline_index])
            baseline_q.append(baseline_value)
            unconstrained = row.legal_actions[index]
            chosen["unconstrained"].append(unconstrained)
            target_q["unconstrained"].append(float(scores[index]))
            prior_index = max(
                range(len(scores)),
                key=lambda item: (
                    float(scores[item])
                    + (0.18 if row.legal_actions[item] == row.baseline_action else 0.0),
                    row.legal_actions[item],
                ),
            )
            chosen["baseline_prior_0_18"].append(row.legal_actions[prior_index])
            target_q["baseline_prior_0_18"].append(float(scores[prior_index]))
            supported = [
                item for item, action in enumerate(row.legal_actions)
                if action == row.baseline_action
                or training_support[(row.source_context, action)] >= 32
            ]
            supported_index = max(
                supported,
                key=lambda item: (float(scores[item]), row.legal_actions[item]),
            )
            for margin in (0.5, 1.0):
                name = f"support32_margin_{str(margin).replace('.', '_')}"
                selected_index = (
                    supported_index
                    if float(scores[supported_index]) >= baseline_value + margin
                    else baseline_index
                )
                chosen[name].append(row.legal_actions[selected_index])
                target_q[name].append(float(scores[selected_index]))
    baseline = [row.baseline_action for row in rows]
    evaluated = {
        name: {
            **_ope(rows, actions, target_q[name], q_logged),
            "action_counts": dict(Counter(actions).most_common()),
            "vs_baseline_disagreements": sum(
                action != baseline_action
                for action, baseline_action in zip(actions, baseline, strict=True)
            ),
        }
        for name, actions in chosen.items()
    }
    return {
        "logged_behavior": {
            "mean_reconstructed_reward": sum(row.reward for row in rows) / len(rows),
            "hit_within_120_rate": sum(row.hit_within_120 for row in rows) / len(rows),
        },
        "model_policy": evaluated["unconstrained"],
        "policy_variants": evaluated,
        "reactive_baseline": _ope(rows, baseline, baseline_q, q_logged),
        "model_action_counts": dict(Counter(chosen["unconstrained"]).most_common()),
        "baseline_action_counts": dict(Counter(baseline).most_common()),
        "model_vs_baseline_disagreements": evaluated["unconstrained"]["vs_baseline_disagreements"],
        "any_variant_selected_outside_native_safe_set": sum(
            action not in row.legal_actions
            for actions in chosen.values()
            for action, row in zip(actions, rows, strict=True)
        ),
        "bomb_selections_across_variants": sum(
            "bomb" in action.casefold()
            for actions in chosen.values()
            for action in actions
        ),
    }


def _load_training_rows(
    dataset: Path,
    runs,
    *,
    exact_context_only: bool,
    limit: int,
    seed: int,
) -> tuple[list[LabeledTransition], int, list[float]]:
    """Keep every rare failure window and reservoir-sample ordinary rows."""
    rng = random.Random(seed)
    important: list[LabeledTransition] = []
    ordinary: list[LabeledTransition] = []
    ordinary_seen = 0
    total = 0
    for run in runs:
        for row in load_labeled_run(
            dataset,
            run,
            exact_context_only=exact_context_only,
        ):
            total += 1
            if row.hit_within_120 or row.reward < 0.0:
                important.append(row)
                continue
            ordinary_seen += 1
            if len(ordinary) < limit:
                ordinary.append(row)
            else:
                index = rng.randrange(ordinary_seen)
                if index < limit:
                    ordinary[index] = row
    if len(important) >= limit:
        rng.shuffle(important)
        selected = important[:limit]
        return selected, total, [len(important) / len(selected)] * len(selected)
    retained_ordinary = ordinary[:limit - len(important)]
    ordinary_weight = ordinary_seen / len(retained_ordinary) if retained_ordinary else 1.0
    return (
        important + retained_ordinary,
        total,
        [1.0] * len(important) + [ordinary_weight] * len(retained_ordinary),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", required=True, help="difficulty/character/shot/stage")
    parser.add_argument("--view", choices=("exact-v5", "common"), default="exact-v5")
    parser.add_argument("--algorithms", default="catboost,lightgbm,xgboost,extra-trees")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--validation-runs", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=6006)
    args = parser.parse_args()
    if min(args.threads, args.iterations, args.validation_runs, args.max_train_rows) <= 0:
        parser.error("resource and split parameters must be positive")
    scope = tuple(int(value) for value in args.scope.split("/"))
    if len(scope) != 4:
        parser.error("scope must contain exactly four integers")
    _, indexed = load_dataset_index(args.dataset)
    runs = [
        run for run in indexed
        if run.scope == scope
        and run.training_eligible
        and (run.transition_schema == "th06-rl-transition-v5" or args.view == "common")
    ]
    if len(runs) <= args.validation_runs:
        raise SystemExit("scope/view needs at least one train and one validation run")
    runs.sort(key=lambda run: run.run_id)
    train_runs = runs[:-args.validation_runs]
    validation_runs = runs[-args.validation_runs:]
    exact = args.view == "exact-v5"
    sampled_train, train_rows_before_cap, corpus_sampling_weights = _load_training_rows(
        args.dataset,
        train_runs,
        exact_context_only=exact,
        limit=args.max_train_rows,
        seed=args.seed,
    )
    validation_rows = [
        row
        for run in validation_runs
        for row in load_labeled_run(args.dataset, run, exact_context_only=exact)
    ]
    encoder = Encoder(sampled_train)
    x_train = encoder.encode([row.features for row in sampled_train])
    y_train = [row.reward for row in sampled_train]
    # Temper inverse propensity to reduce behavior-policy bias without letting
    # rare exploratory rows dominate a small physical-run split.
    weights = [
        corpus_weight * math.sqrt(min(20.0, 1.0 / row.behavior_probability))
        for row, corpus_weight in zip(
            sampled_train,
            corpus_sampling_weights,
            strict=True,
        )
    ]
    training_support = Counter(
        (row.source_context, row.action) for row in sampled_train
    )
    y_validation = [row.reward for row in validation_rows]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "encoder.json").write_text(
        json.dumps(encoder.manifest(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    algorithms = [value.strip() for value in args.algorithms.split(",") if value.strip()]
    results = {}
    versions = {}
    for index, algorithm in enumerate(algorithms):
        started = time.monotonic()
        model, fit_options = _model(
            algorithm,
            threads=args.threads,
            iterations=args.iterations,
            seed=args.seed + index,
        )
        _fit(model, algorithm, x_train, y_train, weights, fit_options)
        fit_seconds = time.monotonic() - started
        predicted = [
            float(value)
            for value in _predict(model, encoder, [row.features for row in validation_rows])
        ]
        model_path = args.output / f"{algorithm}.joblib"
        import joblib

        joblib.dump(model, model_path, compress=3)
        hit_actual = [row.hit_within_120 for row in validation_rows]
        hit_metrics: dict[str, float | None] = {"average_precision_from_negative_q": None, "roc_auc_from_negative_q": None}
        if len(set(hit_actual)) == 2:
            from sklearn.metrics import average_precision_score, roc_auc_score

            hit_metrics = {
                "average_precision_from_negative_q": float(average_precision_score(hit_actual, [-value for value in predicted])),
                "roc_auc_from_negative_q": float(roc_auc_score(hit_actual, [-value for value in predicted])),
            }
        policy_evaluation = _policy_metrics(
            model,
            encoder,
            validation_rows,
            training_support,
        )
        results[algorithm] = {
            "model_file": model_path.name,
            "model_sha256": _sha256(model_path),
            "fit_seconds": fit_seconds,
            "total_seconds": time.monotonic() - started,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "factual_validation": regression_metrics(y_validation, predicted),
            "hit_120_ranking": hit_metrics,
            "policy_evaluation": policy_evaluation,
        }
        module = "sklearn" if algorithm == "extra-trees" else algorithm.replace("-", "_")
        try:
            package = __import__(module)
            versions[algorithm] = str(package.__version__)
        except (ImportError, AttributeError):
            versions[algorithm] = "unknown"
    manifest = {
        "schema": "th06-rl-offline-cpu-policy-zoo-v1",
        "dataset": {"revision": args.revision, "path": str(args.dataset)},
        "scope": list(scope),
        "view": args.view,
        "feature_schema": FEATURE_SCHEMA,
        "label_schema": LABEL_SCHEMA,
        "features": {"names": FEATURE_NAMES, "categorical": CATEGORICAL_FEATURES},
        "split": {
            "kind": "chronological-complete-practice-stage",
            "train_runs": [run.run_id for run in train_runs],
            "validation_runs": [run.run_id for run in validation_runs],
            "train_rows_before_cap": train_rows_before_cap,
            "train_rows": len(sampled_train),
            "validation_rows": len(validation_rows),
            "train_hit_within_120_rate": (
                sum(row.hit_within_120 for row in sampled_train) / len(sampled_train)
            ),
            "validation_hit_within_120_rate": (
                sum(row.hit_within_120 for row in validation_rows) / len(validation_rows)
            ),
            "maximum_corpus_sampling_weight": max(corpus_sampling_weights),
        },
        "resource_limits": {"threads": args.threads, "sequential_models": True, "max_train_rows": args.max_train_rows},
        "libraries": versions,
        "results": results,
        "deployment_boundary": {
            "native_gate_remains_authoritative": True,
            "fresh_issue_revalidation_required": True,
            "bomb_representable": False,
            "offline_metrics_are_not_physical_promotion": True,
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
