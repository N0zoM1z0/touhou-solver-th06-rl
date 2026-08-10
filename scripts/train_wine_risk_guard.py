#!/usr/bin/env python3
"""Train a factual Stage 6 failure-risk guard with grouped Wine holdouts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

from th06_rl.offline import ACTION_NAMES
from th06_rl.wine_risk import (
    FROZEN_INCUMBENT_POLICY_ID,
    RISK_LABEL_SCHEMA,
    RISK_CATEGORICAL_FEATURES,
    RISK_FEATURE_NAMES,
    RISK_FEATURE_NAMES_V2,
    RISK_FEATURE_SCHEMA,
    RISK_FEATURE_SCHEMA_V2,
    RISK_CATEGORICAL_FEATURES_V2,
    FirstFailurePrefix,
    RiskExample,
    load_first_failure_prefix,
)


TRAINING_SCHEMA = "th06-rl-wine-risk-guard-training-v1"
DIAGNOSTIC_SCHEMA = "th06-rl-wine-risk-guard-diagnostic-v1"
ONE_SIDED_95_Z = 1.6448536269514722
FACTUAL_ACTION_AUDIT_SCHEMA = "th06-rl-wine-risk-consensus-replay-v1"
RESIDUAL_PROPOSAL_AUDIT_SCHEMA = "th06-rl-wine-offline-residual-replay-v1"
SOURCE_CONTEXT_FEATURE = "source_context"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _validate_factual_action_audit(
    path: Path,
    *,
    scope: tuple[int, int, int, int],
    prefixes: list[FirstFailurePrefix],
) -> dict[str, object]:
    """Bind shadow-collected rows to a full frozen-incumbent replay audit."""
    report = _object(path)
    totals = report.get("totals")
    runs = report.get("runs")
    if (
        report.get("schema") != FACTUAL_ACTION_AUDIT_SCHEMA
        or report.get("passed") is not True
        or report.get("mode") != "shadow"
        or report.get("expect_recorded_actions") != "incumbent"
        or report.get("scope") != list(scope)
        or not isinstance(totals, dict)
        or not isinstance(runs, list)
    ):
        raise ValueError("factual-action audit contract is invalid")
    for key in (
        "recorded_incumbent_mismatches",
        "recorded_policy_mismatches",
        "shadow_action_contract_violations",
    ):
        if int(totals.get(key, -1)) != 0:
            raise ValueError(f"factual-action audit contains {key}")
    expected = {
        prefix.run_id: (prefix.manifest_sha256, prefix.run_sha256)
        for prefix in prefixes
    }
    observed: dict[str, tuple[str, str]] = {}
    for raw in runs:
        if not isinstance(raw, dict):
            raise TypeError("factual-action audit run is not an object")
        run_id = str(raw.get("run_id", ""))
        if not run_id or run_id in observed:
            raise ValueError("factual-action audit run identity is invalid")
        if int(raw.get("policy_calls", 0)) <= 0:
            raise ValueError("factual-action audit run has no policy calls")
        for key in (
            "recorded_incumbent_mismatches",
            "recorded_policy_mismatches",
            "shadow_action_contract_violations",
        ):
            if raw.get(key) != []:
                raise ValueError(f"factual-action audit run contains {key}")
        observed[run_id] = (
            str(raw.get("manifest_sha256", "")),
            str(raw.get("run_sha256", "")),
        )
    if observed != expected or int(totals.get("runs", -1)) != len(prefixes):
        raise ValueError("factual-action audit does not exactly cover training runs")
    state_path = Path(str(report.get("state", "")))
    state_sha256 = str(report.get("state_sha256", ""))
    if not state_path.is_file() or _sha256(state_path) != state_sha256:
        raise ValueError("factual-action audit state identity is stale")
    return {
        "schema": FACTUAL_ACTION_AUDIT_SCHEMA,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "state": str(state_path.resolve()),
        "state_sha256": state_sha256,
        "runs": sorted(expected),
        "policy_calls": int(totals.get("policy_calls", 0)),
        "contract": "recorded-action-equals-frozen-incumbent",
    }


def _validate_residual_proposal_audit(
    path: Path,
    *,
    scope: tuple[int, int, int, int],
    prefixes: list[FirstFailurePrefix],
) -> tuple[dict[str, object], dict[str, frozenset[int]]]:
    """Bind a ranker-residual eligibility set to exact factual Wine runs."""
    report = _object(path)
    totals = report.get("totals")
    runs = report.get("runs")
    semantics = report.get("semantics")
    if (
        report.get("schema") != RESIDUAL_PROPOSAL_AUDIT_SCHEMA
        or report.get("passed") is not True
        or report.get("scope") != list(scope)
        or not isinstance(totals, dict)
        or not isinstance(runs, list)
        or not isinstance(semantics, dict)
        or semantics.get("published_action") != "none-offline-replay-only"
        or semantics.get("recorded_action_authority") != "frozen-incumbent"
        or semantics.get("residual_guard_eligibility")
        != "exact-factual-risk-example-and-fallback-opportunity"
    ):
        raise ValueError("residual-proposal audit contract is invalid")
    for key in (
        "recorded_incumbent_mismatches",
        "action_contract_violations",
    ):
        if int(totals.get(key, -1)) != 0:
            raise ValueError(f"residual-proposal audit contains {key}")

    expected = {
        prefix.run_id: prefix for prefix in prefixes
    }
    observed: dict[str, tuple[str, str]] = {}
    eligible: dict[str, frozenset[int]] = {}
    for raw in runs:
        if not isinstance(raw, dict):
            raise TypeError("residual-proposal audit run is not an object")
        run_id = str(raw.get("run_id", ""))
        if not run_id or run_id in observed or run_id not in expected:
            raise ValueError("residual-proposal audit run identity is invalid")
        if (
            int(raw.get("policy_calls", 0)) <= 0
            or int(raw.get("recorded_incumbent_mismatch_count", -1)) != 0
            or int(raw.get("action_contract_violation_count", -1)) != 0
        ):
            raise ValueError("residual-proposal audit run did not replay cleanly")
        sequences = raw.get("residual_guard_eligible_sequences")
        if (
            not isinstance(sequences, list)
            or any(not isinstance(value, int) for value in sequences)
            or len(sequences) != len(set(sequences))
            or len(sequences) != int(raw.get("residual_guard_eligible", -1))
        ):
            raise ValueError("residual-proposal eligibility set is invalid")
        prefix_rows = {
            row.transition.sequence: row for row in expected[run_id].examples
        }
        if any(
            sequence not in prefix_rows
            or not prefix_rows[sequence].fallback_opportunity
            for sequence in sequences
        ):
            raise ValueError("residual proposal escaped factual risk eligibility")
        observed[run_id] = (
            str(raw.get("manifest_sha256", "")),
            str(raw.get("run_sha256", "")),
        )
        eligible[run_id] = frozenset(sequences)

    expected_hashes = {
        run_id: (prefix.manifest_sha256, prefix.run_sha256)
        for run_id, prefix in expected.items()
    }
    if observed != expected_hashes or int(totals.get("runs", -1)) != len(prefixes):
        raise ValueError("residual-proposal audit does not exactly cover input runs")
    state_records = {}
    for name in ("incumbent_state", "ranker_state", "risk_state"):
        state_path = Path(str(report.get(name, "")))
        state_sha256 = str(report.get(f"{name}_sha256", ""))
        if not state_path.is_file() or _sha256(state_path) != state_sha256:
            raise ValueError(f"residual-proposal {name} identity is stale")
        state_records[name] = str(state_path.resolve())
        state_records[f"{name}_sha256"] = state_sha256
    return ({
        "schema": RESIDUAL_PROPOSAL_AUDIT_SCHEMA,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "runs": sorted(expected),
        "policy_calls": int(totals.get("policy_calls", 0)),
        "eligible_rows": sum(len(values) for values in eligible.values()),
        "contract": "offline-ranker-supported-alternative-differs-from-incumbent",
        **state_records,
    }, eligible)


class Encoder:
    def __init__(
        self,
        rows: list[RiskExample],
        *,
        feature_names: tuple[str, ...] = RISK_FEATURE_NAMES,
        categorical_features: tuple[str, ...] = RISK_CATEGORICAL_FEATURES,
    ) -> None:
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.categories: dict[str, dict[str, int]] = {}
        for name in categorical_features:
            values = {str(self._value(row, name)) for row in rows}
            if name in ("action", "baseline_action", "current_action"):
                values.update(ACTION_NAMES)
                values.add("unknown")
            self.categories[name] = {
                value: index for index, value in enumerate(sorted(values))
            }

    @staticmethod
    def _value(row: RiskExample, name: str) -> str | float:
        if name == SOURCE_CONTEXT_FEATURE:
            return row.transition.source_context
        return row.features[name]

    def encode(self, rows: list[RiskExample]):
        import numpy as np

        output = np.empty((len(rows), len(self.feature_names)), dtype=np.float32)
        categorical = set(self.categorical_features)
        for column, name in enumerate(self.feature_names):
            if name in categorical:
                mapping = self.categories[name]
                output[:, column] = [
                    mapping.get(str(self._value(row, name)), -1)
                    for row in rows
                ]
            else:
                output[:, column] = [
                    float(self._value(row, name)) for row in rows
                ]
        return output

    def manifest(self) -> dict[str, list[str]]:
        return {
            name: [
                value for value, _index in sorted(
                    mapping.items(), key=lambda item: item[1]
                )
            ]
            for name, mapping in self.categories.items()
        }


def _model(*, threads: int, iterations: int, seed: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=iterations,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=threads,
    )


def _fit(
    rows: list[RiskExample],
    *,
    threads: int,
    iterations: int,
    seed: int,
    feature_names: tuple[str, ...] = RISK_FEATURE_NAMES,
    categorical_features: tuple[str, ...] = RISK_CATEGORICAL_FEATURES,
):
    import numpy as np

    encoder = Encoder(
        rows,
        feature_names=feature_names,
        categorical_features=categorical_features,
    )
    matrix = encoder.encode(rows)
    labels = np.asarray([row.failure_within_120 for row in rows], dtype=np.float32)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("risk training fold requires both positive and negative rows")
    positive_weight = min(20.0, negatives / positives)
    weights = np.where(labels > 0.0, positive_weight, 1.0)
    model = _model(threads=threads, iterations=iterations, seed=seed)
    model.fit(matrix, labels, sample_weight=weights)
    return model, encoder, positive_weight


def _precision_lower_bound(true_positives: int, activations: int) -> float | None:
    """One-sided 95% Wilson lower bound for guard precision."""
    if activations <= 0:
        return None
    probability = true_positives / activations
    z2 = ONE_SIDED_95_Z * ONE_SIDED_95_Z
    denominator = 1.0 + z2 / activations
    center = probability + z2 / (2.0 * activations)
    spread = ONE_SIDED_95_Z * math.sqrt(
        probability * (1.0 - probability) / activations
        + z2 / (4.0 * activations * activations)
    )
    return (center - spread) / denominator


def _metrics(labels, scores, active) -> dict[str, object]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    active = np.asarray(active, dtype=bool)
    tp = int(np.sum(active & labels))
    fp = int(np.sum(active & ~labels))
    positives = int(np.sum(labels))
    negatives = int(np.sum(~labels))
    return {
        "rows": int(len(labels)),
        "positives": positives,
        "positive_rate": positives / len(labels),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "activations": int(np.sum(active)),
        "activation_rate": float(np.mean(active)),
        "true_positives": tp,
        "false_positives": fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "precision_lower_bound_95_one_sided": _precision_lower_bound(
            tp, tp + fp,
        ),
        "recall": tp / positives if positives else None,
        "false_positive_rate": fp / negatives if negatives else None,
    }


def _select_threshold(
    rows: list[RiskExample],
    scores: list[float],
    *,
    minimum_precision: float,
    minimum_precision_lower_bound: float,
    maximum_activation_rate: float,
    minimum_protected_runs: int,
) -> tuple[float, dict[str, object]]:
    opportunities = [row.fallback_opportunity for row in rows]
    labels = [row.failure_within_120 for row in rows]
    run_ids = [row.transition.run_id for row in rows]
    candidates = sorted(set(scores), reverse=True)
    accepted = []
    for threshold in candidates:
        active = [
            opportunity and score >= threshold
            for opportunity, score in zip(opportunities, scores, strict=True)
        ]
        tp = sum(flag and label for flag, label in zip(active, labels, strict=True))
        fp = sum(flag and not label for flag, label in zip(active, labels, strict=True))
        count = tp + fp
        precision = tp / count if count else 0.0
        precision_lower_bound = _precision_lower_bound(tp, count) or 0.0
        protected = len({
            run_id for run_id, flag, label in zip(run_ids, active, labels, strict=True)
            if flag and label
        })
        activation_rate = count / len(rows)
        if (
            count
            and precision >= minimum_precision
            and precision_lower_bound >= minimum_precision_lower_bound
            and activation_rate <= maximum_activation_rate
            and protected >= minimum_protected_runs
        ):
            accepted.append((tp, protected, precision, -activation_rate, threshold))
    if not accepted:
        raise RuntimeError("no OOF risk threshold satisfies the conservative guard constraints")
    threshold = max(accepted)[-1]
    active = [
        opportunity and score >= threshold
        for opportunity, score in zip(opportunities, scores, strict=True)
    ]
    detail = _metrics(labels, scores, active)
    detail.update({
        "threshold": threshold,
        "minimum_precision": minimum_precision,
        "minimum_precision_lower_bound_95_one_sided": (
            minimum_precision_lower_bound
        ),
        "maximum_activation_rate": maximum_activation_rate,
        "minimum_protected_runs": minimum_protected_runs,
        "protected_runs": sorted({
            run_id for run_id, flag, label in zip(run_ids, active, labels, strict=True)
            if flag and label
        }),
        "correctable_positive_rows": sum(
            label and opportunity
            for label, opportunity in zip(labels, opportunities, strict=True)
        ),
    })
    return threshold, detail


def _threshold_frontier(
    rows: list[RiskExample],
    scores: list[float],
    *,
    maximum_activation_rate: float,
    minimum_protected_runs: int,
) -> dict[str, object]:
    """Summarize the best guard evidence without relaxing acceptance gates."""
    labels = [row.failure_within_120 for row in rows]
    opportunities = [row.fallback_opportunity for row in rows]
    run_ids = [row.transition.run_id for row in rows]
    candidates = []
    for threshold in sorted(set(scores), reverse=True):
        active = [
            opportunity and score >= threshold
            for opportunity, score in zip(opportunities, scores, strict=True)
        ]
        count = sum(active)
        if not count or count / len(rows) > maximum_activation_rate:
            continue
        tp = sum(flag and label for flag, label in zip(active, labels, strict=True))
        fp = count - tp
        protected = len({
            run_id
            for run_id, flag, label in zip(run_ids, active, labels, strict=True)
            if flag and label
        })
        candidates.append({
            "threshold": threshold,
            "activations": count,
            "activation_rate": count / len(rows),
            "true_positives": tp,
            "false_positives": fp,
            "precision": tp / count,
            "precision_lower_bound_95_one_sided": _precision_lower_bound(tp, count),
            "protected_runs": protected,
        })

    def best(values: list[dict[str, object]]) -> dict[str, object] | None:
        if not values:
            return None
        return max(
            values,
            key=lambda item: (
                float(item["precision"]),
                int(item["true_positives"]),
                int(item["protected_runs"]),
                -float(item["activation_rate"]),
                float(item["threshold"]),
            ),
        )

    required = [
        item for item in candidates
        if int(item["protected_runs"]) >= minimum_protected_runs
    ]
    maximum_protected_runs = max(
        (int(item["protected_runs"]) for item in candidates),
        default=0,
    )
    return {
        "candidate_thresholds_within_activation_limit": len(candidates),
        "maximum_activation_rate": maximum_activation_rate,
        "required_protected_runs": minimum_protected_runs,
        "maximum_protected_runs": maximum_protected_runs,
        "best_any_nonempty": best(candidates),
        "best_with_required_protected_runs": best(required),
    }


def _run_summary(prefix: FirstFailurePrefix) -> dict[str, object]:
    positives = [row for row in prefix.examples if row.failure_within_120]
    return {
        "run_id": prefix.run_id,
        "path": str(prefix.run_dir),
        "manifest_sha256": prefix.manifest_sha256,
        "run_sha256": prefix.run_sha256,
        "code_commit": prefix.code_commit,
        "failure_kind": prefix.failure_kind,
        "failure_frame": prefix.failure_frame,
        "failure_context": prefix.failure_context,
        "failure_segment_start_frame": prefix.failure_segment_start_frame,
        "positive_window_start_frame": prefix.positive_window_start_frame,
        "transitions": prefix.transitions,
        "eligible_rows": len(prefix.examples),
        "positive_rows": len(positives),
        "correctable_positive_rows": sum(row.fallback_opportunity for row in positives),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        help="write grouped OOF evidence even when conservative selection rejects",
    )
    parser.add_argument("--scope", default="3/0/0/6")
    parser.add_argument("--retail-sha256", required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--iterations", type=int, default=320)
    parser.add_argument("--seed", type=int, default=6006)
    parser.add_argument("--minimum-precision", type=float, default=0.60)
    parser.add_argument(
        "--minimum-precision-lower-bound",
        type=float,
        default=0.60,
        help="minimum one-sided 95%% Wilson lower bound for OOF guard precision",
    )
    parser.add_argument("--maximum-activation-rate", type=float, default=0.02)
    parser.add_argument("--minimum-protected-runs", type=int, default=2)
    parser.add_argument(
        "--feature-profile",
        choices=("legacy-v1", "context-reactive-v2"),
        default="legacy-v1",
        help="versioned deployable feature contract",
    )
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        choices=RISK_FEATURE_NAMES,
        help=(
            "offline ablation only; excluded-feature artifacts use a "
            "non-exportable diagnostic feature schema"
        ),
    )
    parser.add_argument(
        "--include-source-context",
        action="store_true",
        help=(
            "offline ablation only; condition on automatically derived source "
            "context and emit a non-exportable diagnostic feature schema"
        ),
    )
    parser.add_argument(
        "--factual-action-audit",
        type=Path,
        help=(
            "full shadow consensus replay proving that every recorded action "
            "in every input run equals the frozen incumbent; required when "
            "training on shadow-collected factual runs"
        ),
    )
    parser.add_argument(
        "--residual-proposal-audit",
        type=Path,
        help=(
            "exact offline replay whose supported-ranker proposal set limits "
            "this diagnostic to incumbent-retaining residual opportunities"
        ),
    )
    parser.add_argument(
        "--retain-fold-models",
        action="store_true",
        help=(
            "retain every leave-one-physical-run-out model and encoder for "
            "offline consensus auditing; these are not deployment evidence"
        ),
    )
    args = parser.parse_args()
    scope = tuple(int(value) for value in args.scope.split("/"))
    if len(scope) != 4:
        parser.error("scope must contain four integers")
    if min(args.threads, args.iterations, args.minimum_protected_runs) <= 0:
        parser.error("resource and protected-run bounds must be positive")
    if not 0.0 < args.minimum_precision <= 1.0:
        parser.error("minimum precision must be in (0, 1]")
    if not 0.0 < args.minimum_precision_lower_bound <= 1.0:
        parser.error("minimum precision lower bound must be in (0, 1]")
    if not 0.0 < args.maximum_activation_rate <= 1.0:
        parser.error("maximum activation rate must be in (0, 1]")
    if (
        args.diagnostic_output is not None
        and args.diagnostic_output.resolve() == args.output.resolve()
    ):
        parser.error("diagnostic output must differ from the model output directory")
    if args.feature_profile == "context-reactive-v2":
        if args.exclude_feature or args.include_source_context:
            parser.error(
                "context-reactive-v2 cannot be combined with audit ablation flags"
            )
        excluded_features = frozenset(("laser_count", "phase_elapsed_frames"))
        feature_names = RISK_FEATURE_NAMES_V2
        categorical_features = RISK_CATEGORICAL_FEATURES_V2
        source_context_conditioning = True
        feature_schema = RISK_FEATURE_SCHEMA_V2
    else:
        excluded_features = frozenset(args.exclude_feature)
        base_feature_names = tuple(
            name for name in RISK_FEATURE_NAMES if name not in excluded_features
        )
        feature_names = (
            (SOURCE_CONTEXT_FEATURE, *base_feature_names)
            if args.include_source_context
            else base_feature_names
        )
        categorical_features = tuple(
            name for name in RISK_CATEGORICAL_FEATURES if name in feature_names
        )
        if args.include_source_context:
            categorical_features = (SOURCE_CONTEXT_FEATURE, *categorical_features)
        source_context_conditioning = args.include_source_context
        feature_schema = (
            RISK_FEATURE_SCHEMA
            if not excluded_features and not args.include_source_context
            else "th06-rl-wine-factual-risk-feature-ablation-v1"
        )
    if not feature_names:
        parser.error("feature ablation cannot exclude every feature")
    source_prefixes = [
        load_first_failure_prefix(
            path,
            expected_scope=scope,  # type: ignore[arg-type]
            expected_executable_sha256=args.retail_sha256,
            expected_native_kernel_sha256=args.native_sha256,
            expected_policy_id=(
                None
                if args.factual_action_audit
                else FROZEN_INCUMBENT_POLICY_ID
            ),
        )
        for path in args.runs
    ]
    if (
        len(source_prefixes) < 3
        or len({prefix.run_id for prefix in source_prefixes})
        != len(source_prefixes)
    ):
        raise SystemExit("risk training requires at least three distinct Wine prefixes")
    factual_action_audit = (
        _validate_factual_action_audit(
            args.factual_action_audit,
            scope=scope,  # type: ignore[arg-type]
            prefixes=source_prefixes,
        )
        if args.factual_action_audit is not None
        else None
    )
    residual_proposal_audit = None
    eligibility_sequences: dict[str, frozenset[int]] | None = None
    if args.residual_proposal_audit is not None:
        residual_proposal_audit, eligibility_sequences = (
            _validate_residual_proposal_audit(
                args.residual_proposal_audit,
                scope=scope,  # type: ignore[arg-type]
                prefixes=source_prefixes,
            )
        )
    prefixes = source_prefixes
    empty_eligibility_runs: list[str] = []
    if eligibility_sequences is not None:
        filtered = []
        for prefix in source_prefixes:
            sequences = eligibility_sequences[prefix.run_id]
            examples = tuple(
                row for row in prefix.examples
                if row.transition.sequence in sequences
            )
            if not examples:
                empty_eligibility_runs.append(prefix.run_id)
                continue
            filtered.append(replace(prefix, examples=examples))
        prefixes = filtered
        if len(prefixes) < 3:
            raise SystemExit(
                "residual proposal filter retained fewer than three run groups"
            )
    all_rows = [row for prefix in prefixes for row in prefix.examples]
    started = time.monotonic()
    oof_scores: dict[tuple[str, int], float] = {}
    folds = []
    retained_folds = []
    for fold, validation in enumerate(prefixes):
        training_rows = [
            row for prefix in prefixes if prefix.run_id != validation.run_id
            for row in prefix.examples
        ]
        model, encoder, positive_weight = _fit(
            training_rows,
            threads=args.threads,
            iterations=args.iterations,
            seed=args.seed + fold,
            feature_names=feature_names,
            categorical_features=categorical_features,
        )
        scores = [float(value) for value in model.predict(encoder.encode(list(validation.examples)))]
        for row, score in zip(validation.examples, scores, strict=True):
            oof_scores[(row.transition.run_id, row.transition.sequence)] = score
        labels = [row.failure_within_120 for row in validation.examples]
        folds.append({
            "validation_run": validation.run_id,
            "train_runs": [prefix.run_id for prefix in prefixes if prefix.run_id != validation.run_id],
            "positive_weight": positive_weight,
            "ranking": _metrics(labels, scores, [False] * len(scores)),
        })
        if args.retain_fold_models:
            retained_folds.append((fold, validation.run_id, model, encoder))
    ordered_scores = [
        oof_scores[(row.transition.run_id, row.transition.sequence)] for row in all_rows
    ]
    try:
        threshold, selection = _select_threshold(
            all_rows,
            ordered_scores,
            minimum_precision=args.minimum_precision,
            minimum_precision_lower_bound=args.minimum_precision_lower_bound,
            maximum_activation_rate=args.maximum_activation_rate,
            minimum_protected_runs=args.minimum_protected_runs,
        )
    except RuntimeError as error:
        diagnostic = {
            "schema": DIAGNOSTIC_SCHEMA,
            "status": "rejected",
            "reason": str(error),
            "scope": list(scope),
            "feature_schema": feature_schema,
            "feature_names": list(feature_names),
            "excluded_features": sorted(excluded_features),
            "source_context_conditioning": source_context_conditioning,
            "feature_profile": args.feature_profile,
            "label_schema": RISK_LABEL_SCHEMA,
            "factual_action_audit": factual_action_audit,
            "residual_proposal_audit": residual_proposal_audit,
            "eligibility_filter": (
                None if residual_proposal_audit is None else {
                    "kind": "offline-ranker-supported-residual-v1",
                    "empty_runs": empty_eligibility_runs,
                    "source_runs": len(source_prefixes),
                    "training_runs": len(prefixes),
                }
            ),
            "runs": [_run_summary(prefix) for prefix in prefixes],
            "rows": len(all_rows),
            "correctable_rows": sum(row.fallback_opportunity for row in all_rows),
            "correctable_positive_rows": sum(
                row.fallback_opportunity and row.failure_within_120
                for row in all_rows
            ),
            "oof": {
                "split": "leave-one-physical-run-out",
                "folds": folds,
                "ranking": _metrics(
                    [row.failure_within_120 for row in all_rows],
                    ordered_scores,
                    [False] * len(all_rows),
                ),
                "frontier": _threshold_frontier(
                    all_rows,
                    ordered_scores,
                    maximum_activation_rate=args.maximum_activation_rate,
                    minimum_protected_runs=args.minimum_protected_runs,
                ),
            },
            "acceptance": {
                "minimum_precision": args.minimum_precision,
                "minimum_precision_lower_bound_95_one_sided": (
                    args.minimum_precision_lower_bound
                ),
                "maximum_activation_rate": args.maximum_activation_rate,
                "minimum_protected_runs": args.minimum_protected_runs,
            },
            "fit_seconds": time.monotonic() - started,
            "process_peak_rss_mib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
        }
        rendered = json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
        if args.diagnostic_output is not None:
            args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostic_output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 2
    active = [
        row.fallback_opportunity and score >= threshold
        for row, score in zip(all_rows, ordered_scores, strict=True)
    ]
    by_run = {}
    for prefix in prefixes:
        selected = [
            (row, score, flag)
            for row, score, flag in zip(all_rows, ordered_scores, active, strict=True)
            if row.transition.run_id == prefix.run_id
        ]
        run_rows = [item[0] for item in selected]
        run_scores = [item[1] for item in selected]
        run_active = [item[2] for item in selected]
        metrics = _metrics(
            [row.failure_within_120 for row in run_rows],
            run_scores,
            run_active,
        )
        leads = [
            row.frames_to_failure for row, flag in zip(run_rows, run_active, strict=True)
            if flag and row.failure_within_120 and row.frames_to_failure is not None
        ]
        metrics["earliest_true_activation_lead_frames"] = max(leads) if leads else None
        by_run[prefix.run_id] = metrics

    final_model, final_encoder, positive_weight = _fit(
        all_rows,
        threads=args.threads,
        iterations=args.iterations,
        seed=args.seed + len(prefixes),
        feature_names=feature_names,
        categorical_features=categorical_features,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    import joblib

    model_path = args.output / "xgboost-risk.joblib"
    encoder_path = args.output / "encoder.json"
    oof_path = args.output / "oof_predictions.jsonl"
    manifest_path = args.output / "manifest.json"
    joblib.dump(final_model, model_path, compress=3)
    encoder_path.write_text(
        json.dumps(final_encoder.manifest(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with oof_path.open("w", encoding="utf-8") as output:
        for row, score, flag in zip(all_rows, ordered_scores, active, strict=True):
            output.write(json.dumps({
                "run_id": row.transition.run_id,
                "sequence": row.transition.sequence,
                "frame": row.transition.frame,
                "source_context": row.transition.source_context,
                "action": row.transition.action,
                "baseline_action": row.transition.baseline_action,
                "failure_within_120": row.failure_within_120,
                "frames_to_failure": row.frames_to_failure,
                "fallback_opportunity": row.fallback_opportunity,
                "score": score,
                "guard_active": flag,
            }, sort_keys=True, separators=(",", ":")) + "\n")
    retained_fold_files = []
    if retained_folds:
        fold_dir = args.output / "audit-folds"
        fold_dir.mkdir()
        for fold, validation_run, model, encoder in retained_folds:
            fold_model_path = fold_dir / f"fold-{fold:02d}-xgboost-risk.joblib"
            fold_encoder_path = fold_dir / f"fold-{fold:02d}-encoder.json"
            joblib.dump(model, fold_model_path, compress=3)
            fold_encoder_path.write_text(
                json.dumps(
                    encoder.manifest(), sort_keys=True, separators=(",", ":"),
                ) + "\n",
                encoding="utf-8",
            )
            retained_fold_files.append({
                "fold": fold,
                "validation_run": validation_run,
                "model_file": str(fold_model_path.relative_to(args.output)),
                "model_sha256": _sha256(fold_model_path),
                "encoder_file": str(fold_encoder_path.relative_to(args.output)),
                "encoder_sha256": _sha256(fold_encoder_path),
            })
    manifest = {
        "schema": TRAINING_SCHEMA,
        "scope": list(scope),
        "feature_schema": feature_schema,
        "feature_names": list(feature_names),
        "categorical_features": list(categorical_features),
        "excluded_features": sorted(excluded_features),
        "source_context_conditioning": source_context_conditioning,
        "feature_profile": args.feature_profile,
        "label_schema": RISK_LABEL_SCHEMA,
        "factual_action_only": True,
        "factual_action_audit": factual_action_audit,
        "residual_proposal_audit": residual_proposal_audit,
        "eligibility_filter": (
            None if residual_proposal_audit is None else {
                "kind": "offline-ranker-supported-residual-v1",
                "empty_runs": empty_eligibility_runs,
                "source_runs": len(source_prefixes),
                "training_runs": len(prefixes),
            }
        ),
        "fallback_action": (
            "native-reactive-baseline-only"
            if residual_proposal_audit is None
            else "offline-ranker-supported-alternative-only"
        ),
        "runs": [_run_summary(prefix) for prefix in prefixes],
        "rows": len(all_rows),
        "failure_kind_counts": dict(Counter(prefix.failure_kind for prefix in prefixes)),
        "oof": {
            "split": "leave-one-physical-run-out",
            "folds": folds,
            "selection": selection,
            "by_run": by_run,
            "predictions_file": oof_path.name,
            "predictions_sha256": _sha256(oof_path),
        },
        "training": {
            "algorithm": "xgboost-reg-squarederror",
            "iterations": args.iterations,
            "seed": args.seed,
            "threads": args.threads,
            "positive_weight": positive_weight,
            "model_file": model_path.name,
            "model_sha256": _sha256(model_path),
            "encoder_file": encoder_path.name,
            "encoder_sha256": _sha256(encoder_path),
            "fit_seconds": time.monotonic() - started,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "retained_fold_models": retained_fold_files,
            "retained_fold_models_purpose": (
                "offline-consensus-audit-only" if retained_fold_files else None
            ),
        },
        "guard": {
            "threshold": threshold,
            "mode": "shadow-before-active",
            "requires_incumbent_baseline_disagreement": True,
            "requires_residual_proposal": residual_proposal_audit is not None,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "manifest_sha256": _sha256(manifest_path),
        "rows": len(all_rows),
        "threshold": threshold,
        "selection": selection,
        "by_run": by_run,
        "model_sha256": _sha256(model_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
