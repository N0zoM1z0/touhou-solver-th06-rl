#!/usr/bin/env python3
"""Replay a shadow Wine risk guard against strict physical corpus prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from th06_rl.policies.adaptive import AdaptivePolicy
from th06_rl.policies.offline_ranker import NATIVE_SCORER_ENV
from th06_rl.policies.offline_risk_guard import OfflineRiskGuardPolicy
from th06_rl.policy_api import PolicyContext
from th06_rl.wine_risk import (
    RISK_CATEGORICAL_FEATURES,
    RISK_FEATURE_NAMES,
    RISK_FEATURE_SCHEMA,
    _frame,
    _stream_rows,
    load_first_failure_prefix,
    risk_features_for_context,
    risk_feature_contract,
)


ONE_SIDED_95_Z = 1.6448536269514722


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _precision_lower_bound(true_positives: int, labeled_activations: int) -> float | None:
    if labeled_activations <= 0:
        return None
    probability = true_positives / labeled_activations
    z2 = ONE_SIDED_95_Z * ONE_SIDED_95_Z
    denominator = 1.0 + z2 / labeled_activations
    center = probability + z2 / (2.0 * labeled_activations)
    spread = ONE_SIDED_95_Z * math.sqrt(
        probability * (1.0 - probability) / labeled_activations
        + z2 / (4.0 * labeled_activations * labeled_activations)
    )
    return (center - spread) / denominator


def _score_sweep(
    scored: list[dict[str, object]],
    *,
    thresholds: tuple[float, ...],
    policy_calls: int,
) -> list[dict[str, object]]:
    rows = []
    for threshold in thresholds:
        active = [row for row in scored if float(row["score"]) >= threshold]
        positive = sum(row["label"] is True for row in active)
        negative = sum(row["label"] is False for row in active)
        unlabeled = sum(row["label"] is None for row in active)
        labeled = positive + negative
        rows.append({
            "threshold": threshold,
            "activations": len(active),
            "activation_rate": len(active) / policy_calls if policy_calls else None,
            "candidate_positive": positive,
            "candidate_negative": negative,
            "candidate_unlabeled": unlabeled,
            "labeled_precision": positive / labeled if labeled else None,
            "labeled_precision_lower_bound_95_one_sided": (
                _precision_lower_bound(positive, labeled)
            ),
            "activated_runs": sorted({str(row["run_id"]) for row in active}),
            "positive_runs": sorted({
                str(row["run_id"]) for row in active if row["label"] is True
            }),
            "negative_runs": sorted({
                str(row["run_id"]) for row in active if row["label"] is False
            }),
        })
    return rows


def _load_audit_folds(training_dir: Path):
    manifest_path = training_dir / "manifest.json"
    manifest = _object(manifest_path)
    feature_schema = str(manifest.get("feature_schema", ""))
    try:
        feature_names, categorical_features = risk_feature_contract(feature_schema)
    except ValueError as error:
        raise ValueError("retained-fold training manifest is incompatible") from error
    if (
        manifest.get("schema") != "th06-rl-wine-risk-guard-training-v1"
        or tuple(manifest.get("feature_names", ())) != feature_names
        or tuple(manifest.get("categorical_features", ()))
        != categorical_features
    ):
        raise ValueError("retained-fold training manifest is incompatible")
    training = manifest.get("training")
    folds = training.get("retained_fold_models") if isinstance(training, dict) else None
    if not isinstance(folds, list) or len(folds) < 3:
        raise ValueError("retained-fold audit requires at least three fold models")
    import joblib

    loaded = []
    root = training_dir.resolve()
    for expected_fold, row in enumerate(folds):
        if not isinstance(row, dict) or int(row.get("fold", -1)) != expected_fold:
            raise ValueError("retained-fold manifest order is invalid")
        model_path = (training_dir / str(row.get("model_file", ""))).resolve()
        encoder_path = (training_dir / str(row.get("encoder_file", ""))).resolve()
        if root not in model_path.parents or root not in encoder_path.parents:
            raise ValueError("retained-fold file escaped its training directory")
        if (
            not model_path.is_file()
            or _sha256(model_path) != row.get("model_sha256")
            or not encoder_path.is_file()
            or _sha256(encoder_path) != row.get("encoder_sha256")
        ):
            raise ValueError("retained-fold file/hash contract failed")
        raw_encoder = _object(encoder_path)
        encoder = {}
        for name in categorical_features:
            values = raw_encoder.get(name)
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"retained-fold encoder {name!r} is invalid")
            encoder[name] = {value: index for index, value in enumerate(values)}
        loaded.append((
            row,
            joblib.load(model_path),
            encoder,
            feature_schema,
            feature_names,
            categorical_features,
        ))
    return manifest_path, loaded


def _audit_fold_scores(
    scored: list[dict[str, object]], loaded_folds, fold_subsets=(),
) -> None:
    if not scored:
        return
    import numpy as np

    per_row = [[] for _row in scored]
    for fold_index, (
        _manifest,
        model,
        encoder,
        _feature_schema,
        feature_names,
        categorical_features,
    ) in enumerate(loaded_folds):
        categorical = set(categorical_features)
        matrix = np.empty(
            (len(scored), len(feature_names)), dtype=np.float32,
        )
        for row_index, row in enumerate(scored):
            features = row["_features"]
            if not isinstance(features, dict):
                raise TypeError("retained-fold audit features are absent")
            for column, name in enumerate(feature_names):
                value = features[name]
                matrix[row_index, column] = (
                    encoder[name].get(str(value), -1)
                    if name in categorical else float(value)
                )
        predictions = model.predict(matrix)
        for row, value in zip(per_row, predictions, strict=True):
            row.append(float(value))
        for row, value in zip(scored, predictions, strict=True):
            row[f"fold_member_{fold_index:02d}"] = float(value)
    for row, values in zip(scored, per_row, strict=True):
        ordered = sorted(values)
        row["fold_minimum"] = ordered[0]
        row["fold_lower_quartile"] = ordered[(len(ordered) - 1) // 4]
        row["fold_median"] = ordered[len(ordered) // 2]
        row["fold_mean"] = sum(ordered) / len(ordered)
        row["fold_maximum"] = ordered[-1]
        row["final_and_fold_minimum"] = min(float(row["score"]), ordered[0])
        for subset in fold_subsets:
            name = "fold_subset_" + "_".join(f"{index:02d}" for index in subset)
            row[name] = min(
                float(row[f"fold_member_{index:02d}"]) for index in subset
            )


def _fold_score_sweeps(
    scored: list[dict[str, object]],
    *,
    thresholds: tuple[float, ...],
    policy_calls: int,
) -> dict[str, list[dict[str, object]]]:
    output = {}
    names = [
        "fold_minimum",
        "fold_lower_quartile",
        "fold_median",
        "fold_mean",
        "fold_maximum",
        "final_and_fold_minimum",
    ]
    if scored:
        names.extend(sorted(
            name for name in scored[0]
            if name.startswith(("fold_member_", "fold_subset_"))
        ))
    for name in names:
        if not scored or name not in scored[0]:
            continue
        projected = [dict(row, score=row[name]) for row in scored]
        output[name] = _score_sweep(
            projected, thresholds=thresholds, policy_calls=policy_calls,
        )
    return output


def _context(row: dict[str, object], decision: dict[str, object]) -> PolicyContext:
    scope = row.get("scope")
    policy = row.get("policy_context")
    if not isinstance(scope, dict) or not isinstance(policy, dict):
        raise TypeError("replay transition scope/policy context is absent")
    hard_actions = decision.get("hard_actions")
    if not isinstance(hard_actions, list):
        raise TypeError("replay frame has no native Hard evaluations")
    legal = row.get("legal_actions")
    hard = policy.get("hard_admissible_actions")
    if not isinstance(legal, list) or not isinstance(hard, list):
        raise TypeError("replay transition has no native action sets")
    return PolicyContext(
        frame=_frame(row.get("snapshot_ref")),
        scope=tuple(int(scope[name]) for name in (
            "difficulty", "character", "shot_type", "stage",
        )),  # type: ignore[arg-type]
        source_context=str(scope.get("phase_id", "")),
        baseline_action=str(row.get("baseline_action", "")),
        locally_admissible_actions=tuple(str(value) for value in legal),
        player_x=float(policy["player_x"]),
        player_y=float(policy["player_y"]),
        power=int(policy["power"]),
        bullet_count=int(policy["bullet_count"]),
        laser_count=int(policy["laser_count"]),
        hard_action_count=int(policy["hard_action_count"]),
        exploration_rate=0.0,
        current_action=str(policy["current_action"]),
        hard_admissible_actions=tuple(str(value) for value in hard),
        phase_elapsed_frames=int(policy["phase_elapsed_frames"]),
        hard_action_evaluations=tuple(
            (
                str(value[0]),
                None if value[1] is None else float(value[1]),
                float(value[2]),
                float(value[3]),
            )
            for value in hard_actions
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--native-scorer", type=Path, required=True)
    parser.add_argument("--retail-sha256", required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--scope", default="3/0/0/6")
    parser.add_argument(
        "--expect-recorded-actions",
        choices=("state", "incumbent"),
        default="state",
        help="whether corpus actions came from this state or its UCB incumbent",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=[],
        help=(
            "additional final-model thresholds to audit over every incumbent/"
            "baseline disagreement; may be repeated"
        ),
    )
    parser.add_argument(
        "--audit-folds-training-dir",
        type=Path,
        help=(
            "score retained leave-one-run-out models for offline consensus "
            "audit only; does not alter replayed policy actions"
        ),
    )
    parser.add_argument(
        "--fold-subset",
        action="append",
        default=[],
        help=(
            "comma-separated retained-fold indexes whose minimum score should "
            "be audited; requires --audit-folds-training-dir"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = tuple(int(value) for value in args.scope.split("/"))
    if len(scope) != 4:
        parser.error("scope must contain four integers")
    if not args.native_scorer.is_file():
        parser.error("native replay scorer is absent")

    source_state = _object(args.state)
    mode = source_state.get("mode")
    if mode not in ("shadow", "active"):
        raise SystemExit("risk-guard replay state mode is invalid")
    configured_threshold = float(source_state.get("threshold", math.nan))
    requested_thresholds = tuple(sorted({configured_threshold, *args.threshold}))
    if any(not math.isfinite(value) for value in requested_thresholds):
        raise SystemExit("risk-guard replay thresholds must be finite")
    native_contract = source_state.get("native_scorer")
    if not isinstance(native_contract, dict):
        raise SystemExit("shadow state has no production native-scorer contract")
    incumbent_state = source_state.get("incumbent_state")
    if not isinstance(incumbent_state, dict):
        raise SystemExit("shadow state has no incumbent")
    runtime_state = json.loads(json.dumps(source_state))
    runtime_state["native_scorer"]["sha256"] = _sha256(args.native_scorer)
    os.environ[NATIVE_SCORER_ENV] = str(args.native_scorer.resolve())
    fold_manifest_path = None
    loaded_folds = []
    if args.audit_folds_training_dir is not None:
        fold_manifest_path, loaded_folds = _load_audit_folds(
            args.audit_folds_training_dir,
        )
    fold_subsets = []
    for raw in args.fold_subset:
        try:
            subset = tuple(sorted({int(value) for value in raw.split(",")}))
        except ValueError as error:
            parser.error(f"invalid --fold-subset {raw!r}: {error}")
        if (
            not loaded_folds
            or len(subset) < 2
            or any(index < 0 or index >= len(loaded_folds) for index in subset)
        ):
            parser.error("fold subsets require two or more valid retained-fold indexes")
        fold_subsets.append(subset)

    run_reports = []
    total_policy_calls = 0
    total_recorded_mismatches = 0
    total_recorded_policy_mismatches = 0
    total_shadow_mismatches = 0
    total_action_contract_violations = 0
    total_candidates = 0
    total_positive_candidates = 0
    total_negative_candidates = 0
    total_unlabeled_candidates = 0
    all_scored_disagreements: list[dict[str, object]] = []
    started = time.perf_counter()
    for run_dir in args.runs:
        prefix = load_first_failure_prefix(
            run_dir,
            expected_scope=scope,  # type: ignore[arg-type]
            expected_executable_sha256=args.retail_sha256,
            expected_native_kernel_sha256=args.native_sha256,
            expected_policy_id=None,
        )
        manifest = _object(run_dir / "manifest.json")
        decisions = {
            int(row["sequence"]): row["decision"]
            for row in _stream_rows(run_dir, manifest, "frames")
            if isinstance(row.get("decision"), dict)
        }
        examples = {
            example.transition.sequence: example for example in prefix.examples
        }
        incumbent = AdaptivePolicy()
        incumbent.import_state(incumbent_state)
        guard = OfflineRiskGuardPolicy()
        guard.import_state(runtime_state)
        calls = 0
        recorded_mismatches = []
        recorded_policy_mismatches = []
        shadow_mismatches = []
        action_contract_violations = []
        candidates = []
        scored_disagreements = []
        for row in _stream_rows(run_dir, manifest, "transitions"):
            sequence = int(row.get("sequence", -1))
            decision = decisions.get(sequence)
            if not isinstance(decision, dict) or decision.get("reason") != "ok":
                continue
            context = _context(row, decision)
            direct = incumbent.decide(context)
            risk_score = (
                guard.scorer.predict_many([
                    guard._encoded_row(context, direct.action),
                ])[0]
                if direct.action != context.baseline_action
                else None
            )
            shadow = guard.decide(context)
            recorded = row.get("proposed_action")
            calls += 1
            if direct.action != recorded and len(recorded_mismatches) < 20:
                recorded_mismatches.append({
                    "sequence": sequence,
                    "frame": context.frame,
                    "recorded": recorded,
                    "direct_incumbent": direct.action,
                })
            candidate = shadow.policy_id.endswith("-candidate")
            if risk_score is not None:
                example = examples.get(sequence)
                scored_disagreements.append({
                    "run_id": prefix.run_id,
                    "sequence": sequence,
                    "frame": context.frame,
                    "score": risk_score,
                    "label": (
                        None if example is None else example.failure_within_120
                    ),
                    "_features": guard.features_for_context(
                        context, direct.action,
                    ),
                })
            if mode == "shadow" and shadow.action != direct.action and len(shadow_mismatches) < 20:
                shadow_mismatches.append({
                    "sequence": sequence,
                    "frame": context.frame,
                    "direct_incumbent": direct.action,
                    "shadow": shadow.action,
                })
            expected_action = (
                context.baseline_action
                if mode == "active" and candidate
                else direct.action
            )
            expected_recorded = (
                shadow.action
                if args.expect_recorded_actions == "state"
                else direct.action
            )
            if (
                expected_recorded != recorded
                and len(recorded_policy_mismatches) < 20
            ):
                recorded_policy_mismatches.append({
                    "sequence": sequence,
                    "frame": context.frame,
                    "expected": expected_recorded,
                    "recorded": recorded,
                })
            if (
                shadow.action != expected_action
                and len(action_contract_violations) < 20
            ):
                action_contract_violations.append({
                    "sequence": sequence,
                    "frame": context.frame,
                    "mode": mode,
                    "candidate": candidate,
                    "expected": expected_action,
                    "observed": shadow.action,
                })
            if candidate:
                example = examples.get(sequence)
                candidates.append({
                    "sequence": sequence,
                    "frame": context.frame,
                    "source_context": context.source_context,
                    "incumbent_action": direct.action,
                    "baseline_action": context.baseline_action,
                    "score": risk_score,
                    "hard_action_count": context.hard_action_count,
                    "bullet_count": context.bullet_count,
                    "laser_count": context.laser_count,
                    "player_x": context.player_x,
                    "player_y": context.player_y,
                    "label": (
                        None if example is None else example.failure_within_120
                    ),
                    "frames_to_failure": (
                        None if example is None else example.frames_to_failure
                    ),
                })
        candidate_positive = sum(row["label"] is True for row in candidates)
        candidate_negative = sum(row["label"] is False for row in candidates)
        candidate_unlabeled = sum(row["label"] is None for row in candidates)
        _audit_fold_scores(
            scored_disagreements, loaded_folds, fold_subsets,
        )
        total_policy_calls += calls
        total_recorded_mismatches += len(recorded_mismatches)
        total_recorded_policy_mismatches += len(recorded_policy_mismatches)
        total_shadow_mismatches += len(shadow_mismatches)
        total_action_contract_violations += len(action_contract_violations)
        total_candidates += len(candidates)
        total_positive_candidates += candidate_positive
        total_negative_candidates += candidate_negative
        total_unlabeled_candidates += candidate_unlabeled
        all_scored_disagreements.extend(scored_disagreements)
        run_reports.append({
            "run_id": prefix.run_id,
            "policy_calls": calls,
            "recorded_incumbent_mismatches": recorded_mismatches,
            "recorded_policy_mismatches": recorded_policy_mismatches,
            "shadow_action_mismatches": shadow_mismatches,
            "action_contract_violations": action_contract_violations,
            "candidates": len(candidates),
            "candidate_positive": candidate_positive,
            "candidate_negative": candidate_negative,
            "candidate_unlabeled": candidate_unlabeled,
            "candidate_examples": candidates[:40],
            "scored_incumbent_baseline_disagreements": len(scored_disagreements),
            "score_sweep": _score_sweep(
                scored_disagreements,
                thresholds=requested_thresholds,
                policy_calls=calls,
            ),
            "fold_score_sweeps": _fold_score_sweeps(
                scored_disagreements,
                thresholds=requested_thresholds,
                policy_calls=calls,
            ),
            "guard_metrics": guard.metrics(),
        })

    report = {
        "schema": "th06-rl-wine-risk-guard-replay-v1",
        "state": str(args.state.resolve()),
        "state_sha256": _sha256(args.state),
        "production_native_scorer_sha256": native_contract.get("sha256"),
        "replay_native_scorer": str(args.native_scorer.resolve()),
        "replay_native_scorer_sha256": _sha256(args.native_scorer),
        "scope": list(scope),
        "mode": mode,
        "expect_recorded_actions": args.expect_recorded_actions,
        "retained_fold_audit": (
            None if fold_manifest_path is None else {
                "training_manifest": str(fold_manifest_path.resolve()),
                "training_manifest_sha256": _sha256(fold_manifest_path),
                "folds": len(loaded_folds),
                "fold_subsets": [list(value) for value in fold_subsets],
                "deployment_authority": False,
            }
        ),
        "runs": run_reports,
        "totals": {
            "runs": len(run_reports),
            "policy_calls": total_policy_calls,
            "recorded_incumbent_mismatches": total_recorded_mismatches,
            "recorded_policy_mismatches": total_recorded_policy_mismatches,
            "shadow_action_mismatches": total_shadow_mismatches,
            "action_contract_violations": total_action_contract_violations,
            "candidates": total_candidates,
            "candidate_positive": total_positive_candidates,
            "candidate_negative": total_negative_candidates,
            "candidate_unlabeled": total_unlabeled_candidates,
            "scored_incumbent_baseline_disagreements": len(
                all_scored_disagreements
            ),
            "score_sweep": _score_sweep(
                all_scored_disagreements,
                thresholds=requested_thresholds,
                policy_calls=total_policy_calls,
            ),
            "fold_score_sweeps": _fold_score_sweeps(
                all_scored_disagreements,
                thresholds=requested_thresholds,
                policy_calls=total_policy_calls,
            ),
            "replay_seconds": time.perf_counter() - started,
        },
        "passed": (
            total_recorded_policy_mismatches == 0
            and total_action_contract_violations == 0
            and all(
                run["guard_metrics"]["scorer_backend"] == "native-batch"
                for run in run_reports
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "passed": report["passed"],
        "totals": report["totals"],
    }, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
