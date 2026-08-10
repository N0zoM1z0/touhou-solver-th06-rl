#!/usr/bin/env python3
"""Export a shadow-only fixed-fold Wine risk-consensus policy."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

from scripts.export_wine_risk_guard_policy import (
    TRAINING_SCHEMA,
    _canonical,
    _conformance_vectors,
    _object,
    _portable_tree,
    _precision_lower_bound,
    _sha256,
)
from th06_rl.policies.adaptive import (
    LEGACY_STATE_SCHEMAS as INCUMBENT_LEGACY_SCHEMAS,
    REWARD_VERSION as INCUMBENT_REWARD_VERSION,
    STATE_SCHEMA as INCUMBENT_STATE_SCHEMA,
    unpack_state,
)
from th06_rl.policies.offline_ranker import (
    MODEL_CODEC,
    MODEL_SCHEMA,
    NATIVE_SCORER_SCHEMA,
)
from th06_rl.policies.offline_risk_consensus import MAXIMUM_MEMBERS, STATE_SCHEMA
from th06_rl.wine_risk import (
    RISK_CATEGORICAL_FEATURES,
    RISK_FEATURE_NAMES,
    RISK_FEATURE_SCHEMA,
    RISK_LABEL_SCHEMA,
    risk_feature_contract,
)


def _portable_artifact(
    model,
    encoder: dict[str, object],
    *,
    feature_schema: str,
    feature_names: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> bytes:
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    parameters = config["learner"]["learner_model_param"]
    if int(parameters["num_feature"]) != len(feature_names):
        raise ValueError("risk-consensus XGBoost feature count is inconsistent")
    if config["learner"]["objective"]["name"] != "reg:squarederror":
        raise ValueError("risk-consensus exporter requires squared-error XGBoost")
    artifact = {
        "schema": MODEL_SCHEMA,
        "feature_schema": feature_schema,
        "feature_names": list(feature_names),
        "categorical_features": list(categorical_features),
        "encoder": encoder,
        "base_score": float(parameters["base_score"]),
        "trees": [
            _portable_tree(json.loads(tree))
            for tree in booster.get_dump(dump_format="json")
        ],
        "conformance": _conformance_vectors(
            model,
            encoder,  # type: ignore[arg-type]
            feature_names=feature_names,
            categorical_features=categorical_features,
        ),
    }
    return _canonical(artifact)


def _threshold_row(
    report: dict[str, object], threshold: float, score_key: str,
) -> dict[str, object]:
    totals = report.get("totals")
    fold_sweeps = totals.get("fold_score_sweeps") if isinstance(totals, dict) else None
    minimum = fold_sweeps.get(score_key) if isinstance(fold_sweeps, dict) else None
    if not isinstance(minimum, list):
        raise ValueError("consensus audit has no fold-minimum sweep")
    matches = [
        row for row in minimum
        if isinstance(row, dict)
        and math.isclose(float(row.get("threshold", math.nan)), threshold)
    ]
    if len(matches) != 1:
        raise ValueError("consensus audit threshold evidence is ambiguous")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--incumbent-state", type=Path, required=True)
    parser.add_argument("--audit-replay", type=Path, action="append", required=True)
    parser.add_argument(
        "--fold-subset",
        help=(
            "comma-separated retained-fold indexes to export; defaults to all "
            "folds and must match an audited minimum-score subset"
        ),
    )
    parser.add_argument("--native-scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.training_dir / "manifest.json"
    manifest = _object(manifest_path)
    feature_schema = str(manifest.get("feature_schema", ""))
    try:
        feature_names, categorical_features = risk_feature_contract(feature_schema)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if (
        manifest.get("schema") != TRAINING_SCHEMA
        or tuple(manifest.get("feature_names", ())) != feature_names
        or tuple(manifest.get("categorical_features", ()))
        != categorical_features
        or manifest.get("label_schema") != RISK_LABEL_SCHEMA
    ):
        raise SystemExit("Wine risk-consensus training schema is incompatible")
    if manifest.get("eligibility_filter") is not None:
        raise SystemExit(
            "filtered residual-risk folds require a composite residual "
            "exporter and cannot be exported as a standalone consensus"
        )
    scope = manifest.get("scope")
    if not isinstance(scope, list) or len(scope) != 4:
        raise SystemExit("Wine risk-consensus scope is invalid")
    oof = manifest.get("oof")
    selection = oof.get("selection") if isinstance(oof, dict) else None
    guard = manifest.get("guard")
    training = manifest.get("training")
    folds = training.get("retained_fold_models") if isinstance(training, dict) else None
    if (
        not isinstance(selection, dict)
        or not isinstance(guard, dict)
        or not isinstance(folds, list)
        or len(folds) < 3
        or len(folds) != len(manifest.get("runs", ()))
    ):
        raise SystemExit("retained grouped-fold training evidence is absent")
    threshold = float(guard.get("threshold", math.nan))
    if (
        not math.isfinite(threshold)
        or threshold != float(selection.get("threshold", math.nan))
        or float(selection.get("precision_lower_bound_95_one_sided", 0.0)) < 0.80
        or len(selection.get("protected_runs", ())) < 2
    ):
        raise SystemExit("grouped OOF evidence does not meet the consensus floor")
    if training.get("retained_fold_models_purpose") != "offline-consensus-audit-only":
        raise SystemExit("retained folds have no explicit consensus-audit purpose")

    import joblib

    if args.fold_subset is None:
        selected_indexes = tuple(range(len(folds)))
        score_key = "fold_minimum"
    else:
        try:
            selected_indexes = tuple(sorted({
                int(value) for value in args.fold_subset.split(",")
            }))
        except ValueError as error:
            parser.error(f"invalid --fold-subset: {error}")
        if (
            len(selected_indexes) < 3
            or len(selected_indexes) > MAXIMUM_MEMBERS
            or any(index < 0 or index >= len(folds) for index in selected_indexes)
        ):
            parser.error(
                "fold subset requires between three and "
                f"{MAXIMUM_MEMBERS} valid indexes"
            )
        score_key = "fold_subset_" + "_".join(
            f"{index:02d}" for index in selected_indexes
        )

    if len(selected_indexes) > MAXIMUM_MEMBERS:
        parser.error(
            "retained fold count exceeds the runtime consensus-member limit; "
            "select an audited --fold-subset"
        )

    members = []
    validation_runs = set()
    root = args.training_dir.resolve()
    for index, row in enumerate(folds):
        if not isinstance(row, dict) or int(row.get("fold", -1)) != index:
            raise SystemExit("retained fold order is invalid")
        validation_run = str(row.get("validation_run", ""))
        model_path = (args.training_dir / str(row.get("model_file", ""))).resolve()
        encoder_path = (args.training_dir / str(row.get("encoder_file", ""))).resolve()
        if root not in model_path.parents or root not in encoder_path.parents:
            raise SystemExit("retained fold escaped its training directory")
        if (
            not model_path.is_file()
            or _sha256(model_path) != row.get("model_sha256")
            or not encoder_path.is_file()
            or _sha256(encoder_path) != row.get("encoder_sha256")
        ):
            raise SystemExit("retained fold file/hash contract failed")
        if index not in selected_indexes:
            continue
        validation_runs.add(validation_run)
        encoder = _object(encoder_path)
        for name in categorical_features:
            values = encoder.get(name)
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values))
            ):
                raise SystemExit(f"retained fold encoder {name!r} is invalid")
        decoded = _portable_artifact(
            joblib.load(model_path),
            encoder,
            feature_schema=feature_schema,
            feature_names=feature_names,
            categorical_features=categorical_features,
        )
        members.append({
            "index": len(members),
            "source_fold": index,
            "validation_run": validation_run,
            "source_model": str(model_path),
            "source_model_sha256": _sha256(model_path),
            "model_codec": MODEL_CODEC,
            "portable_model_sha256": hashlib.sha256(decoded).hexdigest(),
            "model_payload": base64.b64encode(
                zlib.compress(decoded, level=9),
            ).decode("ascii"),
        })
    if len(validation_runs) != len(members):
        raise SystemExit("retained folds do not have unique validation runs")

    training_run_ids = {
        str(row.get("run_id")) for row in manifest.get("runs", ())
        if isinstance(row, dict)
    }
    audit_run_ids = set()
    audit_positive = audit_negative = audit_unlabeled = 0
    audit_calls = audit_activations = 0
    audit_reports = []
    for path in args.audit_replay:
        replay = _object(path)
        totals = replay.get("totals")
        runs = replay.get("runs")
        fold_audit = replay.get("retained_fold_audit")
        if (
            replay.get("schema") != "th06-rl-wine-risk-guard-replay-v1"
            or replay.get("passed") is not True
            or not isinstance(totals, dict)
            or not isinstance(runs, list)
            or not isinstance(fold_audit, dict)
            or fold_audit.get("training_manifest_sha256") != _sha256(manifest_path)
            or int(totals.get("recorded_policy_mismatches", -1)) != 0
            or int(totals.get("action_contract_violations", -1)) != 0
        ):
            raise SystemExit("consensus audit replay did not pass its exact contract")
        audited_subsets = fold_audit.get("fold_subsets", ())
        if (
            args.fold_subset is not None
            and list(selected_indexes) not in audited_subsets
        ):
            raise SystemExit("consensus audit did not score the exported fold subset")
        row = _threshold_row(replay, threshold, score_key)
        run_ids = {
            str(run.get("run_id")) for run in runs if isinstance(run, dict)
        }
        if not run_ids or audit_run_ids & run_ids or training_run_ids & run_ids:
            raise SystemExit("consensus audits are duplicated or overlap training")
        audit_run_ids.update(run_ids)
        audit_positive += int(row.get("candidate_positive", 0))
        audit_negative += int(row.get("candidate_negative", 0))
        audit_unlabeled += int(row.get("candidate_unlabeled", 0))
        audit_activations += int(row.get("activations", 0))
        audit_calls += int(totals.get("policy_calls", 0))
        audit_reports.append({
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "run_ids": sorted(run_ids),
        })
    audit_labeled = audit_positive + audit_negative
    if (
        len(audit_run_ids) < 4
        or audit_positive <= 0
        or audit_negative != 0
        or _precision_lower_bound(audit_positive, audit_labeled) < 0.90
        or not audit_calls
        or audit_activations / audit_calls > 0.02
    ):
        raise SystemExit("external fold-consensus shadow evidence is insufficient")

    incumbent_state = _object(args.incumbent_state)
    unpacked = unpack_state(incumbent_state)
    if (
        unpacked.get("schema")
        not in (INCUMBENT_STATE_SCHEMA, *INCUMBENT_LEGACY_SCHEMAS)
        or unpacked.get("reward_version") != INCUMBENT_REWARD_VERSION
    ):
        raise SystemExit("incumbent state is not the supported frozen UCB")
    if not args.native_scorer.is_file():
        raise SystemExit(f"native scorer is absent: {args.native_scorer}")

    state = {
        "schema": STATE_SCHEMA,
        "mode": "shadow",
        "scope": [int(value) for value in scope],
        "threshold": threshold,
        "label_schema": RISK_LABEL_SCHEMA,
        "feature_schema": feature_schema,
        "consensus": {
            "aggregation": "minimum",
            "members": len(members),
            "source_folds": list(selected_indexes),
            "audit_score_key": score_key,
            "fixed_bounded_work": True,
            "factual_incumbent_action_only": True,
            "publication": "incumbent-action-only",
        },
        "source_training_manifest": str(manifest_path.resolve()),
        "source_training_manifest_sha256": _sha256(manifest_path),
        "members": members,
        "incumbent_source": str(args.incumbent_state.resolve()),
        "incumbent_source_sha256": _sha256(args.incumbent_state),
        "incumbent_state_sha256": hashlib.sha256(
            _canonical(incumbent_state),
        ).hexdigest(),
        "incumbent_state": incumbent_state,
        "native_scorer": {
            "schema": NATIVE_SCORER_SCHEMA,
            "sha256": _sha256(args.native_scorer),
        },
        "acceptance": {
            "oof_precision_lower_bound_95_one_sided": selection[
                "precision_lower_bound_95_one_sided"
            ],
            "oof_protected_runs": selection["protected_runs"],
            "external_audit_reports": audit_reports,
            "external_run_ids": sorted(audit_run_ids),
            "external_policy_calls": audit_calls,
            "external_activations": audit_activations,
            "external_candidate_positive": audit_positive,
            "external_candidate_negative": audit_negative,
            "external_candidate_unlabeled": audit_unlabeled,
            "external_precision": audit_positive / audit_labeled,
            "external_precision_lower_bound_95_one_sided": (
                _precision_lower_bound(audit_positive, audit_labeled)
            ),
            "external_activation_rate": audit_activations / audit_calls,
            "active_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(state) + b"\n")
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "mode": "shadow",
        "members": len(members),
        "source_folds": list(selected_indexes),
        "threshold": threshold,
        "external_runs": len(audit_run_ids),
        "external_candidate_positive": audit_positive,
        "external_candidate_negative": audit_negative,
        "external_precision_lower_bound_95_one_sided": (
            _precision_lower_bound(audit_positive, audit_labeled)
        ),
        "state_bytes": args.output.stat().st_size,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
