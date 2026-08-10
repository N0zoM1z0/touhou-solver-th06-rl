#!/usr/bin/env python3
"""Export an accepted Wine risk model around the frozen UCB incumbent."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import random
import re
import zlib

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
from th06_rl.policies.offline_risk_guard import STATE_SCHEMA
from th06_rl.wine_risk import (
    RISK_CATEGORICAL_FEATURES,
    RISK_FEATURE_NAMES,
    RISK_FEATURE_SCHEMA,
    RISK_LABEL_SCHEMA,
    risk_feature_contract,
)


TRAINING_SCHEMA = "th06-rl-wine-risk-guard-training-v1"
_FEATURE_RE = re.compile(r"f(\d+)")
ONE_SIDED_95_Z = 1.6448536269514722


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _precision_lower_bound(true_positives: int, activations: int) -> float:
    if activations <= 0:
        return 0.0
    probability = true_positives / activations
    z2 = ONE_SIDED_95_Z * ONE_SIDED_95_Z
    denominator = 1.0 + z2 / activations
    center = probability + z2 / (2.0 * activations)
    spread = ONE_SIDED_95_Z * math.sqrt(
        probability * (1.0 - probability) / activations
        + z2 / (4.0 * activations * activations)
    )
    return (center - spread) / denominator


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _portable_tree(raw: dict[str, object]) -> list[list[float | int]]:
    nodes: list[list[float | int] | None] = []

    def build(node: dict[str, object]) -> int:
        index = len(nodes)
        nodes.append(None)
        if "leaf" in node:
            nodes[index] = [-1, 0.0, -1, -1, -1, float(node["leaf"])]
            return index
        match = _FEATURE_RE.fullmatch(str(node.get("split")))
        if match is None:
            raise ValueError(f"unsupported XGBoost split: {node.get('split')!r}")
        children = node.get("children")
        if not isinstance(children, list) or len(children) != 2:
            raise TypeError("XGBoost split must have two children")
        by_id = {int(child["nodeid"]): child for child in children}
        yes_id = int(node["yes"])
        no_id = int(node["no"])
        missing_id = int(node["missing"])
        if set(by_id) != {yes_id, no_id}:
            raise ValueError("XGBoost child IDs do not match yes/no branches")
        left = build(by_id[yes_id])
        right = build(by_id[no_id])
        missing = left if missing_id == yes_id else right if missing_id == no_id else -1
        if missing < 0:
            raise ValueError("XGBoost missing branch is not a child")
        nodes[index] = [
            int(match.group(1)),
            float(node["split_condition"]),
            left,
            right,
            missing,
            0.0,
        ]
        return index

    if build(raw) != 0 or any(node is None for node in nodes):
        raise RuntimeError("portable XGBoost tree construction failed")
    return [node for node in nodes if node is not None]


def _conformance_vectors(
    model,
    encoder: dict[str, list[str]],
    *,
    feature_names: tuple[str, ...] = RISK_FEATURE_NAMES,
    categorical_features: tuple[str, ...] = RISK_CATEGORICAL_FEATURES,
):
    import numpy as np

    rng = random.Random(6066)
    categorical = set(categorical_features)
    vectors = [
        [0.0] * len(feature_names),
        [-1.0] * len(feature_names),
    ]
    for _ in range(8):
        row = []
        for index, name in enumerate(feature_names):
            if name in categorical:
                row.append(float(rng.randrange(-1, max(1, len(encoder[name])))))
            elif name in ("player_x", "player_y", "incumbent_final_x", "baseline_final_x"):
                row.append(rng.uniform(8.0, 376.0))
            elif name in ("incumbent_final_y", "baseline_final_y"):
                row.append(rng.uniform(16.0, 432.0))
            elif name.endswith("_count") or name in ("bullet_count", "laser_count"):
                row.append(float(rng.randrange(0, 641)))
            elif name.startswith(("legal_", "hard_", "matches_")):
                row.append(float(rng.randrange(0, 2)))
            else:
                row.append(rng.uniform(-4.0, 512.0) + index * 0.001)
        vectors.append(row)
    predictions = model.predict(np.asarray(vectors, dtype=np.float32))
    return [
        {"features": row, "prediction": float(prediction)}
        for row, prediction in zip(vectors, predictions, strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--incumbent-state", type=Path, required=True)
    parser.add_argument("--mode", choices=("shadow", "active"), default="shadow")
    parser.add_argument(
        "--shadow-replay",
        type=Path,
        help="strict unseen physical shadow replay required before active export",
    )
    parser.add_argument("--native-scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "active" and args.shadow_replay is None:
        parser.error("active export requires --shadow-replay")
    if args.mode == "shadow" and args.shadow_replay is not None:
        parser.error("shadow export does not consume active-promotion evidence")

    manifest_path = args.training_dir / "manifest.json"
    manifest = _object(manifest_path)
    if manifest.get("schema") != TRAINING_SCHEMA:
        raise SystemExit("unsupported Wine risk training manifest")
    if manifest.get("eligibility_filter") is not None:
        raise SystemExit(
            "filtered residual-risk training requires a composite residual "
            "exporter and cannot be exported as a standalone risk guard"
        )
    feature_schema = str(manifest.get("feature_schema", ""))
    try:
        feature_names, categorical_features = risk_feature_contract(feature_schema)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if (
        tuple(manifest.get("feature_names", ())) != feature_names
        or tuple(manifest.get("categorical_features", ())) != categorical_features
        or manifest.get("label_schema") != RISK_LABEL_SCHEMA
    ):
        raise SystemExit("Wine risk training feature/label schema is not deployable")
    scope = manifest.get("scope")
    if not isinstance(scope, list) or len(scope) != 4:
        raise SystemExit("Wine risk training scope is invalid")
    oof = manifest.get("oof")
    selection = oof.get("selection") if isinstance(oof, dict) else None
    guard = manifest.get("guard")
    if not isinstance(selection, dict) or not isinstance(guard, dict):
        raise SystemExit("Wine risk training acceptance evidence is absent")
    precision = float(selection.get("precision", 0.0))
    lower_bound = float(selection.get("precision_lower_bound_95_one_sided", 0.0))
    minimum_precision = float(selection.get("minimum_precision", 1.0))
    minimum_lower_bound = float(
        selection.get("minimum_precision_lower_bound_95_one_sided", 1.0)
    )
    if (
        precision < minimum_precision
        or lower_bound < minimum_lower_bound
        or minimum_precision < 0.60
        or minimum_lower_bound < 0.60
        or int(selection.get("minimum_protected_runs", 0)) < 2
        or len(selection.get("protected_runs", ())) < 2
    ):
        raise SystemExit("Wine risk training did not pass conservative OOF acceptance")
    threshold = float(guard.get("threshold", float("nan")))
    if threshold != float(selection.get("threshold", float("nan"))):
        raise SystemExit("Wine risk threshold is inconsistent with OOF selection")

    training = manifest.get("training")
    if not isinstance(training, dict):
        raise SystemExit("Wine risk training files are absent")
    model_path = args.training_dir / str(training.get("model_file", ""))
    encoder_path = args.training_dir / str(training.get("encoder_file", ""))
    if (
        not model_path.is_file()
        or _sha256(model_path) != training.get("model_sha256")
        or not encoder_path.is_file()
        or _sha256(encoder_path) != training.get("encoder_sha256")
    ):
        raise SystemExit("Wine risk training file/hash contract failed")
    encoder = _object(encoder_path)
    for name in categorical_features:
        values = encoder.get(name)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
        ):
            raise SystemExit(f"Wine risk encoder category {name!r} is invalid")

    import joblib

    model = joblib.load(model_path)
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    parameters = config["learner"]["learner_model_param"]
    if int(parameters["num_feature"]) != len(feature_names):
        raise SystemExit("Wine risk XGBoost feature count is inconsistent")
    if config["learner"]["objective"]["name"] != "reg:squarederror":
        raise SystemExit("Wine risk exporter only supports squared-error XGBoost")
    trees = [
        _portable_tree(json.loads(tree))
        for tree in booster.get_dump(dump_format="json")
    ]
    artifact = {
        "schema": MODEL_SCHEMA,
        "feature_schema": feature_schema,
        "feature_names": list(feature_names),
        "categorical_features": list(categorical_features),
        "encoder": encoder,
        "base_score": float(parameters["base_score"]),
        "trees": trees,
        "conformance": _conformance_vectors(
            model,
            encoder,
            feature_names=feature_names,
            categorical_features=categorical_features,
        ),
    }
    decoded = _canonical(artifact)

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
        "mode": args.mode,
        "scope": [int(value) for value in scope],
        "threshold": threshold,
        "label_schema": RISK_LABEL_SCHEMA,
        "feature_schema": feature_schema,
        "source_training_manifest": str(manifest_path.resolve()),
        "source_training_manifest_sha256": _sha256(manifest_path),
        "source_model": str(model_path.resolve()),
        "source_model_sha256": _sha256(model_path),
        "model_codec": MODEL_CODEC,
        "portable_model_sha256": hashlib.sha256(decoded).hexdigest(),
        "model_payload": base64.b64encode(zlib.compress(decoded, level=9)).decode("ascii"),
        "incumbent_source": str(args.incumbent_state.resolve()),
        "incumbent_source_sha256": _sha256(args.incumbent_state),
        "incumbent_state_sha256": hashlib.sha256(_canonical(incumbent_state)).hexdigest(),
        "incumbent_state": incumbent_state,
        "native_scorer": {
            "schema": NATIVE_SCORER_SCHEMA,
            "sha256": _sha256(args.native_scorer),
        },
        "acceptance": {
            "precision": precision,
            "precision_lower_bound_95_one_sided": lower_bound,
            "protected_runs": selection["protected_runs"],
            "activation_rate": selection["activation_rate"],
        },
    }
    if args.mode == "active":
        assert args.shadow_replay is not None
        replay = _object(args.shadow_replay)
        totals = replay.get("totals")
        replay_runs = replay.get("runs")
        if (
            replay.get("schema") != "th06-rl-wine-risk-guard-replay-v1"
            or replay.get("passed") is not True
            or not isinstance(totals, dict)
            or not isinstance(replay_runs, list)
            or not replay_runs
        ):
            raise SystemExit("active export shadow replay did not pass")
        shadow_path = Path(str(replay.get("state", "")))
        if (
            not shadow_path.is_file()
            or _sha256(shadow_path) != replay.get("state_sha256")
        ):
            raise SystemExit("active export shadow state evidence is absent or changed")
        shadow_state = _object(shadow_path)
        if (
            shadow_state.get("mode") != "shadow"
            or shadow_state.get("source_model_sha256") != state["source_model_sha256"]
            or shadow_state.get("portable_model_sha256")
            != state["portable_model_sha256"]
            or shadow_state.get("incumbent_state_sha256")
            != state["incumbent_state_sha256"]
            or float(shadow_state.get("threshold", math.nan)) != threshold
        ):
            raise SystemExit("active export does not match the validated shadow policy")
        training_run_ids = {
            str(run.get("run_id"))
            for run in manifest.get("runs", ())
            if isinstance(run, dict)
        }
        shadow_run_ids = {
            str(run.get("run_id"))
            for run in replay_runs
            if isinstance(run, dict)
        }
        if (
            not shadow_run_ids
            or training_run_ids & shadow_run_ids
            or int(totals.get("recorded_incumbent_mismatches", -1)) != 0
            or int(totals.get("shadow_action_mismatches", -1)) != 0
        ):
            raise SystemExit("active export shadow evidence is not an unseen exact replay")
        candidate_positive = int(totals.get("candidate_positive", 0))
        candidate_negative = int(totals.get("candidate_negative", 0))
        labeled_candidates = candidate_positive + candidate_negative
        candidates = int(totals.get("candidates", 0))
        policy_calls = int(totals.get("policy_calls", 0))
        shadow_precision = (
            candidate_positive / labeled_candidates if labeled_candidates else 0.0
        )
        shadow_lower_bound = _precision_lower_bound(
            candidate_positive, labeled_candidates,
        )
        shadow_activation_rate = candidates / policy_calls if policy_calls else 1.0
        if (
            shadow_precision < 0.65
            or shadow_lower_bound < 0.60
            or shadow_activation_rate > 0.02
            or candidate_positive <= 0
        ):
            raise SystemExit("active export unseen shadow metrics are insufficient")
        state["shadow_validation"] = {
            "schema": replay["schema"],
            "report": str(args.shadow_replay.resolve()),
            "report_sha256": _sha256(args.shadow_replay),
            "state_sha256": replay["state_sha256"],
            "run_ids": sorted(shadow_run_ids),
            "policy_calls": policy_calls,
            "candidates": candidates,
            "candidate_positive": candidate_positive,
            "candidate_negative": candidate_negative,
            "candidate_unlabeled": int(totals.get("candidate_unlabeled", 0)),
            "precision": shadow_precision,
            "precision_lower_bound_95_one_sided": shadow_lower_bound,
            "activation_rate": shadow_activation_rate,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(state) + b"\n")
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "mode": args.mode,
        "scope": scope,
        "threshold": threshold,
        "source_model_sha256": state["source_model_sha256"],
        "portable_model_sha256": state["portable_model_sha256"],
        "incumbent_source_sha256": state["incumbent_source_sha256"],
        "native_scorer_sha256": state["native_scorer"]["sha256"],
        "trees": len(trees),
        "portable_json_bytes": len(decoded),
        "state_bytes": args.output.stat().st_size,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
