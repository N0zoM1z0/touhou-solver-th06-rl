#!/usr/bin/env python3
"""Export one trained XGBoost ranker as an immutable Wine policy state."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import zlib

from th06_rl.offline_learning import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    load_labeled_run,
)
from th06_rl.offline import load_dataset_index
from th06_rl.policies.offline_ranker import (
    MODEL_CODEC,
    MODEL_SCHEMA,
    NATIVE_SCORER_SCHEMA,
    STATE_SCHEMA,
)


_FEATURE_RE = re.compile(r"f(\d+)")


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


def _portable_tree(raw: dict[str, object]) -> list[list[float | int]]:
    nodes: list[list[float | int] | None] = []

    def build(node: dict[str, object]) -> int:
        index = len(nodes)
        nodes.append(None)
        if "leaf" in node:
            nodes[index] = [-1, 0.0, -1, -1, -1, float(node["leaf"])]
            return index
        split = node.get("split")
        match = _FEATURE_RE.fullmatch(str(split))
        if match is None:
            raise ValueError(f"unsupported XGBoost split feature: {split!r}")
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
        if missing_id == yes_id:
            missing = left
        elif missing_id == no_id:
            missing = right
        else:
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


def _conformance_vectors(model, encoder: dict[str, list[str]]) -> list[dict[str, object]]:
    import numpy as np

    rng = random.Random(6006)
    categorical = set(CATEGORICAL_FEATURES)
    vectors: list[list[float]] = []
    vectors.append([0.0] * len(FEATURE_NAMES))
    vectors.append([-1.0] * len(FEATURE_NAMES))
    for _ in range(6):
        row = []
        for index, name in enumerate(FEATURE_NAMES):
            if name in categorical:
                count = len(encoder[name])
                row.append(float(rng.randrange(-1, max(1, count))))
            elif name in ("player_x", "player_y"):
                row.append(rng.uniform(8.0, 432.0))
            elif name.endswith("_count"):
                row.append(float(rng.randrange(0, 641)))
            elif name == "phase_elapsed_frames":
                row.append(float(rng.randrange(0, 36000)))
            else:
                row.append(rng.uniform(-2.0, 32.0) + index * 0.01)
        vectors.append(row)
    matrix = np.asarray(vectors, dtype=np.float32)
    predictions = model.predict(matrix)
    return [
        {"features": row, "prediction": float(prediction)}
        for row, prediction in zip(vectors, predictions, strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("shadow", "active"), required=True)
    parser.add_argument(
        "--selection",
        choices=("baseline-prior", "support-margin"),
        default="baseline-prior",
    )
    parser.add_argument("--baseline-prior", type=float, default=0.18)
    parser.add_argument("--score-margin", type=float, default=1.0)
    parser.add_argument("--minimum-support", type=int, default=32)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="immutable corpus snapshot required by support-margin",
    )
    parser.add_argument(
        "--native-scorer",
        type=Path,
        help="optional isolated scorer DLL/SO whose hash Wine must reproduce",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline_prior < 0.0:
        parser.error("--baseline-prior must be nonnegative")
    if args.score_margin < 0.0 or args.minimum_support <= 0:
        parser.error("score margin/support parameters must be nonnegative/positive")
    if args.selection == "support-margin" and args.dataset is None:
        parser.error("--selection support-margin requires --dataset")
    manifest = _object(args.manifest)
    encoder = _object(args.encoder)
    if manifest.get("schema") != "th06-rl-offline-cpu-policy-zoo-v1":
        raise SystemExit("unsupported offline CPU policy-zoo manifest")
    if manifest.get("feature_schema") != FEATURE_SCHEMA:
        raise SystemExit("offline manifest feature schema is not deployable")
    features = manifest.get("features")
    if not isinstance(features, dict) or tuple(features.get("names", ())) != FEATURE_NAMES:
        raise SystemExit("offline manifest feature order is not deployable")
    if tuple(features.get("categorical", ())) != CATEGORICAL_FEATURES:
        raise SystemExit("offline manifest categorical schema is not deployable")
    for name in CATEGORICAL_FEATURES:
        values = encoder.get(name)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise SystemExit(f"offline encoder category {name!r} is invalid")
    model_sha = _sha256(args.model)
    results = manifest.get("results")
    if not isinstance(results, dict):
        raise SystemExit("offline manifest has no model results")
    matches = [
        (algorithm, result)
        for algorithm, result in results.items()
        if isinstance(result, dict)
        and result.get("model_sha256") == model_sha
        and result.get("model_file") == args.model.name
    ]
    if len(matches) != 1:
        raise SystemExit("offline model hash/file does not identify one manifest result")
    algorithm, _result = matches[0]
    if algorithm != "xgboost":
        raise SystemExit("the initial portable Wine exporter supports XGBoost only")
    import joblib

    model = joblib.load(args.model)
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    parameters = config["learner"]["learner_model_param"]
    if int(parameters["num_feature"]) != len(FEATURE_NAMES):
        raise SystemExit("XGBoost feature count does not match the manifest")
    if config["learner"]["objective"]["name"] != "reg:squarederror":
        raise SystemExit("only scalar squared-error XGBoost rankers are supported")
    trees = [
        _portable_tree(json.loads(tree))
        for tree in booster.get_dump(dump_format="json")
    ]
    artifact = {
        "schema": MODEL_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "encoder": encoder,
        "base_score": float(parameters["base_score"]),
        "trees": trees,
        "conformance": _conformance_vectors(model, encoder),
    }
    decoded = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    scope = manifest.get("scope")
    if not isinstance(scope, list) or len(scope) != 4:
        raise SystemExit("offline manifest scope is invalid")
    selection: dict[str, object] = {
        "kind": "baseline-prior",
        "value": args.baseline_prior,
    }
    if args.selection == "support-margin":
        split = manifest.get("split")
        if not isinstance(split, dict):
            raise SystemExit("offline manifest split is invalid")
        if split.get("train_rows") != split.get("train_rows_before_cap"):
            raise SystemExit(
                "support-margin export refuses a model whose support was sampled"
            )
        _dataset_manifest, indexed_runs = load_dataset_index(args.dataset)
        by_id = {run.run_id: run for run in indexed_runs}
        train_ids = split.get("train_runs")
        if not isinstance(train_ids, list) or not train_ids:
            raise SystemExit("offline manifest has no training-run identities")
        missing_runs = [run_id for run_id in train_ids if run_id not in by_id]
        if missing_runs:
            raise SystemExit(f"dataset is missing manifest train runs: {missing_runs}")
        exact = manifest.get("view") == "exact-v5"
        support: Counter[tuple[str, str]] = Counter()
        for run_id in train_ids:
            for row in load_labeled_run(
                args.dataset,
                by_id[run_id],
                exact_context_only=exact,
            ):
                support[(row.source_context, row.action)] += 1
        supported = sorted(
            [list(pair) for pair, count in support.items() if count >= args.minimum_support]
        )
        selection = {
            "kind": "support-margin",
            "score_margin": args.score_margin,
            "minimum_support": args.minimum_support,
            "supported_actions": supported,
        }
    state = {
        "schema": STATE_SCHEMA,
        "mode": args.mode,
        "scope": [int(value) for value in scope],
        "view": manifest.get("view"),
        "feature_schema": FEATURE_SCHEMA,
        "selection": selection,
        "source_model": str(args.model.resolve()),
        "source_model_sha256": model_sha,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "dataset": manifest.get("dataset"),
        "model_codec": MODEL_CODEC,
        "portable_model_sha256": hashlib.sha256(decoded).hexdigest(),
        "model_payload": base64.b64encode(zlib.compress(decoded, level=9)).decode("ascii"),
    }
    if args.native_scorer is not None:
        if not args.native_scorer.is_file():
            raise SystemExit(f"native scorer does not exist: {args.native_scorer}")
        state["native_scorer"] = {
            "schema": NATIVE_SCORER_SCHEMA,
            "sha256": _sha256(args.native_scorer),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "mode": args.mode,
        "selection": args.selection,
        "scope": scope,
        "source_model_sha256": model_sha,
        "portable_model_sha256": state["portable_model_sha256"],
        "trees": len(trees),
        "portable_json_bytes": len(decoded),
        "state_bytes": args.output.stat().st_size,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
