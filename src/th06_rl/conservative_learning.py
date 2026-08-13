"""Episode-grouped conservative fitted-Q learning from physical Wine rows."""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import zlib

from .autonomous_learning import (
    BEHAVIOR_POLICY,
    _expected_probability,
    _object,
    _transition_rows,
    _validate_run,
)
from .learning_features import (
    TREE_FEATURE_SCHEMA,
    tree_candidate_vector,
    tree_feature_names,
)
from .actions import ACTION_NAMES
from .policies.offline_ranker import MODEL_SCHEMA
from .th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)


STATE_SCHEMA = "autonomous-conservative-fqi-policy-v2"
FIT_REPORT_SCHEMA = "autonomous-conservative-fqi-fit-v2"
MODEL_CODEC = "zlib-base64-json-v1"
NATIVE_SCORER_SCHEMA = "th06-rl-native-xgboost-scorer-v1"
_FEATURE_RE = re.compile(r"f(\d+)")


@dataclass(frozen=True)
class FactualStep:
    episode_id: str
    raw_index: int
    sequence: int
    action: str
    baseline_action: str
    behavior_probability: float
    vector: tuple[float, ...]
    legal_actions: tuple[str, ...]
    candidate_vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class NStepCost:
    state: FactualStep
    observed_hit_cost: float
    next_state: FactualStep | None


def _context(row: dict[str, object]) -> dict[str, object]:
    value = row.get("policy_context")
    if not isinstance(value, dict):
        raise TypeError("transition has no policy context")
    return value


def _vector(row: dict[str, object], action: str) -> tuple[float, ...]:
    context = _context(row)
    return tree_candidate_vector(
        observation_features=context.get("observation_features"),
        action_features=context.get("action_features"),
        action=action,
        baseline_action=str(row.get("baseline_action", "")),
        current_action=str(context.get("current_action", "")),
        observation_names=OBSERVATION_FEATURE_NAMES,
        action_names=ACTION_FEATURE_NAMES,
    )


def load_complete_episode(
    run_dir: Path,
    *,
    exploration_probability: float,
    n_step_frames: int,
) -> tuple[list[NStepCost], dict[str, object]]:
    """Load factual action states and cross passive/HIT gaps without invention."""
    run_dir = run_dir.resolve()
    run, manifest = _validate_run(run_dir)
    outcome = manifest.get("run_outcome")
    if (
        manifest.get("stage_trajectory_complete") is not True
        or not isinstance(outcome, dict)
        or outcome.get("stage_completed") is not True
        or not isinstance(outcome.get("physical_hits"), int)
    ):
        raise ValueError("conservative FQI requires a complete physical Stage")
    rows = list(_transition_rows(run_dir, manifest))
    episode_id = str(run.get("run_id", run_dir.name))
    steps = []
    excluded: Counter[str] = Counter()
    for raw_index, row in enumerate(rows):
        if row.get("learning_eligible") is not True:
            excluded["ineligible"] += 1
            continue
        if row.get("policy_id") != BEHAVIOR_POLICY:
            excluded["behavior-policy"] += 1
            continue
        legal_raw = row.get("legal_actions")
        action = row.get("published_action")
        baseline = str(row.get("baseline_action", ""))
        if not isinstance(legal_raw, list) or not isinstance(action, str):
            excluded["missing-action"] += 1
            continue
        legal = tuple(str(value) for value in legal_raw)
        if action not in legal or baseline not in legal:
            raise ValueError("factual action or incumbent escaped the native set")
        probability = float(row.get("behavior_probability", 0.0))
        expected = _expected_probability(
            action=action,
            baseline=baseline,
            legal=legal,
            exploration_probability=exploration_probability,
        )
        if not math.isclose(probability, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("recorded action propensity does not match generation")
        candidates = tuple(_vector(row, candidate) for candidate in legal)
        steps.append(FactualStep(
            episode_id=episode_id,
            raw_index=raw_index,
            sequence=int(row["sequence"]),
            action=action,
            baseline_action=baseline,
            behavior_probability=probability,
            vector=_vector(row, action),
            legal_actions=legal,
            candidate_vectors=candidates,
        ))
    if not steps:
        raise ValueError("complete Stage has no eligible factual decisions")
    next_step_after = [None] * (len(rows) + 1)
    cursor = len(steps) - 1
    upcoming = None
    for raw_index in range(len(rows), -1, -1):
        while cursor >= 0 and steps[cursor].raw_index >= raw_index:
            upcoming = steps[cursor]
            cursor -= 1
        next_step_after[raw_index] = upcoming
    samples = []
    for step in steps:
        elapsed = 0
        cost = 0.0
        stop_index = step.raw_index
        for stop_index in range(step.raw_index, len(rows)):
            terms = rows[stop_index].get("outcome_terms")
            if not isinstance(terms, dict):
                raise TypeError("transition outcome terms are absent")
            if terms.get("bomb_used") is True:
                raise ValueError("Bomb-bearing transition is never learning eligible")
            if terms.get("authority_lost") is True:
                raise ValueError("complete training Stage contains authority loss")
            frame_step = int(terms.get("elapsed_frames", 0))
            if frame_step < 0:
                raise ValueError("negative physical transition duration")
            cost += float(terms.get("life_lost") is True)
            elapsed += frame_step
            if elapsed >= n_step_frames:
                break
        next_state = (
            next_step_after[min(len(rows), stop_index + 1)]
            if stop_index + 1 < len(rows)
            else None
        )
        samples.append(NStepCost(step, cost, next_state))
    return samples, {
        "episode_id": episode_id,
        "run_dir": str(run_dir),
        "transitions": len(rows),
        "eligible_steps": len(steps),
        "physical_hits": int(outcome["physical_hits"]),
        "excluded": dict(excluded),
    }


def _standardization(samples: list[NStepCost]):
    import numpy as np

    matrix = np.asarray([sample.state.vector for sample in samples], dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _kmeans(rows, *, count: int, iterations: int, seed: int):
    import numpy as np

    if len(rows) <= count:
        return rows.copy()
    generator = np.random.default_rng(seed)
    centers = [rows[generator.integers(len(rows))]]
    nearest = ((rows - centers[0]) ** 2).mean(axis=1)
    for _ in range(1, count):
        index = int(np.argmax(nearest))
        centers.append(rows[index])
        nearest = np.minimum(nearest, ((rows - rows[index]) ** 2).mean(axis=1))
    centers = np.asarray(centers, dtype=np.float64)
    for _ in range(iterations):
        distances = ((rows[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
        assignments = distances.argmin(axis=1)
        updated = centers.copy()
        for index in range(len(centers)):
            selected = rows[assignments == index]
            if len(selected):
                updated[index] = selected.mean(axis=0)
        if np.allclose(updated, centers, rtol=0.0, atol=1e-7):
            break
        centers = updated
    return centers


def fit_local_support(
    train: list[NStepCost],
    validation: list[NStepCost],
    *,
    prototypes_per_action: int,
    quantile: float,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    import numpy as np

    mean, scale = _standardization(train)
    groups = []
    for action_index, action in enumerate(ACTION_NAMES):
        rows = np.asarray([
            sample.state.vector for sample in train if sample.state.action == action
        ], dtype=np.float64)
        if not len(rows):
            raise ValueError(f"training episodes have no factual action {action}")
        groups.append(_kmeans(
            (rows - mean) / scale,
            count=prototypes_per_action,
            iterations=12,
            seed=seed + action_index,
        ))
    distances = []
    by_action: Counter[str] = Counter()
    for sample in validation:
        action_index = ACTION_NAMES.index(sample.state.action)
        row = (np.asarray(sample.state.vector) - mean) / scale
        distance = float(((groups[action_index] - row) ** 2).mean(axis=1).min())
        distances.append(distance)
        by_action[sample.state.action] += 1
    if not distances:
        raise ValueError("validation episodes have no local-support observations")
    threshold = float(np.quantile(np.asarray(distances), quantile))
    artifact = {
        "schema": "autonomous-local-prototype-support-v1",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "prototypes": [group.tolist() for group in groups],
        "threshold": threshold,
        "threshold_source": {
            "kind": "heldout-factual-distance-quantile",
            "quantile": quantile,
            "rows": len(distances),
        },
    }
    report = {
        "validation_rows": len(distances),
        "validation_coverage": sum(value <= threshold for value in distances) / len(distances),
        "threshold": threshold,
        "distances": {
            "mean": float(np.mean(distances)),
            "p95": float(np.quantile(distances, 0.95)),
            "p99": float(np.quantile(distances, 0.99)),
            "max": max(distances),
        },
        "validation_action_rows": dict(sorted(by_action.items())),
        "prototypes_per_action": [len(group) for group in groups],
    }
    return artifact, report


def _portable_tree(raw: dict[str, object]) -> list[list[float | int]]:
    nodes: list[list[float | int] | None] = []

    def build(node: dict[str, object]) -> int:
        index = len(nodes)
        nodes.append(None)
        if "leaf" in node:
            nodes[index] = [-1, 0.0, -1, -1, -1, float(node["leaf"])]
            return index
        match = _FEATURE_RE.fullmatch(str(node.get("split")))
        children = node.get("children")
        if match is None or not isinstance(children, list) or len(children) != 2:
            raise ValueError("unsupported XGBoost tree node")
        by_id = {int(child["nodeid"]): child for child in children}
        yes_id, no_id, missing_id = (
            int(node[name]) for name in ("yes", "no", "missing")
        )
        left, right = build(by_id[yes_id]), build(by_id[no_id])
        nodes[index] = [
            int(match.group(1)),
            float(node["split_condition"]),
            left,
            right,
            left if missing_id == yes_id else right,
            0.0,
        ]
        return index

    build(raw)
    if any(node is None for node in nodes):
        raise RuntimeError("portable tree construction failed")
    return [node for node in nodes if node is not None]


def _export_model(
    model,
    conformance_rows,
    *,
    feature_schema: str = TREE_FEATURE_SCHEMA,
    feature_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    import numpy as np

    booster = model.get_booster()
    config = json.loads(booster.save_config())
    parameters = config["learner"]["learner_model_param"]
    names = feature_names or tree_feature_names(
        OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES
    )
    rows = np.asarray(conformance_rows, dtype=np.float32)
    predictions = model.predict(rows)
    return {
        "schema": MODEL_SCHEMA,
        "feature_schema": feature_schema,
        "feature_names": list(names),
        "base_score": float(parameters["base_score"].strip("[]")),
        "trees": [
            _portable_tree(json.loads(tree))
            for tree in booster.get_dump(dump_format="json")
        ],
        "conformance": [
            {"features": row.tolist(), "prediction": float(prediction)}
            for row, prediction in zip(rows, predictions, strict=True)
        ],
    }


def _encoded_model(artifact: dict[str, object]) -> dict[str, object]:
    raw = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    return {
        "codec": MODEL_CODEC,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": base64.b64encode(zlib.compress(raw, level=9)).decode(),
    }


def _support_distances(artifact: dict[str, object], rows, actions):
    import numpy as np

    mean = np.asarray(artifact["mean"])
    scale = np.asarray(artifact["scale"])
    prototypes = [np.asarray(group) for group in artifact["prototypes"]]
    result = []
    for row, action in zip(rows, actions, strict=True):
        normalized = (np.asarray(row) - mean) / scale
        result.append(float(((prototypes[action] - normalized) ** 2).mean(axis=1).min()))
    return result


def _target_sha256(values) -> str:
    """Commit a fitted target vector without embedding the training corpus."""
    import numpy as np

    array = np.asarray(values, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _backup_layout(samples: list[NStepCost]):
    rows = []
    actions = []
    slices = []
    for sample in samples:
        start = len(rows)
        if sample.next_state is not None:
            rows.extend(sample.next_state.candidate_vectors)
            actions.extend(
                ACTION_NAMES.index(action)
                for action in sample.next_state.legal_actions
            )
        slices.append((start, len(rows), sample.next_state))
    return rows, actions, slices


def _bellman_backup(
    *,
    factual_cost,
    layout,
    models,
    support: dict[str, object],
    uncertainty_scale: float,
):
    """Apply one frozen conservative backup to either train or held-out rows."""
    import numpy as np

    rows, actions, slices = layout
    updated = np.asarray(factual_cost, dtype=np.float32).copy()
    if not rows:
        return updated, 0
    matrix = np.asarray(rows, dtype=np.float32)
    predictions = np.asarray([model.predict(matrix) for model in models])
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    distances = _support_distances(support, rows, actions)
    supported_backups = 0
    for index, (start, stop, next_state) in enumerate(slices):
        if next_state is None or stop == start:
            continue
        permitted = [
            offset for offset in range(start, stop)
            if next_state.legal_actions[offset - start]
            == next_state.baseline_action
            or distances[offset] <= float(support["threshold"])
        ]
        if not permitted:
            permitted = [
                start + next_state.legal_actions.index(
                    next_state.baseline_action
                )
            ]
        updated[index] += min(
            float(mean[offset] + uncertainty_scale * std[offset])
            for offset in permitted
        )
        supported_backups += len(permitted)
    return updated, supported_backups


def fit_conservative_fqi(
    train: list[NStepCost],
    validation: list[NStepCost],
    *,
    ensemble_members: int,
    bellman_iterations: int,
    trees_per_iteration: int,
    propensity_clip: float,
    prototypes_per_action: int,
    support_quantile: float,
    uncertainty_scale: float,
    seed: int,
    threads: int,
    native_scorer_sha256: str,
    compatible_native_scorer_sha256: tuple[str, ...] = (),
    n_step_frames: int = 60,
) -> dict[str, object]:
    import numpy as np
    from xgboost import XGBRegressor

    if len({sample.state.episode_id for sample in train}) < 3:
        raise ValueError("conservative FQI needs at least three training episodes")
    if len({sample.state.episode_id for sample in validation}) < 2:
        raise ValueError("conservative FQI needs at least two validation episodes")
    support, support_report = fit_local_support(
        train,
        validation,
        prototypes_per_action=prototypes_per_action,
        quantile=support_quantile,
        seed=seed,
    )
    x = np.asarray([sample.state.vector for sample in train], dtype=np.float32)
    factual_cost = np.asarray(
        [sample.observed_hit_cost for sample in train], dtype=np.float32
    )
    weights = np.sqrt(np.asarray([
        min(propensity_clip, 1.0 / sample.state.behavior_probability)
        for sample in train
    ], dtype=np.float32))
    groups = sorted({sample.state.episode_id for sample in train})
    group_indices = {
        group: np.asarray([
            index for index, sample in enumerate(train)
            if sample.state.episode_id == group
        ])
        for group in groups
    }
    generator = random.Random(seed)
    bootstrap_counts = []
    for _member in range(ensemble_members):
        chosen = [generator.choice(groups) for _ in groups]
        counts = Counter(chosen)
        bootstrap_counts.append(np.asarray([
            counts[sample.state.episode_id] for sample in train
        ], dtype=np.float32))

    if n_step_frames <= 0:
        raise ValueError("n-step frame horizon must be positive")
    train_layout = _backup_layout(train)
    validation_layout = _backup_layout(validation)
    targets = factual_cost.copy()
    validation_factual_cost = np.asarray(
        [sample.observed_hit_cost for sample in validation], dtype=np.float32
    )
    validation_targets = validation_factual_cost.copy()
    models = []
    iteration_report = []

    def fit_ensemble(fit_targets, *, fit_round: int):
        fitted = []
        for member in range(ensemble_members):
            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=trees_per_iteration,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=16.0,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                reg_alpha=0.05,
                tree_method="hist",
                n_jobs=threads,
                random_state=seed + fit_round * 100 + member,
            )
            member_weights = weights * bootstrap_counts[member]
            model.fit(x, fit_targets, sample_weight=member_weights)
            fitted.append(model)
        return fitted

    for iteration in range(bellman_iterations):
        fitted_target_sha256 = _target_sha256(targets)
        models = fit_ensemble(targets, fit_round=iteration)
        updated, supported_backups = _bellman_backup(
            factual_cost=factual_cost,
            layout=train_layout,
            models=models,
            support=support,
            uncertainty_scale=uncertainty_scale,
        )
        validation_updated, validation_supported_backups = _bellman_backup(
            factual_cost=validation_factual_cost,
            layout=validation_layout,
            models=models,
            support=support,
            uncertainty_scale=uncertainty_scale,
        )
        delta = float(np.mean(np.abs(updated - targets)))
        targets = updated
        validation_targets = validation_updated
        iteration_report.append({
            "iteration": iteration + 1,
            "mean_absolute_target_delta": delta,
            "target_mean": float(targets.mean()),
            "target_max": float(targets.max()),
            "supported_backup_candidates": supported_backups,
            "validation_supported_backup_candidates": validation_supported_backups,
            "fitted_target_sha256": fitted_target_sha256,
            "next_target_sha256": _target_sha256(targets),
            "next_nominal_horizon_frames": n_step_frames * (iteration + 2),
        })

    # The loop's final models were fitted to the previous target generation.
    # Refit once to the last reported target before export.  This explicit step
    # prevents the generation-2 off-by-one model/fit-report mismatch.
    models = fit_ensemble(targets, fit_round=bellman_iterations)
    exported_target_sha256 = _target_sha256(targets)

    validation_x = np.asarray(
        [sample.state.vector for sample in validation], dtype=np.float32
    )
    validation_predictions = np.asarray([
        model.predict(validation_x) for model in models
    ])
    prediction = validation_predictions.mean(axis=0)
    rmse = float(np.sqrt(np.mean((prediction - validation_targets) ** 2)))
    constant = float(np.sqrt(np.mean(
        (validation_targets - targets.mean()) ** 2
    )))
    conformance_indices = np.linspace(
        0, len(train) - 1, min(8, len(train)), dtype=int
    )
    conformance = x[conformance_indices]
    encoded_models = [
        _encoded_model(_export_model(model, conformance)) for model in models
    ]
    gates = {
        "train_groups": len(groups) >= 3,
        "validation_groups": len({sample.state.episode_id for sample in validation}) >= 2,
        "train_has_hits": sum(sample.observed_hit_cost for sample in train) > 0.0,
        "validation_has_hits": float(validation_factual_cost.sum()) > 0.0,
        "all_actions_observed": all(
            any(sample.state.action == action for sample in train)
            for action in ACTION_NAMES
        ),
        "heldout_local_support": support_report["validation_coverage"] >= 0.95,
        "finite_validation_prediction": math.isfinite(rmse),
        "native_scorer_bound": len(native_scorer_sha256) == 64,
    }
    names = tree_feature_names(OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES)
    return {
        "schema": STATE_SCHEMA,
        "mode": "shadow",
        "feature_schema": TREE_FEATURE_SCHEMA,
        "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "feature_names": list(names),
        "models": encoded_models,
        "native_scorer": {
            "schema": NATIVE_SCORER_SCHEMA,
            "sha256": native_scorer_sha256,
            "compatible_sha256": sorted(set((
                native_scorer_sha256, *compatible_native_scorer_sha256
            ))),
        },
        "support": support,
        "selection": {
            "rule": "unanimous-pessimistic-cost-improvement",
            "uncertainty_scale": uncertainty_scale,
            "active_override_budget": 128,
        },
        "authorization": {
            "fit_gates": gates,
            "fit_eligible": all(gates.values()),
            "active_canary": None,
        },
        "fit_report": {
            "schema": FIT_REPORT_SCHEMA,
            "train_groups": groups,
            "validation_groups": sorted({
                sample.state.episode_id for sample in validation
            }),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_n_step_hit_cost": float(factual_cost.sum()),
            "validation_n_step_hit_cost": float(validation_factual_cost.sum()),
            "exported_target_sha256": exported_target_sha256,
            "exported_target_iteration": bellman_iterations + 1,
            "exported_nominal_horizon_frames": n_step_frames * (
                bellman_iterations + 1
            ),
            "matched_horizon_validation_target_mean": float(
                validation_targets.mean()
            ),
            "matched_horizon_validation_rmse": rmse,
            "matched_horizon_constant_rmse": constant,
            "iterations": iteration_report,
            "support": support_report,
        },
    }
