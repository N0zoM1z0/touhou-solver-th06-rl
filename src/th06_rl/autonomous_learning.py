"""Factual episode labeling and grouped ridge fitting for autonomous RL."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path

from .learning_features import FEATURE_SCHEMA, candidate_vector, feature_names


TRANSITION_SCHEMA = "th06-rl-transition-v6"
BEHAVIOR_POLICY = "uniform-safe-exploration-v1"
MODEL_SCHEMA = "autonomous-grouped-linear-q-v1"
POLICY_STATE_SCHEMA = "autonomous-linear-q-policy-v1"


@dataclass(frozen=True)
class LearningSample:
    episode_id: str
    sequence: int
    action: str
    baseline_action: str
    behavior_probability: float
    features: tuple[float, ...]
    target_return: float


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transition_rows(run_dir: Path, manifest: dict[str, object]):
    expected_sequence = 0
    expected_records = 0
    observed_records = 0
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise TypeError("corpus manifest shard list is invalid")
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("stream") != "transitions":
            continue
        name = str(shard.get("path", ""))
        if not name or Path(name).name != name:
            raise ValueError("unsafe transition shard path")
        path = run_dir / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(shard.get("compressed_bytes", -1)):
            raise ValueError(f"transition shard size mismatch: {path}")
        if _sha256(path) != shard.get("sha256"):
            raise ValueError(f"transition shard digest mismatch: {path}")
        expected_records += int(shard.get("records", 0))
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("transition row is not an object")
                if row.get("schema_version") != TRANSITION_SCHEMA:
                    raise ValueError("autonomous learner requires transition v6")
                if int(row.get("sequence", -1)) != expected_sequence:
                    raise ValueError("transition sequence is not contiguous")
                expected_sequence += 1
                observed_records += 1
                yield row
    manifest_records = manifest.get("records")
    recorded = (
        int(manifest_records.get("transitions", -1))
        if isinstance(manifest_records, dict)
        else -1
    )
    if observed_records != expected_records or observed_records != recorded:
        raise ValueError("transition record count mismatch")


def _validate_run(run_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    run = _object(run_dir / "run.json")
    manifest = _object(run_dir / "manifest.json")
    if manifest.get("complete") is not True or int(manifest.get("dropped_records", -1)):
        raise ValueError(f"incomplete physical corpus: {run_dir}")
    schemas = run.get("schemas")
    if not isinstance(schemas, dict) or schemas.get("transition") != TRANSITION_SCHEMA:
        raise ValueError("autonomous learner requires a v6 physical corpus")
    outcome = manifest.get("run_outcome")
    if not isinstance(outcome, dict):
        raise TypeError("physical run outcome is absent")
    for field in (
        "background_reactivations",
        "capture_failures",
        "corpus_failures",
        "infrastructure_failures",
        "trace_failures",
    ):
        if int(outcome.get(field, -1)) != 0:
            raise ValueError(f"physical corpus has infrastructure failure: {field}")
    if outcome.get("corpus_failure") is not None:
        raise ValueError("physical corpus writer failed")
    metadata = run.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("physical run metadata is absent")
    return run, manifest


def _outcome(row: dict[str, object]) -> dict[str, object]:
    value = row.get("outcome_terms")
    if not isinstance(value, dict):
        raise TypeError("transition outcome is absent")
    return value


def _immediate_reward(outcome: dict[str, object]) -> float:
    reward = 1.0
    reward -= 100.0 * float(outcome.get("life_lost") is True)
    reward -= 25.0 * float(outcome.get("control_dead_end") is True)
    return reward


def _terminal(outcome: dict[str, object]) -> bool:
    return bool(
        outcome.get("life_lost") is True
        or outcome.get("control_dead_end") is True
        or outcome.get("authority_lost") is True
        or outcome.get("bomb_used") is True
    )


def _expected_probability(
    *,
    action: str,
    baseline: str,
    legal: tuple[str, ...],
    exploration_probability: float,
) -> float:
    if len(legal) == 1 or exploration_probability == 0.0:
        return float(action == baseline)
    probability = exploration_probability / len(legal)
    if action == baseline:
        probability += 1.0 - exploration_probability
    return probability


def label_episode(
    rows: list[dict[str, object]],
    *,
    episode_id: str,
    exploration_probability: float,
    return_horizon: int,
    gamma: float,
    observation_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> tuple[list[LearningSample], Counter[str]]:
    if not 0.0 <= exploration_probability <= 1.0:
        raise ValueError("exploration probability must be in [0, 1]")
    if return_horizon <= 0 or not 0.0 < gamma <= 1.0:
        raise ValueError("return horizon and gamma are invalid")
    samples = []
    excluded: Counter[str] = Counter()
    for index, row in enumerate(rows):
        outcome = _outcome(row)
        if outcome.get("bomb_used") is True:
            raise ValueError("Bomb-bearing transition is never learning eligible")
        if outcome.get("authority_lost") is True:
            raise ValueError("authority loss is an infrastructure stop")
        action = row.get("published_action")
        proposal = row.get("proposed_action")
        legal_raw = row.get("legal_actions")
        legal = (
            tuple(str(value) for value in legal_raw)
            if isinstance(legal_raw, list)
            else ()
        )
        baseline = str(row.get("baseline_action", ""))
        if (
            row.get("learning_eligible") is not True
            or action is None
            or proposal != action
        ):
            excluded["unpublished-or-ineligible"] += 1
            continue
        action = str(action)
        if not legal or action not in legal or baseline not in legal:
            raise ValueError("published action or baseline escaped the safe set")
        if row.get("policy_id") != BEHAVIOR_POLICY:
            excluded["different-behavior-policy"] += 1
            continue
        probability = float(row.get("behavior_probability", 0.0))
        expected = _expected_probability(
            action=action,
            baseline=baseline,
            legal=legal,
            exploration_probability=exploration_probability,
        )
        if not math.isclose(probability, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("recorded action propensity does not match generation")
        context = row.get("policy_context")
        if not isinstance(context, dict):
            excluded["missing-policy-context"] += 1
            continue
        observation = context.get("observation_features")
        actions = context.get("action_features")
        if not isinstance(observation, list) or not isinstance(actions, list):
            excluded["missing-adapter-features"] += 1
            continue
        try:
            vector = candidate_vector(
                observation_features=observation,
                action_features=actions,
                action=action,
                baseline_action=baseline,
                current_action=str(context.get("current_action", "")),
                observation_names=observation_names,
                action_names=action_names,
            )
        except (TypeError, ValueError):
            excluded["invalid-adapter-features"] += 1
            continue

        target = 0.0
        discount = 1.0
        elapsed = 0
        complete = False
        expected_sequence = int(row["sequence"])
        for future in rows[index:]:
            if int(future.get("sequence", -1)) != expected_sequence:
                break
            future_outcome = _outcome(future)
            step = int(future_outcome.get("elapsed_frames", 0))
            if step != 1 or future_outcome.get("authority_lost") is True:
                break
            target += discount * _immediate_reward(future_outcome)
            elapsed += step
            expected_sequence += 1
            if _terminal(future_outcome):
                complete = True
                break
            if elapsed >= return_horizon:
                complete = True
                break
            discount *= gamma**step
        if not complete:
            excluded["censored-return-window"] += 1
            continue
        samples.append(LearningSample(
            episode_id=episode_id,
            sequence=int(row["sequence"]),
            action=action,
            baseline_action=baseline,
            behavior_probability=probability,
            features=vector,
            target_return=target,
        ))
    return samples, excluded


def load_episode(
    run_dir: Path,
    *,
    exploration_probability: float,
    return_horizon: int,
    gamma: float,
    observation_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> tuple[list[LearningSample], dict[str, object]]:
    run_dir = run_dir.resolve()
    run, manifest = _validate_run(run_dir)
    rows = list(_transition_rows(run_dir, manifest))
    episode_id = str(run.get("run_id", run_dir.name))
    samples, excluded = label_episode(
        rows,
        episode_id=episode_id,
        exploration_probability=exploration_probability,
        return_horizon=return_horizon,
        gamma=gamma,
        observation_names=observation_names,
        action_names=action_names,
    )
    metadata = run["metadata"]
    assert isinstance(metadata, dict)
    return samples, {
        "episode_id": episode_id,
        "run_dir": str(run_dir),
        "run_sha256": _sha256(run_dir / "run.json"),
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
        "transition_rows": len(rows),
        "learning_samples": len(samples),
        "excluded": dict(excluded),
        "provenance": {
            key: metadata.get(key)
            for key in (
                "code_commit",
                "executable_sha256",
                "native_kernel_sha256",
                "input_backend",
            )
        },
    }


def _ridge_model(samples: list[LearningSample], *, alpha: float, clip: float):
    import numpy as np

    if not samples:
        raise ValueError("cannot fit an empty sample set")
    x = np.asarray([sample.features for sample in samples], dtype=np.float64)
    y = np.asarray([sample.target_return for sample in samples], dtype=np.float64)
    weights = np.asarray(
        [min(clip, 1.0 / sample.behavior_probability) for sample in samples],
        dtype=np.float64,
    )
    total = float(weights.sum())
    mean = (x * weights[:, None]).sum(axis=0) / total
    variance = (((x - mean) ** 2) * weights[:, None]).sum(axis=0) / total
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    target_mean = float((y * weights).sum() / total)
    centered = y - target_mean
    gram = (normalized.T * weights) @ normalized / total
    rhs = (normalized.T @ (weights * centered)) / total
    gram.flat[:: gram.shape[0] + 1] += alpha
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": target_mean,
    }


def predict_model(model: dict[str, object], features: tuple[float, ...]) -> float:
    mean = model["mean"]
    scale = model["scale"]
    coefficients = model["coefficients"]
    if not all(isinstance(value, list) for value in (mean, scale, coefficients)):
        raise TypeError("linear model vectors are invalid")
    if not len(features) == len(mean) == len(scale) == len(coefficients):
        raise ValueError("linear model feature length mismatch")
    return float(model["intercept"]) + sum(
        (value - float(center)) / float(width) * float(coefficient)
        for value, center, width, coefficient in zip(
            features, mean, scale, coefficients, strict=True
        )
    )


def _rmse(actual: list[float], predicted: list[float]) -> float:
    if not actual or len(actual) != len(predicted):
        raise ValueError("RMSE vectors must be nonempty and equal")
    return math.sqrt(sum(
        (left - right) ** 2
        for left, right in zip(actual, predicted, strict=True)
    ) / len(actual))


def fit_grouped_ridge(
    train: list[LearningSample],
    validation: list[LearningSample],
    *,
    observation_names: tuple[str, ...],
    action_names: tuple[str, ...],
    alpha: float,
    propensity_clip: float,
    minimum_train_groups: int,
    minimum_validation_groups: int,
    minimum_train_rows: int,
    minimum_non_baseline_rows: int,
    minimum_action_samples: int,
    minimum_action_ess: float,
    required_rmse_ratio: float,
    margin_rmse_fraction: float,
) -> dict[str, object]:
    if alpha <= 0.0 or propensity_clip < 1.0:
        raise ValueError("ridge alpha and propensity clip are invalid")
    train_groups = sorted({sample.episode_id for sample in train})
    validation_groups = sorted({sample.episode_id for sample in validation})
    if set(train_groups) & set(validation_groups):
        raise ValueError("episode appeared in both train and validation")
    if not train or not validation:
        raise ValueError("grouped fit requires train and validation samples")
    names = feature_names(observation_names, action_names)
    if any(len(sample.features) != len(names) for sample in (*train, *validation)):
        raise ValueError("learning sample feature schema mismatch")

    fold_count = min(5, len(train_groups))
    committee = []
    if fold_count >= 2:
        group_fold = {
            group: index % fold_count for index, group in enumerate(train_groups)
        }
        for fold in range(fold_count):
            rows = [
                sample for sample in train
                if group_fold[sample.episode_id] != fold
            ]
            committee.append(_ridge_model(
                rows, alpha=alpha, clip=propensity_clip
            ))
    full = _ridge_model(train, alpha=alpha, clip=propensity_clip)
    actual = [sample.target_return for sample in validation]
    predicted = [predict_model(full, sample.features) for sample in validation]
    train_weights = [
        min(propensity_clip, 1.0 / sample.behavior_probability)
        for sample in train
    ]
    train_mean = sum(
        sample.target_return * weight
        for sample, weight in zip(train, train_weights, strict=True)
    ) / sum(train_weights)
    constant = [train_mean] * len(validation)
    model_rmse = _rmse(actual, predicted)
    constant_rmse = _rmse(actual, constant)
    rmse_ratio = (
        model_rmse / constant_rmse
        if constant_rmse > 1e-12
        else 0.0
        if model_rmse <= 1e-12
        else 1e12
    )

    by_action: dict[str, list[LearningSample]] = {}
    for sample in train:
        by_action.setdefault(sample.action, []).append(sample)
    support = {}
    for action, rows in sorted(by_action.items()):
        weights = [
            min(propensity_clip, 1.0 / sample.behavior_probability)
            for sample in rows
        ]
        ess = sum(weights) ** 2 / sum(weight * weight for weight in weights)
        support[action] = {
            "samples": len(rows),
            "clipped_propensity_ess": ess,
            "authorized": (
                len(rows) >= minimum_action_samples and ess >= minimum_action_ess
            ),
        }
    non_baseline = sum(
        sample.action != sample.baseline_action for sample in train
    )
    supported_actions = sum(
        bool(row["authorized"]) for row in support.values()
    )
    gates = {
        "minimum_train_groups": len(train_groups) >= minimum_train_groups,
        "minimum_validation_groups": (
            len(validation_groups) >= minimum_validation_groups
        ),
        "minimum_train_rows": len(train) >= minimum_train_rows,
        "minimum_non_baseline_rows": non_baseline >= minimum_non_baseline_rows,
        "multiple_supported_actions": supported_actions >= 2,
        "heldout_rmse_improves_constant": rmse_ratio <= required_rmse_ratio,
        "committee_available": len(committee) >= 2,
    }
    return {
        "schema": POLICY_STATE_SCHEMA,
        "mode": "shadow",
        "feature_schema": FEATURE_SCHEMA,
        "observation_feature_names": list(observation_names),
        "action_feature_names": list(action_names),
        "feature_names": list(names),
        "model": {
            "schema": MODEL_SCHEMA,
            "full": full,
            "committee": committee,
        },
        "support": support,
        "selection": {
            "minimum_action_samples": minimum_action_samples,
            "minimum_action_ess": minimum_action_ess,
            "score_margin": margin_rmse_fraction * model_rmse,
            "committee_rule": "unanimous-best-and-margin",
        },
        "authorization": {
            "fit_gates": gates,
            "fit_eligible": all(gates.values()),
            "active_canary": None,
        },
        "fit_report": {
            "train_groups": train_groups,
            "validation_groups": validation_groups,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "non_baseline_train_rows": non_baseline,
            "propensity_clip": propensity_clip,
            "ridge_alpha": alpha,
            "heldout_rmse": model_rmse,
            "heldout_constant_rmse": constant_rmse,
            "heldout_rmse_ratio": rmse_ratio,
            "required_rmse_ratio": required_rmse_ratio,
            "margin_rmse_fraction": margin_rmse_fraction,
        },
    }
