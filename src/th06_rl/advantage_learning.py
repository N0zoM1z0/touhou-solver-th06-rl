"""Generation-3 option-level doubly robust residual advantage learning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

from .autonomous_learning import _expected_probability
from .conservative_learning import (
    _encoded_model,
    _export_model,
    _kmeans,
    _support_distances,
)
from .hazard_representation import (
    HAZARD_PRIMITIVE_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    MAX_HAZARD_PRIMITIVES,
)
from .learning_features import TREE_FEATURE_SCHEMA, tree_candidate_vector, tree_feature_names
from .offline import ACTION_NAMES
from .th06.learning_adapter import ACTION_FEATURE_NAMES, OBSERVATION_FEATURE_NAMES


TRANSITION_SCHEMA = "th06-rl-transition-v9"
BEHAVIOR_POLICY = "safe-option-exploration-v1"
STATE_SCHEMA = "autonomous-dr-option-advantage-policy-v1"
FIT_REPORT_SCHEMA = "autonomous-dr-option-advantage-fit-v1"
POPULATION_MEMBERS = 7
CROSSFIT_FOLDS = 3
NUISANCE_MEMBERS = 3
NUISANCE_TREES = 96
POPULATION_TREES = 128
DISTILLED_TREES = 48
DISTILLATION_P95_ERROR = 0.05
DISTILLATION_MAX_ERROR = 0.25
RICH_FEATURE_SCHEMA = "learned-hazard-codebook-option-tree-v1"
HAZARD_CODEBOOK_SCHEMA = "game-neutral-hazard-codebook-v1"
HAZARD_PROTOTYPES = 24
HAZARD_CODEBOOK_SAMPLE = 65_536
NONEXECUTED_OPTION_TERMINATIONS = frozenset({
    "publication-rejected",
    "hard-empty",
    "authority-loss",
    "stage-transition",
    "bomb",
    "physical-hit",
})


@dataclass(frozen=True)
class OptionStep:
    episode_id: str
    option_id: str
    sequence: int
    frame: int
    action: str
    baseline_action: str
    behavior_probability: float
    vector: tuple[float, ...]
    legal_actions: tuple[str, ...]
    candidate_vectors: tuple[tuple[float, ...], ...]
    option_hit_cost: float
    duration_frames: int
    return_to_go: float = 0.0
    termination_reason: str = ""
    hazard_primitives: tuple[tuple[float, ...], ...] = ()
    history_features: tuple[float, ...] = ()


@dataclass(frozen=True)
class AdvantageSample:
    episode_id: str
    option_id: str
    action: str
    baseline_action: str
    vector: tuple[float, ...]
    pseudo_advantage: float


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


def _rows(run_dir: Path, manifest: dict[str, object]):
    expected_sequence = 0
    observed = 0
    expected = 0
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
        expected += int(shard.get("records", 0))
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("schema_version") != TRANSITION_SCHEMA:
                    raise ValueError("generation 3 requires transition v9")
                if int(row.get("sequence", -1)) != expected_sequence:
                    raise ValueError("transition sequence is not contiguous")
                expected_sequence += 1
                observed += 1
                yield row
    recorded = manifest.get("records")
    recorded_count = (
        int(recorded.get("transitions", -1))
        if isinstance(recorded, dict) else -1
    )
    if observed != expected or observed != recorded_count:
        raise ValueError("transition record count mismatch")


def _frame(snapshot_ref: object) -> int:
    marker = str(snapshot_ref).rsplit(":f", 1)
    if len(marker) != 2:
        raise ValueError("snapshot reference has no physical frame")
    return int(marker[1])


def _vector(row: dict[str, object], action: str) -> tuple[float, ...]:
    context = row.get("policy_context")
    if not isinstance(context, dict):
        raise TypeError("option boundary has no policy context")
    return tree_candidate_vector(
        observation_features=context.get("observation_features"),
        action_features=context.get("action_features"),
        action=action,
        baseline_action=str(row.get("baseline_action", "")),
        current_action=str(context.get("current_action", "")),
        observation_names=OBSERVATION_FEATURE_NAMES,
        action_names=ACTION_FEATURE_NAMES,
    )


def _representation_inputs(
    row: dict[str, object],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    context = row.get("policy_context")
    if not isinstance(context, dict):
        raise TypeError("option boundary has no policy context")
    raw_hazards = context.get("hazard_primitives")
    raw_history = context.get("history_features")
    if not isinstance(raw_hazards, list) or not isinstance(raw_history, list):
        raise TypeError("generation-3 representation inputs are absent")
    if len(raw_hazards) > MAX_HAZARD_PRIMITIVES:
        raise ValueError("hazard primitive set exceeds its online bound")
    hazards = []
    for raw in raw_hazards:
        if not isinstance(raw, list) or len(raw) != len(HAZARD_PRIMITIVE_FEATURE_NAMES):
            raise ValueError("hazard primitive schema mismatch")
        values = tuple(float(value) for value in raw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("hazard primitive contains a non-finite value")
        hazards.append(values)
    history_names = []
    history_values = []
    for raw in raw_history:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("history feature row is invalid")
        history_names.append(str(raw[0]))
        history_values.append(float(raw[1]))
    if tuple(history_names) != HISTORY_FEATURE_NAMES or not all(
        math.isfinite(value) for value in history_values
    ):
        raise ValueError("history feature schema mismatch")
    return tuple(hazards), tuple(history_values)


def _validate_run(run_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    run = _object(run_dir / "run.json")
    manifest = _object(run_dir / "manifest.json")
    schemas = run.get("schemas")
    outcome = manifest.get("run_outcome")
    if (
        manifest.get("complete") is not True
        or manifest.get("stage_trajectory_complete") is not True
        or int(manifest.get("dropped_records", -1)) != 0
        or not isinstance(schemas, dict)
        or schemas.get("transition") != TRANSITION_SCHEMA
        or not isinstance(outcome, dict)
        or outcome.get("stage_completed") is not True
        or not isinstance(outcome.get("physical_hits"), int)
    ):
        raise ValueError("generation-3 learner requires a complete v7 physical Stage")
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
    return run, manifest


def load_option_episode(
    run_dir: Path,
    *,
    exploration_probability: float,
) -> tuple[list[OptionStep], dict[str, object]]:
    """Aggregate factual transition-v9 rows into randomized option treatments."""
    run_dir = run_dir.resolve()
    run, manifest = _validate_run(run_dir)
    rows = list(_rows(run_dir, manifest))
    episode_id = str(run.get("run_id", run_dir.name))
    grouped: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    excluded: Counter[str] = Counter()
    unassigned_hits = 0
    for row in rows:
        option = row.get("option")
        if option is not None and not isinstance(option, dict):
            raise TypeError("option trace is not an object")
        option_id = (
            str(option.get("option_id", "")) if option is not None else ""
        )
        boundary = option is not None and option.get("boundary") is True
        if option is not None and not option_id:
            raise ValueError("option trace has no identity")
        action = str(option.get("intent", "")) if option is not None else ""
        executed = row.get("executed_action")
        if option is not None and executed != action:
            outcome = row.get("outcome_terms")
            if (
                option.get("termination_reason")
                not in NONEXECUTED_OPTION_TERMINATIONS
                or row.get("learning_eligible") is not False
                or not isinstance(outcome, dict)
            ):
                raise ValueError("unpublished option is not explicitly rejected")
            if current is not None and option_id == current["option_id"]:
                if outcome.get("life_lost") is True:
                    raise ValueError("HIT occurred during an unfactual option gap")
                current["termination"] = "publication-rejected"
                grouped.append(current)
                current = None
            elif current is not None and boundary:
                if current["termination"] is None:
                    current["termination"] = str(
                        option.get("preceding_termination_reason")
                        or "next-option-boundary"
                    )
                grouped.append(current)
                current = None
            elif not boundary:
                raise ValueError("unpublished continuation has no factual boundary")
            unassigned_hits += int(outcome.get("life_lost") is True)
            continue
        if boundary:
            if current is not None:
                if current["termination"] is None:
                    current["termination"] = str(
                        option.get("preceding_termination_reason")
                        or "next-option-boundary"
                    )
                grouped.append(current)
            legal_raw = row.get("legal_actions")
            baseline = str(row.get("baseline_action", ""))
            if not isinstance(legal_raw, list):
                raise TypeError("option boundary has no native-safe set")
            legal = tuple(str(value) for value in legal_raw)
            probability = float(option.get("boundary_probability", 0.0))
            expected = _expected_probability(
                action=action,
                baseline=baseline,
                legal=legal,
                exploration_probability=exploration_probability,
            )
            if (
                row.get("policy_id") != BEHAVIOR_POLICY
                or action not in legal
                or baseline not in legal
                or not math.isclose(probability, expected, rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(
                    float(row.get("behavior_probability", 0.0)),
                    probability,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("invalid randomized option boundary")
            current = {
                "option_id": option_id,
                "sequence": int(row["sequence"]),
                "frame": _frame(row.get("snapshot_ref")),
                "action": action,
                "baseline": baseline,
                "probability": probability,
                "legal": legal,
                "vector": _vector(row, action),
                "candidate_vectors": tuple(_vector(row, candidate) for candidate in legal),
                "representation_inputs": _representation_inputs(row),
                "hit_cost": 0.0,
                "physical_elapsed": 0,
                "termination": None,
            }
        elif option is not None:
            if current is None or option_id != current["option_id"]:
                raise ValueError("continuation row escaped its option boundary")
            if str(option.get("intent", "")) != current["action"]:
                raise ValueError("option intent changed inside a treatment")
        outcome = row.get("outcome_terms")
        if not isinstance(outcome, dict):
            raise TypeError("transition row has no physical outcome")
        if outcome.get("bomb_used") is True or outcome.get("authority_lost") is True:
            raise ValueError("invalid physical outcome inside training option")
        if current is None:
            unassigned_hits += int(outcome.get("life_lost") is True)
            continue
        current["hit_cost"] = float(current["hit_cost"]) + float(
            outcome.get("life_lost") is True
        )
        current["physical_elapsed"] = int(current["physical_elapsed"]) + int(
            outcome.get("elapsed_frames", 0)
        )
        termination = (
            option.get("termination_reason") if option is not None else None
        )
        if termination is not None:
            current["termination"] = str(termination)
    if current is not None:
        if current["termination"] is None:
            current["termination"] = "complete-stage-tail"
        grouped.append(current)
    if unassigned_hits:
        raise ValueError("physical HIT occurred before the first option boundary")
    if not grouped:
        raise ValueError("complete Stage has no randomized option boundaries")

    steps = []
    for index, item in enumerate(grouped):
        next_frame = (
            int(grouped[index + 1]["frame"])
            if index + 1 < len(grouped) else None
        )
        duration = (
            next_frame - int(item["frame"])
            if next_frame is not None
            else max(1, int(item["physical_elapsed"]))
        )
        if duration <= 0:
            raise ValueError("option boundary frames are not increasing")
        steps.append(OptionStep(
            episode_id=episode_id,
            option_id=str(item["option_id"]),
            sequence=int(item["sequence"]),
            frame=int(item["frame"]),
            action=str(item["action"]),
            baseline_action=str(item["baseline"]),
            behavior_probability=float(item["probability"]),
            vector=tuple(item["vector"]),
            legal_actions=tuple(item["legal"]),
            candidate_vectors=tuple(item["candidate_vectors"]),
            option_hit_cost=float(item["hit_cost"]),
            duration_frames=duration,
            termination_reason=str(item["termination"]),
            hazard_primitives=tuple(item["representation_inputs"][0]),
            history_features=tuple(item["representation_inputs"][1]),
        ))
    return_value = 0.0
    labeled = []
    for step in reversed(steps):
        return_value += step.option_hit_cost
        labeled.append(replace(step, return_to_go=return_value))
    labeled.reverse()
    outcome = manifest["run_outcome"]
    observed_hits = int(sum(step.option_hit_cost for step in labeled))
    if observed_hits != int(outcome["physical_hits"]):
        raise ValueError("option aggregation did not account for every physical HIT")
    return labeled, {
        "episode_id": episode_id,
        "run_dir": str(run_dir),
        "transitions": len(rows),
        "options": len(labeled),
        "physical_hits": observed_hits,
        "option_terminations": dict(Counter(
            step.termination_reason for step in labeled
        )),
        "excluded": dict(excluded),
    }


def doubly_robust_advantages(
    outcome: float,
    nuisance: list[float],
    *,
    factual_index: int,
    factual_probability: float,
    baseline_index: int,
) -> list[float]:
    """Multi-action AIPW values, differenced against the factual incumbent."""
    if not 0 <= factual_index < len(nuisance) or not 0 <= baseline_index < len(nuisance):
        raise IndexError("factual or baseline action index is invalid")
    if not 0.0 < factual_probability <= 1.0:
        raise ValueError("factual propensity must be in (0, 1]")
    values = list(map(float, nuisance))
    values[factual_index] += (
        float(outcome) - values[factual_index]
    ) / factual_probability
    baseline = values[baseline_index]
    return [value - baseline for value in values]


def hazard_codebook_feature_names() -> tuple[str, ...]:
    return (
        *(f"hazard:prototype_fraction_{index}" for index in range(HAZARD_PROTOTYPES)),
        *(f"hazard:prototype_min_distance_{index}" for index in range(HAZARD_PROTOTYPES)),
        *(f"hazard:mean_{name}" for name in HAZARD_PRIMITIVE_FEATURE_NAMES),
        *(f"hazard:max_abs_{name}" for name in HAZARD_PRIMITIVE_FEATURE_NAMES),
        "hazard:count_log",
        "hazard:empty",
    )


def fit_hazard_codebook(
    samples: list[OptionStep],
    *,
    seed: int,
) -> dict[str, object]:
    """Learn a bounded permutation-invariant primitive codebook."""
    import numpy as np

    generator = random.Random(seed)
    reservoir: list[tuple[float, ...]] = []
    seen = 0
    empty_sets = 0
    for sample in samples:
        empty_sets += not sample.hazard_primitives
        for primitive in sample.hazard_primitives:
            seen += 1
            if len(reservoir) < HAZARD_CODEBOOK_SAMPLE:
                reservoir.append(primitive)
            else:
                index = generator.randrange(seen)
                if index < HAZARD_CODEBOOK_SAMPLE:
                    reservoir[index] = primitive
    if len(reservoir) < HAZARD_PROTOTYPES:
        raise ValueError("training episodes contain too few observed hazards")
    matrix = np.asarray(reservoir, dtype=np.float64)
    if matrix.shape[1] != len(HAZARD_PRIMITIVE_FEATURE_NAMES):
        raise ValueError("hazard codebook primitive width mismatch")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - mean) / scale
    prototypes = _kmeans(
        normalized,
        count=HAZARD_PROTOTYPES,
        iterations=20,
        seed=seed,
    )
    if len(prototypes) != HAZARD_PROTOTYPES:
        raise RuntimeError("hazard codebook did not produce 24 prototypes")
    return {
        "schema": HAZARD_CODEBOOK_SCHEMA,
        "primitive_feature_names": list(HAZARD_PRIMITIVE_FEATURE_NAMES),
        "maximum_primitives": MAX_HAZARD_PRIMITIVES,
        "prototype_count": HAZARD_PROTOTYPES,
        "sample_limit": HAZARD_CODEBOOK_SAMPLE,
        "sampled_primitives": len(reservoir),
        "observed_primitives": seen,
        "empty_training_sets": empty_sets,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "prototypes": prototypes.tolist(),
    }


def encode_hazard_set(
    primitives: tuple[tuple[float, ...], ...],
    artifact: dict[str, object],
) -> tuple[float, ...]:
    import numpy as np

    if (
        artifact.get("schema") != HAZARD_CODEBOOK_SCHEMA
        or tuple(artifact.get("primitive_feature_names", ()))
        != HAZARD_PRIMITIVE_FEATURE_NAMES
        or int(artifact.get("prototype_count", -1)) != HAZARD_PROTOTYPES
        or len(primitives) > MAX_HAZARD_PRIMITIVES
    ):
        raise ValueError("hazard codebook contract mismatch")
    mean = np.asarray(artifact["mean"], dtype=np.float64)
    scale = np.asarray(artifact["scale"], dtype=np.float64)
    prototypes = np.asarray(artifact["prototypes"], dtype=np.float64)
    if primitives:
        matrix = (np.asarray(primitives, dtype=np.float64) - mean) / scale
        distances = ((
            matrix[:, None, :] - prototypes[None, :, :]
        ) ** 2).mean(axis=2)
        assignment = distances.argmin(axis=1)
        fractions = np.bincount(
            assignment, minlength=HAZARD_PROTOTYPES
        ).astype(np.float64) / len(matrix)
        minimum = distances.min(axis=0)
        average = matrix.mean(axis=0)
        max_abs = np.abs(matrix).max(axis=0)
    else:
        fractions = np.zeros(HAZARD_PROTOTYPES)
        minimum = np.zeros(HAZARD_PROTOTYPES)
        average = np.zeros(len(mean))
        max_abs = np.zeros(len(mean))
    encoded = tuple(float(value) for value in (
        *fractions,
        *minimum,
        *average,
        *max_abs,
        math.log1p(len(primitives)),
        float(not primitives),
    ))
    if len(encoded) != len(hazard_codebook_feature_names()) or not all(
        math.isfinite(value) for value in encoded
    ):
        raise RuntimeError("hazard codebook encoding failed")
    return encoded


def rich_feature_names() -> tuple[str, ...]:
    return (
        *tree_feature_names(OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES),
        *hazard_codebook_feature_names(),
        *(f"history:{name}" for name in HISTORY_FEATURE_NAMES),
    )


def rich_candidate_vector(
    base_vector: tuple[float, ...],
    hazard_primitives: tuple[tuple[float, ...], ...],
    history_features: tuple[float, ...],
    artifact: dict[str, object],
) -> tuple[float, ...]:
    if len(history_features) != len(HISTORY_FEATURE_NAMES):
        raise ValueError("four-observation history width mismatch")
    result = (
        *base_vector,
        *encode_hazard_set(hazard_primitives, artifact),
        *history_features,
    )
    if len(result) != len(rich_feature_names()):
        raise RuntimeError("rich candidate vector width mismatch")
    return result


def rich_candidate_vector_from_encoding(
    base_vector: tuple[float, ...],
    hazard_encoding: tuple[float, ...],
    history_features: tuple[float, ...],
) -> tuple[float, ...]:
    if len(hazard_encoding) != len(hazard_codebook_feature_names()):
        raise ValueError("hazard encoding width mismatch")
    if len(history_features) != len(HISTORY_FEATURE_NAMES):
        raise ValueError("four-observation history width mismatch")
    result = (*base_vector, *hazard_encoding, *history_features)
    if len(result) != len(rich_feature_names()):
        raise RuntimeError("rich candidate vector width mismatch")
    return result


def _augment_steps(
    samples: list[OptionStep], artifact: dict[str, object]
) -> list[OptionStep]:
    return [
        replace(
            sample,
            vector=rich_candidate_vector(
                sample.vector,
                sample.hazard_primitives,
                sample.history_features,
                artifact,
            ),
            candidate_vectors=tuple(
                rich_candidate_vector(
                    vector,
                    sample.hazard_primitives,
                    sample.history_features,
                    artifact,
                )
                for vector in sample.candidate_vectors
            ),
        )
        for sample in samples
    ]


def _folds(groups: list[str], *, count: int, seed: int) -> list[tuple[str, ...]]:
    if count < 2 or len(groups) < count * 2:
        raise ValueError("cross-fitting needs at least two episodes per fold")
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    return [tuple(sorted(shuffled[index::count])) for index in range(count)]


def _regressor(*, trees: int, seed: int, threads: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=trees,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=8.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=8.0,
        reg_alpha=0.05,
        tree_method="hist",
        n_jobs=threads,
        random_state=seed,
    )


def _student_regressor(*, seed: int, threads: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=DISTILLED_TREES,
        max_depth=4,
        learning_rate=0.06,
        min_child_weight=8.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=8.0,
        reg_alpha=0.05,
        tree_method="hist",
        n_jobs=threads,
        random_state=seed,
    )


def _fit_rich_support(
    train: list[OptionStep],
    validation: list[OptionStep],
    *,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    import numpy as np

    matrix = np.asarray([sample.vector for sample in train], dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    prototypes = []
    factual_counts = {}
    supported_actions = []
    for action_index, action in enumerate(ACTION_NAMES):
        rows = np.asarray([
            sample.vector for sample in train if sample.action == action
        ], dtype=np.float64)
        factual_counts[action] = len(rows)
        if len(rows):
            supported_actions.append(action)
            prototypes.append(_kmeans(
                (rows - mean) / scale,
                count=12,
                iterations=12,
                seed=seed + action_index,
            ))
        else:
            # Native batch support requires a nonempty group for each stable
            # action index. This placeholder is unreachable because runtime
            # first checks factual_supported_actions.
            prototypes.append(np.zeros((1, matrix.shape[1]), dtype=np.float64))
    if len(supported_actions) < 2:
        raise ValueError("training options support fewer than two actions")
    provisional = {
        "schema": "autonomous-rich-local-prototype-support-v1",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "prototypes": [group.tolist() for group in prototypes],
        "factual_supported_actions": supported_actions,
        "threshold": 0.0,
    }
    validation_rows = [sample.vector for sample in validation]
    validation_actions = [ACTION_NAMES.index(sample.action) for sample in validation]
    if any(sample.action not in supported_actions for sample in validation):
        raise ValueError("validation contains a factual action absent from training")
    distances = _support_distances(
        provisional, validation_rows, validation_actions
    )
    by_episode = {}
    for sample, distance in zip(validation, distances, strict=True):
        by_episode.setdefault(sample.episode_id, []).append(distance)
    episode_scores = {
        episode: max(values) for episode, values in by_episode.items()
    }
    ordered = sorted(episode_scores.values())
    rank = min(len(ordered) - 1, math.ceil(0.90 * (len(ordered) + 1)) - 1)
    threshold = float(ordered[max(0, rank)])
    provisional["threshold"] = threshold
    provisional["threshold_source"] = {
        "kind": "whole-episode-one-sided-conformal-max-distance",
        "nominal_coverage": 0.90,
        "episode_scores": dict(sorted(episode_scores.items())),
        "episode_groups": len(episode_scores),
    }
    report = {
        "threshold": threshold,
        "validation_rows": len(distances),
        "validation_coverage": sum(
            distance <= threshold for distance in distances
        ) / len(distances),
        "training_factual_action_counts": factual_counts,
        "factual_supported_actions": supported_actions,
        "prototypes_per_action": [len(group) for group in prototypes],
        "distance_mean": float(np.mean(distances)),
        "distance_p95": float(np.quantile(distances, 0.95)),
        "distance_max": max(distances),
    }
    return provisional, report


def _calibrate_population_upper(
    validation: list[AdvantageSample],
    predictions,
) -> tuple[dict[str, object], dict[str, object]]:
    import numpy as np

    upper = np.asarray(predictions).max(axis=0)
    actual = np.asarray([
        sample.pseudo_advantage for sample in validation
    ], dtype=np.float64)
    by_episode: dict[str, list[float]] = {}
    eligible = []
    for index, sample in enumerate(validation):
        if sample.action == sample.baseline_action:
            continue
        score = max(0.0, float(actual[index] - upper[index]))
        by_episode.setdefault(sample.episode_id, []).append(score)
        eligible.append(index)
    if len(by_episode) < 3 or not eligible:
        raise ValueError("calibration needs three nonbaseline episode groups")
    episode_scores = {
        episode: max(values) for episode, values in by_episode.items()
    }
    ordered = sorted(episode_scores.values())
    rank = min(len(ordered) - 1, math.ceil(0.90 * (len(ordered) + 1)) - 1)
    radius = float(ordered[max(0, rank)])
    covered = [actual[index] <= upper[index] + radius for index in eligible]
    artifact = {
        "schema": "autonomous-dr-population-upper-conformal-v1",
        "kind": "whole-episode-one-sided-max-residual",
        "nominal_coverage": 0.90,
        "radius": radius,
        "episode_groups": sorted(by_episode),
        "episode_scores": dict(sorted(episode_scores.items())),
    }
    report = {
        "nonbaseline_rows": len(eligible),
        "row_coverage": sum(bool(value) for value in covered) / len(covered),
        "radius": radius,
        "population_upper_mean": float(upper[eligible].mean()),
        "pseudo_advantage_mean": float(actual[eligible].mean()),
    }
    return artifact, report


def _fit_nuisance(
    samples: list[OptionStep],
    *,
    members: int,
    trees: int,
    seed: int,
    threads: int,
):
    import numpy as np

    x = np.asarray([sample.vector for sample in samples], dtype=np.float32)
    y = np.asarray([sample.return_to_go for sample in samples], dtype=np.float32)
    weights = np.asarray([
        1.0 / sample.behavior_probability for sample in samples
    ], dtype=np.float32)
    groups = sorted({sample.episode_id for sample in samples})
    generator = random.Random(seed)
    models = []
    unique_groups = []
    for member in range(members):
        chosen = [generator.choice(groups) for _ in groups]
        counts = Counter(chosen)
        member_weights = weights * np.asarray([
            counts[sample.episode_id] for sample in samples
        ], dtype=np.float32)
        model = _regressor(trees=trees, seed=seed + member, threads=threads)
        model.fit(x, y, sample_weight=member_weights)
        models.append(model)
        unique_groups.append(len(counts))
    return models, unique_groups


def _pseudo_samples(samples: list[OptionStep], nuisance_models) -> list[AdvantageSample]:
    import numpy as np

    rows = np.asarray([
        vector for sample in samples for vector in sample.candidate_vectors
    ], dtype=np.float32)
    predictions = [model.predict(rows) for model in nuisance_models]
    mean = np.asarray(predictions).mean(axis=0)
    result = []
    offset = 0
    for sample in samples:
        stop = offset + len(sample.legal_actions)
        nuisance = mean[offset:stop].tolist()
        factual_index = sample.legal_actions.index(sample.action)
        baseline_index = sample.legal_actions.index(sample.baseline_action)
        advantages = doubly_robust_advantages(
            sample.return_to_go,
            nuisance,
            factual_index=factual_index,
            factual_probability=sample.behavior_probability,
            baseline_index=baseline_index,
        )
        for action, vector, advantage in zip(
            sample.legal_actions,
            sample.candidate_vectors,
            advantages,
            strict=True,
        ):
            result.append(AdvantageSample(
                episode_id=sample.episode_id,
                option_id=sample.option_id,
                action=action,
                baseline_action=sample.baseline_action,
                vector=vector,
                pseudo_advantage=float(advantage),
            ))
        offset = stop
    return result


def _effective_sample_size(samples: list[OptionStep]) -> dict[str, object]:
    report = {}
    for action in ACTION_NAMES:
        weights = [
            1.0 / sample.behavior_probability
            for sample in samples if sample.action == action
        ]
        report[action] = {
            "factual_options": len(weights),
            "inverse_propensity_ess": (
                sum(weights) ** 2 / sum(value * value for value in weights)
                if weights else 0.0
            ),
        }
    return report


def fit_dr_option_advantage(
    train: list[OptionStep],
    validation: list[OptionStep],
    *,
    crossfit_folds: int = CROSSFIT_FOLDS,
    nuisance_members: int = NUISANCE_MEMBERS,
    population_members: int = POPULATION_MEMBERS,
    nuisance_trees: int = NUISANCE_TREES,
    population_trees: int = POPULATION_TREES,
    seed: int = 260812,
    threads: int = 12,
    native_scorer_sha256: str,
) -> dict[str, object]:
    import numpy as np

    train_groups = sorted({sample.episode_id for sample in train})
    validation_groups = sorted({sample.episode_id for sample in validation})
    if len(train_groups) < 9:
        raise ValueError("generation-3 fit needs at least nine training episodes")
    if len(validation_groups) < 3:
        raise ValueError("generation-3 fit needs at least three held-out episodes")
    if set(train_groups) & set(validation_groups):
        raise ValueError("training and validation episodes overlap")
    if crossfit_folds != CROSSFIT_FOLDS:
        raise ValueError("generation-3 fit requires three cross-fit folds")
    if nuisance_members != NUISANCE_MEMBERS:
        raise ValueError("generation-3 fit requires three nuisance members")
    if population_members != POPULATION_MEMBERS:
        raise ValueError("generation-3 population must contain seven members")
    representation = fit_hazard_codebook(train, seed=seed + 30_000)
    conformance_indices = np.linspace(
        0, len(train) - 1, min(4, len(train)), dtype=int
    )
    representation["conformance"] = [
        {
            "primitives": [list(row) for row in train[index].hazard_primitives],
            "encoding": list(encode_hazard_set(
                train[index].hazard_primitives, representation
            )),
        }
        for index in conformance_indices
    ]
    train = _augment_steps(train, representation)
    validation = _augment_steps(validation, representation)
    support, support_report = _fit_rich_support(
        train, validation, seed=seed + 40_000
    )
    folds = _folds(train_groups, count=crossfit_folds, seed=seed)
    pseudo_train = []
    fold_report = []
    for fold_index, heldout in enumerate(folds):
        fit_rows = [sample for sample in train if sample.episode_id not in heldout]
        heldout_rows = [sample for sample in train if sample.episode_id in heldout]
        nuisance, unique = _fit_nuisance(
            fit_rows,
            members=nuisance_members,
            trees=nuisance_trees,
            seed=seed + fold_index * 1000,
            threads=threads,
        )
        rows = _pseudo_samples(heldout_rows, nuisance)
        pseudo_train.extend(rows)
        fold_report.append({
            "fold": fold_index,
            "heldout_episodes": list(heldout),
            "fit_episodes": sorted(set(train_groups) - set(heldout)),
            "heldout_options": len(heldout_rows),
            "pseudo_action_rows": len(rows),
            "nuisance_bootstrap_unique_episodes": unique,
        })
    if {(sample.episode_id, sample.option_id) for sample in pseudo_train} != {
        (sample.episode_id, sample.option_id) for sample in train
    }:
        raise RuntimeError("cross-fitting did not label every training option once")

    full_nuisance, full_nuisance_unique = _fit_nuisance(
        train,
        members=nuisance_members,
        trees=nuisance_trees,
        seed=seed + 10_000,
        threads=threads,
    )
    pseudo_validation = _pseudo_samples(validation, full_nuisance)
    x = np.asarray([sample.vector for sample in pseudo_train], dtype=np.float32)
    y = np.asarray([
        sample.pseudo_advantage for sample in pseudo_train
    ], dtype=np.float32)
    groups = sorted({sample.episode_id for sample in pseudo_train})
    generator = random.Random(seed + 20_000)
    models = []
    population_weights = []
    bootstrap_report = []
    for member in range(population_members):
        chosen = [generator.choice(groups) for _ in groups]
        counts = Counter(chosen)
        weights = np.asarray([
            counts[sample.episode_id] for sample in pseudo_train
        ], dtype=np.float32)
        model = _regressor(
            trees=population_trees,
            seed=seed + 20_000 + member,
            threads=threads,
        )
        model.fit(x, y, sample_weight=weights)
        models.append(model)
        population_weights.append(weights)
        bootstrap_report.append({
            "member": member,
            "unique_episodes": len(counts),
            "episode_counts": dict(sorted(counts.items())),
        })

    validation_x = np.asarray([
        sample.vector for sample in pseudo_validation
    ], dtype=np.float32)
    validation_y = np.asarray([
        sample.pseudo_advantage for sample in pseudo_validation
    ], dtype=np.float32)
    teacher_validation_prediction = np.asarray([
        model.predict(validation_x) for model in models
    ])
    teachers = models
    models = []
    for member, (teacher, weights) in enumerate(zip(
        teachers, population_weights, strict=True
    )):
        student = _student_regressor(
            seed=seed + 50_000 + member,
            threads=threads,
        )
        student.fit(x, teacher.predict(x), sample_weight=weights)
        models.append(student)
    validation_prediction = np.asarray([
        model.predict(validation_x) for model in models
    ])
    distillation_error = np.abs(
        validation_prediction - teacher_validation_prediction
    )
    distillation_p95 = float(np.quantile(distillation_error, 0.95))
    distillation_max = float(distillation_error.max())
    calibration, calibration_report = _calibrate_population_upper(
        pseudo_validation, validation_prediction
    )
    mean_prediction = validation_prediction.mean(axis=0)
    rmse = float(np.sqrt(np.mean((mean_prediction - validation_y) ** 2)))
    constant = float(np.sqrt(np.mean(validation_y ** 2)))
    baseline_identity_error = max(
        (
            abs(sample.pseudo_advantage)
            for sample in (*pseudo_train, *pseudo_validation)
            if sample.action == sample.baseline_action
        ),
        default=0.0,
    )
    conformance_indices = np.linspace(
        0, len(x) - 1, min(8, len(x)), dtype=int
    )
    conformance = x[conformance_indices]
    names = rich_feature_names()
    encoded_models = [
        _encoded_model(_export_model(
            model,
            conformance,
            feature_schema=RICH_FEATURE_SCHEMA,
            feature_names=names,
        ))
        for model in models
    ]
    finite = all(math.isfinite(value) for value in (rmse, constant))
    gates = {
        "train_episode_groups": len(train_groups) >= 9,
        "validation_episode_groups": len(validation_groups) >= 3,
        "disjoint_episode_groups": not bool(set(train_groups) & set(validation_groups)),
        "crossfit_complete": len({
            (sample.episode_id, sample.option_id) for sample in pseudo_train
        }) == len(train),
        "train_has_hits": sum(sample.option_hit_cost for sample in train) > 0.0,
        "validation_has_hits": sum(sample.option_hit_cost for sample in validation) > 0.0,
        "baseline_advantage_identity": baseline_identity_error <= 1e-9,
        "finite_validation_diagnostics": finite,
        "population_complete": len(models) == POPULATION_MEMBERS,
        "distillation_p95": distillation_p95 <= DISTILLATION_P95_ERROR,
        "distillation_max": distillation_max <= DISTILLATION_MAX_ERROR,
        "support_calibrated": support_report["validation_coverage"] >= 0.90,
        "conformal_upper_coverage": calibration_report["row_coverage"] >= 0.90,
        "native_scorer_bound": len(native_scorer_sha256) == 64,
    }
    return {
        "schema": STATE_SCHEMA,
        "mode": "shadow",
        "feature_schema": RICH_FEATURE_SCHEMA,
        "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "feature_names": list(names),
        "representation": {
            "kind": "learned-permutation-invariant-hazard-codebook-plus-factual-history",
            "hazard_codebook": representation,
            "history_feature_names": list(HISTORY_FEATURE_NAMES),
        },
        "models": encoded_models,
        "support": support,
        "selection": {
            "rule": "minimum-calibrated-population-upper-advantage",
            "baseline_advantage": 0.0,
            "conformal_radius": calibration["radius"],
            "active_override_budget": None,
        },
        "native_scorer": {
            "schema": "th06-rl-native-xgboost-scorer-v1",
            "sha256": native_scorer_sha256,
            "compatible_sha256": [native_scorer_sha256],
        },
        "population": {
            "kind": "whole-episode-bootstrap-cross-fitted-dr",
            "members": POPULATION_MEMBERS,
            "bootstrap": bootstrap_report,
        },
        "authorization": {
            "fit_gates": gates,
            "fit_eligible": all(gates.values()),
            "calibration": calibration,
            "active_canary": None,
        },
        "fit_report": {
            "schema": FIT_REPORT_SCHEMA,
            "algorithm": "cross-fitted-multi-action-aipw-option-advantage",
            "return": "undiscounted-complete-stage-physical-hit-count",
            "train_groups": train_groups,
            "validation_groups": validation_groups,
            "train_options": len(train),
            "validation_options": len(validation),
            "train_pseudo_action_rows": len(pseudo_train),
            "validation_pseudo_action_rows": len(pseudo_validation),
            "crossfit_folds": fold_report,
            "full_nuisance_bootstrap_unique_episodes": full_nuisance_unique,
            "population_bootstrap": bootstrap_report,
            "distillation": {
                "teacher_trees_per_member": population_trees,
                "student_trees_per_member": DISTILLED_TREES,
                "validation_p95_absolute_error": distillation_p95,
                "validation_max_absolute_error": distillation_max,
                "p95_gate": DISTILLATION_P95_ERROR,
                "max_gate": DISTILLATION_MAX_ERROR,
            },
            "calibration": calibration_report,
            "support": support_report,
            "hazard_codebook": {
                key: representation[key]
                for key in (
                    "prototype_count",
                    "sample_limit",
                    "sampled_primitives",
                    "observed_primitives",
                    "empty_training_sets",
                )
            },
            "factual_effective_sample_size": _effective_sample_size(train),
            "train_physical_hit_cost": float(sum(
                sample.option_hit_cost for sample in train
            )),
            "validation_physical_hit_cost": float(sum(
                sample.option_hit_cost for sample in validation
            )),
            "heldout_dr_advantage_rmse": rmse,
            "heldout_zero_advantage_rmse": constant,
            "baseline_identity_max_error": baseline_identity_error,
        },
    }


def _causal_smoke_episodes(prefix: str, count: int) -> list[OptionStep]:
    names = tree_feature_names(OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES)
    indices = {name: index for index, name in enumerate(names)}
    result = []
    for episode_index in range(count):
        episode = f"{prefix}-{episode_index}"
        for option_index in range(32):
            state_risk = float(option_index % 2)
            baseline = [0.0] * len(names)
            baseline[indices["observation:position_x_unit"]] = state_risk
            baseline[indices["matches_baseline"]] = 1.0
            candidate = baseline.copy()
            candidate[indices["action:direction_x"]] = -1.0
            candidate[indices["delta_from_baseline:direction_x"]] = -1.0
            candidate[indices["matches_baseline"]] = 0.0
            assigned_candidate = (episode_index + option_index) % 2 == 0
            outcome = 2.0 + 2.0 * state_risk - float(assigned_candidate)
            primitive = [0.0] * len(HAZARD_PRIMITIVE_FEATURE_NAMES)
            primitive[0] = state_risk
            primitive[1] = option_index / 64.0
            primitive[6] = math.hypot(primitive[0], primitive[1])
            primitive[11] = 1.0
            history = [0.0] * len(HISTORY_FEATURE_NAMES)
            history[0] = 1.0
            history[2] = state_risk
            result.append(OptionStep(
                episode_id=episode,
                option_id=f"{episode}:{option_index}",
                sequence=option_index,
                frame=option_index * 8,
                action="left" if assigned_candidate else "stay",
                baseline_action="stay",
                behavior_probability=0.5,
                vector=tuple(candidate if assigned_candidate else baseline),
                legal_actions=("stay", "left"),
                candidate_vectors=(tuple(baseline), tuple(candidate)),
                option_hit_cost=outcome,
                duration_frames=8,
                return_to_go=outcome,
                termination_reason="horizon",
                hazard_primitives=(tuple(primitive),),
                history_features=tuple(history),
            ))
    return result


def run_causal_recovery_smoke(*, threads: int = 4) -> dict[str, object]:
    """Fail-fast proof that DR residual fitting recovers a known effect."""
    from .policies.autonomous_conservative_q import _decode_model
    from .policies.offline_ranker import PortableXGBoostRegressor

    state = fit_dr_option_advantage(
        _causal_smoke_episodes("smoke-train", 9),
        _causal_smoke_episodes("smoke-validation", 3),
        seed=260812,
        threads=threads,
        native_scorer_sha256="0" * 64,
    )
    names = tuple(state["feature_names"])
    base_names = tree_feature_names(
        OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES
    )
    scorers = [
        PortableXGBoostRegressor(
            _decode_model(model),
            expected_feature_schema=RICH_FEATURE_SCHEMA,
            expected_feature_names=names,
        )
        for model in state["models"]
    ]
    predictions = []
    for state_risk in (0.0, 1.0):
        baseline = [0.0] * len(base_names)
        baseline[base_names.index("observation:position_x_unit")] = state_risk
        baseline[base_names.index("matches_baseline")] = 1.0
        candidate = baseline.copy()
        candidate[base_names.index("action:direction_x")] = -1.0
        candidate[base_names.index("delta_from_baseline:direction_x")] = -1.0
        candidate[base_names.index("matches_baseline")] = 0.0
        primitive = [0.0] * len(HAZARD_PRIMITIVE_FEATURE_NAMES)
        primitive[0] = state_risk
        primitive[6] = abs(state_risk)
        primitive[11] = 1.0
        history = [0.0] * len(HISTORY_FEATURE_NAMES)
        history[0] = 1.0
        history[2] = state_risk
        candidate = rich_candidate_vector(
            tuple(candidate),
            (tuple(primitive),),
            tuple(history),
            state["representation"]["hazard_codebook"],
        )
        predictions.append([
            float(scorer.predict_many([candidate])[0]) for scorer in scorers
        ])
    flat = [value for row in predictions for value in row]
    mean = sum(flat) / len(flat)
    leakage = max(
        abs(left - right)
        for left, right in zip(predictions[0], predictions[1], strict=True)
    )
    population_range = max(flat) - min(flat)
    fit = state["fit_report"]
    gates = {
        "fit_contract": state["authorization"]["fit_eligible"] is True,
        "all_members_recover_negative_effect": all(value < 0.0 for value in flat),
        "known_effect_error_at_most_half_hit": abs(mean - (-1.0)) <= 0.5,
        "state_risk_leakage_below_0_15_hit": leakage < 0.15,
        "residual_beats_zero_advantage": (
            fit["heldout_dr_advantage_rmse"]
            < fit["heldout_zero_advantage_rmse"]
        ),
        "population_not_collapsed": population_range > 1e-6,
    }
    return {
        "schema": "autonomous-generation-3-causal-smoke-v1",
        "known_candidate_advantage": -1.0,
        "population_members": POPULATION_MEMBERS,
        "candidate_predictions_by_state_risk": predictions,
        "candidate_prediction_mean": mean,
        "candidate_prediction_range": population_range,
        "state_risk_leakage_max": leakage,
        "heldout_dr_advantage_rmse": fit["heldout_dr_advantage_rmse"],
        "heldout_zero_advantage_rmse": fit["heldout_zero_advantage_rmse"],
        "gates": gates,
        "passed": all(gates.values()),
    }


def audit_wine_option_smoke(
    run_dir: Path,
    *,
    exploration_probability: float = 0.10,
    minimum_boundaries: int = 32,
) -> dict[str, object]:
    """Audit a short, explicitly non-evidence Wine option pipeline run."""
    run_dir = run_dir.resolve()
    run = _object(run_dir / "run.json")
    manifest = _object(run_dir / "manifest.json")
    schemas = run.get("schemas")
    outcome = manifest.get("run_outcome")
    if (
        manifest.get("complete") is not True
        or int(manifest.get("dropped_records", -1)) != 0
        or not isinstance(schemas, dict)
        or schemas.get("transition") != TRANSITION_SCHEMA
        or not isinstance(outcome, dict)
    ):
        raise ValueError("Wine smoke corpus is incomplete or not transition v9")
    infrastructure_fields = (
        "background_reactivations",
        "capture_failures",
        "corpus_failures",
        "infrastructure_failures",
        "trace_failures",
    )
    clean_infrastructure = all(
        int(outcome.get(field, -1)) == 0 for field in infrastructure_fields
    ) and outcome.get("corpus_failure") is None
    boundaries = 0
    continuations = 0
    non_incumbent = 0
    safe_membership = 0
    horizon_terminations = 0
    representation_boundaries = 0
    option_rows = 0
    rejected_option_rows = 0
    option_ids: set[str] = set()
    active_option_id: str | None = None
    for row in _rows(run_dir, manifest):
        option = row.get("option")
        if option is None:
            continue
        if not isinstance(option, dict):
            raise TypeError("Wine smoke option trace is invalid")
        option_rows += 1
        option_id = str(option.get("option_id", ""))
        action = str(option.get("intent", ""))
        legal_raw = row.get("legal_actions")
        if row.get("policy_id") != BEHAVIOR_POLICY:
            raise ValueError("Wine smoke used the wrong behavior policy")
        if not isinstance(legal_raw, list) or action not in legal_raw:
            raise ValueError("Wine smoke option escaped the native-safe set")
        elapsed = int(option.get("elapsed_frames_at_decision", 0))
        if not 1 <= elapsed <= 8:
            raise ValueError("Wine smoke option exceeded its fixed horizon")
        boundary = option.get("boundary") is True
        executed = row.get("executed_action")
        legal = tuple(str(value) for value in legal_raw)
        baseline = str(row.get("baseline_action", ""))
        conditional_expected = (
            _expected_probability(
                action=action,
                baseline=baseline,
                legal=legal,
                exploration_probability=exploration_probability,
            )
            if boundary else 1.0
        )
        if boundary and not math.isclose(
            float(option.get("boundary_probability", 0.0)),
            conditional_expected,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Wine smoke boundary propensity is invalid")
        conditional = float(option.get("conditional_probability", 0.0))
        if not math.isclose(
            conditional,
            conditional_expected,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(row.get("behavior_probability", 0.0)),
            conditional,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Wine smoke conditional propensity is invalid")
        if executed != action:
            if (
                option.get("termination_reason")
                not in NONEXECUTED_OPTION_TERMINATIONS
                or row.get("learning_eligible") is not False
            ):
                raise ValueError("Wine smoke has an ambiguous unpublished option")
            rejected_option_rows += 1
            if not boundary:
                if option_id != active_option_id:
                    raise ValueError("rejected continuation escaped its factual option")
                active_option_id = None
            continue
        safe_membership += 1
        if boundary:
            boundaries += 1
            if option_id in option_ids or elapsed != 1:
                raise ValueError("Wine smoke option boundary identity is invalid")
            option_ids.add(option_id)
            active_option_id = option_id
            _representation_inputs(row)
            representation_boundaries += 1
            non_incumbent += int(action != baseline)
        else:
            continuations += 1
            if option_id != active_option_id:
                raise ValueError("Wine smoke continuation escaped its boundary")
        if option.get("termination_reason") == "horizon" and elapsed != 8:
            raise ValueError("Wine smoke horizon termination occurred off boundary")
        horizon_terminations += option.get("termination_reason") == "horizon"
        if option.get("termination_reason") is not None:
            active_option_id = None
    summary = manifest.get("summary")
    input_lease_rows = (
        int(summary.get("reason_counts", {}).get("input-lease", 0))
        if isinstance(summary, dict)
        and isinstance(summary.get("reason_counts"), dict)
        else 0
    )
    gates = {
        "clean_infrastructure": clean_infrastructure,
        "minimum_option_boundaries": boundaries >= minimum_boundaries,
        "randomized_non_incumbent_witnessed": non_incumbent >= 1,
        "conditional_continuation_witnessed": continuations >= 1,
        "horizon_termination_witnessed": horizon_terminations >= 1,
        "all_executed_option_intents_native_safe": (
            safe_membership == option_rows - rejected_option_rows
        ),
        "input_lease_witnessed": input_lease_rows >= 1,
        "representation_present_at_every_boundary": (
            representation_boundaries == boundaries
        ),
    }
    return {
        "schema": "autonomous-generation-3-wine-option-smoke-v1",
        "run_dir": str(run_dir),
        "evidence_eligible": False,
        "option_rows": option_rows,
        "rejected_option_rows": rejected_option_rows,
        "executed_option_rows": option_rows - rejected_option_rows,
        "option_boundaries": boundaries,
        "option_continuations": continuations,
        "non_incumbent_boundaries": non_incumbent,
        "horizon_terminations": horizon_terminations,
        "representation_boundaries": representation_boundaries,
        "input_lease_rows": input_lease_rows,
        "gates": gates,
        "passed": all(gates.values()),
    }
