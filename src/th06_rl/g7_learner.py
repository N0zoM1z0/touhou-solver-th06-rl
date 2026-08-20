"""Episode-cross-fitted critic and proper linear AWR actor for Generation 7."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math

from th06_rl.feature_contract import FEATURE_AVAILABILITY_SCHEMA
from th06_rl.g7_policy_math import (
    POLICY_DISTRIBUTION_SCHEMA,
    ConstrainedDistribution,
    constrained_cost_distribution,
    reference_distribution,
)
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.learning_features import (
    CAUSAL_TREE_FEATURE_SCHEMA,
    causal_tree_candidate_vector,
    causal_tree_feature_names,
)
from th06_rl.offline_options import (
    OPTION_DATASET_SCHEMA,
    ActorState,
    OfflineOptionTransition,
)
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)


LINEAR_ACTOR_SCHEMA = "th06-rl-g7-linear-awr-actor-v1"
STAGE_NUISANCE_NAMES = tuple(f"nuisance:stage-{stage}" for stage in range(1, 7))
ACTOR_FEATURE_NAMES = causal_tree_feature_names(
    OBSERVATION_FEATURE_NAMES,
    ACTION_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
)
CRITIC_FEATURE_NAMES = (*ACTOR_FEATURE_NAMES, *STAGE_NUISANCE_NAMES)


@dataclass(frozen=True)
class CriticExample:
    episode_id: str
    option_id: str
    factual_action: str
    behavior_probability: float
    reference_probabilities: tuple[tuple[str, float], ...]
    target_cost_to_go: float
    candidate_vectors: tuple[tuple[str, tuple[float, ...]], ...]


@dataclass(frozen=True)
class CriticDataset:
    examples: tuple[CriticExample, ...]
    episode_ids: tuple[str, ...]
    excluded_options: int
    exclusion_reasons: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class CriticPrediction:
    episode_id: str
    option_id: str
    costs: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class CrossFitCriticResult:
    predictions: tuple[CriticPrediction, ...]
    fold_by_episode: tuple[tuple[str, int], ...]
    factual_rmse: float
    example_count: int
    episode_count: int
    maximum_importance_ratio: float


def _actor_vector(state: ActorState, action: str) -> tuple[float, ...]:
    return causal_tree_candidate_vector(
        observation_features=state.observation_features,
        action_features=state.action_features,
        history_features=state.history_features,
        action=action,
        baseline_action=state.baseline_action,
        current_action=state.current_action,
        observation_names=OBSERVATION_FEATURE_NAMES,
        action_names=ACTION_FEATURE_NAMES,
        history_names=HISTORY_FEATURE_NAMES,
    )


def _critic_vector(
    state: ActorState,
    action: str,
    stage: int,
) -> tuple[float, ...]:
    if stage not in range(1, 7):
        raise ValueError("critic nuisance stage must be in 1..6")
    return (
        *_actor_vector(state, action),
        *(float(candidate == stage) for candidate in range(1, 7)),
    )


def _cost_to_go(
    options: tuple[OfflineOptionTransition, ...],
) -> dict[str, float]:
    if not options:
        raise ValueError("physical episode has no option transitions")
    episode = options[0].episode_id
    if any(option.episode_id != episode for option in options):
        raise ValueError("cost-to-go input mixes physical episodes")
    if any(option.schema != OPTION_DATASET_SCHEMA for option in options):
        raise ValueError("physical episode uses an unsupported option schema")
    if any(option.episode_unit != "complete-route" for option in options):
        raise ValueError("Generation-7 training requires complete-route episodes")
    if len({option.behavior_policy_id for option in options}) != 1:
        raise ValueError("physical episode mixes behavior policies")
    if len({option.option_id for option in options}) != len(options):
        raise ValueError("physical episode repeats an option identity")
    if any(option.terminal for option in options[:-1]) or not options[-1].terminal:
        raise ValueError("only the last option may terminate the physical episode")
    if any(
        option.next_state is None if not option.terminal else option.next_state is not None
        for option in options
    ):
        raise ValueError("option terminal and next-state facts disagree")
    if any(
        left.next_state != right.state
        for left, right in zip(options, options[1:])
    ):
        raise ValueError("option next-state linkage is not exact")
    if any(
        right.start_sequence <= left.end_sequence
        for left, right in zip(options, options[1:])
    ):
        raise ValueError("physical episode options are not sequence ordered")
    for option in options:
        probabilities = dict(option.behavior_probabilities)
        if (
            not option.behavior_policy_id
            or option.end_sequence < option.start_sequence
            or option.elapsed_frames <= 0
            or option.interstitial_elapsed_frames < 0
            or isinstance(option.physical_hit_cost, bool)
            or isinstance(option.controlled_hit_cost, bool)
            or isinstance(option.interstitial_hit_cost, bool)
            or min(
                option.physical_hit_cost,
                option.controlled_hit_cost,
                option.interstitial_hit_cost,
            ) < 0
            or option.physical_hit_cost
            != option.controlled_hit_cost + option.interstitial_hit_cost
            or len(probabilities) != len(option.behavior_probabilities)
            or set(probabilities) != set(option.state.legal_actions)
            or option.action not in probabilities
            or any(
                not math.isfinite(probability) or probability < 0.0
                for probability in probabilities.values()
            )
            or not math.isclose(
                sum(probabilities.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9
            )
            or not math.isclose(
                probabilities[option.action],
                option.behavior_probability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or option.behavior_probability <= 0.0
        ):
            raise ValueError("physical option facts violate the causal contract")
    result = {}
    cumulative = 0.0
    for option in reversed(options):
        cumulative += option.physical_hit_cost
        result[option.option_id] = cumulative
    return result


def build_critic_dataset(
    episodes,
    *,
    reference_epsilon: float,
) -> CriticDataset:
    """Build factual MC targets while retaining every excluded interval's HIT."""
    examples = []
    excluded = Counter()
    episode_ids = []
    for raw_episode in episodes:
        options = tuple(raw_episode)
        returns = _cost_to_go(options)
        episode_ids.append(options[0].episode_id)
        for option in options:
            if not option.eligible:
                excluded.update(option.exclusion_reasons or ("ineligible",))
                continue
            reference = reference_distribution(
                option.state.legal_actions,
                option.state.baseline_action,
                epsilon=reference_epsilon,
            )
            candidates = tuple(
                (
                    action,
                    _critic_vector(option.state, action, option.start_stage),
                )
                for action in option.state.legal_actions
            )
            if option.action not in dict(candidates):
                raise ValueError("factual action is absent from critic candidates")
            examples.append(CriticExample(
                option.episode_id,
                option.option_id,
                option.action,
                option.behavior_probability,
                reference,
                returns[option.option_id],
                candidates,
            ))
    unique_episodes = tuple(sorted(set(episode_ids)))
    if len(unique_episodes) != len(episode_ids):
        raise ValueError("critic input repeats a physical episode")
    return CriticDataset(
        tuple(examples),
        unique_episodes,
        sum(excluded.values()),
        tuple(sorted(excluded.items())),
    )


def whole_episode_folds(
    episode_ids: tuple[str, ...],
    *,
    folds: int,
    seed: int,
) -> tuple[tuple[str, int], ...]:
    unique = tuple(sorted(set(episode_ids)))
    if len(unique) != len(episode_ids) or not 2 <= folds <= len(unique):
        raise ValueError("cross-fitting requires unique episodes and 2..N folds")
    if not 0 <= seed < 2**64:
        raise ValueError("cross-fit seed must be unsigned 64-bit")
    ordered = sorted(
        unique,
        key=lambda episode: hashlib.sha256(
            f"{seed}:{episode}".encode("utf-8")
        ).digest(),
    )
    return tuple(sorted(
        (episode, index % folds) for index, episode in enumerate(ordered)
    ))


def cross_fit_cost_critic(
    dataset: CriticDataset,
    *,
    folds: int,
    seed: int,
    max_importance_ratio: float = 20.0,
    n_jobs: int = 1,
    n_estimators: int = 160,
) -> CrossFitCriticResult:
    """Fit a nuisance Q under whole-episode cross-fitting, never row splits."""
    if not dataset.examples:
        raise ValueError("critic dataset has no eligible examples")
    if not math.isfinite(max_importance_ratio) or max_importance_ratio < 1.0:
        raise ValueError("maximum importance ratio must be >=1")
    if n_jobs < 1 or n_estimators < 1:
        raise ValueError("critic resource bounds must be positive")
    import numpy as np
    import xgboost as xgb

    fold_rows = whole_episode_folds(dataset.episode_ids, folds=folds, seed=seed)
    fold_by_episode = dict(fold_rows)
    predictions = []
    errors = []
    observed_maximum_ratio = 0.0
    for validation_fold in range(folds):
        training = [
            example for example in dataset.examples
            if fold_by_episode[example.episode_id] != validation_fold
        ]
        validation = [
            example for example in dataset.examples
            if fold_by_episode[example.episode_id] == validation_fold
        ]
        if not training or not validation:
            raise ValueError("cross-fit fold has no train or validation examples")
        x_train = []
        y_train = []
        weights = []
        for example in training:
            reference = dict(example.reference_probabilities)
            ratio = reference[example.factual_action] / example.behavior_probability
            if not math.isfinite(ratio) or ratio <= 0.0 or ratio > max_importance_ratio:
                raise ValueError("critic importance ratio exceeds the declared support bound")
            observed_maximum_ratio = max(observed_maximum_ratio, ratio)
            x_train.append(dict(example.candidate_vectors)[example.factual_action])
            y_train.append(example.target_cost_to_go)
            weights.append(ratio)
        training_matrix = xgb.DMatrix(
            np.asarray(x_train, dtype=np.float32),
            label=np.asarray(y_train, dtype=np.float32),
            weight=np.asarray(weights, dtype=np.float32),
            feature_names=list(CRITIC_FEATURE_NAMES),
        )
        model = xgb.train(
            {
                "objective": "reg:squarederror",
                "max_depth": 4,
                "eta": 0.05,
                "min_child_weight": 8.0,
                "lambda": 4.0,
                "alpha": 0.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "tree_method": "hist",
                "seed": seed + validation_fold,
                "nthread": n_jobs,
            },
            training_matrix,
            num_boost_round=n_estimators,
        )
        for example in validation:
            actions = tuple(action for action, _vector in example.candidate_vectors)
            matrix = np.asarray(
                [vector for _action, vector in example.candidate_vectors],
                dtype=np.float32,
            )
            values = tuple(float(value) for value in model.predict(xgb.DMatrix(
                matrix,
                feature_names=list(CRITIC_FEATURE_NAMES),
            )))
            costs = tuple(zip(actions, values, strict=True))
            predictions.append(CriticPrediction(
                example.episode_id,
                example.option_id,
                costs,
            ))
            errors.append(
                dict(costs)[example.factual_action] - example.target_cost_to_go
            )
    predictions.sort(key=lambda row: (row.episode_id, row.option_id))
    if len(predictions) != len(dataset.examples):
        raise RuntimeError("cross-fitting did not predict every eligible example once")
    return CrossFitCriticResult(
        tuple(predictions),
        fold_rows,
        math.sqrt(sum(error * error for error in errors) / len(errors)),
        len(dataset.examples),
        len(dataset.episode_ids),
        observed_maximum_ratio,
    )


def fit_linear_awr_actor(
    dataset: CriticDataset,
    critic: CrossFitCriticResult,
    *,
    reference_epsilon: float,
    temperature: float,
    maximum_log_weight: float = 4.0,
    l2: float = 1e-3,
) -> dict[str, object]:
    """Fit reference-offset logits with a bounded nonnegative AWR objective."""
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("AWR temperature must be positive")
    if maximum_log_weight < 0.0 or not math.isfinite(maximum_log_weight):
        raise ValueError("AWR log-weight bound must be finite and nonnegative")
    if l2 < 0.0 or not math.isfinite(l2):
        raise ValueError("AWR L2 must be finite and nonnegative")
    import numpy as np
    from scipy.optimize import minimize

    prediction_by_key = {
        (row.episode_id, row.option_id): dict(row.costs)
        for row in critic.predictions
    }
    if len(prediction_by_key) != len(dataset.examples):
        raise ValueError("critic predictions are duplicated or incomplete")
    states = []
    all_vectors = []
    for example in dataset.examples:
        q = prediction_by_key[(example.episode_id, example.option_id)]
        reference = dict(example.reference_probabilities)
        if set(q) != set(reference):
            raise ValueError("critic prediction and reference action sets differ")
        value = sum(reference[action] * q[action] for action in reference)
        advantage = q[example.factual_action] - value
        log_weight = max(
            -maximum_log_weight,
            min(maximum_log_weight, -advantage / temperature),
        )
        importance = reference[example.factual_action] / example.behavior_probability
        weight = importance * math.exp(log_weight)
        actor_vectors = tuple(
            (action, _actor_vector_from_critic(vector))
            for action, vector in example.candidate_vectors
        )
        all_vectors.extend(vector for _action, vector in actor_vectors)
        states.append((example, reference, actor_vectors, weight))

    matrix = np.asarray(all_vectors, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    dimension = len(ACTOR_FEATURE_NAMES)

    def objective(weights):
        total_weight = 0.0
        loss = 0.0
        gradient = np.zeros(dimension, dtype=np.float64)
        for example, reference, rows, sample_weight in states:
            vectors = np.asarray(
                [(np.asarray(vector) - mean) / scale for _action, vector in rows],
                dtype=np.float64,
            )
            actions = tuple(action for action, _vector in rows)
            logits = np.asarray([
                math.log(reference[action]) + float(vectors[index] @ weights)
                for index, action in enumerate(actions)
            ])
            maximum = float(logits.max())
            exp = np.exp(logits - maximum)
            probabilities = exp / exp.sum()
            factual = actions.index(example.factual_action)
            loss += sample_weight * (
                maximum + math.log(float(exp.sum())) - logits[factual]
            )
            gradient += sample_weight * (
                probabilities @ vectors - vectors[factual]
            )
            total_weight += sample_weight
        loss = loss / total_weight + 0.5 * l2 * float(weights @ weights)
        gradient = gradient / total_weight + l2 * weights
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"proper AWR actor fit failed: {fitted.message}")
    return {
        "schema": LINEAR_ACTOR_SCHEMA,
        "feature_schema": CAUSAL_TREE_FEATURE_SCHEMA,
        "feature_availability_schema": FEATURE_AVAILABILITY_SCHEMA,
        "policy_distribution_schema": POLICY_DISTRIBUTION_SCHEMA,
        "feature_names": list(ACTOR_FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": fitted.x.tolist(),
        "reference_epsilon": reference_epsilon,
        "awr_temperature": temperature,
        "maximum_log_weight": maximum_log_weight,
        "l2": l2,
        "training_examples": len(states),
        "training_episodes": len(dataset.episode_ids),
        "objective": float(fitted.fun),
    }


def _actor_vector_from_critic(vector: tuple[float, ...]) -> tuple[float, ...]:
    if len(vector) != len(CRITIC_FEATURE_NAMES):
        raise ValueError("critic candidate feature length mismatch")
    return vector[:len(ACTOR_FEATURE_NAMES)]


def linear_actor_scores(
    artifact: dict[str, object],
    state: ActorState,
) -> tuple[tuple[str, float], ...]:
    """Portable bounded online scorer; no training framework is imported."""
    if (
        artifact.get("schema") != LINEAR_ACTOR_SCHEMA
        or artifact.get("feature_schema") != CAUSAL_TREE_FEATURE_SCHEMA
        or artifact.get("feature_availability_schema")
        != FEATURE_AVAILABILITY_SCHEMA
        or artifact.get("policy_distribution_schema")
        != POLICY_DISTRIBUTION_SCHEMA
        or tuple(artifact.get("feature_names", ())) != ACTOR_FEATURE_NAMES
    ):
        raise ValueError("linear actor artifact contract mismatch")
    mean = tuple(float(value) for value in artifact.get("mean", ()))
    scale = tuple(float(value) for value in artifact.get("scale", ()))
    weights = tuple(float(value) for value in artifact.get("weights", ()))
    if (
        len(mean) != len(ACTOR_FEATURE_NAMES)
        or len(scale) != len(mean)
        or len(weights) != len(mean)
        or any(not math.isfinite(value) for value in (*mean, *weights))
        or any(not math.isfinite(value) or value <= 0.0 for value in scale)
    ):
        raise ValueError("linear actor numeric artifact is invalid")
    result = []
    for action in state.legal_actions:
        vector = _actor_vector(state, action)
        score = sum(
            ((value - center) / width) * weight
            for value, center, width, weight in zip(
                vector, mean, scale, weights, strict=True
            )
        )
        if not math.isfinite(score):
            raise ValueError("linear actor produced a non-finite score")
        result.append((action, score))
    return tuple(result)


def linear_actor_distribution(
    artifact: dict[str, object],
    state: ActorState,
    *,
    supported_actions,
    forecast_accepted_actions,
    max_kl: float,
) -> ConstrainedDistribution:
    """Use the same reference-offset residual policy in OPE and deployment."""
    scores = linear_actor_scores(artifact, state)
    epsilon = float(artifact.get("reference_epsilon", float("nan")))
    return constrained_cost_distribution(
        safe_actions=state.legal_actions,
        baseline_action=state.baseline_action,
        predicted_costs=tuple((action, -score) for action, score in scores),
        supported_actions=supported_actions,
        forecast_accepted_actions=forecast_accepted_actions,
        epsilon=epsilon,
        temperature=1.0,
        max_kl=max_kl,
    )
