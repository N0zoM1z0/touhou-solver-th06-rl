"""Proper nonnegative AWR extraction for the exact residual policy object."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from statistics import mean

from ..actions import ACTION_NAMES
from .objectives import awr_weight
from .fqe import evaluate_fqe_crosschecks
from .linear_models import ridge_pipeline
from .orthogonal_learning import (
    CompactEpisode,
    OrthogonalConfig,
    _append_nuisance,
    _behavior_expected_features,
    _cluster_summary,
    _fit_effect_model,
    _one_hot_nuisance,
    _state_and_centered,
    episode_folds,
)
from .policy_distribution import ResidualStochasticPolicy, reference_probabilities


@dataclass(frozen=True)
class AwrConfig:
    temperature: float
    maximum_weight: float
    kl_coefficient: float
    epochs: int
    learning_rate: float
    l2: float

    def __post_init__(self) -> None:
        if (
            self.temperature <= 0.0
            or self.maximum_weight < 1.0
            or self.kl_coefficient < 0.0
            or self.epochs <= 0
            or self.learning_rate <= 0.0
            or self.l2 < 0.0
        ):
            raise ValueError("AWR actor configuration is invalid")


@dataclass(frozen=True)
class LinearAwrActor:
    feature_indices: tuple[int, ...]
    coefficients: tuple[float, ...]

    def scores(self, features):
        import numpy as np

        return np.asarray(features)[:, self.feature_indices] @ np.asarray(
            self.coefficients
        )


def _varying_feature_indices(feature_names: tuple[str, ...]) -> tuple[int, ...]:
    indices = tuple(
        index for index, name in enumerate(feature_names)
        if name.startswith("action:")
        or name.startswith("delta_from_baseline:")
        or name in {"matches_baseline", "matches_current"}
    )
    if not indices:
        raise ValueError("actor has no within-safe-set varying features")
    return indices


def _episode_actor_arrays(
    episode: CompactEpisode,
    *,
    feature_indices: tuple[int, ...],
    reference_epsilon: float,
    supported_actions: frozenset[int],
):
    import numpy as np

    sizes = np.diff(episode.offsets).astype(np.int64)
    groups = np.repeat(np.arange(episode.option_count), sizes)
    baselines = episode.offsets[:-1] + episode.baseline_positions
    factual = episode.offsets[:-1] + episode.factual_positions
    reference = np.repeat(reference_epsilon / sizes, sizes).astype(np.float64)
    reference[baselines] += 1.0 - reference_epsilon
    supported = np.asarray(
        [int(action) in supported_actions for action in episode.action_indices],
        dtype=bool,
    )
    baseline_mask = np.zeros(len(supported), dtype=bool)
    baseline_mask[baselines] = True
    supported |= baseline_mask
    return (
        episode.features[:, feature_indices].astype(np.float64, copy=False),
        sizes,
        groups,
        baselines.astype(np.int64),
        factual.astype(np.int64),
        reference,
        supported,
    )


def _expected_actor_policy_features(
    episode: CompactEpisode,
    *,
    actor: LinearAwrActor,
    supported_actions: frozenset[int],
    policy: ResidualStochasticPolicy,
):
    import numpy as np

    target_rows = np.empty(
        (episode.option_count, episode.features.shape[1]), dtype=np.float64
    )
    reference_rows = np.empty_like(target_rows)
    target_probabilities = np.empty(len(episode.features), dtype=np.float64)
    reference_probability_rows = np.empty_like(target_probabilities)
    scores = actor.scores(episode.features)
    for option in range(episode.option_count):
        start = int(episode.offsets[option])
        stop = int(episode.offsets[option + 1])
        features = episode.features[start:stop].astype(np.float64, copy=False)
        actions = tuple(
            ACTION_NAMES[int(value)]
            for value in episode.action_indices[start:stop]
        )
        baseline_position = int(episode.baseline_positions[option])
        supported = tuple(
            int(action) in supported_actions
            for action in episode.action_indices[start:stop]
        )
        decision = policy.distribution(
            safe_actions=actions,
            baseline_action=actions[baseline_position],
            logits=tuple(map(float, scores[start:stop])),
            statistically_supported=supported,
            forecast_risky=(False,) * len(actions),
        )
        target = np.asarray(decision.probabilities, dtype=np.float64)
        reference = np.asarray(reference_probabilities(
            actions, actions[baseline_position], epsilon=policy.epsilon
        ))
        target_rows[option] = target @ features
        reference_rows[option] = reference @ features
        target_probabilities[start:stop] = target
        reference_probability_rows[start:stop] = reference
    return (
        target_rows,
        reference_rows,
        target_probabilities,
        reference_probability_rows,
    )


def fit_linear_awr_actor(
    episodes: tuple[CompactEpisode, ...],
    *,
    weights: dict[str, object],
    feature_names: tuple[str, ...],
    supported_actions: frozenset[int],
    policy: ResidualStochasticPolicy,
    config: AwrConfig,
) -> tuple[LinearAwrActor, dict[str, object]]:
    """Full-batch Adam on bounded proper weighted conditional likelihood."""
    import numpy as np

    feature_indices = _varying_feature_indices(feature_names)
    arrays = tuple(
        _episode_actor_arrays(
            episode,
            feature_indices=feature_indices,
            reference_epsilon=policy.epsilon,
            supported_actions=supported_actions,
        )
        for episode in episodes
    )
    width = len(feature_indices)
    coefficients = np.zeros(width, dtype=np.float64)
    first_moment = np.zeros(width, dtype=np.float64)
    second_moment = np.zeros(width, dtype=np.float64)
    beta1, beta2 = 0.9, 0.999
    total_options = sum(episode.option_count for episode in episodes)
    total_weight = sum(
        float(np.asarray(weights[episode.episode_id]).sum()) for episode in episodes
    )
    if total_weight <= 0.0:
        raise ValueError("AWR actor has no positive optimization weight")
    history = []
    for epoch in range(1, config.epochs + 1):
        gradient = np.zeros(width, dtype=np.float64)
        weighted_nll = 0.0
        reference_kl = 0.0
        maximum_probability_error = 0.0
        for episode, values in zip(episodes, arrays, strict=True):
            (
                features,
                sizes,
                groups,
                baselines,
                factual,
                reference,
                supported,
            ) = values
            option_weights = np.asarray(weights[episode.episode_id], dtype=np.float64)
            if (
                len(option_weights) != episode.option_count
                or np.any(option_weights < 0.0)
                or not np.all(np.isfinite(option_weights))
            ):
                raise ValueError("AWR option weights are invalid")
            scores = features @ coefficients
            raw_delta = (scores - scores[baselines][groups]) / policy.temperature
            eligible = supported.copy()
            eligible[baselines] = False
            clipped = np.zeros_like(raw_delta)
            clipped[eligible] = np.clip(
                raw_delta[eligible],
                -policy.maximum_log_tilt,
                policy.maximum_log_tilt,
            )
            active = eligible & (
                np.abs(raw_delta) < policy.maximum_log_tilt
            )
            unnormalized = reference * np.exp(clipped)
            normalizers = np.add.reduceat(unnormalized, episode.offsets[:-1])
            probabilities = unnormalized / normalizers[groups]
            maximum_probability_error = max(
                maximum_probability_error,
                float(np.max(np.abs(
                    np.add.reduceat(probabilities, episode.offsets[:-1]) - 1.0
                ))),
            )
            log_probabilities = np.log(probabilities)
            weighted_nll += float(
                np.sum(option_weights * -log_probabilities[factual])
            )
            reference_kl += float(np.sum(
                reference * (np.log(reference) - log_probabilities)
            ))
            coefficient_gradient = (
                option_weights[groups] / total_weight * probabilities
                + config.kl_coefficient / total_options
                * (probabilities - reference)
            )
            coefficient_gradient[factual] -= option_weights / total_weight
            h = coefficient_gradient * active / policy.temperature
            group_sums = np.add.reduceat(h, episode.offsets[:-1])
            gradient += features.T @ h - features[baselines].T @ group_sums
        objective = (
            weighted_nll / total_weight
            + config.kl_coefficient * reference_kl / total_options
            + 0.5 * config.l2 * float(coefficients @ coefficients)
        )
        gradient += config.l2 * coefficients
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        corrected_first = first_moment / (1.0 - beta1**epoch)
        corrected_second = second_moment / (1.0 - beta2**epoch)
        coefficients -= config.learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )
        history.append({
            "epoch": epoch,
            "objective": objective,
            "weighted_nll": weighted_nll / total_weight,
            "reference_kl": reference_kl / total_options,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "coefficient_norm": float(np.linalg.norm(coefficients)),
            "maximum_probability_sum_error": maximum_probability_error,
        })
    if any(row["objective"] < 0.0 for row in history):
        raise RuntimeError("proper AWR objective crossed its finite lower bound")
    return LinearAwrActor(
        feature_indices=feature_indices,
        coefficients=tuple(map(float, coefficients)),
    ), {
        "schema": "generation7-linear-awr-fit-v1",
        "feature_count": width,
        "epochs": history,
        "minimum_weight": min(
            float(min(weights[episode.episode_id])) for episode in episodes
        ),
        "maximum_weight": max(
            float(max(weights[episode.episode_id])) for episode in episodes
        ),
    }


def crossfit_proper_awr(
    episodes: tuple[CompactEpisode, ...],
    *,
    feature_names: tuple[str, ...],
    orthogonal_config: OrthogonalConfig,
    awr_config: AwrConfig,
) -> dict[str, object]:
    import numpy as np
    folds = episode_folds(
        episodes,
        folds=orthogonal_config.folds,
        seed=orthogonal_config.fold_seed,
    )
    sources, stages = _one_hot_nuisance(episodes)
    compact = tuple(_state_and_centered(episode) for episode in episodes)
    policy = ResidualStochasticPolicy(
        epsilon=orthogonal_config.reference_epsilon,
        temperature=orthogonal_config.policy_temperature,
        maximum_log_tilt=orthogonal_config.maximum_log_tilt,
    )
    estimates = {
        name: defaultdict(list)
        for name in (
            "one_step_direct",
            "one_step_ips",
            "one_step_dr",
            "one_step_fqe",
            "sequential_fqe",
            "sequential_dr",
        )
    }
    losses = defaultdict(list)
    fold_reports = []
    for held_fold in range(orthogonal_config.folds):
        train_indices = tuple(index for index, fold in enumerate(folds) if fold != held_fold)
        held_indices = tuple(index for index, fold in enumerate(folds) if fold == held_fold)
        train_states = np.concatenate(tuple(
            _append_nuisance(compact[index][0], episodes[index], sources, stages)
            for index in train_indices
        ))
        train_targets = np.concatenate(tuple(episodes[index].targets for index in train_indices))
        nuisance = ridge_pipeline(
            alpha=orthogonal_config.nuisance_ridge_alpha
        ).fit(train_states, train_targets)
        effect, _train_effect_centered = _fit_effect_model(
            episodes,
            compact,
            train_indices=train_indices,
            residual=train_targets - nuisance.predict(train_states),
            config=orthogonal_config,
        )
        action_counts = Counter(
            int(action) for index in train_indices for action in compact[index][3]
        )
        action_episodes = Counter()
        for index in train_indices:
            for action in set(map(int, compact[index][3])):
                action_episodes[action] += 1
        supported_actions = frozenset(
            action for action in range(len(ACTION_NAMES))
            if action_counts[action] >= orthogonal_config.minimum_action_assignments
            and action_episodes[action] >= orthogonal_config.minimum_action_episodes
        )
        weights = {}
        for index in train_indices:
            episode = episodes[index]
            factual_scores, baseline_scores = effect.factual_baseline_predictions(
                episode, compact[index][0]
            )
            reward_advantages = baseline_scores - factual_scores
            weights[episode.episode_id] = np.asarray([
                awr_weight(
                    float(advantage),
                    temperature=awr_config.temperature,
                    maximum_weight=awr_config.maximum_weight,
                )
                for advantage in reward_advantages
            ], dtype=np.float64)
        actor, actor_report = fit_linear_awr_actor(
            tuple(episodes[index] for index in train_indices),
            weights=weights,
            feature_names=feature_names,
            supported_actions=supported_actions,
            policy=policy,
            config=awr_config,
        )
        all_indices = (*train_indices, *held_indices)
        factual_rows = {
            index: _append_nuisance(
                compact[index][2], episodes[index], sources, stages
            )
            for index in all_indices
        }
        behavior_expected_rows = {
            index: _append_nuisance(
                _behavior_expected_features(episodes[index]),
                episodes[index],
                sources,
                stages,
            )
            for index in all_indices
        }
        target_expected_rows = {}
        reference_expected_rows = {}
        target_candidate_probabilities = {}
        reference_candidate_probabilities = {}
        for index in all_indices:
            target_rows, reference_rows, target_probabilities, reference_probabilities_ = (
                _expected_actor_policy_features(
                    episodes[index],
                    actor=actor,
                    supported_actions=supported_actions,
                    policy=policy,
                )
            )
            target_expected_rows[index] = _append_nuisance(
                target_rows, episodes[index], sources, stages
            )
            reference_expected_rows[index] = _append_nuisance(
                reference_rows, episodes[index], sources, stages
            )
            target_candidate_probabilities[index] = target_probabilities
            reference_candidate_probabilities[index] = reference_probabilities_
        fqe = evaluate_fqe_crosschecks(
            episodes,
            train_indices=train_indices,
            held_indices=held_indices,
            factual_rows=factual_rows,
            behavior_expected_rows=behavior_expected_rows,
            target_expected_rows=target_expected_rows,
            reference_expected_rows=reference_expected_rows,
            target_candidate_probabilities=target_candidate_probabilities,
            reference_candidate_probabilities=reference_candidate_probabilities,
            horizon=orthogonal_config.fqe_horizon,
            ridge_alpha=orthogonal_config.fqe_ridge_alpha,
        )
        for name, by_episode in fqe["estimates"].items():
            for episode_id, values_ in by_episode.items():
                estimates[name][episode_id].extend(values_)
        for index in held_indices:
            episode = episodes[index]
            state_rows = _append_nuisance(compact[index][0], episode, sources, stages)
            state_values = nuisance.predict(state_rows)
            effect_states = effect.state_rows(episode, compact[index][0])
            for option in range(episode.option_count):
                start, stop = int(episode.offsets[option]), int(episode.offsets[option + 1])
                features = episode.features[start:stop]
                raw_effects = effect.predict_candidates(
                    features, effect_states[option]
                )
                mu = episode.behavior_probabilities[start:stop]
                q_values = state_values[option] + raw_effects - float(mu @ raw_effects)
                baseline_position = int(episode.baseline_positions[option])
                factual_position = int(episode.factual_positions[option])
                pi = target_candidate_probabilities[index][start:stop]
                reference = reference_candidate_probabilities[index][start:stop]
                target = float(episode.targets[option])
                factual_probability = float(mu[factual_position])
                direct = float((pi - reference) @ q_values)
                ips = float((pi[factual_position] - reference[factual_position]) / factual_probability * target)
                dr = float(direct + (pi[factual_position] - reference[factual_position]) / factual_probability * (target - q_values[factual_position]))
                estimates["one_step_direct"][episode.episode_id].append(direct)
                estimates["one_step_ips"][episode.episode_id].append(ips)
                estimates["one_step_dr"][episode.episode_id].append(dr)
                losses[episode.episode_id].append(-math.log(pi[factual_position]))
        fold_reports.append({
            "fold": held_fold,
            "training_episodes": len(train_indices),
            "heldout_episodes": len(held_indices),
            "supported_actions": [ACTION_NAMES[index] for index in sorted(supported_actions)],
            "actor_fit": actor_report,
            "sequential_dr_cumulative_weights": fqe[
                "cumulative_weight_diagnostics"
            ],
        })
    return {
        "schema": "generation7-proper-awr-crossfit-v1",
        "folds": list(folds),
        "fold_reports": fold_reports,
        "heldout_negative_log_likelihood": _cluster_summary(losses),
        "estimates": {name: _cluster_summary(rows) for name, rows in estimates.items()},
    }
