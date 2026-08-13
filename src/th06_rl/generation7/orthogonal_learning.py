"""Episode-cross-fitted baseline-relative orthogonal/direct improvement."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from statistics import mean, stdev

from ..actions import ACTION_NAMES
from ..th06.learning_adapter import ACTION_FEATURE_NAMES, OBSERVATION_FEATURE_NAMES
from .offline_dataset import EpisodeArrayPaths, load_episode_arrays, proximal_targets
from .fqe import evaluate_fqe_crosschecks
from .feature_contract import compact_actor_feature_names
from .linear_models import ridge_pipeline
from .policy_distribution import ResidualStochasticPolicy, reference_probabilities


@dataclass(frozen=True)
class OrthogonalConfig:
    folds: int
    fold_seed: int
    horizon: int
    nuisance_ridge_alpha: float
    effect_ridge_alpha: float
    reference_epsilon: float
    policy_temperature: float
    maximum_log_tilt: float
    minimum_action_assignments: int
    minimum_action_episodes: int
    fqe_horizon: int
    fqe_ridge_alpha: float
    effect_representation: str = "action_only"

    def __post_init__(self) -> None:
        if (
            self.folds < 2
            or self.horizon <= 0
            or min(self.nuisance_ridge_alpha, self.effect_ridge_alpha) <= 0.0
            or not 0.0 < self.reference_epsilon <= 1.0
            or self.policy_temperature <= 0.0
            or self.maximum_log_tilt <= 0.0
            or min(self.minimum_action_assignments, self.minimum_action_episodes) <= 0
            or self.fqe_horizon <= 0
            or self.fqe_ridge_alpha <= 0.0
            or self.effect_representation not in {
                "action_only",
                "compact_bilinear",
                "richer_bilinear",
            }
        ):
            raise ValueError("orthogonal learner configuration is invalid")


@dataclass(frozen=True)
class CompactEpisode:
    episode_id: str
    source_id: str
    cohort_id: str
    stage: int
    features: object
    causal_context_features: object
    action_indices: object
    offsets: object
    factual_positions: object
    baseline_positions: object
    behavior_probabilities: object
    targets: object
    hit_costs: object

    @property
    def option_count(self) -> int:
        return len(self.factual_positions)


def load_compact_episodes(
    paths: tuple[EpisodeArrayPaths, ...],
    *,
    horizon: int,
) -> tuple[CompactEpisode, ...]:
    result = []
    for item in paths:
        arrays = load_episode_arrays(item)
        result.append(CompactEpisode(
            episode_id=str(arrays["episode_id"]),
            source_id=str(arrays["source_id"]),
            cohort_id=str(arrays["cohort_id"]),
            stage=int(arrays["stage"]),
            features=arrays["candidate_features"],
            causal_context_features=arrays["causal_context_features"],
            action_indices=arrays["candidate_action_indices"],
            offsets=arrays["offsets"],
            factual_positions=arrays["factual_positions"],
            baseline_positions=arrays["baseline_positions"],
            behavior_probabilities=arrays["behavior_probabilities"],
            targets=proximal_targets(arrays["hit_costs"], horizon),
            hit_costs=arrays["hit_costs"].astype("float64"),
        ))
    if len({episode.episode_id for episode in result}) != len(result):
        raise ValueError("duplicate physical episode group")
    return tuple(result)


def episode_folds(
    episodes: tuple[CompactEpisode, ...],
    *,
    folds: int,
    seed: int,
) -> tuple[int, ...]:
    if folds < 2 or len(episodes) < folds:
        raise ValueError("whole-episode fold count is invalid")
    ordered = sorted(
        range(len(episodes)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{episodes[index].episode_id}".encode()
        ).digest(),
    )
    assignments = [-1] * len(episodes)
    for rank, index in enumerate(ordered):
        assignments[index] = rank % folds
    return tuple(assignments)


def _state_and_centered(episode: CompactEpisode):
    import numpy as np

    observation_count = len(OBSERVATION_FEATURE_NAMES)
    action_count = len(ACTION_FEATURE_NAMES)
    state_indices = np.asarray(
        [*range(observation_count + action_count),
         episode.features.shape[1] - 1],
        dtype=np.int64,
    )
    states = np.empty(
        (episode.option_count, len(state_indices)), dtype=np.float64
    )
    centered = np.empty(
        (episode.option_count, episode.features.shape[1]), dtype=np.float64
    )
    factual_features = np.empty_like(centered)
    factual_actions = np.empty(episode.option_count, dtype=np.int16)
    factual_probabilities = np.empty(episode.option_count, dtype=np.float64)
    for option in range(episode.option_count):
        start = int(episode.offsets[option])
        stop = int(episode.offsets[option + 1])
        factual = start + int(episode.factual_positions[option])
        baseline = start + int(episode.baseline_positions[option])
        probabilities = episode.behavior_probabilities[start:stop]
        rows = episode.features[start:stop].astype(np.float64, copy=False)
        expectation = probabilities @ rows
        factual_features[option] = episode.features[factual]
        centered[option] = factual_features[option] - expectation
        states[option] = episode.features[baseline, state_indices]
        factual_actions[option] = episode.action_indices[factual]
        factual_probabilities[option] = probabilities[
            int(episode.factual_positions[option])
        ]
    return states, centered, factual_features, factual_actions, factual_probabilities


def _one_hot_nuisance(
    episodes: tuple[CompactEpisode, ...],
) -> tuple[dict[str, int], dict[int, int]]:
    sources = {
        name: index for index, name in enumerate(sorted({row.source_id for row in episodes}))
    }
    stages = {
        stage: index for index, stage in enumerate(sorted({row.stage for row in episodes}))
    }
    return sources, stages


def _append_nuisance(rows, episode, sources, stages):
    import numpy as np

    result = np.zeros(
        (len(rows), rows.shape[1] + len(sources) + len(stages)),
        dtype=np.float64,
    )
    result[:, :rows.shape[1]] = rows
    result[:, rows.shape[1] + sources[episode.source_id]] = 1.0
    result[:, rows.shape[1] + len(sources) + stages[episode.stage]] = 1.0
    return result


def _cluster_summary(values: dict[str, list[float]]) -> dict[str, float | int]:
    episode_means = tuple(mean(rows) for rows in values.values() if rows)
    flat = tuple(value for rows in values.values() for value in rows)
    if not flat:
        raise ValueError("policy estimate has no rows")
    return {
        "episodes": len(episode_means),
        "rows": len(flat),
        "row_mean": mean(flat),
        "episode_equal_mean": mean(episode_means),
        "episode_cluster_standard_error": (
            stdev(episode_means) / math.sqrt(len(episode_means))
            if len(episode_means) > 1 else math.inf
        ),
    }


def _paired_cluster_difference(left, right):
    if set(left) != set(right):
        raise ValueError("paired estimators cover different episodes")
    differences = {}
    for episode_id in left:
        if len(left[episode_id]) != len(right[episode_id]):
            raise ValueError("paired estimators cover different option rows")
        differences[episode_id] = [
            float(left_value - right_value)
            for left_value, right_value in zip(
                left[episode_id], right[episode_id], strict=True
            )
        ]
    return _cluster_summary(differences)


def _importance_weight_summary(values) -> dict[str, float | int]:
    import numpy as np

    rows = np.asarray(values, dtype=np.float64)
    if not len(rows) or np.any(~np.isfinite(rows)) or np.any(rows < 0.0):
        raise ValueError("importance weights are invalid")
    square_sum = float(rows @ rows)
    total = float(rows.sum())
    return {
        "count": len(rows),
        "mean": float(rows.mean()),
        "maximum": float(rows.max()),
        "effective_sample_size": (
            total * total / square_sum if square_sum > 0.0 else 0.0
        ),
    }


def proposal_propensity_calibration(
    episodes: tuple[CompactEpisode, ...],
    *,
    reference_epsilon: float,
) -> dict[str, object]:
    """Check the randomized proposal law before fitting any outcome model.

    Under a correct logged propensity, the importance ratio from the behavior
    proposal law to any fixed supported policy has expectation one.  The
    shared reference is used here because it is defined for every native-safe
    candidate set.  This gate would expose the old post-assignment compliance
    filter without looking at rewards.
    """
    import numpy as np

    if not episodes or not 0.0 < reference_epsilon <= 1.0:
        raise ValueError("proposal propensity calibration contract is invalid")
    values: dict[str, list[float]] = {}
    for episode in episodes:
        sizes = np.diff(episode.offsets).astype(np.float64)
        factual = np.asarray(episode.factual_positions, dtype=np.int64)
        baseline = np.asarray(episode.baseline_positions, dtype=np.int64)
        factual_rows = episode.offsets[:-1] + factual
        reference = (
            reference_epsilon / sizes
            + (1.0 - reference_epsilon) * (factual == baseline)
        )
        ratios = reference / episode.behavior_probabilities[factual_rows]
        if np.any(~np.isfinite(ratios)) or np.any(ratios < 0.0):
            raise ValueError("proposal propensity ratio is invalid")
        values[episode.episode_id] = ratios.tolist()
    return {
        "policy": "incumbent-uniform-reference",
        "expected_mean": 1.0,
        "aggregate": _cluster_summary(values),
        "strata": _stratified_summaries(episodes, values),
    }


def _stratified_summaries(episodes, values):
    result = {}
    strata = {
        **{
            f"source:{source}": frozenset(
                episode.episode_id
                for episode in episodes if episode.source_id == source
            )
            for source in sorted({episode.source_id for episode in episodes})
        },
        **{
            f"stage:{stage}": frozenset(
                episode.episode_id
                for episode in episodes if episode.stage == stage
            )
            for stage in sorted({episode.stage for episode in episodes})
        },
    }
    for name, episode_ids in strata.items():
        selected = {
            episode_id: rows
            for episode_id, rows in values.items()
            if episode_id in episode_ids and rows
        }
        if selected:
            result[name] = _cluster_summary(selected)
    return result


def _expected_policy_features(
    episode: CompactEpisode,
    *,
    compact_state,
    effect,
    policy: ResidualStochasticPolicy,
    fold_action_counts: Counter,
    fold_action_episodes: Counter,
    config: OrthogonalConfig,
):
    import numpy as np

    q_candidate_rows = effect.q_candidate_rows(episode, compact_state)
    policy_rows = np.empty(
        (episode.option_count, q_candidate_rows.shape[1]), dtype=np.float64
    )
    reference_rows = np.empty_like(policy_rows)
    policy_probabilities = np.empty(len(episode.features), dtype=np.float64)
    reference_probability_rows = np.empty_like(policy_probabilities)
    effect_states = effect.state_rows(episode, compact_state)
    for option in range(episode.option_count):
        start = int(episode.offsets[option])
        stop = int(episode.offsets[option + 1])
        features = episode.features[start:stop].astype(np.float64, copy=False)
        actions = tuple(
            ACTION_NAMES[int(value)] for value in episode.action_indices[start:stop]
        )
        baseline_position = int(episode.baseline_positions[option])
        baseline_action = actions[baseline_position]
        supported = tuple(
            fold_action_counts[int(action)] >= config.minimum_action_assignments
            and fold_action_episodes[int(action)] >= config.minimum_action_episodes
            for action in episode.action_indices[start:stop]
        )
        effects = effect.predict_candidates(features, effect_states[option])
        decision = policy.distribution(
            safe_actions=actions,
            baseline_action=baseline_action,
            logits=tuple(-float(value) for value in effects),
            statistically_supported=supported,
            forecast_risky=(False,) * len(actions),
        )
        pi = np.asarray(decision.probabilities, dtype=np.float64)
        reference = np.asarray(reference_probabilities(
            actions,
            baseline_action,
            epsilon=config.reference_epsilon,
        ))
        policy_rows[option] = pi @ q_candidate_rows[start:stop]
        reference_rows[option] = reference @ q_candidate_rows[start:stop]
        policy_probabilities[start:stop] = pi
        reference_probability_rows[start:stop] = reference
    return (
        policy_rows,
        reference_rows,
        policy_probabilities,
        reference_probability_rows,
    )


def _expected_candidate_rows(episode: CompactEpisode, candidate_rows):
    import numpy as np

    rows = np.empty(
        (episode.option_count, candidate_rows.shape[1]), dtype=np.float64
    )
    for option in range(episode.option_count):
        start = int(episode.offsets[option])
        stop = int(episode.offsets[option + 1])
        rows[option] = (
            episode.behavior_probabilities[start:stop]
            @ candidate_rows[start:stop]
        )
    return rows


_ACTION_CORE_NAMES = (
    "action:direction_x",
    "action:direction_y",
    "action:focused",
    "action:stationary",
    "action:diagonal",
)
_ACTION_CORE_INDICES = tuple(
    compact_actor_feature_names().index(name) for name in _ACTION_CORE_NAMES
)


class ContextualEffectModel:
    """Ridge effect model with an optional small online bilinear scorer."""

    def __init__(self, estimator, *, representation: str) -> None:
        self.estimator = estimator
        self.representation = representation

    def state_rows(self, episode: CompactEpisode, compact_state):
        import numpy as np

        if self.representation == "action_only":
            return np.empty((episode.option_count, 0), dtype=np.float32)
        state = np.asarray(compact_state, dtype=np.float32)
        if self.representation == "richer_bilinear":
            state = np.concatenate((
                state,
                np.asarray(episode.causal_context_features, dtype=np.float32),
            ), axis=1)
        return state

    def nuisance_state_rows(self, episode: CompactEpisode, compact_state):
        import numpy as np

        state = np.asarray(compact_state, dtype=np.float32)
        if self.representation == "richer_bilinear":
            state = np.concatenate((
                state,
                np.asarray(episode.causal_context_features, dtype=np.float32),
            ), axis=1)
        return state

    def _rows(self, candidates, state_row):
        import numpy as np

        base = np.asarray(candidates, dtype=np.float32)
        if self.representation == "action_only":
            return base
        core = base[:, _ACTION_CORE_INDICES]
        interactions = np.einsum(
            "ij,k->ikj", core, np.asarray(state_row, dtype=np.float32)
        ).reshape(len(base), -1)
        return np.concatenate((base, interactions), axis=1)

    def predict_option(self, episode: CompactEpisode, compact_state, option: int):
        start = int(episode.offsets[option])
        stop = int(episode.offsets[option + 1])
        state = self.state_rows(episode, compact_state)[option]
        return self.predict_candidates(episode.features[start:stop], state)

    def predict_candidates(self, candidates, state_row):
        return self.estimator.predict(self._rows(candidates, state_row))

    def q_candidate_rows(self, episode: CompactEpisode, compact_state):
        """Return the Q design corresponding to this causal representation."""
        import numpy as np

        states = self.state_rows(episode, compact_state)
        sizes = np.diff(episode.offsets).astype(np.int64)
        groups = np.repeat(np.arange(episode.option_count), sizes)
        base = np.asarray(episode.features, dtype=np.float32)
        if self.representation == "action_only":
            return base
        core = base[:, _ACTION_CORE_INDICES]
        interactions = np.einsum(
            "ij,ik->ijk", states[groups], core
        ).reshape(len(base), -1)
        rows = np.concatenate((base, interactions), axis=1)
        if self.representation == "richer_bilinear":
            rows = np.concatenate((
                rows,
                np.asarray(
                    episode.causal_context_features, dtype=np.float32
                )[groups],
            ), axis=1)
        return rows

    def factual_baseline_predictions(self, episode: CompactEpisode, compact_state):
        import numpy as np

        factual = episode.offsets[:-1] + episode.factual_positions
        baseline = episode.offsets[:-1] + episode.baseline_positions
        states = self.state_rows(episode, compact_state)
        if self.representation == "action_only":
            return (
                self.estimator.predict(episode.features[factual]),
                self.estimator.predict(episode.features[baseline]),
            )
        factual_rows = np.concatenate((
            episode.features[factual],
            np.einsum(
                "ij,ik->ijk",
                states,
                episode.features[factual][:, _ACTION_CORE_INDICES],
            ).reshape(episode.option_count, -1),
        ), axis=1)
        baseline_rows = np.concatenate((
            episode.features[baseline],
            np.einsum(
                "ij,ik->ijk",
                states,
                episode.features[baseline][:, _ACTION_CORE_INDICES],
            ).reshape(episode.option_count, -1),
        ), axis=1)
        return self.estimator.predict(factual_rows), self.estimator.predict(baseline_rows)


def _effect_centered_rows(
    episode: CompactEpisode,
    compact_state,
    base_centered,
    *,
    representation: str,
):
    import numpy as np

    base = np.asarray(base_centered, dtype=np.float32)
    if representation == "action_only":
        return base
    state = np.asarray(compact_state, dtype=np.float32)
    if representation == "richer_bilinear":
        state = np.concatenate((
            state,
            np.asarray(episode.causal_context_features, dtype=np.float32),
        ), axis=1)
    interactions = np.einsum(
        "ij,ik->ijk", state, base[:, _ACTION_CORE_INDICES]
    ).reshape(episode.option_count, -1)
    return np.concatenate((base, interactions), axis=1)


def _fit_effect_model(
    episodes: tuple[CompactEpisode, ...],
    compact,
    *,
    train_indices: tuple[int, ...],
    residual,
    config: OrthogonalConfig,
) -> tuple[ContextualEffectModel, object]:
    import numpy as np
    centered = np.concatenate(tuple(
        _effect_centered_rows(
            episodes[index],
            compact[index][0],
            compact[index][1],
            representation=config.effect_representation,
        )
        for index in train_indices
    ))
    estimator = ridge_pipeline(alpha=config.effect_ridge_alpha).fit(
        centered, residual
    )
    return (
        ContextualEffectModel(
            estimator, representation=config.effect_representation
        ),
        centered,
    )


def crossfit_orthogonal_policy(
    episodes: tuple[CompactEpisode, ...],
    *,
    config: OrthogonalConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit effects on training episodes and evaluate exact policies held out."""
    import numpy as np
    folds = episode_folds(episodes, folds=config.folds, seed=config.fold_seed)
    sources, stages = _one_hot_nuisance(episodes)
    compact = tuple(_state_and_centered(episode) for episode in episodes)
    policy = ResidualStochasticPolicy(
        epsilon=config.reference_epsilon,
        temperature=config.policy_temperature,
        maximum_log_tilt=config.maximum_log_tilt,
    )
    representation = ContextualEffectModel(
        estimator=None, representation=config.effect_representation
    )
    estimators = {
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
    fold_reports = []
    nuisance_residuals: dict[str, object] = {}
    centered_features: dict[str, object] = {}
    effect_states_by_episode: dict[str, object] = {}
    action_counts = Counter()
    action_episode_counts = Counter()
    distribution_mass = Counter()
    reference_mass = Counter()
    proposal_mass = []
    policy_kls = []
    action_direct = {action: defaultdict(list) for action in ACTION_NAMES}

    for held_fold in range(config.folds):
        train_indices = tuple(
            index for index, fold in enumerate(folds) if fold != held_fold
        )
        held_indices = tuple(
            index for index, fold in enumerate(folds) if fold == held_fold
        )
        train_states = np.concatenate(tuple(
            _append_nuisance(
                representation.nuisance_state_rows(
                    episodes[index], compact[index][0]
                ),
                episodes[index],
                sources,
                stages,
            )
            for index in train_indices
        ))
        train_targets = np.concatenate(tuple(
            episodes[index].targets for index in train_indices
        ))
        train_factual_actions = tuple(compact[index][3] for index in train_indices)
        fold_action_counts = Counter(
            int(action) for rows in train_factual_actions for action in rows
        )
        fold_action_episodes = Counter()
        for index in train_indices:
            for action in set(map(int, compact[index][3])):
                fold_action_episodes[action] += 1
        action_counts.update(fold_action_counts)
        action_episode_counts.update(fold_action_episodes)

        nuisance = ridge_pipeline(alpha=config.nuisance_ridge_alpha)
        nuisance.fit(train_states, train_targets)
        train_residual = train_targets - nuisance.predict(train_states)
        effect, _train_effect_centered = _fit_effect_model(
            episodes,
            compact,
            train_indices=train_indices,
            residual=train_residual,
            config=config,
        )
        all_indices = (*train_indices, *held_indices)
        q_candidate_rows = {
            index: effect.q_candidate_rows(episodes[index], compact[index][0])
            for index in all_indices
        }
        factual_rows = {
            index: _append_nuisance(
                q_candidate_rows[index][
                    episodes[index].offsets[:-1]
                    + episodes[index].factual_positions
                ],
                episodes[index],
                sources,
                stages,
            )
            for index in all_indices
        }
        behavior_expected_rows = {
            index: _append_nuisance(
                _expected_candidate_rows(
                    episodes[index], q_candidate_rows[index]
                ),
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
                _expected_policy_features(
                    episodes[index],
                    compact_state=compact[index][0],
                    effect=effect,
                    policy=policy,
                    fold_action_counts=fold_action_counts,
                    fold_action_episodes=fold_action_episodes,
                    config=config,
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
            horizon=config.fqe_horizon,
            ridge_alpha=config.fqe_ridge_alpha,
        )
        for name, by_episode in fqe["estimates"].items():
            for episode_id, values_ in by_episode.items():
                estimators[name][episode_id].extend(values_)
        fold_direct = []
        fold_dr = []
        fold_target_ratios = []
        fold_reference_ratios = []
        fold_contrast_weights = []
        maximum_absolute_effect_prediction = 0.0

        for index in held_indices:
            episode = episodes[index]
            states, centered, _factual, _actions, _factual_probability = compact[index]
            state_rows = _append_nuisance(
                representation.nuisance_state_rows(episode, states),
                episode,
                sources,
                stages,
            )
            nuisance_prediction = nuisance.predict(state_rows)
            residual = episode.targets - nuisance_prediction
            nuisance_residuals[episode.episode_id] = residual
            centered_features[episode.episode_id] = _effect_centered_rows(
                episode,
                states,
                centered,
                representation=config.effect_representation,
            )
            effect_states = effect.state_rows(episode, states)
            effect_states_by_episode[episode.episode_id] = effect_states
            for option in range(episode.option_count):
                start = int(episode.offsets[option])
                stop = int(episode.offsets[option + 1])
                features = episode.features[start:stop].astype(np.float64, copy=False)
                raw_effects = effect.predict_candidates(
                    features, effect_states[option]
                )
                maximum_absolute_effect_prediction = max(
                    maximum_absolute_effect_prediction,
                    float(np.max(np.abs(raw_effects))),
                )
                mu = episode.behavior_probabilities[start:stop]
                q_values = (
                    nuisance_prediction[option]
                    + raw_effects
                    - float(mu @ raw_effects)
                )
                actions = tuple(
                    ACTION_NAMES[int(value)] for value in episode.action_indices[start:stop]
                )
                baseline_position = int(episode.baseline_positions[option])
                factual_position = int(episode.factual_positions[option])
                baseline_action = actions[baseline_position]
                supported = tuple(
                    fold_action_counts[int(action)] >= config.minimum_action_assignments
                    and fold_action_episodes[int(action)] >= config.minimum_action_episodes
                    for action in episode.action_indices[start:stop]
                )
                decision = policy.distribution(
                    safe_actions=actions,
                    baseline_action=baseline_action,
                    logits=tuple(-float(value) for value in q_values),
                    statistically_supported=supported,
                    forecast_risky=(False,) * len(actions),
                )
                pi = np.asarray(decision.probabilities, dtype=np.float64)
                reference = np.asarray(reference_probabilities(
                    actions,
                    baseline_action,
                    epsilon=config.reference_epsilon,
                ))
                target = float(episode.targets[option])
                probability = float(mu[factual_position])
                target_ratio = float(pi[factual_position] / probability)
                reference_ratio = float(
                    reference[factual_position] / probability
                )
                contrast_weight = target_ratio - reference_ratio
                direct = float((pi - reference) @ q_values)
                ips = float(
                    contrast_weight * target
                )
                dr = float(
                    direct
                    + contrast_weight * (target - q_values[factual_position])
                )
                estimators["one_step_direct"][episode.episode_id].append(direct)
                estimators["one_step_ips"][episode.episode_id].append(ips)
                estimators["one_step_dr"][episode.episode_id].append(dr)
                for action, pi_value, reference_value, q_value in zip(
                    actions, pi, reference, q_values, strict=True
                ):
                    action_direct[action][episode.episode_id].append(float(
                        (pi_value - reference_value) * q_value
                    ))
                fold_direct.append(direct)
                fold_dr.append(dr)
                fold_target_ratios.append(target_ratio)
                fold_reference_ratios.append(reference_ratio)
                fold_contrast_weights.append(abs(contrast_weight))
                proposal_mass.append(1.0 - pi[baseline_position])
                for action, probability_value, reference_value in zip(
                    actions, pi, reference, strict=True
                ):
                    distribution_mass[action] += float(probability_value)
                    reference_mass[action] += float(reference_value)
                policy_kls.append(sum(
                    reference_value * math.log(reference_value / probability_value)
                    for reference_value, probability_value in zip(
                        reference, pi, strict=True
                    )
                ))
        fold_reports.append({
            "fold": held_fold,
            "training_episodes": len(train_indices),
            "heldout_episodes": len(held_indices),
            "heldout_direct_mean": mean(fold_direct),
            "heldout_dr_mean": mean(fold_dr),
            "effect_coefficient_norm": float(np.linalg.norm(
                effect.estimator.named_steps["ridge"].coef_
            )),
            "maximum_absolute_heldout_effect_prediction": (
                maximum_absolute_effect_prediction
            ),
            "sequential_dr_cumulative_weights": fqe[
                "cumulative_weight_diagnostics"
            ],
            "one_step_importance_weights": {
                "target": _importance_weight_summary(fold_target_ratios),
                "reference": _importance_weight_summary(fold_reference_ratios),
                "absolute_contrast": _importance_weight_summary(
                    fold_contrast_weights
                ),
            },
        })

    total_mass = sum(distribution_mass.values())
    report = {
        "schema": "generation7-orthogonal-direct-crossfit-v2",
        "episode_groups": len(episodes),
        "options": sum(episode.option_count for episode in episodes),
        "folds": list(folds),
        "fold_reports": fold_reports,
        "policy": {
            "policy_id": policy.policy_id,
            "reference_epsilon": config.reference_epsilon,
            "temperature": config.policy_temperature,
            "maximum_log_tilt": config.maximum_log_tilt,
            "mean_reference_kl": mean(policy_kls),
            "mean_nonbaseline_probability": mean(proposal_mass),
            "action_probability_mass": {
                action: distribution_mass[action] / total_mass
                for action in ACTION_NAMES
            },
            "reference_action_mass": {
                action: reference_mass[action] / total_mass
                for action in ACTION_NAMES
            },
        },
        "proposal_propensity_calibration": proposal_propensity_calibration(
            episodes,
            reference_epsilon=config.reference_epsilon,
        ),
        "estimates": {
            name: _cluster_summary(rows) for name, rows in estimators.items()
        },
        "paired_calibration_differences": {
            "one_step_direct_minus_dr": _paired_cluster_difference(
                estimators["one_step_direct"], estimators["one_step_dr"]
            ),
            "one_step_ips_minus_dr": _paired_cluster_difference(
                estimators["one_step_ips"], estimators["one_step_dr"]
            ),
            "one_step_fqe_minus_dr": _paired_cluster_difference(
                estimators["one_step_fqe"], estimators["one_step_dr"]
            ),
            "sequential_fqe_minus_dr": _paired_cluster_difference(
                estimators["sequential_fqe"], estimators["sequential_dr"]
            ),
        },
        "strata": {
            name: _stratified_summaries(episodes, rows)
            for name, rows in estimators.items()
        },
        "action_specific_one_step_direct": {
            action: {
                "aggregate": _cluster_summary(rows),
                "strata": _stratified_summaries(episodes, rows),
            }
            for action, rows in action_direct.items() if rows
        },
    }
    diagnostics = {
        "nuisance_residuals": nuisance_residuals,
        "centered_features": centered_features,
        "effect_states": effect_states_by_episode,
        "effect_representation": config.effect_representation,
    }
    return report, diagnostics


def orthogonal_randomization_nulls(
    episodes: tuple[CompactEpisode, ...],
    diagnostics: dict[str, object],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Compare the observed orthogonal score with propensity-aware nulls."""
    import numpy as np

    if replicates < 20:
        raise ValueError("orthogonal null needs at least 20 replicates")
    generator = np.random.default_rng(seed)
    residuals = diagnostics["nuisance_residuals"]
    first_centered = diagnostics["centered_features"][episodes[0].episode_id]
    observed_sum = np.zeros(first_centered.shape[1], dtype=np.float64)
    total = 0
    for episode in episodes:
        centered = diagnostics["centered_features"][episode.episode_id]
        values = residuals[episode.episode_id]
        observed_sum += centered.T @ values
        total += len(values)
    observed_norm = float(np.linalg.norm(observed_sum / total))
    null_designs = {}
    for episode in episodes:
        sizes = np.diff(episode.offsets).astype(np.int64)
        groups = np.repeat(np.arange(episode.option_count), sizes)
        probabilities = np.zeros(
            (episode.option_count, len(ACTION_NAMES)), dtype=np.float64
        )
        candidate_indices = np.full(
            (episode.option_count, len(ACTION_NAMES)), -1, dtype=np.int64
        )
        flat = np.arange(len(episode.features), dtype=np.int64)
        probabilities[groups, episode.action_indices] = (
            episode.behavior_probabilities
        )
        candidate_indices[groups, episode.action_indices] = flat
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0
        factual = episode.offsets[:-1] + episode.factual_positions
        base_width = episode.features.shape[1]
        base_centered = diagnostics["centered_features"][episode.episode_id][
            :, :base_width
        ]
        expectation = episode.features[factual] - base_centered
        null_designs[episode.episode_id] = (
            cumulative,
            candidate_indices,
            expectation,
        )
    action_norms = []
    reward_norms = []
    for _replicate in range(replicates):
        action_sum = np.zeros_like(observed_sum)
        reward_sum = np.zeros_like(observed_sum)
        for episode in episodes:
            values = residuals[episode.episode_id]
            centered = diagnostics["centered_features"][episode.episode_id]
            rotated = np.roll(
                values,
                int(generator.integers(1, len(values))) if len(values) > 1 else 0,
            )
            reward_sum += centered.T @ rotated
            cumulative, candidate_indices, expectation = null_designs[
                episode.episode_id
            ]
            uniforms = generator.random(episode.option_count)
            sampled_actions = np.sum(
                uniforms[:, None] > cumulative, axis=1
            )
            selected = candidate_indices[
                np.arange(episode.option_count), sampled_actions
            ]
            if np.any(selected < 0):
                raise RuntimeError("propensity null selected an unavailable action")
            sampled_base = (
                episode.features[selected].astype(np.float64, copy=False)
                - expectation
            )
            base_width = episode.features.shape[1]
            action_sum[:base_width] += sampled_base.T @ values
            if diagnostics["effect_representation"] != "action_only":
                state = diagnostics["effect_states"][episode.episode_id]
                weighted_core = (
                    sampled_base[:, list(_ACTION_CORE_INDICES)]
                    * values[:, None]
                )
                interaction = (state.T @ weighted_core).reshape(-1)
                action_sum[base_width:] += interaction
        action_norms.append(float(np.linalg.norm(action_sum / total)))
        reward_norms.append(float(np.linalg.norm(reward_sum / total)))
    return {
        "schema": "generation7-orthogonal-randomization-nulls-v1",
        "replicates": replicates,
        "observed_score_norm": observed_norm,
        "action_randomization": {
            "null_mean_norm": float(np.mean(action_norms)),
            "null_p_value": (1 + sum(value >= observed_norm for value in action_norms))
            / (replicates + 1),
        },
        "reward_suffix": {
            "null_mean_norm": float(np.mean(reward_norms)),
            "null_p_value": (1 + sum(value >= observed_norm for value in reward_norms))
            / (replicates + 1),
        },
    }
