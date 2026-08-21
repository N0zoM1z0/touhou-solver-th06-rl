"""Offline-only Generation-7 fit orchestration; no Wine/runtime imports."""

from __future__ import annotations

import hashlib
import math

from th06_rl.g7_contract import CANDIDATE_SCHEMA
from th06_rl.g7_forecast import build_forecast_artifact
from th06_rl.g7_learner import (
    build_critic_dataset,
    cross_fit_cost_critic,
    fit_linear_awr_actor,
)
from th06_rl.g7_support import fit_local_support

def _episode_subsets(
    episodes,
    *,
    members: int,
    fraction: float,
    seed: int,
    minimum_episodes: int,
):
    rows = tuple(episodes)
    if (
        members < 3
        or not 0.5 <= fraction < 1.0
        or not 0 <= seed < 2**64 - 10_000
        or minimum_episodes < 2
        or len(rows) < minimum_episodes + 1
    ):
        raise ValueError("actor ensemble subset contract is invalid")
    count = max(minimum_episodes, math.ceil(len(rows) * fraction))
    count = min(len(rows) - 1, count)
    result = []
    seen = set()
    for member in range(members):
        ordered = sorted(
            rows,
            key=lambda episode: hashlib.sha256(
                f"{seed}:{member}:{episode[0].episode_id}".encode("utf-8")
            ).digest(),
        )
        subset = tuple(ordered[:count])
        identity = tuple(sorted(episode[0].episode_id for episode in subset))
        if identity in seen:
            raise ValueError("actor ensemble produced duplicate episode subsets")
        seen.add(identity)
        result.append(subset)
    return tuple(result)


def fit_g7_candidate(
    training_episodes,
    *,
    seed: int,
    reference_epsilon: float,
    awr_temperature: float,
    crossfit_folds: int,
    critic_estimators: int,
    n_jobs: int,
    maximum_importance_ratio: float,
    support_prototypes_per_action: int,
    support_distance_quantile: float,
    support_minimum_samples: int,
    support_minimum_ess: float,
    ensemble_members: int,
    ensemble_episode_fraction: float,
    required_vote_fraction: float = 1.0,
) -> dict[str, object]:
    """Fit only on a caller-supplied physical-episode training partition."""
    episodes = tuple(tuple(episode) for episode in training_episodes)
    if (
        not 0.0 < reference_epsilon <= 1.0
        or not math.isfinite(reference_epsilon)
        or not 0 <= seed < 2**64 - 10_000
        or crossfit_folds < 2
        or len(episodes) <= crossfit_folds
    ):
        raise ValueError("Generation-7 fit bounds are invalid")
    dataset = build_critic_dataset(
        episodes,
        reference_epsilon=reference_epsilon,
    )
    critic = cross_fit_cost_critic(
        dataset,
        folds=crossfit_folds,
        seed=seed,
        max_importance_ratio=maximum_importance_ratio,
        n_jobs=n_jobs,
        n_estimators=critic_estimators,
    )
    actor = fit_linear_awr_actor(
        dataset,
        critic,
        reference_epsilon=reference_epsilon,
        temperature=awr_temperature,
    )
    support = fit_local_support(
        dataset,
        seed=seed + 1,
        prototypes_per_action=support_prototypes_per_action,
        distance_quantile=support_distance_quantile,
        minimum_samples=support_minimum_samples,
        minimum_ess=support_minimum_ess,
    )
    subsets = _episode_subsets(
        episodes,
        members=ensemble_members,
        fraction=ensemble_episode_fraction,
        seed=seed + 2,
        minimum_episodes=crossfit_folds + 1,
    )
    ensemble_actors = []
    ensemble_metrics = []
    for member, subset in enumerate(subsets):
        member_dataset = build_critic_dataset(
            subset,
            reference_epsilon=reference_epsilon,
        )
        member_critic = cross_fit_cost_critic(
            member_dataset,
            folds=crossfit_folds,
            seed=seed + 1000 + member,
            max_importance_ratio=maximum_importance_ratio,
            n_jobs=n_jobs,
            n_estimators=critic_estimators,
        )
        ensemble_actors.append(fit_linear_awr_actor(
            member_dataset,
            member_critic,
            reference_epsilon=reference_epsilon,
            temperature=awr_temperature,
        ))
        ensemble_metrics.append({
            "member": member,
            "episode_ids": sorted(episode[0].episode_id for episode in subset),
            "episodes": member_critic.episode_count,
            "examples": member_critic.example_count,
            "factual_rmse": member_critic.factual_rmse,
            "maximum_importance_ratio": member_critic.maximum_importance_ratio,
        })
    forecast = build_forecast_artifact(
        ensemble_actors,
        required_vote_fraction=required_vote_fraction,
    )
    return {
        "schema": CANDIDATE_SCHEMA,
        "authorization": "offline-research-only",
        "actor": actor,
        "local_support": support,
        "forecast": forecast,
        "fit": {
            "seed": seed,
            "episode_ids": list(dataset.episode_ids),
            "episodes": critic.episode_count,
            "eligible_examples": critic.example_count,
            "excluded_options": dataset.excluded_options,
            "exclusion_reasons": [list(row) for row in dataset.exclusion_reasons],
            "factual_rmse": critic.factual_rmse,
            "maximum_importance_ratio": critic.maximum_importance_ratio,
            "reference_epsilon": reference_epsilon,
            "awr_temperature": awr_temperature,
            "crossfit_folds": crossfit_folds,
            "critic_estimators": critic_estimators,
            "ensemble_episode_fraction": ensemble_episode_fraction,
            "ensemble": ensemble_metrics,
        },
    }
