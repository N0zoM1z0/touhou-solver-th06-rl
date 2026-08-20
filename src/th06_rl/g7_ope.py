"""Exact-propensity held-out evaluation for a frozen Generation-7 candidate."""

from __future__ import annotations

import math
import random

from th06_rl.g7_forecast import forecast_accepted_actions
from th06_rl.g7_learner import linear_actor_distribution
from th06_rl.g7_support import locally_supported_actions
from th06_rl.g7_training import CANDIDATE_SCHEMA
from th06_rl.offline_options import validate_offline_episode


OPE_SCHEMA = "th06-rl-g7-heldout-pdis-v1"


def _mean(values) -> float:
    rows = tuple(float(value) for value in values)
    if not rows:
        raise ValueError("cannot average empty evidence")
    return sum(rows) / len(rows)


def _ess(weights) -> float:
    rows = tuple(float(value) for value in weights)
    square_sum = sum(value * value for value in rows)
    return sum(rows) ** 2 / square_sum if square_sum else 0.0


def _percentile(rows: list[float], probability: float) -> float:
    ordered = sorted(rows)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _bootstrap_interval(
    values,
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    rows = tuple(float(value) for value in values)
    if (
        not rows
        or not 0.5 < confidence < 1.0
        or resamples < 200
        or not 0 <= seed < 2**64
    ):
        raise ValueError("episode bootstrap contract is invalid")
    generator = random.Random(seed)
    means = [
        _mean(rows[generator.randrange(len(rows))] for _ in rows)
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence) / 2.0
    return _percentile(means, tail), _percentile(means, 1.0 - tail)


def _depth_stratified_null(
    weight_differences_by_depth,
    physical_costs_by_depth,
    *,
    episodes: int,
    observed_difference: float,
    resamples: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    if resamples < 200:
        raise ValueError("association null requires at least 200 permutations")
    generator = random.Random(seed)
    null_values = []
    for _ in range(resamples):
        total = 0.0
        for differences, costs in zip(
            weight_differences_by_depth,
            physical_costs_by_depth,
            strict=True,
        ):
            shuffled = list(differences)
            generator.shuffle(shuffled)
            total += sum(
                difference * cost
                for difference, cost in zip(shuffled, costs, strict=True)
            )
        null_values.append(total / episodes)
    p_value = (
        1 + sum(value <= observed_difference for value in null_values)
    ) / (resamples + 1)
    return p_value, (
        _percentile(null_values, 0.025),
        _percentile(null_values, 0.975),
    )


def evaluate_candidate(
    candidate: dict[str, object],
    validation_episodes,
    *,
    max_kl: float,
    maximum_step_ratio: float,
    maximum_cumulative_ratio: float,
    minimum_effective_sample_size: float,
    minimum_episodes: int,
    confidence: float,
    bootstrap_resamples: int,
    permutation_resamples: int,
    maximum_null_p_value: float,
    seed: int,
) -> dict[str, object]:
    """Paired PDIS on episodes never supplied to the fitter.

    Ratios are neither clipped nor self-normalized. Evidence outside declared
    support bounds is rejected instead of silently biasing the estimate.
    """
    episodes = tuple(tuple(episode) for episode in validation_episodes)
    if (
        candidate.get("schema") != CANDIDATE_SCHEMA
        or candidate.get("authorization") != "offline-research-only"
        or not math.isfinite(max_kl)
        or max_kl < 0.0
        or not math.isfinite(maximum_step_ratio)
        or maximum_step_ratio < 1.0
        or not math.isfinite(maximum_cumulative_ratio)
        or maximum_cumulative_ratio < 1.0
        or not math.isfinite(minimum_effective_sample_size)
        or minimum_effective_sample_size <= 0.0
        or minimum_episodes < 2
        or len(episodes) < minimum_episodes
        or permutation_resamples < 200
        or not math.isfinite(maximum_null_p_value)
        or not 0.0 < maximum_null_p_value < 0.5
    ):
        raise ValueError("held-out OPE contract is invalid")
    actor = candidate.get("actor")
    support_artifact = candidate.get("local_support")
    forecast_artifact = candidate.get("forecast")
    if not all(isinstance(value, dict) for value in (
        actor, support_artifact, forecast_artifact
    )):
        raise ValueError("candidate lacks portable online artifacts")
    assert isinstance(actor, dict)
    assert isinstance(support_artifact, dict)
    assert isinstance(forecast_artifact, dict)

    candidate_values = []
    incumbent_values = []
    observed_values = []
    candidate_final_weights = []
    incumbent_final_weights = []
    candidate_weights_by_depth: list[list[float]] = []
    incumbent_weights_by_depth: list[list[float]] = []
    physical_hits_by_depth: list[int] = []
    physical_costs_by_depth: list[list[int]] = []
    maximum_observed_step_ratio = 0.0
    maximum_observed_cumulative_ratio = 0.0
    nonincumbent_mass = 0.0
    decisions = 0
    episode_ids = set()
    for episode in episodes:
        validate_offline_episode(episode)
        if any(not option.eligible for option in episode):
            raise ValueError("held-out OPE episode contains an ineligible option")
        episode_id = episode[0].episode_id
        if episode_id in episode_ids:
            raise ValueError("held-out OPE repeats a physical episode")
        episode_ids.add(episode_id)
        candidate_weight = 1.0
        incumbent_weight = 1.0
        candidate_value = 0.0
        incumbent_value = 0.0
        observed_value = 0.0
        for depth, option in enumerate(episode):
            supported = locally_supported_actions(support_artifact, option.state)
            forecast = forecast_accepted_actions(
                forecast_artifact,
                option.state,
                supported_actions=supported,
            )
            distribution = linear_actor_distribution(
                actor,
                option.state,
                supported_actions=supported,
                forecast_accepted_actions=forecast,
                max_kl=max_kl,
            )
            target = dict(distribution.probabilities)
            behavior = dict(option.behavior_probabilities)
            if any(
                target[action] > 0.0 and behavior.get(action, 0.0) <= 0.0
                for action in target
            ):
                raise ValueError("candidate is outside logged behavior support")
            factual = option.action
            candidate_ratio = target[factual] / option.behavior_probability
            incumbent_probability = float(factual == option.state.baseline_action)
            incumbent_ratio = incumbent_probability / option.behavior_probability
            if (
                candidate_ratio > maximum_step_ratio + 1e-12
                or incumbent_ratio > maximum_step_ratio + 1e-12
            ):
                raise ValueError("held-out step importance ratio exceeds its bound")
            maximum_observed_step_ratio = max(
                maximum_observed_step_ratio,
                candidate_ratio,
                incumbent_ratio,
            )
            candidate_weight *= candidate_ratio
            incumbent_weight *= incumbent_ratio
            if (
                not math.isfinite(candidate_weight)
                or not math.isfinite(incumbent_weight)
                or candidate_weight > maximum_cumulative_ratio + 1e-12
                or incumbent_weight > maximum_cumulative_ratio + 1e-12
            ):
                raise ValueError("held-out cumulative importance ratio exceeds its bound")
            maximum_observed_cumulative_ratio = max(
                maximum_observed_cumulative_ratio,
                candidate_weight,
                incumbent_weight,
            )
            while len(candidate_weights_by_depth) <= depth:
                candidate_weights_by_depth.append([])
                incumbent_weights_by_depth.append([])
                physical_hits_by_depth.append(0)
                physical_costs_by_depth.append([])
            candidate_weights_by_depth[depth].append(candidate_weight)
            incumbent_weights_by_depth[depth].append(incumbent_weight)
            physical_hits_by_depth[depth] += option.physical_hit_cost
            physical_costs_by_depth[depth].append(option.physical_hit_cost)
            candidate_value += candidate_weight * option.physical_hit_cost
            incumbent_value += incumbent_weight * option.physical_hit_cost
            observed_value += option.physical_hit_cost
            nonincumbent_mass += 1.0 - target[option.state.baseline_action]
            decisions += 1
        candidate_values.append(candidate_value)
        incumbent_values.append(incumbent_value)
        observed_values.append(observed_value)
        candidate_final_weights.append(candidate_weight)
        incumbent_final_weights.append(incumbent_weight)
    differences = [
        candidate_value - incumbent_value
        for candidate_value, incumbent_value in zip(
            candidate_values, incumbent_values, strict=True
        )
    ]
    candidate_interval = _bootstrap_interval(
        candidate_values,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    incumbent_interval = _bootstrap_interval(
        incumbent_values,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed + 1,
    )
    difference_interval = _bootstrap_interval(
        differences,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed + 2,
    )
    weight_differences_by_depth = [
        [
            candidate - incumbent
            for candidate, incumbent in zip(
                candidate_rows, incumbent_rows, strict=True
            )
        ]
        for candidate_rows, incumbent_rows in zip(
            candidate_weights_by_depth,
            incumbent_weights_by_depth,
            strict=True,
        )
    ]
    null_p_value, null_interval = _depth_stratified_null(
        weight_differences_by_depth,
        physical_costs_by_depth,
        episodes=len(episodes),
        observed_difference=_mean(differences),
        resamples=permutation_resamples,
        seed=seed + 3,
    )
    hit_depths = tuple(
        depth for depth, hits in enumerate(physical_hits_by_depth) if hits > 0
    )
    candidate_hit_prefix_ess = tuple(
        _ess(candidate_weights_by_depth[depth]) for depth in hit_depths
    )
    incumbent_hit_prefix_ess = tuple(
        _ess(incumbent_weights_by_depth[depth]) for depth in hit_depths
    )
    candidate_minimum_ess = min(candidate_hit_prefix_ess, default=0.0)
    incumbent_minimum_ess = min(incumbent_hit_prefix_ess, default=0.0)
    checks = {
        "episode_count": len(episodes) >= minimum_episodes,
        "candidate_hit_prefix_ess": (
            candidate_minimum_ess >= minimum_effective_sample_size
        ),
        "incumbent_hit_prefix_ess": (
            incumbent_minimum_ess >= minimum_effective_sample_size
        ),
        "learned_deviation_present": nonincumbent_mass > 0.0,
        "paired_improvement": difference_interval[1] < 0.0,
        "action_outcome_association": null_p_value <= maximum_null_p_value,
    }
    return {
        "schema": OPE_SCHEMA,
        "passed": all(checks.values()),
        "authorization": "offline-evidence-only",
        "episodes": len(episodes),
        "decisions": decisions,
        "observed_mean_physical_hits": _mean(observed_values),
        "candidate_pdis_mean": _mean(candidate_values),
        "candidate_pdis_interval": list(candidate_interval),
        "incumbent_pdis_mean": _mean(incumbent_values),
        "incumbent_pdis_interval": list(incumbent_interval),
        "paired_difference_mean": _mean(differences),
        "paired_difference_interval": list(difference_interval),
        "depth_stratified_null_interval": list(null_interval),
        "depth_stratified_null_p_value": null_p_value,
        "candidate_final_weight_ess": _ess(candidate_final_weights),
        "incumbent_final_weight_ess": _ess(incumbent_final_weights),
        "hit_bearing_option_depths": len(hit_depths),
        "candidate_minimum_hit_prefix_ess": candidate_minimum_ess,
        "incumbent_minimum_hit_prefix_ess": incumbent_minimum_ess,
        "maximum_step_ratio": maximum_observed_step_ratio,
        "maximum_cumulative_ratio": maximum_observed_cumulative_ratio,
        "mean_nonincumbent_probability": nonincumbent_mass / decisions,
        "confidence": confidence,
        "bootstrap_resamples": bootstrap_resamples,
        "permutation_resamples": permutation_resamples,
        "maximum_null_p_value": maximum_null_p_value,
        "checks": checks,
    }
