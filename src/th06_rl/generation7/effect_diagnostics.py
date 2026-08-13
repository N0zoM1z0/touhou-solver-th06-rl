"""Known-propensity proximal action-effect diagnostics grouped by episode."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean, stdev

from ..actions import ACTION_NAMES
from .factual_options import FactualEpisode


@dataclass(frozen=True)
class EffectRow:
    action: str
    baseline_action: str
    legal_actions: tuple[str, ...]
    behavior_probabilities: tuple[float, ...]
    hit_cost: int
    duration_frames: int = 1
    complied: bool = True

    @property
    def treated(self) -> bool:
        return self.action != self.baseline_action

    @property
    def treatment_probability(self) -> float:
        baseline_probability = self.behavior_probabilities[
            self.legal_actions.index(self.baseline_action)
        ]
        return 1.0 - baseline_probability


@dataclass(frozen=True)
class EffectEpisode:
    episode_id: str
    source_id: str
    stage: int
    rows: tuple[EffectRow, ...]


def effect_episode(episode: FactualEpisode) -> EffectEpisode:
    return EffectEpisode(
        episode_id=episode.episode_id,
        source_id=episode.source_id,
        stage=episode.stage,
        rows=tuple(
            EffectRow(
                action=option.proposal_action,
                baseline_action=option.baseline_action,
                legal_actions=option.legal_actions,
                behavior_probabilities=option.behavior_probabilities,
                hit_cost=option.hit_cost,
                duration_frames=option.duration_frames,
                complied=option.complied,
            )
            for option in episode.options
        ),
    )


def _targets(rows: tuple[EffectRow, ...], horizon: int) -> tuple[float, ...]:
    if horizon <= 0:
        raise ValueError("proximal horizon must be positive")
    costs = tuple(row.hit_cost for row in rows)
    prefix = [0]
    for cost in costs:
        prefix.append(prefix[-1] + cost)
    return tuple(
        float(prefix[min(len(rows), index + horizon)] - prefix[index])
        for index in range(len(rows))
    )


def _binary_episode_score(
    episode: EffectEpisode,
    horizon: int,
) -> tuple[float, int]:
    score = 0.0
    eligible = 0
    targets = _targets(episode.rows, horizon)
    for row, target in zip(episode.rows, targets, strict=True):
        p1 = row.treatment_probability
        p0 = 1.0 - p1
        if not 0.0 < p1 < 1.0:
            continue
        eligible += 1
        score += target / p1 if row.treated else -target / p0
    return score, eligible


def binary_ipw_effect(
    episodes: tuple[EffectEpisode, ...],
    *,
    horizon: int,
) -> dict[str, float | int]:
    if not episodes:
        raise ValueError("effect estimate needs episode groups")
    contributions = tuple(_binary_episode_score(episode, horizon) for episode in episodes)
    contributions = tuple(
        (score, count) for score, count in contributions if count > 0
    )
    if not contributions:
        raise ValueError("effect estimate has no randomized opportunities")
    total_rows = sum(count for _score, count in contributions)
    estimate = sum(score for score, _count in contributions) / total_rows
    episode_means = tuple(score / count for score, count in contributions)
    standard_error = (
        stdev(episode_means) / math.sqrt(len(episode_means))
        if len(episode_means) > 1
        else math.inf
    )
    return {
        "horizon": horizon,
        "episodes": len(contributions),
        "rows": total_rows,
        "effect": estimate,
        "episode_equal_effect": mean(episode_means),
        "episode_cluster_standard_error": standard_error,
    }


def binary_ipw_exposure_diagnostics(
    episodes: tuple[EffectEpisode, ...],
) -> dict[str, object]:
    """Expose decision-interval and compliance effects of assignment.

    A hit effect per proposal is not automatically a gameplay effect when the
    proposal changes native compliance and therefore the time until the next
    proposal.  These Horvitz-Thompson diagnostics make that mismatch visible;
    they are not a substitute for a fixed-physical-time or full-Stage value.
    """
    if not episodes:
        raise ValueError("exposure diagnostic needs episode groups")
    episode_rows = []
    for episode in episodes:
        contexts = 0
        potential = {
            "hit_nonbaseline": 0.0,
            "hit_baseline": 0.0,
            "duration_nonbaseline": 0.0,
            "duration_baseline": 0.0,
            "compliance_nonbaseline": 0.0,
            "compliance_baseline": 0.0,
        }
        for row in episode.rows:
            p1 = row.treatment_probability
            p0 = 1.0 - p1
            if not 0.0 < p1 < 1.0:
                continue
            contexts += 1
            arm = "nonbaseline" if row.treated else "baseline"
            probability = p1 if row.treated else p0
            potential[f"hit_{arm}"] += row.hit_cost / probability
            potential[f"duration_{arm}"] += row.duration_frames / probability
            potential[f"compliance_{arm}"] += float(row.complied) / probability
        if contexts:
            episode_rows.append({
                name: value / contexts for name, value in potential.items()
            })
    if not episode_rows:
        raise ValueError("exposure diagnostic has no randomized opportunities")

    def comparison(name: str) -> dict[str, float | int]:
        nonbaseline = tuple(row[f"{name}_nonbaseline"] for row in episode_rows)
        baseline = tuple(row[f"{name}_baseline"] for row in episode_rows)
        differences = tuple(
            left - right
            for left, right in zip(nonbaseline, baseline, strict=True)
        )
        return {
            "episodes": len(episode_rows),
            "nonbaseline_potential_mean": mean(nonbaseline),
            "baseline_potential_mean": mean(baseline),
            "difference": mean(differences),
            "episode_cluster_standard_error": (
                stdev(differences) / math.sqrt(len(differences))
                if len(differences) > 1
                else math.inf
            ),
        }

    rate_differences = tuple(
        row["hit_nonbaseline"] / row["duration_nonbaseline"]
        - row["hit_baseline"] / row["duration_baseline"]
        for row in episode_rows
    )
    return {
        "schema": "generation7-proposal-exposure-diagnostic-v1",
        "warning": "per-proposal HIT is not a fixed-physical-time value",
        "duration_frames": comparison("duration"),
        "compliance_probability": comparison("compliance"),
        "hit_probability": comparison("hit"),
        "hit_rate_per_frame_difference": {
            "episodes": len(rate_differences),
            "episode_equal_mean": mean(rate_differences),
            "episode_cluster_standard_error": (
                stdev(rate_differences) / math.sqrt(len(rate_differences))
                if len(rate_differences) > 1
                else math.inf
            ),
        },
    }


def action_specific_ipw_effect(
    episodes: tuple[EffectEpisode, ...],
    *,
    action: str,
    horizon: int,
) -> dict[str, float | int]:
    if action not in ACTION_NAMES:
        raise ValueError("unknown action")
    episode_values = []
    total_score = 0.0
    total_contexts = 0
    factual_treatments = 0
    factual_controls = 0
    for episode in episodes:
        targets = _targets(episode.rows, horizon)
        score = 0.0
        contexts = 0
        for row, target in zip(episode.rows, targets, strict=True):
            if action == row.baseline_action or action not in row.legal_actions:
                continue
            contexts += 1
            if row.action == action:
                probability = row.behavior_probabilities[
                    row.legal_actions.index(action)
                ]
                score += target / probability
                factual_treatments += 1
            elif row.action == row.baseline_action:
                probability = row.behavior_probabilities[
                    row.legal_actions.index(row.baseline_action)
                ]
                score -= target / probability
                factual_controls += 1
        if contexts:
            episode_values.append(score / contexts)
            total_score += score
            total_contexts += contexts
    return {
        "action": action,
        "horizon": horizon,
        "episodes": len(episode_values),
        "contexts": total_contexts,
        "factual_treatments": factual_treatments,
        "factual_controls": factual_controls,
        "effect": total_score / total_contexts if total_contexts else math.nan,
        "episode_equal_effect": mean(episode_values) if episode_values else math.nan,
        "episode_cluster_standard_error": (
            stdev(episode_values) / math.sqrt(len(episode_values))
            if len(episode_values) > 1
            else math.inf
        ),
    }


def bootstrap_sign_stability(
    episodes: tuple[EffectEpisode, ...],
    *,
    horizon: int,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    if replicates <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    generator = random.Random(seed)
    effects = []
    for _replicate in range(replicates):
        sample = tuple(generator.choice(episodes) for _ in episodes)
        effects.append(float(binary_ipw_effect(sample, horizon=horizon)["effect"]))
    observed = float(binary_ipw_effect(episodes, horizon=horizon)["effect"])
    same_sign = sum(
        value == 0.0 or math.copysign(1.0, value) == math.copysign(1.0, observed)
        for value in effects
    ) / replicates
    ordered = sorted(effects)
    return {
        "horizon": horizon,
        "replicates": replicates,
        "observed_effect": observed,
        "same_sign_fraction": same_sign,
        "lower_95": ordered[int(0.025 * (replicates - 1))],
        "upper_95": ordered[int(0.975 * (replicates - 1))],
    }


def permutation_null(
    episodes: tuple[EffectEpisode, ...],
    *,
    horizon: int,
    replicates: int,
    seed: int,
    kind: str,
) -> dict[str, object]:
    if kind not in {"action", "reward-suffix"} or replicates <= 1:
        raise ValueError("permutation null contract is invalid")
    import numpy as np

    generator = np.random.default_rng(seed)
    arrays = []
    total_rows = 0
    for episode in episodes:
        targets = _targets(episode.rows, horizon)
        eligible_targets = []
        multipliers = []
        treatment_probabilities = []
        for row, target in zip(episode.rows, targets, strict=True):
            p1 = row.treatment_probability
            p0 = 1.0 - p1
            if not 0.0 < p1 < 1.0:
                continue
            eligible_targets.append(target)
            multipliers.append(1.0 / p1 if row.treated else -1.0 / p0)
            treatment_probabilities.append(p1)
        if eligible_targets:
            arrays.append((
                np.asarray(eligible_targets, dtype=np.float64),
                np.asarray(multipliers, dtype=np.float64),
                np.asarray(treatment_probabilities, dtype=np.float64),
            ))
            total_rows += len(eligible_targets)
    effects = []
    for _replicate in range(replicates):
        score = 0.0
        for targets, multipliers, treatment_probabilities in arrays:
            if kind == "action":
                treated = generator.random(len(targets)) < treatment_probabilities
                sampled = np.where(
                    treated,
                    1.0 / treatment_probabilities,
                    -1.0 / (1.0 - treatment_probabilities),
                )
                score += float(targets @ sampled)
            else:
                offset = (
                    int(generator.integers(1, len(targets)))
                    if len(targets) > 1
                    else 0
                )
                score += float(multipliers @ np.roll(targets, -offset))
        effects.append(score / total_rows)
    center = mean(effects)
    spread = stdev(effects)
    return {
        "kind": kind,
        "horizon": horizon,
        "replicates": replicates,
        "null_mean": center,
        "null_standard_deviation": spread,
        "null_z": center / spread if spread > 0.0 else math.inf,
        "passes": spread > 0.0 and abs(center / spread) <= 3.0,
    }


def synthetic_delayed_effect(
    *,
    episodes: int,
    rows_per_episode: int,
    delay: int,
    seed: int,
) -> dict[str, object]:
    if min(episodes, rows_per_episode, delay) <= 0 or delay >= rows_per_episode:
        raise ValueError("synthetic delayed-effect contract is invalid")
    generator = random.Random(seed)
    groups = []
    for group in range(episodes):
        treatments = [generator.random() < 0.35 for _ in range(rows_per_episode)]
        costs = [0] * rows_per_episode
        for index, treated in enumerate(treatments[:-delay]):
            costs[index + delay] += int(treated)
        rows = tuple(
            EffectRow(
                action="up" if treated else "stay",
                baseline_action="stay",
                legal_actions=("stay", "up"),
                behavior_probabilities=(0.65, 0.35),
                hit_cost=cost,
            )
            for treated, cost in zip(treatments, costs, strict=True)
        )
        groups.append(EffectEpisode(f"synthetic-{group}", "synthetic", 0, rows))
    estimate = binary_ipw_effect(tuple(groups), horizon=delay + 1)
    effect = float(estimate["effect"])
    return {
        "schema": "generation7-synthetic-delayed-effect-v1",
        "delay": delay,
        "estimated_effect": effect,
        "expected_sign": "positive-cost",
        "passes": effect > 0.5,
    }
