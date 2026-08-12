"""Generation-5 in-sample implicit fitted-Q learning primitives."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
import random
from typing import Iterable

from .advantage_learning import OptionStep


STATE_SCHEMA = "autonomous-supported-implicit-q-policy-v1"
FIT_REPORT_SCHEMA = "autonomous-supported-implicit-q-fit-v1"
POPULATION_MEMBERS = 7
CROSSFIT_FOLDS = 5
BELLMAN_ITERATIONS = 8
N_STEP_OPTIONS = 8
COST_EXPECTILE = 0.10
Q_TREES = 128
VALUE_TREES = 96


@dataclass(frozen=True)
class ImplicitQMember:
    outcome_model: object
    q_model: object
    value_model: object
    bootstrap: dict[str, int]
    iterations: tuple[dict[str, float], ...]


def _episodes(samples: Iterable[OptionStep]) -> dict[str, list[OptionStep]]:
    result: dict[str, list[OptionStep]] = {}
    for sample in samples:
        result.setdefault(sample.episode_id, []).append(sample)
    for episode, rows in result.items():
        rows.sort(key=lambda row: row.sequence)
        if len({row.option_id for row in rows}) != len(rows) or any(
            right.sequence <= left.sequence for left, right in zip(rows, rows[1:])
        ):
            raise ValueError(f"invalid factual option order in {episode}")
    if not result:
        raise ValueError("implicit Q needs factual episodes")
    return dict(sorted(result.items()))


def _offline_state_vector(
    sample: OptionStep,
    *,
    index: int,
    count: int,
) -> tuple[float, ...]:
    """State-only value input; the final four controls never deploy online."""
    baseline = sample.legal_actions.index(sample.baseline_action)
    denominator = max(1, count - 1)
    remaining = count - index - 1
    return (
        *sample.candidate_vectors[baseline],
        index / denominator,
        remaining / denominator,
        math.log1p(index),
        math.log1p(remaining),
    )


def _arrays(episodes: dict[str, list[OptionStep]]):
    import numpy as np

    q_rows = []
    state_rows = []
    episode_ids = []
    base_weights = []
    mean_episode_options = sum(map(len, episodes.values())) / len(episodes)
    for episode, rows in episodes.items():
        weight = mean_episode_options / len(rows)
        for index, sample in enumerate(rows):
            if sample.action not in sample.legal_actions:
                raise ValueError("factual action is absent from its safe set")
            if sample.baseline_action not in sample.legal_actions:
                raise ValueError("incumbent is absent from its safe set")
            if not math.isfinite(sample.option_hit_cost) or sample.option_hit_cost < 0:
                raise ValueError("implicit Q requires non-negative physical HIT cost")
            q_rows.append(sample.vector)
            state_rows.append(_offline_state_vector(
                sample, index=index, count=len(rows)
            ))
            episode_ids.append(episode)
            base_weights.append(weight)
    return (
        np.asarray(q_rows, dtype=np.float32),
        np.asarray(state_rows, dtype=np.float32),
        tuple(episode_ids),
        np.asarray(base_weights, dtype=np.float64),
    )


def _q_regressor(*, trees: int, seed: int, threads: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=trees,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=4.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=12.0,
        reg_alpha=0.05,
        base_score=0.0,
        tree_method="hist",
        n_jobs=threads,
        random_state=seed,
    )


def _centered_layout(episodes: dict[str, list[OptionStep]]):
    import numpy as np

    rows = []
    coefficients = []
    starts = []
    episode_ids = []
    samples = []
    for episode, episode_rows in episodes.items():
        for sample in episode_rows:
            probabilities = tuple(map(float, sample.behavior_probabilities))
            if (
                len(probabilities) != len(sample.legal_actions)
                or any(
                    not math.isfinite(value) or value <= 0.0
                    for value in probabilities
                )
                or not math.isclose(
                    sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9
                )
            ):
                raise ValueError(
                    "implicit Q needs the complete behavior distribution"
                )
            factual = sample.legal_actions.index(sample.action)
            if not math.isclose(
                probabilities[factual],
                sample.behavior_probability,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError("factual and vector propensity disagree")
            starts.append(len(rows))
            for index, vector in enumerate(sample.candidate_vectors):
                rows.append(vector)
                coefficients.append(
                    float(index == factual) - probabilities[index]
                )
            episode_ids.append(episode)
            samples.append(sample)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(coefficients, dtype=np.float64),
        np.asarray(starts, dtype=np.int64),
        tuple(episode_ids),
        tuple(samples),
    )


def _effect_regressor(
    *,
    rows,
    coefficients,
    starts,
    targets,
    group_weights,
    trees: int,
    seed: int,
    threads: int,
):
    import numpy as np
    from xgboost import XGBRegressor

    group_indices = np.repeat(
        np.arange(len(starts)), np.diff(np.append(starts, len(rows)))
    )
    weights = np.asarray(group_weights, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)

    def objective(_labels, predictions):
        centered = np.add.reduceat(coefficients * predictions, starts)
        error = centered - target_values
        row_weight = weights[group_indices]
        gradient = 2.0 * coefficients * error[group_indices] * row_weight
        hessian = 2.0 * coefficients * coefficients * row_weight + 1e-8
        return gradient, hessian

    model = XGBRegressor(
        objective=objective,
        n_estimators=trees,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=4.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=12.0,
        reg_alpha=0.05,
        base_score=0.0,
        tree_method="hist",
        n_jobs=threads,
        random_state=seed,
    )
    labels = np.repeat(
        target_values, np.diff(np.append(starts, len(rows)))
    )
    model.fit(rows, labels)
    return model


def _value_regressor(
    *,
    trees: int,
    seed: int,
    threads: int,
    expectile: float,
    row_weights,
):
    import numpy as np
    from xgboost import XGBRegressor

    if not 0.0 < expectile < 0.5:
        raise ValueError("cost expectile must be below the mean")
    weights = np.asarray(row_weights, dtype=np.float64)

    def objective(labels, predictions):
        residual = labels - predictions
        asymmetric = np.where(residual < 0.0, 1.0 - expectile, expectile)
        gradient = 2.0 * asymmetric * (predictions - labels) * weights
        hessian = 2.0 * asymmetric * weights + 1e-8
        return gradient, hessian

    return XGBRegressor(
        objective=objective,
        n_estimators=trees,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=4.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=12.0,
        reg_alpha=0.05,
        base_score=0.0,
        tree_method="hist",
        n_jobs=threads,
        random_state=seed,
    )


def _n_step_targets(
    episodes: dict[str, list[OptionStep]],
    value_model,
    *,
    n_step_options: int,
) -> dict[tuple[str, str], float]:
    import numpy as np

    if n_step_options < 1:
        raise ValueError("n-step horizon must be positive")
    result = {}
    for episode, rows in episodes.items():
        values = None
        if value_model is not None:
            states = np.asarray([
                _offline_state_vector(sample, index=index, count=len(rows))
                for index, sample in enumerate(rows)
            ], dtype=np.float32)
            values = value_model.predict(states)
        for index, sample in enumerate(rows):
            stop = min(len(rows), index + n_step_options)
            target = sum(row.option_hit_cost for row in rows[index:stop])
            if stop < len(rows) and values is not None:
                target += float(values[stop])
            result[(episode, sample.option_id)] = float(target)
    return result


def _bootstrap_counts(
    groups: tuple[str, ...], *, seed: int
) -> dict[str, int]:
    generator = random.Random(seed)
    return dict(Counter(generator.choice(groups) for _ in groups))


def fit_implicit_q_member(
    samples: list[OptionStep],
    *,
    iterations: int = BELLMAN_ITERATIONS,
    n_step_options: int = N_STEP_OPTIONS,
    q_trees: int = Q_TREES,
    value_trees: int = VALUE_TREES,
    expectile: float = COST_EXPECTILE,
    seed: int,
    threads: int = 1,
    bootstrap: dict[str, int] | None = None,
) -> ImplicitQMember:
    import numpy as np

    if iterations < 1:
        raise ValueError("implicit Q needs at least one Bellman iteration")
    episodes = _episodes(samples)
    groups = tuple(episodes)
    counts = dict(bootstrap or {episode: 1 for episode in groups})
    if set(counts) - set(groups) or not any(counts.get(group, 0) for group in groups):
        raise ValueError("invalid whole-episode bootstrap")
    _q_rows, state_rows, row_episodes, base_weights = _arrays(episodes)
    effect_rows, coefficients, starts, effect_episodes, _ordered_samples = (
        _centered_layout(episodes)
    )
    if effect_episodes != row_episodes:
        raise RuntimeError("implicit Q factual and action-centered layouts differ")
    row_weights = base_weights * np.asarray(
        [counts.get(episode, 0) for episode in row_episodes], dtype=np.float64
    )
    value_model = None
    reports = []
    outcome_model = None
    q_model = None
    for iteration in range(iterations):
        targets = _n_step_targets(
            episodes, value_model, n_step_options=n_step_options
        )
        labels = np.asarray([
            targets[(episode, sample.option_id)]
            for episode, rows in episodes.items() for sample in rows
        ], dtype=np.float32)
        # The state outcome absorbs common risk. Only the bounded,
        # propensity-centered randomized action effect deploys online.
        outcome_model = _q_regressor(
            trees=q_trees, seed=seed + iteration * 2, threads=threads
        )
        outcome_model.fit(state_rows, labels, sample_weight=row_weights)
        outcome_predictions = outcome_model.predict(state_rows)
        residuals = labels - outcome_predictions
        q_model = _effect_regressor(
            rows=effect_rows,
            coefficients=coefficients,
            starts=starts,
            targets=residuals,
            group_weights=row_weights,
            trees=q_trees,
            seed=seed + iteration * 2 + 1,
            threads=threads,
        )
        effect_predictions = q_model.predict(effect_rows)
        centered_predictions = np.add.reduceat(
            coefficients * effect_predictions, starts
        )
        q_predictions = outcome_predictions + centered_predictions
        value_model = _value_regressor(
            trees=value_trees,
            seed=seed + iteration * 2 + 2,
            threads=threads,
            expectile=expectile,
            row_weights=row_weights,
        )
        value_model.fit(state_rows, q_predictions)
        value_predictions = value_model.predict(state_rows)
        reports.append({
            "iteration": float(iteration + 1),
            "target_mean": float(np.average(labels, weights=row_weights)),
            "q_weighted_mse": float(np.average(
                (q_predictions - labels) ** 2, weights=row_weights
            )),
            "zero_effect_weighted_mse": float(np.average(
                residuals ** 2, weights=row_weights
            )),
            "maximum_centered_coefficient": float(
                np.abs(coefficients).max()
            ),
            "value_mean": float(np.average(
                value_predictions, weights=row_weights
            )),
        })
    assert (
        outcome_model is not None
        and q_model is not None
        and value_model is not None
    )
    return ImplicitQMember(
        outcome_model=outcome_model,
        q_model=q_model,
        value_model=value_model,
        bootstrap=dict(sorted((group, counts.get(group, 0)) for group in groups)),
        iterations=tuple(reports),
    )


def fit_implicit_q_population(
    samples: list[OptionStep],
    *,
    members: int = POPULATION_MEMBERS,
    iterations: int = BELLMAN_ITERATIONS,
    n_step_options: int = N_STEP_OPTIONS,
    q_trees: int = Q_TREES,
    value_trees: int = VALUE_TREES,
    expectile: float = COST_EXPECTILE,
    seed: int = 260_813,
    total_threads: int = 12,
) -> list[ImplicitQMember]:
    episodes = _episodes(samples)
    groups = tuple(episodes)
    if members < 1 or total_threads < 1:
        raise ValueError("population and CPU budget must be positive")
    bootstraps = [
        _bootstrap_counts(groups, seed=seed + member * 10_000)
        for member in range(members)
    ]
    workers = min(members, total_threads)
    member_threads = max(1, total_threads // workers)

    def fit(member: int) -> ImplicitQMember:
        return fit_implicit_q_member(
            samples,
            iterations=iterations,
            n_step_options=n_step_options,
            q_trees=q_trees,
            value_trees=value_trees,
            expectile=expectile,
            seed=seed + member * 10_000,
            threads=member_threads,
            bootstrap=bootstraps[member],
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fit, range(members)))


def pessimistic_action(
    members: list[ImplicitQMember], sample: OptionStep
) -> tuple[str, dict[str, float]]:
    """Rank a native-safe set with the predeclared population-range margin."""
    import numpy as np

    if not members:
        raise ValueError("implicit Q population is empty")
    matrix = np.asarray(sample.candidate_vectors, dtype=np.float32)
    predictions = np.asarray([
        member.q_model.predict(matrix) for member in members
    ], dtype=np.float64)
    baseline = sample.legal_actions.index(sample.baseline_action)
    advantages = predictions - predictions[:, [baseline]]
    bounds = {}
    for index, action in enumerate(sample.legal_actions):
        values = advantages[:, index]
        bounds[action] = float(values.max() + values.max() - values.min())
    candidates = [
        (bound, action) for action, bound in bounds.items()
        if action != sample.baseline_action and bound < 0.0
    ]
    return min(candidates, default=(0.0, sample.baseline_action))[1], bounds


def delayed_effect_episodes(
    *,
    count: int = 64,
    options: int = 72,
    delay: int = 12,
    null_effect: bool = False,
    seed: int = 260_813,
) -> list[OptionStep]:
    """Deterministic complete episodes for delayed-credit and null smokes."""
    if delay < 2 or options <= delay * 2:
        raise ValueError("fixture needs a delayed interior")
    generator = random.Random(seed)
    result = []
    for episode_index in range(count):
        pending = [float((episode_index + index) % 5 == 0) for index in range(delay)]
        rows = []
        for option_index in range(options):
            state = tuple(pending)
            stay = (*state, 0.0)
            left = (*state, 1.0)
            choose_left = generator.random() < 0.5
            action = "left" if choose_left else "stay"
            cost = pending.pop(0)
            if null_effect:
                pending.append(float((episode_index + option_index + delay) % 5 == 0))
            else:
                pending.append(0.0 if choose_left else 1.0)
            rows.append(OptionStep(
                episode_id=f"fixture-{episode_index:03d}",
                option_id=f"fixture-{episode_index:03d}:{option_index:03d}",
                sequence=option_index,
                frame=option_index * 8,
                action=action,
                baseline_action="stay",
                behavior_probability=0.5,
                behavior_probabilities=(0.5, 0.5),
                vector=left if choose_left else stay,
                legal_actions=("stay", "left"),
                candidate_vectors=(stay, left),
                option_hit_cost=cost,
                duration_frames=8,
                termination_reason=(
                    "complete-stage-tail" if option_index == options - 1 else "horizon"
                ),
            ))
        remaining = 0.0
        labeled = []
        for sample in reversed(rows):
            remaining += sample.option_hit_cost
            labeled.append(replace(sample, return_to_go=remaining))
        result.extend(reversed(labeled))
    return result
