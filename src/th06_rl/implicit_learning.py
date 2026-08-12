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
CALIBRATION_Q_TREES = 48
CALIBRATION_VALUE_TREES = 32
CALIBRATION_MEMBERS = 4
MINIMUM_CALIBRATION_EPISODES = 20
MAXIMUM_PROPOSAL_RATE = 0.10
MINIMUM_CONDITIONAL_AGREEMENT = 0.80


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


def _support_artifacts(samples: list[OptionStep], *, seed: int):
    # Reuse the source- and native-audited prototype implementation while
    # keeping Generation-4 outcomes and authorization completely separate.
    from .sequential_learning import OrthogonalOption, _support

    placeholders = [
        OrthogonalOption(
            step=sample,
            n_step_target=0.0,
            outcome_residual=0.0,
            fold=0,
        )
        for sample in samples
    ]
    return _support(placeholders, seed=seed)


def _supported(sample: OptionStep, support: dict[str, object]) -> list[bool]:
    from .sequential_learning import _support_mask

    return _support_mask(support, sample)


def _population_choice(
    predictions,
    sample: OptionStep,
    supported: list[bool],
    member_indices: tuple[int, ...],
) -> str:
    import numpy as np

    baseline = sample.legal_actions.index(sample.baseline_action)
    advantages = predictions[np.asarray(member_indices), :] - predictions[
        np.asarray(member_indices), [baseline]
    ][:, None]
    candidates = []
    for index, action in enumerate(sample.legal_actions):
        if action == sample.baseline_action or not supported[index]:
            continue
        values = advantages[:, index]
        bound = float(2.0 * values.max() - values.min())
        if bound < 0.0:
            candidates.append((bound, action))
    return min(candidates, default=(0.0, sample.baseline_action))[1]


def _evaluate_crossfit_fold(
    train: list[OptionStep],
    heldout: list[OptionStep],
    *,
    fold: int,
    iterations: int,
    n_step_options: int,
    q_trees: int,
    value_trees: int,
    seed: int,
    total_threads: int,
):
    import numpy as np

    population = fit_implicit_q_population(
        train,
        members=CALIBRATION_MEMBERS,
        iterations=iterations,
        n_step_options=n_step_options,
        q_trees=q_trees,
        value_trees=value_trees,
        seed=seed + fold * 100_000,
        total_threads=total_threads,
    )
    support, support_report = _support_artifacts(
        train, seed=seed + 50_000 + fold * 100_000
    )
    episodes = _episodes(heldout)
    _factual, states, row_episodes, _weights = _arrays(episodes)
    layout_rows, coefficients, starts, layout_episodes, samples = (
        _centered_layout(episodes)
    )
    if layout_episodes != row_episodes:
        raise RuntimeError("held-out implicit layouts differ")
    effect_predictions = np.asarray([
        member.q_model.predict(layout_rows) for member in population
    ], dtype=np.float64)
    centered = np.asarray([
        np.add.reduceat(coefficients * values, starts)
        for values in effect_predictions
    ])
    common = np.asarray([
        member.outcome_model.predict(states) for member in population
    ], dtype=np.float64)
    targets = []
    for member in population:
        by_option = _n_step_targets(
            episodes,
            member.value_model,
            n_step_options=n_step_options,
        )
        targets.append([
            by_option[(episode, sample.option_id)]
            for episode, rows in episodes.items() for sample in rows
        ])
    targets = np.asarray(targets, dtype=np.float64)
    zero_errors = np.mean((targets - common) ** 2, axis=0)
    q_errors = np.mean((targets - common - centered) ** 2, axis=0)
    stop_indices = [*starts[1:], len(layout_rows)]
    episode_reports: dict[str, dict[str, object]] = {}
    full_proposals = 0
    union_proposals = 0
    conditional_agreements = 0
    exact_agreements = 0
    unsupported_candidates = 0
    for index, (start, stop) in enumerate(zip(
        starts, stop_indices, strict=True
    )):
        sample = samples[index]
        episode = sample.episode_id
        report = episode_reports.setdefault(episode, {
            "zero_squared_error": 0.0,
            "q_squared_error": 0.0,
            "options": 0,
            "proposals": 0,
        })
        report["zero_squared_error"] += float(zero_errors[index])
        report["q_squared_error"] += float(q_errors[index])
        report["options"] += 1
        mask = _supported(sample, support)
        unsupported_candidates += sum(
            action != sample.baseline_action and not mask[action_index]
            for action_index, action in enumerate(sample.legal_actions)
        )
        predictions = effect_predictions[:, start:stop]
        left = _population_choice(predictions, sample, mask, (0, 1))
        right = _population_choice(predictions, sample, mask, (2, 3))
        full = _population_choice(predictions, sample, mask, (0, 1, 2, 3))
        either = left != sample.baseline_action or right != sample.baseline_action
        union_proposals += int(either)
        exact_agreements += int(left == right)
        conditional_agreements += int(either and left == right)
        if full != sample.baseline_action:
            full_proposals += 1
            report["proposals"] += 1
    for report in episode_reports.values():
        report["q_beats_zero"] = (
            report["q_squared_error"] < report["zero_squared_error"]
        )
    zero = sum(row["zero_squared_error"] for row in episode_reports.values())
    q_loss = sum(row["q_squared_error"] for row in episode_reports.values())
    return {
        "fold": fold,
        "fit_episodes": sorted({sample.episode_id for sample in train}),
        "heldout_episodes": sorted(episode_reports),
        "support": support_report,
        "bootstrap": [member.bootstrap for member in population],
        "episodes": episode_reports,
        "zero_squared_error": zero,
        "q_squared_error": q_loss,
        "options": len(samples),
        "proposals": full_proposals,
        "union_half_proposals": union_proposals,
        "conditional_half_agreements": conditional_agreements,
        "exact_half_agreements": exact_agreements,
        "unsupported_candidates": unsupported_candidates,
    }


def crossfit_implicit_q_report(
    samples: list[OptionStep],
    *,
    new_episode_ids: frozenset[str] = frozenset(),
    iterations: int = BELLMAN_ITERATIONS,
    n_step_options: int = N_STEP_OPTIONS,
    q_trees: int = CALIBRATION_Q_TREES,
    value_trees: int = CALIBRATION_VALUE_TREES,
    seed: int = 260_813,
    total_threads: int = 12,
) -> dict[str, object]:
    from .advantage_learning import _folds

    episodes = _episodes(samples)
    groups = list(episodes)
    folds = _folds(groups, count=CROSSFIT_FOLDS, seed=seed)
    reports = []
    for fold, heldout_groups in enumerate(folds):
        heldout_set = set(heldout_groups)
        train = [
            sample for sample in samples if sample.episode_id not in heldout_set
        ]
        heldout = [
            sample for sample in samples if sample.episode_id in heldout_set
        ]
        reports.append(_evaluate_crossfit_fold(
            train,
            heldout,
            fold=fold,
            iterations=iterations,
            n_step_options=n_step_options,
            q_trees=q_trees,
            value_trees=value_trees,
            seed=seed,
            total_threads=total_threads,
        ))
    episode_reports = {
        episode: report
        for fold in reports
        for episode, report in fold["episodes"].items()
    }

    def cohort(selected: set[str]) -> dict[str, object]:
        rows = [episode_reports[episode] for episode in sorted(selected)]
        zero = sum(row["zero_squared_error"] for row in rows)
        q_loss = sum(row["q_squared_error"] for row in rows)
        return {
            "episode_groups": len(rows),
            "zero_squared_error": zero,
            "q_squared_error": q_loss,
            "relative_q_loss": q_loss / zero if zero > 0.0 else math.inf,
            "episodes_beating_zero": sum(row["q_beats_zero"] for row in rows),
            "options": sum(row["options"] for row in rows),
            "proposals": sum(row["proposals"] for row in rows),
        }

    overall = cohort(set(episode_reports))
    new = cohort(set(new_episode_ids) & set(episode_reports))
    options = sum(report["options"] for report in reports)
    proposals = sum(report["proposals"] for report in reports)
    union = sum(report["union_half_proposals"] for report in reports)
    conditional = sum(
        report["conditional_half_agreements"] for report in reports
    )
    return {
        "folds": reports,
        "episodes": episode_reports,
        "overall": overall,
        "new_cohort": new,
        "options": options,
        "proposals": proposals,
        "proposal_rate": proposals / options,
        "union_half_proposals": union,
        "conditional_half_agreement": conditional / union if union else 0.0,
        "exact_half_agreement": sum(
            report["exact_half_agreements"] for report in reports
        ) / options,
        "unsupported_candidates": sum(
            report["unsupported_candidates"] for report in reports
        ),
    }


def fit_supported_implicit_q(
    samples: list[OptionStep],
    *,
    new_episode_ids: frozenset[str] = frozenset(),
    iterations: int = BELLMAN_ITERATIONS,
    n_step_options: int = N_STEP_OPTIONS,
    q_trees: int = Q_TREES,
    value_trees: int = VALUE_TREES,
    calibration_q_trees: int = CALIBRATION_Q_TREES,
    calibration_value_trees: int = CALIBRATION_VALUE_TREES,
    seed: int = 260_813,
    total_threads: int = 12,
    native_scorer_sha256: str,
    compatible_native_scorer_sha256: tuple[str, ...] = (),
) -> dict[str, object]:
    from .advantage_learning import (
        _augment_steps,
        encode_hazard_set,
        fit_hazard_codebook,
        rich_feature_names,
    )
    from .conservative_learning import _encoded_model, _export_model
    from .hazard_representation import HISTORY_FEATURE_NAMES
    from .sequential_learning import RICH_FEATURE_SCHEMA
    from .th06.learning_adapter import (
        ACTION_FEATURE_NAMES,
        OBSERVATION_FEATURE_NAMES,
    )

    groups = sorted({sample.episode_id for sample in samples})
    if len(groups) < CROSSFIT_FOLDS * 2:
        raise ValueError("Generation 5 fit needs at least ten episode groups")
    if not new_episode_ids <= set(groups):
        raise ValueError("new cohort is not a subset of fitted factual episodes")
    representation = fit_hazard_codebook(samples, seed=seed + 30_000)
    representation["conformance"] = []
    for sample in samples[: min(4, len(samples))]:
        representation["conformance"].append({
            "primitives": [list(row) for row in sample.hazard_primitives],
            "encoding": list(encode_hazard_set(
                sample.hazard_primitives, representation
            )),
        })
    augmented = _augment_steps(samples, representation)
    calibration = crossfit_implicit_q_report(
        augmented,
        new_episode_ids=new_episode_ids,
        iterations=iterations,
        n_step_options=n_step_options,
        q_trees=calibration_q_trees,
        value_trees=calibration_value_trees,
        seed=seed,
        total_threads=total_threads,
    )
    population = fit_implicit_q_population(
        augmented,
        members=POPULATION_MEMBERS,
        iterations=iterations,
        n_step_options=n_step_options,
        q_trees=q_trees,
        value_trees=value_trees,
        seed=seed + 700_000,
        total_threads=total_threads,
    )
    support, support_report = _support_artifacts(
        augmented, seed=seed + 800_000
    )
    conformance = [sample.vector for sample in augmented[: min(8, len(augmented))]]
    names = rich_feature_names()
    models = [
        _encoded_model(_export_model(
            member.q_model,
            conformance,
            feature_schema=RICH_FEATURE_SCHEMA,
            feature_names=names,
        ))
        for member in population
    ]
    compatible = tuple(dict.fromkeys((
        native_scorer_sha256,
        *compatible_native_scorer_sha256,
    )))
    native_bound = all(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in compatible
    )
    overall = calibration["overall"]
    new = calibration["new_cohort"]
    production_contract = (
        iterations == BELLMAN_ITERATIONS
        and n_step_options == N_STEP_OPTIONS
        and q_trees == Q_TREES
        and value_trees == VALUE_TREES
        and calibration_q_trees == CALIBRATION_Q_TREES
        and calibration_value_trees == CALIBRATION_VALUE_TREES
        and seed == 260_813
    )
    gates = {
        "production_contract": production_contract,
        "minimum_crossfit_episode_groups": (
            overall["episode_groups"] >= MINIMUM_CALIBRATION_EPISODES
        ),
        "overall_q_beats_zero": (
            overall["q_squared_error"] < overall["zero_squared_error"]
        ),
        "overall_strict_episode_majority": (
            overall["episodes_beating_zero"] > overall["episode_groups"] / 2
        ),
        "new_cohort_declared": new["episode_groups"] >= 8,
        "new_cohort_q_beats_zero": (
            new["q_squared_error"] < new["zero_squared_error"]
        ),
        "new_cohort_strict_episode_majority": (
            new["episodes_beating_zero"] > new["episode_groups"] / 2
        ),
        "heldout_policy_exercised": calibration["proposals"] > 0,
        "heldout_proposal_rate_bounded": (
            calibration["proposal_rate"] <= MAXIMUM_PROPOSAL_RATE
        ),
        "independent_half_policy_agreement": (
            calibration["conditional_half_agreement"]
            >= MINIMUM_CONDITIONAL_AGREEMENT
        ),
        "population_complete": len(models) == POPULATION_MEMBERS,
        "support_calibrated": support_report["coverage"] >= 0.99,
        "finite_diagnostics": all(math.isfinite(float(value)) for value in (
            overall["zero_squared_error"],
            overall["q_squared_error"],
            overall["relative_q_loss"],
            calibration["proposal_rate"],
            calibration["conditional_half_agreement"],
        )),
        "native_scorer_bound": native_bound,
    }
    return {
        "schema": STATE_SCHEMA,
        "mode": "shadow",
        "feature_schema": RICH_FEATURE_SCHEMA,
        "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "feature_names": list(names),
        "representation": {
            "kind": (
                "learned-permutation-invariant-hazard-codebook-plus-factual-history"
            ),
            "hazard_codebook": representation,
            "history_feature_names": list(HISTORY_FEATURE_NAMES),
        },
        "models": models,
        "support": support,
        "selection": {
            "rule": "population-range-upper-bound-relative-to-incumbent",
            "baseline_advantage": 0.0,
            "uncertainty_range_multiplier": 1.0,
            "active_override_budget": None,
        },
        "native_scorer": {
            "schema": "th06-rl-native-xgboost-scorer-v1",
            "sha256": native_scorer_sha256,
            "compatible_sha256": list(compatible),
        },
        "population": {
            "kind": "whole-episode-bootstrap-action-centered-implicit-q",
            "members": POPULATION_MEMBERS,
            "trees_per_member": q_trees,
            "bellman_iterations": iterations,
            "bootstrap": [member.bootstrap for member in population],
            "iterations": [list(member.iterations) for member in population],
        },
        "authorization": {
            "fit_gates": gates,
            "fit_eligible": all(gates.values()),
            "policy_calibration": calibration,
            "active_canary": None,
        },
        "fit_report": {
            "schema": FIT_REPORT_SCHEMA,
            "algorithm": "action-centered-in-sample-implicit-fitted-q",
            "reward": "physical-HIT-only",
            "gamma": 1.0,
            "cost_expectile": COST_EXPECTILE,
            "bellman_iterations": iterations,
            "n_step_options": n_step_options,
            "episode_groups": groups,
            "new_episode_groups": sorted(new_episode_ids),
            "options": len(augmented),
            "q_trees": q_trees,
            "value_trees": value_trees,
            "calibration_q_trees": calibration_q_trees,
            "calibration_value_trees": calibration_value_trees,
            "centered_objective_uses_inverse_propensity": False,
            "maximum_centered_coefficient": max(
                abs(float(index == sample.legal_actions.index(sample.action)) - probability)
                for sample in augmented
                for index, probability in enumerate(sample.behavior_probabilities)
            ),
            "policy_calibration": calibration,
            "support": support_report,
            "population_bootstrap": [member.bootstrap for member in population],
        },
    }


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
