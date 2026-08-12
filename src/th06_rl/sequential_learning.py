"""Generation-4 sequential semi-Markov action-centered offline RL."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
import random
from typing import Iterable

from .advantage_learning import (
    OptionStep,
    _augment_steps,
    _folds,
    fit_hazard_codebook,
    rich_feature_names,
)
from .conservative_learning import _encoded_model, _export_model, _kmeans
from .hazard_representation import (
    HAZARD_PRIMITIVE_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
)
from .learning_features import tree_feature_names
from .offline import ACTION_NAMES
from .th06.learning_adapter import ACTION_FEATURE_NAMES, OBSERVATION_FEATURE_NAMES


STATE_SCHEMA = "autonomous-sequential-r-option-critic-v1"
FIT_REPORT_SCHEMA = "autonomous-sequential-r-option-fit-v1"
SUPPORT_SCHEMA = "autonomous-rich-local-prototype-support-v1"
RICH_FEATURE_SCHEMA = "generation-4-rich-action-centered-v1"
POPULATION_MEMBERS = 7
CROSSFIT_FOLDS = 5
NUISANCE_MEMBERS = 3
NUISANCE_TREES = 160
CRITIC_TREES = 128
N_STEP_OPTIONS = 8
MINIMUM_CALIBRATION_EPISODES = 20
SUPPORT_QUANTILE = 0.99
TRANSITION_SCHEMA = "th06-rl-transition-v10"
BEHAVIOR_POLICY = "propensity-aware-option-exploration-v1"


@dataclass(frozen=True)
class OrthogonalOption:
    step: OptionStep
    n_step_target: float
    outcome_residual: float
    fold: int


@dataclass(frozen=True)
class _CenteredLayout:
    rows: object
    coefficients: object
    group_indices: object
    starts: object
    targets: object
    episode_ids: tuple[str, ...]
    samples: tuple[OrthogonalOption, ...]


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
    return dict(sorted(result.items()))


def _probabilities(sample: OptionStep) -> tuple[float, ...]:
    values = tuple(map(float, sample.behavior_probabilities))
    if (
        len(values) != len(sample.legal_actions)
        or any(not math.isfinite(value) or value <= 0.0 for value in values)
        or not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError("option has no complete factual behavior distribution")
    factual = sample.legal_actions.index(sample.action)
    if not math.isclose(
        values[factual],
        sample.behavior_probability,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("chosen and complete behavior propensities disagree")
    return values


def _nuisance_vector(
    sample: OptionStep,
    *,
    index: int,
    count: int,
) -> tuple[float, ...]:
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


def _nuisance_regressor(*, trees: int, seed: int, threads: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=trees,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=8.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=12.0,
        reg_alpha=0.05,
        tree_method="hist",
        n_jobs=threads,
        random_state=seed,
    )


def _episode_arrays(
    episodes: dict[str, list[OptionStep]],
    targets: dict[tuple[str, str], float],
):
    import numpy as np

    x = []
    y = []
    weights = []
    mean_episode_options = sum(map(len, episodes.values())) / len(episodes)
    for episode, rows in episodes.items():
        # Equalize complete-episode influence without collapsing the total
        # Hessian mass below the tree learner's fixed min_child_weight.
        weight = mean_episode_options / len(rows)
        for index, sample in enumerate(rows):
            x.append(_nuisance_vector(sample, index=index, count=len(rows)))
            y.append(targets[(episode, sample.option_id)])
            weights.append(weight)
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
    )


def _fit_nuisance_population(
    episodes: dict[str, list[OptionStep]],
    targets: dict[tuple[str, str], float],
    *,
    trees: int,
    seed: int,
    threads: int,
):
    x, y, base_weights = _episode_arrays(episodes, targets)
    groups = tuple(episodes)
    row_groups = [episode for episode, rows in episodes.items() for _ in rows]
    generator = random.Random(seed)
    models = []
    bootstrap = []
    for member in range(NUISANCE_MEMBERS):
        chosen = [generator.choice(groups) for _ in groups]
        counts = Counter(chosen)
        weights = base_weights.copy()
        for index, episode in enumerate(row_groups):
            weights[index] *= counts[episode]
        model = _nuisance_regressor(
            trees=trees,
            seed=seed + member,
            threads=threads,
        )
        model.fit(x, y, sample_weight=weights)
        models.append(model)
        bootstrap.append({
            "member": member,
            "unique_episodes": len(counts),
            "episode_counts": dict(sorted(counts.items())),
        })
    return models, bootstrap


def _predict_nuisance(models, rows) -> list[float]:
    import numpy as np

    matrix = np.asarray(rows, dtype=np.float32)
    predictions = np.asarray([model.predict(matrix) for model in models])
    return predictions.mean(axis=0).astype(float).tolist()


def _return_targets(
    episodes: dict[str, list[OptionStep]],
) -> dict[tuple[str, str], float]:
    return {
        (episode, sample.option_id): float(sample.return_to_go)
        for episode, rows in episodes.items()
        for sample in rows
    }


def _n_step_targets(
    episodes: dict[str, list[OptionStep]],
    value_models,
) -> dict[tuple[str, str], float]:
    result = {}
    for episode, rows in episodes.items():
        nuisance_rows = [
            _nuisance_vector(sample, index=index, count=len(rows))
            for index, sample in enumerate(rows)
        ]
        values = _predict_nuisance(value_models, nuisance_rows)
        for index, sample in enumerate(rows):
            stop = min(len(rows), index + N_STEP_OPTIONS)
            target = sum(row.option_hit_cost for row in rows[index:stop])
            if stop < len(rows):
                target += values[stop]
            result[(episode, sample.option_id)] = float(target)
    return result


def _residual_options(
    episodes: dict[str, list[OptionStep]],
    targets: dict[tuple[str, str], float],
    outcome_models,
    *,
    fold: int,
) -> list[OrthogonalOption]:
    result = []
    for episode, rows in episodes.items():
        vectors = [
            _nuisance_vector(sample, index=index, count=len(rows))
            for index, sample in enumerate(rows)
        ]
        expected = _predict_nuisance(outcome_models, vectors)
        for sample, mean in zip(rows, expected, strict=True):
            target = targets[(episode, sample.option_id)]
            result.append(OrthogonalOption(
                step=sample,
                n_step_target=target,
                outcome_residual=float(target - mean),
                fold=fold,
            ))
    return result


def crossfit_n_step_residuals(
    samples: list[OptionStep],
    *,
    nuisance_trees: int = NUISANCE_TREES,
    seed: int = 260812,
    threads: int = 12,
) -> tuple[list[OrthogonalOption], list[dict[str, object]]]:
    episodes = _episodes(samples)
    groups = list(episodes)
    if len(groups) < CROSSFIT_FOLDS * 2:
        raise ValueError("Generation 4 needs at least ten complete episodes")
    folds = _folds(groups, count=CROSSFIT_FOLDS, seed=seed)
    result = []
    reports = []
    for fold_index, heldout_groups in enumerate(folds):
        heldout_set = set(heldout_groups)
        fit = {
            episode: rows for episode, rows in episodes.items()
            if episode not in heldout_set
        }
        heldout = {
            episode: rows for episode, rows in episodes.items()
            if episode in heldout_set
        }
        value_models, value_bootstrap = _fit_nuisance_population(
            fit,
            _return_targets(fit),
            trees=nuisance_trees,
            seed=seed + fold_index * 10_000,
            threads=threads,
        )
        fit_targets = _n_step_targets(fit, value_models)
        heldout_targets = _n_step_targets(heldout, value_models)
        outcome_models, outcome_bootstrap = _fit_nuisance_population(
            fit,
            fit_targets,
            trees=nuisance_trees,
            seed=seed + fold_index * 10_000 + 1_000,
            threads=threads,
        )
        heldout_rows = _residual_options(
            heldout,
            heldout_targets,
            outcome_models,
            fold=fold_index,
        )
        result.extend(heldout_rows)
        reports.append({
            "fold": fold_index,
            "fit_episodes": sorted(fit),
            "heldout_episodes": sorted(heldout),
            "heldout_options": len(heldout_rows),
            "value_bootstrap": value_bootstrap,
            "outcome_bootstrap": outcome_bootstrap,
            "target_mean": sum(row.n_step_target for row in heldout_rows)
            / len(heldout_rows),
            "residual_mean": sum(row.outcome_residual for row in heldout_rows)
            / len(heldout_rows),
        })
    expected = {(sample.episode_id, sample.option_id) for sample in samples}
    observed = {(row.step.episode_id, row.step.option_id) for row in result}
    if observed != expected or len(result) != len(samples):
        raise RuntimeError("cross-fitted nuisance did not label every option once")
    result.sort(key=lambda row: (row.step.episode_id, row.step.sequence))
    return result, reports


def _centered_layout(samples: list[OrthogonalOption]) -> _CenteredLayout:
    import numpy as np

    rows = []
    coefficients = []
    group_indices = []
    starts = []
    targets = []
    episode_ids = []
    for group_index, sample in enumerate(samples):
        step = sample.step
        probabilities = _probabilities(step)
        factual = step.legal_actions.index(step.action)
        starts.append(len(rows))
        targets.append(sample.outcome_residual)
        episode_ids.append(step.episode_id)
        for candidate_index, vector in enumerate(step.candidate_vectors):
            rows.append(vector)
            coefficients.append(
                float(candidate_index == factual) - probabilities[candidate_index]
            )
            group_indices.append(group_index)
    return _CenteredLayout(
        rows=np.asarray(rows, dtype=np.float32),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        group_indices=np.asarray(group_indices, dtype=np.int64),
        starts=np.asarray(starts, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.float64),
        episode_ids=tuple(episode_ids),
        samples=tuple(samples),
    )


def _critic_regressor(
    layout: _CenteredLayout,
    *,
    episode_weights: dict[str, int],
    trees: int,
    seed: int,
    threads: int,
):
    import numpy as np
    from xgboost import XGBRegressor

    coefficients = layout.coefficients
    group_indices = layout.group_indices
    starts = layout.starts
    targets = layout.targets
    weights = np.asarray([
        episode_weights[episode] for episode in layout.episode_ids
    ], dtype=np.float64)

    def objective(_labels, predictions):
        centered = np.add.reduceat(coefficients * predictions, starts)
        error = centered - targets
        group_weight = weights[group_indices]
        gradient = 2.0 * coefficients * error[group_indices] * group_weight
        hessian = 2.0 * coefficients * coefficients * group_weight + 1e-8
        return gradient, hessian

    model = XGBRegressor(
        objective=objective,
        n_estimators=trees,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=8.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=12.0,
        reg_alpha=0.05,
        base_score=0.0,
        tree_method="hist",
        n_jobs=threads,
        random_state=seed,
    )
    labels = np.repeat(layout.targets, [
        len(sample.step.legal_actions) for sample in layout.samples
    ])
    model.fit(layout.rows, labels)
    return model


def _fit_critic_population(
    samples: list[OrthogonalOption],
    *,
    trees: int,
    seed: int,
    threads: int,
):
    layout = _centered_layout(samples)
    groups = sorted(set(layout.episode_ids))
    generator = random.Random(seed)
    models = []
    reports = []
    for member in range(POPULATION_MEMBERS):
        chosen = [generator.choice(groups) for _ in groups]
        counts = Counter(chosen)
        model = _critic_regressor(
            layout,
            episode_weights={episode: counts[episode] for episode in groups},
            trees=trees,
            seed=seed + member,
            threads=threads,
        )
        models.append(model)
        reports.append({
            "member": member,
            "unique_episodes": len(counts),
            "episode_counts": dict(sorted(counts.items())),
        })
    return models, reports


def _evaluate_population(models, samples: list[OrthogonalOption]):
    import numpy as np

    layout = _centered_layout(samples)
    predictions = np.asarray([model.predict(layout.rows) for model in models])
    result = []
    stop_indices = [*layout.starts[1:], len(layout.rows)]
    for group, (start, stop) in enumerate(zip(
        layout.starts, stop_indices, strict=True
    )):
        sample = samples[group]
        coefficients = layout.coefficients[start:stop]
        member_scores = predictions[:, start:stop]
        centered = member_scores @ coefficients
        baseline = sample.step.legal_actions.index(sample.step.baseline_action)
        advantages = member_scores - member_scores[:, [baseline]]
        upper = advantages.max(axis=0)
        candidates = [
            (float(upper[index]), index)
            for index, action in enumerate(sample.step.legal_actions)
            if action != sample.step.baseline_action and upper[index] < 0.0
        ]
        selected = min(candidates, default=None)
        result.append({
            "sample": sample,
            "centered_prediction": float(centered.mean()),
            "member_centered_predictions": centered.astype(float).tolist(),
            "selected_action": (
                sample.step.legal_actions[selected[1]]
                if selected is not None else sample.step.baseline_action
            ),
            "selected_upper_advantage": (
                selected[0] if selected is not None else 0.0
            ),
            "member_advantages": advantages.astype(float).tolist(),
        })
    return result


def _crossfit_critic_report(
    residuals: list[OrthogonalOption],
    *,
    critic_trees: int,
    seed: int,
    threads: int,
):
    by_episode: dict[str, dict[str, object]] = {}
    fold_reports = []
    for fold in range(CROSSFIT_FOLDS):
        train = [sample for sample in residuals if sample.fold != fold]
        heldout = [sample for sample in residuals if sample.fold == fold]
        models, bootstrap = _fit_critic_population(
            train,
            trees=critic_trees,
            seed=seed + 100_000 + fold * 10_000,
            threads=threads,
        )
        evaluated = _evaluate_population(models, heldout)
        for row in evaluated:
            sample = row["sample"]
            episode = sample.step.episode_id
            report = by_episode.setdefault(episode, {
                "zero_squared_error": 0.0,
                "critic_squared_error": 0.0,
                "options": 0,
                "proposals": 0,
                "proposal_actions": Counter(),
            })
            report["zero_squared_error"] += sample.outcome_residual ** 2
            report["critic_squared_error"] += (
                sample.outcome_residual - row["centered_prediction"]
            ) ** 2
            report["options"] += 1
            proposed = row["selected_action"]
            if proposed != sample.step.baseline_action:
                report["proposals"] += 1
                report["proposal_actions"][proposed] += 1
        fold_reports.append({
            "fold": fold,
            "fit_episodes": sorted({row.step.episode_id for row in train}),
            "heldout_episodes": sorted({row.step.episode_id for row in heldout}),
            "bootstrap": bootstrap,
        })
    episode_reports = {}
    for episode, raw in sorted(by_episode.items()):
        episode_reports[episode] = {
            **raw,
            "proposal_actions": dict(sorted(raw["proposal_actions"].items())),
            "critic_beats_zero": (
                raw["critic_squared_error"] < raw["zero_squared_error"]
            ),
        }
    zero = sum(row["zero_squared_error"] for row in episode_reports.values())
    critic = sum(row["critic_squared_error"] for row in episode_reports.values())
    proposal_episodes = sum(row["proposals"] > 0 for row in episode_reports.values())
    return {
        "folds": fold_reports,
        "episodes": episode_reports,
        "zero_r_loss": zero,
        "critic_r_loss": critic,
        "relative_r_loss": critic / zero if zero > 0.0 else math.inf,
        "episodes_beating_zero": sum(
            row["critic_beats_zero"] for row in episode_reports.values()
        ),
        "episode_groups": len(episode_reports),
        "proposals": sum(row["proposals"] for row in episode_reports.values()),
        "proposal_episodes": proposal_episodes,
    }


def _support(
    residuals: list[OrthogonalOption],
    *,
    seed: int,
):
    import numpy as np

    samples = [row.step for row in residuals]
    matrix = np.asarray([sample.vector for sample in samples], dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    prototypes = []
    supported = []
    counts = {}
    for action_index, action in enumerate(ACTION_NAMES):
        rows = np.asarray([
            sample.vector for sample in samples if sample.action == action
        ], dtype=np.float64)
        counts[action] = len(rows)
        if len(rows):
            supported.append(action)
            prototypes.append(_kmeans(
                (rows - mean) / scale,
                count=min(12, len(rows)),
                iterations=12,
                seed=seed + action_index,
            ))
        else:
            prototypes.append(np.zeros((1, matrix.shape[1]), dtype=np.float64))
    distances = []
    for sample in samples:
        action_index = ACTION_NAMES.index(sample.action)
        normalized = (np.asarray(sample.vector) - mean) / scale
        distances.append(float((
            (prototypes[action_index] - normalized) ** 2
        ).mean(axis=1).min()))
    threshold = float(np.quantile(distances, SUPPORT_QUANTILE))
    artifact = {
        "schema": SUPPORT_SCHEMA,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "prototypes": [group.tolist() for group in prototypes],
        "factual_supported_actions": supported,
        "threshold": threshold,
        "threshold_source": {
            "kind": "cross-fitted-factual-distance-quantile",
            "quantile": SUPPORT_QUANTILE,
            "rows": len(distances),
        },
    }
    report = {
        "threshold": threshold,
        "quantile": SUPPORT_QUANTILE,
        "rows": len(distances),
        "coverage": sum(value <= threshold for value in distances) / len(distances),
        "distance_mean": float(np.mean(distances)),
        "distance_p95": float(np.quantile(distances, 0.95)),
        "distance_p99": float(np.quantile(distances, 0.99)),
        "factual_action_counts": counts,
    }
    return artifact, report


def fit_sequential_r_critic(
    samples: list[OptionStep],
    *,
    nuisance_trees: int = NUISANCE_TREES,
    critic_trees: int = CRITIC_TREES,
    seed: int = 260812,
    threads: int = 12,
    native_scorer_sha256: str,
    compatible_native_scorer_sha256: tuple[str, ...] = (),
) -> dict[str, object]:
    groups = sorted({sample.episode_id for sample in samples})
    if len(groups) < CROSSFIT_FOLDS * 2:
        raise ValueError("Generation 4 fit needs at least ten episode groups")
    representation = fit_hazard_codebook(samples, seed=seed + 30_000)
    representation["conformance"] = []
    for sample in samples[: min(4, len(samples))]:
        from .advantage_learning import encode_hazard_set
        representation["conformance"].append({
            "primitives": [list(row) for row in sample.hazard_primitives],
            "encoding": list(encode_hazard_set(
                sample.hazard_primitives, representation
            )),
        })
    augmented = _augment_steps(samples, representation)
    residuals, nuisance_report = crossfit_n_step_residuals(
        augmented,
        nuisance_trees=nuisance_trees,
        seed=seed,
        threads=threads,
    )
    calibration = _crossfit_critic_report(
        residuals,
        critic_trees=critic_trees,
        seed=seed,
        threads=threads,
    )
    final_models, final_bootstrap = _fit_critic_population(
        residuals,
        trees=critic_trees,
        seed=seed + 200_000,
        threads=threads,
    )
    support, support_report = _support(residuals, seed=seed + 300_000)
    conformance = [row.step.vector for row in residuals[: min(8, len(residuals))]]
    names = rich_feature_names()
    models = [
        _encoded_model(_export_model(
            model,
            conformance,
            feature_schema=RICH_FEATURE_SCHEMA,
            feature_names=names,
        ))
        for model in final_models
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
    episode_groups = int(calibration["episode_groups"])
    gates = {
        "minimum_crossfit_episode_groups": (
            episode_groups >= MINIMUM_CALIBRATION_EPISODES
        ),
        "crossfit_complete": len(residuals) == len(samples),
        "critic_r_loss_beats_zero": (
            calibration["critic_r_loss"] < calibration["zero_r_loss"]
        ),
        "strict_episode_majority_beats_zero": (
            calibration["episodes_beating_zero"] > episode_groups / 2
        ),
        "population_complete": len(models) == POPULATION_MEMBERS,
        "finite_diagnostics": all(math.isfinite(float(value)) for value in (
            calibration["critic_r_loss"],
            calibration["zero_r_loss"],
            calibration["relative_r_loss"],
        )),
        "support_calibrated": support_report["coverage"] >= SUPPORT_QUANTILE,
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
            "kind": "learned-permutation-invariant-hazard-codebook-plus-factual-history",
            "hazard_codebook": representation,
            "history_feature_names": list(HISTORY_FEATURE_NAMES),
        },
        "models": models,
        "support": support,
        "selection": {
            "rule": "all-members-negative-relative-to-incumbent",
            "baseline_advantage": 0.0,
            "active_override_budget": None,
        },
        "native_scorer": {
            "schema": "th06-rl-native-xgboost-scorer-v1",
            "sha256": native_scorer_sha256,
            "compatible_sha256": list(compatible),
        },
        "population": {
            "kind": "whole-episode-bootstrap-action-centered-r-critic",
            "members": POPULATION_MEMBERS,
            "trees_per_member": critic_trees,
            "bootstrap": final_bootstrap,
        },
        "authorization": {
            "fit_gates": gates,
            "fit_eligible": all(gates.values()),
            "policy_calibration": calibration,
            "active_canary": None,
        },
        "fit_report": {
            "schema": FIT_REPORT_SCHEMA,
            "algorithm": "cross-fitted-n-step-generalized-r-option-critic",
            "reward": "physical-HIT-only",
            "gamma": 1.0,
            "n_step_options": N_STEP_OPTIONS,
            "episode_groups": groups,
            "options": len(samples),
            "nuisance_trees": nuisance_trees,
            "critic_trees": critic_trees,
            "nuisance_crossfit": nuisance_report,
            "policy_calibration": calibration,
            "support": support_report,
            "population_bootstrap": final_bootstrap,
            "maximum_inverse_propensity": max(
                1.0 / probability
                for sample in samples for probability in _probabilities(sample)
            ),
            "centered_objective_uses_inverse_propensity": False,
        },
    }


def _causal_episodes(count: int = 160, options: int = 64) -> list[OptionStep]:
    names = tree_feature_names(OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES)
    indices = {name: index for index, name in enumerate(names)}
    generator = random.Random(260812)
    result = []
    for episode_index in range(count):
        episode_rows = []
        for option_index in range(options):
            state_risk = float((episode_index + option_index) % 3) / 2.0
            baseline = [0.0] * len(names)
            baseline[indices["observation:position_x_unit"]] = state_risk
            baseline[indices["matches_baseline"]] = 1.0
            candidate = baseline.copy()
            candidate[indices["action:direction_x"]] = -1.0
            candidate[indices["delta_from_baseline:direction_x"]] = -1.0
            candidate[indices["matches_baseline"]] = 0.0
            assigned_candidate = generator.random() < 0.25
            action = "left" if assigned_candidate else "stay"
            immediate = 2.0 + state_risk - float(assigned_candidate)
            primitive = [0.0] * len(HAZARD_PRIMITIVE_FEATURE_NAMES)
            primitive[option_index % len(primitive)] = 1.0
            primitive[0] += state_risk
            history = [0.0] * len(HISTORY_FEATURE_NAMES)
            history[0] = 1.0
            history[2] = state_risk
            episode_rows.append(OptionStep(
                episode_id=f"causal-{episode_index:02d}",
                option_id=f"causal-{episode_index:02d}:{option_index:03d}",
                sequence=option_index,
                frame=option_index * 8,
                action=action,
                baseline_action="stay",
                behavior_probability=0.25 if assigned_candidate else 0.75,
                vector=tuple(candidate if assigned_candidate else baseline),
                legal_actions=("stay", "left"),
                candidate_vectors=(tuple(baseline), tuple(candidate)),
                option_hit_cost=immediate,
                duration_frames=8,
                termination_reason=(
                    "complete-stage-tail"
                    if option_index == options - 1 else "horizon"
                ),
                hazard_primitives=(tuple(primitive),),
                history_features=tuple(history),
                behavior_probabilities=(0.75, 0.25),
            ))
        remaining = 0.0
        labeled = []
        for sample in reversed(episode_rows):
            remaining += sample.option_hit_cost
            labeled.append(replace(sample, return_to_go=remaining))
        result.extend(reversed(labeled))
    return result


def fit_sequential_causal_fixture(
    *,
    threads: int = 4,
    native_scorer_sha256: str = "0" * 64,
    compatible_native_scorer_sha256: tuple[str, ...] = (),
) -> tuple[list[OptionStep], dict[str, object]]:
    samples = _causal_episodes()
    state = fit_sequential_r_critic(
        samples,
        seed=260812,
        threads=threads,
        native_scorer_sha256=native_scorer_sha256,
        compatible_native_scorer_sha256=compatible_native_scorer_sha256,
    )
    return samples, state


def run_sequential_causal_smoke(*, threads: int = 4) -> dict[str, object]:
    from .policies.autonomous_conservative_q import _decode_model
    from .policies.offline_ranker import PortableXGBoostRegressor

    samples, state = fit_sequential_causal_fixture(threads=threads)
    scorers = [
        PortableXGBoostRegressor(
            _decode_model(model),
            expected_feature_schema=RICH_FEATURE_SCHEMA,
            expected_feature_names=rich_feature_names(),
        )
        for model in state["models"]
    ]
    representation = state["representation"]["hazard_codebook"]
    from .advantage_learning import rich_candidate_vector

    predictions = []
    for state_risk in (0.0, 1.0):
        base_names = tree_feature_names(
            OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES
        )
        indices = {name: index for index, name in enumerate(base_names)}
        baseline = [0.0] * len(base_names)
        baseline[indices["observation:position_x_unit"]] = state_risk
        baseline[indices["matches_baseline"]] = 1.0
        candidate = baseline.copy()
        candidate[indices["action:direction_x"]] = -1.0
        candidate[indices["delta_from_baseline:direction_x"]] = -1.0
        candidate[indices["matches_baseline"]] = 0.0
        primitive = [0.0] * len(HAZARD_PRIMITIVE_FEATURE_NAMES)
        primitive[0] = state_risk
        primitive[11] = 1.0
        history = [0.0] * len(HISTORY_FEATURE_NAMES)
        history[0] = 1.0
        history[2] = state_risk
        rows = [
            rich_candidate_vector(
                tuple(vector),
                (tuple(primitive),),
                tuple(history),
                representation,
            )
            for vector in (baseline, candidate)
        ]
        predictions.append([
            float(scorer.predict_many([rows[1]])[0]
                  - scorer.predict_many([rows[0]])[0])
            for scorer in scorers
        ])
    flat = [value for row in predictions for value in row]
    mean = sum(flat) / len(flat)
    member_leakage = max(
        abs(left - right)
        for left, right in zip(predictions[0], predictions[1], strict=True)
    )
    aggregate_leakage = abs(
        sum(predictions[0]) / len(predictions[0])
        - sum(predictions[1]) / len(predictions[1])
    )
    calibration = state["fit_report"]["policy_calibration"]
    gates = {
        "fit_contract": state["authorization"]["fit_eligible"] is True,
        "all_members_recover_negative_effect": all(value < 0.0 for value in flat),
        "known_effect_error_at_most_half_hit": abs(mean - (-1.0)) <= 0.5,
        "aggregate_state_risk_leakage_below_half_hit": (
            aggregate_leakage < 0.5
        ),
        "every_member_state_risk_leakage_below_one_hit": (
            member_leakage < 1.0
        ),
        "crossfit_r_loss_beats_zero": (
            calibration["critic_r_loss"] < calibration["zero_r_loss"]
        ),
        "no_inverse_propensity_objective": (
            state["fit_report"]["centered_objective_uses_inverse_propensity"]
            is False
        ),
    }
    return {
        "schema": "autonomous-generation-4-sequential-causal-smoke-v1",
        "known_candidate_advantage": -1.0,
        "population_predictions_by_state_risk": predictions,
        "prediction_mean": mean,
        "prediction_range": max(flat) - min(flat),
        "aggregate_state_risk_leakage": aggregate_leakage,
        "member_state_risk_leakage_max": member_leakage,
        "crossfit_zero_r_loss": calibration["zero_r_loss"],
        "crossfit_critic_r_loss": calibration["critic_r_loss"],
        "gates": gates,
        "passed": all(gates.values()),
    }
