"""Finite-horizon FQE and sequential-DR checks for one exact policy object.

The one-step estimand changes only the current option and then returns to the
recorded behavior policy.  The sequential estimand applies the target (or
reference) policy for every option in a window.  Keeping these results under
different names prevents an invalid magnitude comparison between estimands.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from .linear_models import MINIMUM_FEATURE_SCALE


@dataclass(frozen=True)
class _HorizonQ:
    design: object
    models: tuple[tuple[object, float], ...]


class _FixedRidgeDesign:
    """Reuse one standardized Gram matrix for every Bellman iteration."""

    def __init__(self, rows, *, alpha: float) -> None:
        import numpy as np

        self.mean = rows.mean(axis=0)
        self.scale = rows.std(axis=0)
        self.scale = np.maximum(self.scale, MINIMUM_FEATURE_SCALE)
        self.rows = (rows - self.mean) / self.scale
        gram = self.rows.T @ self.rows
        gram.flat[::gram.shape[0] + 1] += alpha
        self.gram = gram

    def fit(self, targets):
        import numpy as np

        target_mean = float(np.mean(targets))
        coefficient = np.linalg.solve(
            self.gram,
            self.rows.T @ (targets - target_mean),
        )
        return coefficient, target_mean

    def transform(self, rows):
        return (rows - self.mean) / self.scale


def _fit_horizon_q(
    episodes,
    *,
    train_indices: tuple[int, ...],
    factual_rows: dict[int, object],
    continuation_rows: dict[int, object],
    horizon: int,
    ridge_alpha: float,
) -> _HorizonQ:
    import numpy as np

    design = _FixedRidgeDesign(
        np.concatenate(tuple(factual_rows[index] for index in train_indices)),
        alpha=ridge_alpha,
    )
    zero = (np.zeros(design.rows.shape[1], dtype=np.float64), 0.0)
    models = [zero]
    transformed = {
        index: design.transform(continuation_rows[index])
        for index in train_indices
    }
    for _remaining in range(1, horizon + 1):
        previous_coefficient, previous_intercept = models[-1]
        targets = []
        for index in train_indices:
            episode = episodes[index]
            target = episode.hit_costs.copy()
            if episode.option_count > 1:
                target[:-1] += (
                    transformed[index][1:] @ previous_coefficient
                    + previous_intercept
                )
            targets.append(target)
        models.append(design.fit(np.concatenate(tuple(targets))))
    return _HorizonQ(design=design, models=tuple(models))


def _predict(fit: _HorizonQ, rows, remaining: int):
    coefficient, intercept = fit.models[remaining]
    return fit.design.transform(rows) @ coefficient + intercept


def _prediction_table(fit: _HorizonQ, rows):
    """Return [remaining, row] predictions with one matrix transform."""
    import numpy as np

    transformed = fit.design.transform(rows)
    return np.asarray([
        transformed @ coefficient + intercept
        for coefficient, intercept in fit.models
    ])


def evaluate_fqe_crosschecks(
    episodes,
    *,
    train_indices: tuple[int, ...],
    held_indices: tuple[int, ...],
    factual_rows: dict[int, object],
    behavior_expected_rows: dict[int, object],
    target_expected_rows: dict[int, object],
    reference_expected_rows: dict[int, object],
    target_candidate_probabilities: dict[int, object],
    reference_candidate_probabilities: dict[int, object],
    horizon: int,
    ridge_alpha: float,
) -> dict[str, object]:
    """Evaluate matched one-step and repeated-policy estimands.

    Returns per-episode rows.  No probability clipping is permitted: the
    cumulative-weight diagnostics expose whether sequential DR has usable
    support instead of silently changing its estimand.
    """
    import numpy as np

    behavior_q = _fit_horizon_q(
        episodes,
        train_indices=train_indices,
        factual_rows=factual_rows,
        continuation_rows=behavior_expected_rows,
        horizon=horizon,
        ridge_alpha=ridge_alpha,
    )
    target_q = _fit_horizon_q(
        episodes,
        train_indices=train_indices,
        factual_rows=factual_rows,
        continuation_rows=target_expected_rows,
        horizon=horizon,
        ridge_alpha=ridge_alpha,
    )
    reference_q = _fit_horizon_q(
        episodes,
        train_indices=train_indices,
        factual_rows=factual_rows,
        continuation_rows=reference_expected_rows,
        horizon=horizon,
        ridge_alpha=ridge_alpha,
    )
    results = {
        name: defaultdict(list)
        for name in ("one_step_fqe", "sequential_fqe", "sequential_dr")
    }
    weight_moments = {
        name: {"count": 0, "sum": 0.0, "sum_squares": 0.0, "maximum": 0.0}
        for name in ("target", "reference")
    }
    for index in held_indices:
        episode = episodes[index]
        factual = episode.offsets[:-1] + episode.factual_positions
        behavior_factual = episode.behavior_probabilities[factual]
        probability_rows = {
            "target": np.asarray(target_candidate_probabilities[index])[factual],
            "reference": np.asarray(reference_candidate_probabilities[index])[factual],
        }
        fits = {"target": target_q, "reference": reference_q}
        expected = {
            "target": target_expected_rows[index],
            "reference": reference_expected_rows[index],
        }
        one_step_tables = {
            "target": _prediction_table(
                behavior_q, target_expected_rows[index]
            ),
            "reference": _prediction_table(
                behavior_q, reference_expected_rows[index]
            ),
        }
        expected_tables = {
            name: _prediction_table(fits[name], expected[name])
            for name in ("target", "reference")
        }
        factual_tables = {
            name: _prediction_table(fits[name], factual_rows[index])
            for name in ("target", "reference")
        }
        for start in range(episode.option_count):
            length = min(horizon, episode.option_count - start)
            one_step_target = float(one_step_tables["target"][length, start])
            one_step_reference = float(one_step_tables["reference"][length, start])
            results["one_step_fqe"][episode.episode_id].append(
                one_step_target - one_step_reference
            )
            sequential_values = {}
            sequential_dr_values = {}
            for name in ("target", "reference"):
                fit = fits[name]
                value = float(expected_tables[name][length, start])
                dr_value = value
                cumulative_ratio = 1.0
                for offset in range(length):
                    option = start + offset
                    remaining = length - offset
                    cumulative_ratio *= float(
                        probability_rows[name][option] / behavior_factual[option]
                    )
                    if not math.isfinite(cumulative_ratio):
                        raise FloatingPointError("sequential DR weight is non-finite")
                    moment = weight_moments[name]
                    moment["count"] += 1
                    moment["sum"] += cumulative_ratio
                    moment["sum_squares"] += cumulative_ratio * cumulative_ratio
                    moment["maximum"] = max(moment["maximum"], cumulative_ratio)
                    factual_q = float(factual_tables[name][remaining, option])
                    next_value = 0.0
                    if offset + 1 < length:
                        next_value = float(
                            expected_tables[name][remaining - 1, option + 1]
                        )
                    dr_value += cumulative_ratio * (
                        float(episode.hit_costs[option]) + next_value - factual_q
                    )
                sequential_values[name] = value
                sequential_dr_values[name] = dr_value
            results["sequential_fqe"][episode.episode_id].append(
                sequential_values["target"] - sequential_values["reference"]
            )
            results["sequential_dr"][episode.episode_id].append(
                sequential_dr_values["target"]
                - sequential_dr_values["reference"]
            )
    diagnostics = {}
    for name, values in weight_moments.items():
        diagnostics[name] = {
            **values,
            "effective_sample_size": (
                values["sum"] * values["sum"] / values["sum_squares"]
                if values["sum_squares"] > 0.0 else 0.0
            ),
        }
    return {"estimates": results, "cumulative_weight_diagnostics": diagnostics}
