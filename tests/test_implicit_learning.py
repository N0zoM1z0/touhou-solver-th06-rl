from __future__ import annotations

import numpy as np

from th06_rl.implicit_learning import (
    _episodes,
    _n_step_targets,
    delayed_effect_episodes,
    fit_implicit_q_population,
    pessimistic_action,
)


def _small_population(samples):
    return fit_implicit_q_population(
        samples,
        members=7,
        iterations=4,
        n_step_options=4,
        q_trees=40,
        value_trees=32,
        total_threads=4,
    )


def test_terminal_zero_n_step_target_conserves_physical_hits() -> None:
    samples = delayed_effect_episodes(count=1, options=32, delay=8)
    rows = _episodes(samples)["fixture-000"]

    targets = _n_step_targets(
        {"fixture-000": rows}, None, n_step_options=8
    )

    assert targets[("fixture-000", rows[-1].option_id)] == rows[-1].option_hit_cost
    assert targets[("fixture-000", rows[-8].option_id)] == sum(
        row.option_hit_cost for row in rows[-8:]
    )
    assert rows[0].return_to_go == sum(row.option_hit_cost for row in rows)


def test_implicit_q_recovers_effect_beyond_single_backup_horizon() -> None:
    samples = delayed_effect_episodes(count=64, options=72, delay=12)
    population = _small_population(samples)
    interior = [
        sample for sample in samples
        if 8 <= sample.sequence < 72 - 12
    ]

    decisions = [pessimistic_action(population, sample)[0] for sample in interior]
    member_effects = []
    for member in population:
        member_effects.append(float(np.mean([
            member.q_model.predict(
                np.asarray(sample.candidate_vectors, dtype=np.float32)
            )[1]
            - member.q_model.predict(
                np.asarray(sample.candidate_vectors, dtype=np.float32)
            )[0]
            for sample in interior
        ])))

    assert all(effect < 0.0 for effect in member_effects)
    assert decisions.count("left") > 0
    assert all(
        member.iterations[-1]["q_weighted_mse"]
        < member.iterations[-1]["zero_effect_weighted_mse"]
        for member in population
    )
    assert all(
        member.iterations[-1]["maximum_centered_coefficient"] == 0.5
        for member in population
    )


def test_implicit_q_abstains_when_randomized_action_has_no_effect() -> None:
    samples = delayed_effect_episodes(
        count=64, options=72, delay=12, null_effect=True
    )
    population = _small_population(samples)

    decisions = [pessimistic_action(population, sample)[0] for sample in samples]

    assert set(decisions) == {"stay"}
