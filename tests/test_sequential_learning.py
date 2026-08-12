from __future__ import annotations

import numpy as np

from th06_rl.sequential_learning import (
    _causal_episodes,
    _centered_layout,
    _critic_regressor,
    _episodes,
    _n_step_targets,
    _probabilities,
    OrthogonalOption,
)


def test_semi_markov_return_is_recursive_sum_of_interval_hits() -> None:
    samples = _causal_episodes(count=1, options=16)
    rows = _episodes(samples)["causal-00"]

    class FrozenValue:
        def predict(self, matrix):
            return np.zeros(len(matrix), dtype=np.float32)

    targets = _n_step_targets(
        {"causal-00": rows},
        [FrozenValue(), FrozenValue(), FrozenValue()],
    )

    assert targets[("causal-00", rows[-1].option_id)] == rows[-1].option_hit_cost
    assert targets[("causal-00", rows[-8].option_id)] == sum(
        row.option_hit_cost for row in rows[-8:]
    )
    assert rows[0].return_to_go == sum(row.option_hit_cost for row in rows)


def test_action_centered_objective_recovers_effect_without_inverse_propensity() -> None:
    raw = _causal_episodes(count=20, options=32)
    orthogonal = [
        OrthogonalOption(
            step=sample,
            n_step_target=sample.option_hit_cost,
            outcome_residual=(
                -0.75 if sample.action == "left" else 0.25
            ),
            fold=0,
        )
        for sample in raw
    ]
    layout = _centered_layout(orthogonal)
    assert np.abs(layout.coefficients).max() <= 0.75
    assert all(_probabilities(sample) == (0.75, 0.25) for sample in raw)

    model = _critic_regressor(
        layout,
        episode_weights={f"causal-{index:02d}": 1 for index in range(20)},
        trees=48,
        seed=260812,
        threads=1,
    )
    first = raw[0]
    scores = model.predict(np.asarray(first.candidate_vectors, dtype=np.float32))
    assert float(scores[1] - scores[0]) < -0.8
    assert float(scores[1] - scores[0]) > -1.2
