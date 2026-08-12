from __future__ import annotations

from dataclasses import replace

import numpy as np

from th06_rl.implicit_learning import delayed_effect_episodes
from th06_rl.iql_actor_learning import actor_arrays
from th06_rl.low_rank_learning import FeatureRoleLayout


def test_actor_arrays_keep_variable_safe_sets_and_factual_index() -> None:
    raw = delayed_effect_episodes(count=2, options=6, delay=2)
    samples = []
    for index, sample in enumerate(raw):
        if index % 2:
            samples.append(replace(
                sample,
                action="stay",
                behavior_probability=1.0,
                behavior_probabilities=(1.0,),
                vector=sample.candidate_vectors[0],
                legal_actions=("stay",),
                candidate_vectors=(sample.candidate_vectors[0],),
            ))
        else:
            samples.append(sample)
    layout = FeatureRoleLayout(
        names=("observation:p0", "observation:p1", "action:treatment"),
        state_indices=(0, 1),
        action_indices=(2,),
    )

    prepared = actor_arrays(samples, layout)

    assert prepared.actions.shape == (12, 2, 1)
    assert np.array_equal(prepared.masks.sum(axis=1), [2, 1] * 6)
    assert np.allclose(prepared.behavior_probabilities.sum(axis=1), 1.0)
    assert all(
        sample.legal_actions[index] == sample.action
        for sample, index in zip(samples, prepared.factual, strict=True)
    )
