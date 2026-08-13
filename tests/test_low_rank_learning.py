from __future__ import annotations

import numpy as np

from th06_rl.low_rank_learning import (
    FeatureRoleLayout,
    LowRankEffectModel,
    named_feature_roles,
)


def test_named_feature_roles_partition_state_and_action() -> None:
    layout = named_feature_roles((
        "observation:x",
        "action:dx",
        "delta_from_baseline:dx",
        "matches_baseline",
        "history:lag0:x",
    ))

    assert layout.state_indices == (0, 4)
    assert layout.action_indices == (1, 2, 3)


def test_centered_prediction_equals_propensity_weighted_candidate_scores() -> None:
    layout = FeatureRoleLayout(
        names=("observation:x", "action:dx"),
        state_indices=(0,),
        action_indices=(1,),
    )
    model = LowRankEffectModel(
        layout=layout,
        state_mean=np.asarray([0.25], dtype=np.float32),
        state_scale=np.asarray([2.0], dtype=np.float32),
        action_mean=np.asarray([0.4], dtype=np.float32),
        action_scale=np.asarray([0.5], dtype=np.float32),
        state_hidden_weight=np.asarray([[0.7]], dtype=np.float32),
        state_hidden_bias=np.asarray([0.1], dtype=np.float32),
        state_latent_weight=np.asarray([[1.3]], dtype=np.float32),
        state_latent_bias=np.asarray([-0.2], dtype=np.float32),
        action_latent_weight=np.asarray([[0.8]], dtype=np.float32),
        action_bias_weight=np.asarray([0.35], dtype=np.float32),
    )
    candidates = np.asarray([[1.5, -1.0], [1.5, 1.0]], dtype=np.float32)
    probabilities = np.asarray([0.3, 0.7], dtype=np.float32)
    factual = 0
    scores = model.predict(candidates)
    direct = scores[factual] - probabilities @ scores
    centered_action = np.asarray([[
        candidates[factual, 1] - probabilities @ candidates[:, 1]
    ]], dtype=np.float32)
    centered = model.predict_centered(
        np.asarray([[1.5]], dtype=np.float32), centered_action
    )[0]

    assert np.isclose(centered, direct, atol=1e-6)
