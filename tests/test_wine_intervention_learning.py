from th06_rl.wine_intervention_learning import (
    FEATURE_NAMES,
    action_relative_features,
)


def test_action_relative_features_are_phase_and_rng_free() -> None:
    features = action_relative_features(
        player_x=100.0,
        player_y=432.0,
        bullet_count=300,
        hard_action_count=3,
        local_action_count=2,
        effort_horizon=4,
        current_action="left",
        baseline_action="up",
        action="up",
        incumbent_action="left",
        evaluations=(
            ("left", 2.0, 92.0, 432.0),
            ("up", 3.0, 100.0, 424.0),
        ),
    )
    assert tuple(features) == FEATURE_NAMES
    assert features["clearance_delta_incumbent"] == 1.0
    assert features["edge_delta_incumbent"] == 8.0
    assert features["delta_dx_incumbent"] == 1.0
    assert features["delta_dy_incumbent"] == -1.0
    assert not any("rng" in name or "phase" in name or "frame" in name for name in features)
