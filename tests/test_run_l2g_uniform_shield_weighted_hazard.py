from __future__ import annotations

import json
from pathlib import Path

from scripts.run_l2g_uniform_shield_weighted_hazard import load_prereg


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    REPOSITORY / "experiments/l2g-uniform-shield-weighted-hazard-v1.json"
)


def test_l2g_preregistration_changes_only_the_bounded_action_measure() -> None:
    prereg = load_prereg(PREREGISTRATION)

    assert prereg["fit"]["horizon_game_frames"] == 16
    assert prereg["fit"]["importance_weight"] == (
        "(1 / observed_shield_action_count) / behavior_probability"
    )
    assert prereg["fit"]["maximum_importance_weight"] == 5.0
    assert prereg["fit"]["maximum_depth"] == 3
    assert prereg["fit"]["boosted_rounds"] == 64
    assert prereg["data"]["reuses_previously_evaluated_l2d"] is True
    assert prereg["data"]["independent_confirmation"] is False
    assert prereg["gate"]["fresh_confirmation_required_if_selected"] is True
    assert prereg["gate"]["history_admitted"] is False
    assert prereg["gate"]["value_learning_admitted"] is False
    assert prereg["gate"]["online_policy_admitted"] is False
    assert prereg["online_wine"] is False
    assert prereg["fits_policy"] is False


def test_l2g_preregistration_is_canonical_json_data() -> None:
    observed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert observed["schema"] == (
        "th06-rl-l2g-uniform-shield-weighted-hazard-prereg-v1"
    )
    assert observed["non_goals"]
    assert len(observed["sha256_bindings"]["weighted_hazard_module"]) == 64
    assert len(observed["sha256_bindings"]["weighted_hazard_runner"]) == 64
