from __future__ import annotations

import json
from pathlib import Path

from scripts.run_l2e_train_only_calibration import load_prereg


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPOSITORY / "experiments/l2e-train-only-calibration-v1.json"


def test_l2e_preregistration_changes_only_the_probability_surface() -> None:
    prereg = load_prereg(PREREGISTRATION)

    assert prereg["fit"]["horizon_game_frames"] == 16
    assert prereg["fit"]["probability_surface"] == (
        "two-parameter-affine-logistic-platt"
    )
    assert prereg["fit"]["source_ridge_l2"] == 0.001
    assert prereg["data"]["reuses_previously_evaluated_l2d"] is True
    assert prereg["data"]["independent_confirmation"] is False
    assert prereg["gate"]["fresh_confirmation_required_if_selected"] is True
    assert prereg["gate"]["history_admitted"] is False
    assert prereg["gate"]["value_learning_admitted"] is False
    assert prereg["online_wine"] is False
    assert prereg["fits_policy"] is False


def test_l2e_preregistration_is_canonical_json_data() -> None:
    observed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert observed["schema"] == "th06-rl-l2e-train-only-calibration-prereg-v1"
    assert observed["non_goals"]
