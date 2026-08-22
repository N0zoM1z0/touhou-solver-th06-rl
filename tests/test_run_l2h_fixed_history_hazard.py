from __future__ import annotations

import json
from pathlib import Path

from scripts.run_l2h_fixed_history_hazard import load_prereg


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPOSITORY / "experiments/l2h-fixed-history-hazard-v1.json"


def test_l2h_preregistration_changes_only_fixed_causal_history() -> None:
    prereg = load_prereg(PREREGISTRATION)

    assert prereg["fit"]["history_length_decision_roots"] == 16
    assert prereg["fit"]["history_fields_per_root"] == 15
    assert prereg["fit"]["history_total_fields"] == 255
    assert prereg["fit"]["history_padding"] == (
        "none-drop-row-without-complete-prefix"
    )
    assert prereg["fit"]["history_requires_unit_game_frame_intervals"] is True
    assert prereg["fit"]["training_proper_score"] == "mean-unweighted-row-brier"
    assert prereg["fit"]["maximum_depth"] == 3
    assert prereg["fit"]["boosted_rounds"] == 64
    assert prereg["fit"]["threads"] == 16
    assert prereg["data"]["reuses_previously_evaluated_l2d"] is True
    assert prereg["gate"]["fresh_confirmation_required_if_selected"] is True
    assert prereg["gate"]["history_admitted"] is True
    assert prereg["gate"]["value_learning_admitted"] is False
    assert prereg["gate"]["online_policy_admitted"] is False
    assert prereg["online_wine"] is False
    assert prereg["fits_policy"] is False


def test_l2h_preregistration_is_canonical_json_data() -> None:
    observed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert observed["schema"] == "th06-rl-l2h-fixed-history-hazard-prereg-v1"
    assert observed["non_goals"]
    assert len(observed["sha256_bindings"]["history_dataset_module"]) == 64
    assert len(observed["sha256_bindings"]["history_hazard_module"]) == 64
    assert len(observed["sha256_bindings"]["history_hazard_runner"]) == 64
