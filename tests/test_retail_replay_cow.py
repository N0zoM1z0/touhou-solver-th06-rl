from __future__ import annotations

import pytest

from scripts.label_retail_replay_cow import retail_checkpoint_contract
from scripts.label_headless_cow_counterfactuals import label_checkpoint


def _row():
    transition = {"published_action": "right_fast"}
    frame = {
        "snapshot_id": "run:00000001:f3032",
        "scope": {"phase_id": "boss:0:sub10"},
        "snapshot": {
            "frame": 3032,
            "difficulty": 3,
            "character": 0,
            "shot_type": 0,
            "stage": 6,
            "rng_seed": 10,
            "rng_generation": 2,
            "input_mask": 0x01,
            "x": 120.0,
            "y": 432.0,
            "player_state": 0,
            "half_width": 1.25,
            "half_height": 1.25,
            "lives_remaining": 2,
            "current_power": 128,
            "rank": 20,
            "timeline_time": 2800,
            "live_bullet_count": 400,
            "laser_count": 0,
        },
        "decision": {
            "current_action": "stay_fast",
            "published_action": "right_fast",
            "baseline_action": "left_fast",
            "hard_actions": [
                ["left_fast", 10.0, 116.0, 432.0],
                ["right_fast", 9.0, 124.0, 432.0],
            ],
        },
    }
    return transition, frame


def test_retail_checkpoint_contract_preserves_native_set_and_actions() -> None:
    transition, frame = _row()

    result = retail_checkpoint_contract(transition, frame)

    assert result["tick"] == 3032
    assert result["factual_action"] == "right_fast"
    assert result["local_teacher_action"] == "left_fast"
    assert result["native_legal_actions"] == ["left_fast", "right_fast"]
    assert result["source_context"] == "boss:0:sub10"


def test_retail_checkpoint_contract_refuses_factual_outside_native_set() -> None:
    transition, frame = _row()
    transition["published_action"] = "up"

    with pytest.raises(ValueError, match="native-hard-safe"):
        retail_checkpoint_contract(transition, frame)


def test_targeted_label_api_exposes_a_native_legal_subset() -> None:
    # Signature-level regression: retail COW may compare a targeted pair
    # without redefining the full native authority set stored in the row.
    assert "evaluated_first_actions" in label_checkpoint.__annotations__
