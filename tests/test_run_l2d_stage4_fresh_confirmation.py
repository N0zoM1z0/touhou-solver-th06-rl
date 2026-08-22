from __future__ import annotations

import json
from pathlib import Path

from scripts.run_l2d_stage4_fresh_confirmation import load_prereg


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPOSITORY / "experiments/l2d-stage4-fresh-confirmation-v1.json"


def test_l2d_preregistration_freezes_collection_and_confirmation_boundaries() -> None:
    prereg = load_prereg(PREREGISTRATION)

    collection = prereg["collection"]
    assert len(collection["episodes"]) == 8
    assert collection["serial_wine_workers"] == 1
    assert collection["exploration_probability"] == 0.2
    assert collection["physical_hit_retry"] is False
    assert collection["peek_or_sequential_stop"] is False
    assert prereg["evaluation"]["primary_horizon_game_frames"] == 16
    assert prereg["gate"]["history_admitted"] is False
    assert prereg["gate"]["value_learning_admitted"] is False
    assert prereg["gate"]["online_policy_admitted"] is False
    assert prereg["fits_model"] is False
    assert prereg["runs_learned_policy"] is False


def test_l2d_preregistration_is_canonical_json_data() -> None:
    observed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert observed["schema"] == "th06-rl-l2d-stage4-fresh-confirmation-prereg-v1"
    assert all(
        row["split"] == "independent-confirmation"
        for row in observed["collection"]["episodes"]
    )
