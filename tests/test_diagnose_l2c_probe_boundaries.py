from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_l2c_probe_boundaries import load_prereg


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPOSITORY / "experiments/l2c-stage4-boundary-diagnosis-v1.json"


def test_l2c_preregistration_is_read_only_and_requires_fresh_confirmation() -> None:
    prereg = load_prereg(PREREGISTRATION)

    assert prereg["data"]["reuses_previously_evaluated_validation"] is True
    assert prereg["data"]["independent_confirmation"] is False
    assert prereg["diagnosis"]["horizons_game_frames"] == [16, 64]
    assert prereg["gate"]["descriptive_only"] is True
    assert prereg["gate"]["fresh_confirmation_required"] is True
    assert prereg["gate"]["history_admitted"] is False
    assert prereg["gate"]["value_learning_admitted"] is False


def test_l2c_preregistration_is_canonical_json_data() -> None:
    observed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert observed["schema"] == "th06-rl-l2c-boundary-prereg-v1"
    assert observed["online_wine"] is False
    assert observed["fits_policy"] is False
