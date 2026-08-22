from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_l2b_incremental_action import load_prereg
from th06_rl.factual_probe_diagnostics import STATE_ONLY_FEATURE_NAMES


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPOSITORY / "experiments/l2b-incremental-action-v1.json"


def test_l2b_preregistration_is_one_reused_holdout_feature_ablation() -> None:
    prereg = load_prereg(PREREGISTRATION)

    assert prereg["diagnosis"]["state_only_feature_names"] == list(
        STATE_ONLY_FEATURE_NAMES
    )
    assert prereg["diagnosis"]["supported_hit_horizons"] == [16, 64]
    assert prereg["data"]["reuses_previously_evaluated_validation"] is True
    assert prereg["data"]["independent_confirmation"] is False
    assert prereg["gate"]["history_admitted"] is False
    assert prereg["gate"]["value_learning_admitted"] is False


def test_l2b_preregistration_is_canonical_json_data() -> None:
    observed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert observed["schema"] == "th06-rl-l2b-incremental-action-prereg-v1"
    assert observed["online_wine"] is False
    assert observed["fits_policy"] is False
