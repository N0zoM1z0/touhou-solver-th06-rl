from __future__ import annotations

import json
from pathlib import Path

from scripts.run_l2_stage4_factual_probes import load_prereg
from th06_rl.factual_probes import PROBE_FEATURE_NAMES


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPOSITORY / "experiments/l2-stage4-factual-probes-v1.json"


def test_l2_factual_probe_preregistration_freezes_the_small_pilot() -> None:
    prereg = load_prereg(PREREGISTRATION)

    assert prereg["probes"]["feature_names"] == list(PROBE_FEATURE_NAMES)
    assert prereg["probes"]["horizons_game_frames"] == [1, 4, 16, 64]
    assert prereg["data"]["validation_episode_indices"] == [2, 5, 8, 11]
    assert prereg["gate"]["history_admitted_by_this_experiment"] is False
    assert prereg["online_wine"] is False
    assert prereg["fits_policy"] is False


def test_l2_factual_probe_preregistration_is_canonical_json_data() -> None:
    observed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert observed["schema"] == "th06-rl-l2-stage4-factual-probes-prereg-v1"
    assert observed["probes"]["risk_model"] == (
        "standardized-ridge-linear-probability"
    )
