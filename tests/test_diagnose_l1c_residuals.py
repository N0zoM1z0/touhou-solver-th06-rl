from __future__ import annotations

from pathlib import Path

from scripts.diagnose_l1c_residuals import load_prereg


REPOSITORY = Path(__file__).resolve().parents[1]
PREREG = REPOSITORY / "experiments/l1c-stage4-residual-diagnosis-v1.json"


def test_l1c_residual_prereg_is_read_only_and_train_selected() -> None:
    prereg = load_prereg(PREREG)

    assert prereg["data"]["reuse_without_mutation"] is True
    assert prereg["diagnosis"]["fit_deployable_model"] is False
    assert prereg["diagnosis"]["evaluate_scaled_validation"] is False
    assert prereg["selection_rule"]["selection_uses_transformed_validation"] is False
    assert prereg["selection_rule"]["history_for_behavior_target"] is False
