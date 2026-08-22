from __future__ import annotations

from pathlib import Path

from scripts.diagnose_l1d_target_contract import load_prereg


REPOSITORY = Path(__file__).resolve().parents[1]
PREREG = REPOSITORY / "experiments/l1d-stage4-target-diagnosis-v1.json"


def test_l1d_target_diagnosis_is_train_only_and_non_deployable() -> None:
    prereg = load_prereg(PREREG)
    diagnosis = prereg["diagnosis"]

    assert prereg["data"]["reuse_without_mutation"] is True
    assert diagnosis["continuation_updates"] == 500
    assert diagnosis["continuation_checkpoints"] == [0, 100, 250, 500]
    assert diagnosis["evaluate_continuation_validation"] is False
    assert diagnosis["serialize_continuation_parameters"] is False
    assert diagnosis["fit_deployable_model"] is False
