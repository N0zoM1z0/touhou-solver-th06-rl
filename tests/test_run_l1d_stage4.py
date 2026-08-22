from __future__ import annotations

import json
from pathlib import Path

from scripts.run_l1d_stage4 import fit_command, load_prereg, result_decision


REPOSITORY = Path(__file__).resolve().parents[1]
PREREG = REPOSITORY / "experiments/l1d-stage4-bc-mlp-v1.json"
L1C_PREREG = REPOSITORY / "experiments/l1c-stage4-bc-timebox-v1.json"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_l1d_prereg_changes_only_architecture_and_required_initialization() -> None:
    prereg = load_prereg(PREREG)
    l1c = _json(L1C_PREREG)

    assert prereg["data"] == l1c["data"]
    assert prereg["fit"]["model"] == "masked-one-hidden-relu-softmax"
    assert prereg["fit"]["hidden_width"] == 32
    assert prereg["fit"]["initialization"] == "fixed-seed-he-normal"
    for key in (
        "bootstrap_samples",
        "calibration_schema",
        "calibration_tolerance",
        "decision_epoch_schema",
        "feature_schema",
        "l2",
        "learning_rate",
        "max_rows_per_split",
        "maximum_updates",
        "minimum_updates",
        "optimizer",
        "relative_gradient_l2_tolerance",
        "seed",
        "target_schema",
        "validation_use",
    ):
        assert prereg["fit"][key] == l1c["fit"][key]


def test_l1d_fit_command_is_offline_and_uses_the_frozen_l1c_comparator(
    tmp_path: Path,
) -> None:
    prereg = load_prereg(PREREG)
    inventory = {
        index: {"run_dir": f"corpora/l1/run-{index}"}
        for index in range(12)
    }

    command = fit_command(prereg, inventory, tmp_path / "model.json")
    train = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--train-run"
    ]
    validation = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--validation-run"
    ]

    assert len(train) == 8
    assert len(validation) == 4
    assert set(train).isdisjoint(validation)
    assert command[command.index("--hidden-width") + 1] == "32"
    comparator = command[command.index("--linear-comparator-state") + 1]
    assert comparator.endswith("l1c-stage4-bc-timebox-v1/model/linear-bc.json")
    assert all("wine" not in value.lower() for value in command)


def test_l1d_decision_requires_convergence_and_the_direct_joint_gate() -> None:
    assert result_decision({
        "fit": {
            "optimization": {"converged": False},
            "learnability_gate_passed": False,
        }
    }) == "inconclusive-l1d-mlp-optimization-not-converged"
    assert result_decision({
        "fit": {
            "optimization": {"converged": True},
            "learnability_gate_passed": False,
        }
    }) == "stop-l1d-small-current-observation-mlp"
    assert result_decision({
        "fit": {
            "optimization": {"converged": True},
            "learnability_gate_passed": True,
        }
    }) == "admit-stage4-mlp-bc-integration-canary"
