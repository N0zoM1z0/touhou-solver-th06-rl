from __future__ import annotations

import json
from pathlib import Path

from scripts.run_l1c_stage4 import fit_command, load_prereg, result_decision


REPOSITORY = Path(__file__).resolve().parents[1]
PREREG = REPOSITORY / "experiments/l1c-stage4-bc-timebox-v1.json"
L1B_PREREG = REPOSITORY / "experiments/l1b-stage4-bc-convergence-v1.json"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _without(mapping: object, *keys: str) -> dict[str, object]:
    assert isinstance(mapping, dict)
    return {key: value for key, value in mapping.items() if key not in keys}


def test_l1c_prereg_changes_only_maximum_update_timebox() -> None:
    prereg = load_prereg(PREREG)
    l1b = _json(L1B_PREREG)

    assert prereg["fit"]["maximum_updates"] == 10000
    assert l1b["fit"]["maximum_updates"] == 2000
    assert _without(prereg["fit"], "maximum_updates") == _without(
        l1b["fit"], "maximum_updates"
    )
    assert _without(prereg["gate"], "timebox") == _without(
        l1b["gate"], "timebox"
    )
    for key in ("auxiliary_targets", "comparators", "data", "online_canary"):
        assert prereg[key] == l1b[key]


def test_l1c_fit_command_restarts_from_zero_without_wine(tmp_path: Path) -> None:
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
    assert command[command.index("--epochs") + 1] == "10000"
    assert command[command.index("--minimum-updates") + 1] == "100"
    assert command[command.index("--relative-gradient-l2-tolerance") + 1] == "0.01"
    assert all("wine" not in value.lower() for value in command)


def test_l1c_decision_requires_convergence_before_any_canary() -> None:
    assert result_decision({
        "fit": {
            "optimization": {"converged": False},
            "learnability_gate_passed": False,
        }
    }) == "inconclusive-l1c-optimization-not-converged"
    assert result_decision({
        "fit": {
            "optimization": {"converged": True},
            "learnability_gate_passed": False,
        }
    }) == "stop-l1c-linear-current-observation"
    assert result_decision({
        "fit": {
            "optimization": {"converged": True},
            "learnability_gate_passed": True,
        }
    }) == "admit-stage4-bc-integration-canary"
