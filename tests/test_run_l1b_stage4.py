from __future__ import annotations

from pathlib import Path

from scripts.run_l1b_stage4 import fit_command, load_prereg, result_decision


REPOSITORY = Path(__file__).resolve().parents[1]
PREREG = REPOSITORY / "experiments/l1b-stage4-bc-convergence-v1.json"


def test_l1b_prereg_changes_only_optimization_convergence() -> None:
    prereg = load_prereg(PREREG)

    assert prereg["data"]["reuse_without_mutation"] is True
    assert prereg["data"]["train_episode_indices"] == [0, 1, 3, 4, 6, 7, 9, 10]
    assert prereg["data"]["validation_episode_indices"] == [2, 5, 8, 11]
    assert prereg["fit"]["model"] == "masked-linear-softmax"
    assert prereg["fit"]["feature_schema"] == (
        "th06-rl-current-observation-features-v1"
    )
    assert prereg["auxiliary_targets"] == []
    assert prereg["online_canary"]["run_in_this_experiment"] is False


def test_l1b_fit_command_reuses_exact_split_without_wine(tmp_path) -> None:
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
    assert "--relative-gradient-l2-tolerance" in command
    assert "--minimum-updates" in command
    assert all("wine" not in value.lower() for value in command)


def test_l1b_decision_requires_convergence_before_any_canary() -> None:
    assert result_decision({
        "fit": {
            "optimization": {"converged": False},
            "learnability_gate_passed": False,
        }
    }) == "inconclusive-l1b-optimization-not-converged"
    assert result_decision({
        "fit": {
            "optimization": {"converged": True},
            "learnability_gate_passed": False,
        }
    }) == "stop-l1b-linear-current-observation"
    assert result_decision({
        "fit": {
            "optimization": {"converged": True},
            "learnability_gate_passed": True,
        }
    }) == "admit-stage4-bc-integration-canary"
