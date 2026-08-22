from __future__ import annotations

import json
from pathlib import Path

from scripts.run_l2j_logscore_primitive_hazard import load_prereg


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    REPOSITORY / "experiments/l2j-logscore-primitive-hazard-v1.json"
)


def test_l2j_preregistration_changes_only_training_proper_score() -> None:
    prereg = load_prereg(PREREGISTRATION)
    fit = prereg["fit"]

    assert fit["training_proper_score"] == (
        "mean-unweighted-bernoulli-log-score-with-logits"
    )
    assert fit["loss_implementation"] == "torch.nn.BCEWithLogitsLoss"
    assert fit["positive_class_weight"] == 1.0
    assert fit["negative_class_weight"] == 1.0
    assert fit["primitive_token_cap"] == 48
    assert fit["epochs"] == 40
    assert fit["batch_size"] == 4096
    assert fit["seed"] == 20260846
    assert fit["loader_workers"] == 8
    assert fit["threads"] == 32
    assert prereg["data"]["reuses_previously_evaluated_l2d"] is True
    assert prereg["data"]["independent_confirmation"] is False
    assert prereg["gate"]["minimum_loss_correction_episodes"] == 6
    assert prereg["gate"]["fresh_confirmation_required_if_selected"] is True
    assert prereg["gate"]["history_admitted"] is False
    assert prereg["gate"]["value_learning_admitted"] is False
    assert prereg["gate"]["online_policy_admitted"] is False
    assert prereg["online_wine"] is False
    assert prereg["fits_policy"] is False


def test_l2j_preregistration_binds_both_probability_models() -> None:
    observed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert observed["schema"] == (
        "th06-rl-l2j-logscore-primitive-hazard-prereg-v1"
    )
    assert observed["non_goals"]
    for key in (
        "primitive_brier_module",
        "primitive_log_hazard_module",
        "primitive_log_hazard_runner",
        "source_l2i_fit",
        "source_l2i_result",
    ):
        assert len(observed["sha256_bindings"][key]) == 64
