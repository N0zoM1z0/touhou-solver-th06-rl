from __future__ import annotations

import json
from pathlib import Path

from scripts.run_l2i_observed_primitive_set_hazard import load_prereg
from th06_rl.factual_primitive_dataset import TOKEN_FEATURE_NAMES
from th06_rl.factual_primitive_hazard_model import PrimitiveSetHazard


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    REPOSITORY / "experiments/l2i-observed-primitive-set-hazard-v1.json"
)


def test_l2i_preregistration_freezes_one_bounded_object_set_test() -> None:
    prereg = load_prereg(PREREGISTRATION)
    fit = prereg["fit"]

    assert fit["primitive_contract"] == "observed-hazard-kinematics-v1"
    assert fit["primitive_projection_frames"] == 4
    assert fit["primitive_token_cap"] == 48
    assert fit["primitive_token_width"] == len(TOKEN_FEATURE_NAMES) == 13
    assert fit["model"] == "shared-sigmoid-deepsets-brier-regressor"
    assert fit["models"] == [
        "object_full",
        "scalar_only",
        "object_current_action_ablated",
    ]
    assert fit["training_proper_score"] == "mean-unweighted-row-brier"
    assert fit["epochs"] == 40
    assert fit["threads"] == 32
    assert fit["loader_workers"] == 8
    assert fit["shared_initialization"] is True
    assert fit["shared_minibatch_order"] is True
    assert prereg["data"]["reuses_previously_evaluated_l2d"] is True
    assert prereg["data"]["independent_confirmation"] is False
    assert prereg["gate"]["fresh_confirmation_required_if_selected"] is True
    assert prereg["gate"]["history_admitted"] is False
    assert prereg["gate"]["value_learning_admitted"] is False
    assert prereg["gate"]["online_policy_admitted"] is False
    assert prereg["online_wine"] is False
    assert prereg["fits_policy"] is False


def test_l2i_frozen_architecture_is_bounded_below_gate() -> None:
    model = PrimitiveSetHazard(
        token_width=13,
        token_hidden=32,
        head_hidden_1=64,
        head_hidden_2=32,
        token_cap=48,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == 8929
    assert 8929 <= load_prereg(PREREGISTRATION)["gate"][
        "maximum_parameter_count"
    ]


def test_l2i_preregistration_is_canonical_json_data() -> None:
    observed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert observed["schema"] == (
        "th06-rl-l2i-observed-primitive-set-hazard-prereg-v1"
    )
    assert observed["non_goals"]
    assert "no history" in observed["non_goals"][2]
    for key in (
        "primitive_dataset_module",
        "primitive_hazard_module",
        "primitive_hazard_runner",
    ):
        assert len(observed["sha256_bindings"][key]) == 64
