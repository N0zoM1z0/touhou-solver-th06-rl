from __future__ import annotations

from th06_rl.conservative_learning import (
    FactualStep,
    NStepCost,
    fit_conservative_fqi,
)
from th06_rl.learning_features import tree_feature_names
from th06_rl.offline import ACTION_NAMES
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)


def _episodes(names: tuple[str, ...]) -> list[NStepCost]:
    width = len(tree_feature_names(OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES))
    samples = []
    for episode_index, episode in enumerate(names):
        for action_index, action in enumerate(ACTION_NAMES):
            vector = [0.0] * width
            vector[0] = episode_index / max(1, len(names) - 1)
            vector[len(OBSERVATION_FEATURE_NAMES)] = action_index / len(ACTION_NAMES)
            state = FactualStep(
                episode_id=episode,
                raw_index=action_index,
                sequence=action_index,
                action=action,
                baseline_action="stay",
                behavior_probability=0.1,
                vector=tuple(vector),
                legal_actions=("stay", action) if action != "stay" else ("stay",),
                candidate_vectors=(tuple(vector),) if action == "stay" else (
                    tuple([0.0] * width), tuple(vector)
                ),
            )
            samples.append(NStepCost(
                state=state,
                observed_hit_cost=float(action_index == episode_index),
                next_state=None,
            ))
    return samples


def test_conservative_fit_is_grouped_nonlinear_and_exportable() -> None:
    state = fit_conservative_fqi(
        _episodes(("train-a", "train-b", "train-c")),
        _episodes(("validation-a", "validation-b")),
        ensemble_members=3,
        bellman_iterations=1,
        trees_per_iteration=4,
        propensity_clip=20.0,
        prototypes_per_action=1,
        support_quantile=0.99,
        uncertainty_scale=1.0,
        seed=7,
        threads=1,
        native_scorer_sha256="a" * 64,
    )
    assert state["authorization"]["fit_eligible"] is True
    assert len(state["models"]) == 3
    assert len(state["support"]["prototypes"]) == len(ACTION_NAMES)
    assert state["fit_report"]["train_groups"] == [
        "train-a", "train-b", "train-c"
    ]
    report = state["fit_report"]
    assert report["exported_target_sha256"] == report["iterations"][-1][
        "next_target_sha256"
    ]
    assert report["exported_target_iteration"] == 2
    assert report["exported_nominal_horizon_frames"] == 120
    assert "heldout_factual_cost_rmse" not in report
    assert "matched_horizon_validation_rmse" in report
