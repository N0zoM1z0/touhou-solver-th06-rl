from __future__ import annotations

from scripts.train_headless_teacher import (
    Decision,
    apply_counterfactual_labels,
    candidate_features,
    candidate_sample_weight,
    generic_choice,
)
from th06_rl.headless_corpus import source_context_id


def decision() -> Decision:
    return Decision(
        run="run",
        seed=7,
        sequence=0,
        source_context="timeline:440/0/1",
        state={
            "player_x": 192.0,
            "player_y": 384.0,
            "previous_action": "stay",
            "boundary_reserve": 48.0,
            "game_frame": 10,
            "lives": 2,
            "power": 64,
            "rank": 32,
            "graze": 0,
            "bullet_count": 2,
            "laser_count": 0,
            "enemy_count": 1,
            "boss_count": 0,
        },
        legal_actions=("stay", "left"),
        candidates=(
            {
                "action": "stay",
                "min_clearance": None,
                "final_x": 192.0,
                "final_y": 384.0,
                "final_boundary_reserve": 48.0,
            },
            {
                "action": "left",
                "min_clearance": 10.0,
                "final_x": 184.0,
                "final_y": 384.0,
                "final_boundary_reserve": 48.0,
            },
        ),
        teacher_action="left",
        selected_action="left",
    )


def test_teacher_features_exclude_seed_rng_and_supervision_leakage() -> None:
    value = candidate_features(decision(), decision().candidates[1])

    assert value["action"] == "left"
    assert "seed" not in value
    assert not any("rng" in name for name in value)
    assert not any("teacher" in name for name in value)
    assert not any("selected" in name for name in value)


def test_generic_choice_prefers_unbounded_clearance() -> None:
    assert generic_choice(decision()) == "stay"


def test_ranker_context_contract_uses_derived_identity_not_raw_json() -> None:
    observation = {
        "enemies": [],
        "source_context": {
            "timeline_time": 100,
            "next": {"time": 440, "opcode": 0, "arg0": 1},
        },
    }

    assert source_context_id(observation) == decision().source_context


def test_failure_weighting_only_emphasizes_disagreeing_corrective_pair() -> None:
    failed = Decision(
        **{
            **decision().__dict__,
            "selected_action": "stay",
            "authority_failure_distance": 1,
        }
    )

    teacher_weight = candidate_sample_weight(
        failed,
        failed.candidates[1],
        failure_horizon=120,
        failure_weight=8.0,
    )
    failed_behavior_weight = candidate_sample_weight(
        failed,
        failed.candidates[0],
        failure_horizon=120,
        failure_weight=8.0,
    )

    assert teacher_weight == 9.0
    assert failed_behavior_weight == 9.0


def test_failure_weighting_does_not_invent_a_counterfactual_on_agreement() -> None:
    agreed = Decision(
        **{
            **decision().__dict__,
            "authority_failure_distance": 1,
        }
    )

    assert candidate_sample_weight(
        agreed,
        agreed.candidates[1],
        failure_horizon=120,
        failure_weight=8.0,
    ) == 1.0


def test_dynamic_counterfactual_can_override_local_teacher_label(tmp_path) -> None:
    current = decision()
    current = Decision(**{**current.__dict__, "observation_sha256": "abc"})
    label = tmp_path / "label.json"
    label.write_text(
        __import__("json").dumps({
            "schema": "th06-rl-headless-cow-counterfactual-v1",
            "scope": {"stage": 6},
            "input_source": {"commit": "source"},
            "runtime_source": {"commit": "runtime", "clean": True},
            "checkpoints": [{"observation_sha256": "abc", "best_actions": ["stay"]}],
        }),
        encoding="utf-8",
    )

    updated, report = apply_counterfactual_labels(
        [current],
        {"scope": {"stage": 6}, "source": {"commit": "source"}},
        [label],
    )

    assert updated[0].teacher_action == "stay"
    assert updated[0].counterfactual_original_action == "left"
    assert report["changed_local_teacher_labels"] == 1
    weight = candidate_sample_weight(
        updated[0],
        updated[0].candidates[0],
        failure_horizon=0,
        failure_weight=0.0,
        counterfactual_weight=16.0,
    )
    assert weight == 17.0
