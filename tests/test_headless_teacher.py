from __future__ import annotations

from scripts.train_headless_teacher import Decision, candidate_features, generic_choice
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
