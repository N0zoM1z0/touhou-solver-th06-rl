from __future__ import annotations

import hashlib
import subprocess

from scripts.train_headless_teacher import (
    Decision,
    Encoder,
    apply_counterfactual_labels,
    candidate_features,
    candidate_sample_weight,
    generic_choice,
    repository_commit,
)
from th06_rl.headless_corpus import (
    HAZARD_FEATURE_NAMES,
    SOURCE_CONTEXT_FEATURE_NAMES,
    compact_hazard_sector_features,
    source_context_id,
)
from scripts.collect_headless_dagger import borda_consensus, source_compatible
from scripts.collect_headless_dagger import _benchmark_ranker_decision
from th06_rl.native import ACTIONS, NativeCertifiedAction


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
    assert all(name in value for name in HAZARD_FEATURE_NAMES)
    assert all(name in value for name in SOURCE_CONTEXT_FEATURE_NAMES)


def test_repository_commit_is_independent_of_caller_working_directory(
    tmp_path,
    monkeypatch,
) -> None:
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)

    assert repository_commit() == expected


def test_hazard_sector_features_retain_position_and_approach_without_authority() -> None:
    observation = {
        "player": {
            "x": 100.0,
            "y": 100.0,
            "half_width": 1.25,
            "half_height": 1.25,
        },
        "bullets": [
            {
                "x": 140.0,
                "y": 100.0,
                "vx": -2.0,
                "vy": 0.0,
                "half_width": 2.5,
                "half_height": 2.5,
            }
        ],
    }

    features = compact_hazard_sector_features(observation)

    assert sum(features[name] for name in features if name.endswith("near_count")) == 1.0
    assert sum(
        features[name] for name in features if name.endswith("approaching_count")
    ) == 1.0
    assert min(
        features[name] for name in features if name.endswith("min_projected_surface")
    ) == 0.0


def test_encoder_can_replay_an_older_prefix_feature_schema() -> None:
    names = ("action", "previous_action", "source_context", "player_x")
    encoder = Encoder([decision()], feature_names=names)

    matrix = encoder.encode([candidate_features(decision(), decision().candidates[0])])

    assert matrix.shape == (1, len(names))


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
            "terminal_failure_distance": 1,
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
            "terminal_failure_distance": 1,
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
    assert report["files_used"] == [{
        "path": str(label.resolve()),
        "sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
    }]
    assert len(report["file_set_sha256"]) == 64
    weight = candidate_sample_weight(
        updated[0],
        updated[0].candidates[0],
        failure_horizon=0,
        failure_weight=0.0,
        counterfactual_weight=16.0,
    )
    assert weight == 17.0


def test_survivable_counterfactual_target_keeps_all_full_horizon_actions(tmp_path) -> None:
    current = Decision(**{**decision().__dict__, "observation_sha256": "abc"})
    label = tmp_path / "survivable.json"
    label.write_text(
        __import__("json").dumps({
            "schema": "th06-rl-headless-cow-counterfactual-v1",
            "scope": {"stage": 6},
            "input_source": {"commit": "source"},
            "runtime_source": {"commit": "runtime", "clean": True},
            "checkpoints": [{
                "observation_sha256": "abc",
                "branch_frames": 180,
                "best_actions": ["stay"],
                "outcomes": [
                    {
                        "first_action": "stay",
                        "survival_ticks": 180,
                        "physical_deaths_delta": 0,
                        "termination_reason": "tick-limit",
                    },
                    {
                        "first_action": "left",
                        "survival_ticks": 180,
                        "physical_deaths_delta": 0,
                        "termination_reason": "tick-limit",
                    },
                ],
            }],
        }),
        encoding="utf-8",
    )

    updated, report = apply_counterfactual_labels(
        [current],
        {"scope": {"stage": 6}, "source": {"commit": "source"}},
        [label],
        target="survivable",
    )

    assert updated[0].teacher_action == "left"
    assert updated[0].counterfactual_acceptable_actions == ("stay", "left")
    assert updated[0].counterfactual_original_action is None
    assert report["mean_acceptable_actions"] == 2.0


def test_runtime_compatibility_requires_exact_clean_commit_and_binary() -> None:
    allowed = [{"commit": "new", "binary_sha256": "abc", "clean": True}]

    assert source_compatible(
        allowed,
        {"commit": "new", "binary_sha256": "abc", "clean": True},
    )
    assert not source_compatible(
        allowed,
        {"commit": "new", "binary_sha256": "different", "clean": True},
    )
    assert not source_compatible(
        allowed,
        {"commit": "new", "binary_sha256": "abc", "clean": False},
    )


def test_continuation_ranker_metadata_does_not_claim_a_teacher_label() -> None:
    certified = (
        NativeCertifiedAction(
            action=ACTIONS[0],
            min_clearance=10.0,
            final_x=192.0,
            final_y=384.0,
        ),
    )

    decision_metadata = _benchmark_ranker_decision(
        certified[0].action.name,
        certified,
    )

    assert decision_metadata.kind == "benchmark-ranker-only"
    assert decision_metadata.effort_horizon == 0
    assert decision_metadata.action == certified[0].action.name


def test_borda_consensus_rewards_cross_model_support_without_score_calibration() -> None:
    actions = ["left", "stay", "right"]

    selected = borda_consensus(actions, [
        [1000.0, 900.0, -500.0],
        [-20.0, 0.2, 0.1],
        [0.0, 0.8, 0.7],
    ])

    assert selected == "stay"
