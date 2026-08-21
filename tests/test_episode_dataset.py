from __future__ import annotations

from dataclasses import replace
import json

import pytest

from th06_rl.corpus import CorpusRecorder, FrameEvidence, RunMetadata
from th06_rl.bc_features import (
    FEATURE_NAMES,
    features_from_policy_context,
    features_from_portable_root,
)
from th06_rl.episode_dataset import (
    EpisodeDatasetError,
    iter_decision_epochs,
    validate_decision_epochs,
)
from th06_rl.th06.control_capture import ControlSnapshot
from th06_rl.policy_api import PolicyContext


def _snapshot(
    frame: int,
    *,
    player_state: int = 0,
    lives: int = 2,
    x: float = 192.0,
    current_power: int = 64,
    bullet_count: int = 10,
    in_menu: bool = False,
) -> ControlSnapshot:
    return ControlSnapshot(
        capture_tier="control-v2",
        frame=frame,
        stage=4,
        player_state=player_state,
        x=x,
        y=400.0,
        half_width=1.5,
        half_height=1.5,
        normal_speed=4.0,
        focus_speed=2.0,
        normal_diagonal_speed=2.8,
        focus_diagonal_speed=1.4,
        frame_multiplier=1.0,
        input_mask=1,
        bullets=(),
        live_bullet_count=bullet_count,
        raw_bullet_tails=b"",
        bullet_sprite_dimensions=(),
        bullets_are_reachable_subset=True,
        laser_count=0,
        in_menu=in_menu,
        time_stopped=False,
        replay_or_demo=False,
        lasers=(),
        enemies=(),
        difficulty=3,
        character=0,
        shot_type=0,
        bomb_active=False,
        spell_active=False,
        rank=10,
        subrank=0,
        max_rank=32,
        min_rank=0,
        rng_seed=123,
        rng_generation=frame,
        current_power=current_power,
        lives_remaining=lives,
        source_context="forbidden-test-source-context",
        boss_life=None,
        timeline_time=frame,
        timeline_time_float=float(frame),
        capture_attempts=1,
        bullet_read_retries=0,
    )


def _decision(
    reason: str,
    *,
    current: str = "stay",
    published: str | None = None,
    legal: tuple[str, ...] = (),
) -> FrameEvidence:
    evaluations = tuple(
        (action, 10.0 + index, 190.0 + index, 399.0 + index)
        for index, action in enumerate(legal)
    )
    probabilities = (
        tuple((action, float(action == published)) for action in legal)
        if published is not None
        else ()
    )
    return FrameEvidence(
        phase_id="stage:4",
        current_action=current,
        shield_actions=evaluations,
        baseline_action=legal[0] if legal else None,
        locally_admissible_actions=legal,
        proposed_action=published,
        published_action=published,
        behavior_probability=1.0,
        behavior_probabilities=probabilities,
        policy_id="fixture-policy-v1",
        policy_generation=1,
        policy_sha256="fixture",
        capture_ms=1.0,
        solve_ms=0.1,
        reason=reason,
        shield_collision_margin=0.35,
    )


def _episode(tmp_path):
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata("test", "exe", "native", "test", 3, 0, 0, 4, {}),
    )
    rows = (
        (_snapshot(0), _decision("ok", published="left", legal=("left", "right"))),
        (
            _snapshot(1),
            _decision(
                "input-lease",
                current="left",
                published="left",
                legal=("left",),
            ),
        ),
        (
            _snapshot(2, x=190.0),
            _decision(
                "ok",
                current="left",
                published="right",
                legal=("left", "right"),
            ),
        ),
        (_snapshot(3, player_state=2, lives=1), _decision("physical-hit", current="right")),
        (_snapshot(4, player_state=2, lives=1), _decision("player-not-active")),
        (
            _snapshot(5, player_state=0, lives=1),
            _decision("ok", published="stay", legal=("stay", "left")),
        ),
        (_snapshot(6, player_state=0, lives=1, in_menu=True), _decision("passive")),
    )
    for snapshot, decision in rows:
        recorder.record(snapshot, decision)
    return recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "controller_exit_code": 0,
        "physical_hits": 1,
    })


def test_decision_epochs_collapse_input_leases_and_conserve_hits(tmp_path) -> None:
    run_dir = _episode(tmp_path)

    epochs = list(iter_decision_epochs(run_dir))

    assert [epoch.transition_sequences for epoch in epochs] == [
        (0, 1),
        (2, 3, 4),
        (5,),
    ]
    assert [epoch.published_action for epoch in epochs] == ["left", "right", "stay"]
    assert {epoch.policy_id for epoch in epochs} == {"fixture-policy-v1"}
    assert epochs[0].commanded_actions == ("left", "left")
    assert epochs[0].executed_actions == ("left", "left")
    assert [epoch.hit_cost for epoch in epochs] == [0, 1, 0]
    assert [epoch.elapsed_game_frames for epoch in epochs] == [2, 3, 1]
    assert [epoch.terminal for epoch in epochs] == [False, False, True]
    assert all(epoch.learning_eligible for epoch in epochs)
    assert epochs[1].next_observation is not None
    assert epochs[1].next_observation.player_x == 192.0
    assert "forbidden-test-source-context" not in repr(epochs)
    assert validate_decision_epochs(run_dir) == {
        "decision_epochs": 3,
        "learning_eligible_decision_epochs": 3,
        "excluded_decision_epochs": 0,
        "physical_hits": 1,
        "learning_eligible_physical_hits": 1,
        "excluded_physical_hits": 0,
    }


def test_offline_and_online_scalar_features_are_bit_exact(tmp_path) -> None:
    epoch = next(iter_decision_epochs(_episode(tmp_path)))
    root = epoch.observation
    context = PolicyContext(
        baseline_action="right",
        locally_admissible_actions=root.locally_admissible_actions,
        player_x=root.player_x,
        player_y=root.player_y,
        power=root.power,
        bullet_count=root.bullet_count,
        laser_count=root.laser_count,
        shield_action_count=len(root.locally_admissible_actions),
        current_action=root.current_action,
        shield_admissible_actions=root.locally_admissible_actions,
        shield_action_evaluations=root.shield_action_evaluations,
    )

    offline = features_from_portable_root(root)
    online = features_from_policy_context(context)

    assert len(offline) == len(FEATURE_NAMES)
    assert offline == online
    assert online == features_from_policy_context(replace(context, baseline_action="left"))


def test_unexecuted_publication_has_no_action_conditioned_successor(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata("test", "exe", "native", "test", 3, 0, 0, 4, {}),
    )
    recorder.record(
        _snapshot(0),
        _decision("ok", current="stay", published="left", legal=("left", "stay")),
    )
    recorder.record(_snapshot(1), _decision("passive", current="stay"))
    run_dir = recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "physical_hits": 0,
    })

    epoch = next(iter_decision_epochs(run_dir))

    assert epoch.learning_eligible is False
    assert "published-action-not-executed" in epoch.exclusion_reasons
    assert epoch.executed_actions == ("stay",)


def test_decision_view_rejects_declared_hit_mismatch(tmp_path) -> None:
    run_dir = _episode(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_outcome"]["physical_hits"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EpisodeDatasetError, match="conserve"):
        list(iter_decision_epochs(run_dir))


def test_observation_gap_is_retained_but_not_trainable(tmp_path) -> None:
    run_dir = _episode(tmp_path)
    # The fixture itself remains immutable; construct a separate gap episode.
    recorder = CorpusRecorder(
        tmp_path / "gap",
        RunMetadata("test", "exe", "native", "test", 3, 0, 0, 4, {}),
    )
    recorder.record(
        _snapshot(10),
        _decision("ok", published="stay", legal=("stay",)),
    )
    recorder.record(
        _snapshot(12),
        replace(_decision("passive", current="stay"), observation_gap=2),
    )
    gap_run = recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "physical_hits": 0,
    })

    epoch = next(iter_decision_epochs(gap_run))

    assert epoch.elapsed_game_frames == 2
    assert epoch.learning_eligible is False
    assert "invalid-factual-link" in epoch.exclusion_reasons
