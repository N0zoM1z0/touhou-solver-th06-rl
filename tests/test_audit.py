from dataclasses import replace
from types import SimpleNamespace

from scripts.audit_run import (
    _collision_evidence,
    _measure_player_successor,
    _new_player_successor_parity,
    audit,
)
from th06_rl.corpus import CorpusRecorder, FrameEvidence, RunMetadata
from th06_rl.retail import native
from th06_rl.retail.model import Bullet, PlayerAttackState
from th06_rl.th06.control_capture import (
    CONTROL_CAPTURE_TIER,
    OFFLINE_FACT_SCHEMA,
    SOURCE_RECORD_SCHEMA,
    ControlSnapshot,
)


def _online_contract() -> dict[str, object]:
    return {
        "algorithm": "observed-shield4-paused-publication-v1",
        "shield_contract": "observed-hazard-kinematics-v1",
        "publication_epoch": "coherent-root-process-suspended-v1",
        "factual_state_schema": OFFLINE_FACT_SCHEMA,
        "shield_horizon": 4,
        "predicts_future_births": False,
    }


def _snapshot(
    frame: int,
    *,
    player_state: int = 0,
    x: float = 192.0,
    input_mask: int = 0x01,
    bullets=(),
) -> ControlSnapshot:
    return ControlSnapshot(
        capture_tier=CONTROL_CAPTURE_TIER,
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
        input_mask=input_mask,
        bullets=tuple(bullets),
        live_bullet_count=len(bullets),
        raw_bullet_tails=b"",
        bullet_sprite_dimensions=(),
        bullets_are_reachable_subset=False,
        laser_count=0,
        in_menu=False,
        time_stopped=False,
        replay_or_demo=False,
        lasers=(),
        enemies=(),
        difficulty=3,
        character=0,
        shot_type=0,
        bomb_active=False,
        spell_active=False,
        rank=0,
        subrank=0,
        max_rank=32,
        min_rank=0,
        rng_seed=0,
        rng_generation=0,
        current_power=0,
        lives_remaining=2,
        source_context="observed-world",
        boss_life=None,
        timeline_time=0,
        timeline_time_float=0.0,
        capture_attempts=1,
        bullet_read_retries=0,
        raw_enemy_manager_tail=bytes(
            native.ENEMY_MANAGER_SIZE
            - native.ENEMY_ARRAY_OFFSET
            - native.ENEMY_COUNT * native.ENEMY_STRIDE
        ),
        source_record_schema=SOURCE_RECORD_SCHEMA,
        factual_state_schema=OFFLINE_FACT_SCHEMA,
        player_attack=PlayerAttackState(
            (), 0.0, 0.0, 0, False,
            0, 0, 0.0, 0, 0, 0.0,
            ((0.0, 0.0), (0.0, 0.0)), 0, False, False,
        ),
        item_active_upper_bound=0,
    )


def _evidence() -> FrameEvidence:
    return FrameEvidence(
        phase_id="stage:4",
        current_action="stay",
        shield_actions=(("stay", None, 192.0, 400.0),),
        baseline_action="stay",
        locally_admissible_actions=("stay",),
        proposed_action="stay",
        published_action="stay",
        behavior_probability=1.0,
        behavior_probabilities=(("stay", 1.0),),
        policy_id="runtime-smoke-reactive-baseline-v1",
        policy_generation=1,
        policy_sha256="test",
        capture_ms=1.0,
        solve_ms=0.1,
        reason="ok",
    )


def _record_episode(
    tmp_path,
    before: ControlSnapshot,
    after: ControlSnapshot,
    *,
    hits: int,
):
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata(
            "test",
            "exe",
            "native",
            "test",
            3,
            0,
            0,
            4,
            _online_contract(),
            expected_stages=(4,),
        ),
    )
    evidence = _evidence()
    recorder.record(before, evidence)
    recorder.record(after, evidence)
    return recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "physical_hits": hits,
        "policy_failures": 0,
        "policy_last_error": None,
    })


def test_complete_physical_episode_passes_with_only_generic_streams(tmp_path) -> None:
    run_dir = _record_episode(tmp_path, _snapshot(10), _snapshot(11), hits=0)

    report = audit(run_dir)

    assert report["integrity_errors"] == []
    assert report["episode_dataset_admission"] == {
        "checked_frames": 2,
        "checked_transitions": 1,
        "passes": True,
        "error": None,
    }


def test_new_overlapping_bullet_is_unknown_dynamics_outcome(tmp_path) -> None:
    bullet = Bullet(
        x=192.0,
        y=400.0,
        vx=0.0,
        vy=0.0,
        half_width=2.0,
        half_height=2.0,
        state=1,
        slot=7,
    )
    run_dir = _record_episode(
        tmp_path,
        _snapshot(10),
        _snapshot(11, player_state=2, bullets=(bullet,)),
        hits=1,
    )

    report = audit(run_dir)

    assert report["integrity_errors"] == []
    assert report["physical_hits"] == 1
    assert report["hit_classifications"] == {"future-unobserved-hazard": 1}


def test_online_contract_drift_is_an_infrastructure_error(tmp_path) -> None:
    run_dir = _record_episode(tmp_path, _snapshot(10), _snapshot(11), hits=0)
    run_path = run_dir / "run.json"
    import json

    run = json.loads(run_path.read_text())
    run["metadata"]["online_contract"]["predicts_future_births"] = True
    run_path.write_text(json.dumps(run), encoding="utf-8")

    report = audit(run_dir)

    assert "observed-shield-contract-invalid" in report["integrity_errors"]


def test_player_successor_uses_witnessed_next_root_input() -> None:
    before = SimpleNamespace(
        frame=100,
        stage=1,
        player_state=0,
        x=192.0,
        y=400.0,
        normal_speed=4.0,
        focus_speed=2.0,
        normal_diagonal_speed=2.8,
        focus_diagonal_speed=1.4,
    )
    after = SimpleNamespace(
        frame=101,
        stage=1,
        x=192.0,
        y=398.0,
        input_mask=0x14,
        time_stopped=False,
    )
    parity = _new_player_successor_parity()

    _measure_player_successor(before, after, 9, parity)

    assert parity["checked_links"] == 1
    assert parity["bit_exact_links"] == 1
    assert parity["mismatches"] == 0


def test_collision_evidence_does_not_relabel_future_birth_as_observed() -> None:
    before = _snapshot(10)
    bullet = Bullet(
        x=192.0,
        y=400.0,
        vx=0.0,
        vy=0.0,
        half_width=2.0,
        half_height=2.0,
        state=1,
        slot=9,
    )
    after = replace(_snapshot(11), bullets=(bullet,))

    evidence = _collision_evidence(before, after, 1)

    assert evidence["new_after_overlapping_bullet_slots"] == [9]
    assert evidence["projected_observed_bullet_slots"] == []
