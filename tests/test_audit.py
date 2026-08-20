from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from th06_rl.corpus import CorpusRecorder, RunMetadata
from th06_rl.retail.model import Bullet, PlayerAttackState, Snapshot


_SPEC = importlib.util.spec_from_file_location(
    "th06_rl_audit_script",
    Path(__file__).resolve().parents[1] / "scripts/audit_run.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)
audit = _AUDIT.audit


def _source_anchor(stage: int) -> Snapshot:
    return Snapshot(
        frame=stage,
        stage=stage,
        player_state=0,
        x=192.0,
        y=400.0,
        half_width=1.25,
        half_height=1.25,
        normal_speed=4.0,
        focus_speed=2.0,
        normal_diagonal_speed=2.8,
        focus_diagonal_speed=1.4,
        frame_multiplier=1.0,
        input_mask=1,
        bullets=(),
        laser_count=0,
        in_menu=False,
        time_stopped=False,
        replay_or_demo=False,
        difficulty=3,
        character=0,
        player_attack=PlayerAttackState(
            shots=(),
            last_enemy_hit_x=0.0,
            last_enemy_hit_y=0.0,
            orb_state=0,
            is_focus=True,
            focus_timer_previous=0,
            focus_timer=0,
            focus_timer_float=0.0,
            fire_timer_previous=0,
            fire_timer=0,
            fire_timer_float=0.0,
            orb_positions=((0.0, 0.0), (0.0, 0.0)),
            shot_type=0,
            bomb_active=False,
            spell_active=False,
        ),
    )


def test_empty_complete_stage_audit_is_structurally_stable(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata(
            "test", "exe", "native", "test", 3, 0, 0, 4,
            {
                "source_commitment": "source-complete-hard-v1",
                "factual_state_schema": "th06-1.02h-offline-facts-v2",
            },
        ),
    )
    run_dir = recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "physical_hits": 0,
    })

    report = audit(run_dir)

    assert report["stage_completed"] is True
    assert report["physical_hits"] == 0
    assert report["integrity_errors"] == []
    assert report["infra_stable_for_learning"] is True


def test_route_audit_accepts_only_declared_complete_stage_coverage(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata(
            "test", "exe", "native", "test", 3, 0, 0, 1,
            {
                "source_commitment": "source-complete-hard-v1",
                "factual_state_schema": "th06-1.02h-offline-facts-v2",
            },
            episode_unit="route",
            expected_stages=(1, 2, 3, 4, 5, 6),
        ),
    )
    for stage in range(1, 7):
        recorder.record_anchor(
            _source_anchor(stage),
            phase_id="source:test",
            reason="stage-root",
            control_snapshot_ref=None,
        )
    run_dir = recorder.close({
        "termination_reason": "route-complete",
        "stage_completed": True,
        "physical_hits": 0,
    })
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["summary"]["phases"] = [
        {"scope": f"3/0/0/{stage}/source:test"}
        for stage in range(1, 7)
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit(run_dir)

    assert report["integrity_errors"] == []
    assert report["scope"]["episode_unit"] == "route"
    assert report["scope"]["observed_stages"] == [1, 2, 3, 4, 5, 6]
    assert report["source_anchor_coverage"] == {
        "anchored_stages": [1, 2, 3, 4, 5, 6],
        "missing_observed_stages": [],
    }


def test_route_audit_rejects_stage_without_source_anchor(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata(
            "test", "exe", "native", "test", 3, 0, 0, 1,
            {
                "source_commitment": "source-complete-hard-v1",
                "factual_state_schema": "th06-1.02h-offline-facts-v2",
            },
            episode_unit="route",
            expected_stages=(1, 2),
        ),
    )
    recorder.record_anchor(
        _source_anchor(1),
        phase_id="source:test",
        reason="stage-root",
        control_snapshot_ref=None,
    )
    run_dir = recorder.close({
        "termination_reason": "route-complete",
        "stage_completed": True,
        "physical_hits": 0,
    })
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["summary"]["phases"] = [
        {"scope": f"3/0/0/{stage}/source:test"} for stage in (1, 2)
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit(run_dir)

    assert "source-anchor-stage-coverage" in report["integrity_errors"]
    assert report["source_anchor_coverage"]["missing_observed_stages"] == [2]


def test_audit_rejects_physical_hit_count_disagreement(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata(
            "test", "exe", "native", "test", 3, 0, 0, 4,
            {
                "source_commitment": "source-complete-hard-v1",
                "factual_state_schema": "th06-1.02h-offline-facts-v2",
            },
        ),
    )
    run_dir = recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "physical_hits": 1,
    })

    assert "physical-hit-count-mismatch" in audit(run_dir)["integrity_errors"]


def test_collision_evidence_identifies_new_enemy_body() -> None:
    common = {
        "x": 193.0,
        "y": 121.0,
        "half_width": 1.25,
        "half_height": 1.25,
        "bullets": (),
        "lasers": (),
    }
    before = SimpleNamespace(**common, enemies=())
    after = SimpleNamespace(
        **common,
        enemies=(SimpleNamespace(
            x=192.0,
            y=128.0,
            half_width=16.0,
            half_height=18.0,
        ),),
    )

    evidence = _AUDIT._collision_evidence(before, after, 1)

    assert evidence["after_overlapping_enemy_indices"] == [0]
    assert evidence["new_after_overlapping_enemy_indices"] == [0]


def test_successor_aabb_comparison_is_one_sided_and_float_tolerant() -> None:
    actual = (10.0, 20.0, 14.0, 24.0)

    assert _AUDIT._aabb_contains((9.9995, 19.9995, 13.9995, 23.9995), actual)
    assert not _AUDIT._aabb_contains((10.01, 20.0, 14.0, 24.0), actual)


def test_post_update_laser_recovers_source_midpoint_bug() -> None:
    laser = SimpleNamespace(
        x=100.0,
        y=120.0,
        angle=0.25,
        start_offset=8.0,
        end_offset=108.0,
        width=16.0,
        start_time=10,
        hitbox_start_time=3,
        duration=30,
        despawn_duration=10,
        hitbox_end_delay=5,
        timer=6,
        timer_float=6.0,
        flags=0,
        state=0,
    )

    hazards, reason = _AUDIT._retained_post_update_laser_hazards(laser)

    assert reason == "checked"
    assert len(hazards) == 1
    # The source used timer 5 before Tick: (5 * width / startTime) / 2.
    assert hazards[0].size_x == 4.0
    assert hazards[0].size_y == 8.0


def test_post_update_laser_unions_zero_delay_natural_predecessor() -> None:
    laser = SimpleNamespace(
        x=100.0,
        y=120.0,
        angle=0.25,
        start_offset=8.0,
        end_offset=108.0,
        width=16.0,
        start_time=10,
        hitbox_start_time=3,
        duration=30,
        despawn_duration=10,
        hitbox_end_delay=0,
        timer=1,
        timer_float=1.0,
        flags=0,
        state=2,
    )

    hazards, reason = _AUDIT._retained_post_update_laser_hazards(laser)

    assert reason == "checked"
    assert len(hazards) == 1
    assert hazards[0].size_x == 100.0
    assert hazards[0].size_y == 8.0


def test_successor_laser_accepts_exact_or_conservative_aabb_coverage() -> None:
    actual = _AUDIT.LaserHazard(100.0, 120.0, 0.4, 60.0, 80.0, 8.0)
    exact = _AUDIT.LaserHazard(100.0, 120.0, 0.4, 60.0, 80.0, 8.0)
    enclosing = _AUDIT._laser_aabb(actual)

    assert _AUDIT._laser_is_covered(actual, (), (exact,))
    assert _AUDIT._laser_is_covered(actual, (enclosing,), ())
    assert not _AUDIT._laser_is_covered(actual, (), ())


def test_numeric_successor_parity_is_bit_exact_for_linear_bullets() -> None:
    bullet = Bullet(
        x=10.0,
        y=20.0,
        vx=0.25,
        vy=-0.5,
        half_width=2.0,
        half_height=2.0,
        state=1,
        timer=7,
        slot=3,
        sprite=4,
    )
    correct = Bullet(
        **{
            **bullet.__dict__,
            "x": _AUDIT._f32(bullet.x + bullet.vx),
            "y": _AUDIT._f32(bullet.y + bullet.vy),
            "timer": 8,
        }
    )
    before = SimpleNamespace(frame=100, bullets=(bullet,))
    after = SimpleNamespace(frame=101, bullets=(correct,))
    parity = _AUDIT._new_bullet_successor_parity()

    _AUDIT._measure_bullet_successors(before, after, 9, parity)
    result = _AUDIT._finish_bullet_successor_parity(parity)

    assert result["linear_exact_checked"] == 1
    assert result["exact_mismatches"] == 0

    wrong = Bullet(**{**correct.__dict__, "x": correct.x + 0.001})
    parity = _AUDIT._new_bullet_successor_parity()
    _AUDIT._measure_bullet_successors(
        before,
        SimpleNamespace(frame=101, bullets=(wrong,)),
        9,
        parity,
    )
    result = _AUDIT._finish_bullet_successor_parity(parity)
    assert result["exact_mismatches"] == 1
    assert result["counterexamples"][0]["category"] == "linear-exact"


def test_numeric_successor_parity_uses_declared_global_mutation_union() -> None:
    bullet = Bullet(
        x=100.0,
        y=120.0,
        vx=0.5,
        vy=-0.25,
        half_width=2.0,
        half_height=2.0,
        state=1,
        timer=20,
        slot=7,
        sprite=2,
    )
    before = SimpleNamespace(frame=200, bullets=(bullet,))

    stopped = Bullet(**{
        **bullet.__dict__,
        "vx": 0.0,
        "vy": 0.0,
        "timer": 21,
    })
    parity = _AUDIT._new_bullet_successor_parity()
    _AUDIT._measure_bullet_successors(
        before,
        SimpleNamespace(frame=201, bullets=(stopped,)),
        10,
        parity,
        {"source_bullet_stop_frames": [0]},
    )
    result = _AUDIT._finish_bullet_successor_parity(parity)
    assert result["global_stop_union_checked"] == 1
    assert result["global_mutation_union_violations"] == 0
    assert result["linear_exact_checked"] == 0

    predicted_x = _AUDIT._f32(bullet.x + bullet.vx)
    predicted_y = _AUDIT._f32(bullet.y + bullet.vy)
    released = Bullet(**{
        **bullet.__dict__,
        "x": _AUDIT._f32(predicted_x + 0.009),
        "y": _AUDIT._f32(predicted_y - 0.008),
        "vx": _AUDIT._f32(bullet.vx + 0.009),
        "vy": _AUDIT._f32(bullet.vy - 0.008),
        "timer": 1,
    })
    parity = _AUDIT._new_bullet_successor_parity()
    _AUDIT._measure_bullet_successors(
        before,
        SimpleNamespace(frame=201, bullets=(released,)),
        11,
        parity,
        {"source_bullet_release_frames": [0]},
    )
    result = _AUDIT._finish_bullet_successor_parity(parity)
    assert result["global_release_union_checked"] == 1
    assert result["global_mutation_union_violations"] == 0

    outside = Bullet(**{**released.__dict__, "x": predicted_x + 0.1})
    parity = _AUDIT._new_bullet_successor_parity()
    _AUDIT._measure_bullet_successors(
        before,
        SimpleNamespace(frame=201, bullets=(outside,)),
        12,
        parity,
        {"source_bullet_release_frames": [0]},
    )
    result = _AUDIT._finish_bullet_successor_parity(parity)
    assert result["global_mutation_union_violations"] == 1
    assert result["counterexamples"][0]["category"] == (
        "global-release-or-ordinary-union"
    )
