from __future__ import annotations

from types import SimpleNamespace

from th06_rl.hazard_representation import (
    HAZARD_SUMMARY_FEATURE_NAMES,
    HAZARD_PRIMITIVE_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    MAX_HAZARD_PRIMITIVES,
    make_history_observation,
    project_hazard_primitives,
    project_history_features,
    summarize_hazard_primitives,
)


def test_fixed_hazard_summary_is_bounded_width_and_empty_aware() -> None:
    empty = summarize_hazard_primitives(())
    assert len(empty) == len(HAZARD_SUMMARY_FEATURE_NAMES)
    assert empty[-1] == 1.0
    row = tuple(float(index) for index in range(14))
    populated = summarize_hazard_primitives((row, row))
    assert populated[-1] == 0.0
    assert populated[-2] > 0.0


def _snapshot(*, bullets=(), lasers=(), enemies=(), frame=10):
    return SimpleNamespace(
        frame=frame,
        stage=3,
        x=192.0,
        y=400.0,
        live_bullet_count=len(bullets),
        laser_count=len(lasers),
        bullets=bullets,
        lasers=lasers,
        enemies=enemies,
    )


def _bullet(x: float):
    return SimpleNamespace(
        x=x,
        y=380.0,
        vx=1.0,
        vy=2.0,
        half_width=2.0,
        half_height=3.0,
        timer=12,
    )


def test_hazard_projection_is_permutation_invariant_and_bounded() -> None:
    bullets = tuple(_bullet(float(index)) for index in range(300))
    forward = project_hazard_primitives(_snapshot(bullets=bullets))
    reverse = project_hazard_primitives(_snapshot(bullets=tuple(reversed(bullets))))

    assert forward == reverse
    assert len(forward) == MAX_HAZARD_PRIMITIVES
    assert all(len(row) == len(HAZARD_PRIMITIVE_FEATURE_NAMES) for row in forward)


def test_hazard_projection_uses_only_generic_observed_geometry() -> None:
    laser = SimpleNamespace(
        x=100.0,
        y=120.0,
        angle=0.0,
        start_offset=10.0,
        end_offset=30.0,
        width=8.0,
        speed=2.0,
        timer=5,
        duration=20,
        flags=0xDEAD,
        slot=99,
    )
    row = project_hazard_primitives(_snapshot(lasers=(laser,)))[0]

    assert row[-3:] == (0.0, 1.0, 0.0)
    assert row[10] == 1.0
    assert "flags" not in HAZARD_PRIMITIVE_FEATURE_NAMES
    assert "slot" not in HAZARD_PRIMITIVE_FEATURE_NAMES


def test_four_observation_history_is_factual_and_fixed_width() -> None:
    previous = tuple(
        make_history_observation(_snapshot(frame=frame), "left")
        for frame in (7, 8, 9)
    )
    current = make_history_observation(_snapshot(frame=10), "stay")
    features = project_history_features(current, previous)

    assert tuple(name for name, _value in features) == HISTORY_FEATURE_NAMES
    assert len(features) == 4 * 9
    assert dict(features)["lag0:age_frames_log"] == 0.0
    assert dict(features)["lag3:age_frames_log"] > 0.0
