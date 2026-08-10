from __future__ import annotations

from scripts.audit_headless_event_observation import event_summary, physical_observation


def test_physical_observation_removes_only_events() -> None:
    observation = {
        "tick": 7,
        "player": {"x": 1.0},
        "events": {"bullet_births": [], "laser_births": [], "hit": None},
    }
    assert physical_observation(observation) == {
        "tick": 7,
        "player": {"x": 1.0},
    }


def test_event_summary_retains_causal_hit_and_counts_births() -> None:
    observation = {
        "tick": 9,
        "events": {
            "bullet_births": [{"slot": 1}],
            "laser_births": [],
            "hit": {"kind": "enemy", "slot": 0},
        },
    }
    summary = event_summary(observation, offset=2)
    assert summary is not None
    assert summary["offset"] == 2
    assert summary["bullet_births"] == 1
    assert summary["laser_births"] == 0
    assert summary["hit"]["kind"] == "enemy"
