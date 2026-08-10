from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retail_source_replay import compare_retail_source_states


def _frame(sequence: int, frame: int, *, mask: int, x: float) -> dict[str, object]:
    return {
        "sequence": sequence,
        "snapshot": {
            "frame": frame,
            "difficulty": 3,
            "character": 0,
            "shot_type": 0,
            "stage": 6,
            "rng_seed": 10,
            "rng_generation": 2,
            "input_mask": mask,
            "x": x,
            "y": 384.0,
            "player_state": 3,
            "half_width": 1.25,
            "half_height": 1.25,
            "lives_remaining": 2,
            "current_power": 128,
            "rank": 16,
            "timeline_time": frame,
            "live_bullet_count": 0,
            "laser_count": 0,
        },
    }


def _source(frame: int, *, mask: int, x: float) -> dict[str, object]:
    return {
        "schema": "th06-headless-observation-v2",
        "tick": frame,
        "game_frame": frame,
        "scope": {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 6},
        "rng_seed": 10,
        "rng_generation": 2,
        "input": mask,
        "player": {
            "x": x,
            "y": 384.0,
            "state": 3,
            "half_width": 1.25,
            "half_height": 1.25,
            "focused": bool(mask & 0x04),
        },
        "lives": 2,
        "power": 128,
        "rank": 16,
        "source_context": {"timeline_time": frame},
        "bullets": [],
        "lasers": [],
    }


def test_retail_source_comparison_separates_numeric_and_input_drift(tmp_path: Path) -> None:
    frames = [
        _frame(0, 127, mask=0, x=192.0),
        _frame(1, 131, mask=0x11, x=193.41421508789062),
        _frame(2, 2805, mask=0, x=200.0),
    ]
    source = [
        _source(1, mask=0, x=192.0),
        _source(127, mask=0, x=192.0),
        _source(131, mask=0x11, x=193.414215),
        _source(2805, mask=1, x=200.0),
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in source), encoding="utf-8"
    )

    result = compare_retail_source_states(frames, trace)

    assert result["common_snapshots"] == 3
    assert result["missing_retail_snapshots"] == 0
    assert result["exact_shared_state"]["first_divergence"]["frame"] == 131
    assert result["categories"]["player_at_1e_6"]["equal"] is True
    assert result["categories"]["player_geometry_at_1e_6"]["equal"] is True
    assert result["categories"]["input"]["first_divergence"]["frame"] == 2805
    assert result["categories"]["rng"]["equal"] is True
    tolerance = {
        row["absolute_tolerance"]: row for row in result["tolerance_ladder"]
    }
    assert tolerance[1e-6]["first_divergence"]["frame"] == 2805


def test_retail_source_comparison_can_retain_an_early_source_terminal(tmp_path: Path) -> None:
    frames = [
        _frame(0, 127, mask=0, x=192.0),
        _frame(1, 131, mask=0x11, x=192.0),
    ]
    trace = tmp_path / "short.jsonl"
    trace.write_text(json.dumps(_source(127, mask=0, x=192.0)) + "\n", encoding="utf-8")

    result = compare_retail_source_states(
        frames, trace, require_full_coverage=False
    )

    assert result["common_snapshots"] == 1
    assert result["missing_retail_snapshots"] == 1
    assert result["first_missing_retail_frame"] == 131


def test_retail_source_comparison_audits_dialogue_input_samples(tmp_path: Path) -> None:
    frames = [
        _frame(0, 127, mask=0, x=192.0),
        _frame(1, 131, mask=1, x=192.0),
    ]
    frames[1]["decision"] = {
        "dialogue_delivery": [
            {"game_frame": 128, "current_input_mask": 1},
            {"game_frame": 129, "current_input_mask": 0},
            {"game_frame": 129, "current_input_mask": 0},
        ]
    }
    source = [
        _source(127, mask=0, x=192.0),
        _source(128, mask=1, x=192.0),
        _source(129, mask=0, x=192.0),
        _source(131, mask=1, x=192.0),
    ]
    trace = tmp_path / "dialogue.jsonl"
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in source), encoding="utf-8"
    )

    result = compare_retail_source_states(frames, trace)

    assert result["dialogue_delivery"] == {
        "available": True,
        "equal": True,
        "sample_records": 3,
        "unique_sample_frames": 2,
        "matched_sample_frames": 2,
        "missing_sample_frames": 0,
        "first_missing_sample_frame": None,
        "first_sample_frame": 128,
        "last_sample_frame": 129,
        "first_divergence": None,
    }

    source[2]["input"] = 1
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in source), encoding="utf-8"
    )
    mismatched = compare_retail_source_states(frames, trace)
    assert mismatched["dialogue_delivery"]["equal"] is False
    assert mismatched["dialogue_delivery"]["first_divergence"]["frame"] == 129


def test_retail_source_comparison_audits_native_safe_set_and_reconvergence(
    tmp_path: Path,
) -> None:
    frames = [
        _frame(0, 127, mask=0, x=192.0),
        _frame(1, 128, mask=0, x=192.0),
        _frame(2, 129, mask=0, x=192.0),
    ]
    for frame, actions in zip(
        frames,
        (("stay",), ("left",), ("up",)),
        strict=True,
    ):
        frame["decision"] = {
            "hard_actions": [
                [action, None, 192.0, 384.0] for action in actions
            ],
            "dialogue_delivery": [],
        }
    trace = tmp_path / "native-sets.jsonl"
    trace.write_text(
        "".join(
            json.dumps(_source(frame, mask=0, x=192.0)) + "\n"
            for frame in (127, 128, 129)
        ),
        encoding="utf-8",
    )

    source_sets = {
        127: ("stay",),
        128: ("right",),
        129: ("up",),
    }
    result = compare_retail_source_states(
        frames,
        trace,
        native_safe_set_resolver=lambda observation: source_sets[
            int(observation["tick"])
        ],
    )

    native = result["native_safe_set"]
    assert native["available"] is True
    assert native["delivery_delays"] == [0, 1, 2, 3]
    assert native["common_snapshots"] == 3
    assert native["equal_snapshots"] == 2
    assert native["differing_snapshots"] == 1
    assert native["first_divergence"] == {
        "frame": 128,
        "retail_hard_actions": ["left"],
        "source_hard_actions": ["right"],
        "only_retail": ["left"],
        "only_source": ["right"],
    }
    assert native["last_divergence"]["frame"] == 128
    assert native["first_reconvergence_after_divergence"] == 129
    assert native["terminal_window"]["equal_snapshots"] == 2
