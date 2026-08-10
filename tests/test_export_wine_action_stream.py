from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scripts.export_wine_action_stream import (
    build_retail_action_stream,
    recover_stage_rng_seed,
)
from th06_rl.wine_risk import FirstFailurePrefix


def _prefix(tmp_path: Path) -> FirstFailurePrefix:
    return FirstFailurePrefix(
        schema="th06-rl-wine-first-failure-prefix-v1",
        run_id="run",
        run_dir=tmp_path,
        scope=(3, 0, 0, 6),
        executable_sha256="a" * 64,
        native_kernel_sha256="b" * 64,
        code_commit="c" * 40,
        manifest_sha256="d" * 64,
        run_sha256="e" * 64,
        failure_kind="control-dead-end",
        failure_frame=132,
        failure_context="boss:0:sub10",
        failure_segment_start_frame=130,
        positive_window_start_frame=130,
        transitions=2,
        examples=(),
    )


def _frame(
    sequence: int,
    frame: int,
    mask: int,
    current: str,
    published: str | None,
    *,
    dialogue_delivery: list[dict[str, object]] | None = None,
):
    return {
        "sequence": sequence,
        "snapshot": {
            "frame": frame,
            "difficulty": 3,
            "character": 0,
            "shot_type": 0,
            "stage": 6,
            "input_mask": mask,
            "rng_seed": 33367,
            "rng_generation": 2,
        },
        "decision": {
            "current_action": current,
            "published_action": published,
            **(
                {"dialogue_delivery": dialogue_delivery}
                if dialogue_delivery is not None
                else {}
            ),
        },
    }


def _transition(
    sequence: int,
    source: int,
    target: int,
    proposed: str,
    published: str | None,
):
    return {
        "sequence": sequence,
        "snapshot_ref": f"run:{sequence:08d}:f{source}",
        "next_snapshot_ref": f"run:{sequence + 1:08d}:f{target}",
        "proposed_action": proposed,
        "published_action": published,
        "outcome_terms": {
            "elapsed_frames": target - source,
            "bomb_used": False,
        },
    }


def test_recover_stage_rng_seed_reconstructs_observed_generation() -> None:
    assert recover_stage_rng_seed(33367, 2) == 3193


def test_retail_export_reconstructs_prelude_gaps_and_stale_retry(tmp_path: Path) -> None:
    frames = [
        _frame(0, 127, 0x00, "stay_fast", "up_fast"),
        _frame(1, 131, 0x11, "up_fast", None),
        _frame(2, 132, 0x11, "up_fast", "up_fast"),
    ]
    transitions = [
        _transition(0, 127, 131, "up_fast", "up_fast"),
        _transition(1, 131, 132, "up_fast", None),
    ]

    stream = build_retail_action_stream(_prefix(tmp_path), transitions, frames)

    assert stream.max_ticks == 132
    assert stream.action_count == 132
    assert stream.stage_rng_seed == 3193
    assert stream.auto_shoot is True
    assert stream.auto_shoot_after_tick == 127
    assert stream.retail_dialogue_control is True
    assert stream.retail_dialogue_control_after_tick == 132
    assert [(segment.count, segment.action) for segment in stream.segments] == [
        (126, "stay_fast"),
        (6, "up_fast"),
    ]
    assert stream.provenance["maximum_observation_gap"] == 4
    assert stream.provenance["coverage_padding_actions"] == 1


def test_retail_export_refuses_bomb_evidence(tmp_path: Path) -> None:
    frames = [
        _frame(0, 127, 0x00, "stay_fast", "up_fast"),
        _frame(1, 131, 0x11, "up_fast", None),
        _frame(2, 132, 0x11, "up_fast", "up_fast"),
    ]
    transitions = [
        _transition(0, 127, 131, "up_fast", "up_fast"),
        _transition(1, 131, 132, "up_fast", None),
    ]
    transitions[0]["outcome_terms"]["bomb_used"] = True

    with pytest.raises(ValueError, match="Bomb-free"):
        build_retail_action_stream(_prefix(tmp_path), transitions, frames)

    transitions[0]["outcome_terms"]["bomb_used"] = False
    frames[1]["snapshot"]["input_mask"] = 0x13
    with pytest.raises(ValueError, match="Bomb bit"):
        build_retail_action_stream(_prefix(tmp_path), transitions, frames)


def test_retail_export_delays_shoot_through_initial_stale_retry(tmp_path: Path) -> None:
    frames = [
        _frame(0, 127, 0x00, "stay_fast", None),
        _frame(1, 131, 0x00, "stay_fast", "up_fast"),
        _frame(2, 132, 0x11, "up_fast", "up_fast"),
    ]
    transitions = [
        _transition(0, 127, 131, "stay_fast", None),
        _transition(1, 131, 132, "up_fast", "up_fast"),
    ]

    stream = build_retail_action_stream(_prefix(tmp_path), transitions, frames)

    assert stream.auto_shoot_after_tick == 131
    assert [(segment.count, segment.action) for segment in stream.segments] == [
        (130, "stay_fast"),
        (2, "up_fast"),
    ]


def test_retail_export_uses_sampled_target_action_over_publication_intent(
    tmp_path: Path,
) -> None:
    frames = [
        _frame(0, 127, 0x00, "stay_fast", "up_fast"),
        _frame(1, 128, 0x00, "stay_fast", "up_fast"),
        _frame(2, 132, 0x11, "up_fast", "up_fast"),
    ]
    transitions = [
        _transition(0, 127, 128, "up_fast", "up_fast"),
        _transition(1, 128, 132, "up_fast", "up_fast"),
    ]

    stream = build_retail_action_stream(_prefix(tmp_path), transitions, frames)

    assert [(segment.count, segment.action) for segment in stream.segments] == [
        (127, "stay_fast"),
        (5, "up_fast"),
    ]
    assert "coherent target snapshot action" in stream.provenance[
        "delivery_reconstruction"
    ]


def test_retail_export_delays_dialogue_control_to_largest_observed_gap(
    tmp_path: Path,
) -> None:
    frames = [
        _frame(0, 127, 0x00, "stay_fast", "up_fast"),
        _frame(1, 131, 0x11, "up_fast", "up_fast"),
        _frame(2, 140, 0x11, "up_fast", "up_fast"),
        _frame(3, 150, 0x11, "up_fast", "up_fast"),
    ]
    transitions = [
        _transition(0, 127, 131, "up_fast", "up_fast"),
        _transition(1, 131, 140, "up_fast", "up_fast"),
        _transition(2, 140, 150, "up_fast", "up_fast"),
    ]

    stream = build_retail_action_stream(
        replace(_prefix(tmp_path), failure_frame=150, transitions=3),
        transitions,
        frames,
    )

    assert stream.retail_dialogue_control_after_tick == 140
    assert stream.provenance["maximum_observation_gap"] == 10


def test_retail_export_preserves_sampled_dialogue_edges_and_held_backfill(
    tmp_path: Path,
) -> None:
    samples = [
        {
            "game_frame": 129,
            "current_input_mask": 1,
            "previous_input_mask": 1,
            "published_input_mask": 0,
            "held_repeat": 0,
            "held_frames": 1,
            "active": True,
            "skippable": False,
            "pulsed_shoot": False,
        },
        {
            "game_frame": 130,
            "current_input_mask": 0,
            "previous_input_mask": 1,
            "published_input_mask": 0,
            "held_repeat": 0,
            "held_frames": 0,
            "active": True,
            "skippable": False,
            "pulsed_shoot": False,
        },
        {
            "game_frame": 131,
            "current_input_mask": 1,
            "previous_input_mask": 0,
            "published_input_mask": 1,
            "held_repeat": 0,
            "held_frames": 0,
            "active": False,
            "skippable": False,
            "pulsed_shoot": True,
        },
    ]
    frames = [
        _frame(0, 127, 0x00, "stay_fast", "up_fast"),
        _frame(
            1,
            131,
            0x01,
            "stay_fast",
            None,
            dialogue_delivery=samples,
        ),
        _frame(2, 132, 0x11, "up_fast", "up_fast"),
    ]
    transitions = [
        _transition(0, 127, 131, "up_fast", "up_fast"),
        _transition(1, 131, 132, "up_fast", None),
    ]

    stream = build_retail_action_stream(_prefix(tmp_path), transitions, frames)

    assert [
        (segment.start_tick, segment.count, segment.input_mask)
        for segment in stream.retail_dialogue_inputs
    ] == [(127, 2, 1), (129, 1, 0), (130, 1, 1)]
    assert stream.provenance["retail_dialogue_input_evidence"] == {
        "sample_records": 3,
        "sampled_current_frames": 3,
        "sampled_previous_frames": 3,
        "held_backfilled_frames": 0,
        "exact_frames": 4,
        "segments": 3,
        "first_exact_frame": 128,
        "last_exact_frame": 131,
        "first_runtime_input_tick": 127,
        "last_runtime_input_tick": 130,
    }
    assert "rendering, numeric, and RNG" in stream.provenance["known_limit"]


def test_retail_export_refuses_bomb_in_dialogue_delivery(tmp_path: Path) -> None:
    sample = {
        "game_frame": 130,
        "current_input_mask": 0x02,
        "previous_input_mask": 0,
        "published_input_mask": 0,
        "held_repeat": 0,
        "held_frames": 0,
        "active": True,
        "skippable": False,
        "pulsed_shoot": False,
    }
    frames = [
        _frame(0, 127, 0x00, "stay_fast", "up_fast"),
        _frame(1, 131, 0x11, "up_fast", None, dialogue_delivery=[sample]),
        _frame(2, 132, 0x11, "up_fast", "up_fast"),
    ]
    transitions = [
        _transition(0, 127, 131, "up_fast", "up_fast"),
        _transition(1, 131, 132, "up_fast", None),
    ]

    with pytest.raises(ValueError, match="Bomb-bearing"):
        build_retail_action_stream(_prefix(tmp_path), transitions, frames)


def test_retail_export_refuses_terminal_or_scope_drift(tmp_path: Path) -> None:
    frames = [
        _frame(0, 127, 0x00, "stay_fast", "up_fast"),
        _frame(1, 131, 0x11, "up_fast", None),
        _frame(2, 132, 0x11, "up_fast", "up_fast"),
    ]
    transitions = [
        _transition(0, 127, 131, "up_fast", "up_fast"),
        _transition(1, 131, 132, "up_fast", None),
    ]

    with pytest.raises(ValueError, match="terminal frame"):
        build_retail_action_stream(
            replace(_prefix(tmp_path), failure_frame=133), transitions, frames
        )

    frames[0]["snapshot"]["stage"] = 5
    with pytest.raises(ValueError, match="scope mismatch"):
        build_retail_action_stream(_prefix(tmp_path), transitions, frames)
