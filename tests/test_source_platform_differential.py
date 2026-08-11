from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_source_platform_differential import (
    ACTION_STREAM_SCHEMA,
    DELIVERY_CONTRACT,
    _runtime_command,
    _windows_path,
    compare_source_traces,
    parse_action_stream,
    render_action_file,
    render_dialogue_input_file,
    _trace_summary,
)


def _stream() -> dict[str, object]:
    return {
        "schema": ACTION_STREAM_SCHEMA,
        "delivery_contract": DELIVERY_CONTRACT,
        "scope": {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 6},
        "initial_seed": 7,
        "max_ticks": 12,
        "auto_shoot": True,
        "segments": [
            {"count": 4, "action": "stay"},
            {"count": 8, "action": "left_fast"},
        ],
    }


def _observation(tick: int, *, x: float = 10.0, births: int = 0) -> dict[str, object]:
    return {
        "schema": "th06-headless-observation-v2",
        "tick": tick,
        "terminal_reason": None,
        "player": {"x": x},
        "events": {"bullet_births": [{"slot": index} for index in range(births)]},
    }


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_action_stream_is_strict_bomb_free_and_covers_tick_bound() -> None:
    stream = parse_action_stream(_stream())

    assert stream.action_count == 12
    assert render_action_file(stream) == "4 stay\n8 left_fast\n"
    command = _runtime_command(stream, binary="th06", actions="a.txt", trace="t.jsonl")
    assert "--continue-after-hit" not in command
    assert command[-1] == "--auto-shoot"
    assert "--stage-rng-seed" not in command

    invalid = _stream()
    invalid["segments"] = [{"count": 12, "action": "bomb"}]
    with pytest.raises(ValueError, match="unknown or forbidden"):
        parse_action_stream(invalid)

    exhausted = _stream()
    exhausted["segments"] = [{"count": 11, "action": "stay"}]
    with pytest.raises(ValueError, match="cover max_ticks"):
        parse_action_stream(exhausted)


def test_action_stream_passes_optional_stage_rng_seed_to_both_runtimes() -> None:
    raw = _stream()
    raw["stage_rng_seed"] = 3193

    stream = parse_action_stream(raw)
    command = _runtime_command(stream, binary="th06", actions="a.txt", trace="t.jsonl")

    assert stream.stage_rng_seed == 3193
    assert command[command.index("--stage-rng-seed") + 1] == "3193"

    raw["stage_rng_seed"] = 65536
    with pytest.raises(ValueError, match="stage_rng_seed"):
        parse_action_stream(raw)


def test_action_stream_can_delay_shoot_for_a_retail_capture_prelude() -> None:
    raw = _stream()
    raw["auto_shoot_after_tick"] = 7

    stream = parse_action_stream(raw)
    command = _runtime_command(stream, binary="th06", actions="a.txt", trace="t.jsonl")

    assert stream.auto_shoot_after_tick == 7
    assert command[command.index("--auto-shoot-after-tick") + 1] == "7"

    raw["auto_shoot"] = False
    with pytest.raises(ValueError, match="requires auto_shoot"):
        parse_action_stream(raw)


def test_action_stream_can_enable_separate_retail_dialogue_delivery() -> None:
    raw = _stream()
    raw["retail_dialogue_control"] = True
    raw["retail_dialogue_control_after_tick"] = 7

    stream = parse_action_stream(raw)
    command = _runtime_command(stream, binary="th06", actions="a.txt", trace="t.jsonl")

    assert stream.retail_dialogue_control is True
    assert command[command.index("--retail-dialogue-control-after-tick") + 1] == "7"

    raw["auto_shoot"] = False
    with pytest.raises(ValueError, match="requires auto_shoot"):
        parse_action_stream(raw)

    raw = _stream()
    raw["retail_dialogue_control_after_tick"] = 7
    with pytest.raises(ValueError, match="requires retail_dialogue_control"):
        parse_action_stream(raw)


def test_action_stream_can_replay_exact_bomb_free_dialogue_inputs() -> None:
    raw = _stream()
    raw["retail_dialogue_control"] = True
    raw["retail_dialogue_inputs"] = [
        {"start_tick": 7, "count": 2, "input_mask": 0x01},
        {"start_tick": 9, "count": 1, "input_mask": 0x100},
    ]

    stream = parse_action_stream(raw)
    assert render_dialogue_input_file(stream) == "7 2 1\n9 1 256\n"
    command = _runtime_command(
        stream,
        binary="th06",
        actions="a.txt",
        trace="t.jsonl",
        dialogue_inputs="dialogue.txt",
    )
    assert command[command.index("--retail-dialogue-inputs") + 1] == "dialogue.txt"

    with pytest.raises(ValueError, match="rendered input file"):
        _runtime_command(stream, binary="th06", actions="a.txt", trace="t.jsonl")

    raw["retail_dialogue_inputs"][1]["start_tick"] = 8
    with pytest.raises(ValueError, match="overlap or are unordered"):
        parse_action_stream(raw)

    raw["retail_dialogue_inputs"][1] = {
        "start_tick": 9,
        "count": 1,
        "input_mask": 0x02,
    }
    with pytest.raises(ValueError, match="forbidden or Bomb-bearing"):
        parse_action_stream(raw)


def test_windows_path_uses_wine_z_drive(tmp_path: Path) -> None:
    assert _windows_path(tmp_path / "trace.jsonl").startswith("Z:\\")
    assert _windows_path(tmp_path / "trace.jsonl").endswith("\\trace.jsonl")


def test_trace_comparison_keeps_events_separate_from_physical_state(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_trace(left, [_observation(1), _observation(2, births=1)])
    _write_trace(right, [_observation(1), _observation(2, births=2)])

    result = compare_source_traces(left, right, absolute_tolerance=1e-6)

    assert result["exact_physical"]["equal"] is True
    assert result["tolerant_physical"]["equal"] is True
    assert result["events"]["equal"] is False
    assert result["discrete_delivery"]["equal"] is True
    assert result["events"]["first_divergence"]["line"] == 2


def test_trace_comparison_reports_exact_and_tolerant_drift_independently(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_trace(left, [_observation(1), _observation(2, x=10.0)])
    _write_trace(right, [_observation(1), _observation(2, x=10.0000005)])

    result = compare_source_traces(left, right, absolute_tolerance=1e-6)

    assert result["exact_physical"]["equal"] is False
    assert result["exact_physical"]["first_divergence"]["difference"]["path"] == "$.player.x"
    assert result["tolerant_physical"]["equal"] is True
    assert result["events"]["equal"] is True
    assert result["discrete_delivery"]["equal"] is True
    by_tolerance = {
        row["absolute_tolerance"]: row for row in result["tolerance_ladder"]
    }
    assert by_tolerance[1e-7]["equal"] is False
    assert by_tolerance[1e-6]["equal"] is True


def test_trace_summary_normalizes_windows_line_endings_without_hiding_them(tmp_path: Path) -> None:
    linux = tmp_path / "linux.jsonl"
    windows = tmp_path / "windows.jsonl"
    encoded = json.dumps(_observation(1)).encode()
    linux.write_bytes(encoded + b"\n")
    windows.write_bytes(encoded + b"\r\n")

    linux_summary = _trace_summary(linux)
    windows_summary = _trace_summary(windows)

    assert linux_summary["sha256"] != windows_summary["sha256"]
    assert linux_summary["lf_normalized_sha256"] == windows_summary["lf_normalized_sha256"]
    assert linux_summary["lf_only_lines"] == 1
    assert windows_summary["crlf_lines"] == 1
