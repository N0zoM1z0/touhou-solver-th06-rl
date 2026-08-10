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

    invalid = _stream()
    invalid["segments"] = [{"count": 12, "action": "bomb"}]
    with pytest.raises(ValueError, match="unknown or forbidden"):
        parse_action_stream(invalid)

    exhausted = _stream()
    exhausted["segments"] = [{"count": 11, "action": "stay"}]
    with pytest.raises(ValueError, match="cover max_ticks"):
        parse_action_stream(exhausted)


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
