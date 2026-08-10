from argparse import ArgumentTypeError
from pathlib import Path

import pytest

from scripts.run_headless_paired_continuation import (
    Candidate,
    _unique,
    child_command,
    parse_candidate,
)


def test_candidate_parser_rejects_unsafe_labels() -> None:
    assert parse_candidate("incumbent=model.joblib") == Candidate(
        "incumbent", Path("model.joblib")
    )
    with pytest.raises(ArgumentTypeError):
        parse_candidate("../escape=model.joblib")


def test_panel_requires_two_unique_seeds() -> None:
    assert _unique([163, 164, 163]) == (163, 164)
    with pytest.raises(ValueError, match="at least two"):
        _unique([163, 163])


def test_child_command_fixes_continuation_contract(tmp_path: Path) -> None:
    command = child_command(
        root=tmp_path,
        candidate=Candidate("candidate", tmp_path / "model.joblib"),
        seed=165,
        stage=6,
        difficulty=3,
        character=0,
        shot_type=0,
        threads=2,
        anchor_stride=4096,
        teacher_horizon=12,
        binary=tmp_path / "th06",
        game_directory=tmp_path / "game",
        output_root=tmp_path / "panel",
    )
    assert command[command.index("--max-ticks") + 1] == "0"
    assert "--continue-after-hit" in command
    assert command[command.index("--stage") + 1] == "6"
    assert command[command.index("--seed") + 1] == "165"
    assert command[command.index("--output-root") + 1] == str(
        tmp_path / "panel/candidate"
    )
