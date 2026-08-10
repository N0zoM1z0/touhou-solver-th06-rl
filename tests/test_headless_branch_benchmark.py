from __future__ import annotations

from pathlib import Path

import pytest

from scripts.benchmark_headless_branches import _normalized_prefix


def test_prefix_normalization_truncates_surplus_rle_without_bomb(tmp_path: Path) -> None:
    actions = tmp_path / "actions.txt"
    actions.write_text(
        "# common prefix\n2 stay\nright_fast\n100 left\n",
        encoding="utf-8",
    )

    assert _normalized_prefix(actions, 5) == (
        (2, "stay"),
        (1, "right_fast"),
        (2, "left"),
    )


def test_prefix_normalization_rejects_short_or_forbidden_input(tmp_path: Path) -> None:
    short = tmp_path / "short.txt"
    short.write_text("stay\n", encoding="utf-8")
    with pytest.raises(ValueError, match="short"):
        _normalized_prefix(short, 2)

    bomb = tmp_path / "bomb.txt"
    bomb.write_text("bomb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        _normalized_prefix(bomb, 1)
