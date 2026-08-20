from pathlib import Path

import pytest

from scripts.collect_wine_parallel import _schedule, _stages, parse_args


def _workers():
    return [
        {"worker": 0},
        {"worker": 1},
    ]


def test_collection_schedule_is_outcome_independent_and_round_robin(
    tmp_path: Path,
) -> None:
    schedule = _schedule(
        episodes=7,
        stages=(1, 2, 3, 4, 5, 6),
        workers=_workers(),
        gate_sha256="a" * 64,
        commit="commit",
    )
    assert [row["worker"] for row in schedule["episodes"]] == [0, 1, 0, 1, 0, 1, 0]
    assert [row["stage"] for row in schedule["episodes"]] == [1, 2, 3, 4, 5, 6, 1]
    assert schedule["natural_rng"] is True


def test_collection_stage_scope_is_explicit_and_valid() -> None:
    assert _stages("6,4,2") == (6, 4, 2)
    with pytest.raises(Exception, match="1..6"):
        _stages("0,6")


def test_collection_requires_a_positive_predeclared_episode_count(
    tmp_path: Path,
) -> None:
    common = [
        "--gate", str(tmp_path / "gate.json"),
        "--policy-plugin", str(tmp_path / "policy.py"),
        "--policy-state", str(tmp_path / "policy.json"),
        "--artifact-root", str(tmp_path / "artifacts"),
        "--corpus-root", str(tmp_path / "corpus"),
        "--output", str(tmp_path / "collection.json"),
    ]
    with pytest.raises(SystemExit):
        parse_args([*common, "--episodes", "0"])
    args = parse_args([*common, "--episodes", "12", "--stages", "1,6"])
    assert args.episodes == 12
    assert args.stages == (1, 6)
