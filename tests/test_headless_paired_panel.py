from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scripts.compare_headless_paired_panel import compare


SCOPE = {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 5}
SOURCE = {"commit": "source-commit", "binary_sha256": "b" * 64, "clean": True}


def _run(
    root: Path,
    label: str,
    seed: int,
    *,
    hits: int,
    forced: int,
    ranker: str,
    termination: str = "stage-clear-success",
) -> None:
    run = root / label / f"run-seed{seed}"
    run.mkdir(parents=True)
    rows = []
    row_count = max(3, hits, forced)
    for sequence in range(row_count):
        rows.append({
            "sequence": sequence,
            "tick": sequence,
            "next_tick": sequence + 1,
            "scope": SCOPE,
            "source_context": "timeline:end",
            "behavior": {"policy": "candidate", "selected_action": "stay_fast"},
            "benchmark_forced_action": sequence < forced,
            "outcome_terms": {
                "deaths_delta": 1 if sequence < hits else 0,
                "bombs_used_delta": 0,
                "terminal_reason": termination if sequence == row_count - 1 else None,
            },
        })
    with gzip.open(run / "transitions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (run / "manifest.json").write_text(json.dumps({
        "transaction_complete": True,
        "training_eligible": False,
        "scope": SCOPE,
        "source": SOURCE,
        "initial_seed": seed,
        "termination_reason": termination,
        "continue_after_hit": True,
        "ranker": {"sha256": ranker},
    }), encoding="utf-8")


def test_compare_reports_aggregate_and_seedwise_dominance(tmp_path: Path) -> None:
    for seed, left_hits, left_forced, right_hits, right_forced in (
        (91, 2, 4, 3, 5),
        (92, 1, 2, 1, 3),
    ):
        _run(tmp_path, "left", seed, hits=left_hits, forced=left_forced, ranker="l" * 64)
        _run(tmp_path, "right", seed, hits=right_hits, forced=right_forced, ranker="r" * 64)

    result = compare([
        ("left", tmp_path / "left"),
        ("right", tmp_path / "right"),
    ])

    verdict = next(row for row in result["comparisons"] if row["left"] == "left")
    assert result["seeds"] == [91, 92]
    assert verdict["aggregate_dominates"] is True
    assert verdict["seedwise_dominates"] is True
    assert result["seedwise_dominators"] == ["left"]
    assert result["promotion_allowed"] is False


def test_compare_rejects_different_seed_panels(tmp_path: Path) -> None:
    for seed in (91, 92):
        _run(tmp_path, "left", seed, hits=0, forced=0, ranker="l" * 64)
    for seed in (91, 93):
        _run(tmp_path, "right", seed, hits=0, forced=0, ranker="r" * 64)

    with pytest.raises(ValueError, match="exact same seed panel"):
        compare([("left", tmp_path / "left"), ("right", tmp_path / "right")])


def test_compare_rejects_non_natural_run(tmp_path: Path) -> None:
    for label, ranker in (("left", "l" * 64), ("right", "r" * 64)):
        _run(tmp_path, label, 91, hits=0, forced=0, ranker=ranker)
        _run(
            tmp_path,
            label,
            92,
            hits=0,
            forced=0,
            ranker=ranker,
            termination="tick-limit",
        )

    with pytest.raises(ValueError, match="did not finish naturally"):
        compare([("left", tmp_path / "left"), ("right", tmp_path / "right")])
