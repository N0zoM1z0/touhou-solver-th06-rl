from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.summarize_headless_continuation import summarize, summarize_transition_file


def _row(sequence: int, *, hit: int = 0, forced: bool = False) -> dict:
    return {
        "sequence": sequence,
        "tick": 10 + sequence,
        "next_tick": 11 + sequence,
        "scope": {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 4},
        "source_context": f"timeline:{sequence}",
        "behavior": {"policy": "candidate", "selected_action": "stay_fast"},
        "benchmark_forced_action": forced,
        "outcome_terms": {
            "deaths_delta": hit,
            "bombs_used_delta": 0,
            "terminal_reason": None,
        },
    }


def test_interrupted_gzip_counts_only_complete_records(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    complete = gzip.compress(
        b"".join(json.dumps(row).encode() + b"\n" for row in (_row(0), _row(1, hit=1)))
    )
    path = run / "transitions.jsonl.gz.partial"
    path.write_bytes(complete[:-8])

    result = summarize_transition_file(path)

    assert result["status"] == "interrupted-partial"
    assert result["training_eligible"] is False
    assert result["rows"] == 2
    assert result["physical_hits"] == 1
    assert result["physical_hit_ticks"] == [12]
    assert result["first_hit_tick"] == 12
    assert result["bombs_used"] == 0


def test_summary_keeps_hit_and_forced_rates_and_reasons(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    path = run / "transitions.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in (_row(0), _row(1, hit=1, forced=True), _row(2)):
            stream.write(json.dumps(row) + "\n")
    (run / "manifest.json").write_text(
        json.dumps({
            "termination_reason": "tick-limit",
            "training_eligible": False,
            "authority_failure_reasons": {
                "laser slot 0 lacks angular history": 1,
            },
        }),
        encoding="utf-8",
    )

    result = summarize([tmp_path])

    assert result["complete_runs"] == 1
    assert result["physical_hits"] == 1
    assert result["benchmark_forced_rows"] == 1
    assert result["benchmark_forced_reason_rows"] == {
        "laser slot 0 lacks angular history": 1,
    }
    assert result["unattributed_benchmark_forced_rows"] == 0
    assert result["hits_per_1000_ticks"] == 1000 / 3


def test_partial_run_intent_preserves_full_ranker_identity(tmp_path: Path) -> None:
    run = tmp_path / "run-seed23"
    run.mkdir()
    path = run / "transitions.jsonl.gz.partial"
    path.write_bytes(gzip.compress(json.dumps(_row(0)).encode() + b"\n")[:-8])
    (run / "run-intent.json").write_text(json.dumps({
        "scope": _row(0)["scope"],
        "initial_seed": 23,
        "continue_after_hit": True,
        "ranker": {"sha256": "a" * 64},
    }), encoding="utf-8")

    result = summarize_transition_file(path)

    assert result["seed"] == 23
    assert result["continue_after_hit"] is True
    assert result["ranker_sha256"] == "a" * 64
    assert result["benchmark_forced_reason_rows"] == {}
    assert result["unattributed_benchmark_forced_rows"] == 0


def test_partial_forced_rows_remain_explicitly_unattributed(tmp_path: Path) -> None:
    run = tmp_path / "run-seed24"
    run.mkdir()
    path = run / "transitions.jsonl.gz.partial"
    path.write_bytes(gzip.compress(json.dumps(_row(0, forced=True)).encode() + b"\n")[:-8])

    result = summarize([run])

    assert result["benchmark_forced_rows"] == 1
    assert result["benchmark_forced_reason_rows"] == {}
    assert result["unattributed_benchmark_forced_rows"] == 1
