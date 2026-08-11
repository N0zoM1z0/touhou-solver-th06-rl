from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_headless_action_stream import export_corpus_action_stream


def _write_run(tmp_path: Path, *, forced: bool = False) -> Path:
    rows = []
    for index, action in enumerate(("stay", "stay", "left_fast", "left_fast")):
        rows.append(
            {
                "sequence": index,
                "next_tick": index + 2,
                "scope": {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 6},
                "benchmark_forced_action": forced and index == 2,
                "behavior": {"selected_action": action},
                "outcome_terms": {"bombs_used_delta": 0, "terminal_reason": None},
            }
        )
    transitions = tmp_path / "transitions.jsonl.gz"
    with gzip.open(transitions, "wt", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(transitions.read_bytes()).hexdigest()
    manifest = {
        "schema": "th06-rl-headless-corpus-v1",
        "transaction_complete": True,
        "continue_after_hit": False,
        "scope": {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 6},
        "initial_seed": 73,
        "behavior_policy": "frozen-test",
        "source": {"commit": "a" * 40},
        "ranker": {"sha256": "b" * 64},
        "files": {"transitions": {"path": transitions.name, "sha256": digest}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_export_headless_action_stream_verifies_and_run_length_encodes(tmp_path: Path) -> None:
    stream = export_corpus_action_stream(_write_run(tmp_path), max_ticks=4)

    assert stream.initial_seed == 73
    assert [(segment.count, segment.action) for segment in stream.segments] == [
        (2, "stay"),
        (2, "left_fast"),
    ]
    assert stream.provenance["kind"] == "verified-headless-corpus-action-prefix"


def test_export_refuses_benchmark_forced_action(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="benchmark-forced"):
        export_corpus_action_stream(_write_run(tmp_path, forced=True), max_ticks=4)
