#!/usr/bin/env python3
"""Freeze a verified Bomb-free action prefix from one headless corpus run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from th06_rl.offline import ACTION_SET

try:
    from run_source_platform_differential import (
        ACTION_STREAM_SCHEMA,
        DELIVERY_CONTRACT,
        ActionSegment,
        SourceActionStream,
    )
except ModuleNotFoundError:  # Imported as scripts.export_headless_action_stream.
    from scripts.run_source_platform_differential import (
        ACTION_STREAM_SCHEMA,
        DELIVERY_CONTRACT,
        ActionSegment,
        SourceActionStream,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid transition JSON at line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"transition line {line_number} is not an object")
            yield row


def _rle(actions: list[str]) -> tuple[ActionSegment, ...]:
    segments: list[ActionSegment] = []
    for action in actions:
        if segments and segments[-1].action == action:
            previous = segments[-1]
            segments[-1] = ActionSegment(count=previous.count + 1, action=action)
        else:
            segments.append(ActionSegment(count=1, action=action))
    return tuple(segments)


def export_corpus_action_stream(run_directory: Path, *, max_ticks: int) -> SourceActionStream:
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != "th06-rl-headless-corpus-v1":
        raise ValueError("unsupported headless corpus manifest")
    if manifest.get("transaction_complete") is not True:
        raise ValueError("headless corpus transaction is incomplete")
    if manifest.get("continue_after_hit") is not False:
        raise ValueError("HIT-continuation corpus cannot become an action differential input")
    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("headless corpus scope is missing")
    files = manifest.get("files")
    transition_file = files.get("transitions") if isinstance(files, dict) else None
    if not isinstance(transition_file, dict) or not isinstance(transition_file.get("path"), str):
        raise ValueError("headless corpus transition file is missing")
    transitions_path = run_directory / transition_file["path"]
    expected_sha256 = transition_file.get("sha256")
    actual_sha256 = _sha256(transitions_path)
    if expected_sha256 != actual_sha256:
        raise ValueError("headless transition SHA-256 does not match the manifest")

    actions: list[str] = []
    for row in _rows(transitions_path):
        index = len(actions)
        if index >= max_ticks:
            break
        if row.get("sequence") != index or row.get("next_tick") != index + 2:
            raise ValueError(f"non-contiguous transition sequence at {index}")
        if row.get("scope") != scope:
            raise ValueError(f"mixed transition scope at {index}")
        if row.get("benchmark_forced_action") is not False:
            raise ValueError(f"benchmark-forced action at transition {index}")
        behavior = row.get("behavior")
        action = behavior.get("selected_action") if isinstance(behavior, dict) else None
        if not isinstance(action, str) or action not in ACTION_SET:
            raise ValueError(f"unknown or forbidden action at transition {index}")
        outcome = row.get("outcome_terms")
        if not isinstance(outcome, dict) or outcome.get("bombs_used_delta") != 0:
            raise ValueError(f"Bomb-free outcome is not established at transition {index}")
        if outcome.get("terminal_reason") is not None and index + 1 < max_ticks:
            raise ValueError(f"corpus terminates before requested prefix at transition {index}")
        actions.append(action)
    if len(actions) != max_ticks:
        raise ValueError(
            f"corpus has only {len(actions)} verified actions; {max_ticks} requested"
        )

    source = manifest.get("source")
    ranker = manifest.get("ranker")
    return SourceActionStream(
        difficulty=int(scope["difficulty"]),
        character=int(scope["character"]),
        shot_type=int(scope["shot_type"]),
        stage=int(scope["stage"]),
        initial_seed=int(manifest["initial_seed"]),
        max_ticks=max_ticks,
        auto_shoot=True,
        segments=_rle(actions),
        description="Frozen Bomb-free headless policy prefix for source platform differential",
        provenance={
            "kind": "verified-headless-corpus-action-prefix",
            "run_directory": str(run_directory.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "transitions_sha256": actual_sha256,
            "source": source,
            "behavior_policy": manifest.get("behavior_policy"),
            "ranker": ranker,
            "exported_actions": len(actions),
            "auto_shoot_source": "HeadlessClient default contract",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--max-ticks", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        stream = export_corpus_action_stream(
            args.run_directory.resolve(), max_ticks=args.max_ticks
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(stream.as_object(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": ACTION_STREAM_SCHEMA,
                "delivery_contract": DELIVERY_CONTRACT,
                "output": str(args.output.resolve()),
                "actions": stream.action_count,
                "segments": len(stream.segments),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
