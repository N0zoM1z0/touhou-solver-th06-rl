#!/usr/bin/env python3
"""Summarize complete or interrupted headless HIT-continuation trajectories.

Interrupted ``*.jsonl.gz.partial`` files are deliberately useful benchmark
evidence.  Only newline-terminated JSON records are counted; a truncated final
record and the missing gzip footer are reported rather than treated as corpus
corruption.  Partial runs remain training-ineligible.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping


def _complete_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield complete records, tolerating only an interrupted gzip tail."""

    stream = gzip.open(path, "rb")
    try:
        while True:
            try:
                line = stream.readline()
            except (EOFError, gzip.BadGzipFile):
                break
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid complete JSON record in {path}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON record in {path}")
            yield row
    finally:
        stream.close()


def _longest_no_hit_interval(start_tick: int, final_tick: int, hits: list[int]) -> int:
    boundaries = [start_tick, *hits, final_tick]
    return max((right - left for left, right in zip(boundaries, boundaries[1:])), default=0)


def summarize_transition_file(path: Path) -> dict[str, Any]:
    run = path.parent
    manifest_path = run / "manifest.json"
    manifest: Mapping[str, Any] = {}
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            manifest = raw

    rows = 0
    first_tick: int | None = None
    final_tick: int | None = None
    scope: Mapping[str, Any] | None = None
    policy: str | None = None
    hits: list[int] = []
    forced = 0
    bombs = 0
    actions: Counter[str] = Counter()
    source_contexts: set[str] = set()
    observed_terminal_reason: str | None = None
    for row in _complete_json_lines(path):
        rows += 1
        tick = int(row["tick"])
        next_tick = int(row["next_tick"])
        first_tick = tick if first_tick is None else first_tick
        final_tick = next_tick
        if scope is None and isinstance(row.get("scope"), Mapping):
            scope = row["scope"]
        behavior = row.get("behavior")
        if isinstance(behavior, Mapping):
            if policy is None and isinstance(behavior.get("policy"), str):
                policy = behavior["policy"]
            action = behavior.get("selected_action")
            if isinstance(action, str):
                actions[action] += 1
        if row.get("benchmark_forced_action") is True:
            forced += 1
        context = row.get("source_context")
        if isinstance(context, str):
            source_contexts.add(context)
        outcome = row.get("outcome_terms")
        if isinstance(outcome, Mapping):
            deaths_delta = max(int(outcome.get("deaths_delta", 0)), 0)
            hits.extend([next_tick] * deaths_delta)
            bombs += int(outcome.get("bombs_used_delta", 0))
            terminal = outcome.get("terminal_reason")
            if isinstance(terminal, str):
                observed_terminal_reason = terminal

    is_partial = path.name.endswith(".partial") or not manifest
    termination_reason = (
        str(manifest["termination_reason"])
        if isinstance(manifest.get("termination_reason"), str)
        else observed_terminal_reason or "interrupted-partial"
    )
    start = first_tick or 0
    end = final_tick or start
    ranker = manifest.get("ranker")
    ranker_sha = ranker.get("sha256") if isinstance(ranker, Mapping) else None
    seed_match = re.search(r"-seed(\d+)$", run.name)
    inferred_seed = int(seed_match.group(1)) if seed_match else None
    return {
        "run": str(run),
        "status": "interrupted-partial" if is_partial else "complete",
        "training_eligible": False if is_partial else manifest.get("training_eligible") is True,
        "scope": dict(scope or manifest.get("scope") or {}),
        "seed": manifest.get("initial_seed", inferred_seed),
        "policy": policy or manifest.get("behavior_policy"),
        "ranker_sha256": ranker_sha,
        "termination_reason": termination_reason,
        "rows": rows,
        "start_tick": first_tick,
        "final_tick": final_tick,
        "observed_ticks": max(end - start, 0),
        "physical_hits": len(hits),
        "physical_hit_ticks": hits,
        "first_hit_tick": hits[0] if hits else None,
        "hits_per_1000_ticks": len(hits) * 1000.0 / rows if rows else 0.0,
        "longest_no_hit_interval_ticks": _longest_no_hit_interval(start, end, hits),
        "benchmark_forced_rows": forced,
        "bombs_used": bombs,
        "unique_source_contexts": len(source_contexts),
        "selected_action_counts": dict(sorted(actions.items())),
    }


def _transition_files(paths: Iterable[Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.name.startswith("transitions.jsonl.gz"):
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("transitions.jsonl.gz"))
            files.extend(path.rglob("transitions.jsonl.gz.partial"))
    return tuple(sorted(dict.fromkeys(item.resolve() for item in files)))


def summarize(paths: Iterable[Path]) -> dict[str, Any]:
    files = _transition_files(paths)
    if not files:
        raise ValueError("no complete or partial transition streams found")
    runs = [summarize_transition_file(path) for path in files]
    total_rows = sum(run["rows"] for run in runs)
    total_hits = sum(run["physical_hits"] for run in runs)
    return {
        "schema": "th06-rl-headless-hit-continuation-summary-v1",
        "runs": len(runs),
        "complete_runs": sum(run["status"] == "complete" for run in runs),
        "partial_runs": sum(run["status"] == "interrupted-partial" for run in runs),
        "rows": total_rows,
        "physical_hits": total_hits,
        "hits_per_1000_ticks": total_hits * 1000.0 / total_rows if total_rows else 0.0,
        "bombs_used": sum(run["bombs_used"] for run in runs),
        "benchmark_forced_rows": sum(run["benchmark_forced_rows"] for run in runs),
        "run_results": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.paths)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
