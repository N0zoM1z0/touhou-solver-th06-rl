#!/usr/bin/env python3
"""Independently audit compact headless trajectories and effective row yield."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from th06_rl.headless_corpus import (
    MANIFEST_SCHEMA,
    TRANSITION_SCHEMA,
    canonical_observation_sha256,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def _manifest_file(
    run: Path,
    manifest: Mapping[str, Any],
    name: str,
    errors: list[str],
) -> Path:
    files = manifest.get("files")
    entry = files.get(name) if isinstance(files, Mapping) else None
    if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
        errors.append(f"manifest lacks file entry {name}")
        return run / f"missing-{name}"
    path = run / entry["path"]
    if not path.is_file():
        errors.append(f"missing {name} file {path.name}")
        return path
    if entry.get("bytes") != path.stat().st_size:
        errors.append(f"{name} byte count mismatch")
    if entry.get("sha256") != _sha256(path):
        errors.append(f"{name} SHA-256 mismatch")
    return path


def audit_run(run: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        return {"run": str(run), "valid": False, "errors": ["manifest missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("unsupported manifest schema")
    transitions_path = _manifest_file(run, manifest, "transitions", errors)
    anchors_path = _manifest_file(run, manifest, "anchors", errors)
    transitions = _jsonl_gzip(transitions_path) if transitions_path.is_file() else []
    anchors = _jsonl_gzip(anchors_path) if anchors_path.is_file() else []
    if manifest.get("transition_count") != len(transitions):
        errors.append("transition count mismatch")
    if manifest.get("anchor_count") != len(anchors):
        errors.append("anchor count mismatch")

    valid_successors = 0
    valid_legal = 0
    valid_probability = 0
    valid_bombless = 0
    contexts: set[str] = set()
    actions: Counter[str] = Counter()
    for index, row in enumerate(transitions):
        prefix = f"transition {index}"
        if row.get("schema") != TRANSITION_SCHEMA:
            errors.append(f"{prefix}: unsupported schema")
        if row.get("sequence") != index:
            errors.append(f"{prefix}: nondense sequence")
        try:
            tick = int(row["tick"])
            next_tick = int(row["next_tick"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{prefix}: invalid tick")
        else:
            if next_tick == tick + 1:
                valid_successors += 1
            else:
                errors.append(f"{prefix}: nonconsecutive successor")
        if index and transitions[index - 1].get("next_observation_sha256") != row.get(
            "observation_sha256"
        ):
            errors.append(f"{prefix}: broken observation digest chain")
        legal = row.get("legal_actions")
        behavior = row.get("behavior")
        action = behavior.get("selected_action") if isinstance(behavior, Mapping) else None
        if isinstance(legal, list) and action in legal and "bomb" not in legal:
            valid_legal += 1
        else:
            errors.append(f"{prefix}: selected action is not Bomb-free native legal")
        try:
            probability = float(behavior["probability"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            probability = 0.0
        if 0.0 < probability <= 1.0:
            valid_probability += 1
        else:
            errors.append(f"{prefix}: invalid behavior probability")
        outcome = row.get("outcome_terms")
        if isinstance(outcome, Mapping) and outcome.get("bombs_used_delta") == 0:
            valid_bombless += 1
        else:
            errors.append(f"{prefix}: Bomb delta is not zero")
        if isinstance(row.get("source_context"), str):
            contexts.add(row["source_context"])
        if isinstance(action, str):
            actions[action] += 1

    expected_anchor_digests: dict[int, str] = {}
    for index, row in enumerate(transitions):
        digest = row.get("observation_sha256")
        if isinstance(digest, str):
            expected_anchor_digests[index] = digest
    if transitions and isinstance(transitions[-1].get("next_observation_sha256"), str):
        expected_anchor_digests[len(transitions)] = transitions[-1]["next_observation_sha256"]
    valid_anchors = 0
    for index, anchor in enumerate(anchors):
        prefix = f"anchor {index}"
        observation = anchor.get("observation")
        if not isinstance(observation, Mapping):
            errors.append(f"{prefix}: observation missing")
            continue
        digest = canonical_observation_sha256(observation)
        sequence = anchor.get("sequence")
        if anchor.get("observation_sha256") != digest:
            errors.append(f"{prefix}: self digest mismatch")
            continue
        if not isinstance(sequence, int) or expected_anchor_digests.get(sequence) != digest:
            errors.append(f"{prefix}: transition digest does not reference anchor")
            continue
        if int(observation.get("input", 0)) & 0x02:
            errors.append(f"{prefix}: Bomb input observed")
            continue
        valid_anchors += 1

    rows = len(transitions)
    return {
        "run": str(run),
        "valid": not errors,
        "errors": errors,
        "scope": manifest.get("scope"),
        "initial_seed": manifest.get("initial_seed"),
        "termination_reason": manifest.get("termination_reason"),
        "rows": rows,
        "factual_successor_rows": valid_successors,
        "native_legal_rows": valid_legal,
        "valid_propensity_rows": valid_probability,
        "bombless_rows": valid_bombless,
        "valid_anchors": valid_anchors,
        "anchors": len(anchors),
        "unique_source_contexts": len(contexts),
        "selected_action_counts": dict(sorted(actions.items())),
        "compressed_bytes": sum(
            path.stat().st_size for path in (transitions_path, anchors_path) if path.is_file()
        ),
    }


def _run_directories(paths: Iterable[Path]) -> tuple[Path, ...]:
    result = []
    for path in paths:
        if (path / "manifest.json").is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(item.parent for item in path.rglob("manifest.json")))
    return tuple(dict.fromkeys(item.resolve() for item in result))


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sum(int(run.get("rows", 0)) for run in runs)
    field_totals = {
        name: sum(int(run.get(name, 0)) for run in runs)
        for name in (
            "factual_successor_rows",
            "native_legal_rows",
            "valid_propensity_rows",
            "bombless_rows",
        )
    }
    return {
        "schema": "th06-rl-headless-corpus-audit-v1",
        "runs": len(runs),
        "valid_runs": sum(run.get("valid") is True for run in runs),
        "rows": rows,
        **field_totals,
        "factual_successor_ratio": field_totals["factual_successor_rows"] / rows if rows else 0.0,
        "native_legal_ratio": field_totals["native_legal_rows"] / rows if rows else 0.0,
        "valid_propensity_ratio": field_totals["valid_propensity_rows"] / rows if rows else 0.0,
        "bombless_ratio": field_totals["bombless_rows"] / rows if rows else 0.0,
        "compressed_bytes": sum(int(run.get("compressed_bytes", 0)) for run in runs),
        "terminations": dict(sorted(Counter(
            str(run.get("termination_reason")) for run in runs
        ).items())),
        "run_results": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directories = _run_directories(args.paths)
    if not directories:
        parser.error("no compact headless corpus manifests found")
    result = summarize([audit_run(path) for path in directories])
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["valid_runs"] == result["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
