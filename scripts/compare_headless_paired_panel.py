#!/usr/bin/env python3
"""Compare complete headless policies on one exact paired seed panel.

Rolling unseen seeds measure robustness, but they do not identify a policy
effect when seed difficulty differs.  This tool deliberately accepts only
complete, natural HIT-continuation runs whose seed set, learning scope, and
authoritative headless source are identical for every candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from summarize_headless_continuation import summarize
except ModuleNotFoundError:  # Imported as scripts.compare_headless_paired_panel.
    from scripts.summarize_headless_continuation import summarize


NATURAL_TERMINATIONS = frozenset({"chain-exit-success", "stage-clear-success"})


def _source_key(source: Mapping[str, Any]) -> tuple[str, str]:
    return str(source.get("commit", "")), str(source.get("binary_sha256", ""))


def _candidate(label: str, root: Path) -> dict[str, Any]:
    summary = summarize([root])
    if summary["partial_runs"]:
        raise ValueError(f"{label}: paired evidence contains interrupted partial runs")
    runs = summary["run_results"]
    if len(runs) < 2:
        raise ValueError(f"{label}: paired evidence requires at least two complete seeds")

    by_seed: dict[int, dict[str, Any]] = {}
    sources: set[tuple[str, str]] = set()
    rankers: set[str] = set()
    scopes: set[str] = set()
    for run in runs:
        seed = run.get("seed")
        if not isinstance(seed, int):
            raise ValueError(f"{label}: run has no integer seed: {run['run']}")
        if seed in by_seed:
            raise ValueError(f"{label}: duplicate completed seed {seed}")
        if run.get("continue_after_hit") is not True:
            raise ValueError(f"{label}: seed {seed} is not a HIT-continuation run")
        if run.get("termination_reason") not in NATURAL_TERMINATIONS:
            raise ValueError(
                f"{label}: seed {seed} did not finish naturally: "
                f"{run.get('termination_reason')}"
            )
        if int(run.get("bombs_used", 0)) != 0:
            raise ValueError(f"{label}: seed {seed} used a Bomb")

        manifest_path = Path(run["run"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("transaction_complete") is not True:
            raise ValueError(f"{label}: seed {seed} manifest is not transaction-complete")
        source = manifest.get("source")
        if not isinstance(source, Mapping) or not all(_source_key(source)):
            raise ValueError(f"{label}: seed {seed} lacks authoritative source identity")
        sources.add(_source_key(source))
        ranker = run.get("ranker_sha256")
        if not isinstance(ranker, str) or not ranker:
            raise ValueError(f"{label}: seed {seed} lacks ranker identity")
        rankers.add(ranker)
        scopes.add(json.dumps(run.get("scope", {}), sort_keys=True))
        by_seed[seed] = {
            "physical_hits": int(run["physical_hits"]),
            "benchmark_forced_rows": int(run["benchmark_forced_rows"]),
            "rows": int(run["rows"]),
            "termination_reason": run["termination_reason"],
            "strict_nmnb": bool(
                int(run["physical_hits"]) == 0
                and int(run["benchmark_forced_rows"]) == 0
            ),
        }
    if len(sources) != 1:
        raise ValueError(f"{label}: runs mix authoritative source identities")
    if len(rankers) != 1:
        raise ValueError(f"{label}: runs mix ranker identities")
    if len(scopes) != 1:
        raise ValueError(f"{label}: runs mix learning scopes")
    return {
        "label": label,
        "root": str(root.resolve()),
        "ranker_sha256": next(iter(rankers)),
        "source": {
            "commit": next(iter(sources))[0],
            "binary_sha256": next(iter(sources))[1],
        },
        "scope": json.loads(next(iter(scopes))),
        "seeds": sorted(by_seed),
        "physical_hits": sum(row["physical_hits"] for row in by_seed.values()),
        "benchmark_forced_rows": sum(
            row["benchmark_forced_rows"] for row in by_seed.values()
        ),
        "strict_nmnb_seeds": sum(row["strict_nmnb"] for row in by_seed.values()),
        "per_seed": {str(seed): by_seed[seed] for seed in sorted(by_seed)},
    }


def _dominance(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    aggregate_weak = (
        left["physical_hits"] <= right["physical_hits"]
        and left["benchmark_forced_rows"] <= right["benchmark_forced_rows"]
    )
    aggregate_strict = (
        left["physical_hits"] < right["physical_hits"]
        or left["benchmark_forced_rows"] < right["benchmark_forced_rows"]
    )
    per_seed_weak = True
    per_seed_strict = False
    hit_wins = 0
    forced_wins = 0
    for seed in left["seeds"]:
        left_row = left["per_seed"][str(seed)]
        right_row = right["per_seed"][str(seed)]
        per_seed_weak &= (
            left_row["physical_hits"] <= right_row["physical_hits"]
            and left_row["benchmark_forced_rows"] <= right_row["benchmark_forced_rows"]
        )
        per_seed_strict |= (
            left_row["physical_hits"] < right_row["physical_hits"]
            or left_row["benchmark_forced_rows"] < right_row["benchmark_forced_rows"]
        )
        hit_wins += left_row["physical_hits"] < right_row["physical_hits"]
        forced_wins += (
            left_row["benchmark_forced_rows"]
            < right_row["benchmark_forced_rows"]
        )
    return {
        "left": left["label"],
        "right": right["label"],
        "aggregate_dominates": aggregate_weak and aggregate_strict,
        "seedwise_dominates": per_seed_weak and per_seed_strict,
        "hit_seed_wins": hit_wins,
        "forced_seed_wins": forced_wins,
    }


def compare(entries: Iterable[tuple[str, Path]]) -> dict[str, Any]:
    candidates = [_candidate(label, path) for label, path in entries]
    if len(candidates) < 2:
        raise ValueError("paired comparison requires at least two candidates")
    labels = [candidate["label"] for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique")
    seed_sets = {tuple(candidate["seeds"]) for candidate in candidates}
    scopes = {json.dumps(candidate["scope"], sort_keys=True) for candidate in candidates}
    sources = {
        (candidate["source"]["commit"], candidate["source"]["binary_sha256"])
        for candidate in candidates
    }
    if len(seed_sets) != 1:
        raise ValueError("candidates do not use the exact same seed panel")
    if len(scopes) != 1:
        raise ValueError("candidates do not use the exact same learning scope")
    if len(sources) != 1:
        raise ValueError("candidates do not use the exact same authoritative source")

    comparisons = [
        _dominance(left, right)
        for left in candidates
        for right in candidates
        if left is not right
    ]
    return {
        "schema": "th06-rl-headless-paired-policy-panel-v1",
        "scope": candidates[0]["scope"],
        "source": candidates[0]["source"],
        "seeds": candidates[0]["seeds"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "comparisons": comparisons,
        "seedwise_dominators": sorted({
            row["left"] for row in comparisons if row["seedwise_dominates"]
        }),
        "promotion_allowed": False,
        "promotion_blocker": (
            "one panel is experiment evidence only; require independent paired "
            "replication and later Windows validation"
        ),
    }


def _entry(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    return label, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="+", type=_entry, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.candidates)
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
