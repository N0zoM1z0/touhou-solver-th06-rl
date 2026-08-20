#!/usr/bin/env python3
"""Verify one complete natural-RNG baseline route and its factual corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_CLEAN_OUTCOME_FIELDS = (
    "background_reactivations",
    "capture_failures",
    "corpus_failures",
    "infrastructure_failures",
    "trace_failures",
)


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def verify(
    report: dict[str, object],
    run: dict[str, object],
    manifest: dict[str, object],
    audit: dict[str, object],
) -> dict[str, object]:
    completion = report.get("controller_completion")
    trace = report.get("trace")
    metadata = run.get("metadata")
    outcome = manifest.get("run_outcome")
    episode = manifest.get("episode")
    records = manifest.get("records")
    summary = manifest.get("summary")
    audit_scope = audit.get("scope")
    successor_coverage = audit.get("source_successor_coverage")
    for name, value in (
        ("controller completion", completion),
        ("trace", trace),
        ("run metadata", metadata),
        ("run outcome", outcome),
        ("episode", episode),
        ("records", records),
        ("summary", summary),
        ("audit scope", audit_scope),
        ("source successor coverage", successor_coverage),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"baseline route is missing {name}")

    hits = completion.get("physical_hits")
    frames = records.get("frames")
    transitions = records.get("transitions")
    checks = {
        "runner_clean": (
            report.get("error") is None
            and report.get("controller_returncode") == 0
            and report.get("gdb_normalized") is True
        ),
        "route_completed": completion.get("route_completed") is True,
        "natural_rng": report.get("diagnostic_rng_seed") is None,
        "immutable_policy": report.get("immutable_policy_state_equal") is True,
        "exact_cleanup": report.get("leftover_prefix_processes") == [],
        "route_scope": (
            metadata.get("episode_unit") == "route"
            and metadata.get("expected_stages") == [1, 2, 3, 4, 5, 6]
            and audit_scope.get("observed_stages") == [1, 2, 3, 4, 5, 6]
        ),
        "source_complete_online_authority": (
            isinstance(metadata.get("planner"), dict)
            and metadata["planner"].get("source_commitment")
            == "source-complete-hard-v1"
        ),
        "comprehensive_offline_facts": (
            isinstance(metadata.get("planner"), dict)
            and metadata["planner"].get("factual_state_schema")
            == "th06-1.02h-offline-facts-v1"
        ),
        "durable_complete": (
            manifest.get("complete") is True
            and manifest.get("stage_trajectory_complete") is True
            and manifest.get("dropped_records") == 0
            and episode.get("unit") == "route"
            and episode.get("complete") is True
            and outcome.get("stage_completed") is True
            and outcome.get("termination_reason") == "route-complete"
        ),
        "factual_streams": (
            isinstance(frames, int)
            and isinstance(transitions, int)
            and frames > 1
            and transitions == frames - 1
            and isinstance(records.get("anchors"), int)
            and records["anchors"] > 0
            and isinstance(summary.get("learning_eligible_transitions"), int)
            and summary["learning_eligible_transitions"] > 0
        ),
        "zero_bomb": audit.get("bomb_events", 0) == 0,
        "zero_infra_failure": (
            outcome.get("corpus_failure") is None
            and all(outcome.get(field) == 0 for field in _CLEAN_OUTCOME_FIELDS)
        ),
        "hit_conservation": (
            isinstance(hits, int)
            and hits == trace.get("physical_hits_in_run")
            and hits == outcome.get("physical_hits")
            and hits == audit.get("physical_hits")
        ),
        "audit_integrity": audit.get("integrity_errors") == [],
        "causal_source_successors": (
            successor_coverage.get("method")
            == "retained-next-root-one-sided-coverage-v1"
            and isinstance(successor_coverage.get("checked_links"), int)
            and successor_coverage["checked_links"] > 0
            and successor_coverage.get("uncovered_aabbs") == 0
            and successor_coverage.get("uncovered_lasers") == 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("baseline route verification failed: " + ", ".join(failed))
    return {
        "schema": "th06-rl-baseline-route-verification-v1",
        "passed": True,
        "physical_hits": hits,
        "frames": frames,
        "transitions": transitions,
        "anchors": records["anchors"],
        "learning_eligible_transitions": summary["learning_eligible_transitions"],
        "hit_classifications": audit.get("hit_classifications"),
        "latency": audit.get("latency"),
        "source_successor_coverage": successor_coverage,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audit", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(*(_object(path) for path in (
            args.report, args.run, args.manifest, args.audit
        )))
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
