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
    numeric_successor_parity = audit.get("source_numeric_successor_parity")
    anchor_coverage = audit.get("source_anchor_coverage")
    dataset_admission = audit.get("source_dataset_admission")
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
        ("source numeric successor parity", numeric_successor_parity),
        ("source anchor coverage", anchor_coverage),
        ("source dataset admission", dataset_admission),
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
            and report.get("repository_worktree_clean") is True
            and report.get("repository_commit") == metadata.get("code_commit")
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
            and metadata["planner"].get("algorithm")
            == "source-hard4-paused-publication-v2"
            and metadata["planner"].get("source_commitment")
            == "source-complete-hard-v1"
            and metadata["planner"].get("hard_horizon") == 4
            and metadata["planner"].get("learner_feature_horizon") == 4
            and metadata["planner"].get("minimum_collision_margin") == 0.35
            and metadata["planner"].get("zero_margin_fallback") is False
            and trace.get("zero_margin_frames") == 0
            and trace.get("invalid_hard_collision_margin_frames") == 0
        ),
        "comprehensive_offline_facts": (
            isinstance(metadata.get("planner"), dict)
            and metadata["planner"].get("factual_state_schema")
            == "th06-1.02h-offline-facts-v2"
        ),
        "stage_local_source_anchors": (
            anchor_coverage.get("anchored_stages") == [1, 2, 3, 4, 5, 6]
            and anchor_coverage.get("missing_observed_stages") == []
        ),
        "self_contained_source_dataset": (
            dataset_admission.get("passes") is True
            and dataset_admission.get("checked_frames") == frames
            and dataset_admission.get("error") is None
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
            and successor_coverage.get("actual_lasers_checked", 0) > 0
            and successor_coverage.get("uncovered_aabbs") == 0
            and successor_coverage.get("uncovered_lasers") == 0
            and (successor_coverage.get(
                "retained_laser_geometry_unavailable", {}
            ) or {}).get("invalid-state", 0) == 0
        ),
        "numeric_source_successors": (
            numeric_successor_parity.get("method")
            == "stable-retained-bullet-center-successor-v2"
            and numeric_successor_parity.get("arithmetic_comparison")
            == "float32-bit-exact"
            and numeric_successor_parity.get("required_collision_margin") == 0.35
            and isinstance(
                numeric_successor_parity.get("transcendental_axis_error_budget"),
                (int, float),
            )
            and 0
            < numeric_successor_parity["transcendental_axis_error_budget"]
            < numeric_successor_parity["required_collision_margin"]
            and 0
            < numeric_successor_parity.get(
                "global_release_acceleration_axis_bound", 1.0
            )
            < numeric_successor_parity["required_collision_margin"]
            and numeric_successor_parity.get("global_mutation_semantics")
            == "source branch union"
            and (
                numeric_successor_parity.get("linear_exact_checked", 0)
                + numeric_successor_parity.get("acceleration_exact_checked", 0)
            ) > 0
            and numeric_successor_parity.get("transcendental_checked", 0) > 0
            and (
                numeric_successor_parity.get("global_stop_union_checked", 0)
                + numeric_successor_parity.get("global_release_union_checked", 0)
                + numeric_successor_parity.get("global_combined_union_checked", 0)
            ) > 0
            and numeric_successor_parity.get("exact_mismatches") == 0
            and numeric_successor_parity.get(
                "transcendental_budget_violations"
            ) == 0
            and numeric_successor_parity.get("nonfinite_successors") == 0
            and numeric_successor_parity.get(
                "global_mutation_union_violations"
            ) == 0
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
        "source_numeric_successor_parity": numeric_successor_parity,
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
