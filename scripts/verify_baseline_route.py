#!/usr/bin/env python3
"""Verify one complete natural-RNG baseline route and factual corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from th06_rl.th06.control_capture import OFFLINE_FACT_SCHEMA


_CLEAN_OUTCOME_FIELDS = (
    "background_reactivations",
    "capture_failures",
    "corpus_failures",
    "infrastructure_failures",
    "policy_failures",
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
    dataset = audit.get("episode_dataset_admission")
    player = audit.get("player_successor_parity")
    shield = audit.get("dense_shield_parity")
    named = (
        ("controller completion", completion),
        ("trace", trace),
        ("run metadata", metadata),
        ("run outcome", outcome),
        ("episode", episode),
        ("records", records),
        ("summary", summary),
        ("audit scope", audit_scope),
        ("episode dataset admission", dataset),
        ("player successor parity", player),
        ("observed shield replay", shield),
    )
    for name, value in named:
        if not isinstance(value, dict):
            raise ValueError(f"baseline route is missing {name}")

    online_contract = (
        metadata.get("online_contract") if isinstance(metadata, dict) else None
    )
    if not isinstance(online_contract, dict):
        raise ValueError("baseline route is missing online contract")
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
        "observed_shield_contract": (
            online_contract.get("algorithm")
            == "observed-shield4-paused-publication-v1"
            and online_contract.get("shield_contract")
            == "observed-hazard-kinematics-v1"
            and online_contract.get("publication_epoch")
            == "coherent-root-process-suspended-v1"
            and online_contract.get("shield_horizon") == 4
            and online_contract.get("predicts_future_births") is False
            and online_contract.get("minimum_collision_margin") == 0.35
            and trace.get("zero_margin_frames") == 0
            and trace.get("invalid_shield_collision_margin_frames") == 0
        ),
        "physical_facts": (
            online_contract.get("factual_state_schema") == OFFLINE_FACT_SCHEMA
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
        ),
        "algorithm_independent_episode": (
            dataset.get("passes") is True
            and dataset.get("checked_frames") == frames
            and dataset.get("checked_transitions") == transitions
            and dataset.get("error") is None
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
        "player_input_successors": (
            player.get("method") == "contiguous-player-center-successor-v1"
            and player.get("arithmetic_comparison") == "float32-bit-exact"
            and player.get("input_semantics")
            == "next-completed-root-sampled-input"
            and player.get("checked_links", 0) > 0
            and player.get("mismatches") == 0
        ),
        "observed_shield_replay": (
            shield.get("method") == "stored-observed-primitives-native-replay-v1"
            and shield.get("checked", 0) > 0
            and shield.get("unsafe_divergences") == []
            and shield.get("conservative_divergences") == []
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("baseline route verification failed: " + ", ".join(failed))
    return {
        "schema": "th06-rl-baseline-route-verification-v2",
        "passed": True,
        "physical_hits": hits,
        "frames": frames,
        "transitions": transitions,
        "hit_classifications": audit.get("hit_classifications"),
        "latency": audit.get("latency"),
        "player_successor_parity": player,
        "dense_shield_parity": shield,
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
        result = verify(*(
            _object(path)
            for path in (args.report, args.run, args.manifest, args.audit)
        ))
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
