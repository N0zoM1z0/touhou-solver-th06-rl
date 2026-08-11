#!/usr/bin/env python3
"""Audit repeated Stage 6 failure regions with one vote per physical Wine run.

The audit does not fit a model.  It validates the prior exact incumbent-action
replay, loads only strict original-retail first-failure prefixes, reduces each
terminal episode to fixed generic bins, and reports repeated residual
opportunities without treating adjacent positive frames as independent runs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

from th06_rl.wine_risk import FirstFailurePrefix, load_first_failure_prefix


AUDIT_SCHEMA = "th06-rl-wine-episode-failure-region-audit-v1"
ACTION_AUDIT_SCHEMA = "th06-rl-wine-risk-consensus-replay-v1"
SUMMARY_FEATURES = (
    "player_x",
    "player_y",
    "edge_reserve",
    "bullet_count",
    "laser_count",
    "hard_action_count",
    "legal_action_count",
    "incumbent_hard_clearance",
    "baseline_hard_clearance",
    "clearance_delta_baseline",
    "incumbent_clearance_rank",
    "incumbent_final_edge_reserve",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _count_events(sequences: Iterable[int]) -> int:
    events = 0
    previous: int | None = None
    for sequence in sorted(set(sequences)):
        if previous is None or sequence != previous + 1:
            events += 1
        previous = sequence
    return events


def _boundary_regime(features: Mapping[str, str | float]) -> str:
    return "boundary" if float(features["edge_reserve"]) <= 16.0 else "interior"


def _hazard_regime(features: Mapping[str, str | float]) -> str:
    if float(features["laser_count"]) > 0.0:
        return "lasers-present"
    bullets = float(features["bullet_count"])
    if bullets >= 384.0:
        return "dense-bullets"
    if bullets >= 128.0:
        return "medium-bullets"
    return "light-bullets"


def _safe_set_regime(features: Mapping[str, str | float]) -> str:
    count = float(features["hard_action_count"])
    if count <= 2.0:
        return "critical-1-2"
    if count <= 4.0:
        return "narrow-3-4"
    return "broad-5-plus"


def _clearance_regime(features: Mapping[str, str | float]) -> str:
    clearance = float(features["incumbent_hard_clearance"])
    if clearance <= 0.5:
        return "at-most-0.5"
    if clearance <= 1.0:
        return "0.5-to-1"
    if clearance <= 2.0:
        return "1-to-2"
    if clearance <= 4.0:
        return "2-to-4"
    return "over-4"


def _boundary_sides(features: Mapping[str, str | float]) -> tuple[str, ...]:
    x = float(features["player_x"])
    y = float(features["player_y"])
    sides = []
    if x <= 24.0:
        sides.append("left")
    if x >= 360.0:
        sides.append("right")
    if y <= 32.0:
        sides.append("top")
    if y >= 416.0:
        sides.append("bottom")
    return tuple(sides) or ("interior",)


def terminal_family(features: Mapping[str, str | float]) -> str:
    return "/".join((_boundary_regime(features), _hazard_regime(features)))


def window_atom(features: Mapping[str, str | float]) -> str:
    return "/".join(
        (_boundary_regime(features), _hazard_regime(features), _safe_set_regime(features))
    )


def opportunity_region(features: Mapping[str, str | float]) -> str:
    return "/".join(
        (
            window_atom(features),
            f"incumbent={features['action']}",
            f"baseline={features['baseline_action']}",
        )
    )


def _feature_ranges(examples) -> dict[str, dict[str, float]]:
    result = {}
    for name in SUMMARY_FEATURES:
        values = [float(example.features[name]) for example in examples]
        result[name] = {
            "minimum": min(values),
            "median": float(statistics.median(values)),
            "maximum": max(values),
        }
    return result


def _episode(prefix: FirstFailurePrefix, *, policy_calls: int) -> dict[str, Any]:
    positives = sorted(
        (example for example in prefix.examples if example.failure_within_120),
        key=lambda example: example.transition.sequence,
    )
    closest = min(
        positives,
        key=lambda example: (
            example.frames_to_failure if example.frames_to_failure is not None else 10**9,
            -example.transition.sequence,
        ),
    )
    opportunities = [example for example in positives if example.fallback_opportunity]
    terminal = closest.features
    return {
        "run_id": prefix.run_id,
        "path": str(prefix.run_dir),
        "collection_stratum": prefix.run_dir.parent.name,
        "manifest_sha256": prefix.manifest_sha256,
        "run_sha256": prefix.run_sha256,
        "code_commit": prefix.code_commit,
        "failure_kind": prefix.failure_kind,
        "failure_frame": prefix.failure_frame,
        "failure_context": prefix.failure_context,
        "failure_segment_start_frame": prefix.failure_segment_start_frame,
        "positive_window_start_frame": prefix.positive_window_start_frame,
        "transitions": prefix.transitions,
        "policy_calls_in_action_audit": policy_calls,
        "eligible_policy_rows": len(prefix.examples),
        "positive_rows": len(positives),
        "positive_row_inflation_vs_episode": len(positives),
        "fallback_opportunity_rows": len(opportunities),
        "fallback_opportunity_events": _count_events(
            example.transition.sequence for example in opportunities
        ),
        "closest_policy_lag_frames": closest.frames_to_failure,
        "terminal_family": terminal_family(terminal),
        "terminal_signature": "/".join(
            (
                terminal_family(terminal),
                _safe_set_regime(terminal),
                _clearance_regime(terminal),
                "sides=" + "+".join(_boundary_sides(terminal)),
            )
        ),
        "terminal_features": {name: terminal[name] for name in SUMMARY_FEATURES},
        "terminal_actions": {
            "incumbent": terminal["action"],
            "baseline": terminal["baseline_action"],
            "current": terminal["current_action"],
        },
        "positive_window_feature_ranges": _feature_ranges(positives),
        "window_atoms": sorted({window_atom(example.features) for example in positives}),
        "opportunity_regions": sorted(
            {opportunity_region(example.features) for example in opportunities}
        ),
    }


def aggregate_memberships(
    episodes: list[dict[str, Any]],
    *,
    membership_key: str,
    plural: bool,
) -> list[dict[str, Any]]:
    support: dict[str, set[str]] = defaultdict(set)
    contexts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    row_occurrences: Counter[str] = Counter()
    opportunities: dict[str, set[str]] = defaultdict(set)
    episode_by_run = {str(episode["run_id"]): episode for episode in episodes}
    for episode in episodes:
        values = episode[membership_key]
        memberships = set(values) if plural else {str(values)}
        for region in memberships:
            region = str(region)
            run_id = str(episode["run_id"])
            support[region].add(run_id)
            contexts[region][str(episode["failure_context"])].add(run_id)
            if int(episode["fallback_opportunity_rows"]) > 0:
                opportunities[region].add(run_id)
    # Row occurrence is intentionally secondary.  It never changes support.
    for episode in episodes:
        if membership_key == "window_atoms":
            for region in episode["window_atoms"]:
                row_occurrences[str(region)] += 1
        elif membership_key == "opportunity_regions":
            for region in episode["opportunity_regions"]:
                row_occurrences[str(region)] += 1
        else:
            row_occurrences[str(episode[membership_key])] += 1
    rows = []
    for region, runs in support.items():
        ordered_runs = sorted(runs)
        failure_frames = [int(episode_by_run[run]["failure_frame"]) for run in ordered_runs]
        rows.append(
            {
                "region": region,
                "episode_support": len(ordered_runs),
                "classification": "repeated" if len(ordered_runs) >= 2 else "singleton",
                "runs": ordered_runs,
                "contexts": {
                    context: len(context_runs)
                    for context, context_runs in sorted(contexts[region].items())
                },
                "episodes_with_any_fallback_opportunity": len(opportunities[region]),
                "membership_occurrences_after_per_episode_dedup": row_occurrences[region],
                "failure_frame_minimum": min(failure_frames),
                "failure_frame_median": float(statistics.median(failure_frames)),
                "failure_frame_maximum": max(failure_frames),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["episode_support"]), str(row["region"])))


def _validate_action_audit(
    path: Path,
    *,
    scope: tuple[int, int, int, int],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    audit = _object(path)
    totals = audit.get("totals")
    runs = audit.get("runs")
    if (
        audit.get("schema") != ACTION_AUDIT_SCHEMA
        or audit.get("passed") is not True
        or audit.get("mode") != "shadow"
        or audit.get("expect_recorded_actions") != "incumbent"
        or audit.get("scope") != list(scope)
        or not isinstance(totals, dict)
        or not isinstance(runs, list)
    ):
        raise ValueError("factual action audit contract is invalid")
    for name in (
        "recorded_incumbent_mismatches",
        "recorded_policy_mismatches",
        "shadow_action_contract_violations",
    ):
        if int(totals.get(name, -1)) != 0:
            raise ValueError(f"factual action audit has nonzero {name}")
    indexed: dict[str, dict[str, Any]] = {}
    policy_calls = 0
    for raw in runs:
        if not isinstance(raw, dict):
            raise TypeError("factual action audit run row is invalid")
        run_id = str(raw.get("run_id", ""))
        if not run_id or run_id in indexed:
            raise ValueError("factual action audit run identity is invalid")
        for name in (
            "recorded_incumbent_mismatches",
            "recorded_policy_mismatches",
            "shadow_action_contract_violations",
        ):
            if raw.get(name) != []:
                raise ValueError(f"factual action audit run has {name}")
        indexed[run_id] = raw
        policy_calls += int(raw.get("policy_calls", -1))
    if (
        int(totals.get("runs", -1)) != len(indexed)
        or int(totals.get("policy_calls", -1)) != policy_calls
    ):
        raise ValueError("factual action audit totals do not match run rows")
    return audit, indexed


def _index_runs(corpus_roots: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in corpus_roots:
        if not root.is_dir():
            raise NotADirectoryError(root)
        for path in root.iterdir():
            if not path.is_dir() or not (path / "manifest.json").is_file():
                continue
            if path.name in result:
                raise ValueError(f"duplicate corpus run identity: {path.name}")
            result[path.name] = path.resolve()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factual-action-audit", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scope", default="3/0/0/6")
    parser.add_argument("--retail-sha256", required=True)
    parser.add_argument("--native-sha256", required=True)
    args = parser.parse_args(argv)
    scope = tuple(int(value) for value in args.scope.split("/"))
    if len(scope) != 4:
        parser.error("scope must contain four integers")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        action_audit, audit_runs = _validate_action_audit(
            args.factual_action_audit.resolve(), scope=scope  # type: ignore[arg-type]
        )
        indexed_paths = _index_runs([path.resolve() for path in args.corpus_root])
        missing = sorted(set(audit_runs) - set(indexed_paths))
        if missing:
            raise ValueError(f"action-audited runs are absent from corpus roots: {missing}")
        prefixes = []
        for run_id, audit_row in audit_runs.items():
            prefix = load_first_failure_prefix(
                indexed_paths[run_id],
                expected_scope=scope,  # type: ignore[arg-type]
                expected_executable_sha256=args.retail_sha256,
                expected_native_kernel_sha256=args.native_sha256,
                expected_policy_id=None,
            )
            if (
                prefix.manifest_sha256 != audit_row.get("manifest_sha256")
                or prefix.run_sha256 != audit_row.get("run_sha256")
            ):
                raise ValueError(f"action audit corpus identity mismatch: {run_id}")
            prefixes.append(prefix)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    episodes = [
        _episode(prefix, policy_calls=int(audit_runs[prefix.run_id]["policy_calls"]))
        for prefix in prefixes
    ]
    episodes.sort(key=lambda episode: str(episode["run_id"]))
    terminal_families = aggregate_memberships(
        episodes, membership_key="terminal_family", plural=False
    )
    contexts = aggregate_memberships(
        episodes, membership_key="failure_context", plural=False
    )
    window_atoms = aggregate_memberships(
        episodes, membership_key="window_atoms", plural=True
    )
    opportunity_regions = aggregate_memberships(
        episodes, membership_key="opportunity_regions", plural=True
    )
    totals = action_audit["totals"]
    report = {
        "schema": AUDIT_SCHEMA,
        "scope": list(scope),
        "evidence_boundary": (
            "episode-grouped original-retail Wine first-failure audit; fixed bins are "
            "hypothesis-generation metadata, not online control or promotion evidence"
        ),
        "bin_contract": {
            "boundary": "edge_reserve <= 16",
            "hazards": {
                "lasers-present": "laser_count > 0",
                "dense-bullets": "no lasers and bullet_count >= 384",
                "medium-bullets": "no lasers and 128 <= bullet_count < 384",
                "light-bullets": "no lasers and bullet_count < 128",
            },
            "native_safe_set": {
                "critical-1-2": "hard_action_count <= 2",
                "narrow-3-4": "3 <= hard_action_count <= 4",
                "broad-5-plus": "hard_action_count >= 5",
            },
            "support_unit": "one physical run contributes at most one vote per region",
            "repeated_minimum_episode_support": 2,
        },
        "factual_action_audit": {
            "path": str(args.factual_action_audit.resolve()),
            "sha256": _sha256(args.factual_action_audit.resolve()),
            "schema": action_audit["schema"],
            "passed": action_audit["passed"],
            "state_sha256": action_audit["state_sha256"],
            "policy_calls": totals["policy_calls"],
            "runs": totals["runs"],
            "recorded_incumbent_mismatches": totals["recorded_incumbent_mismatches"],
            "recorded_policy_mismatches": totals["recorded_policy_mismatches"],
            "shadow_action_contract_violations": totals["shadow_action_contract_violations"],
        },
        "totals": {
            "episodes": len(episodes),
            "eligible_policy_rows": sum(int(episode["eligible_policy_rows"]) for episode in episodes),
            "positive_frame_rows": sum(int(episode["positive_rows"]) for episode in episodes),
            "effective_independent_positive_units": len(episodes),
            "fallback_opportunity_rows": sum(
                int(episode["fallback_opportunity_rows"]) for episode in episodes
            ),
            "fallback_opportunity_events": sum(
                int(episode["fallback_opportunity_events"]) for episode in episodes
            ),
            "episodes_with_fallback_opportunity": sum(
                int(episode["fallback_opportunity_rows"]) > 0 for episode in episodes
            ),
            "failure_kinds": dict(sorted(Counter(
                str(episode["failure_kind"]) for episode in episodes
            ).items())),
            "collection_strata": dict(sorted(Counter(
                str(episode["collection_stratum"]) for episode in episodes
            ).items())),
            "repeated_context_episodes": sum(
                int(region["episode_support"])
                for region in contexts if region["classification"] == "repeated"
            ),
            "singleton_context_episodes": sum(
                int(region["episode_support"])
                for region in contexts if region["classification"] == "singleton"
            ),
        },
        "episodes": episodes,
        "regions": {
            "failure_contexts": contexts,
            "terminal_generic_families": terminal_families,
            "positive_window_atoms": window_atoms,
            "native_baseline_opportunities": opportunity_regions,
        },
        "next_gate": (
            "targeted multi-seed headless COW may inspect only repeated native-baseline "
            "opportunity regions; this audit alone authorizes no residual or active action"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "episodes": len(episodes),
                "positive_frame_rows": report["totals"]["positive_frame_rows"],
                "repeated_context_episodes": report["totals"]["repeated_context_episodes"],
                "repeated_opportunity_regions": sum(
                    region["classification"] == "repeated" for region in opportunity_regions
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
