#!/usr/bin/env python3
"""Audit shared state in retail Wine versus replayed portable-source traces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from th06_rl.wine_risk import FROZEN_INCUMBENT_POLICY_ID, load_first_failure_prefix

try:
    from compare_headless_traces import first_difference
    from export_wine_action_stream import _object, _verified_stream_rows
    from run_source_platform_differential import DRIFT_TOLERANCES, _trace_summary
except ModuleNotFoundError:  # Imported as scripts.audit_retail_source_replay.
    from scripts.compare_headless_traces import first_difference
    from scripts.export_wine_action_stream import _object, _verified_stream_rows
    from scripts.run_source_platform_differential import DRIFT_TOLERANCES, _trace_summary


REPORT_SCHEMA = "th06-rl-retail-source-replay-audit-v1"
PLATFORM_REPORT_SCHEMA = "th06-rl-source-platform-differential-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_player(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "x": float(values["x"]),
        "y": float(values["y"]),
        "state": int(values["state"]),
        "half_width": float(values["half_width"]),
        "half_height": float(values["half_height"]),
        "focused": bool(values["focused"]),
    }


def _retail_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    mask = int(snapshot["input_mask"])
    return {
        "frame": int(snapshot["frame"]),
        "scope": {
            name: int(snapshot[name])
            for name in ("difficulty", "character", "shot_type", "stage")
        },
        "rng_seed": int(snapshot["rng_seed"]),
        "rng_generation": int(snapshot["rng_generation"]),
        "input": mask,
        "player": _normalized_player(
            {
                "x": snapshot["x"],
                "y": snapshot["y"],
                "state": snapshot["player_state"],
                "half_width": snapshot["half_width"],
                "half_height": snapshot["half_height"],
                "focused": bool(mask & 0x04),
            }
        ),
        "lives": int(snapshot["lives_remaining"]),
        "power": int(snapshot["current_power"]),
        "rank": int(snapshot["rank"]),
        "timeline_time": int(snapshot["timeline_time"]),
        "bullet_count": int(snapshot["live_bullet_count"]),
        "laser_count": int(snapshot["laser_count"]),
    }


def _source_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    player = observation.get("player")
    context = observation.get("source_context")
    scope = observation.get("scope")
    bullets = observation.get("bullets")
    lasers = observation.get("lasers")
    if (
        not isinstance(player, Mapping)
        or not isinstance(context, Mapping)
        or not isinstance(scope, Mapping)
        or not isinstance(bullets, list)
        or not isinstance(lasers, list)
    ):
        raise TypeError("source replay observation lacks shared-state fields")
    return {
        "frame": int(observation["game_frame"]),
        "scope": {
            name: int(scope[name])
            for name in ("difficulty", "character", "shot_type", "stage")
        },
        "rng_seed": int(observation["rng_seed"]),
        "rng_generation": int(observation["rng_generation"]),
        "input": int(observation["input"]),
        "player": _normalized_player(player),
        "lives": int(observation["lives"]),
        "power": int(observation["power"]),
        "rank": int(observation["rank"]),
        "timeline_time": int(context["timeline_time"]),
        "bullet_count": len(bullets),
        "laser_count": len(lasers),
    }


def _record(frame: int, difference: Mapping[str, Any]) -> dict[str, Any]:
    return {"frame": frame, "difference": dict(difference)}


def compare_retail_source_states(
    frame_rows: Sequence[Mapping[str, Any]],
    trace_path: Path,
    *,
    require_full_coverage: bool = True,
) -> dict[str, Any]:
    retail_by_frame: dict[int, dict[str, Any]] = {}
    for expected_sequence, row in enumerate(frame_rows):
        if row.get("sequence") != expected_sequence:
            raise ValueError("retail frame rows are not sequence-contiguous")
        snapshot = row.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise TypeError(f"retail frame {expected_sequence} lacks a snapshot")
        state = _retail_state(snapshot)
        frame = state["frame"]
        if frame in retail_by_frame:
            raise ValueError(f"duplicate retail snapshot frame {frame}")
        retail_by_frame[frame] = state

    first_exact: dict[str, Any] | None = None
    first_by_tolerance: dict[float, dict[str, Any] | None] = {
        tolerance: None for tolerance in DRIFT_TOLERANCES
    }
    categories = {
        "discrete_delivery": (
            "frame", "scope", "rng_seed", "rng_generation", "input",
            "lives", "power", "rank", "timeline_time", "bullet_count", "laser_count",
        ),
        "rng": ("rng_seed", "rng_generation"),
        "input": ("input",),
        "game": ("lives", "power", "rank", "timeline_time"),
        "hazard_counts": ("bullet_count", "laser_count"),
        "player_discrete": (),
        "player_at_1e_6": (),
        "player_geometry_at_1e_6": (),
    }
    first_category: dict[str, dict[str, Any] | None] = {
        name: None for name in categories
    }
    matched_frames: list[int] = []
    seen_trace_ticks: set[int] = set()
    with trace_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                observation = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid source replay JSON at {trace_path}:{line_number}: {error}"
                ) from error
            if not isinstance(observation, dict):
                raise TypeError(f"source replay row {line_number} is not an object")
            tick = observation.get("tick")
            if type(tick) is not int or tick in seen_trace_ticks:
                raise ValueError(f"invalid source replay tick at line {line_number}")
            seen_trace_ticks.add(tick)
            retail = retail_by_frame.get(tick)
            if retail is None:
                continue
            source_state = _source_state(observation)
            matched_frames.append(tick)
            if first_exact is None:
                difference = first_difference(retail, source_state)
                if difference is not None:
                    first_exact = _record(tick, difference)
            for tolerance in first_by_tolerance:
                if first_by_tolerance[tolerance] is not None:
                    continue
                difference = first_difference(
                    retail,
                    source_state,
                    absolute_tolerance=tolerance,
                )
                if difference is not None:
                    first_by_tolerance[tolerance] = _record(tick, difference)
            for name, fields in categories.items():
                if first_category[name] is not None:
                    continue
                if name == "player_discrete":
                    left = {
                        key: retail["player"][key]
                        for key in ("state", "focused")
                    }
                    right = {
                        key: source_state["player"][key]
                        for key in ("state", "focused")
                    }
                    tolerance = 0.0
                elif name == "player_at_1e_6":
                    left = retail["player"]
                    right = source_state["player"]
                    tolerance = 1e-6
                elif name == "player_geometry_at_1e_6":
                    geometry = ("x", "y", "state", "half_width", "half_height")
                    left = {key: retail["player"][key] for key in geometry}
                    right = {key: source_state["player"][key] for key in geometry}
                    tolerance = 1e-6
                else:
                    left = {field: retail[field] for field in fields}
                    right = {field: source_state[field] for field in fields}
                    tolerance = 0.0
                difference = first_difference(
                    left,
                    right,
                    absolute_tolerance=tolerance,
                )
                if difference is not None:
                    first_category[name] = _record(tick, difference)

    missing = sorted(set(retail_by_frame) - set(matched_frames))
    if missing and require_full_coverage:
        raise ValueError(
            f"source replay does not cover {len(missing)} retail snapshots; first {missing[0]}"
        )
    common = len(matched_frames)
    category_results = {
        name: {
            "equal": difference is None,
            "first_divergence": difference,
        }
        for name, difference in first_category.items()
    }
    return {
        "retail_snapshots": len(retail_by_frame),
        "common_snapshots": common,
        "first_retail_frame": min(retail_by_frame),
        "last_retail_frame": max(retail_by_frame),
        "missing_retail_snapshots": len(missing),
        "first_missing_retail_frame": missing[0] if missing else None,
        "exact_shared_state": {
            "equal": first_exact is None,
            "first_divergence": first_exact,
        },
        "tolerance_ladder": [
            {
                "absolute_tolerance": tolerance,
                "equal": difference is None,
                "first_divergence": difference,
            }
            for tolerance, difference in first_by_tolerance.items()
        ],
        "categories": category_results,
    }


def _validate_platform_report(
    report: Mapping[str, Any],
    *,
    prefix,
    expected_source_commit: str,
) -> None:
    if report.get("schema") != PLATFORM_REPORT_SCHEMA or report.get("completed") is not True:
        raise ValueError("source platform differential is incomplete or unsupported")
    source = report.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("commit") != expected_source_commit
        or source.get("dirty") is not False
    ):
        raise ValueError("source platform differential source revision mismatch")
    action = report.get("action_stream")
    provenance = action.get("provenance") if isinstance(action, Mapping) else None
    expected_publication_frame = (
        provenance.get(
            "first_battle_publication_frame",
            provenance.get("first_retail_frame"),
        )
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(action, Mapping)
        or not isinstance(provenance, Mapping)
        or provenance.get("kind")
        != "verified-original-retail-wine-first-failure-action-prefix"
        or provenance.get("run_id") != prefix.run_id
        or provenance.get("manifest_sha256") != prefix.manifest_sha256
        or provenance.get("run_sha256") != prefix.run_sha256
        or provenance.get("failure_kind") != prefix.failure_kind
        or provenance.get("last_retail_frame") != prefix.failure_frame
        or action.get("max_ticks") != prefix.failure_frame
        or action.get("stage_rng_seed") != provenance.get("recovered_stage_rng_seed")
        or action.get("auto_shoot_after_tick")
        != expected_publication_frame
    ):
        raise ValueError("source action stream is not anchored to the retail prefix")


def audit_retail_source_replay(
    run_directory: Path,
    platform_directory: Path,
    *,
    expected_scope: tuple[int, int, int, int],
    expected_executable_sha256: str,
    expected_native_kernel_sha256: str,
    expected_source_commit: str,
    require_full_source_coverage: bool = True,
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    platform_directory = platform_directory.resolve()
    prefix = load_first_failure_prefix(
        run_directory,
        expected_scope=expected_scope,
        expected_executable_sha256=expected_executable_sha256,
        expected_native_kernel_sha256=expected_native_kernel_sha256,
        expected_policy_id=FROZEN_INCUMBENT_POLICY_ID,
    )
    manifest = _object(run_directory / "manifest.json")
    frames, frame_evidence = _verified_stream_rows(run_directory, manifest, "frames")
    platform_path = platform_directory / "report.json"
    platform_report = _object(platform_path)
    _validate_platform_report(
        platform_report,
        prefix=prefix,
        expected_source_commit=expected_source_commit,
    )
    traces = platform_report.get("traces")
    if not isinstance(traces, Mapping):
        raise TypeError("source platform report trace evidence is absent")
    comparisons: dict[str, Any] = {}
    trace_evidence: dict[str, Any] = {}
    for domain in ("linux", "wine"):
        stated = traces.get(domain)
        if not isinstance(stated, Mapping):
            raise TypeError(f"source platform report lacks {domain} trace evidence")
        trace = Path(str(stated.get("path", ""))).resolve()
        if trace.parent != platform_directory or not trace.is_file():
            raise ValueError(f"unsafe or absent {domain} source replay trace")
        if _sha256(trace) != stated.get("sha256"):
            raise ValueError(f"{domain} source replay trace SHA-256 mismatch")
        comparisons[domain] = compare_retail_source_states(
            frames,
            trace,
            require_full_coverage=require_full_source_coverage,
        )
        trace_evidence[domain] = _trace_summary(trace)

    known_gap_frames: list[int] = []
    action = platform_report["action_stream"]
    provenance = action["provenance"]
    gaps = provenance.get("observation_gaps")
    if isinstance(gaps, list):
        large = [
            gap for gap in gaps
            if isinstance(gap, Mapping) and int(gap.get("frames", 0)) >= 10
        ]
        if large:
            known_gap_frames = sorted({
                int(gap["source_frame"]) + int(gap["frames"])
                for gap in large
            })
    category_names = ("rng", "game", "hazard_counts", "player_geometry_at_1e_6")
    stable_shared_state = all(
        comparisons[domain]["categories"][name]["equal"]
        for domain in comparisons
        for name in category_names
    )
    input_divergences = [
        comparisons[domain]["categories"]["input"]["first_divergence"]
        for domain in comparisons
    ]
    input_only_at_known_gap = (
        bool(known_gap_frames)
        and all(
            divergence is not None and divergence["frame"] in known_gap_frames
            for divergence in input_divergences
        )
    )
    source_hit_ticks = [
        trace_evidence[domain]["first_hit_tick"] for domain in trace_evidence
    ]
    if any(tick is not None for tick in source_hit_ticks):
        conclusion = "source-physical-hit-before-retail-failure"
    elif stable_shared_state and input_only_at_known_gap:
        conclusion = "shared-dynamics-match-with-known-dialogue-input-gap"
    elif stable_shared_state and all(divergence is None for divergence in input_divergences):
        conclusion = "shared-state-match"
    else:
        conclusion = "retail-source-shared-state-divergence"
    return {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conclusion": conclusion,
        "retail": {
            "run_id": prefix.run_id,
            "run_directory": str(run_directory),
            "failure_kind": prefix.failure_kind,
            "failure_frame": prefix.failure_frame,
            "failure_context": prefix.failure_context,
            "manifest_sha256": prefix.manifest_sha256,
            "run_sha256": prefix.run_sha256,
            "executable_sha256": prefix.executable_sha256,
            "native_kernel_sha256": prefix.native_kernel_sha256,
            "frame_shards": frame_evidence,
        },
        "source_platform_report": {
            "path": str(platform_path),
            "sha256": _sha256(platform_path),
            "source_commit": expected_source_commit,
            "conclusion": platform_report.get("conclusion"),
        },
        "known_dialogue_gap_target_frames": known_gap_frames,
        "require_full_source_coverage": require_full_source_coverage,
        "comparisons": comparisons,
        "traces": trace_evidence,
        "evidence_boundary": {
            "promotion_authority": False,
            "training_corpus": False,
            "purpose": "platform and input-delivery drift localization",
            "retail_failure_remains_authoritative": True,
            "headless_hit_continuation": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--platform-directory", required=True, type=Path)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--expected-native-kernel-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--character", type=int, default=0)
    parser.add_argument("--shot-type", type=int, default=0)
    parser.add_argument("--stage", type=int, default=6)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-source-terminal-prefix",
        action="store_true",
        help="retain a source trace that HITs or terminates before the retail failure",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = audit_retail_source_replay(
            args.run_directory,
            args.platform_directory,
            expected_scope=(
                args.difficulty,
                args.character,
                args.shot_type,
                args.stage,
            ),
            expected_executable_sha256=args.expected_executable_sha256,
            expected_native_kernel_sha256=args.expected_native_kernel_sha256,
            expected_source_commit=args.expected_source_commit,
            require_full_source_coverage=not args.allow_source_terminal_prefix,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": report["schema"],
        "conclusion": report["conclusion"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
