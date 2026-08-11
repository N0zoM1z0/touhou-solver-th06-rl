#!/usr/bin/env python3
"""Recompute Wine Hard-empty roots against source-exact TH06 geometry."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.replay_corpus import _hydrate, _load_objects  # noqa: E402
from th06_rl.autonomous_learning import _transition_rows, _validate_run  # noqa: E402
from th06_rl.native import ACTIONS, NativeKernel  # noqa: E402
from th06_rl.th06.control_capture import decode_control_snapshot  # noqa: E402
from th06_rl.th06.source import (  # noqa: E402
    COLLISION_MARGIN,
    core_action_from_input,
    kinematics_from_snapshot,
    lower_observed_hazards,
)


SCHEMA = "wine-hard-empty-native-audit-v2"
REASONS = {
    "authority-stop:Hard safe set empty",
    "control-dead-end:Hard safe set empty",
}


def _rows(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)


def _source_binding(source_root: Path) -> dict[str, object]:
    files = (
        "src/Player.cpp",
        "src/BulletManager.cpp",
        "src/EnemyManager.cpp",
        "src/ChainPriorities.hpp",
    )
    bound = {}
    for relative in files:
        path = source_root / relative
        bound[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    commit = subprocess.run(
        ("git", "-C", str(source_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "repository": "https://github.com/GensokyoClub/th06.git",
        "commit": commit,
        "files_sha256": bound,
        "source_collision_contract": {
            "player_half_width": 1.25,
            "player_half_height": 1.25,
            "aabb_contact_is_collision": True,
            "extra_collision_margin": 0.0,
            "player_moves_before_bullet_collision": True,
        },
    }


def audit_run(run_dir: Path, native_library: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    _run, manifest = _validate_run(run_dir)
    objects = _load_objects(run_dir)
    frames = list(_rows(sorted(run_dir.glob("frames-*.jsonl.gz"))))
    transitions = list(_transition_rows(run_dir, manifest))
    transition_by_next = {
        str(row.get("next_snapshot_ref")): index
        for index, row in enumerate(transitions)
    }
    kernel = NativeKernel(native_library)
    roots = []
    for row in frames:
        decision = row.get("decision")
        if not isinstance(decision, dict) or decision.get("reason") not in REASONS:
            continue
        raw = _hydrate(row["snapshot"], objects)
        snapshot = decode_control_snapshot(raw)
        forecast = lower_observed_hazards(snapshot, 12)
        prepared = kernel.prepare_hazards(forecast.hazards)
        common = {
            "x": snapshot.x,
            "y": snapshot.y,
            "half_width": snapshot.half_width,
            "half_height": snapshot.half_height,
            "kinematics": kinematics_from_snapshot(snapshot),
            "current_action": core_action_from_input(snapshot.input_mask),
        }
        recomputed = kernel.certify_actions(
            **common, hazards=prepared.prefix(4)
        )
        source_exact = kernel.certify_actions(
            **common,
            hazards=prepared.prefix(4),
            collision_margin=0.0,
        )
        profiles = kernel.profile_actions(
            **common,
            hazards=prepared.prefix(4),
            candidates=ACTIONS,
            checkpoints=(1, 2, 3, 4),
            collision_margin=0.0,
        )
        witnesses = {}
        for profile in profiles:
            first_source_collision = next((
                checkpoint
                for checkpoint, clearance in zip(
                    profile.checkpoints, profile.min_clearances, strict=True
                )
                if clearance <= 0.0
            ), None)
            first_conservative_closure = next((
                checkpoint
                for checkpoint, clearance in zip(
                    profile.checkpoints, profile.min_clearances, strict=True
                )
                if clearance <= COLLISION_MARGIN
            ), None)
            witnesses[profile.action.name] = {
                "first_source_collision_frame": first_source_collision,
                "first_conservative_closure_frame": first_conservative_closure,
                "checkpoint_min_clearance": list(profile.min_clearances),
            }
        transition_index = transition_by_next.get(str(row["snapshot_id"]))
        followup = "unavailable-stage-stopped"
        followup_frames = None
        if transition_index is not None:
            start_frame = int(snapshot.frame)
            for future in transitions[transition_index + 1:]:
                terms = future.get("outcome_terms")
                if not isinstance(terms, dict):
                    continue
                next_ref = str(future.get("next_snapshot_ref", ""))
                try:
                    future_frame = int(next_ref.rsplit(":f", 1)[1])
                except (IndexError, ValueError):
                    future_frame = start_frame
                if terms.get("life_lost") is True:
                    followup = "physical-hit"
                    followup_frames = future_frame - start_frame
                    break
                if int(terms.get("hard_count_after", 0)) > 0:
                    followup = "native-safe-set-recovered"
                    followup_frames = future_frame - start_frame
                    break
        recorded = tuple(decision.get("hard_actions", ()))
        roots.append({
            "sequence": int(row["sequence"]),
            "frame": int(snapshot.frame),
            "reason": decision["reason"],
            "recorded_hard_count": len(recorded),
            "recomputed_hard_actions": [item.action.name for item in recomputed],
            "source_exact_hard_actions": [
                item.action.name for item in source_exact
            ],
            "closure_classification": (
                "source-geometry-closure"
                if not source_exact
                else "conservative-margin-closure"
            ),
            "all_actions_have_four_frame_conservative_closure": all(
                item["first_conservative_closure_frame"] is not None
                for item in witnesses.values()
            ),
            "all_actions_have_four_frame_source_collision": all(
                item["first_source_collision_frame"] is not None
                for item in witnesses.values()
            ),
            "witnesses": witnesses,
            "followup": followup,
            "followup_frames": followup_frames,
        })
    gates = {
        "hard_empty_roots_present": bool(roots),
        "recorded_masks_empty": all(row["recorded_hard_count"] == 0 for row in roots),
        "recomputed_masks_empty": all(not row["recomputed_hard_actions"] for row in roots),
        "every_action_has_conservative_closure_witness": all(
            row["all_actions_have_four_frame_conservative_closure"] for row in roots
        ),
    }
    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "stage_trajectory_complete": manifest.get("stage_trajectory_complete"),
        "roots": roots,
        "gates": gates,
        "passes": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPOSITORY / "reference/GensokyoClub-th06",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [audit_run(path, args.native_library) for path in args.runs]
    result = {
        "schema": SCHEMA,
        "native_library": str(args.native_library.resolve()),
        "authoritative_source": _source_binding(args.source_root.resolve()),
        "runs": reports,
        "passes": all(report["passes"] for report in reports),
        "continuous_followup_roots": sum(
            row["followup"] != "unavailable-stage-stopped"
            for report in reports for row in report["roots"]
        ),
        "source_geometry_closures": sum(
            row["closure_classification"] == "source-geometry-closure"
            for report in reports for row in report["roots"]
        ),
        "conservative_margin_closures": sum(
            row["closure_classification"] == "conservative-margin-closure"
            for report in reports for row in report["roots"]
        ),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
