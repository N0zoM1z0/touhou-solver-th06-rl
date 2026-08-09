#!/usr/bin/env python3
"""A/B one replay checkpoint across an additive headless event upgrade.

The diagnostic replays the same immutable prefix and every requested ordinary
action through both binaries.  It removes only the top-level ``events`` member
before comparing every emitted physical observation.  This is offline source
evidence; it neither changes nor enlarges resident collision authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

try:
    from label_headless_feasibility_oracle import _load_run, _write_prefix
except ModuleNotFoundError:
    from scripts.label_headless_feasibility_oracle import _load_run, _write_prefix

from th06_rl.headless import HeadlessScope
from th06_rl.headless_forkserver import HeadlessForkserver
from th06_rl.native import ACTIONS


SCHEMA = "th06-rl-headless-event-observation-differential-v1"
ACTION_NAMES = tuple(action.name for action in ACTIONS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def physical_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Remove exactly the additive diagnostic event member."""
    return {key: value for key, value in observation.items() if key != "events"}


def event_summary(observation: Mapping[str, Any], *, offset: int) -> dict[str, Any] | None:
    events = observation.get("events")
    if not isinstance(events, Mapping):
        return None
    bullet_births = events.get("bullet_births")
    laser_births = events.get("laser_births")
    hit = events.get("hit")
    bullet_count = len(bullet_births) if isinstance(bullet_births, list) else -1
    laser_count = len(laser_births) if isinstance(laser_births, list) else -1
    if bullet_count == 0 and laser_count == 0 and hit is None:
        return None
    return {
        "offset": offset,
        "tick": observation.get("tick"),
        "bullet_births": bullet_count,
        "laser_births": laser_count,
        "hit": hit,
        "events_sha256": _canonical_digest(events),
    }


def _git_source(binary: Path) -> dict[str, Any]:
    source_root = binary.resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "binary_sha256": _sha256(binary), "clean": not dirty}


def _replay(
    *,
    binary: Path,
    game_directory: Path,
    scope: HeadlessScope,
    seed: int,
    checkpoint_tick: int,
    prefix: Path,
    actions: tuple[str, ...],
    branch_frames: int,
) -> dict[str, list[dict[str, Any]]]:
    branches: dict[str, list[dict[str, Any]]] = {}
    server = HeadlessForkserver(
        binary=binary,
        game_directory=game_directory,
        scope=scope,
        seed=seed,
    )
    try:
        if server.start() != 1:
            raise ValueError("unexpected stage-entry checkpoint tick")
        server.enter_checkpoint(terminal_tick=checkpoint_tick, actions_path=prefix)
        try:
            for action in actions:
                observations = [
                    server.begin_step_session(
                        terminal_tick=checkpoint_tick + branch_frames
                    )
                ]
                while observations[-1].get("terminal_reason") is None:
                    observations.append(server.step_session(action))
                server.finish_step_session()
                branches[action] = observations
        finally:
            server.leave_checkpoint()
    finally:
        server.close()
    return branches


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint-sequence", type=int, required=True)
    parser.add_argument("--branch-frames", type=int, default=4)
    parser.add_argument("--action", action="append")
    parser.add_argument("--old-binary", type=Path, required=True)
    parser.add_argument(
        "--new-binary",
        type=Path,
        default=root / "reference/GensokyoClub-th06-portable/th06",
    )
    parser.add_argument(
        "--game-directory",
        type=Path,
        default=root / "reference/th06-game-original/th06",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.branch_frames <= 0:
        parser.error("branch frames must be positive")
    actions = tuple(dict.fromkeys(args.action or ACTION_NAMES))
    if not actions or any(action not in ACTION_NAMES for action in actions):
        parser.error("event differential actions must be ordinary Bomb-free actions")

    run = args.run.resolve()
    manifest, rows = _load_run(run)
    sequence = args.checkpoint_sequence
    if sequence <= 0 or sequence >= len(rows):
        parser.error("checkpoint sequence is outside the reconstructable run")
    old_binary = args.old_binary.resolve()
    new_binary = args.new_binary.resolve()
    game_directory = args.game_directory.resolve()
    if not old_binary.is_file() or not new_binary.is_file():
        parser.error("both differential binaries must exist")
    old_source = dict(manifest.get("source", {}))
    if old_source.get("binary_sha256") != _sha256(old_binary):
        parser.error("old differential binary does not match the input run")
    new_source = _git_source(new_binary)
    if not new_source["clean"]:
        parser.error("new differential runtime source is dirty")
    scope_data = manifest["scope"]
    scope = HeadlessScope(
        int(scope_data["difficulty"]),
        int(scope_data["character"]),
        int(scope_data["shot_type"]),
        int(scope_data["stage"]),
    )
    checkpoint_tick = int(rows[sequence]["tick"])
    with tempfile.TemporaryDirectory(prefix="th06-event-differential-") as raw:
        prefix = Path(raw) / "prefix.txt"
        _write_prefix(prefix, rows, sequence)
        common = {
            "game_directory": game_directory,
            "scope": scope,
            "seed": int(manifest["initial_seed"]),
            "checkpoint_tick": checkpoint_tick,
            "prefix": prefix,
            "actions": actions,
            "branch_frames": args.branch_frames,
        }
        old_branches = _replay(binary=old_binary, **common)
        new_branches = _replay(binary=new_binary, **common)

    branches = []
    for action in actions:
        old = old_branches[action]
        new = new_branches[action]
        old_physical = [physical_observation(item) for item in old]
        new_physical = [physical_observation(item) for item in new]
        mismatches = [
            offset
            for offset in range(max(len(old_physical), len(new_physical)))
            if offset >= len(old_physical)
            or offset >= len(new_physical)
            or old_physical[offset] != new_physical[offset]
        ]
        summaries = [
            summary
            for offset, item in enumerate(new)
            if (summary := event_summary(item, offset=offset)) is not None
        ]
        branches.append({
            "action": action,
            "physical_observations_equal": not mismatches,
            "old_observation_count": len(old),
            "new_observation_count": len(new),
            "mismatch_offsets": mismatches,
            "old_physical_sha256": _canonical_digest(old_physical),
            "new_physical_sha256": _canonical_digest(new_physical),
            "terminal_reason": new[-1].get("terminal_reason"),
            "deaths_delta": int(new[-1]["deaths"]) - int(new[0]["deaths"]),
            "bombs_delta": int(new[-1]["bombs_used"]) - int(new[0]["bombs_used"]),
            "eventful_observations": summaries,
        })
    physical_equal = all(branch["physical_observations_equal"] for branch in branches)
    eventful_count = sum(len(branch["eventful_observations"]) for branch in branches)
    hit_kinds = sorted({
        str(summary["hit"]["kind"])
        for branch in branches
        for summary in branch["eventful_observations"]
        if isinstance(summary.get("hit"), Mapping)
    })
    result = {
        "schema": SCHEMA,
        "authority": "additive-diagnostics-only-no-native-set-revision",
        "interpretation": (
            "events are removed before exact physical observation comparison; "
            "this artifact does not enlarge resident collision authority"
        ),
        "input_run": str(run),
        "input_manifest_sha256": _sha256(run / "manifest.json"),
        "scope": scope_data,
        "initial_seed": manifest["initial_seed"],
        "checkpoint_sequence": sequence,
        "checkpoint_tick": checkpoint_tick,
        "branch_frames": args.branch_frames,
        "actions": list(actions),
        "old_runtime_source": old_source,
        "new_runtime_source": new_source,
        "removed_observation_members": ["events"],
        "physical_observations_equal": physical_equal,
        "eventful_observation_count": eventful_count,
        "hit_kinds": hit_kinds,
        "branches": branches,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if physical_equal and eventful_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
