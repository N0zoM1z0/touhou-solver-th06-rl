#!/usr/bin/env python3
"""Generate compact native-gated TH06 trajectories from Linux headless mode."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from th06_rl.headless import HeadlessClient, HeadlessScope
from th06_rl.headless_corpus import (
    CompactHeadlessCorpusWriter,
    EpsilonTeacherBehavior,
    NativeOfflineTeacher,
    build_transition,
)
from th06_rl.headless_geometry import (
    HARD_HORIZON,
    KINEMATICS,
    HeadlessAuthorityUnavailable,
    action_from_input,
    certify_lowered_headless_actions,
    lower_headless_hazards,
)
from th06_rl.native import NativeKernel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance(binary: Path) -> dict[str, Any]:
    source = binary.resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "commit": commit,
        "clean": not dirty,
        "binary_sha256": _sha256(binary),
    }


def _run_id(scope: HeadlessScope, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{timestamp}-d{scope.difficulty}-c{scope.character}-"
        f"s{scope.shot_type}-stage{scope.stage}-seed{seed}"
    )


def generate_episode(
    *,
    binary: Path,
    game_directory: Path,
    output_root: Path,
    scope: HeadlessScope,
    seed: int,
    behavior_seed: int,
    epsilon: float,
    max_ticks: int,
    anchor_stride: int,
    teacher_horizon: int,
    provenance: dict[str, Any],
) -> Path:
    run_directory = output_root / _run_id(scope, seed)
    writer = CompactHeadlessCorpusWriter(run_directory, anchor_stride=anchor_stride)
    kernel = NativeKernel()
    teacher = NativeOfflineTeacher(kernel=kernel, horizon=teacher_horizon)
    behavior_policy = EpsilonTeacherBehavior(epsilon=epsilon, seed=behavior_seed)
    termination_reason = "generator-error"
    authority_failure: str | None = None
    terminal_observation: dict[str, Any] | None = None
    try:
        client = HeadlessClient(
            binary=binary,
            game_directory=game_directory,
            scope=scope,
            seed=seed,
            max_ticks=max_ticks,
        )
        try:
            observation = client.start()
            writer.anchor(observation, sequence=0, role="initial", force=True)
            sequence = 0
            while observation.get("terminal_reason") is None:
                try:
                    hard_hazards = lower_headless_hazards(observation, HARD_HORIZON)
                    prepared_hard = kernel.prepare_hazards(hard_hazards)
                    certified = certify_lowered_headless_actions(
                        observation,
                        prepared_hard,
                        kernel=kernel,
                    )
                    if not certified:
                        raise HeadlessAuthorityUnavailable("headless native safe set is empty")
                    teacher_hazards = lower_headless_hazards(observation, teacher_horizon)
                    player = observation["player"]
                    profiles = kernel.profile_actions(
                        x=float(player["x"]),
                        y=float(player["y"]),
                        half_width=float(player["half_width"]),
                        half_height=float(player["half_height"]),
                        kinematics=KINEMATICS,
                        current_action=action_from_input(int(observation["input"])),
                        hazards=teacher_hazards,
                        candidates=tuple(item.action for item in certified),
                    )
                    teacher_decision = teacher.rank(
                        observation,
                        certified,
                        hazards=teacher_hazards,
                    )
                    behavior = behavior_policy.select(teacher_decision, certified)

                    # The step process cannot advance asynchronously, but keep
                    # the same explicit issue-gate boundary as the real agent.
                    issue_certified = certify_lowered_headless_actions(
                        observation,
                        prepared_hard,
                        kernel=kernel,
                    )
                    if behavior.selected_action not in {
                        item.action.name for item in issue_certified
                    }:
                        raise HeadlessAuthorityUnavailable(
                            "selected action failed the immediate issue gate"
                        )
                except HeadlessAuthorityUnavailable as error:
                    authority_failure = str(error)
                    termination_reason = "authority-failure"
                    writer.anchor(
                        observation,
                        sequence=sequence,
                        role="authority-failure",
                        force=True,
                    )
                    break

                next_observation = client.step(behavior.selected_action)
                transition = build_transition(
                    sequence=sequence,
                    observation=observation,
                    next_observation=next_observation,
                    certified=certified,
                    behavior=behavior,
                    epsilon=epsilon,
                    profiles=profiles,
                )
                if transition["outcome_terms"]["bombs_used_delta"] != 0:
                    raise HeadlessAuthorityUnavailable("Bomb use appeared in headless outcome")
                writer.transition(transition)
                sequence += 1
                writer.anchor(
                    next_observation,
                    sequence=sequence,
                    role="periodic",
                )
                observation = next_observation
                if observation.get("terminal_reason") is not None:
                    terminal_observation = observation
                    termination_reason = str(observation["terminal_reason"])
                    writer.anchor(
                        observation,
                        sequence=sequence,
                        role="terminal",
                        force=True,
                    )
                    break
        finally:
            client.close()
    except Exception:
        writer.abort()
        raise

    manifest = writer.close({
        "transaction_complete": True,
        "training_eligible": writer.transition_count > 0,
        "scope": {
            "difficulty": scope.difficulty,
            "character": scope.character,
            "shot_type": scope.shot_type,
            "stage": scope.stage,
        },
        "initial_seed": seed,
        "behavior_seed": behavior_seed,
        "behavior_policy": "epsilon-native-offline-teacher-v1",
        "behavior_epsilon": epsilon,
        "teacher_horizon": teacher_horizon,
        "native_gate_horizon": 4,
        "anchor_stride": anchor_stride,
        "termination_reason": termination_reason,
        "authority_failure": authority_failure,
        "physical_hit": termination_reason == "physical-hit",
        "nmnb_stage_clear": bool(
            terminal_observation is not None
            and termination_reason == "chain-exit-success"
            and int(terminal_observation["deaths"]) == 0
            and int(terminal_observation["bombs_used"]) == 0
        ),
        "effective_transition_ratio": 1.0 if writer.transition_count else 0.0,
        "selected_actions_native_legal_ratio": 1.0 if writer.transition_count else 0.0,
        "source": provenance,
    })
    return manifest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=root / "reference/GensokyoClub-th06-portable/th06",
    )
    parser.add_argument(
        "--game-directory",
        type=Path,
        default=root / "reference/th06-game-original/th06",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "artifacts/headless-corpus",
    )
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--behavior-seed", type=int)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-ticks", type=int, default=3600)
    parser.add_argument("--anchor-stride", type=int, default=120)
    parser.add_argument("--teacher-horizon", type=int, default=12)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--character", type=int, default=0)
    parser.add_argument("--shot-type", type=int, default=0)
    parser.add_argument("--stage", type=int, default=6)
    args = parser.parse_args()
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("epsilon must be in 0..1")
    if min(args.max_ticks, args.anchor_stride, args.teacher_horizon) <= 0:
        parser.error("tick, anchor, and teacher bounds must be positive")
    scope = HeadlessScope(
        args.difficulty,
        args.character,
        args.shot_type,
        args.stage,
    )
    binary = args.binary.resolve()
    game_directory = args.game_directory.resolve()
    if not binary.is_file():
        parser.error(f"headless binary not found: {binary}")
    if not game_directory.is_dir():
        parser.error(f"game directory not found: {game_directory}")
    provenance = _source_provenance(binary)
    manifests = []
    for index, seed in enumerate(args.seed):
        behavior_seed = (
            args.behavior_seed + index
            if args.behavior_seed is not None
            else seed ^ 0x6A09E667
        )
        manifests.append(generate_episode(
            binary=binary,
            game_directory=game_directory,
            output_root=args.output_root.resolve(),
            scope=scope,
            seed=seed,
            behavior_seed=behavior_seed,
            epsilon=args.epsilon,
            max_ticks=args.max_ticks,
            anchor_stride=args.anchor_stride,
            teacher_horizon=args.teacher_horizon,
            provenance=provenance,
        ))
    print(json.dumps({"manifests": [str(path) for path in manifests]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
