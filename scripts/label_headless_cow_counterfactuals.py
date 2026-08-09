#!/usr/bin/env python3
"""Label native-legal first actions with dynamic COW teacher continuations."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

from th06_rl.headless import HeadlessScope
from th06_rl.headless_corpus import NativeOfflineTeacher, canonical_observation_sha256
from th06_rl.headless_forkserver import HeadlessForkserver
from th06_rl.headless_geometry import (
    HARD_HORIZON,
    HeadlessAuthorityUnavailable,
    certify_lowered_headless_actions,
    lower_headless_hard_hazards,
    lower_headless_hazards,
)
from th06_rl.native import NativeKernel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_provenance(binary: Path) -> dict[str, Any]:
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
    return {"commit": commit, "clean": not dirty, "binary_sha256": _sha256(binary)}


def _load_run(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("transaction_complete") is not True:
        raise ValueError("counterfactual input run is not transaction complete")
    rows = []
    with gzip.open(run / "transitions.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows.extend(json.loads(line) for line in stream)
    if len(rows) != manifest.get("transition_count"):
        raise ValueError("counterfactual input transition count is inconsistent")
    if any(row.get("sequence") != index for index, row in enumerate(rows)):
        raise ValueError("counterfactual input sequence is not dense")
    return manifest, rows


def _write_prefix(path: Path, rows: list[dict[str, Any]], sequence: int) -> None:
    path.write_text(
        "".join(f"{row['behavior']['selected_action']}\n" for row in rows[:sequence]),
        encoding="utf-8",
    )


def _boundary_reserve(observation: Mapping[str, Any]) -> float:
    player = observation["player"]
    x = float(player["x"])
    y = float(player["y"])
    return min(x - 8.0, 376.0 - x, y - 16.0, 432.0 - y)


def outcome_rank(outcome: Mapping[str, Any]) -> tuple[int, int, int, int, float]:
    """Order physical survival first, then retain future maneuverability."""
    terminal = str(outcome["termination_reason"])
    completed = terminal in {"tick-limit", "chain-exit-success", "stage-clear-success"}
    return (
        int(completed),
        int(terminal != "physical-hit"),
        int(outcome["survival_ticks"]),
        int(outcome["minimum_native_legal_actions"]),
        float(outcome["terminal_boundary_reserve"]),
    )


def _certify(observation: Mapping[str, Any], kernel: NativeKernel):
    hazards = lower_headless_hard_hazards(observation, HARD_HORIZON)
    prepared = kernel.prepare_hazards(hazards)
    return certify_lowered_headless_actions(observation, prepared, kernel=kernel), prepared


def label_checkpoint(
    *,
    server: HeadlessForkserver,
    row: Mapping[str, Any],
    sequence: int,
    branch_frames: int,
    teacher: NativeOfflineTeacher,
    kernel: NativeKernel,
) -> dict[str, Any]:
    checkpoint_tick = int(row["tick"])
    terminal_tick = checkpoint_tick + branch_frames
    outcomes = []
    for first_action in row["legal_actions"]:
        observation = server.begin_step_session(terminal_tick=terminal_tick)
        digest = canonical_observation_sha256(observation)
        if digest != row["observation_sha256"]:
            server.abort_step_session()
            raise ValueError("COW checkpoint is not byte-logically identical to corpus observation")
        certified, _ = _certify(observation, kernel)
        certified_names = tuple(item.action.name for item in certified)
        if tuple(row["legal_actions"]) != certified_names:
            server.abort_step_session()
            raise ValueError("COW checkpoint native legal set differs from corpus")
        checkpoint_deaths = int(observation["deaths"])
        minimum_legal = len(certified)
        actions_issued = 0
        authority_failure_tick: int | None = None
        observation = server.step_session(str(first_action))
        actions_issued += 1
        while observation.get("terminal_reason") is None:
            try:
                certified, prepared = _certify(observation, kernel)
                if not certified:
                    raise HeadlessAuthorityUnavailable("counterfactual native safe set is empty")
                minimum_legal = min(minimum_legal, len(certified))
                hazards = lower_headless_hazards(observation, teacher.horizon)
                decision = teacher.rank(observation, certified, hazards=hazards)
                issue = certify_lowered_headless_actions(
                    observation,
                    prepared,
                    kernel=kernel,
                )
                if decision.action not in {item.action.name for item in issue}:
                    raise HeadlessAuthorityUnavailable(
                        "counterfactual teacher failed the fresh issue gate"
                    )
            except HeadlessAuthorityUnavailable:
                authority_failure_tick = int(observation["tick"])
                result = server.abort_step_session()
                terminal_observation = result.terminal_observation
                break
            observation = server.step_session(decision.action)
            actions_issued += 1
        else:
            result = server.finish_step_session()
            terminal_observation = result.terminal_observation

        if authority_failure_tick is None:
            termination_reason = str(terminal_observation["terminal_reason"])
            end_tick = int(terminal_observation["tick"])
            terminal_reserve = _boundary_reserve(terminal_observation)
        else:
            termination_reason = "authority-failure"
            end_tick = authority_failure_tick
            terminal_reserve = _boundary_reserve(observation)
        outcomes.append({
            "first_action": first_action,
            "termination_reason": termination_reason,
            "survival_ticks": end_tick - checkpoint_tick,
            "actions_issued": actions_issued,
            "minimum_native_legal_actions": minimum_legal,
            "terminal_boundary_reserve": terminal_reserve,
            "physical_deaths_delta": int(terminal_observation["deaths"])
            - checkpoint_deaths,
        })
    best_rank = max(outcome_rank(outcome) for outcome in outcomes)
    best_actions = sorted(
        str(outcome["first_action"])
        for outcome in outcomes
        if outcome_rank(outcome) == best_rank
    )
    return {
        "sequence": sequence,
        "checkpoint_tick": checkpoint_tick,
        "observation_sha256": row["observation_sha256"],
        "source_context": row["source_context"],
        "factual_action": row["behavior"]["selected_action"],
        "local_teacher_action": row["behavior"]["teacher_action"],
        "native_legal_actions": row["legal_actions"],
        "branch_frames": branch_frames,
        "best_actions": best_actions,
        "factual_action_is_best": row["behavior"]["selected_action"] in best_actions,
        "local_teacher_action_is_best": row["behavior"]["teacher_action"] in best_actions,
        "outcomes": outcomes,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint-sequence", type=int, action="append", required=True)
    parser.add_argument("--branch-frames", type=int, default=240)
    parser.add_argument("--teacher-horizon", type=int, default=12)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.branch_frames <= 0 or args.teacher_horizon < HARD_HORIZON:
        parser.error("branch and teacher horizons are outside safe bounds")
    run = args.run.resolve()
    manifest, rows = _load_run(run)
    sequences = sorted(set(args.checkpoint_sequence))
    if any(sequence <= 0 or sequence >= len(rows) for sequence in sequences):
        parser.error("checkpoint sequence is outside the reconstructable run")
    scope_data = manifest["scope"]
    scope = HeadlessScope(
        int(scope_data["difficulty"]),
        int(scope_data["character"]),
        int(scope_data["shot_type"]),
        int(scope_data["stage"]),
    )
    binary = args.binary.resolve()
    kernel = NativeKernel()
    teacher = NativeOfflineTeacher(kernel=kernel, horizon=args.teacher_horizon)
    labels = []
    with tempfile.TemporaryDirectory(prefix="th06-cow-label-") as raw:
        workspace = Path(raw)
        server = HeadlessForkserver(
            binary=binary,
            game_directory=args.game_directory.resolve(),
            scope=scope,
            seed=int(manifest["initial_seed"]),
        )
        try:
            root_tick = server.start()
            if root_tick != 1:
                raise ValueError(f"unexpected stage-entry checkpoint tick {root_tick}")
            for sequence in sequences:
                prefix = workspace / f"prefix-{sequence}.txt"
                _write_prefix(prefix, rows, sequence)
                server.enter_checkpoint(
                    terminal_tick=int(rows[sequence]["tick"]),
                    actions_path=prefix,
                )
                try:
                    labels.append(label_checkpoint(
                        server=server,
                        row=rows[sequence],
                        sequence=sequence,
                        branch_frames=args.branch_frames,
                        teacher=teacher,
                        kernel=kernel,
                    ))
                finally:
                    server.leave_checkpoint()
        finally:
            server.close()
    result = {
        "schema": "th06-rl-headless-cow-counterfactual-v1",
        "authority": "first-action-native-legal-and-dynamic-continuation-revalidated",
        "scope": scope_data,
        "initial_seed": manifest["initial_seed"],
        "input_run": str(run),
        "input_source": manifest["source"],
        "runtime_source": _runtime_provenance(binary),
        "teacher_horizon": args.teacher_horizon,
        "branch_frames": args.branch_frames,
        "checkpoints": labels,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
