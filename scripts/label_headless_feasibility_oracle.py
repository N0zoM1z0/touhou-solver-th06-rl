#!/usr/bin/env python3
"""Build an exact-checkpoint empirical feasibility lower bound for TH06.

Every native-safe first action is evaluated under every declared continuation.
The result is an offline diagnostic only: failure to find a witness is not a
proof that no action sequence exists, and no oracle result enters the resident
controller or enlarges the native action set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

try:
    from collect_headless_dagger import DistilledRanker, source_compatible
except ModuleNotFoundError:
    from scripts.collect_headless_dagger import DistilledRanker, source_compatible

from th06_rl.headless import HeadlessScope
from th06_rl.headless_corpus import (
    NativeOfflineTeacher,
    canonical_observation_sha256,
)
from th06_rl.headless_forkserver import HeadlessForkserver
from th06_rl.headless_geometry import (
    HARD_HORIZON,
    KINEMATICS,
    HeadlessAuthorityUnavailable,
    action_from_input,
    certify_lowered_headless_actions,
    lower_headless_hazards,
    reactive_headless_action,
)
from th06_rl.native import NativeCertifiedAction, NativeKernel


SCHEMA = "th06-rl-headless-feasibility-oracle-v1"
AUTHORITY = "exact-checkpoint-native-first-actions-multi-continuation-v1"
COMPLETED_TERMINATIONS = frozenset({
    "tick-limit",
    "chain-exit-success",
    "stage-clear-success",
})
MAX_BULLET_FEATURES = 32
MAX_LASER_FEATURES = 8
MAX_ENEMY_FEATURES = 8


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


def _repository_provenance(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "clean": not dirty}


def _load_run(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("transaction_complete") is not True:
        raise ValueError("feasibility input run is not transaction complete")
    with gzip.open(run / "transitions.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != manifest.get("transition_count"):
        raise ValueError("feasibility input transition count is inconsistent")
    if any(row.get("sequence") != index for index, row in enumerate(rows)):
        raise ValueError("feasibility input sequence is not dense")
    transitions = manifest.get("files", {}).get("transitions", {})
    transition_path = run / "transitions.jsonl.gz"
    if (
        transitions.get("path") != transition_path.name
        or int(transitions.get("bytes", -1)) != transition_path.stat().st_size
        or transitions.get("sha256") != _sha256(transition_path)
    ):
        raise ValueError("feasibility input transition artifact fails manifest hash")
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
    return (
        int(outcome.get("feasible") is True),
        int(int(outcome.get("physical_deaths_delta", 0)) == 0),
        int(outcome.get("survival_ticks", 0)),
        int(outcome.get("minimum_native_legal_actions", 0)),
        float(outcome.get("terminal_boundary_reserve", -math.inf)),
    )


def checkpoint_verdict(
    *,
    feasible_actions: tuple[str, ...],
    factual_action: str,
) -> str:
    if not feasible_actions:
        return "oracle-no-witness"
    if factual_action in feasible_actions:
        return "factual-action-has-witness"
    return "policy-selection-witness"


def _finite(row: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _slot_features(
    output: dict[str, float],
    *,
    prefix: str,
    rows: list[Mapping[str, Any]],
    fields: tuple[str, ...],
    count: int,
    player_x: float,
    player_y: float,
) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            math.hypot(_finite(row, "x") - player_x, _finite(row, "y") - player_y),
            _finite(row, "x"),
            _finite(row, "y"),
            _finite(row, "slot", -1.0),
        ),
    )[:count]
    for index in range(count):
        item = ordered[index] if index < len(ordered) else None
        output[f"{prefix}_{index}_present"] = float(item is not None)
        for field in fields:
            name = f"{prefix}_{index}_{field}"
            if item is None:
                output[name] = 0.0
            elif field == "dx":
                output[name] = _finite(item, "x") - player_x
            elif field == "dy":
                output[name] = _finite(item, "y") - player_y
            else:
                output[name] = _finite(item, field)


def exact_snapshot_features(observation: Mapping[str, Any]) -> dict[str, float]:
    """Retain a bounded source-physical probe richer than the deployable view.

    These values never become collision authority.  They exist only to test
    whether a richer exact-snapshot-derived representation generalizes better
    than the compact resident feature contract.
    """
    player = observation.get("player")
    if not isinstance(player, Mapping):
        raise HeadlessAuthorityUnavailable("oracle exact snapshot has no player")
    player_x = _finite(player, "x")
    player_y = _finite(player, "y")
    bullets = observation.get("bullets")
    lasers = observation.get("lasers")
    enemies = observation.get("enemies")
    if not all(isinstance(value, list) for value in (bullets, lasers, enemies)):
        raise HeadlessAuthorityUnavailable("oracle exact snapshot collections are incoherent")
    output = {
        "exact_player_x": player_x,
        "exact_player_y": player_y,
        "exact_bullet_count": float(len(bullets)),
        "exact_laser_count": float(len(lasers)),
        "exact_enemy_count": float(len(enemies)),
    }
    _slot_features(
        output,
        prefix="bullet",
        rows=[row for row in bullets if isinstance(row, Mapping)],
        fields=(
            "dx", "dy", "vx", "vy", "half_width", "half_height", "state",
            "ex_flags", "acceleration_x", "acceleration_y", "speed", "angle",
            "curve_speed_acceleration", "curve_angular_velocity", "turn_speed",
            "direction_rotation", "timer", "direction_interval",
        ),
        count=MAX_BULLET_FEATURES,
        player_x=player_x,
        player_y=player_y,
    )
    _slot_features(
        output,
        prefix="laser",
        rows=[row for row in lasers if isinstance(row, Mapping)],
        fields=(
            "dx", "dy", "angle", "angular_velocity", "width", "speed",
            "start", "end", "start_length", "start_time", "end_time", "timer",
            "duration", "state", "flags", "angle_tracked",
        ),
        count=MAX_LASER_FEATURES,
        player_x=player_x,
        player_y=player_y,
    )
    _slot_features(
        output,
        prefix="enemy",
        rows=[row for row in enemies if isinstance(row, Mapping)],
        fields=(
            "dx", "dy", "vx", "vy", "hitbox_width", "hitbox_height", "life",
            "boss", "contact_active", "ecl_sub", "ecl_time", "slot",
        ),
        count=MAX_ENEMY_FEATURES,
        player_x=player_x,
        player_y=player_y,
    )
    return output


def _certify(
    observation: Mapping[str, Any],
    kernel: NativeKernel,
) -> tuple[tuple[NativeCertifiedAction, ...], Any]:
    hazards = lower_headless_hazards(observation, HARD_HORIZON)
    prepared = kernel.prepare_hazards(hazards)
    certified = certify_lowered_headless_actions(observation, prepared, kernel=kernel)
    return certified, prepared


@dataclass(frozen=True)
class Continuation:
    name: str
    kind: str
    horizon: int
    teacher: NativeOfflineTeacher | None = None
    ranker: DistilledRanker | None = None
    ranker_sha256: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "horizon": self.horizon,
            "ranker_sha256": self.ranker_sha256,
        }

    def choose(
        self,
        observation: dict[str, Any],
        certified: tuple[NativeCertifiedAction, ...],
        *,
        kernel: NativeKernel,
        sequence: int,
        seed: int,
    ) -> str:
        if self.kind == "generic-clearance":
            return reactive_headless_action(observation, certified).name
        if self.kind == "native-local-plan":
            assert self.teacher is not None
            hazards = lower_headless_hazards(observation, self.horizon)
            return self.teacher.rank(observation, certified, hazards=hazards).action
        if self.kind == "distilled-ranker":
            assert self.ranker is not None
            hazards = lower_headless_hazards(observation, self.horizon)
            player = observation["player"]
            profiles = kernel.profile_actions(
                x=float(player["x"]),
                y=float(player["y"]),
                half_width=float(player["half_width"]),
                half_height=float(player["half_height"]),
                kinematics=KINEMATICS,
                current_action=action_from_input(int(observation["input"])),
                hazards=hazards,
                candidates=tuple(item.action for item in certified),
            )
            return self.ranker.rank(
                observation,
                certified,
                sequence=sequence,
                seed=seed,
                profiles=profiles,
            )
        raise ValueError(f"unsupported continuation kind {self.kind}")


def _run_branch(
    *,
    server: HeadlessForkserver,
    row: Mapping[str, Any],
    first_action: str,
    continuation: Continuation,
    branch_frames: int,
    kernel: NativeKernel,
    seed: int,
) -> dict[str, Any]:
    checkpoint_tick = int(row["tick"])
    observation = server.begin_step_session(
        terminal_tick=checkpoint_tick + branch_frames,
    )
    if canonical_observation_sha256(observation) != row["observation_sha256"]:
        server.abort_step_session()
        raise ValueError("oracle checkpoint is not byte-logically identical")
    certified, _ = _certify(observation, kernel)
    if tuple(item.action.name for item in certified) != tuple(row["legal_actions"]):
        server.abort_step_session()
        raise ValueError("oracle checkpoint native legal set differs from corpus")
    checkpoint_deaths = int(observation["deaths"])
    checkpoint_bombs = int(observation["bombs_used"])
    minimum_legal = len(certified)
    actions_issued = 1
    authority_failure_reason: str | None = None
    observation = server.step_session(first_action)
    while observation.get("terminal_reason") is None:
        try:
            certified, prepared = _certify(observation, kernel)
            if not certified:
                raise HeadlessAuthorityUnavailable("native safe set is empty")
            minimum_legal = min(minimum_legal, len(certified))
            selected = continuation.choose(
                observation,
                certified,
                kernel=kernel,
                sequence=actions_issued,
                seed=seed,
            )
            issued = certify_lowered_headless_actions(
                observation,
                prepared,
                kernel=kernel,
            )
            if selected not in {item.action.name for item in issued}:
                raise HeadlessAuthorityUnavailable("continuation failed fresh issue gate")
        except HeadlessAuthorityUnavailable as error:
            authority_failure_reason = str(error)
            result = server.abort_step_session()
            terminal_observation = result.terminal_observation
            break
        observation = server.step_session(selected)
        actions_issued += 1
    else:
        result = server.finish_step_session()
        terminal_observation = result.terminal_observation

    if authority_failure_reason is None:
        termination_reason = str(terminal_observation["terminal_reason"])
        end_tick = int(terminal_observation["tick"])
        terminal_reserve = _boundary_reserve(terminal_observation)
    else:
        termination_reason = "authority-failure"
        end_tick = int(observation["tick"])
        terminal_reserve = _boundary_reserve(observation)
    deaths_delta = int(terminal_observation["deaths"]) - checkpoint_deaths
    bombs_delta = int(terminal_observation["bombs_used"]) - checkpoint_bombs
    completed = termination_reason in COMPLETED_TERMINATIONS
    feasible = completed and deaths_delta == 0 and bombs_delta == 0
    return {
        "continuation": continuation.name,
        "first_action": first_action,
        "termination_reason": termination_reason,
        "authority_failure_reason": authority_failure_reason,
        "survival_ticks": end_tick - checkpoint_tick,
        "actions_issued": actions_issued,
        "minimum_native_legal_actions": minimum_legal,
        "terminal_boundary_reserve": terminal_reserve,
        "physical_deaths_delta": deaths_delta,
        "bombs_used_delta": bombs_delta,
        "feasible": feasible,
    }


def label_checkpoint(
    *,
    server: HeadlessForkserver,
    row: Mapping[str, Any],
    sequence: int,
    branch_frames: int,
    continuations: tuple[Continuation, ...],
    kernel: NativeKernel,
    seed: int,
) -> dict[str, Any]:
    first_observation = server.begin_step_session(
        terminal_tick=int(row["tick"]) + 1,
    )
    if canonical_observation_sha256(first_observation) != row["observation_sha256"]:
        server.abort_step_session()
        raise ValueError("oracle feature checkpoint is not byte-logically identical")
    exact_features = exact_snapshot_features(first_observation)
    # Close the one-tick feature probe with an ordinary certified action.  This
    # avoids using the deliberate authority-abort sentinel for a successful
    # observation read and keeps runtime diagnostics free of false failures.
    server.step_session(str(row["legal_actions"][0]))
    server.finish_step_session()
    branches = [
        _run_branch(
            server=server,
            row=row,
            first_action=str(first_action),
            continuation=continuation,
            branch_frames=branch_frames,
            kernel=kernel,
            seed=seed,
        )
        for first_action in row["legal_actions"]
        for continuation in continuations
    ]
    feasible_actions = tuple(sorted({
        str(branch["first_action"])
        for branch in branches
        if branch["feasible"] is True
    }))
    best_rank = max(outcome_rank(branch) for branch in branches)
    best_actions = tuple(sorted({
        str(branch["first_action"])
        for branch in branches
        if outcome_rank(branch) == best_rank
    }))
    factual = str(row["behavior"]["selected_action"])
    local = str(row["behavior"]["teacher_action"])
    return {
        "sequence": sequence,
        "checkpoint_tick": int(row["tick"]),
        "observation_sha256": row["observation_sha256"],
        "source_context": row["source_context"],
        "compact_state": row["state"],
        "action_candidates": row["action_candidates"],
        "exact_snapshot_features": exact_features,
        "factual_action": factual,
        "local_teacher_action": local,
        "native_legal_actions": row["legal_actions"],
        "branch_frames": branch_frames,
        "continuation_count": len(continuations),
        "feasible_actions": list(feasible_actions),
        "best_actions": list(best_actions),
        "factual_action_has_witness": factual in feasible_actions,
        "local_teacher_action_has_witness": local in feasible_actions,
        "verdict": checkpoint_verdict(
            feasible_actions=feasible_actions,
            factual_action=factual,
        ),
        "branches": branches,
    }


def _continuations(
    *,
    horizons: tuple[int, ...],
    model_paths: tuple[Path, ...],
    kernel: NativeKernel,
    threads: int,
    expected_scope: Mapping[str, Any],
    runtime_source: Mapping[str, Any],
) -> tuple[Continuation, ...]:
    result = [Continuation("generic-clearance", "generic-clearance", 0)]
    for horizon in horizons:
        result.append(Continuation(
            f"native-local-h{horizon}",
            "native-local-plan",
            horizon,
            teacher=NativeOfflineTeacher(kernel=kernel, horizon=horizon),
        ))
    for index, path in enumerate(model_paths):
        ranker = DistilledRanker(path, threads=threads)
        if ranker.scope != dict(expected_scope):
            raise ValueError(f"oracle ranker scope mismatch: {path}")
        if not source_compatible(ranker.compatible_headless_sources, dict(runtime_source)):
            raise ValueError(f"oracle ranker source mismatch: {path}")
        digest = _sha256(path)
        result.append(Continuation(
            f"ranker-{index:02d}-{digest[:12]}",
            "distilled-ranker",
            12,
            ranker=ranker,
            ranker_sha256=digest,
        ))
    return tuple(result)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint-sequence", type=int, action="append", required=True)
    parser.add_argument("--branch-frames", type=int, default=600)
    parser.add_argument("--planner-horizon", type=int, action="append")
    parser.add_argument("--model", type=Path, action="append")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--allow-dirty-code", action="store_true")
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
    horizons = tuple(sorted(set(args.planner_horizon or (4, 12, 30, 60))))
    if args.branch_frames <= 0 or any(horizon < HARD_HORIZON for horizon in horizons):
        parser.error("branch and planner horizons are outside safe bounds")
    if not 1 <= args.threads <= 12:
        parser.error("threads must be in 1..12")
    run = args.run.resolve()
    manifest, rows = _load_run(run)
    sequences = tuple(sorted(set(args.checkpoint_sequence)))
    if any(sequence <= 0 or sequence >= len(rows) for sequence in sequences):
        parser.error("checkpoint sequence is outside the reconstructable run")
    if any(not rows[sequence].get("legal_actions") for sequence in sequences):
        parser.error("checkpoint sequence has no native-safe first action")
    scope_data = manifest["scope"]
    scope = HeadlessScope(
        int(scope_data["difficulty"]),
        int(scope_data["character"]),
        int(scope_data["shot_type"]),
        int(scope_data["stage"]),
    )
    binary = args.binary.resolve()
    code_source = _repository_provenance(root)
    if not code_source["clean"] and not args.allow_dirty_code:
        parser.error("feasibility benchmark requires a clean repository checkout")
    runtime_source = _runtime_provenance(binary)
    kernel = NativeKernel()
    continuations = _continuations(
        horizons=horizons,
        model_paths=tuple(path.resolve() for path in (args.model or ())),
        kernel=kernel,
        threads=args.threads,
        expected_scope=scope_data,
        runtime_source=runtime_source,
    )
    labels = []
    with tempfile.TemporaryDirectory(prefix="th06-feasibility-oracle-") as raw:
        workspace = Path(raw)
        server = HeadlessForkserver(
            binary=binary,
            game_directory=args.game_directory.resolve(),
            scope=scope,
            seed=int(manifest["initial_seed"]),
        )
        try:
            if server.start() != 1:
                raise ValueError("unexpected stage-entry checkpoint tick")
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
                        continuations=continuations,
                        kernel=kernel,
                        seed=int(manifest["initial_seed"]),
                    ))
                finally:
                    server.leave_checkpoint()
        finally:
            server.close()
    result = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "interpretation": (
            "empirical feasibility lower bound; oracle-no-witness is not an "
            "infeasibility proof"
        ),
        "scope": scope_data,
        "initial_seed": manifest["initial_seed"],
        "input_run": str(run),
        "input_source": manifest["source"],
        "input_corpus": {
            "manifest_sha256": _sha256(run / "manifest.json"),
            "transitions": manifest["files"]["transitions"],
        },
        "code_source": code_source,
        "runtime_source": runtime_source,
        "branch_frames": args.branch_frames,
        "continuations": [item.describe() for item in continuations],
        "checkpoints": labels,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
