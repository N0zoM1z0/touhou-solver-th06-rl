#!/usr/bin/env python3
"""Branch one Wine checkpoint, then continue immutable frozen UCB in source."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from th06_rl.headless import HeadlessScope
from th06_rl.headless_forkserver import HeadlessForkserver
from th06_rl.headless_geometry import HeadlessAuthorityUnavailable
from th06_rl.native import NativeKernel
from th06_rl.policies.adaptive import AdaptivePolicy
from th06_rl.wine_risk import FROZEN_INCUMBENT_POLICY_ID, load_first_failure_prefix

try:
    from audit_retail_policy_continuation import (
        RETAIL_DELIVERY_DELAYS,
        _sha256,
        _source_policy_context,
    )
    from audit_retail_source_replay import _retail_state, _source_state
    from compare_headless_traces import first_difference
    from export_wine_action_stream import (
        _object,
        _verified_stream_rows,
        export_wine_action_stream,
    )
    from label_headless_cow_counterfactuals import _runtime_provenance
    from replay_wine_risk_guard import _context as recorded_policy_context
    from run_source_platform_differential import (
        render_action_file,
        render_dialogue_input_file,
    )
except ModuleNotFoundError:  # Imported as scripts.label_retail_policy_cow.
    from scripts.audit_retail_policy_continuation import (
        RETAIL_DELIVERY_DELAYS,
        _sha256,
        _source_policy_context,
    )
    from scripts.audit_retail_source_replay import _retail_state, _source_state
    from scripts.compare_headless_traces import first_difference
    from scripts.export_wine_action_stream import (
        _object,
        _verified_stream_rows,
        export_wine_action_stream,
    )
    from scripts.label_headless_cow_counterfactuals import _runtime_provenance
    from scripts.replay_wine_risk_guard import (
        _context as recorded_policy_context,
    )
    from scripts.run_source_platform_differential import (
        render_action_file,
        render_dialogue_input_file,
    )


SCHEMA = "th06-rl-retail-policy-cow-v1"


def _boundary_reserve(observation: Mapping[str, Any]) -> float:
    player = observation["player"]
    if not isinstance(player, Mapping):
        raise HeadlessAuthorityUnavailable("headless player is incoherent")
    x = float(player["x"])
    y = float(player["y"])
    return min(x - 8.0, 376.0 - x, y - 16.0, 432.0 - y)


def _action_sha256(actions: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{action}\n" for action in actions).encode("ascii")
    ).hexdigest()


def _restore_policy_before_checkpoint(
    transitions: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    *,
    checkpoint_sequence: int,
    policy_state: Mapping[str, Any],
) -> tuple[AdaptivePolicy, dict[str, Any]]:
    policy = AdaptivePolicy()
    policy.import_state(dict(policy_state))
    calls = 0
    mismatches = []
    for sequence in range(checkpoint_sequence):
        transition = transitions[sequence]
        proposed = transition.get("proposed_action")
        if proposed is None:
            continue
        decision = frames[sequence].get("decision")
        if not isinstance(decision, dict):
            raise TypeError(f"retail decision {sequence} is absent")
        context = recorded_policy_context(dict(transition), decision)
        selected = policy.decide(context)
        calls += 1
        if selected.action != proposed and len(mismatches) < 20:
            mismatches.append({
                "sequence": sequence,
                "frame": context.frame,
                "recorded": proposed,
                "replayed": selected.action,
            })
    if mismatches:
        raise ValueError(f"frozen policy restore mismatch: {mismatches[0]}")
    return policy, {"calls": calls, "action_mismatches": mismatches}


def label_policy_cow(
    run_directory: Path,
    *,
    checkpoint_sequence: int,
    first_actions: Sequence[str],
    policy_state_path: Path,
    expected_policy_state_sha256: str,
    expected_scope: tuple[int, int, int, int],
    expected_executable_sha256: str,
    expected_native_kernel_sha256: str,
    expected_source_commit: str,
    expected_source_binary_sha256: str,
    binary: Path,
    game_directory: Path,
    branch_frames: int = 600,
    horizon: int = 12,
    initial_seed: int = 0,
) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(str(action) for action in first_actions))
    if (
        checkpoint_sequence < 0
        or not requested
        or len(requested) != len(first_actions)
        or branch_frames <= 0
        or horizon < 4
    ):
        raise ValueError("checkpoint, actions, or horizons are invalid")
    run_directory = run_directory.resolve()
    policy_state_path = policy_state_path.resolve()
    binary = binary.resolve()
    game_directory = game_directory.resolve()
    if _sha256(policy_state_path) != expected_policy_state_sha256:
        raise ValueError("frozen policy state hash mismatch")
    source = _runtime_provenance(binary)
    if (
        source.get("clean") is not True
        or source.get("commit") != expected_source_commit
        or source.get("binary_sha256") != expected_source_binary_sha256
    ):
        raise ValueError("source runtime identity mismatch")

    prefix = load_first_failure_prefix(
        run_directory,
        expected_scope=expected_scope,
        expected_executable_sha256=expected_executable_sha256,
        expected_native_kernel_sha256=expected_native_kernel_sha256,
        expected_policy_id=FROZEN_INCUMBENT_POLICY_ID,
    )
    stream = export_wine_action_stream(
        run_directory,
        expected_scope=expected_scope,
        expected_executable_sha256=expected_executable_sha256,
        expected_native_kernel_sha256=expected_native_kernel_sha256,
        expected_policy_id=FROZEN_INCUMBENT_POLICY_ID,
        max_source_tick=None,
        initial_seed=initial_seed,
    )
    manifest = _object(run_directory / "manifest.json")
    transitions, transition_evidence = _verified_stream_rows(
        run_directory, manifest, "transitions"
    )
    frames, frame_evidence = _verified_stream_rows(
        run_directory, manifest, "frames"
    )
    if checkpoint_sequence >= len(transitions):
        raise ValueError("checkpoint is outside the retail prefix")
    policy, restore = _restore_policy_before_checkpoint(
        transitions,
        frames,
        checkpoint_sequence=checkpoint_sequence,
        policy_state=_object(policy_state_path),
    )
    checkpoint_transition = transitions[checkpoint_sequence]
    checkpoint_frame = frames[checkpoint_sequence]
    checkpoint_decision = checkpoint_frame.get("decision")
    checkpoint_snapshot = checkpoint_frame.get("snapshot")
    if not isinstance(checkpoint_decision, dict) or not isinstance(
        checkpoint_snapshot, dict
    ):
        raise TypeError("retail checkpoint evidence is absent")
    retail_context = recorded_policy_context(
        checkpoint_transition, checkpoint_decision
    )
    factual_action = str(checkpoint_transition.get("proposed_action"))
    if factual_action not in requested:
        raise ValueError("requested actions omit the factual incumbent")
    checkpoint_tick = retail_context.frame
    kernel = NativeKernel()
    outcomes = []
    checkpoint_contract: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="th06-retail-policy-cow-") as raw:
        workspace = Path(raw)
        actions_path = workspace / "retail-prefix-actions.txt"
        actions_path.write_text(render_action_file(stream), encoding="ascii")
        dialogue_inputs = None
        if stream.retail_dialogue_inputs:
            dialogue_inputs = workspace / "retail-dialogue-inputs.txt"
            dialogue_inputs.write_text(
                render_dialogue_input_file(stream), encoding="ascii"
            )
        server = HeadlessForkserver(
            binary=binary,
            game_directory=game_directory,
            scope=HeadlessScope(*expected_scope),
            seed=stream.initial_seed,
            auto_shoot=stream.auto_shoot,
            stage_rng_seed=stream.stage_rng_seed,
            auto_shoot_after_tick=stream.auto_shoot_after_tick,
            retail_dialogue_control=stream.retail_dialogue_control,
            retail_dialogue_control_after_tick=(
                stream.retail_dialogue_control_after_tick
            ),
            retail_dialogue_inputs_path=dialogue_inputs,
        )
        try:
            if server.start() != 1:
                raise ValueError("unexpected source root tick")
            server.enter_checkpoint(
                terminal_tick=checkpoint_tick,
                actions_path=actions_path,
            )
            try:
                for first_action in requested:
                    phase_state: dict[str, int | str] = {
                        "context": retail_context.source_context,
                        "start_frame": (
                            checkpoint_tick - retail_context.phase_elapsed_frames
                        ),
                    }
                    branch_policy = deepcopy(policy)
                    session_active = False
                    observation = server.begin_step_session(
                        terminal_tick=checkpoint_tick + branch_frames
                    )
                    session_active = True
                    try:
                        difference = first_difference(
                            _retail_state(checkpoint_snapshot),
                            _source_state(observation),
                            absolute_tolerance=1e-6,
                        )
                        if difference is not None:
                            raise ValueError(
                                f"retail/source checkpoint mismatch: {difference}"
                            )
                        context, diagnostics = _source_policy_context(
                            observation,
                            scope=expected_scope,
                            kernel=kernel,
                            phase_state=phase_state,
                            horizon=horizon,
                        )
                        direct = branch_policy.decide(context)
                        if direct.action != factual_action:
                            raise ValueError(
                                "restored checkpoint policy action mismatch"
                            )
                        if first_action not in context.locally_admissible_actions:
                            raise ValueError(
                                f"first action {first_action} is not locally admissible"
                            )
                        if checkpoint_contract is None:
                            checkpoint_contract = {
                                "sequence": checkpoint_sequence,
                                "tick": checkpoint_tick,
                                "source_context": context.source_context,
                                "phase_elapsed_frames": context.phase_elapsed_frames,
                                "factual_action": factual_action,
                                "baseline_action": context.baseline_action,
                                "hard_actions": list(
                                    context.hard_admissible_actions
                                ),
                                "legal_actions": list(
                                    context.locally_admissible_actions
                                ),
                                "retail_delivery_delays": list(
                                    RETAIL_DELIVERY_DELAYS
                                ),
                                "retail_source_state_match_at_1e_6": True,
                                "restored_policy_action_match": True,
                            }
                        issued = [first_action]
                        checkpoint_deaths = int(observation["deaths"])
                        minimum_width = len(context.hard_admissible_actions)
                        observation = server.step_session(first_action)
                        termination_reason = None
                        authority_reason = None
                        while observation.get("terminal_reason") is None:
                            try:
                                context, _diagnostics = _source_policy_context(
                                    observation,
                                    scope=expected_scope,
                                    kernel=kernel,
                                    phase_state=phase_state,
                                    horizon=horizon,
                                )
                            except HeadlessAuthorityUnavailable as error:
                                authority_reason = str(error)
                                termination_reason = "authority-failure"
                                result = server.abort_step_session()
                                session_active = False
                                terminal_observation = result.terminal_observation
                                terminal_tick = int(observation["tick"])
                                terminal_reserve = _boundary_reserve(observation)
                                break
                            minimum_width = min(
                                minimum_width,
                                len(context.hard_admissible_actions),
                            )
                            selected = branch_policy.decide(context)
                            issued.append(selected.action)
                            observation = server.step_session(selected.action)
                        else:
                            result = server.finish_step_session()
                            session_active = False
                            terminal_observation = result.terminal_observation
                            termination_reason = str(
                                terminal_observation["terminal_reason"]
                            )
                            terminal_tick = int(terminal_observation["tick"])
                            terminal_reserve = _boundary_reserve(
                                terminal_observation
                            )
                        expected_factual = [
                            str(row.get("proposed_action"))
                            for row in transitions[checkpoint_sequence:]
                            if row.get("proposed_action") is not None
                        ]
                        factual_suffix_matches = (
                            first_action != factual_action
                            or issued == expected_factual[:len(issued)]
                        )
                        outcomes.append({
                            "first_action": first_action,
                            "termination_reason": termination_reason,
                            "authority_reason": authority_reason,
                            "terminal_tick": terminal_tick,
                            "survival_ticks": terminal_tick - checkpoint_tick,
                            "actions_issued": len(issued),
                            "action_sha256": _action_sha256(issued),
                            "first_32_actions": issued[:32],
                            "action_counts": dict(sorted(Counter(issued).items())),
                            "minimum_native_legal_actions": minimum_width,
                            "terminal_boundary_reserve": terminal_reserve,
                            "physical_deaths_delta": (
                                int(terminal_observation["deaths"])
                                - checkpoint_deaths
                            ),
                            "factual_suffix_matches": factual_suffix_matches,
                        })
                    finally:
                        if session_active:
                            server.abort_step_session()
            finally:
                server.leave_checkpoint()
        finally:
            server.close()

    assert checkpoint_contract is not None
    factual = next(
        outcome for outcome in outcomes
        if outcome["first_action"] == factual_action
    )
    factual_regression = {
        "action_suffix_matches": factual["factual_suffix_matches"],
        "termination_reason": factual["termination_reason"],
        "terminal_tick": factual["terminal_tick"],
        "expected_failure_kind": prefix.failure_kind,
        "expected_failure_tick": prefix.failure_frame,
        "passed": (
            factual["factual_suffix_matches"] is True
            and factual["termination_reason"] == "authority-failure"
            and factual["terminal_tick"] == prefix.failure_frame
        ),
    }
    if not factual_regression["passed"]:
        raise ValueError("factual frozen-UCB branch regression failed")
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "run": str(run_directory),
            "run_id": prefix.run_id,
            "manifest_sha256": prefix.manifest_sha256,
            "run_sha256": prefix.run_sha256,
            "transition_shards": transition_evidence,
            "frame_shards": frame_evidence,
        },
        "scope": {
            "difficulty": expected_scope[0],
            "character": expected_scope[1],
            "shot_type": expected_scope[2],
            "stage": expected_scope[3],
        },
        "source": source,
        "policy": {
            "id": FROZEN_INCUMBENT_POLICY_ID,
            "state": str(policy_state_path),
            "state_sha256": _sha256(policy_state_path),
            "immutable_observe_suppressed": True,
            "restore": restore,
        },
        "branch_frames": branch_frames,
        "horizon": horizon,
        "requested_first_actions": list(requested),
        "checkpoint": checkpoint_contract,
        "factual_regression": factual_regression,
        "outcomes": outcomes,
        "evidence_boundary": {
            "training_corpus": False,
            "promotion_authority": False,
            "native_gate_unchanged": True,
            "bomb_forbidden": True,
            "wine_shadow_required": True,
            "complete_wine_stage_hit_count_required": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--checkpoint-sequence", type=int, required=True)
    parser.add_argument("--first-action", action="append", required=True)
    parser.add_argument("--policy-state", type=Path, required=True)
    parser.add_argument("--expected-policy-state-sha256", required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--expected-native-kernel-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-binary-sha256", required=True)
    parser.add_argument("--branch-frames", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--initial-seed", type=int, default=0)
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
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = label_policy_cow(
            args.run_directory,
            checkpoint_sequence=args.checkpoint_sequence,
            first_actions=args.first_action,
            policy_state_path=args.policy_state,
            expected_policy_state_sha256=args.expected_policy_state_sha256,
            expected_scope=(3, 0, 0, 6),
            expected_executable_sha256=args.expected_executable_sha256,
            expected_native_kernel_sha256=args.expected_native_kernel_sha256,
            expected_source_commit=args.expected_source_commit,
            expected_source_binary_sha256=args.expected_source_binary_sha256,
            binary=args.binary,
            game_directory=args.game_directory,
            branch_frames=args.branch_frames,
            horizon=args.horizon,
            initial_seed=args.initial_seed,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": report["schema"],
        "run_id": report["input"]["run_id"],
        "checkpoint": report["checkpoint"]["sequence"],
        "outcomes": len(report["outcomes"]),
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
