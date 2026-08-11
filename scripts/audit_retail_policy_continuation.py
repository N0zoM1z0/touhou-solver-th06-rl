#!/usr/bin/env python3
"""Audit factual frozen-UCB continuation from an exact retail COW checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from th06_rl.headless import HeadlessScope
from th06_rl.headless_corpus import retail_policy_source_context_id
from th06_rl.headless_forkserver import HeadlessForkserver
from th06_rl.headless_geometry import (
    HARD_HORIZON,
    HeadlessAuthorityUnavailable,
    action_from_input,
    certify_lowered_headless_actions,
    lower_headless_hazards,
    reactive_headless_action,
)
from th06_rl.native import NativeKernel
from th06_rl.policies.adaptive import AdaptivePolicy
from th06_rl.policy_api import PolicyContext
from th06_rl.wine_risk import FROZEN_INCUMBENT_POLICY_ID, load_first_failure_prefix

try:
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
except ModuleNotFoundError:  # Imported as scripts.audit_retail_policy_continuation.
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


REPORT_SCHEMA = "th06-rl-retail-policy-continuation-audit-v1"
RETAIL_DELIVERY_DELAYS = (0, 1, 2, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_keys(policy: AdaptivePolicy, context: PolicyContext) -> dict[str, str]:
    return {
        "coarse": policy._context_key(context),
        "middle": policy._middle_context_key(context),
        "fine": policy._fine_context_key(context),
    }


def _is_recorded_policy_call(
    transition: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> bool:
    """Exclude delivery-only rows that carried an earlier proposal.

    An input lease can retain ``proposed_action`` in the transition even
    though the policy was not called and the decision has no reactive
    baseline. Replaying it would advance immutable UCB state on a fictitious
    call and can fail before an otherwise valid later checkpoint.
    """
    return (
        transition.get("proposed_action") is not None and decision.get("reason") == "ok"
    )


def _source_policy_context(
    observation: Mapping[str, Any],
    *,
    scope: tuple[int, int, int, int],
    kernel: NativeKernel,
    phase_state: dict[str, int | str],
    horizon: int,
) -> tuple[PolicyContext, dict[str, Any]]:
    hazards = lower_headless_hazards(observation, horizon)
    prepared = kernel.prepare_hazards(hazards)
    hard_hazards = prepared.prefix(HARD_HORIZON)
    hard = certify_lowered_headless_actions(
        observation,
        hard_hazards,
        kernel=kernel,
        delivery_delays=RETAIL_DELIVERY_DELAYS,
    )
    if not hard:
        raise HeadlessAuthorityUnavailable("retail-envelope Hard safe set empty")
    lookahead = certify_lowered_headless_actions(
        observation,
        prepared,
        kernel=kernel,
        delivery_delays=RETAIL_DELIVERY_DELAYS,
        candidates=tuple(item.action for item in hard),
    )
    legal = lookahead or hard
    baseline = reactive_headless_action(observation, legal)
    source_context = retail_policy_source_context_id(observation)
    frame = int(observation["game_frame"])
    if phase_state["context"] != source_context:
        phase_state["context"] = source_context
        phase_state["start_frame"] = frame
    phase_elapsed = frame - int(phase_state["start_frame"])
    player = observation["player"]
    if not isinstance(player, Mapping):
        raise HeadlessAuthorityUnavailable("headless player is incoherent")
    context = PolicyContext(
        frame=frame,
        scope=scope,
        source_context=source_context,
        baseline_action=baseline.name,
        locally_admissible_actions=tuple(item.action.name for item in legal),
        player_x=float(player["x"]),
        player_y=float(player["y"]),
        power=int(observation["power"]),
        bullet_count=len(observation["bullets"]),
        laser_count=len(observation["lasers"]),
        hard_action_count=len(hard),
        exploration_rate=0.0,
        current_action=action_from_input(int(observation["input"])).name,
        hard_admissible_actions=tuple(item.action.name for item in hard),
        phase_elapsed_frames=phase_elapsed,
        hard_action_evaluations=tuple(
            (
                item.action.name,
                item.min_clearance,
                item.final_x,
                item.final_y,
            )
            for item in hard
        ),
    )
    return context, {
        "hard_actions": list(context.hard_admissible_actions),
        "legal_actions": list(context.locally_admissible_actions),
        "baseline_action": context.baseline_action,
        "source_context": source_context,
        "phase_elapsed_frames": phase_elapsed,
    }


def audit_factual_continuation(
    run_directory: Path,
    *,
    checkpoint_sequence: int,
    policy_state_path: Path,
    expected_policy_state_sha256: str,
    expected_scope: tuple[int, int, int, int],
    expected_executable_sha256: str,
    expected_native_kernel_sha256: str,
    expected_source_commit: str,
    expected_source_binary_sha256: str,
    binary: Path,
    game_directory: Path,
    branch_frames: int = 120,
    horizon: int = 12,
    initial_seed: int = 0,
) -> dict[str, Any]:
    if checkpoint_sequence < 0 or branch_frames <= 0 or horizon < HARD_HORIZON:
        raise ValueError("checkpoint or horizons are outside safe bounds")
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
    frames, frame_evidence = _verified_stream_rows(run_directory, manifest, "frames")
    if checkpoint_sequence >= len(transitions):
        raise ValueError("checkpoint is outside the retail prefix")

    policy = AdaptivePolicy()
    policy.import_state(_object(policy_state_path))
    replay_mismatches = []
    replay_calls = 0
    for sequence in range(checkpoint_sequence):
        transition = transitions[sequence]
        decision = frames[sequence].get("decision")
        if not isinstance(decision, dict):
            raise TypeError(f"retail decision {sequence} is absent")
        if not _is_recorded_policy_call(transition, decision):
            continue
        proposed = transition["proposed_action"]
        context = recorded_policy_context(transition, decision)
        selected = policy.decide(context)
        replay_calls += 1
        if selected.action != proposed and len(replay_mismatches) < 20:
            replay_mismatches.append(
                {
                    "sequence": sequence,
                    "frame": context.frame,
                    "recorded": proposed,
                    "replayed": selected.action,
                }
            )

    checkpoint_transition = transitions[checkpoint_sequence]
    checkpoint_frame = frames[checkpoint_sequence]
    checkpoint_decision = checkpoint_frame.get("decision")
    checkpoint_snapshot = checkpoint_frame.get("snapshot")
    if not isinstance(checkpoint_decision, dict) or not isinstance(
        checkpoint_snapshot, dict
    ):
        raise TypeError("retail checkpoint evidence is absent")
    retail_context = recorded_policy_context(checkpoint_transition, checkpoint_decision)
    checkpoint_tick = retail_context.frame
    phase_state: dict[str, int | str] = {
        "context": retail_context.source_context,
        "start_frame": checkpoint_tick - retail_context.phase_elapsed_frames,
    }
    kernel = NativeKernel()
    action_rows = []
    context_mismatches = []
    policy_action_mismatches = []
    checkpoint_state_difference = None
    termination_reason = "branch-error"
    terminal_tick = checkpoint_tick
    session_active = False

    with tempfile.TemporaryDirectory(prefix="th06-retail-policy-cow-") as raw:
        workspace = Path(raw)
        actions = workspace / "retail-prefix-actions.txt"
        actions.write_text(render_action_file(stream), encoding="ascii")
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
                actions_path=actions,
            )
            try:
                observation = server.begin_step_session(
                    terminal_tick=checkpoint_tick + branch_frames
                )
                session_active = True
                checkpoint_state_difference = first_difference(
                    _retail_state(checkpoint_snapshot),
                    _source_state(observation),
                    absolute_tolerance=1e-6,
                )
                sequence = checkpoint_sequence
                while observation.get("terminal_reason") is None:
                    terminal_tick = int(observation["tick"])
                    if terminal_tick >= checkpoint_tick + branch_frames:
                        termination_reason = "tick-limit"
                        break
                    try:
                        source_context, diagnostics = _source_policy_context(
                            observation,
                            scope=expected_scope,
                            kernel=kernel,
                            phase_state=phase_state,
                            horizon=horizon,
                        )
                    except HeadlessAuthorityUnavailable as error:
                        termination_reason = f"authority-failure:{error}"
                        server.abort_step_session()
                        session_active = False
                        break

                    expected_transition = (
                        transitions[sequence] if sequence < len(transitions) else None
                    )
                    expected_frame = (
                        frames[sequence] if sequence < len(frames) else None
                    )
                    expected_context = None
                    expected_action = None
                    if expected_transition is not None and expected_frame is not None:
                        expected_decision = expected_frame.get("decision")
                        if isinstance(expected_decision, dict):
                            expected_context = recorded_policy_context(
                                expected_transition, expected_decision
                            )
                            expected_action = expected_transition.get("proposed_action")
                    source_keys = _policy_keys(policy, source_context)
                    expected_keys = (
                        None
                        if expected_context is None
                        else _policy_keys(policy, expected_context)
                    )
                    if expected_keys is not None and source_keys != expected_keys:
                        context_mismatches.append(
                            {
                                "sequence": sequence,
                                "tick": terminal_tick,
                                "source_keys": source_keys,
                                "retail_keys": expected_keys,
                                "source": diagnostics,
                            }
                        )
                    selected = policy.decide(source_context)
                    if (
                        expected_action is not None
                        and selected.action != expected_action
                    ):
                        policy_action_mismatches.append(
                            {
                                "sequence": sequence,
                                "tick": terminal_tick,
                                "source": selected.action,
                                "retail": expected_action,
                            }
                        )
                    action_rows.append(
                        {
                            "sequence": sequence,
                            "tick": terminal_tick,
                            "action": selected.action,
                            "expected_retail_action": expected_action,
                            "policy_keys_equal": expected_keys == source_keys,
                            **diagnostics,
                        }
                    )
                    observation = server.step_session(selected.action)
                    sequence += 1
                if session_active:
                    result = server.finish_step_session()
                    session_active = False
                    terminal = result.terminal_observation
                    terminal_tick = int(terminal["tick"])
                    termination_reason = str(terminal["terminal_reason"])
            finally:
                if session_active:
                    server.abort_step_session()
                server.leave_checkpoint()
        finally:
            server.close()

    expected_terminal = {
        "kind": prefix.failure_kind,
        "frame": prefix.failure_frame,
    }
    terminal_matches = (
        termination_reason == "authority-failure:retail-envelope Hard safe set empty"
        and terminal_tick == prefix.failure_frame
    )
    passed = (
        not replay_mismatches
        and checkpoint_state_difference is None
        and not context_mismatches
        and not policy_action_mismatches
        and terminal_matches
    )
    return {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "scope": list(expected_scope),
        "input": {
            "run": str(run_directory),
            "run_id": prefix.run_id,
            "manifest_sha256": prefix.manifest_sha256,
            "run_sha256": prefix.run_sha256,
            "checkpoint_sequence": checkpoint_sequence,
            "checkpoint_tick": checkpoint_tick,
            "policy_state": str(policy_state_path),
            "policy_state_sha256": _sha256(policy_state_path),
            "transition_shards": transition_evidence,
            "frame_shards": frame_evidence,
        },
        "source": source,
        "policy_restore": {
            "calls_before_checkpoint": replay_calls,
            "action_mismatches": replay_mismatches,
        },
        "checkpoint_state_difference_at_1e_6": checkpoint_state_difference,
        "continuation": {
            "branch_frames": branch_frames,
            "horizon": horizon,
            "delivery_delays": list(RETAIL_DELIVERY_DELAYS),
            "immutable_policy_observe_suppressed": True,
            "actions": action_rows,
            "context_key_mismatches": context_mismatches[:20],
            "policy_action_mismatches": policy_action_mismatches[:20],
            "termination_reason": termination_reason,
            "terminal_tick": terminal_tick,
            "expected_retail_terminal": expected_terminal,
            "terminal_matches": terminal_matches,
        },
        "evidence_boundary": {
            "factual_feasibility_audit_only": True,
            "counterfactual_candidate_selection": False,
            "training_corpus": False,
            "promotion_authority": False,
            "complete_wine_stage_hit_count_required": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--checkpoint-sequence", type=int, required=True)
    parser.add_argument("--policy-state", type=Path, required=True)
    parser.add_argument("--expected-policy-state-sha256", required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--expected-native-kernel-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-binary-sha256", required=True)
    parser.add_argument("--branch-frames", type=int, default=120)
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
        report = audit_factual_continuation(
            args.run_directory,
            checkpoint_sequence=args.checkpoint_sequence,
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
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "passed": report["passed"],
                "termination_reason": report["continuation"]["termination_reason"],
                "terminal_tick": report["continuation"]["terminal_tick"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
