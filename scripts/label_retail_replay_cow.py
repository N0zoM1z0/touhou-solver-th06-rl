#!/usr/bin/env python3
"""Run native-safe COW branches from exact retail-Wine replay checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from th06_rl.headless import HeadlessScope
from th06_rl.headless_corpus import NativeOfflineTeacher, canonical_observation_sha256
from th06_rl.headless_forkserver import HeadlessForkserver
from th06_rl.headless_geometry import (
    HEADLESS_DELIVERY_DELAYS,
    certify_lowered_headless_actions,
)
from th06_rl.native import NativeKernel
from th06_rl.wine_risk import FROZEN_INCUMBENT_POLICY_ID

try:
    from audit_retail_source_replay import _retail_state, _source_state
    from compare_headless_traces import first_difference
    from export_wine_action_stream import (
        _object,
        _verified_stream_rows,
        export_wine_action_stream,
    )
    from label_headless_cow_counterfactuals import (
        _certify,
        _runtime_provenance,
        label_checkpoint,
    )
    from run_source_platform_differential import (
        render_action_file,
        render_dialogue_input_file,
    )
except ModuleNotFoundError:  # Imported as scripts.label_retail_replay_cow.
    from scripts.audit_retail_source_replay import _retail_state, _source_state
    from scripts.compare_headless_traces import first_difference
    from scripts.export_wine_action_stream import (
        _object,
        _verified_stream_rows,
        export_wine_action_stream,
    )
    from scripts.label_headless_cow_counterfactuals import (
        _certify,
        _runtime_provenance,
        label_checkpoint,
    )
    from scripts.run_source_platform_differential import (
        render_action_file,
        render_dialogue_input_file,
    )


SCHEMA = "th06-rl-retail-replay-cow-v2"
RETAIL_NATIVE_DELIVERY_DELAYS = (0, 1, 2, 3)


def retail_checkpoint_contract(
    transition: Mapping[str, Any],
    frame_row: Mapping[str, Any],
) -> dict[str, Any]:
    decision = frame_row.get("decision")
    snapshot = frame_row.get("snapshot")
    scope = frame_row.get("scope")
    if (
        not isinstance(decision, Mapping)
        or not isinstance(snapshot, Mapping)
        or not isinstance(scope, Mapping)
    ):
        raise TypeError("retail checkpoint lacks snapshot/decision/scope evidence")
    hard = decision.get("hard_actions")
    if not isinstance(hard, list) or not hard:
        raise ValueError("retail checkpoint has no native hard action set")
    hard_names: list[str] = []
    for index, row in enumerate(hard):
        if not isinstance(row, list) or len(row) < 1 or not isinstance(row[0], str):
            raise TypeError(f"retail hard action {index} is malformed")
        hard_names.append(row[0])
    if len(set(hard_names)) != len(hard_names):
        raise ValueError("retail hard action set contains duplicates")
    published = transition.get("published_action")
    current = decision.get("current_action")
    factual = current if published is None else published
    baseline = decision.get("baseline_action")
    if not isinstance(factual, str) or factual not in hard_names:
        raise ValueError("retail factual action is not native-hard-safe")
    if not isinstance(baseline, str):
        raise ValueError("retail checkpoint baseline action is absent")
    phase_id = scope.get("phase_id")
    if not isinstance(phase_id, str) or not phase_id:
        raise ValueError("retail checkpoint source context is absent")
    frame = int(snapshot.get("frame", -1))
    if frame < 2:
        raise ValueError("retail checkpoint frame is not reconstructable")
    return {
        "tick": frame,
        "source_context": phase_id,
        "factual_action": factual,
        "local_teacher_action": baseline,
        "native_legal_actions": hard_names,
        "retail_state": _retail_state(snapshot),
        "snapshot_id": frame_row.get("snapshot_id"),
    }


def label_retail_checkpoints(
    run_directory: Path,
    *,
    sequences: Sequence[int],
    expected_scope: tuple[int, int, int, int],
    expected_executable_sha256: str,
    expected_native_kernel_sha256: str,
    expected_policy_id: str,
    binary: Path,
    game_directory: Path,
    branch_frames: int,
    teacher_horizon: int,
    evaluated_first_actions: Sequence[str] | None = None,
    initial_seed: int = 0,
) -> dict[str, Any]:
    if branch_frames <= 0 or teacher_horizon < 4:
        raise ValueError("branch and teacher horizons are outside safe bounds")
    chosen = tuple(sorted(set(int(sequence) for sequence in sequences)))
    if not chosen:
        raise ValueError("at least one retail checkpoint sequence is required")
    run_directory = run_directory.resolve()
    binary = binary.resolve()
    game_directory = game_directory.resolve()
    stream = export_wine_action_stream(
        run_directory,
        expected_scope=expected_scope,
        expected_executable_sha256=expected_executable_sha256,
        expected_native_kernel_sha256=expected_native_kernel_sha256,
        expected_policy_id=expected_policy_id,
        max_source_tick=None,
        initial_seed=initial_seed,
    )
    manifest = _object(run_directory / "manifest.json")
    transitions, transition_evidence = _verified_stream_rows(
        run_directory, manifest, "transitions"
    )
    frames, frame_evidence = _verified_stream_rows(run_directory, manifest, "frames")
    if any(sequence < 0 or sequence >= len(transitions) for sequence in chosen):
        raise ValueError("retail checkpoint sequence is outside the prefix")
    contracts = {
        sequence: retail_checkpoint_contract(transitions[sequence], frames[sequence])
        for sequence in chosen
    }
    scope = HeadlessScope(*expected_scope)
    kernel = NativeKernel()
    teacher = NativeOfflineTeacher(kernel=kernel, horizon=teacher_horizon)
    labels: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="th06-retail-cow-") as raw:
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
            scope=scope,
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
            root_tick = server.start()
            if root_tick != 1:
                raise ValueError(f"unexpected retail replay root tick {root_tick}")
            for sequence in chosen:
                contract = contracts[sequence]
                checkpoint_tick = int(contract["tick"])
                server.enter_checkpoint(
                    terminal_tick=checkpoint_tick,
                    actions_path=actions,
                )
                try:
                    observation = server.begin_step_session(
                        terminal_tick=checkpoint_tick + 1
                    )
                    source_state = _source_state(observation)
                    difference = first_difference(
                        contract["retail_state"],
                        source_state,
                        absolute_tolerance=1e-6,
                    )
                    if difference is not None:
                        server.abort_step_session()
                        raise ValueError(
                            f"retail/source checkpoint mismatch at sequence {sequence}: "
                            f"{difference}"
                        )
                    # The reconstructed source branch publishes synchronously,
                    # but the recorded retail Hard certificate covered the
                    # asynchronous Windows adapter's complete bounded pickup
                    # set.  Compare like with like.  Using the source STEP set
                    # here falsely broadened boundary checkpoints by up to
                    # three movement ticks and rejected valid anchors.
                    certified, prepared = _certify(observation, kernel)
                    certified_names = [item.action.name for item in certified]
                    retail_delivery_certified = certify_lowered_headless_actions(
                        observation,
                        prepared,
                        kernel=kernel,
                        delivery_delays=RETAIL_NATIVE_DELIVERY_DELAYS,
                    )
                    retail_delivery_names = [
                        item.action.name for item in retail_delivery_certified
                    ]
                    if retail_delivery_names != contract["native_legal_actions"]:
                        server.abort_step_session()
                        only_retail = sorted(
                            set(contract["native_legal_actions"])
                            - set(retail_delivery_names)
                        )
                        only_source = sorted(
                            set(retail_delivery_names)
                            - set(contract["native_legal_actions"])
                        )
                        raise ValueError(
                            "retail/source native hard set mismatch under the retail "
                            f"delivery contract at sequence {sequence}: "
                            f"only_retail={only_retail}, only_source={only_source}"
                        )
                    requested_actions = (
                        ()
                        if evaluated_first_actions is None
                        else tuple(dict.fromkeys(evaluated_first_actions))
                    )
                    if any(
                        action not in retail_delivery_names
                        for action in requested_actions
                    ):
                        server.abort_step_session()
                        raise ValueError(
                            "requested first action is not native-safe under the "
                            f"retail delivery contract at sequence {sequence}"
                        )
                    digest = canonical_observation_sha256(observation)
                    terminal = server.step_session(str(contract["factual_action"]))
                    if terminal.get("terminal_reason") != "tick-limit":
                        server.finish_step_session()
                        raise ValueError(
                            f"retail factual action is not one-tick source-safe at sequence {sequence}"
                        )
                    server.finish_step_session()
                    row = {
                        "tick": checkpoint_tick,
                        "observation_sha256": digest,
                        "source_context": contract["source_context"],
                        "legal_actions": certified_names,
                        "behavior": {
                            "selected_action": contract["factual_action"],
                            "teacher_action": contract["local_teacher_action"],
                        },
                    }
                    label = label_checkpoint(
                        server=server,
                        row=row,
                        sequence=sequence,
                        branch_frames=branch_frames,
                        teacher=teacher,
                        kernel=kernel,
                        evaluated_first_actions=evaluated_first_actions,
                    )
                    label.update(
                        {
                            "retail_snapshot_id": contract["snapshot_id"],
                            "retail_source_state_match_at_1e_6": True,
                            "retail_source_native_hard_set_match": True,
                            "retail_native_delivery_delays": list(
                                RETAIL_NATIVE_DELIVERY_DELAYS
                            ),
                            "source_branch_delivery_delays": list(
                                HEADLESS_DELIVERY_DELAYS
                            ),
                        }
                    )
                    labels.append(label)
                finally:
                    server.leave_checkpoint()
        finally:
            server.close()
    provenance = stream.provenance or {}
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_run": str(run_directory),
        "input_run_id": provenance.get("run_id"),
        "input_manifest_sha256": provenance.get("manifest_sha256"),
        "input_run_sha256": provenance.get("run_sha256"),
        "scope": {
            "difficulty": scope.difficulty,
            "character": scope.character,
            "shot_type": scope.shot_type,
            "stage": scope.stage,
        },
        "source": _runtime_provenance(binary),
        "retail_replay": stream.as_object(),
        "retail_shards": {
            "transitions": transition_evidence,
            "frames": frame_evidence,
        },
        "branch_frames": branch_frames,
        "teacher_horizon": teacher_horizon,
        "delivery_contracts": {
            "retail_native_gate": list(RETAIL_NATIVE_DELIVERY_DELAYS),
            "source_step_branch": list(HEADLESS_DELIVERY_DELAYS),
        },
        "requested_first_actions": (
            None
            if evaluated_first_actions is None
            else list(dict.fromkeys(evaluated_first_actions))
        ),
        "checkpoint_sequences": list(chosen),
        "checkpoints": labels,
        "evidence_boundary": {
            "training_corpus": False,
            "promotion_authority": False,
            "purpose": "targeted counterfactual hypothesis generation from Wine failures",
            "native_gate_unchanged": True,
            "bomb_forbidden": True,
            "retail_shadow_required_before_activation": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--checkpoint-sequence", type=int, action="append", required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--expected-native-kernel-sha256", required=True)
    parser.add_argument(
        "--expected-policy-id",
        default=FROZEN_INCUMBENT_POLICY_ID,
    )
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--character", type=int, default=0)
    parser.add_argument("--shot-type", type=int, default=0)
    parser.add_argument("--stage", type=int, default=6)
    parser.add_argument("--initial-seed", type=int, default=0)
    parser.add_argument("--branch-frames", type=int, default=600)
    parser.add_argument("--teacher-horizon", type=int, default=12)
    parser.add_argument(
        "--first-action",
        action="append",
        help="evaluate only this native-legal first action; repeat for a targeted pair",
    )
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        document = label_retail_checkpoints(
            args.run_directory,
            sequences=args.checkpoint_sequence,
            expected_scope=(
                args.difficulty,
                args.character,
                args.shot_type,
                args.stage,
            ),
            expected_executable_sha256=args.expected_executable_sha256,
            expected_native_kernel_sha256=args.expected_native_kernel_sha256,
            expected_policy_id=args.expected_policy_id,
            binary=args.binary,
            game_directory=args.game_directory,
            branch_frames=args.branch_frames,
            teacher_horizon=args.teacher_horizon,
            evaluated_first_actions=args.first_action,
            initial_seed=args.initial_seed,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": document["schema"],
        "run_id": document["input_run_id"],
        "checkpoints": len(document["checkpoints"]),
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
