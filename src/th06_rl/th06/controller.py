"""Restartable physical TH06 shell around the phase-agnostic native planner."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from ..corpus import (
    FRAME_BUDGET_MS,
    CorpusError,
    CorpusRecorder,
    DialogueDeliverySample,
    FrameEvidence,
    RunMetadata,
)
from ..native import NativeKernel
from ..policy_api import PolicyContext, PolicyFailureEvent, PolicyOutcome
from ..policy_loader import HotReloadPolicy
from ..policy_transaction import StagePolicyTransaction
from .background_activity import BackgroundActivityLease
from .background_input import BackgroundInputBridge
from .background_keyboard import BackgroundKeyboard
from .control_capture import (
    CONTROL_CAPTURE_TIER,
    observe_passive_control_clock,
    read_control_snapshot,
    read_passive_input_delivery,
)
from .donor import enable_donor_imports
from .menu import (
    MenuNavigationError,
    start_reimu_a_practice,
    start_reimu_a_route,
)
from .learning_adapter import project_learning_features
from .source import (
    AuthorityUnavailable,
    automatic_source_context,
    core_action_from_input,
    donor_action,
    kinematics_from_snapshot,
    lower_observed_hazards,
)
from .system_health import GIB, below_commit_reserve, read_system_memory


ACTIVE_PLAYER_STATES = (0, 3)
RELOAD_POLL_SECONDS = 60.0
HEALTH_SAMPLE_SECONDS = 1.0
HEALTH_TRACE_SECONDS = 10.0
DIALOGUE_PROBE_SECONDS = 1.0 / 60.0
LOW_COMMIT_EXIT_CODE = 75
MENU_RETRY_EXIT_CODE = 77
DIFFICULTIES = {"normal": 1, "hard": 2, "lunatic": 3}
SUPERVISOR_GAMEPLAY = 2
SUPERVISOR_GAMEPLAY_REINIT = 3
SUPERVISOR_ENDING = 10


class RouteTrial:
    """Complete only when a played ordinary route reaches the Ending state."""

    def __init__(self) -> None:
        self.gameplay_seen = False

    def observe_supervisor(self, current_state: int) -> bool:
        if current_state in (SUPERVISOR_GAMEPLAY, SUPERVISOR_GAMEPLAY_REINIT):
            self.gameplay_seen = True
        return self.gameplay_seen and current_state == SUPERVISOR_ENDING


def _advance_route_scope(
    expected: tuple[int, int, int, int | None],
    observed: tuple[int, int, int, int | None],
) -> tuple[int, int, int, int | None]:
    """Accept only the source-defined next Stage in the same learning scope."""
    expected_stage = expected[3]
    observed_stage = observed[3]
    if (
        observed[:3] != expected[:3]
        or not isinstance(expected_stage, int)
        or observed_stage != expected_stage + 1
        or not 1 <= observed_stage <= 6
    ):
        raise AuthorityUnavailable(f"route scope changed unexpectedly to {observed}")
    return observed


def _physical_bomb(snapshot) -> bool:
    return bool(
        snapshot.input_mask & 0x02
        or getattr(snapshot, "bomb_active", False)
        or (
            getattr(snapshot, "player_attack", None) is not None
            and snapshot.player_attack.bomb_active
        )
    )


def _valid_executable_basename(value: str) -> bool:
    return bool(
        value
        and value not in (".", "..")
        and "/" not in value
        and "\\" not in value
        and value.lower().endswith(".exe")
    )


def _authority_loss(reason: str) -> bool:
    return (
        reason.startswith("authority-stop:")
        and not _control_dead_end(reason)
        and reason
        not in (
            "authority-stop:physical HIT",
            "authority-stop:physical Bomb state/input",
        )
    )


def _control_dead_end(reason: str) -> bool:
    return reason in (
        "control-dead-end:Hard safe set empty",
        "control-dead-end:local forecast has no safe continuation",
        "authority-stop:Hard safe set empty",
        "authority-stop:local forecast has no safe continuation",
    )


def _snapshot_scope(snapshot) -> tuple[int, int, int, int | None]:
    shot_type = getattr(snapshot, "shot_type", None)
    if shot_type is None:
        attack = getattr(snapshot, "player_attack", None)
        shot_type = attack.shot_type if attack is not None else None
    return (
        snapshot.difficulty,
        snapshot.character,
        shot_type,
        snapshot.stage,
    )


def _anchor_partition(source_context: str) -> str:
    """Coarse source boundary for exhaustive roots, never movement control."""
    if source_context.startswith("boss:"):
        return source_context
    if source_context == "timeline-complete":
        return source_context
    return "timeline"


def _reload_poll_window(snapshot) -> bool:
    """Poll source metadata only outside active physical control."""
    return bool(
        snapshot.in_menu
        or snapshot.time_stopped
        or snapshot.player_state not in ACTIVE_PLAYER_STATES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _boundary_reserve(x: float, y: float) -> float:
    return min(x - 8.0, 376.0 - x, y - 16.0, 432.0 - y)


def _reactive_baseline(candidates, current_action):
    """Constant-time cold-start ranking inside the native legal frontier."""
    if not candidates:
        raise ValueError("reactive baseline requires a legal action")
    return max(
        candidates,
        key=lambda item: (
            item.min_clearance,
            _boundary_reserve(item.final_x, item.final_y),
            item.action == current_action,
            item.action.dx == 0 and item.action.dy == 0,
            item.action.focused,
            item.action.name,
        ),
    )


def _code_commit(repository: Path) -> str:
    head = (repository / ".git/HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = repository / ".git" / head[5:]
    return reference.read_text(encoding="utf-8").strip()


def run(args: argparse.Namespace) -> int:
    if os.name != "nt":
        raise RuntimeError("physical controller must run with Windows Python")
    if (args.practice_stage is not None or args.start_route) and not args.armed:
        raise RuntimeError("menu automation requires --armed")
    if args.patch_lives and not args.armed:
        raise RuntimeError("--patch-lives requires --armed")
    if args.patch_lives and not args.stop_game:
        raise RuntimeError("--patch-lives requires exact-process cleanup via --stop-game")
    if args.diagnostic_rng_seed is not None and not args.armed:
        raise RuntimeError("--diagnostic-rng-seed requires --armed")
    if args.diagnostic_rng_seed is not None and (
        args.practice_stage is None and not args.start_route
    ):
        raise RuntimeError(
            "--diagnostic-rng-seed requires a fresh menu-started trial"
        )
    if args.continuous_stage and (
        not args.armed
        or (args.practice_stage is None and not args.start_route)
        or not args.patch_lives
    ):
        raise RuntimeError(
            "--continuous-stage requires armed Practice or route play with "
            "--patch-lives"
        )
    enable_donor_imports()
    from th06.dialogue import DialogueSkipper
    from th06.hazards.lasers import (
        track_motion,
        unknown_motion_may_reach_player,
    )
    from th06.input_lease import InputLease
    from th06.model import PLAYER_ALIVE, action_from_input
    from th06.native import (
        ADDR_LIFE_PATCH,
        NativeDecodeError,
        TARGET_SHA256,
        attach_exact,
        read_dialogue_state,
        read_game_frame,
        read_snapshot as read_authoritative_snapshot,
        read_supervisor_state,
    )
    import th06.native as donor_native
    from th06.trial import PracticeTrial, physical_hit

    expected_stage = args.practice_stage or (1 if args.start_route else args.expected_stage)
    expected_scope = (DIFFICULTIES[args.difficulty], 0, 0, expected_stage)
    donor_native.TARGET_EXE = args.game_executable_name
    process = attach_exact(Path(args.game_dir).resolve())
    bridge = None
    activity = None
    keyboard = None
    dialogue = None
    trace = None
    recorder = None
    plugin = None
    policy_transaction = None
    corpus_path = None
    exit_code = 0
    stage_completed = False
    hit_count = 0
    control_dead_end_count = 0
    capture_failure_count = 0
    infrastructure_failure_count = 0
    infrastructure_failures: dict[str, int] = {}
    trace_failure_count = 0
    corpus_failure_count = 0
    anchor_failure_count = 0
    corpus_failure: str | None = None
    termination_reason = "controller-interrupted"
    minimum_commit_headroom_bytes: int | None = None
    maximum_controller_private_bytes = 0
    policy_state_recovered = False
    policy_state_committed = False
    policy_state_rolled_back = False
    policy_transaction_failure: str | None = None
    started = time.monotonic()
    try:
        if args.armed:
            bridge = BackgroundInputBridge(process)
            bridge.install()
            activity = BackgroundActivityLease(process)
            activity.maintain()
            keyboard = BackgroundKeyboard(process.pid, bridge)
            dialogue = DialogueSkipper(process, keyboard)
        if args.patch_lives:
            print(
                f"life patch: {process.patch_lives()} "
                f"at 0x{ADDR_LIFE_PATCH:08X}; physical HIT remains observable",
                flush=True,
            )
        if args.diagnostic_rng_seed is not None:
            old_seed, old_generation = process.set_diagnostic_rng_seed(
                args.diagnostic_rng_seed
            )
            print(
                "diagnostic RNG: "
                f"0x{old_seed:04X}/{old_generation} -> "
                f"0x{args.diagnostic_rng_seed:04X}/0; "
                "original generator and consumer order retained; "
                "training evidence only",
                flush=True,
            )
        if args.practice_stage is not None or args.start_route:
            assert keyboard is not None
            try:
                if args.start_route:
                    start_reimu_a_route(
                        process,
                        keyboard,
                        difficulty=DIFFICULTIES[args.difficulty],
                        maintain_activity=(
                            activity.maintain if activity is not None else None
                        ),
                    )
                else:
                    start_reimu_a_practice(
                        process,
                        keyboard,
                        args.practice_stage,
                        difficulty=DIFFICULTIES[args.difficulty],
                        maintain_activity=(
                            activity.maintain if activity is not None else None
                        ),
                    )
            except MenuNavigationError as error:
                termination_reason = "menu-navigation-retry"
                exit_code = MENU_RETRY_EXIT_CODE
                print(
                    "transient menu navigation failure; exact process will "
                    f"be cleaned up for a bounded fresh retry: {error}",
                    flush=True,
                )
                return exit_code

        kernel = NativeKernel(args.native_library)
        if not args.immutable_policy:
            policy_transaction = StagePolicyTransaction(args.policy_state)
            policy_state_recovered = policy_transaction.begin()
        plugin = HotReloadPolicy(
            args.policy_plugin,
            state_path=args.policy_state,
            immutable=args.immutable_policy,
        )
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        # Corpus is the durable evidence. Live status may lag by one second;
        # buffering avoids one WSL UNC filesystem transaction per game frame.
        trace = args.trace.open("a", encoding="utf-8", buffering=256 * 1024)
        if not args.no_corpus:
            recorder = CorpusRecorder(
                args.corpus_root,
                RunMetadata(
                    code_commit=_code_commit(args.repository),
                    executable_sha256=TARGET_SHA256,
                    native_kernel_sha256=_sha256(kernel.path),
                    input_backend="supervisor-callsite-v2",
                    difficulty=expected_scope[0],
                    character=expected_scope[1],
                    shot_type=expected_scope[2],
                    stage=expected_scope[3],
                    planner={
                        "algorithm": "observed-native-gate-v1",
                        "observed_horizon": args.horizon,
                        "hard_horizon": 4,
                        "exploration_rate": args.exploration_rate,
                        "diagnostic_rng_seed": args.diagnostic_rng_seed,
                    },
                ),
                deferred_compression=args.defer_corpus_compression,
            )
        lease = InputLease()
        trial = (
            RouteTrial()
            if args.start_route
            else PracticeTrial()
            if args.practice_stage is not None
            else None
        )
        previous_snapshot = None
        previous_player_state = None
        pending_learning = None
        policy_source_context = None
        policy_source_start_frame = None
        last_frame = None
        last_anchor_frame = None
        last_anchor_partition = None
        pending_anchor_reason = "stage-root"
        anchor_retry_after = 0
        last_reported_reason = None
        dialogue_active = False
        dialogue_delivery: list[DialogueDeliverySample] = []
        last_dialogue_delivery = None
        next_dialogue_probe = started
        next_reload_poll = started + RELOAD_POLL_SECONDS
        next_health_sample = started
        next_health_trace = started

        def emit_trace(record: dict[str, object]) -> None:
            """Telemetry failure must never take physical input authority."""
            nonlocal trace, trace_failure_count
            if trace is None:
                return
            try:
                trace.write(json.dumps(record, separators=(",", ":")) + "\n")
                frame = record.get("frame")
                if record.get("event") is not None or (
                    isinstance(frame, int) and frame % 60 == 0
                ):
                    trace.flush()
            except OSError as error:
                trace_failure_count += 1
                print(f"live trace disabled after write failure: {error}", flush=True)
                try:
                    trace.close()
                finally:
                    trace = None

        def retain_dialogue_delivery(dialogue_state) -> None:
            """Retain sampled menu delivery without creating a battle observation."""
            nonlocal last_dialogue_delivery
            if recorder is None or keyboard is None:
                return
            frame, current, previous, held_repeat, held_frames = (
                read_passive_input_delivery(process)
            )
            sample = DialogueDeliverySample(
                game_frame=frame,
                current_input_mask=current,
                previous_input_mask=previous,
                published_input_mask=keyboard.published_input_mask,
                held_repeat=held_repeat,
                held_frames=held_frames,
                active=bool(dialogue_state.active),
                skippable=bool(dialogue_state.skippable),
                pulsed_shoot=bool(dialogue_state.pulsed_shoot),
            )
            if sample != last_dialogue_delivery:
                dialogue_delivery.append(sample)
                last_dialogue_delivery = sample

        def retain_continuous_stage(kind: str, error: BaseException) -> bool:
            """Fail closed on transient infra faults without ending the episode."""
            nonlocal infrastructure_failure_count
            capture_gap_resume = (
                args.resume_after_incoherent_capture
                and kind == "coherent-snapshot"
            )
            if not args.continuous_stage and not capture_gap_resume:
                return False
            # WAIT_TIMEOUT (258) proves the exact process handle is still live.
            if (
                not process.handle
                or process.kernel32.WaitForSingleObject(process.handle, 0) != 258
            ):
                return False
            if keyboard is not None:
                # Failure to publish this release is intentionally not caught:
                # an input backend that cannot fail-close is a real authority
                # loss and must reach exact-process cleanup.
                keyboard.release_all()
            lease.cleared()
            infrastructure_failure_count += 1
            count = infrastructure_failures.get(kind, 0) + 1
            infrastructure_failures[kind] = count
            emit_trace({
                "time": time.time(),
                "event": (
                    "capture-gap-fail-close"
                    if capture_gap_resume and not args.continuous_stage
                    else "continuous-fail-close"
                ),
                "kind": kind,
                "count": count,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            if count == 1 or count % 60 == 0:
                print(
                    f"{kind} unavailable; input released and Stage retained "
                    f"(count={count}): {type(error).__name__}: {error}",
                    flush=True,
                )
            time.sleep(0.001)
            return True

        print(
            f"attached pid={process.pid} sha256={TARGET_SHA256}; "
            f"native={kernel.path}; policy={plugin.status()}; "
            + ("armed" if args.armed else "observe-only"),
            flush=True,
        )
        while not args.seconds or time.monotonic() - started < args.seconds:
            now = time.monotonic()
            if activity is not None:
                try:
                    if activity.maintain():
                        emit_trace({
                            "time": time.time(),
                            "event": "background-reactivated",
                            "count": activity.reactivations,
                        })
                except (OSError, RuntimeError) as error:
                    if retain_continuous_stage("background-activity", error):
                        continue
                    raise
            if now >= next_health_sample:
                next_health_sample = now + HEALTH_SAMPLE_SECONDS
                try:
                    memory = read_system_memory()
                    headroom = memory.commit_headroom_bytes
                    minimum_commit_headroom_bytes = (
                        headroom
                        if minimum_commit_headroom_bytes is None
                        else min(minimum_commit_headroom_bytes, headroom)
                    )
                    maximum_controller_private_bytes = max(
                        maximum_controller_private_bytes,
                        memory.controller_private_bytes,
                    )
                    if now >= next_health_trace:
                        emit_trace({
                            "time": time.time(),
                            "event": "system-memory",
                            "commit_total_bytes": memory.commit_total_bytes,
                            "commit_limit_bytes": memory.commit_limit_bytes,
                            "commit_headroom_bytes": headroom,
                            "physical_available_bytes": (
                                memory.physical_available_bytes
                            ),
                            "controller_private_bytes": (
                                memory.controller_private_bytes
                            ),
                        })
                        next_health_trace = now + HEALTH_TRACE_SECONDS
                    reserve = int(args.min_commit_headroom_gib * GIB)
                    if reserve and below_commit_reserve(memory, reserve):
                        termination_reason = "system-commit-headroom-low"
                        exit_code = LOW_COMMIT_EXIT_CODE
                        emit_trace({
                            "time": time.time(),
                            "event": "system-memory-stop",
                            "commit_headroom_bytes": headroom,
                            "required_headroom_bytes": reserve,
                            "controller_private_bytes": (
                                memory.controller_private_bytes
                            ),
                        })
                        print(
                            "host commit headroom low; input will be released "
                            f"and batch paused ({headroom / GIB:.2f} GiB < "
                            f"{reserve / GIB:.2f} GiB)",
                            flush=True,
                        )
                        break
                except OSError as error:
                    infrastructure_failure_count += 1
                    infrastructure_failures["system-memory-sample"] = (
                        infrastructure_failures.get("system-memory-sample", 0)
                        + 1
                    )
                    if infrastructure_failures["system-memory-sample"] == 1:
                        emit_trace({
                            "time": time.time(),
                            "event": "system-memory-unavailable",
                            "error": str(error),
                        })
                    next_health_sample = now + 10.0
            if trial is not None:
                try:
                    _wanted, current_supervisor = read_supervisor_state(process)
                except (OSError, RuntimeError) as error:
                    if retain_continuous_stage("supervisor-capture", error):
                        continue
                    raise
                if trial.observe_supervisor(current_supervisor):
                    stage_completed = True
                    termination_reason = (
                        "route-complete" if args.start_route else "practice-stage-complete"
                    )
                    print(
                        (
                            "Full route reached Ending"
                            if args.start_route
                            else f"Practice Stage {expected_stage} complete"
                        )
                        + "; "
                        f"physical_hits={hit_count}",
                        flush=True,
                    )
                    break
            if dialogue_active:
                assert dialogue is not None and keyboard is not None
                try:
                    dialogue_state = dialogue.update(True)
                    retain_dialogue_delivery(dialogue_state)
                    if dialogue_state.active:
                        observe_passive_control_clock(process)
                except (OSError, RuntimeError) as error:
                    if retain_continuous_stage("dialogue-control", error):
                        continue
                    raise
                if dialogue_state.active:
                    # Dialogue owns only Shoot/Ctrl. No battle movement is
                    # proposed, captured, or learned while its source state
                    # deliberately pauses the BulletManager calc chain.
                    lease.cleared()
                    time.sleep(0.001)
                    continue
                dialogue_active = False
                keyboard.release_all()
                lease.cleared()
            if (
                dialogue is not None
                and keyboard is not None
                and time.monotonic() >= next_dialogue_probe
            ):
                next_dialogue_probe = time.monotonic() + DIALOGUE_PROBE_SECONDS
                message_active = False
                try:
                    message_active, _skippable = read_dialogue_state(process)
                    if message_active:
                        # GuiImpl::msg is the authoritative dialogue state.
                        # Some messages leave isTimeStopped false, so message
                        # state owns fast-forward while the tiny clock sample
                        # independently records any actual source time-stop.
                        keyboard.release_all()
                        keyboard.apply(action_from_input(0))
                        dialogue_state = dialogue.update(True)
                        retain_dialogue_delivery(dialogue_state)
                        dialogue_active = dialogue_state.active
                        observe_passive_control_clock(process)
                except (OSError, RuntimeError) as error:
                    kind = "dialogue-control" if message_active else "dialogue-probe"
                    if retain_continuous_stage(kind, error):
                        continue
                    raise
                if message_active:
                    lease.cleared()
                    if dialogue_active:
                        time.sleep(0.001)
                        continue
                    keyboard.release_all()
            if last_frame is not None:
                try:
                    if read_game_frame(process) == last_frame:
                        time.sleep(0.001)
                        continue
                except (OSError, RuntimeError) as error:
                    if retain_continuous_stage("frame-capture", error):
                        continue
                    raise
            capture_started = time.perf_counter()
            try:
                snapshot = read_control_snapshot(
                    process,
                    horizon=args.horizon,
                    suspend=(bridge.suspended if bridge is not None else None),
                )
            except (NativeDecodeError, OSError, RuntimeError) as error:
                # A compact control root is still epoch/manager-phase strict.
                # A torn observation can never reach the Hard gate.
                if not retain_continuous_stage("coherent-snapshot", error):
                    termination_reason = (
                        "authority-stop:coherent snapshot unavailable:"
                        f"{type(error).__name__}:{str(error)[:160]}"
                    )
                    exit_code = 2
                    if keyboard is not None:
                        keyboard.release_all()
                    lease.cleared()
                    emit_trace({
                        "time": time.time(),
                        "event": "capture-stop",
                        "kind": "coherent-snapshot",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
                    print(termination_reason, flush=True)
                    break
                capture_failure_count += 1
                emit_trace({
                    "time": time.time(),
                    "event": "capture-incoherent",
                    "count": capture_failure_count,
                    "error": str(error),
                })
                continue
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            prior_frame = last_frame
            if snapshot.frame == prior_frame:
                time.sleep(0.001)
                continue
            observed_scope = _snapshot_scope(snapshot)
            if args.start_route and observed_scope != expected_scope:
                previous_scope = expected_scope
                expected_scope = _advance_route_scope(expected_scope, observed_scope)
                expected_stage = expected_scope[3]
                emit_trace({
                    "time": time.time(),
                    "event": "route-stage-transition",
                    "from_scope": list(previous_scope),
                    "to_scope": list(expected_scope),
                    "frame": snapshot.frame,
                })
            try:
                if (
                    previous_snapshot is not None
                    and snapshot.stage == previous_snapshot.stage
                ):
                    snapshot = replace(
                        snapshot,
                        lasers=track_motion(
                            previous_snapshot.lasers,
                            snapshot.lasers,
                            snapshot.frame - previous_snapshot.frame,
                        ),
                    )
                source_context = automatic_source_context(snapshot)
            except (OSError, RuntimeError, ValueError) as error:
                if retain_continuous_stage("source-context", error):
                    continue
                raise
            if source_context != policy_source_context:
                policy_source_context = source_context
                policy_source_start_frame = snapshot.frame
            assert policy_source_start_frame is not None
            phase_elapsed_frames = max(
                0,
                snapshot.frame - policy_source_start_frame,
            )
            last_frame = snapshot.frame
            observation_gap = (
                1 if prior_frame is None else max(0, snapshot.frame - prior_frame)
            )
            hit = physical_hit(previous_player_state, snapshot.player_state)
            previous_player_state = snapshot.player_state
            previous_snapshot = snapshot
            reason = "ok"
            selected = None
            proposed = None
            published = None
            policy = None
            hard = ()
            legal = ()
            locally_admissible = ()
            current_action_name = None
            baseline_action = None
            selected_evaluation = None
            hard_count = 0
            effort_horizon = 0
            observation_features = ()
            action_features = ()
            solve_started = time.perf_counter()
            try:
                if hit:
                    hit_count += 1
                    if args.continuous_stage:
                        reason = "physical-hit"
                        if keyboard is not None:
                            keyboard.release_all()
                            lease.cleared()
                    else:
                        raise AuthorityUnavailable("physical HIT")
                elif _physical_bomb(snapshot):
                    raise AuthorityUnavailable("physical Bomb state/input")
                elif snapshot.in_menu or snapshot.time_stopped:
                    reason = "passive"
                    if keyboard is not None:
                        keyboard.release_all()
                        lease.cleared()
                    if (
                        dialogue is not None
                        and keyboard is not None
                        and not snapshot.in_menu
                        and not snapshot.replay_or_demo
                    ):
                        # Hold Shoot without movement, then let the independent
                        # dialogue controller own Ctrl or the required fresh-Z
                        # edge. release_all() above intentionally ran first so
                        # no previous dodge direction survives into dialogue.
                        keyboard.apply(action_from_input(0))
                        dialogue_state = dialogue.update(True)
                        retain_dialogue_delivery(dialogue_state)
                        dialogue_active = dialogue_state.active
                        if not dialogue_active:
                            keyboard.release_all()
                elif snapshot.replay_or_demo:
                    raise AuthorityUnavailable("replay/demo input authority")
                elif snapshot.player_state not in ACTIVE_PLAYER_STATES:
                    reason = "player-not-active"
                    if keyboard is not None:
                        keyboard.release_all()
                        lease.cleared()
                elif not 0.99 <= snapshot.frame_multiplier <= 1.01:
                    raise AuthorityUnavailable("unsupported frame multiplier")
                elif snapshot.laser_count != len(snapshot.lasers):
                    raise AuthorityUnavailable("incoherent laser decode")
                elif any(
                    not laser.motion_known
                    and unknown_motion_may_reach_player(snapshot, laser, 4)
                    for laser in snapshot.lasers
                ):
                    raise AuthorityUnavailable("unknown reachable laser motion")
                elif observed_scope != expected_scope:
                    raise AuthorityUnavailable(
                        f"scope changed to {_snapshot_scope(snapshot)}"
                    )
                else:
                    current_core = core_action_from_input(snapshot.input_mask)
                    current_action_name = current_core.name
                    kinematics = kinematics_from_snapshot(snapshot)
                    lease_status = (
                        lease.status(snapshot.input_mask, snapshot.frame)
                        if keyboard is not None
                        else None
                    )
                    if lease_status is not None and lease_status.timed_out:
                        raise AuthorityUnavailable("input pickup timeout")
                    if lease_status is not None and lease_status.action is not None:
                        desired_core = core_action_from_input(
                            # Donor action -> source control mask through its
                            # own exact conversion path.
                            _donor_action_mask(lease_status.action)
                        )
                        hard_forecast = lower_observed_hazards(snapshot, 4)
                        retained = kernel.certify_actions(
                            x=snapshot.x,
                            y=snapshot.y,
                            half_width=snapshot.half_width,
                            half_height=snapshot.half_height,
                            kinematics=kinematics,
                            current_action=current_core,
                            hazards=hard_forecast.hazards,
                            candidates=(desired_core,),
                            delivery_delays=lease_status.delivery_delays,
                        )
                        if not retained:
                            raise AuthorityUnavailable("in-flight input unsafe")
                        selected = desired_core
                        proposed = desired_core
                        published = desired_core.name if keyboard is not None else None
                        hard = retained
                        legal = retained
                        locally_admissible = (desired_core.name,)
                        hard_count = 1
                        effort_horizon = 4
                        reason = "input-lease"
                    else:
                        forecast = lower_observed_hazards(
                            snapshot,
                            args.horizon,
                        )
                        prepared_forecast = kernel.prepare_hazards(
                            forecast.hazards
                        )
                        prepared_hard = prepared_forecast.prefix(
                            forecast.hard_horizon,
                        )
                        hard = kernel.certify_actions(
                            x=snapshot.x,
                            y=snapshot.y,
                            half_width=snapshot.half_width,
                            half_height=snapshot.half_height,
                            kinematics=kinematics,
                            current_action=current_core,
                            hazards=prepared_hard,
                        )
                        hard_count = len(hard)
                        if not hard:
                            raise AuthorityUnavailable("Hard safe set empty")
                        lookahead = kernel.certify_actions(
                            x=snapshot.x,
                            y=snapshot.y,
                            half_width=snapshot.half_width,
                            half_height=snapshot.half_height,
                            kinematics=kinematics,
                            current_action=current_core,
                            hazards=prepared_forecast,
                            candidates=tuple(item.action for item in hard),
                        )
                        # A longer constant-action gate is advisory: when every
                        # constant path closes, retain the immediate Hard set so
                        # the learned policy can re-decide next frame instead of
                        # requiring an online combinatorial search.
                        legal = lookahead or hard
                        effort_horizon = (
                            forecast.source_coverage if lookahead else 4
                        )
                        action_profiles = kernel.profile_actions(
                            x=snapshot.x,
                            y=snapshot.y,
                            half_width=snapshot.half_width,
                            half_height=snapshot.half_height,
                            kinematics=kinematics,
                            current_action=current_core,
                            hazards=prepared_forecast,
                            candidates=tuple(item.action for item in hard),
                            checkpoints=(1, 2, 3, 4, 6, 8, 10, 12),
                        )
                        locally_admissible = tuple(
                            item.action.name for item in legal
                        )
                        baseline = _reactive_baseline(legal, current_core)
                        baseline_action = baseline.action.name
                        observation_features, action_features = (
                            project_learning_features(
                                snapshot,
                                hard,
                                locally_admissible,
                                effort_horizon,
                                action_profiles,
                            )
                        )
                        policy = plugin.decide(PolicyContext(
                            frame=snapshot.frame,
                            scope=expected_scope,
                            source_context=source_context,
                            baseline_action=baseline_action,
                            locally_admissible_actions=locally_admissible,
                            player_x=snapshot.x,
                            player_y=snapshot.y,
                            power=snapshot.current_power,
                            bullet_count=snapshot.live_bullet_count,
                            laser_count=snapshot.laser_count,
                            hard_action_count=len(hard),
                            exploration_rate=args.exploration_rate,
                            current_action=current_action_name,
                            hard_admissible_actions=tuple(
                                item.action.name for item in hard
                            ),
                            phase_elapsed_frames=phase_elapsed_frames,
                            hard_action_evaluations=tuple(
                                (
                                    item.action.name,
                                    _finite(item.min_clearance),
                                    item.final_x,
                                    item.final_y,
                                )
                                for item in hard
                            ),
                            effort_horizon=effort_horizon,
                            observation_features=observation_features,
                            action_features=action_features,
                        ))
                        selected = next(
                            item.action for item in legal
                            if item.action.name == policy.action
                        )
                        selected_evaluation = next(
                            item for item in legal if item.action == selected
                        )
                        proposed = selected
                        # Re-run the selected Hard certificate after soft work,
                        # then reject publication if the physical frame moved.
                        fresh = kernel.certify_actions(
                            x=snapshot.x,
                            y=snapshot.y,
                            half_width=snapshot.half_width,
                            half_height=snapshot.half_height,
                            kinematics=kinematics,
                            current_action=current_core,
                            hazards=prepared_hard,
                            candidates=(selected,),
                        )
                        if not fresh:
                            raise AuthorityUnavailable(
                                "selected action lost fresh Hard"
                            )
                        if read_game_frame(process) != snapshot.frame:
                            selected = None
                            reason = "stale-retry"
                        elif keyboard is not None:
                            events = keyboard.apply(donor_action(selected))
                            published = selected.name
                            if events and selected != current_core:
                                lease.issued(
                                    read_game_frame(process),
                                    donor_action(selected),
                                    action_from_input(snapshot.input_mask),
                                )
            except AuthorityUnavailable as error:
                error_text = str(error)
                dead_end = error_text in (
                        "Hard safe set empty",
                        "local forecast has no safe continuation",
                )
                recoverable = (
                    args.continuous_stage
                    and error_text != "physical Bomb state/input"
                )
                if recoverable and dead_end:
                    reason = f"control-dead-end:{error_text}"
                    control_dead_end_count += 1
                elif recoverable:
                    reason = f"authority-retry:{error_text}"
                    infrastructure_failure_count += 1
                    infrastructure_failures["authority"] = (
                        infrastructure_failures.get("authority", 0) + 1
                    )
                else:
                    reason = f"authority-stop:{error_text}"
                    termination_reason = reason
                    exit_code = (
                        10
                        if error_text == "physical HIT"
                        else 11
                        if error_text == "physical Bomb state/input"
                        else 12
                        if error_text in (
                            "Hard safe set empty",
                            "local forecast has no safe continuation",
                        )
                        else 2
                    )
                if keyboard is not None:
                    keyboard.release_all()
                    lease.cleared()
            except (OSError, RuntimeError, ValueError) as error:
                if not retain_continuous_stage("control-infrastructure", error):
                    raise
                reason = (
                    "infrastructure-retry:"
                    f"{type(error).__name__}:{str(error)[:160]}"
                )

            solve_ms = (time.perf_counter() - solve_started) * 1000.0
            if pending_learning is not None:
                learning_elapsed = max(
                    0, snapshot.frame - pending_learning["frame"]
                )
                plugin.observe(PolicyOutcome(
                    frame=pending_learning["frame"],
                    scope=pending_learning["scope"],
                    source_context=pending_learning["source_context"],
                    action=pending_learning["action"],
                    published=True,
                    elapsed_frames=learning_elapsed,
                    life_lost=hit,
                    bomb_used=_physical_bomb(snapshot),
                    control_dead_end=_control_dead_end(reason),
                    authority_lost=_authority_loss(reason),
                    phase_changed=(
                        pending_learning["scope"] != _snapshot_scope(snapshot)
                        or pending_learning["source_context"] != source_context
                    ),
                    next_hard_action_count=(
                        -1 if reason == "input-lease" else hard_count
                    ),
                    next_player_x=snapshot.x,
                    next_player_y=snapshot.y,
                    learning_eligible=(
                        learning_elapsed == 1
                        and pending_learning["capture_ms"] <= FRAME_BUDGET_MS
                    ),
                ))
                pending_learning = None
            if hit:
                # A HIT usually arrives after the controller has already
                # stopped publishing (death/invulnerability transition), so
                # it cannot reliably ride on the one-step PolicyOutcome.
                # Deliver it separately for bounded delayed credit while the
                # native gate remains the sole action authority.
                plugin.observe_failure(PolicyFailureEvent(
                    frame=snapshot.frame,
                    scope=_snapshot_scope(snapshot),
                    source_context=source_context,
                    kind="physical-hit",
                ))
            if policy is not None and published is None:
                plugin.observe(PolicyOutcome(
                    frame=snapshot.frame,
                    scope=expected_scope,
                    source_context=source_context,
                    action=policy.action,
                    published=False,
                    elapsed_frames=0,
                    life_lost=False,
                    bomb_used=False,
                    control_dead_end=False,
                    authority_lost=False,
                    phase_changed=False,
                    next_hard_action_count=hard_count,
                    next_player_x=snapshot.x,
                    next_player_y=snapshot.y,
                    learning_eligible=False,
                ))
            if policy is not None and published is not None:
                pending_learning = {
                    "frame": snapshot.frame,
                    "scope": expected_scope,
                    "source_context": source_context,
                    "action": published,
                    "capture_ms": capture_ms,
                }
            policy_status = plugin.status()
            if recorder is not None:
                evidence = FrameEvidence(
                    phase_id=source_context,
                    current_action=current_action_name,
                    hard_actions=tuple(
                        (
                            item.action.name,
                            _finite(item.min_clearance),
                            item.final_x,
                            item.final_y,
                        )
                        for item in hard
                    ),
                    baseline_action=baseline_action,
                    locally_admissible_actions=locally_admissible,
                    proposed_action=proposed.name if proposed is not None else None,
                    published_action=published,
                    behavior_probability=(
                        policy.behavior_probability if policy is not None else 1.0
                    ),
                    policy_id=(
                        policy.policy_id
                        if policy is not None
                        else policy_status.get("policy_id")
                    ),
                    policy_generation=int(policy_status["generation"]),
                    policy_sha256=policy_status.get("sha256"),
                    effort_horizon=effort_horizon,
                    plan_min_clearance=(
                        _finite(selected_evaluation.min_clearance)
                        if selected_evaluation is not None else None
                    ),
                    cumulative_risk=None,
                    terminal_x=(
                        selected_evaluation.final_x
                        if selected_evaluation is not None else None
                    ),
                    terminal_y=(
                        selected_evaluation.final_y
                        if selected_evaluation is not None else None
                    ),
                    endpoint_count=len(legal),
                    continuation_action_count=len(legal),
                    capture_ms=capture_ms,
                    solve_ms=solve_ms,
                    reason=reason,
                    capture_attempts=snapshot.capture_attempts,
                    observation_gap=observation_gap,
                    snapshot_tier=CONTROL_CAPTURE_TIER,
                    phase_elapsed_frames=phase_elapsed_frames,
                    dialogue_delivery=tuple(dialogue_delivery),
                    observation_features=observation_features,
                    action_features=action_features,
                )
                try:
                    snapshot_ref = recorder.record(snapshot, evidence)
                    dialogue_delivery.clear()
                    last_dialogue_delivery = None
                    partition = _anchor_partition(source_context)
                    if (
                        last_anchor_partition is not None
                        and partition != last_anchor_partition
                    ):
                        pending_anchor_reason = "source-context-change"
                    elif (
                        args.full_anchor_frames
                        and last_anchor_frame is not None
                        and snapshot.frame - last_anchor_frame
                        >= args.full_anchor_frames
                    ):
                        pending_anchor_reason = "periodic"
                    # Exhaustive decoding is intentionally admitted only in a
                    # quiet/passive window.  An anchor may be delayed; it may
                    # never steal an urgent high-density control frame.
                    anchor_safe = bool(
                        snapshot.in_menu
                        or snapshot.time_stopped
                        or snapshot.player_state not in ACTIVE_PLAYER_STATES
                        or (
                            snapshot.live_bullet_count <= 8
                            and snapshot.laser_count == 0
                        )
                    )
                    if (
                        pending_anchor_reason is not None
                        and anchor_safe
                        and snapshot.frame >= anchor_retry_after
                    ):
                        try:
                            anchor = read_authoritative_snapshot(process)
                            anchor_context = automatic_source_context(anchor)
                            if (
                                anchor.stage != expected_stage
                                or _snapshot_scope(anchor) != expected_scope
                            ):
                                raise RuntimeError(
                                    "authoritative anchor changed learning scope"
                                )
                            recorder.record_anchor(
                                anchor,
                                phase_id=anchor_context,
                                reason=pending_anchor_reason,
                                control_snapshot_ref=snapshot_ref,
                            )
                            last_anchor_frame = anchor.frame
                            last_anchor_partition = _anchor_partition(
                                anchor_context
                            )
                            pending_anchor_reason = None
                        except (
                            NativeDecodeError,
                            OSError,
                            RuntimeError,
                            ValueError,
                        ) as error:
                            anchor_failure_count += 1
                            anchor_retry_after = snapshot.frame + 60
                            emit_trace({
                                "time": time.time(),
                                "event": "anchor-capture-missed",
                                "frame": snapshot.frame,
                                "count": anchor_failure_count,
                                "error": str(error),
                            })
                except CorpusError as error:
                    # Durable evidence failure is not input authority. Preserve
                    # the same physical Stage, stop claiming this corpus is a
                    # complete trajectory, and let the next full Stage start a
                    # fresh recorder instead of killing the game here.
                    corpus_failure_count += 1
                    corpus_failure = f"{type(error).__name__}: {error}"
                    failed_recorder = recorder
                    recorder = None
                    dialogue_delivery.clear()
                    last_dialogue_delivery = None
                    if keyboard is not None:
                        keyboard.release_all()
                        lease.cleared()
                    emit_trace({
                        "time": time.time(),
                        "event": "corpus-disabled",
                        "error": corpus_failure,
                        "run_dir": str(failed_recorder.run_dir),
                    })
                    print(
                        "corpus writer disabled for this Stage; physical loop "
                        f"retained: {corpus_failure}",
                        flush=True,
                    )
                    corpus_path = failed_recorder.run_dir
                    try:
                        failed_recorder.close({
                            "termination_reason": "corpus-failure",
                            "stage_completed": False,
                            "controller_exit_code": 0,
                            "physical_hits": hit_count,
                            "control_dead_ends": control_dead_end_count,
                            "capture_failures": capture_failure_count,
                            "anchor_failures": anchor_failure_count,
                            "corpus_failure": corpus_failure,
                            "elapsed_wall_seconds": time.monotonic() - started,
                            "minimum_commit_headroom_bytes": (
                                minimum_commit_headroom_bytes
                            ),
                            "maximum_controller_private_bytes": (
                                maximum_controller_private_bytes
                            ),
                        })
                    except CorpusError:
                        # The run directory and its last atomic manifest remain
                        # as explicit partial evidence. Cleanup must continue.
                        pass
            record = {
                "time": time.time(),
                "run_id": recorder.run_id if recorder is not None else None,
                "frame": snapshot.frame,
                "scope": list(_snapshot_scope(snapshot)),
                "source_context": source_context,
                "x": snapshot.x,
                "y": snapshot.y,
                "bullets": snapshot.live_bullet_count,
                "lasers": snapshot.laser_count,
                "hard_count": hard_count,
                "effort_horizon": effort_horizon,
                "action": selected.name if selected is not None else None,
                "reason": reason,
                "capture_ms": capture_ms,
                "solve_ms": solve_ms,
                "capture_attempts": snapshot.capture_attempts,
                "observation_gap": observation_gap,
                "policy": policy_status,
            }
            emit_trace(record)
            exceptional_change = (
                reason not in (
                    "ok",
                    "stale-retry",
                    "input-lease",
                    "passive",
                    "player-not-active",
                )
                and reason != last_reported_reason
            )
            if snapshot.frame % 60 == 0 or exceptional_change:
                print(
                    f"f={snapshot.frame} bullets={snapshot.live_bullet_count} "
                    f"hard={hard_count} h={effort_horizon} "
                    f"action={record['action']} capture={capture_ms:.2f}ms "
                    f"solve={solve_ms:.2f}ms "
                    f"reason={reason}",
                    flush=True,
                )
            last_reported_reason = reason
            if reason.startswith("authority-stop:"):
                break
            if (
                time.monotonic() >= next_reload_poll
                and _reload_poll_window(snapshot)
            ):
                poll_started = time.perf_counter()
                reloaded = False
                try:
                    reloaded = plugin.maybe_reload(0)
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    if not retain_continuous_stage("policy-reload-poll", error):
                        raise
                emit_trace({
                    "time": time.time(),
                    "event": "policy-reload-poll",
                    "frame": snapshot.frame,
                    "reloaded": reloaded,
                    "duration_ms": (
                        time.perf_counter() - poll_started
                    ) * 1000.0,
                })
                next_reload_poll = time.monotonic() + RELOAD_POLL_SECONDS
        else:
            termination_reason = "time-limit"
    finally:
        if policy_transaction is not None:
            try:
                if (
                    stage_completed
                    and exit_code == 0
                    and sys.exc_info()[0] is None
                    and plugin is not None
                ):
                    plugin.checkpoint()
                    policy_transaction.commit()
                    policy_state_committed = True
                else:
                    policy_transaction.rollback()
                    policy_state_rolled_back = True
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                policy_transaction_failure = f"{type(error).__name__}: {error}"
                infrastructure_failure_count += 1
                infrastructure_failures["policy-transaction"] = 1
                try:
                    policy_transaction.rollback()
                    policy_state_rolled_back = True
                except (OSError, RuntimeError, TypeError, ValueError) as rollback_error:
                    policy_transaction_failure += (
                        "; rollback "
                        f"{type(rollback_error).__name__}: {rollback_error}"
                    )
                exit_code = 76
                print(
                    "Stage policy transaction failed; batch will stop: "
                    f"{policy_transaction_failure}",
                    flush=True,
                )
        if trace is not None:
            trace.close()
        try:
            if dialogue is not None:
                dialogue.release()
            if keyboard is not None:
                keyboard.release_all()
        finally:
            try:
                if recorder is not None:
                    try:
                        corpus_path = recorder.close({
                            "termination_reason": termination_reason,
                            "stage_completed": stage_completed,
                            "controller_exit_code": exit_code,
                            "physical_hits": hit_count,
                            "control_dead_ends": control_dead_end_count,
                            "capture_failures": capture_failure_count,
                            "anchor_failures": anchor_failure_count,
                            "infrastructure_failures": infrastructure_failure_count,
                            "infrastructure_failures_by_kind": infrastructure_failures,
                            "trace_failures": trace_failure_count,
                            "corpus_failures": corpus_failure_count,
                            "corpus_failure": corpus_failure,
                            "elapsed_wall_seconds": time.monotonic() - started,
                            "minimum_commit_headroom_bytes": (
                                minimum_commit_headroom_bytes
                            ),
                            "maximum_controller_private_bytes": (
                                maximum_controller_private_bytes
                            ),
                            "policy_state_recovered": policy_state_recovered,
                            "policy_state_committed": policy_state_committed,
                            "policy_state_rolled_back": policy_state_rolled_back,
                            "policy_transaction_failure": (
                                policy_transaction_failure
                            ),
                            "background_reactivations": (
                                activity.reactivations
                                if activity is not None
                                else 0
                            ),
                        })
                    except CorpusError as error:
                        corpus_failure_count += 1
                        corpus_failure = f"{type(error).__name__}: {error}"
                        corpus_path = recorder.run_dir
                        print(
                            f"corpus finalization failed; cleanup retained: {corpus_failure}",
                            flush=True,
                        )
            finally:
                try:
                    if activity is not None:
                        activity.release()
                finally:
                    try:
                        if bridge is not None:
                            bridge.close()
                    finally:
                        try:
                            if args.stop_game:
                                # A user/window-manager initiated exit can race the
                                # controller cleanup. A signalled process handle means
                                # the exact trial is already gone and needs no second
                                # TerminateProcess call.
                                if process.kernel32.WaitForSingleObject(
                                    process.handle, 0
                                ) != 0:
                                    process.terminate()
                        finally:
                            process.close()
        print(
            "released input and restored bridge; "
            + (
                f"stopped exact pid {process.pid}"
                if args.stop_game
                else f"left exact pid {process.pid} running for re-attach"
            ),
            flush=True,
        )
        if corpus_path is not None:
            print(f"complete corpus: {corpus_path}", flush=True)
    if corpus_path is not None and not args.no_post_run_audit:
        try:
            audit_command = [
                    sys.executable,
                    str(args.repository / "scripts/audit_run.py"),
                    str(corpus_path),
                ]
            if args.native_library is not None:
                audit_command.extend((
                    "--native-library",
                    str(args.native_library),
                ))
            result = subprocess.run(
                audit_command,
                check=False,
            )
            if result.returncode:
                print(
                    "post-run infra audit reported status "
                    f"{result.returncode}; learning batch retained",
                    flush=True,
                )
        except OSError as error:
            print(
                f"post-run infra audit unavailable; batch retained: {error}",
                flush=True,
            )
    return exit_code


def _donor_action_mask(action) -> int:
    mask = 0x04 if action.focused else 0
    if action.dx < 0:
        mask |= 0x40
    elif action.dx > 0:
        mask |= 0x80
    if action.dy < 0:
        mask |= 0x10
    elif action.dy > 0:
        mask |= 0x20
    return mask


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-dir",
        default=r"D:\Entertainment\Game\Touhou\th06",
    )
    parser.add_argument("--game-executable-name", default="th06.exe")
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--practice-stage", type=int, choices=range(1, 7))
    parser.add_argument(
        "--start-route",
        action="store_true",
        help="enter ordinary Start and continue through the six-Stage route",
    )
    parser.add_argument("--expected-stage", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="hard")
    parser.add_argument(
        "--patch-lives",
        action="store_true",
        help="apply the verified 1.02h HIT-continuation life patch",
    )
    parser.add_argument(
        "--continuous-stage",
        action="store_true",
        help=(
            "record HIT/dead-end feedback and keep playing Practice or the "
            "ordinary route until its natural terminal"
        ),
    )
    parser.add_argument(
        "--resume-after-incoherent-capture",
        action="store_true",
        help=(
            "release every input on a torn compact snapshot and resume only "
            "after a later coherent capture; HIT and geometry authority "
            "failures still stop the run"
        ),
    )
    parser.add_argument("--stop-game", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument(
        "--full-anchor-frames",
        type=int,
        default=1800,
        help=(
            "retain an exhaustive source snapshot at phase changes and at "
            "this low-frequency interval; 0 disables periodic anchors"
        ),
    )
    parser.add_argument("--exploration-rate", type=float, default=0.03)
    parser.add_argument(
        "--diagnostic-rng-seed",
        type=lambda value: int(value, 0),
        choices=range(0x10000),
        metavar="0..0xffff",
        help=(
            "training diagnostic: fix the original source RNG initial seed "
            "while retaining its generator and consumer order"
        ),
    )
    parser.add_argument("--native-library", type=Path)
    parser.add_argument(
        "--policy-plugin",
        type=Path,
        default=repository / "src/th06_rl/policies/adaptive.py",
    )
    parser.add_argument(
        "--policy-state",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--immutable-policy",
        action="store_true",
        help=(
            "disable policy feedback, hot reload, checkpoint writes, and the "
            "Stage state transaction for a frozen evaluation"
        ),
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=repository / "artifacts/corpus",
    )
    parser.add_argument(
        "--defer-corpus-compression",
        action="store_true",
        help=(
            "write gzip-0 shards to a fast local spool during play; a "
            "post-Stage finalizer must recompress and archive them"
        ),
    )
    parser.add_argument("--no-corpus", action="store_true")
    parser.add_argument("--no-post-run-audit", action="store_true")
    parser.add_argument(
        "--min-commit-headroom-gib",
        type=float,
        default=4.0,
        help=(
            "release input and stop the batch when Windows system commit "
            "headroom falls below this reserve; 0 disables the guard"
        ),
    )
    args = parser.parse_args(argv)
    if args.start_route and args.practice_stage is not None:
        parser.error("--start-route and --practice-stage are mutually exclusive")
    if not _valid_executable_basename(args.game_executable_name):
        parser.error("--game-executable-name must be one .exe basename")
    if args.seconds < 0.0:
        parser.error("--seconds cannot be negative")
    if args.horizon < 4:
        parser.error("--horizon must cover Hard-4")
    if args.full_anchor_frames < 0:
        parser.error("--full-anchor-frames cannot be negative")
    if not 0.0 <= args.exploration_rate <= 1.0:
        parser.error("--exploration-rate must be in [0, 1]")
    if args.immutable_policy and args.exploration_rate != 0.0:
        parser.error("--immutable-policy requires --exploration-rate 0")
    if args.min_commit_headroom_gib < 0.0:
        parser.error("--min-commit-headroom-gib cannot be negative")
    expected_stage = args.practice_stage or args.expected_stage
    label = (
        f"{args.difficulty}_reimu_a_route"
        if args.start_route
        else f"{args.difficulty}_reimu_a_stage{expected_stage}"
    )
    if args.policy_state is None:
        args.policy_state = repository / f"artifacts/policy/{label}.json"
    if args.trace is None:
        args.trace = repository / f"artifacts/live/{label}.jsonl"
    args.repository = repository
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))
