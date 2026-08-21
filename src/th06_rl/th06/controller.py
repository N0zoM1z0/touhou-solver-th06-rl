"""Restartable physical TH06 shell for shielded policy execution."""

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
    CorpusError,
    CorpusRecorder,
    DialogueDeliverySample,
    FrameEvidence,
    RunMetadata,
)
from ..native import NativeKernel
from ..policy_api import PolicyContext
from ..policy_loader import ImmutablePolicy
from .background_activity import BackgroundActivityLease
from .background_input import BackgroundInputBridge
from .background_keyboard import BackgroundKeyboard
from .control_capture import (
    CONTROL_CAPTURE_TIER,
    OFFLINE_FACT_SCHEMA,
    observe_passive_control_clock,
    read_control_snapshot,
    read_passive_input_delivery,
)
from ..retail import native as retail_native
from ..retail.dialogue import DialogueSkipper
from ..retail.hazards.lasers import track_motion
from ..retail.input_lease import InputLease
from ..retail.model import PLAYER_ALIVE, PLAYER_INVULNERABLE, action_from_input
from ..retail.native import (
    ADDR_LIFE_PATCH,
    NativeDecodeError,
    TARGET_SHA256,
    attach_exact,
    read_dialogue_state,
    read_game_frame,
    read_supervisor_state,
)
from ..retail.trial import PracticeTrial, physical_hit
from .menu import (
    MenuNavigationError,
    start_reimu_a_practice,
    start_reimu_a_route,
)
from .source import (
    ControlUnavailable,
    COLLISION_MARGIN,
    SHIELD_HORIZON,
    core_action_from_input,
    retail_action,
    kinematics_from_snapshot,
    lower_observed_hazards,
)
from .system_health import GIB, below_commit_reserve, read_system_memory


ACTIVE_PLAYER_STATES = (PLAYER_ALIVE, PLAYER_INVULNERABLE)
HEALTH_SAMPLE_SECONDS = 1.0
HEALTH_TRACE_SECONDS = 10.0
PAUSED_CAPTURE_ATTEMPTS = 8
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
        raise ControlUnavailable(f"route scope changed unexpectedly to {observed}")
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


def _control_dead_end(reason: str) -> bool:
    return reason in (
        "control-dead-end:in-flight input rejected by shield",
        "control-dead-end:observed shield set empty",
        "infrastructure-stop:in-flight input rejected by shield",
        "infrastructure-stop:observed shield set empty",
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


def _capture_control_root_while_paused(process, bridge):
    """Resume the exact process between rejected paused-root attempts.

    The successful pause remains entered so source certification and input
    publication belong to that same epoch. A calc phase cannot finish while
    the exact process is suspended, so every rejected attempt is resumed
    before the next bounded retry.
    """
    failures = []
    for attempt in range(1, PAUSED_CAPTURE_ATTEMPTS + 1):
        pause = bridge.suspended()
        pause.__enter__()
        try:
            snapshot = read_control_snapshot(
                process,
                horizon=SHIELD_HORIZON,
                suspend=None,
                max_attempts=1,
            )
        except (NativeDecodeError, OSError, RuntimeError, ValueError) as error:
            pause.__exit__(type(error), error, error.__traceback__)
            failures.append(f"{type(error).__name__}:{str(error)[:160]}")
            if attempt < PAUSED_CAPTURE_ATTEMPTS:
                time.sleep(0.002)
            continue
        if snapshot.capture_attempts != attempt:
            snapshot = replace(snapshot, capture_attempts=attempt)
        return snapshot, pause
    raise RuntimeError(
        "paused coherent physical capture exhausted resume-separated retries; "
        f"failures={failures}"
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


def _apply_observed_shield(kernel, **kwargs):
    """Apply the fixed margin to one observed-hazard action set."""
    return kernel.certify_actions(
        **kwargs,
        collision_margin=COLLISION_MARGIN,
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
    expected_stage = args.practice_stage or (1 if args.start_route else args.expected_stage)
    expected_scope = (DIFFICULTIES[args.difficulty], 0, 0, expected_stage)
    retail_native.TARGET_EXE = args.game_executable_name
    process = attach_exact(Path(args.game_dir).resolve())
    bridge = None
    activity = None
    keyboard = None
    dialogue = None
    trace = None
    recorder = None
    plugin = None
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
    corpus_failure: str | None = None
    decision_pause = None
    termination_reason = "controller-interrupted"
    minimum_commit_headroom_bytes: int | None = None
    maximum_controller_private_bytes = 0
    started = time.monotonic()

    def resume_decision_pause() -> None:
        """Resume an exact process paused across root capture and publication."""
        nonlocal decision_pause
        pause = decision_pause
        decision_pause = None
        if pause is not None:
            pause.__exit__(None, None, None)

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
        plugin = ImmutablePolicy(
            args.policy_plugin,
            state_path=args.policy_state,
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
                    online_contract={
                        "algorithm": "observed-shield4-paused-publication-v1",
                        "shield_contract": "observed-hazard-kinematics-v1",
                        "publication_epoch": "coherent-root-process-suspended-v1",
                        "factual_state_schema": OFFLINE_FACT_SCHEMA,
                        "shield_horizon": SHIELD_HORIZON,
                        "minimum_collision_margin": COLLISION_MARGIN,
                        "predicts_future_births": False,
                        "diagnostic_rng_seed": args.diagnostic_rng_seed,
                    },
                    episode_unit=("route" if args.start_route else "practice-stage"),
                    expected_stages=(
                        tuple(range(1, 7))
                        if args.start_route
                        else (expected_stage,)
                    ),
                ),
                max_run_bytes=int(args.max_corpus_gib * GIB),
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
        last_frame = None
        last_reported_reason = None
        dialogue_active = False
        dialogue_delivery: list[DialogueDeliverySample] = []
        last_dialogue_delivery = None
        next_dialogue_probe = started
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
            stage, frame, current, previous, held_repeat, held_frames = (
                read_passive_input_delivery(process)
            )
            sample = DialogueDeliverySample(
                stage=stage,
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
            """Release input and force exact-trial cleanup on infra failure."""
            nonlocal infrastructure_failure_count
            if keyboard is not None:
                keyboard.release_all()
            lease.cleared()
            infrastructure_failure_count += 1
            count = infrastructure_failures.get(kind, 0) + 1
            infrastructure_failures[kind] = count
            emit_trace({
                "time": time.time(),
                "event": "infrastructure-fail-stop",
                "kind": kind,
                "count": count,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            print(
                f"{kind} unavailable; input released and exact trial will stop: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            return False

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
                if bridge is not None:
                    snapshot, decision_pause = _capture_control_root_while_paused(
                        process,
                        bridge,
                    )
                else:
                    snapshot = read_control_snapshot(
                        process,
                        horizon=SHIELD_HORIZON,
                        suspend=None,
                    )
                if snapshot.factual_state_schema != OFFLINE_FACT_SCHEMA:
                    raise RuntimeError("dense offline factual root is incomplete")
            except (NativeDecodeError, OSError, RuntimeError) as error:
                # A compact control root is still epoch/manager-phase strict.
                # A torn observation can never reach the action shield.
                if not retain_continuous_stage("coherent-snapshot", error):
                    termination_reason = (
                        "infrastructure-stop:coherent snapshot unavailable:"
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
                    resume_decision_pause()
                    break
                capture_failure_count += 1
                emit_trace({
                    "time": time.time(),
                    "event": "capture-incoherent",
                    "count": capture_failure_count,
                    "error": str(error),
                })
                resume_decision_pause()
                continue
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            prior_frame = last_frame
            if snapshot.frame == prior_frame:
                resume_decision_pause()
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
            phase_id = f"stage:{snapshot.stage}"
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
            shield_evaluations = ()
            legal = ()
            locally_admissible = ()
            # This is a witnessed game input at the completed-calc root, not
            # the action most recently sent through the Wine input bridge.
            # Preserve it for passive/dead frames too so the next-root corpus
            # transition can separate command intent from sampled execution.
            current_action_name = core_action_from_input(snapshot.input_mask).name
            baseline_action = None
            shield_count = 0
            shield_collision_margin = None
            shield_contract = ""
            shield_horizon = 0
            shield_aabb_frames = ()
            shield_laser_frames = ()
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
                        raise ControlUnavailable("physical HIT")
                elif _physical_bomb(snapshot):
                    raise ControlUnavailable("physical Bomb state/input")
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
                    raise ControlUnavailable("replay/demo input authority")
                elif snapshot.player_state not in ACTIVE_PLAYER_STATES:
                    reason = "player-not-active"
                    if keyboard is not None:
                        keyboard.release_all()
                        lease.cleared()
                elif snapshot.frame_multiplier != 1.0:
                    raise ControlUnavailable("unsupported frame multiplier")
                elif snapshot.laser_count != len(snapshot.lasers):
                    raise ControlUnavailable("incoherent laser decode")
                elif observed_scope != expected_scope:
                    raise ControlUnavailable(
                        f"scope changed to {_snapshot_scope(snapshot)}"
                    )
                else:
                    current_core = core_action_from_input(snapshot.input_mask)
                    current_action_name = current_core.name
                    kinematics = kinematics_from_snapshot(snapshot)
                    shield_projection = lower_observed_hazards(
                        snapshot,
                        SHIELD_HORIZON,
                    )
                    shield_contract = shield_projection.contract
                    shield_horizon = shield_projection.horizon
                    shield_aabb_frames = tuple(
                        tuple((
                            hazard.left,
                            hazard.top,
                            hazard.right,
                            hazard.bottom,
                        ) for hazard in frame)
                        for frame in shield_projection.hazards.aabb_frames[:4]
                    )
                    shield_laser_frames = tuple(
                        tuple((
                            hazard.origin_x,
                            hazard.origin_y,
                            hazard.angle,
                            hazard.center_offset,
                            hazard.size_x,
                            hazard.size_y,
                        ) for hazard in frame)
                        for frame in shield_projection.hazards.laser_frames[:4]
                    )
                    prepared_shield = kernel.prepare_hazards(
                        shield_projection.hazards
                    )
                    lease_status = (
                        lease.status(snapshot.input_mask, snapshot.frame)
                        if keyboard is not None
                        else None
                    )
                    if lease_status is not None and lease_status.timed_out:
                        raise ControlUnavailable("input pickup timeout")
                    if lease_status is not None and lease_status.action is not None:
                        desired_core = core_action_from_input(
                            # Donor action -> source control mask through its
                            # own exact conversion path.
                            _retail_action_mask(lease_status.action)
                        )
                        retained = _apply_observed_shield(
                            kernel,
                            x=snapshot.x,
                            y=snapshot.y,
                            half_width=snapshot.half_width,
                            half_height=snapshot.half_height,
                            kinematics=kinematics,
                            current_action=current_core,
                            hazards=prepared_shield,
                            candidates=(desired_core,),
                            delivery_delays=lease_status.delivery_delays,
                        )
                        shield_collision_margin = COLLISION_MARGIN
                        if not retained:
                            raise ControlUnavailable(
                                "in-flight input rejected by shield"
                            )
                        selected = desired_core
                        proposed = desired_core
                        published = desired_core.name if keyboard is not None else None
                        shield_evaluations = retained
                        legal = retained
                        locally_admissible = (desired_core.name,)
                        shield_count = 1
                        reason = "input-lease"
                    else:
                        shield = _apply_observed_shield(
                            kernel,
                            x=snapshot.x,
                            y=snapshot.y,
                            half_width=snapshot.half_width,
                            half_height=snapshot.half_height,
                            kinematics=kinematics,
                            current_action=current_core,
                            hazards=prepared_shield,
                        )
                        shield_collision_margin = COLLISION_MARGIN
                        shield_count = len(shield)
                        if not shield:
                            raise ControlUnavailable("observed shield set empty")
                        shield_evaluations = shield
                        legal = shield
                        baseline = _reactive_baseline(legal, current_core)
                        baseline_action = baseline.action.name
                        locally_admissible = tuple(
                            item.action.name for item in legal
                        )
                        policy = plugin.decide(PolicyContext(
                            baseline_action=baseline_action,
                            locally_admissible_actions=locally_admissible,
                            player_x=snapshot.x,
                            player_y=snapshot.y,
                            power=snapshot.current_power,
                            bullet_count=snapshot.live_bullet_count,
                            laser_count=snapshot.laser_count,
                            shield_action_count=len(shield),
                            current_action=current_action_name,
                            shield_admissible_actions=tuple(
                                item.action.name for item in shield
                            ),
                            shield_action_evaluations=tuple(
                                (
                                    item.action.name,
                                    _finite(item.min_clearance),
                                    item.final_x,
                                    item.final_y,
                                )
                                for item in shield
                            ),
                        ))
                        selected = next(
                            item.action for item in legal
                            if item.action.name == policy.action
                        )
                        proposed = selected
                        # Re-run the selected observed shield after policy work,
                        # then reject publication if the physical frame moved.
                        fresh = kernel.certify_actions(
                            x=snapshot.x,
                            y=snapshot.y,
                            half_width=snapshot.half_width,
                            half_height=snapshot.half_height,
                            kinematics=kinematics,
                            current_action=current_core,
                            hazards=prepared_shield,
                            candidates=(selected,),
                            collision_margin=shield_collision_margin,
                        )
                        if not fresh:
                            raise ControlUnavailable(
                                "selected action lost fresh observed shield"
                            )
                        issue_frame = read_game_frame(process)
                        if issue_frame != snapshot.frame:
                            stale_delta = issue_frame - snapshot.frame
                            current_is_shielded = any(
                                item.action == current_core for item in shield
                            )
                            if 0 < stale_delta <= 3 and current_is_shielded:
                                selected = None
                                reason = "stale-retain-observed-shield-current"
                            else:
                                raise ControlUnavailable(
                                    "stale coherent observation"
                                )
                        elif keyboard is not None:
                            events = keyboard.apply(retail_action(selected))
                            published = selected.name
                            if events and selected != current_core:
                                lease.issued(
                                    read_game_frame(process),
                                    retail_action(selected),
                                    action_from_input(snapshot.input_mask),
                                )
            except ControlUnavailable as error:
                error_text = str(error)
                dead_end = error_text in (
                    "in-flight input rejected by shield",
                    "observed shield set empty",
                )
                # HIT-continuation may cross a geometry dead end so its
                # factual outcome can be recorded. Coherent capture, delivery,
                # Bomb, and process failures remain non-recoverable.
                recoverable = args.continuous_stage and dead_end
                if recoverable and dead_end:
                    reason = f"control-dead-end:{error_text}"
                    control_dead_end_count += 1
                else:
                    reason = f"infrastructure-stop:{error_text}"
                    termination_reason = reason
                    exit_code = (
                        10
                        if error_text == "physical HIT"
                        else 11
                        if error_text == "physical Bomb state/input"
                        else 12
                        if error_text in (
                            "in-flight input rejected by shield",
                            "observed shield set empty",
                        )
                        else 2
                    )
                if keyboard is not None:
                    keyboard.release_all()
                    lease.cleared()
            except (OSError, RuntimeError, ValueError) as error:
                raise RuntimeError(
                    "control infrastructure lost its coherent transaction"
                ) from error

            # The exact process remains suspended from coherent capture through
            # the final shield replay and bridge publication.
            resume_decision_pause()
            solve_ms = (time.perf_counter() - solve_started) * 1000.0
            # Policy identity is required in every factual frame, but complete
            # diagnostic counters are not.  Full metrics sort a rolling
            # latency window and materialize several action dictionaries; doing
            # that on every 60 Hz frame created allocation/scheduler pressure in
            # the same process as the deadline-sensitive scorer.  Sample full
            # metrics once per game second and emit an exact final snapshot.
            policy_status = plugin.status(
                include_metrics=snapshot.frame % 60 == 0
            )
            if recorder is not None:
                evidence = FrameEvidence(
                    phase_id=phase_id,
                    current_action=current_action_name,
                    shield_actions=tuple(
                        (
                            item.action.name,
                            _finite(item.min_clearance),
                            item.final_x,
                            item.final_y,
                        )
                        for item in shield_evaluations
                    ),
                    baseline_action=baseline_action,
                    locally_admissible_actions=locally_admissible,
                    proposed_action=proposed.name if proposed is not None else None,
                    published_action=published,
                    behavior_probability=(
                        policy.behavior_probability if policy is not None else 1.0
                    ),
                    behavior_probabilities=(
                        policy.behavior_probabilities
                        if policy is not None
                        else tuple(
                            (action, float(action == published))
                            for action in locally_admissible
                        )
                        if published is not None
                        else ()
                    ),
                    policy_id=(
                        policy.policy_id
                        if policy is not None
                        else policy_status.get("policy_id")
                    ),
                    policy_generation=int(policy_status["generation"]),
                    policy_sha256=policy_status.get("sha256"),
                    capture_ms=capture_ms,
                    solve_ms=solve_ms,
                    reason=reason,
                    capture_attempts=snapshot.capture_attempts,
                    observation_gap=observation_gap,
                    snapshot_tier=CONTROL_CAPTURE_TIER,
                    dialogue_delivery=tuple(dialogue_delivery),
                    shield_collision_margin=shield_collision_margin,
                    shield_contract=shield_contract,
                    shield_horizon=shield_horizon,
                    shield_aabb_frames=shield_aabb_frames,
                    shield_laser_frames=shield_laser_frames,
                )
                try:
                    recorder.record(snapshot, evidence)
                    dialogue_delivery.clear()
                    last_dialogue_delivery = None
                except CorpusError as error:
                    # A physical frame without its promised dense factual root
                    # is not valid collection. Release input and terminate the
                    # exact trial; never continue a route with a silent hole.
                    corpus_failure_count += 1
                    corpus_failure = f"{type(error).__name__}: {error}"
                    if keyboard is not None:
                        keyboard.release_all()
                        lease.cleared()
                    emit_trace({
                        "time": time.time(),
                        "event": "corpus-fail-stop",
                        "error": corpus_failure,
                        "run_dir": str(recorder.run_dir),
                    })
                    raise
            record = {
                "time": time.time(),
                "run_id": recorder.run_id if recorder is not None else None,
                "frame": snapshot.frame,
                "scope": list(_snapshot_scope(snapshot)),
                "phase_id": phase_id,
                "x": snapshot.x,
                "y": snapshot.y,
                "bullets": snapshot.live_bullet_count,
                "lasers": snapshot.laser_count,
                "shield_count": shield_count,
                "shield_collision_margin": shield_collision_margin,
                "shield_horizon": shield_horizon,
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
                    "stale-retain-observed-shield-current",
                    "input-lease",
                    "passive",
                    "player-not-active",
                )
                and reason != last_reported_reason
            )
            if snapshot.frame % 60 == 0 or exceptional_change:
                print(
                    f"f={snapshot.frame} bullets={snapshot.live_bullet_count} "
                    f"shield={shield_count} h={shield_horizon} "
                    f"action={record['action']} capture={capture_ms:.2f}ms "
                    f"solve={solve_ms:.2f}ms "
                    f"reason={reason}",
                    flush=True,
                )
            last_reported_reason = reason
            if reason.startswith("infrastructure-stop:"):
                break
        else:
            termination_reason = "time-limit"
    finally:
        resume_decision_pause()
        if trace is not None and plugin is not None:
            emit_trace({
                "time": time.time(),
                "event": "policy-final-status",
                "policy": plugin.status(),
            })
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
                            "infrastructure_failures": infrastructure_failure_count,
                            "infrastructure_failures_by_kind": infrastructure_failures,
                            "trace_failures": trace_failure_count,
                            "policy_failures": int(
                                plugin.status(include_metrics=False)[
                                    "policy_failures"
                                ]
                            ),
                            "policy_last_error": plugin.status(
                                include_metrics=False
                            ).get("last_error"),
                            "corpus_failures": corpus_failure_count,
                            "corpus_failure": corpus_failure,
                            "elapsed_wall_seconds": time.monotonic() - started,
                            "minimum_commit_headroom_bytes": (
                                minimum_commit_headroom_bytes
                            ),
                            "maximum_controller_private_bytes": (
                                maximum_controller_private_bytes
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


def _retail_action_mask(action) -> int:
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
    hit_mode = parser.add_mutually_exclusive_group()
    hit_mode.add_argument(
        "--continuous-stage",
        dest="continuous_stage",
        action="store_true",
        default=None,
        help=(
            "record HIT/dead-end feedback and keep playing Practice or the "
            "ordinary route until its natural terminal (the default for "
            "armed menu-started physical episodes)"
        ),
    )
    hit_mode.add_argument(
        "--stop-on-hit",
        dest="continuous_stage",
        action="store_false",
        help=(
            "explicit diagnostic-only first-HIT prefix; incomplete output is "
            "not a training or promotion episode"
        ),
    )
    parser.add_argument("--stop-game", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0)
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
        required=True,
    )
    parser.add_argument(
        "--policy-state",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--immutable-policy",
        action="store_true",
        required=True,
        help=(
            "require a frozen policy; online feedback and hot reload are removed"
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
    parser.add_argument(
        "--max-corpus-gib",
        type=float,
        default=0.5,
        help="fail closed if this physical episode exceeds the storage bound",
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
    if args.continuous_stage is None:
        # A physical menu-started episode is complete by default. Direct
        # attach/observe commands retain their bounded first-failure behavior
        # because they do not own the game lifecycle or life patch.
        args.continuous_stage = bool(
            args.armed and (args.practice_stage is not None or args.start_route)
        )
    if args.start_route and args.practice_stage is not None:
        parser.error("--start-route and --practice-stage are mutually exclusive")
    if not _valid_executable_basename(args.game_executable_name):
        parser.error("--game-executable-name must be one .exe basename")
    if args.seconds < 0.0:
        parser.error("--seconds cannot be negative")
    if args.min_commit_headroom_gib < 0.0:
        parser.error("--min-commit-headroom-gib cannot be negative")
    if args.max_corpus_gib <= 0.0:
        parser.error("--max-corpus-gib must be positive")
    expected_stage = args.practice_stage or args.expected_stage
    label = (
        f"{args.difficulty}_reimu_a_route"
        if args.start_route
        else f"{args.difficulty}_reimu_a_stage{expected_stage}"
    )
    if args.trace is None:
        args.trace = repository / f"artifacts/live/{label}.jsonl"
    args.repository = repository
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))
