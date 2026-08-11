"""No-PTY client for the Linux TH06 stage-entry COW forkserver."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import selectors
import signal
import shutil
import subprocess
import time

from .headless import HeadlessProtocolError, HeadlessScope


@dataclass(frozen=True)
class ForkRunResult:
    child_pid: int
    status: int
    trace_path: Path
    summary_only: bool


@dataclass(frozen=True)
class ForkCheckpointResult:
    child_pid: int
    status: int
    restored_tick: int


@dataclass(frozen=True)
class ForkStepResult:
    child_pid: int
    status: int
    terminal_observation: dict[str, object]
    aborted: bool


class HeadlessForkserver:
    """Own one immutable stage-entry parent and run serial COW children."""

    def __init__(
        self,
        *,
        binary: Path,
        game_directory: Path,
        scope: HeadlessScope,
        seed: int,
        auto_shoot: bool = True,
        stage_rng_seed: int | None = None,
        auto_shoot_after_tick: int | None = None,
        retail_dialogue_control: bool = False,
        retail_dialogue_control_after_tick: int | None = None,
        retail_dialogue_inputs_path: Path | None = None,
        read_timeout: float = 30.0,
    ) -> None:
        if seed not in range(1 << 16):
            raise ValueError("headless seed must be 0..65535")
        if read_timeout <= 0.0:
            raise ValueError("forkserver read timeout must be positive")
        if stage_rng_seed is not None and stage_rng_seed not in range(1 << 16):
            raise ValueError("stage RNG seed must be 0..65535")
        if auto_shoot_after_tick is not None and auto_shoot_after_tick < 0:
            raise ValueError("auto-shoot threshold must be nonnegative")
        if auto_shoot_after_tick is not None and not auto_shoot:
            raise ValueError("auto-shoot threshold requires auto_shoot")
        if retail_dialogue_control and not auto_shoot:
            raise ValueError("retail dialogue control requires auto_shoot")
        if (
            retail_dialogue_control_after_tick is not None
            and retail_dialogue_control_after_tick < 0
        ):
            raise ValueError("retail dialogue control threshold must be nonnegative")
        if retail_dialogue_control_after_tick is not None and not retail_dialogue_control:
            raise ValueError(
                "retail dialogue control threshold requires retail dialogue control"
            )
        if retail_dialogue_inputs_path is not None and not retail_dialogue_control:
            raise ValueError(
                "retail dialogue input stream requires retail dialogue control"
            )
        if (
            retail_dialogue_inputs_path is not None
            and not retail_dialogue_inputs_path.is_file()
        ):
            raise ValueError("retail dialogue input file does not exist")
        self.binary = binary.resolve()
        self.game_directory = game_directory.resolve()
        self.scope = scope
        self.seed = seed
        self.auto_shoot = auto_shoot
        self.stage_rng_seed = stage_rng_seed
        self.auto_shoot_after_tick = auto_shoot_after_tick
        self.retail_dialogue_control = retail_dialogue_control
        self.retail_dialogue_control_after_tick = retail_dialogue_control_after_tick
        self.retail_dialogue_inputs_path = (
            retail_dialogue_inputs_path.resolve()
            if retail_dialogue_inputs_path is not None
            else None
        )
        self.read_timeout = read_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._read_buffer = bytearray()
        self.checkpoint_tick: int | None = None
        self.checkpoint_parent_tick: int | None = None
        self.step_child_active = False
        self.step_child_terminal: dict[str, object] | None = None

    def _command(self) -> list[str]:
        command = [
            str(self.binary),
            "--headless",
            "--forkserver",
            "--seed",
            str(self.seed),
            "--practice-stage",
            str(self.scope.stage),
            "--difficulty",
            str(self.scope.difficulty),
            "--character",
            str(self.scope.character),
            "--shot-type",
            str(self.scope.shot_type),
        ]
        if self.auto_shoot:
            command.append("--auto-shoot")
        if self.stage_rng_seed is not None:
            command.extend(("--stage-rng-seed", str(self.stage_rng_seed)))
        if self.auto_shoot_after_tick is not None:
            command.extend(
                ("--auto-shoot-after-tick", str(self.auto_shoot_after_tick))
            )
        if self.retail_dialogue_control:
            command.append("--retail-dialogue-control")
        if self.retail_dialogue_control_after_tick is not None:
            command.extend(
                (
                    "--retail-dialogue-control-after-tick",
                    str(self.retail_dialogue_control_after_tick),
                )
            )
        if self.retail_dialogue_inputs_path is not None:
            command.extend(
                ("--retail-dialogue-inputs", str(self.retail_dialogue_inputs_path))
            )
        nice = shutil.which("nice")
        ionice = shutil.which("ionice")
        if nice is not None:
            command = [nice, "-n", "15", *command]
        if ionice is not None:
            command = [ionice, "-c", "2", "-n", "7", *command]
        return command

    def _readline(self) -> str:
        process = self.process
        if process is None or process.stdout is None:
            raise HeadlessProtocolError("forkserver process is not running")
        deadline = time.monotonic() + self.read_timeout
        while b"\n" not in self._read_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise HeadlessProtocolError(
                    f"forkserver response timed out; process status={process.poll()}"
                )
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                if not selector.select(remaining):
                    raise HeadlessProtocolError(
                        f"forkserver response timed out; process status={process.poll()}"
                    )
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise HeadlessProtocolError(
                    f"forkserver ended before a response; status={process.poll()}"
                )
            self._read_buffer.extend(chunk)
        raw, _, remainder = self._read_buffer.partition(b"\n")
        self._read_buffer = bytearray(remainder)
        return raw.decode("utf-8").strip()

    @staticmethod
    def _write(process: subprocess.Popen[bytes], line: str) -> None:
        if process.stdin is None:
            raise HeadlessProtocolError("forkserver control input is closed")
        process.stdin.write(line.encode("utf-8"))
        process.stdin.flush()

    def start(self) -> int:
        if self.process is not None:
            raise HeadlessProtocolError("forkserver already started")
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        if not self.game_directory.is_dir():
            raise NotADirectoryError(self.game_directory)
        self.process = subprocess.Popen(
            self._command(),
            cwd=self.game_directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=False,
            bufsize=0,
            start_new_session=True,
        )
        ready = self._readline().split()
        if len(ready) != 2 or ready[0] != "READY":
            self.close()
            raise HeadlessProtocolError(f"invalid forkserver greeting: {' '.join(ready)}")
        try:
            self.checkpoint_tick = int(ready[1])
        except ValueError as error:
            self.close()
            raise HeadlessProtocolError("invalid forkserver checkpoint tick") from error
        return self.checkpoint_tick

    @staticmethod
    def _protocol_path(path: Path) -> Path:
        resolved = path.resolve()
        if any(character.isspace() for character in str(resolved)):
            raise ValueError("forkserver protocol paths cannot contain whitespace")
        return resolved

    def run(
        self,
        *,
        terminal_tick: int,
        actions_path: Path,
        trace_path: Path,
        summary_only: bool = False,
    ) -> ForkRunResult:
        process = self.process
        checkpoint = self.checkpoint_tick
        if process is None or process.stdin is None or checkpoint is None:
            raise HeadlessProtocolError("forkserver process is not running")
        if terminal_tick <= checkpoint:
            raise ValueError("fork child terminal tick must follow its checkpoint")
        actions = self._protocol_path(actions_path)
        trace = self._protocol_path(trace_path)
        if not actions.is_file():
            raise FileNotFoundError(actions)
        trace.parent.mkdir(parents=True, exist_ok=True)
        command = "RUN_FINAL" if summary_only else "RUN"
        self._write(process, f"{command} {terminal_tick} {actions} {trace}\n")
        response = self._readline().split()
        if response and response[0] == "ERROR":
            raise HeadlessProtocolError(f"forkserver rejected run: {' '.join(response[1:])}")
        if len(response) != 3 or response[0] != "DONE":
            raise HeadlessProtocolError(f"invalid forkserver completion: {' '.join(response)}")
        try:
            child_pid = int(response[1])
            status = int(response[2])
        except ValueError as error:
            raise HeadlessProtocolError(
                f"non-numeric forkserver completion: {' '.join(response)}"
            ) from error
        if status != 0:
            raise HeadlessProtocolError(
                f"forked headless child {child_pid} failed with status {status}"
            )
        if not trace.is_file():
            raise HeadlessProtocolError("forked headless child produced no trace")
        return ForkRunResult(child_pid, status, trace, summary_only)

    def _read_step_observation(self) -> dict[str, object]:
        line = self._readline()
        try:
            observation = json.loads(line)
        except json.JSONDecodeError as error:
            raise HeadlessProtocolError(
                f"invalid fork STEP JSON observation: {error}"
            ) from error
        if not isinstance(observation, dict):
            raise HeadlessProtocolError("fork STEP observation is not a JSON object")
        if observation.get("terminal_reason") is not None:
            self.step_child_terminal = observation
        return observation

    def begin_step_session(self, *, terminal_tick: int) -> dict[str, object]:
        """Fork an interactive child from the current immutable checkpoint."""
        process = self.process
        checkpoint = self.checkpoint_tick
        if process is None or process.stdin is None or checkpoint is None:
            raise HeadlessProtocolError("forkserver process is not running")
        if self.step_child_active:
            raise HeadlessProtocolError("fork STEP child is already active")
        if terminal_tick <= checkpoint:
            raise ValueError("fork STEP terminal tick must follow its checkpoint")
        self._write(process, f"STEP {terminal_tick}\n")
        self.step_child_active = True
        self.step_child_terminal = None
        return self._read_step_observation()

    def step_session(self, action: str) -> dict[str, object]:
        """Advance one Bomb-free action in the active interactive child."""
        from .offline import ACTION_SET

        process = self.process
        if process is None or process.stdin is None or not self.step_child_active:
            raise HeadlessProtocolError("no fork STEP child is active")
        if self.step_child_terminal is not None:
            raise HeadlessProtocolError("fork STEP child is already terminal")
        if action not in ACTION_SET:
            raise ValueError(f"unknown or forbidden fork STEP action: {action}")
        self._write(process, action + "\n")
        return self._read_step_observation()

    def _finish_step_session(self, *, aborted: bool, allow_error: bool) -> ForkStepResult:
        if not self.step_child_active or self.step_child_terminal is None:
            raise HeadlessProtocolError("fork STEP child has no terminal observation")
        response = self._readline().split()
        terminal = self.step_child_terminal
        self.step_child_active = False
        self.step_child_terminal = None
        if len(response) != 3 or response[0] != "DONE":
            raise HeadlessProtocolError(
                f"invalid fork STEP completion: {' '.join(response)}"
            )
        try:
            child_pid = int(response[1])
            status = int(response[2])
        except ValueError as error:
            raise HeadlessProtocolError(
                f"non-numeric fork STEP completion: {' '.join(response)}"
            ) from error
        if status != 0 and not allow_error:
            raise HeadlessProtocolError(
                f"fork STEP child {child_pid} failed with status {status}"
            )
        return ForkStepResult(child_pid, status, terminal, aborted)

    def finish_step_session(self) -> ForkStepResult:
        """Consume DONE after a physical terminal or requested tick limit."""
        return self._finish_step_session(aborted=False, allow_error=False)

    def abort_step_session(self) -> ForkStepResult:
        """Fail-close a branch whose external native authority became empty."""
        process = self.process
        if process is None or process.stdin is None or not self.step_child_active:
            raise HeadlessProtocolError("no fork STEP child is active")
        if self.step_child_terminal is None:
            # Deliberately outside the movement vocabulary. The runtime emits
            # one input-error terminal observation and never publishes Bomb.
            self._write(process, "__authority_abort__\n")
            terminal = self._read_step_observation()
            if terminal.get("terminal_reason") != "input-error":
                raise HeadlessProtocolError("fork STEP authority abort did not fail closed")
        return self._finish_step_session(aborted=True, allow_error=True)

    def enter_checkpoint(
        self,
        *,
        terminal_tick: int,
        actions_path: Path,
    ) -> int:
        """Replay one common prefix and enter its nested immutable server."""
        process = self.process
        checkpoint = self.checkpoint_tick
        if process is None or process.stdin is None or checkpoint is None:
            raise HeadlessProtocolError("forkserver process is not running")
        if self.checkpoint_parent_tick is not None:
            raise HeadlessProtocolError("nested forkserver checkpoint already active")
        if terminal_tick <= checkpoint:
            raise ValueError("nested checkpoint tick must follow its parent")
        actions = self._protocol_path(actions_path)
        if not actions.is_file():
            raise FileNotFoundError(actions)
        self._write(process, f"CHECKPOINT {terminal_tick} {actions}\n")
        response = self._readline().split()
        if response and response[0] == "ERROR":
            raise HeadlessProtocolError(
                f"forkserver rejected checkpoint: {' '.join(response[1:])}"
            )
        if len(response) != 2 or response[0] != "READY":
            raise HeadlessProtocolError(
                f"prefix terminated before checkpoint: {' '.join(response)}"
            )
        try:
            reached = int(response[1])
        except ValueError as error:
            raise HeadlessProtocolError("invalid nested checkpoint tick") from error
        if reached != terminal_tick:
            raise HeadlessProtocolError(
                f"nested checkpoint reached tick {reached}, expected {terminal_tick}"
            )
        self.checkpoint_parent_tick = checkpoint
        self.checkpoint_tick = reached
        return reached

    def leave_checkpoint(self) -> ForkCheckpointResult:
        """Stop one nested server and restore its immutable parent."""
        process = self.process
        parent_tick = self.checkpoint_parent_tick
        if process is None or process.stdin is None or parent_tick is None:
            raise HeadlessProtocolError("no nested forkserver checkpoint is active")
        self._write(process, "QUIT\n")
        response = self._readline().split()
        self.checkpoint_parent_tick = None
        self.checkpoint_tick = parent_tick
        if len(response) != 3 or response[0] != "CHECKPOINT_DONE":
            raise HeadlessProtocolError(
                f"invalid checkpoint completion: {' '.join(response)}"
            )
        try:
            child_pid = int(response[1])
            status = int(response[2])
        except ValueError as error:
            raise HeadlessProtocolError(
                f"non-numeric checkpoint completion: {' '.join(response)}"
            ) from error
        if status != 0:
            raise HeadlessProtocolError(
                f"checkpoint child {child_pid} failed with status {status}"
            )
        return ForkCheckpointResult(child_pid, status, parent_tick)

    def close(self, timeout: float = 5.0) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None and self.step_child_active:
            try:
                self.abort_step_session()
            except (BrokenPipeError, HeadlessProtocolError):
                pass
        try:
            if process.poll() is None and self.checkpoint_parent_tick is not None:
                self.leave_checkpoint()
        except (BrokenPipeError, HeadlessProtocolError):
            pass
        if process.poll() is None and process.stdin is not None:
            try:
                self._write(process, "QUIT\n")
            except BrokenPipeError:
                pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # start_new_session=True makes this exact forkserver PID its process
            # group leader.  Stop a currently running child together with the
            # parent instead of orphaning it behind a blocked waitpid().
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=timeout)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        self.process = None
        self.checkpoint_tick = None
        self.checkpoint_parent_tick = None
        self.step_child_active = False
        self.step_child_terminal = None
        self._read_buffer.clear()

    def __enter__(self) -> HeadlessForkserver:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
