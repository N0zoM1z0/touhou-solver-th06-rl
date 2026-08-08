"""No-PTY client for the Linux TH06 stage-entry COW forkserver."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import selectors
import signal
import shutil
import subprocess

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
        read_timeout: float = 30.0,
    ) -> None:
        if seed not in range(1 << 16):
            raise ValueError("headless seed must be 0..65535")
        if read_timeout <= 0.0:
            raise ValueError("forkserver read timeout must be positive")
        self.binary = binary.resolve()
        self.game_directory = game_directory.resolve()
        self.scope = scope
        self.seed = seed
        self.auto_shoot = auto_shoot
        self.read_timeout = read_timeout
        self.process: subprocess.Popen[str] | None = None
        self.checkpoint_tick: int | None = None
        self.checkpoint_parent_tick: int | None = None

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
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(self.read_timeout):
                raise HeadlessProtocolError(
                    f"forkserver response timed out; process status={process.poll()}"
                )
        line = process.stdout.readline()
        if not line:
            raise HeadlessProtocolError(
                f"forkserver ended before a response; status={process.poll()}"
            )
        return line.strip()

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
            text=True,
            bufsize=1,
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
        process.stdin.write(f"{command} {terminal_tick} {actions} {trace}\n")
        process.stdin.flush()
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
        process.stdin.write(f"CHECKPOINT {terminal_tick} {actions}\n")
        process.stdin.flush()
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
        process.stdin.write("QUIT\n")
        process.stdin.flush()
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
        try:
            if process.poll() is None and self.checkpoint_parent_tick is not None:
                self.leave_checkpoint()
        except (BrokenPipeError, HeadlessProtocolError):
            pass
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write("QUIT\n")
                process.stdin.flush()
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

    def __enter__(self) -> HeadlessForkserver:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
