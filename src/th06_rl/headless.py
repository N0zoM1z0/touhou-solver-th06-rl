"""Small no-PTY client for the source-derived TH06 headless step protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import selectors
import shutil
import subprocess
from typing import Any

from .offline import ACTION_SET


@dataclass(frozen=True)
class HeadlessScope:
    difficulty: int
    character: int
    shot_type: int
    stage: int

    def __post_init__(self) -> None:
        if self.difficulty not in range(4):
            raise ValueError("headless difficulty must be 0..3")
        if self.character not in range(2):
            raise ValueError("headless character must be 0..1")
        if self.shot_type not in range(2):
            raise ValueError("headless shot type must be 0..1")
        if self.stage not in range(1, 7):
            raise ValueError("headless Practice stage must be 1..6")


class HeadlessProtocolError(RuntimeError):
    pass


class HeadlessClient:
    """Own one deterministic TH06 subprocess and exchange one action per tick."""

    def __init__(
        self,
        *,
        binary: Path,
        game_directory: Path,
        scope: HeadlessScope,
        seed: int,
        max_ticks: int = 0,
        auto_shoot: bool = True,
        continue_after_hit: bool = False,
        read_timeout: float = 30.0,
    ) -> None:
        if seed not in range(1 << 16):
            raise ValueError("headless seed must be 0..65535")
        if max_ticks < 0:
            raise ValueError("maximum ticks must be nonnegative")
        if read_timeout <= 0.0:
            raise ValueError("read timeout must be positive")
        self.binary = binary.resolve()
        self.game_directory = game_directory.resolve()
        self.scope = scope
        self.seed = seed
        self.max_ticks = max_ticks
        self.auto_shoot = auto_shoot
        self.continue_after_hit = continue_after_hit
        self.read_timeout = read_timeout
        self.process: subprocess.Popen[str] | None = None
        self.terminal = False

    def _command(self) -> list[str]:
        command = [
            str(self.binary),
            "--headless",
            "--step",
            "--seed",
            str(self.seed),
            "--max-ticks",
            str(self.max_ticks),
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
        if self.continue_after_hit:
            command.append("--continue-after-hit")
        nice = shutil.which("nice")
        ionice = shutil.which("ionice")
        if nice is not None:
            command = [nice, "-n", "15", *command]
        if ionice is not None:
            command = [ionice, "-c", "2", "-n", "7", *command]
        return command

    def start(self) -> dict[str, Any]:
        if self.process is not None:
            raise HeadlessProtocolError("headless process already started")
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
        return self._read_observation()

    def step(self, action: str) -> dict[str, Any]:
        if action not in ACTION_SET:
            raise ValueError(f"unknown or forbidden headless action: {action}")
        if self.process is None or self.process.stdin is None:
            raise HeadlessProtocolError("headless process is not running")
        if self.terminal:
            raise HeadlessProtocolError("cannot step a terminal headless episode")
        self.process.stdin.write(action + "\n")
        self.process.stdin.flush()
        return self._read_observation()

    def _read_observation(self) -> dict[str, Any]:
        assert self.process is not None
        assert self.process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.read_timeout):
                status = self.process.poll()
                raise HeadlessProtocolError(
                    f"headless observation timed out; process status={status}"
                )
        line = self.process.stdout.readline()
        if not line:
            status = self.process.poll()
            raise HeadlessProtocolError(
                f"headless process ended before an observation; status={status}"
            )
        try:
            observation = json.loads(line)
        except json.JSONDecodeError as error:
            raise HeadlessProtocolError(f"invalid headless JSON observation: {error}") from error
        if not isinstance(observation, dict):
            raise HeadlessProtocolError("headless observation is not a JSON object")
        self.terminal = observation.get("terminal_reason") is not None
        return observation

    def close(self, timeout: float = 5.0) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)
        if process.stdout is not None:
            process.stdout.close()

    def __enter__(self) -> "HeadlessClient":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
