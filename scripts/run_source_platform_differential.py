#!/usr/bin/env python3
"""Replay one bounded action stream in Linux source and Win32 source under Wine.

This is a platform/delivery diagnostic, not promotion evidence for the shipped
game.  Both runtimes receive the same fixed scope, seed, maximum tick count,
and Bomb-free run-length action file.  The report preserves exact physical and
event divergence; a configurable float tolerance is reported separately and
never changes the exact result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from th06_rl.headless_corpus import canonical_observation_sha256
from th06_rl.offline import ACTION_SET

try:
    from compare_headless_traces import first_difference
except ModuleNotFoundError:  # Imported as scripts.run_source_platform_differential.
    from scripts.compare_headless_traces import first_difference


ACTION_STREAM_SCHEMA = "th06-rl-source-action-stream-v1"
DELIVERY_CONTRACT = "source-headless-run-length-one-action-per-tick-v1"
REPORT_SCHEMA = "th06-rl-source-platform-differential-v1"
PREFIX_MARKER = ".th06-rl-source-differential-v1"
OBSERVATION_SCHEMA = "th06-headless-observation-v2"
DRIFT_TOLERANCES = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
DISCRETE_DELIVERY_FIELDS = (
    "schema",
    "tick",
    "terminal_reason",
    "scope",
    "initial_seed",
    "supervisor_state",
    "stage",
    "game_frame",
    "rng_seed",
    "rng_generation",
    "input",
    "lives",
    "bombs",
    "score",
    "power",
    "rank",
    "deaths",
    "bombs_used",
    "graze",
)
_MISSING = object()


@dataclass(frozen=True)
class ActionSegment:
    count: int
    action: str


@dataclass(frozen=True)
class SourceActionStream:
    difficulty: int
    character: int
    shot_type: int
    stage: int
    initial_seed: int
    max_ticks: int
    auto_shoot: bool
    segments: tuple[ActionSegment, ...]
    stage_rng_seed: int | None = None
    auto_shoot_after_tick: int | None = None
    retail_dialogue_control: bool = False
    retail_dialogue_control_after_tick: int | None = None
    description: str | None = None
    provenance: dict[str, Any] | None = None

    @property
    def action_count(self) -> int:
        return sum(segment.count for segment in self.segments)

    def as_object(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": ACTION_STREAM_SCHEMA,
            "delivery_contract": DELIVERY_CONTRACT,
            "scope": {
                "difficulty": self.difficulty,
                "character": self.character,
                "shot_type": self.shot_type,
                "stage": self.stage,
            },
            "initial_seed": self.initial_seed,
            "max_ticks": self.max_ticks,
            "auto_shoot": self.auto_shoot,
            "segments": [
                {"count": segment.count, "action": segment.action}
                for segment in self.segments
            ],
        }
        if self.description is not None:
            result["description"] = self.description
        if self.stage_rng_seed is not None:
            result["stage_rng_seed"] = self.stage_rng_seed
        if self.auto_shoot_after_tick is not None:
            result["auto_shoot_after_tick"] = self.auto_shoot_after_tick
        if self.retail_dialogue_control:
            result["retail_dialogue_control"] = True
        if self.retail_dialogue_control_after_tick is not None:
            result["retail_dialogue_control_after_tick"] = (
                self.retail_dialogue_control_after_tick
            )
        if self.provenance is not None:
            result["provenance"] = self.provenance
        return result


def _strict_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def parse_action_stream(raw: object) -> SourceActionStream:
    if not isinstance(raw, dict):
        raise ValueError("action stream root must be an object")
    allowed = {
        "schema",
        "delivery_contract",
        "scope",
        "initial_seed",
        "stage_rng_seed",
        "auto_shoot_after_tick",
        "retail_dialogue_control",
        "retail_dialogue_control_after_tick",
        "max_ticks",
        "auto_shoot",
        "segments",
        "description",
        "provenance",
    }
    if set(raw) - allowed:
        raise ValueError(f"unknown action stream members: {sorted(set(raw) - allowed)}")
    if raw.get("schema") != ACTION_STREAM_SCHEMA:
        raise ValueError("unsupported source action stream schema")
    if raw.get("delivery_contract") != DELIVERY_CONTRACT:
        raise ValueError("unsupported source action delivery contract")
    scope = raw.get("scope")
    if not isinstance(scope, dict) or set(scope) != {
        "difficulty",
        "character",
        "shot_type",
        "stage",
    }:
        raise ValueError("scope must contain exactly difficulty/character/shot_type/stage")
    difficulty = _strict_int(scope["difficulty"], "difficulty", 0, 3)
    character = _strict_int(scope["character"], "character", 0, 1)
    shot_type = _strict_int(scope["shot_type"], "shot_type", 0, 1)
    stage = _strict_int(scope["stage"], "stage", 1, 6)
    initial_seed = _strict_int(raw.get("initial_seed"), "initial_seed", 0, 65535)
    stage_rng_seed_raw = raw.get("stage_rng_seed")
    stage_rng_seed = (
        None
        if stage_rng_seed_raw is None
        else _strict_int(stage_rng_seed_raw, "stage_rng_seed", 0, 65535)
    )
    max_ticks = _strict_int(raw.get("max_ticks"), "max_ticks", 1, 10_000_000)
    if type(raw.get("auto_shoot")) is not bool:
        raise ValueError("auto_shoot must be a boolean")
    auto_shoot_after_tick_raw = raw.get("auto_shoot_after_tick")
    auto_shoot_after_tick = (
        None
        if auto_shoot_after_tick_raw is None
        else _strict_int(
            auto_shoot_after_tick_raw,
            "auto_shoot_after_tick",
            0,
            max_ticks,
        )
    )
    if auto_shoot_after_tick is not None and raw["auto_shoot"] is not True:
        raise ValueError("auto_shoot_after_tick requires auto_shoot")
    retail_dialogue_control = raw.get("retail_dialogue_control", False)
    if type(retail_dialogue_control) is not bool:
        raise ValueError("retail_dialogue_control must be a boolean")
    if retail_dialogue_control and raw["auto_shoot"] is not True:
        raise ValueError("retail_dialogue_control requires auto_shoot")
    retail_dialogue_after_raw = raw.get("retail_dialogue_control_after_tick")
    retail_dialogue_control_after_tick = (
        None
        if retail_dialogue_after_raw is None
        else _strict_int(
            retail_dialogue_after_raw,
            "retail_dialogue_control_after_tick",
            0,
            max_ticks,
        )
    )
    if retail_dialogue_control_after_tick is not None and not retail_dialogue_control:
        raise ValueError(
            "retail_dialogue_control_after_tick requires retail_dialogue_control"
        )
    segments_raw = raw.get("segments")
    if not isinstance(segments_raw, list) or not segments_raw:
        raise ValueError("segments must be a non-empty list")
    segments: list[ActionSegment] = []
    for index, segment in enumerate(segments_raw):
        if not isinstance(segment, dict) or set(segment) != {"count", "action"}:
            raise ValueError(f"segment {index} must contain exactly count/action")
        count = _strict_int(segment["count"], f"segment {index} count", 1, 10_000_000)
        action = segment["action"]
        if not isinstance(action, str) or action not in ACTION_SET:
            raise ValueError(f"segment {index} has unknown or forbidden action {action!r}")
        segments.append(ActionSegment(count=count, action=action))
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError("description must be a string when present")
    provenance = raw.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object when present")
        try:
            json.dumps(provenance, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("provenance must contain finite JSON values") from error
    stream = SourceActionStream(
        difficulty=difficulty,
        character=character,
        shot_type=shot_type,
        stage=stage,
        initial_seed=initial_seed,
        max_ticks=max_ticks,
        auto_shoot=raw["auto_shoot"],
        segments=tuple(segments),
        stage_rng_seed=stage_rng_seed,
        auto_shoot_after_tick=auto_shoot_after_tick,
        retail_dialogue_control=retail_dialogue_control,
        retail_dialogue_control_after_tick=retail_dialogue_control_after_tick,
        description=description,
        provenance=dict(provenance) if provenance is not None else None,
    )
    if stream.action_count < stream.max_ticks:
        raise ValueError(
            "action stream must cover max_ticks so delivery cannot fail by exhaustion"
        )
    return stream


def load_action_stream(path: Path) -> SourceActionStream:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid action stream JSON: {error}") from error
    return parse_action_stream(raw)


def render_action_file(stream: SourceActionStream) -> str:
    return "".join(f"{segment.count} {segment.action}\n" for segment in stream.segments)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_path(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_absolute():
        raise ValueError("Wine path must be absolute")
    return "Z:" + str(resolved).replace("/", "\\")


def _git_snapshot(repository: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"path": str(repository.resolve()), "commit": commit, "dirty": bool(status), "status": status}


def _file_evidence(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    ]


def _runtime_command(
    stream: SourceActionStream,
    *,
    binary: str,
    actions: str,
    trace: str,
) -> list[str]:
    command = [
        binary,
        "--headless",
        "--seed",
        str(stream.initial_seed),
        "--max-ticks",
        str(stream.max_ticks),
        "--practice-stage",
        str(stream.stage),
        "--difficulty",
        str(stream.difficulty),
        "--character",
        str(stream.character),
        "--shot-type",
        str(stream.shot_type),
        "--actions",
        actions,
        "--trace",
        trace,
    ]
    if stream.stage_rng_seed is not None:
        command.extend(("--stage-rng-seed", str(stream.stage_rng_seed)))
    if stream.auto_shoot:
        command.append("--auto-shoot")
    if stream.auto_shoot_after_tick is not None:
        command.extend(
            ("--auto-shoot-after-tick", str(stream.auto_shoot_after_tick))
        )
    if stream.retail_dialogue_control:
        command.append("--retail-dialogue-control")
    if stream.retail_dialogue_control_after_tick is not None:
        command.extend(
            (
                "--retail-dialogue-control-after-tick",
                str(stream.retail_dialogue_control_after_tick),
            )
        )
    return command


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=5.0)
    return {
        "command": list(command),
        "cwd": str(cwd),
        "pid": process.pid,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _prefix_processes(prefix: Path) -> list[dict[str, Any]]:
    wanted = os.path.realpath(prefix)
    matches: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            values: dict[str, str] = {}
            for item in (entry / "environ").read_bytes().split(b"\0"):
                if b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                if key == b"WINEPREFIX":
                    values["WINEPREFIX"] = value.decode(errors="replace")
            if os.path.realpath(values.get("WINEPREFIX", "")) != wanted:
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            ).strip()
            matches.append({"pid": int(entry.name), "command": command})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(matches, key=lambda item: item["pid"])


def _owned_wine_prefix(prefix: Path) -> None:
    marker = prefix / PREFIX_MARKER
    if prefix.exists() and any(prefix.iterdir()) and not marker.is_file():
        raise ValueError(f"refusing unmarked/non-dedicated Wine prefix: {prefix}")
    prefix.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text(REPORT_SCHEMA + "\n", encoding="utf-8")


def _shutdown_wine_prefix(
    *,
    wineserver: Path,
    environment: Mapping[str, str],
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for option in ("-k", "-w"):
        try:
            result = subprocess.run(
                [str(wineserver), option],
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            results.append(
                {
                    "command": [str(wineserver), option],
                    "returncode": result.returncode,
                    "stdout": result.stdout.decode(errors="replace"),
                    "stderr": result.stderr.decode(errors="replace"),
                }
            )
        except subprocess.TimeoutExpired as error:
            results.append(
                {
                    "command": [str(wineserver), option],
                    "timed_out": True,
                    "stdout": (error.stdout or b"").decode(errors="replace"),
                    "stderr": (error.stderr or b"").decode(errors="replace"),
                }
            )
            break
    return {"commands": results}


def _wait_for_prefix_idle(
    prefix: Path,
    *,
    stable_seconds: float = 3.0,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Require a stable idle interval so delayed Wine helpers are not missed."""
    deadline = time.monotonic() + timeout_seconds
    idle_since: float | None = None
    observed: dict[int, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        current = _prefix_processes(prefix)
        for process in current:
            observed[process["pid"]] = process
        if current:
            idle_since = None
        else:
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since >= stable_seconds:
                return {
                    "idle": True,
                    "stable_seconds": stable_seconds,
                    "observed_processes": sorted(observed.values(), key=lambda item: item["pid"]),
                    "remaining_processes": [],
                }
        time.sleep(0.1)
    remaining = _prefix_processes(prefix)
    return {
        "idle": False,
        "stable_seconds": stable_seconds,
        "observed_processes": sorted(observed.values(), key=lambda item: item["pid"]),
        "remaining_processes": remaining,
    }


def _trace_summary(path: Path) -> dict[str, Any]:
    rows = 0
    first_tick: int | None = None
    last_tick: int | None = None
    terminal_reason: str | None = None
    hit_tick: int | None = None
    final_discrete: dict[str, Any] | None = None
    terminal_hit: object = None
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                observation = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid trace JSON at {path}:{line_number}: {error}") from error
            if not isinstance(observation, dict) or observation.get("schema") != OBSERVATION_SCHEMA:
                raise ValueError(f"invalid observation schema at {path}:{line_number}")
            tick = observation.get("tick")
            if type(tick) is not int:
                raise ValueError(f"invalid tick at {path}:{line_number}")
            if last_tick is not None and tick <= last_tick:
                raise ValueError(f"non-increasing tick at {path}:{line_number}")
            rows += 1
            first_tick = tick if first_tick is None else first_tick
            last_tick = tick
            terminal = observation.get("terminal_reason")
            terminal_reason = terminal if isinstance(terminal, str) else terminal_reason
            events = observation.get("events")
            if hit_tick is None and isinstance(events, dict) and events.get("hit") is not None:
                hit_tick = tick
            if hit_tick is None and terminal == "physical-hit":
                hit_tick = tick
            final_discrete = {
                key: observation.get(key) for key in DISCRETE_DELIVERY_FIELDS
            }
            terminal_hit = events.get("hit") if isinstance(events, dict) else None
    if rows == 0:
        raise ValueError(f"empty trace: {path}")
    raw_digest = hashlib.sha256()
    normalized_digest = hashlib.sha256()
    crlf_lines = 0
    lf_only_lines = 0
    with path.open("rb") as source:
        for line in source:
            raw_digest.update(line)
            if line.endswith(b"\r\n"):
                crlf_lines += 1
                normalized_digest.update(line[:-2] + b"\n")
            else:
                if line.endswith(b"\n"):
                    lf_only_lines += 1
                normalized_digest.update(line)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": raw_digest.hexdigest(),
        "lf_normalized_sha256": normalized_digest.hexdigest(),
        "crlf_lines": crlf_lines,
        "lf_only_lines": lf_only_lines,
        "rows": rows,
        "first_tick": first_tick,
        "last_tick": last_tick,
        "terminal_reason": terminal_reason,
        "first_hit_tick": hit_tick,
        "final_discrete": final_discrete,
        "terminal_hit": terminal_hit,
    }


def _discrete_delivery(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {key: observation.get(key) for key in DISCRETE_DELIVERY_FIELDS}


def _difference_record(
    *,
    line: int,
    left: Mapping[str, Any] | object,
    right: Mapping[str, Any] | object,
    difference: dict[str, Any],
) -> dict[str, Any]:
    left_tick = left.get("tick") if isinstance(left, Mapping) else None
    right_tick = right.get("tick") if isinstance(right, Mapping) else None
    result = {
        "line": line,
        "left_tick": left_tick,
        "right_tick": right_tick,
        "difference": difference,
    }
    if isinstance(left, Mapping):
        result["left_physical_sha256"] = canonical_observation_sha256(left)
    if isinstance(right, Mapping):
        result["right_physical_sha256"] = canonical_observation_sha256(right)
    return result


def compare_source_traces(
    left_path: Path,
    right_path: Path,
    *,
    absolute_tolerance: float,
) -> dict[str, Any]:
    """Compare exact physical state, events, and tolerant physical state separately."""
    if absolute_tolerance < 0.0:
        raise ValueError("absolute tolerance must be nonnegative")
    first_exact: dict[str, Any] | None = None
    first_events: dict[str, Any] | None = None
    first_tolerant: dict[str, Any] | None = None
    first_discrete: dict[str, Any] | None = None
    first_by_tolerance: dict[float, dict[str, Any] | None] = {
        tolerance: None
        for tolerance in dict.fromkeys((absolute_tolerance, *DRIFT_TOLERANCES))
    }
    overlap_rows = 0
    with left_path.open(encoding="utf-8") as left_file, right_path.open(encoding="utf-8") as right_file:
        for line_number, (left_line, right_line) in enumerate(
            zip_longest(left_file, right_file, fillvalue=_MISSING), start=1
        ):
            if left_line is _MISSING or right_line is _MISSING:
                difference = {
                    "path": "$",
                    "left": "missing" if left_line is _MISSING else "present",
                    "right": "missing" if right_line is _MISSING else "present",
                    "reason": "trace-length",
                }
                record = {
                    "line": line_number,
                    "left_tick": None,
                    "right_tick": None,
                    "difference": difference,
                }
                first_exact = first_exact or record
                first_events = first_events or record
                first_tolerant = first_tolerant or record
                first_discrete = first_discrete or record
                for tolerance in first_by_tolerance:
                    first_by_tolerance[tolerance] = first_by_tolerance[tolerance] or record
                break
            try:
                left = json.loads(left_line)
                right = json.loads(right_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid comparison JSONL at line {line_number}: {error}") from error
            if not isinstance(left, dict) or not isinstance(right, dict):
                raise ValueError(f"comparison row {line_number} is not an object")
            overlap_rows += 1
            left_physical = {key: value for key, value in left.items() if key != "events"}
            right_physical = {key: value for key, value in right.items() if key != "events"}
            if first_exact is None:
                difference = first_difference(left_physical, right_physical)
                if difference is not None:
                    first_exact = _difference_record(
                        line=line_number, left=left, right=right, difference=difference
                    )
            if first_events is None:
                difference = first_difference(left.get("events"), right.get("events"))
                if difference is not None:
                    first_events = _difference_record(
                        line=line_number, left=left, right=right, difference=difference
                    )
            if first_discrete is None:
                difference = first_difference(
                    _discrete_delivery(left), _discrete_delivery(right)
                )
                if difference is not None:
                    first_discrete = _difference_record(
                        line=line_number, left=left, right=right, difference=difference
                    )
            for tolerance, recorded in first_by_tolerance.items():
                if recorded is not None:
                    continue
                difference = first_difference(
                    left_physical,
                    right_physical,
                    absolute_tolerance=tolerance,
                )
                if difference is not None:
                    first_by_tolerance[tolerance] = _difference_record(
                        line=line_number, left=left, right=right, difference=difference
                    )
    first_tolerant = first_by_tolerance[absolute_tolerance]
    return {
        "overlap_rows": overlap_rows,
        "exact_physical": {
            "equal": first_exact is None,
            "matched_prefix_rows": overlap_rows if first_exact is None else first_exact["line"] - 1,
            "first_divergence": first_exact,
        },
        "events": {
            "equal": first_events is None,
            "matched_prefix_rows": overlap_rows if first_events is None else first_events["line"] - 1,
            "first_divergence": first_events,
        },
        "discrete_delivery": {
            "fields": list(DISCRETE_DELIVERY_FIELDS),
            "equal": first_discrete is None,
            "matched_prefix_rows": overlap_rows if first_discrete is None else first_discrete["line"] - 1,
            "first_divergence": first_discrete,
        },
        "tolerant_physical": {
            "absolute_tolerance": absolute_tolerance,
            "equal": first_tolerant is None,
            "matched_prefix_rows": overlap_rows if first_tolerant is None else first_tolerant["line"] - 1,
            "first_divergence": first_tolerant,
        },
        "tolerance_ladder": [
            {
                "absolute_tolerance": tolerance,
                "equal": difference is None,
                "matched_prefix_rows": overlap_rows if difference is None else difference["line"] - 1,
                "first_divergence": difference,
            }
            for tolerance, difference in first_by_tolerance.items()
        ],
    }


def _validate_paths(paths: Sequence[tuple[str, Path]], parser: argparse.ArgumentParser) -> None:
    for label, path in paths:
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-stream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-repository",
        type=Path,
        default=repository / "reference/GensokyoClub-th06-portable",
    )
    parser.add_argument(
        "--linux-binary",
        type=Path,
        default=repository / "reference/GensokyoClub-th06-portable/th06",
    )
    parser.add_argument(
        "--windows-binary",
        type=Path,
        default=repository / "reference/GensokyoClub-th06-portable/th06.exe",
    )
    parser.add_argument(
        "--game-directory",
        type=Path,
        default=repository / "reference/th06-game-original/th06",
    )
    parser.add_argument("--wine", type=Path, default=Path("/usr/bin/wine"))
    parser.add_argument("--wineboot", type=Path, default=Path("/usr/bin/wineboot"))
    parser.add_argument("--wineserver", type=Path, default=Path("/usr/bin/wineserver"))
    parser.add_argument(
        "--wine-prefix",
        type=Path,
        default=Path("/home/c/.wine-th06-rl-source-diff"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)

    if args.timeout_seconds <= 0.0:
        parser.error("timeout must be positive")
    if args.absolute_tolerance < 0.0:
        parser.error("absolute tolerance must be nonnegative")
    try:
        stream = load_action_stream(args.action_stream)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    source_repository = args.source_repository.resolve()
    linux_binary = args.linux_binary.resolve()
    windows_binary = args.windows_binary.resolve()
    game_directory = args.game_directory.resolve()
    # Wine's launcher tools can be symlinks into one multicall executable.  The
    # invoked basename selects wine vs wineboot, so resolving those symlinks
    # changes behavior (for example, ``wine -u`` tries to launch a file).
    wine = args.wine.absolute()
    wineboot = args.wineboot.absolute()
    wineserver = args.wineserver.absolute()
    wine_prefix = args.wine_prefix.resolve()
    output = args.output.resolve()
    _validate_paths(
        (
            ("Linux source binary", linux_binary),
            ("Windows source binary", windows_binary),
            ("Wine", wine),
            ("wineboot", wineboot),
            ("wineserver", wineserver),
        ),
        parser,
    )
    if not source_repository.is_dir():
        parser.error(f"source repository does not exist: {source_repository}")
    if not game_directory.is_dir():
        parser.error(f"game directory does not exist: {game_directory}")
    if output.exists() and any(output.iterdir()):
        parser.error("output must be absent or empty")
    if _prefix_processes(wine_prefix):
        parser.error(f"dedicated Wine prefix already has live processes: {wine_prefix}")
    try:
        _owned_wine_prefix(wine_prefix)
    except ValueError as error:
        parser.error(str(error))

    output.mkdir(parents=True, exist_ok=True)
    normalized_stream = output / "action-stream.json"
    normalized_stream.write_text(
        json.dumps(stream.as_object(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    actions = output / "actions.txt"
    actions.write_text(render_action_file(stream), encoding="ascii")
    linux_trace = output / "linux.trace.jsonl"
    wine_trace = output / "wine.trace.jsonl"

    base_environment = os.environ.copy()
    linux_command = _runtime_command(
        stream,
        binary=str(linux_binary),
        actions=str(actions),
        trace=str(linux_trace),
    )
    linux_run = _run_process(
        linux_command,
        cwd=game_directory,
        environment=base_environment,
        stdout_path=output / "linux.stdout.log",
        stderr_path=output / "linux.stderr.log",
        timeout_seconds=args.timeout_seconds,
    )

    wine_environment = base_environment.copy()
    wine_environment.update(
        {
            "WINEPREFIX": str(wine_prefix),
            "WINEARCH": "win32",
            "WINEDEBUG": "-all",
            "WINEDLLOVERRIDES": "mscoree,mshtml=",
        }
    )
    wineboot_run = _run_process(
        [str(wineboot), "-u"],
        cwd=game_directory,
        environment=wine_environment,
        stdout_path=output / "wineboot.stdout.log",
        stderr_path=output / "wineboot.stderr.log",
        timeout_seconds=max(args.timeout_seconds, 120.0),
    )
    wineboot_cleanup = _shutdown_wine_prefix(
        wineserver=wineserver, environment=wine_environment
    )
    wineboot_cleanup["process_audit"] = _wait_for_prefix_idle(wine_prefix)
    wineboot_cleanup["remaining_processes"] = wineboot_cleanup["process_audit"][
        "remaining_processes"
    ]
    if wineboot_cleanup["remaining_processes"]:
        raise RuntimeError("Wine prefix did not become idle after wineboot")

    windows_runtime_command = _runtime_command(
        stream,
        binary=_windows_path(windows_binary),
        actions=_windows_path(actions),
        trace=_windows_path(wine_trace),
    )
    wine_command = [str(wine), *windows_runtime_command]
    wine_run = _run_process(
        wine_command,
        cwd=game_directory,
        environment=wine_environment,
        stdout_path=output / "wine.stdout.log",
        stderr_path=output / "wine.stderr.log",
        timeout_seconds=args.timeout_seconds,
    )
    wine_cleanup = _shutdown_wine_prefix(
        wineserver=wineserver, environment=wine_environment
    )
    wine_cleanup["process_audit"] = _wait_for_prefix_idle(wine_prefix)
    wine_cleanup["remaining_processes"] = wine_cleanup["process_audit"][
        "remaining_processes"
    ]

    linux_summary = _trace_summary(linux_trace)
    wine_summary = _trace_summary(wine_trace)
    comparison = compare_source_traces(
        linux_trace,
        wine_trace,
        absolute_tolerance=args.absolute_tolerance,
    )
    comparison["raw_serialization"] = {
        "byte_equal": linux_summary["sha256"] == wine_summary["sha256"],
        "lf_normalized_equal": (
            linux_summary["lf_normalized_sha256"]
            == wine_summary["lf_normalized_sha256"]
        ),
        "linux_crlf_lines": linux_summary["crlf_lines"],
        "wine_crlf_lines": wine_summary["crlf_lines"],
    }
    outcome_agreement = {
        "terminal_reason": linux_summary["terminal_reason"] == wine_summary["terminal_reason"],
        "last_tick": linux_summary["last_tick"] == wine_summary["last_tick"],
        "first_hit_tick": linux_summary["first_hit_tick"] == wine_summary["first_hit_tick"],
    }
    completed = (
        linux_run["returncode"] == 0
        and not linux_run["timed_out"]
        and wineboot_run["returncode"] == 0
        and not wineboot_run["timed_out"]
        and wine_run["returncode"] == 0
        and not wine_run["timed_out"]
        and not wine_cleanup["remaining_processes"]
    )
    if not completed:
        conclusion = "runner-failure"
    elif comparison["exact_physical"]["equal"] and comparison["events"]["equal"]:
        conclusion = (
            "exact-source-observation-match"
            if comparison["raw_serialization"]["byte_equal"]
            else "exact-source-observation-match-with-serialization-drift"
        )
    elif comparison["tolerant_physical"]["equal"] and all(outcome_agreement.values()):
        conclusion = "bounded-numeric-drift-with-common-outcome"
    else:
        conclusion = "source-platform-divergence"

    data_files = sorted(
        path
        for path in game_directory.iterdir()
        if path.is_file() and (path.suffix.lower() == ".dat" or path.name == "msgothic.ttc")
    )
    dlls = sorted(windows_binary.parent.glob("*.dll"), key=lambda path: path.name.lower())
    report = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_boundary": (
            "paired reconstructed-source platform/delivery diagnostic only; "
            "not original-retail or real-Windows promotion evidence"
        ),
        "conclusion": conclusion,
        "completed": completed,
        "action_stream": {
            **stream.as_object(),
            "path": str(normalized_stream),
            "sha256": _sha256(normalized_stream),
            "action_file_sha256": _sha256(actions),
            "action_count": stream.action_count,
        },
        "source": _git_snapshot(source_repository),
        "binaries": {
            "linux": _file_evidence([linux_binary])[0],
            "windows_bundle": _file_evidence([windows_binary, *dlls]),
        },
        "game_data": _file_evidence(data_files),
        "wine": {
            "binary": str(wine),
            "prefix": str(wine_prefix),
            "wineboot": wineboot_run,
            "wineboot_cleanup": wineboot_cleanup,
            "cleanup": wine_cleanup,
        },
        "runs": {"linux": linux_run, "wine": wine_run},
        "traces": {"linux": linux_summary, "wine": wine_summary},
        "comparison": comparison,
        "outcome_agreement": outcome_agreement,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "conclusion": conclusion}, sort_keys=True))
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
