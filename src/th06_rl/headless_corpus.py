"""Compact factual trajectories from the deterministic TH06 step environment.

The learned/teacher decision ranks only a native-certified action set. Full
source observations are retained as sparse compressed anchors; every compact
transition links two coherent observations by canonical SHA-256 digests.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

from .core.planner import LocalPlannerConfig
from .headless_geometry import (
    HARD_HORIZON,
    KINEMATICS,
    HeadlessAuthorityUnavailable,
    action_from_input,
    lower_headless_hazards,
    reactive_headless_action,
    validate_headless_observation,
)
from .native import ACTIONS, NativeCertifiedAction, NativeKernel, PackedHazards


TRANSITION_SCHEMA = "th06-rl-headless-transition-v1"
MANIFEST_SCHEMA = "th06-rl-headless-corpus-v1"
BEHAVIOR_POLICY = "epsilon-native-offline-teacher-v1"
HAZARD_SECTOR_COUNT = 8
HAZARD_FEATURE_DEFAULTS = {
    **{f"hazard_sector_{index}_near_count": 0.0 for index in range(HAZARD_SECTOR_COUNT)},
    **{
        f"hazard_sector_{index}_approaching_count": 0.0
        for index in range(HAZARD_SECTOR_COUNT)
    },
    **{
        f"hazard_sector_{index}_min_surface": 512.0
        for index in range(HAZARD_SECTOR_COUNT)
    },
    **{
        f"hazard_sector_{index}_min_projected_surface": 512.0
        for index in range(HAZARD_SECTOR_COUNT)
    },
}
HAZARD_FEATURE_NAMES = tuple(HAZARD_FEATURE_DEFAULTS)


def canonical_observation_sha256(observation: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        observation,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_context_id(observation: Mapping[str, Any]) -> str:
    """Derive a partition key without selecting a movement script."""
    enemies = observation.get("enemies")
    if not isinstance(enemies, list):
        raise HeadlessAuthorityUnavailable("headless enemies are incoherent")
    bosses = sorted(
        (
            int(enemy["slot"]),
            int(enemy["ecl_sub"]),
        )
        for enemy in enemies
        if isinstance(enemy, Mapping) and enemy.get("boss") is True
    )
    if bosses:
        return "boss:" + ",".join(f"{slot}/{sub}" for slot, sub in bosses)
    raw = observation.get("source_context")
    if not isinstance(raw, Mapping):
        return "source-context-unavailable"
    next_instruction = raw.get("next")
    if next_instruction is None:
        return "timeline:end"
    if not isinstance(next_instruction, Mapping):
        raise HeadlessAuthorityUnavailable("headless timeline context is incoherent")
    try:
        time = int(next_instruction["time"])
        opcode = int(next_instruction["opcode"])
        argument = int(next_instruction["arg0"])
    except (KeyError, TypeError, ValueError) as error:
        raise HeadlessAuthorityUnavailable("headless timeline identity is invalid") from error
    return f"timeline:{time}/{opcode}/{argument}"


def _optional_finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _boundary_reserve(x: float, y: float) -> float:
    return min(x - 8.0, 376.0 - x, y - 16.0, 432.0 - y)


def _finite_feature(item: Mapping[str, Any], name: str) -> float:
    try:
        value = float(item[name])
    except (KeyError, TypeError, ValueError) as error:
        raise HeadlessAuthorityUnavailable(f"headless feature {name} is invalid") from error
    if not math.isfinite(value):
        raise HeadlessAuthorityUnavailable(f"headless feature {name} is non-finite")
    return value


def compact_hazard_sector_features(observation: Mapping[str, Any]) -> dict[str, float]:
    """Summarize observed bullets without becoming collision authority.

    Eight fixed angular sectors retain local spatial and velocity information
    that a global bullet count discards. The work is one bounded pass over the
    already-observed physical bullet snapshot. These approximate descriptors
    are model inputs only; native certification remains the sole safety gate.
    """
    player = observation.get("player")
    bullets = observation.get("bullets")
    if not isinstance(player, Mapping) or not isinstance(bullets, list):
        raise HeadlessAuthorityUnavailable("headless hazard feature input is incoherent")
    player_x = _finite_feature(player, "x")
    player_y = _finite_feature(player, "y")
    player_half_width = _finite_feature(player, "half_width")
    player_half_height = _finite_feature(player, "half_height")
    values = dict(HAZARD_FEATURE_DEFAULTS)
    for bullet in bullets:
        if not isinstance(bullet, Mapping):
            raise HeadlessAuthorityUnavailable("headless bullet feature is incoherent")
        dx = _finite_feature(bullet, "x") - player_x
        dy = _finite_feature(bullet, "y") - player_y
        vx = _finite_feature(bullet, "vx")
        vy = _finite_feature(bullet, "vy")
        combined_x = player_half_width + _finite_feature(bullet, "half_width")
        combined_y = player_half_height + _finite_feature(bullet, "half_height")
        combined_radius = math.hypot(combined_x, combined_y)
        center_distance = math.hypot(dx, dy)
        surface = max(center_distance - combined_radius, 0.0)
        angle = math.atan2(dy, dx)
        sector = min(
            int((angle + math.pi) * HAZARD_SECTOR_COUNT / (2.0 * math.pi)),
            HAZARD_SECTOR_COUNT - 1,
        )
        near_name = f"hazard_sector_{sector}_near_count"
        approaching_name = f"hazard_sector_{sector}_approaching_count"
        surface_name = f"hazard_sector_{sector}_min_surface"
        projected_name = f"hazard_sector_{sector}_min_projected_surface"
        values[surface_name] = min(values[surface_name], surface, 512.0)
        if surface <= 160.0:
            values[near_name] += 1.0
        speed_squared = vx * vx + vy * vy
        closest_tick = 0.0
        if speed_squared > 1e-9:
            closest_tick = max(0.0, min(60.0, -(dx * vx + dy * vy) / speed_squared))
        projected_surface = max(
            math.hypot(dx + vx * closest_tick, dy + vy * closest_tick) - combined_radius,
            0.0,
        )
        values[projected_name] = min(values[projected_name], projected_surface, 512.0)
        if closest_tick > 0.0 and projected_surface <= 128.0:
            values[approaching_name] += 1.0
    return values


def compact_state_features(observation: Mapping[str, Any]) -> dict[str, Any]:
    player = observation["player"]
    assert isinstance(player, Mapping)
    enemies = observation["enemies"]
    assert isinstance(enemies, list)
    return {
        "player_x": float(player["x"]),
        "player_y": float(player["y"]),
        "player_focused": bool(player["focused"]),
        "previous_action": action_from_input(int(observation["input"])).name,
        "boundary_reserve": _boundary_reserve(float(player["x"]), float(player["y"])),
        "game_frame": int(observation["game_frame"]),
        "lives": int(observation["lives"]),
        "bombs_observed": int(observation["bombs"]),
        "power": int(observation["power"]),
        "rank": int(observation["rank"]),
        "graze": int(observation["graze"]),
        "bullet_count": len(observation["bullets"]),
        "laser_count": len(observation["lasers"]),
        "enemy_count": len(enemies),
        "boss_count": sum(
            isinstance(enemy, Mapping) and enemy.get("boss") is True
            for enemy in enemies
        ),
        **compact_hazard_sector_features(observation),
    }


def compact_candidate_records(
    certified: tuple[NativeCertifiedAction, ...],
    *,
    selected: str,
    teacher: str,
) -> list[dict[str, Any]]:
    return [
        {
            "action": item.action.name,
            "min_clearance": _optional_finite(item.min_clearance),
            "final_x": item.final_x,
            "final_y": item.final_y,
            "final_boundary_reserve": _boundary_reserve(item.final_x, item.final_y),
            "selected": item.action.name == selected,
            "teacher": item.action.name == teacher,
        }
        for item in certified
    ]


@dataclass(frozen=True)
class TeacherDecision:
    action: str
    kind: str
    effort_horizon: int
    surviving_actions: tuple[str, ...]


class NativeOfflineTeacher:
    """Offline-only local search; it never enlarges the Hard first-action set."""

    def __init__(self, *, kernel: NativeKernel | None = None, horizon: int = 12) -> None:
        if horizon < HARD_HORIZON:
            raise ValueError("teacher horizon cannot be shorter than the Hard gate")
        self.kernel = kernel or NativeKernel()
        self.horizon = horizon

    def rank(
        self,
        observation: Mapping[str, Any],
        certified: tuple[NativeCertifiedAction, ...],
        *,
        hazards: PackedHazards | None = None,
    ) -> TeacherDecision:
        if not certified:
            raise HeadlessAuthorityUnavailable("headless native safe set is empty")
        player = observation["player"]
        assert isinstance(player, Mapping)
        try:
            hazards = hazards or lower_headless_hazards(observation, self.horizon)
            plan = self.kernel.plan(
                x=float(player["x"]),
                y=float(player["y"]),
                half_width=float(player["half_width"]),
                half_height=float(player["half_height"]),
                kinematics=KINEMATICS,
                current_action=action_from_input(int(observation["input"])),
                hazards=hazards,
                hard=certified,
                config=LocalPlannerConfig(horizon=self.horizon),
            )
        except HeadlessAuthorityUnavailable:
            plan = None
        if plan is None:
            action = reactive_headless_action(observation, certified)
            return TeacherDecision(action.name, "generic-clearance-fallback", 0, ())
        return TeacherDecision(
            plan.action.name,
            "native-offline-local-plan",
            plan.effort_horizon,
            tuple(action.name for action in plan.surviving_actions),
        )


@dataclass(frozen=True)
class BehaviorDecision:
    selected_action: str
    probability: float
    teacher: TeacherDecision
    policy: str = BEHAVIOR_POLICY


class EpsilonTeacherBehavior:
    def __init__(self, *, epsilon: float, seed: int) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("behavior epsilon must be in 0..1")
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.seed = seed

    def select(
        self,
        teacher: TeacherDecision,
        certified: tuple[NativeCertifiedAction, ...],
    ) -> BehaviorDecision:
        names = tuple(item.action.name for item in certified)
        if not names or teacher.action not in names:
            raise HeadlessAuthorityUnavailable("teacher escaped the native safe set")
        if self.rng.random() < self.epsilon:
            selected = self.rng.choice(names)
        else:
            selected = teacher.action
        probability = self.epsilon / len(names)
        if selected == teacher.action:
            probability += 1.0 - self.epsilon
        return BehaviorDecision(selected, probability, teacher, BEHAVIOR_POLICY)


def build_transition(
    *,
    sequence: int,
    observation: Mapping[str, Any],
    next_observation: Mapping[str, Any],
    certified: tuple[NativeCertifiedAction, ...],
    behavior: BehaviorDecision,
    epsilon: float,
) -> dict[str, Any]:
    validate_headless_observation(observation)
    validate_headless_observation(next_observation)
    current_scope = observation.get("scope")
    if current_scope != next_observation.get("scope"):
        raise HeadlessAuthorityUnavailable("headless scope changed inside one transition")
    tick = int(observation["tick"])
    next_tick = int(next_observation["tick"])
    if next_tick != tick + 1:
        raise HeadlessAuthorityUnavailable("headless transition is not one physical tick")
    legal = tuple(item.action.name for item in certified)
    if behavior.selected_action not in legal:
        raise HeadlessAuthorityUnavailable("behavior action escaped the native safe set")
    terminal_reason = next_observation.get("terminal_reason")
    hit = terminal_reason == "physical-hit"
    next_player = next_observation["player"]
    assert isinstance(next_player, Mapping)
    return {
        "schema": TRANSITION_SCHEMA,
        "sequence": sequence,
        "tick": tick,
        "next_tick": next_tick,
        "scope": current_scope,
        "source_context": source_context_id(observation),
        "next_source_context": source_context_id(next_observation),
        "observation_sha256": canonical_observation_sha256(observation),
        "next_observation_sha256": canonical_observation_sha256(next_observation),
        "state": compact_state_features(observation),
        "legal_actions": list(legal),
        "action_candidates": compact_candidate_records(
            certified,
            selected=behavior.selected_action,
            teacher=behavior.teacher.action,
        ),
        "behavior": {
            "policy": behavior.policy,
            "epsilon": epsilon,
            "probability": behavior.probability,
            "selected_action": behavior.selected_action,
            "teacher_action": behavior.teacher.action,
        },
        "teacher": {
            "kind": behavior.teacher.kind,
            "effort_horizon": behavior.teacher.effort_horizon,
            "surviving_actions": list(behavior.teacher.surviving_actions),
        },
        "outcome_terms": {
            "physical_hit": hit,
            "terminal_reason": terminal_reason,
            "next_player_x": float(next_player["x"]),
            "next_player_y": float(next_player["y"]),
            "next_boundary_reserve": _boundary_reserve(
                float(next_player["x"]),
                float(next_player["y"]),
            ),
            "deaths_delta": int(next_observation["deaths"]) - int(observation["deaths"]),
            "bombs_used_delta": int(next_observation["bombs_used"])
            - int(observation["bombs_used"]),
        },
    }


class CompactHeadlessCorpusWriter:
    """Transactional gzip writer for compact transitions plus sparse anchors."""

    def __init__(self, run_directory: Path, *, anchor_stride: int = 120) -> None:
        if anchor_stride <= 0:
            raise ValueError("anchor stride must be positive")
        self.run_directory = run_directory.resolve()
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self.anchor_stride = anchor_stride
        self.transitions_partial = self.run_directory / "transitions.jsonl.gz.partial"
        self.anchors_partial = self.run_directory / "anchors.jsonl.gz.partial"
        self.transitions_path = self.run_directory / "transitions.jsonl.gz"
        self.anchors_path = self.run_directory / "anchors.jsonl.gz"
        self.transitions = gzip.open(self.transitions_partial, "wt", encoding="utf-8", compresslevel=3)
        self.anchors = gzip.open(self.anchors_partial, "wt", encoding="utf-8", compresslevel=3)
        self.transition_count = 0
        self.anchor_count = 0
        self.last_anchor_digest: str | None = None
        self.closed = False

    @staticmethod
    def _line(value: Mapping[str, Any]) -> str:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"

    def anchor(
        self,
        observation: Mapping[str, Any],
        *,
        sequence: int,
        role: str,
        force: bool = False,
    ) -> None:
        digest = canonical_observation_sha256(observation)
        if not force and sequence % self.anchor_stride != 0:
            return
        if digest == self.last_anchor_digest:
            return
        self.anchors.write(self._line({
            "schema": "th06-rl-headless-anchor-v1",
            "sequence": sequence,
            "role": role,
            "observation_sha256": digest,
            "observation": observation,
        }))
        self.last_anchor_digest = digest
        self.anchor_count += 1

    def transition(self, value: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("headless corpus writer is closed")
        self.transitions.write(self._line(value))
        self.transition_count += 1

    @staticmethod
    def _file_manifest(path: Path) -> dict[str, Any]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}

    def close(self, manifest: Mapping[str, Any]) -> Path:
        if self.closed:
            raise RuntimeError("headless corpus writer is already closed")
        self.closed = True
        self.transitions.close()
        self.anchors.close()
        os.replace(self.transitions_partial, self.transitions_path)
        os.replace(self.anchors_partial, self.anchors_path)
        complete = {
            "schema": MANIFEST_SCHEMA,
            **manifest,
            "transition_count": self.transition_count,
            "anchor_count": self.anchor_count,
            "files": {
                "transitions": self._file_manifest(self.transitions_path),
                "anchors": self._file_manifest(self.anchors_path),
            },
        }
        temporary = self.run_directory / "manifest.json.partial"
        temporary.write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        final = self.run_directory / "manifest.json"
        os.replace(temporary, final)
        return final

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.transitions.close()
        self.anchors.close()
