"""Stable hot-reload boundary above the non-learning reactive baseline."""

from __future__ import annotations

from dataclasses import dataclass


POLICY_API_VERSION = 1


@dataclass(frozen=True)
class PolicyContext:
    frame: int
    scope: tuple[int, int, int, int]
    source_context: str
    baseline_action: str
    locally_admissible_actions: tuple[str, ...]
    player_x: float
    player_y: float
    power: int
    bullet_count: int
    laser_count: int
    hard_action_count: int
    exploration_rate: float
    current_action: str = "stay"
    hard_admissible_actions: tuple[str, ...] = ()
    phase_elapsed_frames: int = 0
    # Already-computed four-frame native certificates.  Policies may use this
    # bounded geometry as ranking evidence, but cannot add an action to the
    # authoritative safe set.
    hard_action_evaluations: tuple[
        tuple[str, float | None, float, float], ...
    ] = ()
    # The controller's already-computed advisory horizon. Four means the
    # longer constant-action frontier closed and only the authoritative
    # Hard-4 set remains. It is generic pressure evidence, not a phase key.
    effort_horizon: int = 0


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    policy_id: str
    behavior_probability: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.behavior_probability <= 1.0:
            raise ValueError("behavior probability must be in (0, 1]")


@dataclass(frozen=True)
class PolicyOutcome:
    frame: int
    scope: tuple[int, int, int, int]
    source_context: str
    action: str
    published: bool
    elapsed_frames: int
    life_lost: bool
    bomb_used: bool
    control_dead_end: bool
    authority_lost: bool
    phase_changed: bool
    next_hard_action_count: int
    next_player_x: float
    next_player_y: float
    learning_eligible: bool = True


@dataclass(frozen=True)
class PolicyFailureEvent:
    """Confirmed physical failure delivered independently of publication."""

    frame: int
    scope: tuple[int, int, int, int]
    source_context: str
    kind: str
