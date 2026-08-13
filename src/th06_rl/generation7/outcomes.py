"""Physical-HIT-only cost, terminal, and conservation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class OptionOutcome:
    hit_cost: int
    duration_frames: int
    terminal: bool

    def __post_init__(self) -> None:
        if self.hit_cost < 0 or self.duration_frames <= 0:
            raise ValueError("option outcome cost/duration is invalid")


def reward_from_hit_cost(hit_cost: int) -> float:
    if hit_cost < 0:
        raise ValueError("physical HIT cost cannot be negative")
    return -float(hit_cost)


def undiscounted_hit_returns(
    outcomes: tuple[OptionOutcome, ...],
) -> tuple[float, ...]:
    """Return factual future HIT cost with gamma=1 and terminal value zero."""
    if not outcomes or not outcomes[-1].terminal:
        raise ValueError("episode must end at an explicit terminal option")
    if any(outcome.terminal for outcome in outcomes[:-1]):
        raise ValueError("no option may follow a terminal option")
    value = 0.0
    returns = []
    for outcome in reversed(outcomes):
        value = float(outcome.hit_cost) + (0.0 if outcome.terminal else value)
        returns.append(value)
    returns.reverse()
    return tuple(returns)


def assert_hit_conservation(
    outcomes: tuple[OptionOutcome, ...],
    *,
    pre_option_hits: int,
    manifest_hits: int,
) -> None:
    if pre_option_hits < 0 or manifest_hits < 0:
        raise ValueError("HIT counts cannot be negative")
    accounted = pre_option_hits + sum(outcome.hit_cost for outcome in outcomes)
    if accounted != manifest_hits:
        raise ValueError(
            f"HIT conservation failed: {accounted} != {manifest_hits}"
        )
    returns = undiscounted_hit_returns(outcomes)
    if not math.isclose(
        returns[0] + pre_option_hits,
        manifest_hits,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("return target and HIT ledger disagree")
