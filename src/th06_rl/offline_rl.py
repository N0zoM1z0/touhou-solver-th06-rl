"""Sequential offline-RL views over immutable physical TH06 trajectories.

The contextual-bandit trainer deliberately reconstructs the online trace
credit.  This module instead exposes conservative one-step MDP tuples: a
physical HIT is attached once to the latest eligible action, and bootstrapping
never crosses an observation/control gap, a source-context boundary, or a HIT.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterable

from .offline import ACTION_SET, RunDescriptor
from .offline_learning import (
    HIT_CREDIT_PENALTY,
    LabeledTransition,
    _candidate_features,
    _frame,
    _immediate_reward,
    _mapping,
)


@dataclass(frozen=True)
class SequentialTransition:
    state: LabeledTransition
    reward: float
    next_state: LabeledTransition | None
    terminal_reason: str | None
    elapsed_frames: int

    @property
    def terminal(self) -> bool:
        return self.next_state is None


def _trainable(
    row: dict[str, object],
    run: RunDescriptor,
    *,
    exact_context_only: bool,
) -> tuple[str, tuple[str, ...], float] | None:
    outcome = _mapping(row.get("outcome_terms"))
    policy = _mapping(row.get("policy_context"))
    legal_raw = row.get("legal_actions")
    legal = tuple(str(item) for item in legal_raw) if isinstance(legal_raw, list) else ()
    action_raw = row.get("published_action")
    proposal_raw = row.get("proposed_action")
    action = str(action_raw) if action_raw is not None else None
    proposal = str(proposal_raw) if proposal_raw is not None else None
    try:
        propensity = float(row.get("behavior_probability", 0.0))
    except (TypeError, ValueError):
        propensity = 0.0
    if not (
        run.training_eligible
        and row.get("learning_eligible") is True
        and action in ACTION_SET
        and action in legal
        and proposal == action
        and not outcome.get("bomb_used")
        and 0.0 < propensity <= 1.0
        and (policy or not exact_context_only)
    ):
        return None
    assert action is not None
    return action, legal, propensity


def sequential_transitions(
    raw_rows: Iterable[dict[str, object]],
    run: RunDescriptor,
    *,
    exact_context_only: bool,
) -> list[SequentialTransition]:
    """Build gap-safe one-step tuples from one complete physical Stage."""
    rows = list(raw_rows)
    states: list[tuple[int, LabeledTransition]] = []
    phase: str | None = None
    phase_start_frame = 0
    previous_action = "stay"
    for index, row in enumerate(rows):
        sequence = int(row.get("sequence", -1))
        if sequence != index:
            raise ValueError(f"sequential view requires dense sequence: {sequence} != {index}")
        frame = _frame(row.get("snapshot_ref"), sequence)
        scope = _mapping(row.get("scope"))
        source_context = str(scope.get("phase_id", "unknown"))
        if source_context != phase:
            phase = source_context
            phase_start_frame = frame
        selected = _trainable(row, run, exact_context_only=exact_context_only)
        if selected is not None:
            action, legal, propensity = selected
            outcome = _mapping(row.get("outcome_terms"))
            states.append((index, LabeledTransition(
                run_id=run.run_id,
                sequence=sequence,
                frame=frame,
                source_context=source_context,
                action=action,
                baseline_action=str(row.get("baseline_action") or "unknown"),
                legal_actions=legal,
                behavior_probability=propensity,
                features=_candidate_features(
                    row,
                    phase_elapsed=frame - phase_start_frame,
                    previous_action=previous_action,
                    action=action,
                ),
                reward=_immediate_reward(outcome),
            )))
            previous_action = action
        elif row.get("published_action") is not None:
            previous_action = str(row["published_action"])

    transitions: list[SequentialTransition] = []
    for position, (row_index, state) in enumerate(states):
        row = rows[row_index]
        next_pair = states[position + 1] if position + 1 < len(states) else None
        next_index = next_pair[0] if next_pair is not None else len(rows)
        intervening = rows[row_index:next_index]
        hit = any(
            _mapping(item.get("outcome_terms")).get("life_lost") is True
            for item in intervening
        )
        reward = state.reward - (HIT_CREDIT_PENALTY if hit else 0.0)
        terminal_reason: str | None = None
        next_state: LabeledTransition | None = None
        if hit:
            terminal_reason = "physical-hit"
        elif next_pair is None:
            terminal_reason = "stage-end"
        else:
            candidate_index, candidate = next_pair
            same_context = candidate.source_context == state.source_context
            coherent_root = (
                row.get("next_snapshot_ref")
                == rows[candidate_index].get("snapshot_ref")
            )
            adjacent = candidate_index == row_index + 1
            if not same_context:
                terminal_reason = "source-context-boundary"
            elif not adjacent or not coherent_root:
                terminal_reason = "observation-or-control-gap"
            else:
                next_state = candidate
        elapsed = max(1, int(_mapping(row.get("outcome_terms")).get("elapsed_frames", 1)))
        transitions.append(SequentialTransition(
            state=replace(
                state,
                reward=reward,
                hit_within_30=hit,
                hit_within_60=hit,
                hit_within_120=hit,
            ),
            reward=reward,
            next_state=next_state,
            terminal_reason=terminal_reason,
            elapsed_frames=elapsed,
        ))
    return transitions


def summarize_sequential(transitions: Iterable[SequentialTransition]) -> dict[str, object]:
    rows = list(transitions)
    reasons = Counter(
        row.terminal_reason or "nonterminal"
        for row in rows
    )
    return {
        "rows": len(rows),
        "nonterminal_rows": reasons["nonterminal"],
        "nonterminal_ratio": reasons["nonterminal"] / len(rows) if rows else 0.0,
        "terminal_reasons": {
            name: count for name, count in sorted(reasons.items()) if name != "nonterminal"
        },
        "physical_hit_terminals": reasons["physical-hit"],
        "unique_source_contexts": len({row.state.source_context for row in rows}),
        "all_next_actions_native_legal": all(
            row.next_state is None
            or row.next_state.action in row.next_state.legal_actions
            for row in rows
        ),
    }
