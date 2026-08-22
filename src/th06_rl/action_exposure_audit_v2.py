"""Audit fixed-horizon randomized action-intention episodes and HIT support."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Iterable

from .actions import ACTION_NAMES
from .episode_dataset import iter_decision_epochs, iter_episode_transitions
from .policies.fixed_shield_action_exposure import OVERRIDE_REASON, POLICY_NAME


AUDIT_SCHEMA = "th06-rl-action-exposure-audit-v2"
TARGET_SCHEMA = "th06-rl-action-exposure-hit-target-audit-v1"


def audit_episode(run_dir: Path, *, exposure_roots: int) -> dict[str, object]:
    epochs = list(iter_decision_epochs(run_dir))
    transitions = list(iter_episode_transitions(run_dir))
    transitions_by_sequence = {item.sequence: item for item in transitions}
    violations: list[str] = []
    groups: dict[int, list[object]] = {}
    assignment_counts: Counter[str] = Counter()
    prior_group = -1

    for epoch in epochs:
        exposure = epoch.action_exposure
        if epoch.policy_id != POLICY_NAME:
            violations.append(f"epoch-{epoch.index}:policy-id")
        if exposure is None:
            violations.append(f"epoch-{epoch.index}:missing-exposure")
            continue
        if exposure.horizon != exposure_roots:
            violations.append(f"epoch-{epoch.index}:horizon")
        if exposure.group_id < prior_group:
            violations.append(f"epoch-{epoch.index}:group-order")
        if exposure.group_id > prior_group + 1:
            violations.append(f"epoch-{epoch.index}:group-gap")
        if exposure.group_id != prior_group:
            if exposure.step != 0:
                violations.append(f"epoch-{epoch.index}:group-without-step-zero")
            prior_group = exposure.group_id
        groups.setdefault(exposure.group_id, []).append(epoch)

        assignment = dict(exposure.assignment_probabilities)
        legal = tuple(epoch.observation.locally_admissible_actions)
        behavior = dict(epoch.behavior_probabilities)
        intended_is_legal = exposure.intended_action in legal
        if exposure.step == 0:
            assignment_counts[exposure.intended_action] += 1
            expected = 1.0 / len(legal)
            if (
                set(assignment) != set(legal)
                or any(
                    not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
                    for value in assignment.values()
                )
                or epoch.published_action != exposure.intended_action
                or behavior != assignment
                or exposure.override_reason is not None
            ):
                violations.append(f"epoch-{epoch.index}:assignment-rule")
        else:
            expected_action = (
                exposure.intended_action if intended_is_legal else epoch.baseline_action
            )
            expected_override = None if intended_is_legal else OVERRIDE_REASON
            expected_behavior = {
                action: float(action == expected_action) for action in legal
            }
            if (
                epoch.published_action != expected_action
                or exposure.override_reason != expected_override
                or behavior != expected_behavior
            ):
                violations.append(f"epoch-{epoch.index}:continuation-rule")

    complete_groups = 0
    eligible_complete_groups = 0
    no_override_groups = 0
    full_execution_groups = 0
    interrupted_groups = 0
    for group_id, group in groups.items():
        exposures = [epoch.action_exposure for epoch in group]
        if any(exposure is None for exposure in exposures):
            continue
        first = exposures[0]
        assert first is not None
        steps = [exposure.step for exposure in exposures if exposure is not None]
        if any(
            exposure is not None
            and (
                exposure.group_id != group_id
                or exposure.horizon != first.horizon
                or exposure.intended_action != first.intended_action
                or exposure.assignment_probability != first.assignment_probability
                or exposure.assignment_probabilities != first.assignment_probabilities
            )
            for exposure in exposures
        ):
            violations.append(f"group-{group_id}:assignment-mutated")
        complete = steps == list(range(exposure_roots))
        if not complete:
            interrupted_groups += 1
            last = group[-1]
            factual_dead_end = any(
                bool(transitions_by_sequence[sequence].outcome.get("control_dead_end"))
                for sequence in last.transition_sequences
            )
            if not (
                steps == list(range(len(steps)))
                and (
                    last.hit_cost > 0
                    or not last.learning_eligible
                    or last.terminal
                    or factual_dead_end
                )
            ):
                violations.append(f"group-{group_id}:unexplained-interruption")
            continue
        complete_groups += 1
        if not all(epoch.learning_eligible for epoch in group):
            continue
        eligible_complete_groups += 1
        if all(
            exposure is not None and exposure.override_reason is None
            for exposure in exposures
        ):
            no_override_groups += 1
        executed = [
            action
            for epoch in group
            for action in epoch.executed_actions
            if action is not None
        ]
        if sum(action == first.intended_action for action in executed) >= exposure_roots:
            full_execution_groups += 1

    control_dead_ends = sum(
        bool(item.outcome.get("control_dead_end")) for item in transitions
    )
    bombs = sum(bool(item.outcome.get("bomb_used")) for item in transitions)
    infrastructure_failures = sum(
        bool(item.outcome.get("infrastructure_failed")) for item in transitions
    )
    denominator = len(transitions)
    return {
        "episode_id": epochs[0].episode_id,
        "transitions": denominator,
        "policy_invocations": len(epochs),
        "groups": len(groups),
        "complete_groups": complete_groups,
        "eligible_complete_groups": eligible_complete_groups,
        "interrupted_groups": interrupted_groups,
        "no_override_groups": no_override_groups,
        "full_intended_executions_groups": full_execution_groups,
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "control_dead_ends": control_dead_ends,
        "control_dead_end_rate": control_dead_ends / denominator,
        "bombs": bombs,
        "infrastructure_failures": infrastructure_failures,
        "contract_violation_count": len(violations),
        "contract_violation_sample": violations[:20],
    }


def audit_hit_target_episode(
    run_dir: Path,
    *,
    exposure_roots: int,
) -> dict[str, object]:
    starts = []
    for epoch in iter_decision_epochs(run_dir):
        exposure = epoch.action_exposure
        if exposure is not None and exposure.step == 0:
            starts.append((
                exposure.group_id,
                epoch.start_sequence,
                exposure.intended_action,
            ))
    needed = {
        sequence + offset
        for _group, sequence, _action in starts
        for offset in range(exposure_roots)
    }
    transitions = {}
    for item in iter_episode_transitions(run_dir):
        if item.sequence not in needed:
            continue
        exposure = item.action_exposure
        transitions[item.sequence] = {
            "elapsed": int(item.outcome.get("elapsed_frames", -1)),
            "hit": bool(item.outcome.get("life_lost")),
            "dead_end": bool(item.outcome.get("control_dead_end")),
            "executed": item.executed_action,
            "group": None if exposure is None else exposure.group_id,
            "step": None if exposure is None else exposure.step,
            "override": None if exposure is None else exposure.override_reason,
        }

    status: Counter[str] = Counter()
    accepted_actions: Counter[str] = Counter()
    positive_actions: Counter[str] = Counter()
    hit_offsets: Counter[int] = Counter()
    for group_id, start, intended in starts:
        window = [transitions.get(start + offset) for offset in range(exposure_roots)]
        if any(item is None for item in window):
            status["unsupported-end"] += 1
            continue
        assert all(item is not None for item in window)
        if any(int(item["elapsed"]) != 1 for item in window if item is not None):
            status["censored-observation-gap"] += 1
            continue
        hits = [
            offset for offset, item in enumerate(window)
            if item is not None and bool(item["hit"])
        ]
        first_hit = hits[0] if hits else None
        outcome_length = exposure_roots if first_hit is None else first_hit + 1
        observed = window[:outcome_length]
        first = observed[0]
        assert first is not None
        if first["executed"] != intended:
            status["censored-assignment-not-executed"] += 1
            continue
        if any(
            item is not None
            and item["group"] is not None
            and item["group"] != group_id
            for item in observed
        ):
            status["censored-next-assignment"] += 1
            continue
        if first_hit is None and [
            item["step"] if item is not None else None for item in window
        ] != list(range(exposure_roots)):
            status["censored-incomplete-protocol"] += 1
            continue
        label = int(first_hit is not None)
        status[f"accepted-label-{label}"] += 1
        accepted_actions[intended] += 1
        if label:
            positive_actions[intended] += 1
            hit_offsets[first_hit] += 1
            if any(
                bool(item["dead_end"]) for item in observed if item is not None
            ):
                status["positive-after-control-dead-end"] += 1
        elif any(
            item["override"] is not None for item in window if item is not None
        ):
            status["accepted-negative-with-shield-override"] += 1
    return {
        "schema": TARGET_SCHEMA,
        "episode_id": run_dir.name,
        "exposure_roots": exposure_roots,
        "group_starts": len(starts),
        "status": dict(sorted(status.items())),
        "accepted_actions": dict(sorted(accepted_actions.items())),
        "positive_actions": dict(sorted(positive_actions.items())),
        "hit_offsets": {str(key): value for key, value in sorted(hit_offsets.items())},
    }


def summarize_audits(
    episode_audits: Iterable[dict[str, object]],
    target_audits: Iterable[dict[str, object]],
    *,
    exposure_roots: int,
    minimum_complete_groups_per_episode: int,
    minimum_assignments_per_action: int,
    minimum_no_override_fraction: float,
    minimum_full_execution_fraction: float,
    maximum_control_dead_end_rate: float,
    hit_support_diagnostic_minimum: int,
) -> dict[str, object]:
    episodes = list(episode_audits)
    targets = list(target_audits)
    assignments: Counter[str] = Counter()
    accepted_actions: Counter[str] = Counter()
    positive_actions: Counter[str] = Counter()
    target_status: Counter[str] = Counter()
    for episode in episodes:
        assignments.update(episode["assignment_counts"])
    for target in targets:
        accepted_actions.update(target["accepted_actions"])
        positive_actions.update(target["positive_actions"])
        target_status.update(target["status"])
    eligible = sum(int(row["eligible_complete_groups"]) for row in episodes)
    no_override = sum(int(row["no_override_groups"]) for row in episodes)
    full_execution = sum(
        int(row["full_intended_executions_groups"]) for row in episodes
    )
    transitions = sum(int(row["transitions"]) for row in episodes)
    dead_ends = sum(int(row["control_dead_ends"]) for row in episodes)
    no_override_fraction = no_override / eligible if eligible else 0.0
    full_execution_fraction = full_execution / eligible if eligible else 0.0
    dead_end_rate = dead_ends / transitions if transitions else 1.0
    positives = int(target_status["accepted-label-1"])
    gates = {
        "exact_exposure_contract": all(
            int(row["contract_violation_count"]) == 0 for row in episodes
        ),
        "complete_group_support": all(
            int(row["eligible_complete_groups"])
            >= minimum_complete_groups_per_episode
            for row in episodes
        ),
        "all_action_assignment_support": all(
            assignments[action] >= minimum_assignments_per_action
            for action in ACTION_NAMES
        ),
        "no_override_fraction": no_override_fraction >= minimum_no_override_fraction,
        "full_intended_executions_fraction": (
            full_execution_fraction >= minimum_full_execution_fraction
        ),
        "control_dead_end_rate": dead_end_rate <= maximum_control_dead_end_rate,
        "zero_bomb_and_infrastructure_failure": all(
            int(row["bombs"]) == 0 and int(row["infrastructure_failures"]) == 0
            for row in episodes
        ),
    }
    contract_passed = all(gates.values())
    support_ready = positives >= hit_support_diagnostic_minimum
    decision = (
        "proceed-serial-action-exposure-training-collection"
        if contract_passed and support_ready
        else "pass-action-exposure-contract-insufficient-hit-support"
        if contract_passed
        else "reject-action-exposure-collection-contract"
    )
    return {
        "schema": AUDIT_SCHEMA,
        "policy_id": POLICY_NAME,
        "exposure_roots": exposure_roots,
        "episodes": episodes,
        "target_episodes": targets,
        "aggregate": {
            "eligible_complete_groups": eligible,
            "assignment_counts": dict(sorted(assignments.items())),
            "no_override_fraction": no_override_fraction,
            "full_intended_executions_fraction": full_execution_fraction,
            "control_dead_end_rate": dead_end_rate,
            "target_status": dict(sorted(target_status.items())),
            "target_accepted_actions": dict(sorted(accepted_actions.items())),
            "target_positive_actions": dict(sorted(positive_actions.items())),
            "hit_support_diagnostic_minimum": hit_support_diagnostic_minimum,
            "hit_support_ready": support_ready,
        },
        "gates": gates,
        "decision": decision,
    }
