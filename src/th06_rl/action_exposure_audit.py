"""Audit randomized multi-root action intentions from factual Wine episodes."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Iterable

from .actions import ACTION_NAMES
from .episode_dataset import iter_decision_epochs, iter_episode_transitions
from .policies.fixed_shield_action_exposure import OVERRIDE_REASON, POLICY_NAME


AUDIT_SCHEMA = "th06-rl-action-exposure-audit-v1"


def audit_episode(run_dir: Path, *, exposure_roots: int) -> dict[str, object]:
    epochs = list(iter_decision_epochs(run_dir))
    transitions = list(iter_episode_transitions(run_dir))
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
                or any(not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
                       for value in assignment.values())
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
    four_executed_groups = 0
    interrupted_groups = 0
    group_starts: list[int] = []
    for group_id, group in groups.items():
        exposures = [epoch.action_exposure for epoch in group]
        if any(exposure is None for exposure in exposures):
            continue
        assert all(exposure is not None for exposure in exposures)
        steps = [exposure.step for exposure in exposures if exposure is not None]
        first = exposures[0]
        assert first is not None
        if steps and steps[0] == 0:
            group_starts.append(group[0].start_sequence)
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
        if complete:
            complete_groups += 1
        else:
            interrupted_groups += 1
            last = group[-1]
            if not (
                steps == list(range(len(steps)))
                and (last.hit_cost > 0 or not last.learning_eligible or last.terminal)
            ):
                violations.append(f"group-{group_id}:unexplained-interruption")
            continue
        eligible = all(epoch.learning_eligible for epoch in group)
        if not eligible:
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
            four_executed_groups += 1

    h16_positive = 0
    h16_negative = 0
    h16_unsupported = 0
    for start in group_starts:
        if start + 16 > len(transitions):
            h16_unsupported += 1
            continue
        window = transitions[start : start + 16]
        if any(
            transition.sequence != start + offset
            or int(transition.outcome.get("elapsed_frames", -1)) != 1
            for offset, transition in enumerate(window)
        ):
            h16_unsupported += 1
            continue
        positive = any(bool(item.outcome.get("life_lost")) for item in window)
        h16_positive += int(positive)
        h16_negative += int(not positive)

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
        "four_intended_executions_groups": four_executed_groups,
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "h16_group_starts": {
            "positive": h16_positive,
            "negative": h16_negative,
            "unsupported": h16_unsupported,
        },
        "control_dead_ends": control_dead_ends,
        "control_dead_end_rate": control_dead_ends / denominator,
        "bombs": bombs,
        "infrastructure_failures": infrastructure_failures,
        "contract_violation_count": len(violations),
        "contract_violation_sample": violations[:20],
    }


def audit_action_exposure_runs(
    run_dirs: Iterable[Path],
    *,
    exposure_roots: int,
    minimum_complete_groups_per_episode: int,
    minimum_assignments_per_action: int,
    minimum_no_override_fraction: float,
    minimum_four_execution_fraction: float,
    maximum_control_dead_end_rate: float,
    h16_support_diagnostic_minimum: int,
) -> dict[str, object]:
    episodes = [
        audit_episode(Path(run_dir), exposure_roots=exposure_roots)
        for run_dir in run_dirs
    ]
    return summarize_action_exposure_audits(
        episodes,
        exposure_roots=exposure_roots,
        minimum_complete_groups_per_episode=minimum_complete_groups_per_episode,
        minimum_assignments_per_action=minimum_assignments_per_action,
        minimum_no_override_fraction=minimum_no_override_fraction,
        minimum_four_execution_fraction=minimum_four_execution_fraction,
        maximum_control_dead_end_rate=maximum_control_dead_end_rate,
        h16_support_diagnostic_minimum=h16_support_diagnostic_minimum,
    )


def summarize_action_exposure_audits(
    episodes: list[dict[str, object]],
    *,
    exposure_roots: int,
    minimum_complete_groups_per_episode: int,
    minimum_assignments_per_action: int,
    minimum_no_override_fraction: float,
    minimum_four_execution_fraction: float,
    maximum_control_dead_end_rate: float,
    h16_support_diagnostic_minimum: int,
) -> dict[str, object]:
    assignments: Counter[str] = Counter()
    for episode in episodes:
        assignments.update(episode["assignment_counts"])
    eligible = sum(int(row["eligible_complete_groups"]) for row in episodes)
    no_override = sum(int(row["no_override_groups"]) for row in episodes)
    four_executed = sum(
        int(row["four_intended_executions_groups"]) for row in episodes
    )
    transitions_weight = sum(int(row["transitions"]) for row in episodes)
    control_dead_ends = sum(int(row["control_dead_ends"]) for row in episodes)
    h16_positive = sum(
        int(row["h16_group_starts"]["positive"]) for row in episodes
    )
    exact_contract = all(int(row["contract_violation_count"]) == 0 for row in episodes)
    group_support = all(
        int(row["eligible_complete_groups"]) >= minimum_complete_groups_per_episode
        for row in episodes
    )
    action_support = all(
        assignments[action] >= minimum_assignments_per_action for action in ACTION_NAMES
    )
    no_override_fraction = no_override / eligible if eligible else 0.0
    four_execution_fraction = four_executed / eligible if eligible else 0.0
    control_dead_end_rate = control_dead_ends / transitions_weight if transitions_weight else 1.0
    gates = {
        "exact_exposure_contract": exact_contract,
        "complete_group_support": group_support,
        "all_action_assignment_support": action_support,
        "no_override_fraction": no_override_fraction >= minimum_no_override_fraction,
        "four_intended_executions_fraction": (
            four_execution_fraction >= minimum_four_execution_fraction
        ),
        "control_dead_end_rate": control_dead_end_rate <= maximum_control_dead_end_rate,
        "zero_bomb_and_infrastructure_failure": all(
            int(row["bombs"]) == 0 and int(row["infrastructure_failures"]) == 0
            for row in episodes
        ),
    }
    contract_passed = all(gates.values())
    support_ready = h16_positive >= h16_support_diagnostic_minimum
    decision = (
        "proceed-action-exposure-training-collection"
        if contract_passed and support_ready
        else "pass-action-exposure-contract-insufficient-h16-support"
        if contract_passed
        else "reject-action-exposure-collection-contract"
    )
    return {
        "schema": AUDIT_SCHEMA,
        "policy_id": POLICY_NAME,
        "exposure_roots": exposure_roots,
        "episodes": episodes,
        "aggregate": {
            "eligible_complete_groups": eligible,
            "assignment_counts": dict(sorted(assignments.items())),
            "no_override_fraction": no_override_fraction,
            "four_intended_executions_fraction": four_execution_fraction,
            "control_dead_end_rate": control_dead_end_rate,
            "h16_positive_group_starts": h16_positive,
            "h16_support_diagnostic_minimum": h16_support_diagnostic_minimum,
            "h16_support_ready": support_ready,
        },
        "gates": gates,
        "decision": decision,
    }
