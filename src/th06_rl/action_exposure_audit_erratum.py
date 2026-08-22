"""Bounded L2k correction for a declared control-dead-end interruption."""

from __future__ import annotations

import copy
from pathlib import Path

from .action_exposure_audit import (
    audit_episode as audit_episode_v1,
    summarize_action_exposure_audits,
)
from .episode_dataset import iter_decision_epochs, iter_episode_transitions


def _factual_control_dead_end_interruptions(
    run_dir: Path,
    *,
    exposure_roots: int,
) -> set[str]:
    """Return only v1 labels explained by a recorded dead-end transition."""
    epochs = list(iter_decision_epochs(run_dir))
    transitions = {
        item.sequence: item for item in iter_episode_transitions(run_dir)
    }
    groups: dict[int, list[object]] = {}
    for epoch in epochs:
        exposure = epoch.action_exposure
        if exposure is not None:
            groups.setdefault(exposure.group_id, []).append(epoch)

    explained: set[str] = set()
    for group_id, group in groups.items():
        steps = [epoch.action_exposure.step for epoch in group]
        if steps == list(range(exposure_roots)):
            continue
        last = group[-1]
        if (
            steps == list(range(len(steps)))
            and any(
                bool(transitions[sequence].outcome.get("control_dead_end"))
                for sequence in last.transition_sequences
            )
        ):
            explained.add(f"group-{group_id}:unexplained-interruption")
    return explained


def audit_episode(run_dir: Path, *, exposure_roots: int) -> dict[str, object]:
    """Apply only the L2k control-dead-end erratum to the frozen v1 audit."""
    result = copy.deepcopy(audit_episode_v1(run_dir, exposure_roots=exposure_roots))
    count = int(result["contract_violation_count"])
    sample = list(result["contract_violation_sample"])
    if count != len(sample):
        raise ValueError("bounded erratum refuses a truncated violation list")
    explained = _factual_control_dead_end_interruptions(
        run_dir,
        exposure_roots=exposure_roots,
    )
    corrected = [item for item in sample if item not in explained]
    result["contract_violation_count"] = len(corrected)
    result["contract_violation_sample"] = corrected
    return result
