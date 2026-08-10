from __future__ import annotations

from th06_rl.offline import RunDescriptor
from th06_rl.offline_rl import sequential_transitions, summarize_sequential


def _run() -> RunDescriptor:
    return RunDescriptor(
        run_id="run",
        remote_path="runs/run",
        scope=(3, 0, 0, 6),
        transition_schema="th06-rl-transition-v5",
        transitions=3,
        storage_complete=True,
        stage_complete=True,
        training_eligible=True,
        code_commit="abc",
        native_kernel_sha256=None,
        physical_hits=1,
        manifest_sha256="0" * 64,
        run_sha256="0" * 64,
    )


def _row(sequence: int, *, eligible: bool = True, phase: str = "phase:a", hit: bool = False):
    action = "left" if eligible else None
    return {
        "schema_version": "th06-rl-transition-v5",
        "sequence": sequence,
        "snapshot_ref": f"run:{sequence:08d}:f{100 + sequence}",
        "next_snapshot_ref": f"run:{sequence + 1:08d}:f{101 + sequence}",
        "scope": {"phase_id": phase},
        "legal_actions": ["stay", "left"],
        "baseline_action": "stay",
        "proposed_action": action,
        "published_action": action,
        "behavior_probability": 0.5,
        "learning_eligible": eligible,
        "policy_context": {
            "current_action": "stay",
            "player_x": 192,
            "player_y": 384,
            "hard_admissible_actions": ["stay", "left"],
        },
        "outcome_terms": {
            "elapsed_frames": 1,
            "life_lost": hit,
            "bomb_used": False,
        },
    }


def test_sequence_bootstraps_only_across_adjacent_coherent_roots() -> None:
    rows = [_row(0), _row(1), _row(2, eligible=False, hit=True)]

    transitions = sequential_transitions(rows, _run(), exact_context_only=True)

    assert len(transitions) == 2
    assert transitions[0].terminal is False
    assert transitions[0].next_state.sequence == transitions[1].state.sequence
    assert transitions[1].terminal_reason == "physical-hit"
    assert transitions[1].reward == -99.0
    assert summarize_sequential(transitions)["physical_hit_terminals"] == 1


def test_sequence_closes_context_boundaries_and_control_gaps() -> None:
    context_rows = [_row(0), _row(1, phase="phase:b")]
    gap_rows = [_row(0), _row(1, eligible=False), _row(2)]

    context = sequential_transitions(context_rows, _run(), exact_context_only=True)
    gap = sequential_transitions(gap_rows, _run(), exact_context_only=True)

    assert context[0].terminal_reason == "source-context-boundary"
    assert gap[0].terminal_reason == "observation-or-control-gap"
    assert all(row.terminal for row in gap)
