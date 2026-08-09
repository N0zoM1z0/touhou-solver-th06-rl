from scripts.batch_label_headless_cow import (
    checkpoint_sequences,
    event_checkpoint_sequences,
)


def test_checkpoint_sequences_include_terminal_neighborhood_and_final_row():
    assert checkpoint_sequences(1000, tail_transitions=240, stride=80) == (
        760,
        840,
        920,
        999,
    )


def test_checkpoint_sequences_bound_short_runs_and_reject_unusable_runs():
    assert checkpoint_sequences(42, tail_transitions=600, stride=80) == (1, 41)
    assert checkpoint_sequences(1, tail_transitions=600, stride=80) == ()


def _row(*, hit: bool = False, forced: bool = False, legal: bool = True):
    return {
        "legal_actions": ["stay_fast"] if legal else [],
        "benchmark_forced_action": forced,
        "outcome_terms": {"deaths_delta": int(hit)},
    }


def test_event_sequences_target_hit_and_first_forced_release_neighborhoods():
    rows = [_row() for _ in range(20)]
    rows[10] = _row(hit=True)
    rows[15] = _row(forced=True, legal=False)
    rows[16] = _row(forced=True, legal=False)

    assert event_checkpoint_sequences(rows, event_window=4, stride=2) == (
        6,
        8,
        10,
        12,
        14,
    )


def test_event_sequences_ignore_runs_without_hit_or_forced_release():
    assert event_checkpoint_sequences(
        [_row() for _ in range(20)], event_window=4, stride=2
    ) == ()
