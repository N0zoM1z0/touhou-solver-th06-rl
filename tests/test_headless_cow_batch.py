from scripts.batch_label_headless_cow import (
    _output_path,
    checkpoint_groups,
    checkpoint_sequences,
    event_checkpoint_sequences,
    round_robin_task_groups,
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


def test_checkpoint_groups_preserve_order_and_default_to_one_replay():
    sequences = (1, 41, 81, 121, 161)
    assert checkpoint_groups(sequences, checkpoints_per_task=0) == (sequences,)
    assert checkpoint_groups(sequences, checkpoints_per_task=2) == (
        (1, 41),
        (81, 121),
        (161,),
    )
    assert checkpoint_groups((), checkpoints_per_task=2) == ()


def test_round_robin_task_groups_balances_early_run_coverage() -> None:
    assert round_robin_task_groups((("a1", "a2", "a3"), ("b1", "b2"))) == (
        "a1",
        "b1",
        "a2",
        "b2",
        "a3",
    )
    assert round_robin_task_groups(()) == ()


def test_output_path_disambiguates_identical_run_basenames(tmp_path) -> None:
    manifest = {"scope": {"stage": 3}, "initial_seed": 73}
    first = tmp_path / "model-a" / "same-run"
    second = tmp_path / "model-b" / "same-run"

    first_output = _output_path(
        tmp_path,
        first,
        manifest,
        disambiguate_run=True,
    )
    second_output = _output_path(
        tmp_path,
        second,
        manifest,
        disambiguate_run=True,
    )

    assert first_output != second_output
    assert first_output == _output_path(
        tmp_path,
        first,
        manifest,
        disambiguate_run=True,
    )


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
    )


def test_event_sequences_keep_pre_hit_forced_release_but_ignore_later_hits():
    rows = [_row() for _ in range(20)]
    rows[5] = _row(forced=True, legal=False)
    rows[10] = _row(hit=True)
    rows[15] = _row(hit=True)

    assert event_checkpoint_sequences(rows, event_window=2, stride=2) == (
        2,
        4,
        8,
        10,
    )


def test_event_sequences_ignore_runs_without_hit_or_forced_release():
    assert event_checkpoint_sequences(
        [_row() for _ in range(20)], event_window=4, stride=2
    ) == ()


def test_event_sequences_include_fail_close_terminal_neighborhood():
    rows = [_row() for _ in range(20)]

    assert event_checkpoint_sequences(
        rows,
        event_window=4,
        stride=2,
        termination_reason="authority-failure",
    ) == (15, 17, 19)


def test_event_sequences_back_up_from_unlabelable_terminal_row():
    rows = [_row() for _ in range(20)]
    rows[19] = _row(legal=False)

    assert event_checkpoint_sequences(
        rows,
        event_window=4,
        stride=2,
        termination_reason="physical-hit",
    ) == (14, 16, 18)
