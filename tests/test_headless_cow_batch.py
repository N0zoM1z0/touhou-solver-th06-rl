from scripts.batch_label_headless_cow import checkpoint_sequences


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
