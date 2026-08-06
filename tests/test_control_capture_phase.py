from th06_rl.th06.control_capture import _completed_calc_lag


def test_active_calc_phase_uses_initial_equal_clock_witness():
    assert _completed_calc_lag(
        None,
        None,
        stage=5,
        game_frame=120,
        bullet_time=120,
        passive=False,
    ) == (5, 0, True)


def test_time_stop_rebases_completed_calc_lag():
    assert _completed_calc_lag(
        5,
        0,
        stage=5,
        game_frame=420,
        bullet_time=120,
        passive=True,
    ) == (5, 300, True)
    assert _completed_calc_lag(
        5,
        300,
        stage=5,
        game_frame=421,
        bullet_time=121,
        passive=False,
    ) == (5, 300, True)


def test_priority_window_is_one_frame_beyond_completed_lag():
    assert _completed_calc_lag(
        5,
        300,
        stage=5,
        game_frame=421,
        bullet_time=120,
        passive=False,
    ) == (5, 300, False)


def test_unobserved_nonzero_midstage_lag_fails_closed():
    assert _completed_calc_lag(
        None,
        None,
        stage=5,
        game_frame=420,
        bullet_time=120,
        passive=False,
    ) == (5, None, False)


def test_stage_change_does_not_reuse_prior_lag():
    assert _completed_calc_lag(
        4,
        300,
        stage=5,
        game_frame=1,
        bullet_time=1,
        passive=False,
    ) == (5, 0, True)
