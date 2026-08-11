import pytest

from th06_rl.th06.controller import (
    RouteTrial,
    _advance_route_scope,
    _valid_executable_basename,
    parse_args,
)


def test_retail_executable_name_accepts_ascii_and_original_japanese_names() -> None:
    assert _valid_executable_basename("th06.exe")
    assert _valid_executable_basename("東方紅魔郷.exe")


def test_retail_executable_name_rejects_paths_and_non_executables() -> None:
    assert not _valid_executable_basename("")
    assert not _valid_executable_basename("../th06.exe")
    assert not _valid_executable_basename(r"folder\th06.exe")
    assert not _valid_executable_basename("th06.com")


def test_route_trial_completes_only_after_gameplay_reaches_ending() -> None:
    trial = RouteTrial()
    assert not trial.observe_supervisor(10)
    assert not trial.observe_supervisor(2)
    assert not trial.observe_supervisor(3)
    assert trial.observe_supervisor(10)


def test_route_scope_accepts_only_next_stage_in_same_scope() -> None:
    assert _advance_route_scope((3, 0, 0, 1), (3, 0, 0, 2)) == (3, 0, 0, 2)
    with pytest.raises(Exception, match="route scope changed unexpectedly"):
        _advance_route_scope((3, 0, 0, 2), (3, 0, 0, 4))
    with pytest.raises(Exception, match="route scope changed unexpectedly"):
        _advance_route_scope((3, 0, 0, 2), (2, 0, 0, 3))


def test_controller_rejects_route_and_practice_together() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--start-route", "--practice-stage", "1"])


def test_controller_accepts_immutable_policy_evaluation() -> None:
    args = parse_args([
        "--practice-stage",
        "6",
        "--immutable-policy",
        "--exploration-rate",
        "0",
    ])
    assert args.immutable_policy


def test_controller_rejects_exploration_for_immutable_policy() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--practice-stage", "6", "--immutable-policy"])


def test_controller_accepts_fail_closed_capture_gap_resume() -> None:
    args = parse_args([
        "--practice-stage",
        "6",
        "--resume-after-incoherent-capture",
    ])
    assert args.resume_after_incoherent_capture


def test_controller_accepts_source_u16_diagnostic_rng_seed() -> None:
    args = parse_args([
        "--armed",
        "--practice-stage",
        "6",
        "--diagnostic-rng-seed",
        "0x1234",
    ])
    assert args.diagnostic_rng_seed == 0x1234
