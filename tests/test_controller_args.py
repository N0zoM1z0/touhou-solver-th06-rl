import pytest

from th06_rl.th06.controller import (
    RouteTrial,
    _control_dead_end,
    _advance_route_scope,
    _valid_executable_basename,
    parse_args as _parse_args,
)


def parse_args(args: list[str]):
    return _parse_args([
        *args,
        "--policy-plugin", "candidate.py",
        "--policy-state", "candidate.json",
        "--immutable-policy",
    ])


def test_in_flight_source_unsafe_is_a_control_dead_end_not_infra_loss() -> None:
    assert _control_dead_end("control-dead-end:in-flight input unsafe")
    assert _control_dead_end("authority-stop:in-flight input unsafe")


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
        parse_args([
            "--practice-stage", "6", "--exploration-rate", "0.01"
        ])


def test_controller_rejects_capture_gap_resume_without_source_authority() -> None:
    with pytest.raises(SystemExit):
        parse_args([
            "--practice-stage",
            "6",
            "--resume-after-incoherent-capture",
        ])


def test_controller_accepts_source_u16_diagnostic_rng_seed() -> None:
    args = parse_args([
        "--armed",
        "--practice-stage",
        "6",
        "--diagnostic-rng-seed",
        "0x1234",
    ])
    assert args.diagnostic_rng_seed == 0x1234


def test_controller_requires_a_positive_corpus_storage_bound() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--armed", "--start-route", "--max-corpus-gib", "0"])
    args = parse_args(["--armed", "--start-route", "--max-corpus-gib", "4"])
    assert args.max_corpus_gib == 4.0
