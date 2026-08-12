from scripts.run_generation6_stage6_pilot import pilot_verdict


def _rows(incumbent, candidate, *, exercised=3, passed=True):
    rows = []
    for index, (left, right) in enumerate(zip(incumbent, candidate, strict=True)):
        rows.extend((
            {"role": "incumbent", "physical_hits": left, "passed": passed,
             "interventions": 0},
            {"role": "candidate", "physical_hits": right, "passed": passed,
             "interventions": int(index < exercised)},
        ))
    return rows


def test_pilot_verdict_requires_strict_aggregate_reduction() -> None:
    result = pilot_verdict(
        _rows([10, 8, 9], [8, 7, 8]), expected_runs=6,
        required_exercised_candidate_stages=2,
    )
    assert result["verdict"] == "effective-pilot-signal"
    assert result["effect_hits"] == 4

    tied = pilot_verdict(
        _rows([10, 8, 9], [9, 9, 9]), expected_runs=6,
        required_exercised_candidate_stages=2,
    )
    assert tied["verdict"] == "no-effective-pilot-signal"


def test_pilot_verdict_separates_exposure_and_runtime_failures() -> None:
    unexercised = pilot_verdict(
        _rows([10, 8, 9], [8, 7, 8], exercised=1), expected_runs=6,
        required_exercised_candidate_stages=2,
    )
    assert unexercised["verdict"] == "inconclusive"

    invalid = pilot_verdict(
        _rows([10, 8, 9], [8, 7, 8], passed=False), expected_runs=6,
        required_exercised_candidate_stages=2,
    )
    assert invalid["verdict"] == "invalid"
