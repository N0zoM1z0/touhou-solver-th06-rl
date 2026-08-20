from __future__ import annotations

import pytest

from scripts.verify_wine_smoke import verify


def _report() -> dict[str, object]:
    return {
        "artifact_dir": "artifacts/smoke",
        "wine_version": "wine-test",
        "error": None,
        "controller_returncode": 0,
        "gdb_normalized": True,
        "retail_sha256": "same",
        "expected_retail_sha256": "same",
        "immutable_policy_state_equal": True,
        "leftover_prefix_processes": [],
        "trace": {
            "rows": 12,
            "first_frame": 40,
            "last_frame": 50,
            "decisions": 11,
            "max_bullets": 3,
            "last_policy_metrics": {"purpose": "infrastructure-smoke-only"},
        },
    }


def test_smoke_verifier_requires_real_controlled_gameplay() -> None:
    result = verify(_report())

    assert result["passed"] is True
    assert result["decisions"] == 11
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("controller_returncode", 1),
        ("gdb_normalized", False),
        ("immutable_policy_state_equal", False),
        ("leftover_prefix_processes", [{"pid": 1}]),
    ],
)
def test_smoke_verifier_fails_closed_on_runtime_contracts(
    field: str, value: object,
) -> None:
    report = _report()
    report[field] = value

    with pytest.raises(ValueError, match="failed"):
        verify(report)


def test_smoke_verifier_rejects_an_empty_trace() -> None:
    report = _report()
    report["trace"] = {"rows": 0, "decisions": 0}

    with pytest.raises(ValueError, match="coherent_gameplay_trace"):
        verify(report)
