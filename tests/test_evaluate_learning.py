import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "evaluate_learning",
    Path(__file__).parents[1] / "scripts" / "evaluate_learning.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_blind_complete_stage_is_not_comparison_eligible():
    assert _MODULE._comparison_exclusions({
        "stage_trajectory_complete": True,
        "capture_failures": 10917,
        "infrastructure_failures": 10917,
    }) == ["capture-failures", "infrastructure-failures"]


def test_fully_observed_complete_stage_is_comparison_eligible():
    assert _MODULE._comparison_exclusions({
        "stage_trajectory_complete": True,
        "capture_failures": 0,
        "infrastructure_failures": 0,
    }) == []

