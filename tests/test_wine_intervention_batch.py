import pytest

from scripts.run_wine_intervention_batch import _arm_order, _parse_seeds


def test_batch_alternates_temporal_arm_order() -> None:
    assert _arm_order(0) == ("incumbent", "alternative")
    assert _arm_order(1) == ("alternative", "incumbent")


def test_batch_seed_parser_accepts_source_notation() -> None:
    assert _parse_seeds("0x1234, 9029") == (0x1234, 9029)
    with pytest.raises(Exception):
        _parse_seeds("1,1")
    with pytest.raises(Exception):
        _parse_seeds("0x10000")
