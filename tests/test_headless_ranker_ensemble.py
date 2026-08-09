from pathlib import Path

import pytest

from scripts.build_headless_ranker_ensemble import delivery_contract, validate_output_directory


def test_ensemble_output_is_a_directory_not_the_joblib_filename() -> None:
    validate_output_directory(Path("artifacts/models/stage3-ensemble"))

    with pytest.raises(ValueError, match="output is a directory"):
        validate_output_directory(Path("artifacts/models/ensemble-ranker.joblib"))


def test_ensemble_delivery_contract_is_explicit() -> None:
    assert delivery_contract({
        "native_delivery_contract": "synchronous-step-v1",
        "native_delivery_delays": [0],
    }) == ("synchronous-step-v1", (0,))
    assert delivery_contract({}) == ("legacy-unspecified-v0", ())

    with pytest.raises(ValueError, match="exactly"):
        delivery_contract({
            "native_delivery_contract": "synchronous-step-v1",
            "native_delivery_delays": [0, 1],
        })
