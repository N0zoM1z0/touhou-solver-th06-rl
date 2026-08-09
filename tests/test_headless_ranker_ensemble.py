from pathlib import Path

import pytest

from scripts.build_headless_ranker_ensemble import validate_output_directory


def test_ensemble_output_is_a_directory_not_the_joblib_filename() -> None:
    validate_output_directory(Path("artifacts/models/stage3-ensemble"))

    with pytest.raises(ValueError, match="output is a directory"):
        validate_output_directory(Path("artifacts/models/ensemble-ranker.joblib"))
