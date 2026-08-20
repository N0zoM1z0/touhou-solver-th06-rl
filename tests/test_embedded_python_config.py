from __future__ import annotations

from pathlib import Path

import pytest

from scripts.configure_embedded_python import configure, configured_pth


def test_embedded_python_paths_are_relative_to_the_runtime(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    runtime = repository / "reference/tools/python"
    runtime.mkdir(parents=True)

    rendered = configured_pth(runtime, repository, "python311.zip")

    assert str(repository) not in rendered
    assert rendered.splitlines() == [
        "python311.zip",
        ".",
        "..\\..\\..",
        "..\\..\\..\\src",
        "import site",
    ]


def test_configure_embedded_python_is_idempotent(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    runtime = repository / "reference/tools/python"
    runtime.mkdir(parents=True)
    (runtime / "python311.zip").write_bytes(b"standard-library")
    path = runtime / "python311._pth"
    path.write_text("python311.zip\n.\n", encoding="utf-8")

    assert configure(runtime, repository) == path
    first = path.read_bytes()
    assert configure(runtime, repository) == path
    assert path.read_bytes() == first


def test_configure_embedded_python_rejects_ambiguous_runtime(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected one"):
        configure(tmp_path, tmp_path)
