#!/usr/bin/env python3
"""Bind the pinned Windows embeddable Python to this checkout portably."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


def configured_pth(runtime: Path, repository: Path, archive_name: str) -> str:
    relative_repository = os.path.relpath(repository.resolve(), runtime.resolve())
    windows_repository = relative_repository.replace("/", "\\")
    return "\n".join(
        (
            archive_name,
            ".",
            windows_repository,
            windows_repository + "\\src",
            "import site",
            "",
        )
    )


def configure(runtime: Path, repository: Path) -> Path:
    candidates = sorted(runtime.glob("python*._pth"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one embeddable Python _pth file, found {len(candidates)}"
        )
    path = candidates[0]
    archive_name = path.name.removesuffix("._pth") + ".zip"
    if not (runtime / archive_name).is_file():
        raise ValueError(f"embeddable standard-library archive is absent: {archive_name}")
    rendered = configured_pth(runtime, repository, archive_name)
    if path.read_text(encoding="utf-8") == rendered:
        return path
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=runtime
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--repository", type=Path, default=repository)
    args = parser.parse_args(argv)
    path = configure(args.runtime.resolve(), args.repository.resolve())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
