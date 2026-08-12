#!/usr/bin/env python3
"""Rebuild or verify the ignored, source-pinned offline-RL reference cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY / "config" / "offline_rl_references.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"reference path escapes cache root: {relative}")
    return target


def _paper(root: Path, row: dict[str, str], *, verify_only: bool) -> None:
    target = _safe_target(root, row["path"])
    expected = row["sha256"]
    if target.is_file() and _sha256(target) == expected:
        print(f"verified paper {row['path']} {expected}")
        return
    if target.exists():
        raise RuntimeError(f"refusing to overwrite mismatched reference: {target}")
    if verify_only:
        raise FileNotFoundError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        row["url"], headers={"User-Agent": "th06-rl-reference-cache/1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
    actual = _sha256(temporary)
    if actual != expected:
        temporary.unlink()
        raise RuntimeError(
            f"download digest mismatch for {row['path']}: {actual} != {expected}"
        )
    temporary.replace(target)
    print(f"downloaded paper {row['path']} {expected}")


def _git(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _repository(root: Path, row: dict[str, str], *, verify_only: bool) -> None:
    target = _safe_target(root, row["path"])
    if not target.exists():
        if verify_only:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--no-checkout", row["url"], str(target))
        _git("fetch", "--depth=1", "origin", row["commit"], cwd=target)
        _git("checkout", "--detach", row["commit"], cwd=target)
    if not (target / ".git").is_dir():
        raise RuntimeError(f"reference is not a Git checkout: {target}")
    remote = _git("remote", "get-url", "origin", cwd=target)
    head = _git("rev-parse", "HEAD", cwd=target)
    if remote != row["url"] or head != row["commit"]:
        raise RuntimeError(
            f"reference checkout mismatch for {row['path']}: {remote} @ {head}"
        )
    print(f"verified repository {row['path']} {head} ({row['license']})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    root = _safe_target(REPOSITORY, manifest["root"])
    for row in manifest["papers"]:
        _paper(root, row, verify_only=args.verify_only)
    for row in manifest["repositories"]:
        _repository(root, row, verify_only=args.verify_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
