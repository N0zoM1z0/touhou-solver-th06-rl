#!/usr/bin/env python3
"""Build an immutable calibration-free ensemble of native-action rankers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compatible_sources(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = artifact.get("compatible_headless_sources")
    if not isinstance(raw, list):
        raw = [artifact.get("headless_source")]
    return [dict(source) for source in raw if isinstance(source, Mapping)]


def _source_key(source: Mapping[str, Any]) -> tuple[str, str, bool]:
    return (
        str(source.get("commit", "")),
        str(source.get("binary_sha256", "")),
        source.get("clean") is True,
    )


def validate_output_directory(path: Path) -> None:
    """Reject the common mistake of passing the artifact filename itself."""
    if path.name.endswith(".joblib"):
        raise ValueError(
            "--output is a directory; do not append ensemble-ranker.joblib"
        )


def delivery_contract(artifact: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    name = str(artifact.get("native_delivery_contract", "legacy-unspecified-v0"))
    raw_delays = artifact.get("native_delivery_delays", [])
    delays = tuple(int(value) for value in raw_delays) if isinstance(raw_delays, list) else ()
    if name == "synchronous-step-v1":
        if delays != (0,):
            raise ValueError("synchronous ensemble member delivery must be exactly [0]")
    elif name != "legacy-unspecified-v0":
        raise ValueError(f"unsupported ensemble delivery contract {name}")
    return name, delays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.error("an ensemble requires at least two model artifacts")
    try:
        validate_output_directory(args.output)
    except ValueError as error:
        parser.error(str(error))

    import joblib

    loaded = []
    scope = None
    common_sources: dict[tuple[str, str, bool], dict[str, Any]] | None = None
    common_delivery: tuple[str, tuple[int, ...]] | None = None
    member_report = []
    for path in args.models:
        resolved = path.resolve()
        artifact = joblib.load(resolved)
        if artifact.get("ensemble_members") is not None:
            parser.error("nested ensembles are not supported")
        if scope is None:
            scope = artifact.get("scope")
        elif artifact.get("scope") != scope:
            parser.error("ensemble members silently mix scopes")
        try:
            current_delivery = delivery_contract(artifact)
        except ValueError as error:
            parser.error(str(error))
        if common_delivery is None:
            common_delivery = current_delivery
        elif current_delivery != common_delivery:
            parser.error("ensemble members silently mix delivery contracts")
        sources = {_source_key(source): source for source in _compatible_sources(artifact)}
        common_sources = sources if common_sources is None else {
            key: common_sources[key] for key in common_sources.keys() & sources.keys()
        }
        loaded.append({
            "model": artifact["model"],
            "feature_names": artifact.get("feature_names"),
            "categories": artifact["categories"],
        })
        member_report.append({"path": str(resolved), "sha256": _sha256(resolved)})
    if scope is None or not common_sources:
        parser.error("ensemble members have no exact compatible runtime in common")
    compatible = [source for key, source in sorted(common_sources.items()) if key[2]]
    if not compatible:
        parser.error("ensemble members have no clean compatible runtime in common")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "ensemble_members": loaded,
        "scope": scope,
        "headless_source": compatible[0],
        "compatible_headless_sources": compatible,
        "ensemble_contract": "borda-native-safe-action-consensus-v1",
        "native_delivery_contract": common_delivery[0],
        "native_delivery_delays": list(common_delivery[1]),
    }
    model_path = output / "ensemble-ranker.joblib"
    joblib.dump(artifact, model_path, compress=3)
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": "th06-rl-headless-ranker-ensemble-v1",
        "algorithm": "borda-native-safe-action-consensus",
        "authority": "rank-native-legal-set-only",
        "scope": scope,
        "headless_source": compatible[0],
        "compatible_headless_sources": compatible,
        "members": member_report,
        "member_count": len(member_report),
        "native_delivery_contract": common_delivery[0],
        "native_delivery_delays": list(common_delivery[1]),
        "code_commit": code_commit,
        "promotion_allowed": False,
        "promotion_blocker": (
            "ensemble requires unseen-seed full-stage continuation and Windows validation"
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "report.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
