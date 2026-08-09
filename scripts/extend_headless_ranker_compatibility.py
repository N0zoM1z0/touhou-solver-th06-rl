#!/usr/bin/env python3
"""Copy a ranker across one audited additive headless observation upgrade."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from th06_rl.native import ACTIONS


ALLOWED_NEWBORN_LASER_SOURCE_PATHS = frozenset({
    "HEADLESS.md",
    "src/HeadlessRuntime.cpp",
    "src/HeadlessRuntime.hpp",
})
ALLOWED_EVENT_SOURCE_PATHS = frozenset({
    "HEADLESS.md",
    "src/BulletManager.cpp",
    "src/EnemyManager.cpp",
    "src/GameWindow.cpp",
    "src/HeadlessRuntime.cpp",
    "src/HeadlessRuntime.hpp",
    "src/Player.cpp",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(source: Mapping[str, Any]) -> tuple[str, str, bool]:
    return (
        str(source.get("commit", "")),
        str(source.get("binary_sha256", "")),
        source.get("clean") is True,
    )


def compatible_sources(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = artifact.get("compatible_headless_sources")
    if not isinstance(raw, list):
        raw = [artifact.get("headless_source")]
    return [dict(source) for source in raw if isinstance(source, Mapping)]


def validate_upgrade_evidence(
    evidence: Mapping[str, Any],
    *,
    old_sources: Iterable[Mapping[str, Any]],
    changed_paths: Iterable[str],
) -> None:
    if evidence.get("schema") == "th06-rl-headless-event-observation-differential-v1":
        if evidence.get("authority") != "additive-diagnostics-only-no-native-set-revision":
            raise ValueError("event evidence does not preserve native-set authority")
        if evidence.get("removed_observation_members") != ["events"]:
            raise ValueError("event evidence removed more than the additive event member")
        if evidence.get("physical_observations_equal") is not True:
            raise ValueError("event upgrade changed a physical observation")
        expected_actions = {action.name for action in ACTIONS}
        if set(evidence.get("actions", ())) != expected_actions:
            raise ValueError("event evidence does not replay every ordinary action")
        branches = evidence.get("branches")
        if not isinstance(branches, list) or len(branches) != len(expected_actions):
            raise ValueError("event evidence has an incomplete branch set")
        if {str(branch.get("action")) for branch in branches} != expected_actions:
            raise ValueError("event evidence has duplicate or missing action branches")
        if any(
            branch.get("physical_observations_equal") is not True
            or branch.get("mismatch_offsets") != []
            or int(branch.get("old_observation_count", 0)) <= 0
            or branch.get("old_observation_count") != branch.get("new_observation_count")
            or branch.get("old_physical_sha256") != branch.get("new_physical_sha256")
            for branch in branches
        ):
            raise ValueError("event evidence contains a physical branch mismatch")
        if int(evidence.get("eventful_observation_count", 0)) <= 0:
            raise ValueError("event evidence did not exercise the diagnostic extension")
        if "enemy" not in evidence.get("hit_kinds", ()):
            raise ValueError("event evidence did not retain the audited enemy collision")
        old_keys = {_source_key(source) for source in old_sources}
        if _source_key(evidence.get("old_runtime_source", {})) not in old_keys:
            raise ValueError("event evidence does not use the model's old runtime")
        if set(changed_paths) != ALLOWED_EVENT_SOURCE_PATHS:
            raise ValueError("runtime upgrade differs from the audited event-only source patch")
        return
    if evidence.get("schema") != "th06-rl-headless-authority-failure-differential-v1":
        raise ValueError("compatibility evidence has an unsupported schema")
    if evidence.get("classification") != "source-safe-but-native-observation-incomplete":
        raise ValueError("compatibility evidence did not isolate an observation gap")
    if evidence.get("native_comparison_available") is not False:
        raise ValueError("compatibility evidence did not fail at observation authority")
    if "angular history" not in str(evidence.get("native_authority_error", "")):
        raise ValueError("compatibility evidence is not the newborn-laser gap")
    expected_actions = {action.name for action in ACTIONS}
    if set(evidence.get("source_safe_constant_actions", ())) != expected_actions:
        raise ValueError("compatibility evidence does not source-test every ordinary action")
    old_keys = {_source_key(source) for source in old_sources}
    if _source_key(evidence.get("runtime_source", {})) not in old_keys:
        raise ValueError("compatibility evidence does not use the model's old runtime")
    changed = set(changed_paths)
    if not changed or not changed.issubset(ALLOWED_NEWBORN_LASER_SOURCE_PATHS):
        raise ValueError("runtime upgrade changes files outside the additive laser ABI")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--runtime-binary", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import joblib

    model_path = args.model.resolve()
    binary = args.runtime_binary.resolve()
    evidence_path = args.evidence.resolve()
    source_root = binary.parent
    artifact = joblib.load(model_path)
    old_sources = compatible_sources(artifact)
    if not old_sources:
        parser.error("model has no exact old runtime source")
    new_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    new_source = {
        "commit": new_commit,
        "binary_sha256": _sha256(binary),
        "clean": not dirty,
    }
    if not new_source["clean"]:
        parser.error("new headless runtime source is dirty")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("schema") == "th06-rl-headless-event-observation-differential-v1":
        if _source_key(evidence.get("new_runtime_source", {})) != _source_key(new_source):
            parser.error("event evidence does not use the requested new runtime")
        evidence_source = evidence.get("old_runtime_source", {})
    else:
        evidence_source = evidence.get("runtime_source", {})
    old_commit = str(evidence_source.get("commit", ""))
    ancestor = subprocess.run(
        ["git", "-C", str(source_root), "merge-base", "--is-ancestor", old_commit, new_commit],
        check=False,
    ).returncode == 0
    if not ancestor:
        parser.error("new runtime is not a descendant of the audited old runtime")
    changed_paths = subprocess.run(
        ["git", "-C", str(source_root), "diff", "--name-only", old_commit, new_commit],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    try:
        validate_upgrade_evidence(
            evidence,
            old_sources=old_sources,
            changed_paths=changed_paths,
        )
    except ValueError as error:
        parser.error(str(error))
    if _source_key(new_source) in {_source_key(source) for source in old_sources}:
        parser.error("model already declares the requested runtime")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("compatibility output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    artifact["compatible_headless_sources"] = [*old_sources, new_source]
    output_model = output / model_path.name
    joblib.dump(artifact, output_model, compress=3)
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": "th06-rl-headless-ranker-compatibility-extension-v1",
        "authority": "additive-observation-upgrade-only",
        "code_commit": code_commit,
        "input_model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "output_model": {"path": output_model.name, "sha256": _sha256(output_model)},
        "old_compatible_sources": old_sources,
        "new_compatible_source": new_source,
        "changed_runtime_paths": changed_paths,
        "evidence": {"path": str(evidence_path), "sha256": _sha256(evidence_path)},
        "promotion_allowed": False,
        "promotion_blocker": "new runtime still requires unseen-seed full-stage rollout",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "compatibility-report.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
