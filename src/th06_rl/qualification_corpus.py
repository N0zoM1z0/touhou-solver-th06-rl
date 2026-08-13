"""Immutable complete-episode partitions for offline learner qualification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


QUALIFICATION_SCHEMA = "autonomous-offline-learner-qualification-v1"


@dataclass(frozen=True)
class QualificationRun:
    role: str
    source: str
    stage: int
    chronological_index: int
    path: Path
    manifest_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _portable_run_path(raw: str, repository: Path) -> Path:
    path = Path(raw)
    if path.is_dir():
        return path.resolve()
    parts = path.parts
    if "artifacts" in parts:
        relocated = repository.joinpath(*parts[parts.index("artifacts"):])
        if relocated.is_dir():
            return relocated.resolve()
    raise FileNotFoundError(f"qualification Wine corpus is absent: {raw}")


def load_qualification_partition(
    contract_path: Path, *, repository: Path
) -> tuple[dict[str, object], tuple[QualificationRun, ...]]:
    contract_path = contract_path.resolve()
    repository = repository.resolve()
    contract = _object(contract_path)
    if contract.get("schema") != QUALIFICATION_SCHEMA:
        raise ValueError("unsupported learner qualification schema")
    if contract.get("evidence_eligible") is not False:
        raise ValueError("historical qualification must be evidence-ineligible")
    sources = contract.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("qualification sources are absent")

    result: list[QualificationRun] = []
    seen_paths: set[Path] = set()
    for raw_source in sources:
        if not isinstance(raw_source, dict):
            raise TypeError("qualification source is not an object")
        name = str(raw_source.get("name", ""))
        report_path = repository / str(raw_source.get("report", ""))
        expected_sha256 = str(raw_source.get("report_sha256", ""))
        stage = raw_source.get("stage")
        development_prefix = raw_source.get("development_prefix")
        qualification_suffix = raw_source.get("qualification_suffix")
        if (
            not name
            or not report_path.is_file()
            or _sha256(report_path) != expected_sha256
            or not isinstance(stage, int)
            or not isinstance(development_prefix, int)
            or not isinstance(qualification_suffix, int)
            or development_prefix < 1
            or qualification_suffix < 1
        ):
            raise ValueError(f"invalid or drifting qualification source: {name!r}")
        report = _object(report_path)
        if report.get("schema") != raw_source.get("report_schema"):
            raise ValueError(f"qualification report schema drifted: {name}")
        rows = report.get("runs")
        if (
            not isinstance(rows, list)
            or len(rows) != development_prefix + qualification_suffix
        ):
            raise ValueError(f"qualification source count drifted: {name}")
        for index, raw_run in enumerate(rows):
            if not isinstance(raw_run, dict):
                raise TypeError(f"qualification run is not an object: {name}")
            run = _portable_run_path(str(raw_run.get("path", "")), repository)
            manifest_sha256 = str(raw_run.get("manifest_sha256", ""))
            if _sha256(run / "manifest.json") != manifest_sha256:
                raise ValueError(f"qualification run manifest drifted: {run}")
            if run in seen_paths:
                raise ValueError(f"duplicate qualification run: {run}")
            seen_paths.add(run)
            role = "development" if index < development_prefix else "qualification"
            result.append(QualificationRun(
                role=role,
                source=name,
                stage=stage,
                chronological_index=index,
                path=run,
                manifest_sha256=manifest_sha256,
            ))

    expected = contract.get("expected")
    development = sum(run.role == "development" for run in result)
    qualification = sum(run.role == "qualification" for run in result)
    if expected != {
        "development_episode_groups": development,
        "qualification_episode_groups": qualification,
        "total_episode_groups": len(result),
    }:
        raise ValueError("qualification partition totals drifted")
    return contract, tuple(result)
