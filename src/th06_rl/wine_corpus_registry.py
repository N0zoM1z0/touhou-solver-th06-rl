"""Capability-indexed registry for immutable original-Wine factual corpora."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


REGISTRY_SCHEMA = "immutable-wine-corpus-registry-v1"
_CLEAN_FIELDS = (
    "background_reactivations",
    "capture_failures",
    "corpus_failures",
    "infrastructure_failures",
    "trace_failures",
)


@dataclass(frozen=True)
class WineCorpusEntry:
    source: str
    access: str
    capabilities: frozenset[str]
    transition_schema: str
    stage: int
    run_id: str
    path: Path
    manifest_sha256: str
    run_sha256: str
    physical_hits: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _clean_complete(manifest: dict[str, object]) -> bool:
    outcome = manifest.get("run_outcome")
    return bool(
        isinstance(outcome, dict)
        and manifest.get("complete") is True
        and manifest.get("stage_trajectory_complete") is True
        and int(manifest.get("dropped_records", -1)) == 0
        and outcome.get("stage_completed") is True
        and outcome.get("corpus_failure") is None
        and all(int(outcome.get(field, -1)) == 0 for field in _CLEAN_FIELDS)
    )


def build_wine_corpus_source(
    *,
    repository: Path,
    root: Path,
    source_id: str,
    access: str,
    capabilities: tuple[str, ...],
    transition_schema: str,
    executable_sha256: str,
) -> dict[str, object]:
    """Build the exact immutable registry row for a finished source.

    Every manifest under ``root`` must be a clean complete run. This prevents
    outcome- or failure-conditioned omission while constructing a new
    autonomous collection inventory.
    """
    repository = repository.resolve()
    root = root.resolve()
    if (
        not source_id
        or access != "training"
        or not capabilities
        or len(set(capabilities)) != len(capabilities)
        or not transition_schema
        or len(executable_sha256) != 64
        or not root.is_dir()
    ):
        raise ValueError("new Wine corpus source contract is invalid")
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise ValueError("new Wine corpus source has no manifests")
    inventory = hashlib.sha256()
    count = 0
    for manifest_path in manifests:
        manifest = _object(manifest_path)
        if not _clean_complete(manifest):
            raise ValueError(
                f"new Wine corpus source contains a rejected run: {manifest_path.parent}"
            )
        run_path = manifest_path.parent / "run.json"
        run = _object(run_path)
        schemas = run.get("schemas")
        metadata = run.get("metadata")
        outcome = manifest.get("run_outcome")
        if (
            not isinstance(schemas, dict)
            or schemas.get("transition") != transition_schema
            or not isinstance(metadata, dict)
            or metadata.get("executable_sha256") != executable_sha256
            or not isinstance(metadata.get("stage"), int)
            or not isinstance(outcome, dict)
            or not isinstance(outcome.get("physical_hits"), int)
        ):
            raise ValueError(
                f"new Wine corpus source semantics drifted: {manifest_path.parent}"
            )
        relative = manifest_path.parent.resolve().relative_to(repository)
        inventory.update(str(relative).encode())
        inventory.update(b"\0")
        inventory.update(_sha256(manifest_path).encode())
        inventory.update(b"\0")
        inventory.update(_sha256(run_path).encode())
        inventory.update(b"\n")
        count += 1
    return {
        "access": access,
        "capabilities": list(capabilities),
        "expected_clean_complete_runs": count,
        "id": source_id,
        "inventory_sha256": inventory.hexdigest(),
        "root": str(root.relative_to(repository)),
        "transition_schema": transition_schema,
    }


def load_wine_corpus_registry(
    registry_path: Path, *, repository: Path
) -> tuple[dict[str, object], tuple[WineCorpusEntry, ...]]:
    repository = repository.resolve()
    registry = _object(registry_path.resolve())
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unsupported Wine corpus registry schema")
    executable_sha256 = str(registry.get("original_retail_executable_sha256", ""))
    sources = registry.get("sources")
    if len(executable_sha256) != 64 or not isinstance(sources, list) or not sources:
        raise ValueError("Wine corpus registry header is invalid")
    entries: list[WineCorpusEntry] = []
    seen_sources: set[str] = set()
    seen_runs: set[Path] = set()
    for raw_source in sources:
        if not isinstance(raw_source, dict):
            raise TypeError("Wine corpus registry source is not an object")
        source = str(raw_source.get("id", ""))
        access = str(raw_source.get("access", ""))
        transition = str(raw_source.get("transition_schema", ""))
        capabilities = frozenset(map(str, raw_source.get("capabilities", ())))
        root = (repository / str(raw_source.get("root", ""))).resolve()
        expected_count = raw_source.get("expected_clean_complete_runs")
        expected_inventory = str(raw_source.get("inventory_sha256", ""))
        if (
            not source
            or source in seen_sources
            or access not in {"training", "infrastructure-regression-only"}
            or not transition
            or not capabilities
            or not root.is_dir()
            or not isinstance(expected_count, int)
            or expected_count < 1
            or len(expected_inventory) != 64
        ):
            raise ValueError(f"invalid Wine corpus source: {source!r}")
        seen_sources.add(source)
        source_rows: list[WineCorpusEntry] = []
        inventory = hashlib.sha256()
        for manifest_path in sorted(root.rglob("manifest.json")):
            manifest = _object(manifest_path)
            if not _clean_complete(manifest):
                continue
            run_path = manifest_path.parent / "run.json"
            run = _object(run_path)
            schemas = run.get("schemas")
            metadata = run.get("metadata")
            outcome = manifest["run_outcome"]
            if (
                not isinstance(schemas, dict)
                or schemas.get("transition") != transition
                or not isinstance(metadata, dict)
                or metadata.get("executable_sha256") != executable_sha256
                or not isinstance(metadata.get("stage"), int)
                or not isinstance(outcome.get("physical_hits"), int)
            ):
                raise ValueError(f"Wine corpus semantics drifted: {manifest_path.parent}")
            path = manifest_path.parent.resolve()
            if path in seen_runs:
                raise ValueError(f"Wine corpus appears in multiple sources: {path}")
            seen_runs.add(path)
            manifest_sha256 = _sha256(manifest_path)
            run_sha256 = _sha256(run_path)
            relative = path.relative_to(repository)
            inventory.update(str(relative).encode())
            inventory.update(b"\0")
            inventory.update(manifest_sha256.encode())
            inventory.update(b"\0")
            inventory.update(run_sha256.encode())
            inventory.update(b"\n")
            source_rows.append(WineCorpusEntry(
                source=source,
                access=access,
                capabilities=capabilities,
                transition_schema=transition,
                stage=int(metadata["stage"]),
                run_id=str(run.get("run_id", path.name)),
                path=path,
                manifest_sha256=manifest_sha256,
                run_sha256=run_sha256,
                physical_hits=int(outcome["physical_hits"]),
            ))
        if len(source_rows) != expected_count:
            raise ValueError(f"Wine corpus source count drifted: {source}")
        if inventory.hexdigest() != expected_inventory:
            raise ValueError(f"Wine corpus source inventory drifted: {source}")
        entries.extend(source_rows)
    return registry, tuple(entries)


def select_wine_corpora(
    entries: tuple[WineCorpusEntry, ...],
    *,
    required_capabilities: frozenset[str],
    access: str = "training",
) -> tuple[WineCorpusEntry, ...]:
    if not required_capabilities:
        raise ValueError("corpus selection needs an explicit semantic capability")
    return tuple(
        entry for entry in entries
        if entry.access == access
        and required_capabilities <= entry.capabilities
    )
