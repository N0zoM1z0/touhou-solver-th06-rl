#!/usr/bin/env python3
"""Incrementally mirror immutable TH06-RL runs and committed policies to HF."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone
import zlib


DATASET_SCHEMA = "th06-rl-hf-dataset-v1"
CHECKPOINT_SCHEMA = "th06-rl-hf-checkpoint-snapshot-v1"
DEFAULT_REPO_ID = "Joh1rreq/touhou-solver-th06-rl-corpus"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def validate_run(run_dir: Path, *, verify_content: bool = False) -> dict[str, object]:
    """Validate one immutable recorder run and return its portable index row."""
    manifest_path = run_dir / "manifest.json"
    run_path = run_dir / "run.json"
    if not manifest_path.is_file() or not run_path.is_file():
        raise FileNotFoundError(f"run metadata is incomplete: {run_dir}")
    manifest = _json(manifest_path)
    run = _json(run_path)
    run_id = str(manifest.get("run_id", ""))
    if not run_id or run_id != run_dir.name or run.get("run_id") != run_id:
        raise ValueError(f"run identity mismatch: {run_dir}")
    if manifest.get("complete") is not True:
        raise ValueError(f"run is still open: {run_id}")
    metadata = run.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"run metadata is invalid: {run_id}")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise TypeError(f"run shards are invalid: {run_id}")
    seen: set[str] = set()
    shard_bytes = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise TypeError(f"invalid shard entry: {run_id}")
        relative = str(shard.get("path", ""))
        path = Path(relative)
        if not relative or path.name != relative or relative in seen:
            raise ValueError(f"unsafe or duplicate shard path in {run_id}: {relative}")
        seen.add(relative)
        local = run_dir / relative
        if not local.is_file() or local.is_symlink():
            raise FileNotFoundError(f"missing immutable shard: {local}")
        expected_bytes = int(shard.get("compressed_bytes", -1))
        if expected_bytes < 0 or local.stat().st_size != expected_bytes:
            raise ValueError(f"shard size mismatch: {local}")
        expected_sha = str(shard.get("sha256", ""))
        named_digest = relative.rsplit("-", 1)[-1].split(".", 1)[0]
        if len(expected_sha) != 64 or named_digest != expected_sha[:16]:
            raise ValueError(f"shard digest/name mismatch: {local}")
        if verify_content and _sha256(local) != expected_sha:
            raise ValueError(f"shard digest mismatch: {local}")
        shard_bytes += expected_bytes
    if shard_bytes != int(manifest.get("compressed_bytes", -1)):
        raise ValueError(f"manifest compressed byte mismatch: {run_id}")
    outcome = manifest.get("run_outcome")
    outcome = outcome if isinstance(outcome, dict) else {}
    episode = manifest.get("episode")
    episode = episode if isinstance(episode, dict) else {}
    stage_complete = bool(
        episode.get("complete") is True
        or manifest.get("stage_trajectory_complete") is True
        or outcome.get("stage_completed") is True
    )
    infra_failures = outcome.get("infrastructure_failures")
    training_eligible = bool(
        stage_complete
        and infra_failures == 0
        and int(manifest.get("dropped_records", 0)) == 0
        and not outcome.get("corpus_failure")
    )
    return {
        "run_id": run_id,
        "remote_path": f"runs/{run_id}",
        "manifest_schema": manifest.get("schema_version"),
        "manifest_sha256": _sha256(manifest_path),
        "run_sha256": _sha256(run_path),
        "difficulty": metadata.get("difficulty"),
        "character": metadata.get("character"),
        "shot_type": metadata.get("shot_type"),
        "stage": metadata.get("stage"),
        "code_commit": metadata.get("code_commit"),
        "schemas": run.get("schemas"),
        "records": manifest.get("records"),
        "compressed_bytes": shard_bytes,
        "shards": len(shards),
        "storage_complete": True,
        "stage_trajectory_complete": stage_complete,
        "termination_reason": episode.get("termination_reason")
        or outcome.get("termination_reason"),
        "physical_hits": outcome.get("physical_hits"),
        "infrastructure_failures": infra_failures,
        "training_eligible_complete_stage": training_eligible,
    }


def discover_runs(
    corpus_root: Path, *, verify_content: bool = False
) -> list[tuple[Path, dict[str, object]]]:
    runs: list[tuple[Path, dict[str, object]]] = []
    if not corpus_root.is_dir():
        return runs
    for run_dir in sorted(path for path in corpus_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _json(manifest_path)
        if manifest.get("complete") is not True:
            continue
        runs.append((run_dir, validate_run(run_dir, verify_content=verify_content)))
    return runs


def _decode_policy_summary(path: Path) -> dict[str, object]:
    outer = _json(path)
    state = outer
    if outer.get("schema") == "th06-rl-online-ucb-packed-v1":
        if outer.get("codec") != "zlib-base64-v1":
            raise ValueError(f"unsupported policy codec: {path}")
        payload = outer.get("payload")
        if not isinstance(payload, str):
            raise TypeError(f"invalid packed policy: {path}")
        decoded = zlib.decompress(base64.b64decode(payload, validate=True))
        state = json.loads(decoded)
        if not isinstance(state, dict):
            raise TypeError(f"invalid decoded policy: {path}")
    trials = state.get("trials")
    trials = trials if isinstance(trials, dict) else {}
    middle = state.get("middle_trials")
    middle = middle if isinstance(middle, dict) else {}
    fine = state.get("fine_trials")
    fine = fine if isinstance(fine, dict) else {}
    return {
        "outer_schema": outer.get("schema"),
        "state_schema": state.get("schema"),
        "reward_version": state.get("reward_version"),
        "decisions": state.get("decisions"),
        "exploratory_decisions": state.get("exploratory_decisions"),
        "observed_trials": sum(int(value) for value in trials.values()),
        "coarse_action_states": len(trials),
        "middle_action_states": len(middle),
        "fine_action_states": len(fine),
    }


def stage_committed_policy_files(policy_root: Path) -> list[tuple[str, Path]]:
    """Select last committed states, never a live partial-Stage checkpoint."""
    selected: list[tuple[str, Path]] = []
    if not policy_root.is_dir():
        return selected
    for state_path in sorted(policy_root.glob("*.json")):
        marker = state_path.with_name(f".{state_path.name}.stage-transaction.json")
        backup = state_path.with_name(f".{state_path.name}.stage-start")
        source = backup if marker.is_file() else state_path
        if marker.is_file() and not backup.is_file():
            raise FileNotFoundError(
                f"active policy transaction has no committed backup: {state_path}"
            )
        selected.append((state_path.name, source))
    return selected


def build_checkpoint_snapshot(policy_root: Path, output_dir: Path) -> dict[str, object]:
    policies: dict[str, dict[str, object]] = {}
    fingerprint_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, source in stage_committed_policy_files(policy_root):
        digest = _sha256(source)
        row = {
            "path": name,
            "sha256": digest,
            "bytes": source.stat().st_size,
            "selection": (
                "stage-start-committed" if source.name.startswith(".") else "checkpoint"
            ),
            **_decode_policy_summary(source),
        }
        policies[name] = row
        fingerprint_rows.append((name, digest))
        shutil.copyfile(source, output_dir / name)
    fingerprint = hashlib.sha256(_canonical(fingerprint_rows)).hexdigest()
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "snapshot_id": f"committed-{fingerprint[:16]}",
        "policies": policies,
    }
    (output_dir / "checkpoint_manifest.json").write_bytes(_canonical(manifest) + b"\n")
    return manifest


def _dataset_card() -> str:
    return """---
pretty_name: TH06 Source-Grounded RL Corpus
license: other
tags:
- touhou
- th06
- game-ai
- contextual-bandit
- offline-rl
---

# TH06 source-grounded RL corpus

This private, append-only dataset mirrors physical trajectories produced by
[touhou-solver-th06-rl](https://github.com/N0zoM1z0/touhou-solver-th06-rl).
Difficulty, character, shot type, Stage, and automatically derived source phase
remain explicit learning scopes.

`runs/<run_id>/` preserves recorder-native gzip shards plus their immutable
`run.json`, `manifest.json`, and optional infrastructure audit. The dataset
index labels complete-stage/training eligibility; diagnostic and legacy runs
are retained but must not silently enter that stratum. `checkpoints/` contains
content-addressed snapshots of the last Stage-committed online sufficient
statistics. A checkpoint is deployment state, not a replacement for raw data
when training a new offline model or reward definition.

The corpus contains decoded gameplay state and outcomes, not the game
executable, DAT assets, screenshots, credentials, or process-memory dumps.
Bomb remains forbidden and every deployed action remains subject to the native
safety gate. Observational trajectories are not counterfactual proof; physical
complete-Stage play is the final evaluation.
"""


def _load_remote_inventory(repo_id: str) -> dict[str, object]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="dataset_manifest.json",
            repo_type="dataset",
        )
    except EntryNotFoundError:
        return {"schema": DATASET_SCHEMA, "runs": [], "checkpoints": []}
    value = _json(Path(path))
    if value.get("schema") != DATASET_SCHEMA:
        raise ValueError("remote dataset manifest schema mismatch")
    return value


def _merge_rows(
    prior: list[object], current: list[dict[str, object]], *, key: str, digest: str
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for value in prior:
        if isinstance(value, dict) and isinstance(value.get(key), str):
            merged[str(value[key])] = value
    for row in current:
        identity = str(row[key])
        old = merged.get(identity)
        if old is not None and old.get(digest) != row.get(digest):
            raise ValueError(f"remote identity collision: {identity}")
        merged[identity] = row
    return [merged[name] for name in sorted(merged)]


def verify_remote_runs(api, repo_id: str, runs) -> str:
    """Require every recorder shard to exist remotely with its exact SHA-256."""
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    remote = {sibling.rfilename: sibling for sibling in info.siblings}
    for run_dir, row in runs:
        prefix = str(row["remote_path"])
        for metadata_name in ("run.json", "manifest.json"):
            local = run_dir / metadata_name
            sibling = remote.get(f"{prefix}/{metadata_name}")
            if (
                sibling is None
                or sibling.size != local.stat().st_size
                or sibling.blob_id != _git_blob_sha1(local)
            ):
                raise RuntimeError(f"remote metadata mismatch: {prefix}/{metadata_name}")
        manifest = _json(run_dir / "manifest.json")
        for shard in manifest["shards"]:
            relative = str(shard["path"])
            sibling = remote.get(f"{prefix}/{relative}")
            lfs = getattr(sibling, "lfs", None) if sibling is not None else None
            if (
                sibling is None
                or sibling.size != int(shard["compressed_bytes"])
                or lfs is None
                or lfs.sha256 != str(shard["sha256"])
            ):
                raise RuntimeError(f"remote shard mismatch: {prefix}/{relative}")
        row["remote_verified_revision"] = info.sha
        row["remote_verified_shards"] = len(manifest["shards"])
    return str(info.sha)


def verify_remote_checkpoint(api, repo_id: str, snapshot_dir: Path, snapshot_id: str) -> str:
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    remote = {sibling.rfilename: sibling for sibling in info.siblings}
    for local in snapshot_dir.iterdir():
        if not local.is_file():
            continue
        path = f"checkpoints/{snapshot_id}/{local.name}"
        sibling = remote.get(path)
        lfs = getattr(sibling, "lfs", None) if sibling is not None else None
        exact = (
            lfs.sha256 == _sha256(local)
            if lfs is not None
            else sibling is not None and sibling.blob_id == _git_blob_sha1(local)
        )
        if sibling is None or sibling.size != local.stat().st_size or not exact:
            raise RuntimeError(f"remote checkpoint mismatch: {path}")
    return str(info.sha)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--corpus-root", type=Path, default=Path("artifacts/corpus"))
    parser.add_argument("--policy-root", type=Path, default=Path("artifacts/policy"))
    parser.add_argument("--source-repo", type=Path, default=Path("."))
    parser.add_argument("--verify-content", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs = discover_runs(args.corpus_root, verify_content=args.verify_content)
    if not runs:
        raise RuntimeError("no closed corpus runs found")
    with tempfile.TemporaryDirectory(prefix="th06-rl-hf-") as temporary:
        snapshot_dir = Path(temporary) / "checkpoint"
        checkpoint = build_checkpoint_snapshot(args.policy_root, snapshot_dir)
        print(
            json.dumps(
                {
                    "repo_id": args.repo_id,
                    "closed_runs": len(runs),
                    "compressed_bytes": sum(
                        int(row["compressed_bytes"]) for _, row in runs
                    ),
                    "checkpoint": checkpoint["snapshot_id"],
                    "dry_run": args.dry_run,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.dry_run:
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        remote_files = set(
            api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
        )
        card = Path(temporary) / "README.md"
        card.write_text(_dataset_card(), encoding="utf-8")
        api.upload_file(
            path_or_fileobj=card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Document TH06-RL corpus",
        )
        for index, (run_dir, row) in enumerate(runs, 1):
            prefix = str(row["remote_path"])
            manifest = _json(run_dir / "manifest.json")
            expected_remote = {
                f"{prefix}/run.json",
                f"{prefix}/manifest.json",
                *(f"{prefix}/{shard['path']}" for shard in manifest["shards"]),
            }
            if expected_remote <= remote_files:
                print(f"[{index}/{len(runs)}] present {row['run_id']}", flush=True)
                continue
            print(f"[{index}/{len(runs)}] upload {row['run_id']}", flush=True)
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=run_dir,
                path_in_repo=str(row["remote_path"]),
                ignore_patterns=[".*", "*.partial", ".cache/**"],
                commit_message=f"Add physical run {row['run_id']}",
            )

        snapshot_id = str(checkpoint["snapshot_id"])
        remote_checkpoint = f"checkpoints/{snapshot_id}/checkpoint_manifest.json"
        if remote_checkpoint not in remote_files:
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=snapshot_dir,
                path_in_repo=f"checkpoints/{snapshot_id}",
                commit_message=f"Add committed policy snapshot {snapshot_id}",
            )

        verified_revision = verify_remote_runs(api, args.repo_id, runs)
        checkpoint_verified_revision = verify_remote_checkpoint(
            api, args.repo_id, snapshot_dir, snapshot_id
        )

        prior = _load_remote_inventory(args.repo_id)
        run_rows = _merge_rows(
            list(prior.get("runs", [])),
            [row for _, row in runs],
            key="run_id",
            digest="manifest_sha256",
        )
        checkpoint_row = {
            "snapshot_id": snapshot_id,
            "remote_path": f"checkpoints/{snapshot_id}",
            "manifest_sha256": _sha256(snapshot_dir / "checkpoint_manifest.json"),
            "remote_verified_revision": checkpoint_verified_revision,
            "policies": checkpoint["policies"],
        }
        checkpoint_rows = _merge_rows(
            list(prior.get("checkpoints", [])),
            [checkpoint_row],
            key="snapshot_id",
            digest="manifest_sha256",
        )
        try:
            import subprocess

            source_commit = subprocess.check_output(
                ["git", "-C", str(args.source_repo), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = None
        inventory = {
            "schema": DATASET_SCHEMA,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source_repository": "N0zoM1z0/touhou-solver-th06-rl",
            "source_commit_at_sync": source_commit,
            "runs": run_rows,
            "checkpoints": checkpoint_rows,
            "statistics": {
                "runs": len(run_rows),
                "compressed_bytes": sum(
                    int(row.get("compressed_bytes", 0)) for row in run_rows
                ),
                "complete_stage_runs": sum(
                    row.get("stage_trajectory_complete") is True for row in run_rows
                ),
                "training_eligible_complete_stage_runs": sum(
                    row.get("training_eligible_complete_stage") is True
                    for row in run_rows
                ),
            },
        }
        inventory_path = Path(temporary) / "dataset_manifest.json"
        inventory_path.write_bytes(_canonical(inventory) + b"\n")
        result = api.upload_file(
            path_or_fileobj=inventory_path,
            path_in_repo="dataset_manifest.json",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Index {len(run_rows)} physical runs",
        )
        print(
            json.dumps(
                {
                    "repo_id": args.repo_id,
                    "revision": getattr(result, "oid", None),
                    "runs": len(run_rows),
                    "compressed_bytes": inventory["statistics"]["compressed_bytes"],
                    "checkpoint": snapshot_id,
                    "data_verified_revision": verified_revision,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
