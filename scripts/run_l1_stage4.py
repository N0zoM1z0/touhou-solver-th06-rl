#!/usr/bin/env python3
"""Run the frozen serial Stage 4 L1 collection, BC fit, and online canary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from scripts.gate_parallel_wine import (  # noqa: E402
    _atomic_json,
    _audit,
    _object,
    _pool,
    _repository_commit,
    _run_dir,
    _sha256,
    build_runner_command,
    run_batch,
    validate_gate_run,
)
from th06_rl.corpus_digest import normalized_factual_digest  # noqa: E402
from th06_rl.episode_dataset import validate_decision_epochs  # noqa: E402
from th06_rl.policy_loader import ImmutablePolicy  # noqa: E402
from th06_rl.policies.uniform_shield_exploration import (  # noqa: E402
    STATE_SCHEMA as UNIFORM_STATE_SCHEMA,
)


PREREG_SCHEMA = "th06-rl-l1-stage4-bc-prereg-v1"
SCHEDULE_SCHEMA = "th06-rl-l1-stage4-serial-schedule-v1"
COLLECTION_SCHEMA = "th06-rl-l1-stage4-collection-v1"
RESULT_SCHEMA = "th06-rl-l1-stage4-bc-result-v1"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY))
    except ValueError as error:
        raise ValueError(f"experiment path must be inside the repository: {path}") from error


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _require_clean_worktree() -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("L1 evidence requires a clean committed worktree")


def _work_event(work_log: Path, event: str, **details: object) -> None:
    record = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **details,
    }
    encoded = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    with (work_log / "events.jsonl").open("a", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def _repository_path(value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("preregistered paths must be nonempty and repository-relative")
    path = (REPOSITORY / value).resolve()
    _relative(path)
    return path


def _derived_policy_seed(domain: str, base_seed: int, episode: int) -> int:
    digest = hashlib.sha256(
        f"{domain}:{base_seed:016x}:{episode:016x}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def load_prereg(path: Path) -> dict[str, Any]:
    prereg = _object(path.resolve())
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("Stage 4 L1 preregistration schema mismatch")
    collection = prereg.get("collection")
    fit = prereg.get("fit")
    gate = prereg.get("gate")
    canary = prereg.get("online_canary")
    paths = prereg.get("paths")
    bindings = prereg.get("sha256_bindings")
    if not all(
        isinstance(value, dict)
        for value in (collection, fit, gate, canary, paths, bindings)
    ):
        raise ValueError("Stage 4 L1 preregistration is incomplete")
    assert isinstance(collection, dict)
    assert isinstance(fit, dict)
    assert isinstance(gate, dict)
    assert isinstance(canary, dict)
    assert isinstance(paths, dict)
    assert isinstance(bindings, dict)
    rows = collection.get("episodes")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("Stage 4 L1 requires exactly 12 preregistered episodes")
    if (
        collection.get("stage") != 4
        or collection.get("difficulty") != "lunatic"
        or collection.get("serial_wine_workers") != 1
        or collection.get("natural_retail_rng") is not True
        or collection.get("game_clock") != "original-retail-normal-speed"
        or collection.get("exploration_probability") != 0.2
        or collection.get("complete_episode_required") is not True
        or collection.get("physical_hit_retry") is not False
        or collection.get("max_attempts_per_slot") != 3
    ):
        raise ValueError("Stage 4 L1 collection contract changed")
    admission = collection.get("infrastructure_admission")
    if not isinstance(admission, dict) or admission != {
        "capture_p99_ms_max": 40.0,
        "dense_shield_parity_required": True,
        "observation_gap_rate_max": 0.005,
        "player_successor_bit_exact_required": True,
        "solve_p99_ms_max": 16.7,
        "stale_retry_rate_max": 0.0,
        "validator": "scripts.gate_parallel_wine.validate_gate_run",
    }:
        raise ValueError("Stage 4 L1 infrastructure admission changed")
    base_seed = int(collection.get("base_policy_seed", -1))
    domain = collection.get("policy_seed_derivation_domain")
    if not isinstance(domain, str) or not 0 <= base_seed < 2**64:
        raise ValueError("Stage 4 L1 seed contract is invalid")
    indices = []
    splits: dict[str, list[int]] = {"train": [], "validation": []}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("Stage 4 L1 episode row is malformed")
        index = int(raw.get("index", -1))
        split = raw.get("split")
        seed = int(raw.get("policy_seed", -1))
        if split not in splits or seed != _derived_policy_seed(domain, base_seed, index):
            raise ValueError("Stage 4 L1 episode split or seed changed")
        indices.append(index)
        splits[str(split)].append(index)
    if indices != list(range(12)) or [len(splits[key]) for key in splits] != [8, 4]:
        raise ValueError("Stage 4 L1 whole-episode split changed")
    if prereg.get("auxiliary_targets") != []:
        raise ValueError("L1 may not add auxiliary targets")
    if (
        fit.get("decision_epoch_schema") != "th06-rl-decision-epoch-v1"
        or fit.get("feature_schema")
        != "th06-rl-current-observation-features-v1"
        or fit.get("target_schema")
        != "th06-rl-published-executed-action-target-v1"
        or fit.get("model") != "masked-linear-softmax"
        or gate.get("minimum_validation_episodes") != 4
        or canary.get("run_only_if_bc_gate_passes") is not True
        or canary.get("corpus_is_training_data") is not False
        or canary.get("natural_retail_rng") is not True
        or canary.get("stage") != 4
    ):
        raise ValueError("Stage 4 L1 learner or canary contract changed")
    for key, binding_key in (
        ("base_policy_state", "base_policy_state"),
        ("uniform_policy_plugin", "uniform_policy_plugin"),
        ("bc_policy_plugin", "bc_policy_plugin"),
    ):
        resolved = _repository_path(paths[key])
        if not resolved.is_file() or _sha256(resolved) != bindings.get(binding_key):
            raise ValueError(f"preregistered input hash differs: {key}")
    for key in (
        "artifact_root",
        "corpus_root",
        "collection_ledger",
        "fit_artifact",
        "experiment_result",
        "wine_pool",
        "work_log_root",
    ):
        _repository_path(paths[key])
    base_state = _object(_repository_path(paths["base_policy_state"]))
    if (
        base_state.get("schema") != UNIFORM_STATE_SCHEMA
        or base_state.get("policy_seed") != base_seed
        or base_state.get("exploration_probability") != 0.2
    ):
        raise ValueError("base exploration policy differs from the preregistration")
    return prereg


def derive_episode_policy_state(
    prereg: dict[str, Any],
    *,
    episode: int,
) -> dict[str, object]:
    collection = prereg["collection"]
    paths = prereg["paths"]
    row = collection["episodes"][episode]
    state = _object(_repository_path(paths["base_policy_state"]))
    state["policy_seed"] = int(row["policy_seed"])
    return state


def build_schedule(
    prereg: dict[str, Any],
    *,
    prereg_path: Path,
    pool_path: Path,
    commit: str,
    work_log_path: Path,
) -> dict[str, object]:
    rows = []
    for raw in prereg["collection"]["episodes"]:
        index = int(raw["index"])
        state = derive_episode_policy_state(prereg, episode=index)
        name = f"stage4-{index:03d}"
        rows.append({
            **raw,
            "name": name,
            "policy_state_path": f"policy-states/{name}.json",
            "policy_state_sha256": hashlib.sha256(_canonical(state)).hexdigest(),
        })
    return {
        "schema": SCHEDULE_SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "pool_sha256": _sha256(pool_path),
        "work_log_path": _relative(work_log_path),
        "game_clock": "original-retail-normal-speed",
        "natural_retail_rng": True,
        "serial_wine_workers": 1,
        "episodes": rows,
    }


def _episode_evidence(
    *,
    row: dict[str, Any],
    artifact_dir: Path,
    corpus_dir: Path,
    policy_plugin: Path,
    policy_state: Path,
) -> dict[str, object]:
    run_dir = _run_dir(corpus_dir)
    audit = _audit(run_dir, artifact_dir)
    report = _object(artifact_dir / "report.json")
    run = _object(run_dir / "run.json")
    manifest = _object(run_dir / "manifest.json")
    verification = validate_gate_run(
        report=report,
        run=run,
        manifest=manifest,
        audit=audit,
        stage=4,
    )
    if (
        report.get("policy_plugin_sha256") != _sha256(policy_plugin)
        or report.get("policy_state_sha256_before") != _sha256(policy_state)
        or report.get("immutable_policy_state_equal") is not True
    ):
        raise ValueError("Wine episode policy binding differs")
    decisions = validate_decision_epochs(run_dir)
    return {
        **row,
        "run_id": manifest.get("run_id"),
        "run_dir": _relative(run_dir),
        "artifact_dir": _relative(artifact_dir),
        "report_sha256": _sha256(artifact_dir / "report.json"),
        "audit_sha256": _sha256(artifact_dir / "infra-audit.json"),
        "run_sha256": _sha256(run_dir / "run.json"),
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
        "normalized_factual_digest": normalized_factual_digest(run_dir),
        "decision_view": decisions,
        "verification": verification,
    }


def _verify_recorded_evidence(evidence: dict[str, object]) -> None:
    bindings = (
        ("run_dir", "run.json", "run_sha256"),
        ("run_dir", "manifest.json", "manifest_sha256"),
        ("artifact_dir", "report.json", "report_sha256"),
        ("artifact_dir", "infra-audit.json", "audit_sha256"),
    )
    for root_key, name, digest_key in bindings:
        root = _repository_path(evidence.get(root_key))
        path = root / name
        if not path.is_file() or _sha256(path) != evidence.get(digest_key):
            raise ValueError(f"recorded episode evidence differs: {path}")
    run_dir = _repository_path(evidence.get("run_dir"))
    if validate_decision_epochs(run_dir) != evidence.get("decision_view"):
        raise ValueError("recorded decision view differs")


def _run_attempts(
    *,
    prereg: dict[str, Any],
    row: dict[str, Any],
    worker: dict[str, Any],
    score_template: Path,
    artifact_root: Path,
    corpus_root: Path,
    policy_plugin: Path,
    policy_state: Path,
    group: str,
    max_attempts: int,
    work_log: Path,
) -> dict[str, object]:
    name = str(row["name"])
    for attempt in range(1, max_attempts + 1):
        attempt_name = f"{name}-attempt-{attempt}"
        record_path = artifact_root / "attempt-results" / group / f"{attempt_name}.json"
        artifact_dir = artifact_root / group / attempt_name
        corpus_dir = corpus_root / group / attempt_name
        launcher_log = artifact_root / "logs" / group / f"{attempt_name}.log"
        if record_path.is_file():
            record = _object(record_path)
            if (
                record.get("episode") != row.get("index")
                or record.get("attempt") != attempt
                or record.get("policy_state_sha256") != _sha256(policy_state)
            ):
                raise ValueError(f"attempt record differs: {attempt_name}")
            if record.get("accepted") is True:
                evidence = record.get("evidence")
                if not isinstance(evidence, dict):
                    raise ValueError(f"accepted attempt lacks evidence: {attempt_name}")
                _verify_recorded_evidence(evidence)
                _work_event(
                    work_log,
                    "attempt-resumed-accepted",
                    group=group,
                    episode=row.get("index"),
                    attempt=attempt,
                    run_id=evidence.get("run_id"),
                )
                return evidence
            continue
        if artifact_dir.exists() or corpus_dir.exists() or launcher_log.exists():
            raise ValueError(f"partial attempt requires manual triage: {attempt_name}")
        command = build_runner_command(
            worker=worker,
            score_template=score_template,
            policy_plugin=policy_plugin,
            policy_state=policy_state,
            stage=4,
            rng_seed=None,
            artifact_dir=artifact_dir,
            corpus_root=corpus_dir,
        )
        _work_event(
            work_log,
            "attempt-started",
            group=group,
            episode=row.get("index"),
            attempt=attempt,
            command=command,
            launcher_log=_relative(launcher_log),
        )
        try:
            run_batch([(attempt_name, command, launcher_log)])
            evidence = _episode_evidence(
                row=row,
                artifact_dir=artifact_dir,
                corpus_dir=corpus_dir,
                policy_plugin=policy_plugin,
                policy_state=policy_state,
            )
        except Exception as error:
            _atomic_json(record_path, {
                "schema": "th06-rl-l1-stage4-attempt-v1",
                "accepted": False,
                "episode": row.get("index"),
                "attempt": attempt,
                "policy_state_sha256": _sha256(policy_state),
                "error": f"{type(error).__name__}: {error}",
            })
            _work_event(
                work_log,
                "attempt-rejected",
                group=group,
                episode=row.get("index"),
                attempt=attempt,
                error=f"{type(error).__name__}: {error}",
                record_path=_relative(record_path),
            )
            continue
        _atomic_json(record_path, {
            "schema": "th06-rl-l1-stage4-attempt-v1",
            "accepted": True,
            "episode": row.get("index"),
            "attempt": attempt,
            "policy_state_sha256": _sha256(policy_state),
            "evidence": evidence,
        })
        _work_event(
            work_log,
            "attempt-accepted",
            group=group,
            episode=row.get("index"),
            attempt=attempt,
            run_id=evidence.get("run_id"),
            physical_hits=(evidence.get("verification") or {}).get("physical_hits"),
            decision_view=evidence.get("decision_view"),
            record_path=_relative(record_path),
        )
        return evidence
    raise RuntimeError(f"scheduled slot exhausted its infrastructure retries: {name}")


def _fit_command(
    prereg: dict[str, Any],
    evidence: dict[int, dict[str, object]],
    output: Path,
) -> list[str]:
    fit = prereg["fit"]
    command = [sys.executable, str(REPOSITORY / "scripts/fit_behavior_clone.py")]
    for split, option in (("train", "--train-run"), ("validation", "--validation-run")):
        for row in prereg["collection"]["episodes"]:
            if row["split"] == split:
                command.extend((option, str(REPOSITORY / evidence[int(row["index"])]["run_dir"])))
    command.extend((
        "--output", str(output),
        "--epochs", str(fit["epochs"]),
        "--learning-rate", str(fit["learning_rate"]),
        "--l2", str(fit["l2"]),
        "--seed", str(fit["seed"]),
        "--bootstrap-samples", str(fit["bootstrap_samples"]),
        "--calibration-tolerance", str(fit["calibration_tolerance"]),
        "--max-rows", str(fit["max_rows_per_split"]),
    ))
    return command


def _validate_model(
    prereg: dict[str, Any],
    evidence: dict[int, dict[str, object]],
    model_path: Path,
    *,
    commit: str,
) -> dict[str, Any]:
    model = _object(model_path)
    paths = prereg["paths"]
    fit = prereg["fit"]
    plugin = _repository_path(paths["bc_policy_plugin"])
    ImmutablePolicy(plugin, state_path=model_path)
    provenance = model.get("provenance")
    recorded_fit = model.get("fit")
    inventory = model.get("inventory")
    if not all(isinstance(value, dict) for value in (provenance, recorded_fit, inventory)):
        raise ValueError("BC artifact lacks frozen provenance, fit, or inventory")
    assert isinstance(provenance, dict)
    assert isinstance(recorded_fit, dict)
    assert isinstance(inventory, dict)
    if (
        provenance.get("code_commit") != commit
        or provenance.get("policy_plugin_sha256") != _sha256(plugin)
        or recorded_fit.get("epochs") != fit["epochs"]
        or recorded_fit.get("learning_rate") != fit["learning_rate"]
        or recorded_fit.get("l2") != fit["l2"]
        or recorded_fit.get("seed") != fit["seed"]
        or recorded_fit.get("bootstrap_samples") != fit["bootstrap_samples"]
        or recorded_fit.get("calibration_tolerance")
        != fit["calibration_tolerance"]
    ):
        raise ValueError("BC artifact differs from the preregistered fit")
    expected_ids = {
        split: [
            evidence[int(row["index"])]["run_id"]
            for row in prereg["collection"]["episodes"]
            if row["split"] == split
        ]
        for split in ("train", "validation")
    }
    observed_ids = {
        split: [row.get("episode_id") for row in inventory.get(split, ())]
        for split in ("train", "validation")
    }
    if observed_ids != expected_ids:
        raise ValueError("BC artifact whole-episode inventory differs")
    return model


def run(prereg_path: Path) -> dict[str, object]:
    _require_clean_worktree()
    prereg_path = prereg_path.resolve()
    prereg = load_prereg(prereg_path)
    paths = prereg["paths"]
    pool_path = _repository_path(paths["wine_pool"])
    pool = _pool(pool_path)
    workers = sorted(pool["workers"], key=lambda item: int(item["worker"]))
    worker_index = int(prereg["collection"]["worker_index"])
    worker = next(
        item for item in workers if int(item["worker"]) == worker_index
    )
    artifact_root = _repository_path(paths["artifact_root"])
    corpus_root = _repository_path(paths["corpus_root"])
    collection_path = _repository_path(paths["collection_ledger"])
    model_path = _repository_path(paths["fit_artifact"])
    result_path = _repository_path(paths["experiment_result"])
    if result_path.is_file():
        result = _object(result_path)
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("preregistration_sha256") != _sha256(prereg_path)
        ):
            raise ValueError("completed Stage 4 L1 result differs")
        return result
    commit = _repository_commit()
    schedule_path = artifact_root / "schedule.json"
    if schedule_path.is_file():
        existing_schedule = _object(schedule_path)
        work_log = _repository_path(existing_schedule.get("work_log_path"))
        schedule = build_schedule(
            prereg,
            prereg_path=prereg_path,
            pool_path=pool_path,
            commit=commit,
            work_log_path=work_log,
        )
        if _object(schedule_path) != schedule:
            raise ValueError("Stage 4 L1 resume schedule differs")
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("Stage 4 L1 artifact root lacks its schedule")
        started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        work_log = _repository_path(paths["work_log_root"]) / (
            f"{started}-{prereg['experiment_id']}"
        )
        session_path = work_log / "session.json"
        if work_log.exists():
            raise ValueError(f"work log already exists: {work_log}")
        _atomic_json(session_path, {
            "schema": "th06-rl-work-log-session-v1",
            "experiment_id": prereg["experiment_id"],
            "repository_commit": commit,
            "preregistration_path": _relative(prereg_path),
            "preregistration_sha256": _sha256(prereg_path),
        })
        schedule = build_schedule(
            prereg,
            prereg_path=prereg_path,
            pool_path=pool_path,
            commit=commit,
            work_log_path=work_log,
        )
        _atomic_json(schedule_path, schedule)
    _work_event(
        work_log,
        "experiment-started-or-resumed",
        repository_commit=commit,
        schedule_path=_relative(schedule_path),
        schedule_sha256=_sha256(schedule_path),
    )
    base_policy = _repository_path(paths["uniform_policy_plugin"])
    for row in schedule["episodes"]:
        state = derive_episode_policy_state(prereg, episode=int(row["index"]))
        state_path = artifact_root / str(row["policy_state_path"])
        if state_path.is_file():
            if _object(state_path) != state:
                raise ValueError(f"episode policy state differs: {state_path}")
        else:
            _atomic_json(state_path, state)
        if _sha256(state_path) != row["policy_state_sha256"]:
            raise ValueError(f"episode policy state hash differs: {state_path}")
    score_template = Path(pool["score_template"]).resolve()
    evidence: dict[int, dict[str, object]] = {}
    for row in schedule["episodes"]:
        index = int(row["index"])
        evidence[index] = _run_attempts(
            prereg=prereg,
            row=row,
            worker=worker,
            score_template=score_template,
            artifact_root=artifact_root,
            corpus_root=corpus_root,
            policy_plugin=base_policy,
            policy_state=artifact_root / str(row["policy_state_path"]),
            group="collection",
            max_attempts=int(prereg["collection"]["max_attempts_per_slot"]),
            work_log=work_log,
        )
    collection = {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "repository_commit": commit,
        "preregistration_sha256": _sha256(prereg_path),
        "schedule_sha256": _sha256(schedule_path),
        "pool_sha256": _sha256(pool_path),
        "game_clock": "original-retail-normal-speed",
        "natural_retail_rng": True,
        "serial_wine_workers": 1,
        "episodes": [evidence[index] for index in sorted(evidence)],
    }
    if collection_path.is_file():
        if _object(collection_path) != collection:
            raise ValueError("Stage 4 L1 collection ledger differs")
    else:
        _atomic_json(collection_path, collection)
    _work_event(
        work_log,
        "collection-complete",
        collection_ledger=_relative(collection_path),
        collection_ledger_sha256=_sha256(collection_path),
        episodes=len(evidence),
    )
    if not model_path.is_file():
        fit_log = artifact_root / "logs" / "fit.log"
        if fit_log.exists():
            raise ValueError("partial L1 fit requires manual triage")
        fit_command = _fit_command(prereg, evidence, model_path)
        _work_event(
            work_log,
            "fit-started",
            command=fit_command,
            launcher_log=_relative(fit_log),
        )
        run_batch([("fit", fit_command, fit_log)])
    model = _validate_model(prereg, evidence, model_path, commit=commit)
    model_gate = bool((model.get("fit") or {}).get("learnability_gate_passed"))
    _work_event(
        work_log,
        "fit-complete",
        fit_artifact=_relative(model_path),
        fit_artifact_sha256=_sha256(model_path),
        policy_id=model.get("policy_id"),
        learnability_gate_passed=model_gate,
        fit_metrics=model.get("fit"),
    )
    canary_evidence = None
    if model_gate:
        canary_evidence = _run_attempts(
            prereg=prereg,
            row={"index": 12, "name": "stage4-bc-canary", "split": "canary"},
            worker=worker,
            score_template=score_template,
            artifact_root=artifact_root,
            corpus_root=corpus_root,
            policy_plugin=_repository_path(paths["bc_policy_plugin"]),
            policy_state=model_path,
            group="canary",
            max_attempts=int(prereg["online_canary"]["max_attempts"]),
            work_log=work_log,
        )
    result = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "decision": (
            "proceed-to-l2-preregistration" if model_gate
            else "stop-l1-bc-learnability"
        ),
        "repository_commit": commit,
        "preregistration_path": _relative(prereg_path),
        "preregistration_sha256": _sha256(prereg_path),
        "schedule_sha256": _sha256(schedule_path),
        "collection_ledger_sha256": _sha256(collection_path),
        "fit_artifact_path": _relative(model_path),
        "fit_artifact_sha256": _sha256(model_path),
        "policy_id": model.get("policy_id"),
        "learnability_gate_passed": model_gate,
        "fit_metrics": model.get("fit"),
        "online_canary": canary_evidence,
        "claim": (
            "behavior learnability and online integration only; no HIT reduction "
            "or NMNB improvement claim"
        ),
    }
    _atomic_json(result_path, result)
    _work_event(
        work_log,
        "experiment-complete",
        result_path=_relative(result_path),
        result_sha256=_sha256(result_path),
        decision=result["decision"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=REPOSITORY / "experiments/l1-stage4-bc-v1.json",
    )
    args = parser.parse_args(argv)
    result = run(args.preregistration)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
