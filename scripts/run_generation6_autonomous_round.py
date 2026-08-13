#!/usr/bin/env python3
"""Run frozen G6 Wine collection, registry refit, smoke, and Stage evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.export_generation6_policy import export_state  # noqa: E402
from scripts.export_generation6_round_policy import (  # noqa: E402
    export_round_preflight_state,
    export_round_state,
)
from scripts.run_generation5_wine import (  # noqa: E402
    _validate_complete_run,
    complete_run,
)
from scripts.run_generation6_wine_canary import (  # noqa: E402
    CANDIDATE as BASE_CANDIDATE,
    DEPLOYABLE_AUDIT,
    INFRA_EVENTS,
    NATIVE_SCORER as WINE_SCORER,
    POLICY_PLUGIN as ACTOR_PLUGIN,
    QUALIFICATION,
    _last_policy_status,
)
from th06_rl.advantage_learning import _validate_run  # noqa: E402
from th06_rl.audited_option_loader import load_audited_option_episode  # noqa: E402
from th06_rl.actions import ACTION_NAMES  # noqa: E402
from th06_rl.process_priority import parse_cpu_list, validate_nice  # noqa: E402
from th06_rl.policies.autonomous_iql_actor import (  # noqa: E402
    ALLOWED_AUTONOMOUS_ROUND_CONTRACT_SHA256,
    POLICY_NAME as ACTOR_POLICY_NAME,
)
from th06_rl.policies.generation6_collection_behavior import (  # noqa: E402
    ACTOR_MASS,
    ALLOWED_COLLECTION_CONTRACT_SHA256,
    ESS_MASS,
    OPTION_HORIZON_FRAMES,
    POLICY_NAME as COLLECTION_POLICY_NAME,
    STATE_SCHEMA as COLLECTION_STATE_SCHEMA,
    UNIFORM_MASS,
)
from th06_rl.resource_control import enforce_training_cpu_affinity  # noqa: E402
from th06_rl.sequential_learning import TRANSITION_SCHEMA  # noqa: E402
from th06_rl.wine_corpus_registry import (  # noqa: E402
    build_wine_corpus_source,
    load_wine_corpus_registry,
    select_wine_corpora,
)
from th06_rl.wine_workers import prepare_wine_worker  # noqa: E402


SCHEMA = "autonomous-generation-6-round-ledger-v1"
CONTRACT_SCHEMA = "autonomous-generation-6-round-contract-v1"
COLLECTION_PLUGIN = (
    REPOSITORY / "src/th06_rl/policies/generation6_collection_behavior.py"
)
BASE_CANARY = REPOSITORY / "config/autonomous_generation6_stage6_pilot.json"
HOST_SCORER = REPOSITORY / "build/native/libth06_rl_ranker.so"
WINDOWS_PYTHON = (
    REPOSITORY / "reference/tools/windows-python-3.11.9-embed-win32/python.exe"
)
REGISTRY = REPOSITORY / "config/wine_corpus_registry.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=REPOSITORY, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"autonomous round subprocess failed ({completed.returncode}): "
            + " ".join(command[:3])
        )


def _fit_report_matches_returncode(
    returncode: int, report: dict[str, object] | None,
) -> bool:
    """Distinguish an explicit gate rejection from a crashed production fit."""
    return bool(
        returncode in (0, 1)
        and isinstance(report, dict)
        and isinstance(report.get("passed"), bool)
        and report["passed"] == (returncode == 0)
    )


def _cpu_set(value: object) -> frozenset[int]:
    result: set[int] = set()
    for component in str(value).split(","):
        bounds = component.split("-", 1)
        try:
            first = int(bounds[0])
            last = int(bounds[-1])
        except ValueError as error:
            raise ValueError("invalid autonomous round CPU list") from error
        if first < 0 or last < first:
            raise ValueError("invalid autonomous round CPU range")
        result.update(range(first, last + 1))
    if not result:
        raise ValueError("autonomous round CPU list is empty")
    return frozenset(result)


def _validate_contract_shape(contract: dict[str, object]) -> None:
    collection = contract.get("collection")
    canary = contract.get("canary")
    evaluation = contract.get("evaluation")
    environment = contract.get("environment")
    offline = contract.get("offline")
    append = contract.get("registry_append")
    frozen = contract.get("frozen_inputs")
    latency_audit = contract.get("latency_tail_audit")
    startup_smoke = contract.get("startup_smoke")
    if not all(isinstance(value, dict) for value in (
        collection, canary, evaluation, environment, offline, append, frozen
    )):
        raise ValueError("Generation-6 round contract sections are absent")
    assert isinstance(collection, dict)
    assert isinstance(canary, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(environment, dict)
    assert isinstance(offline, dict)
    assert isinstance(append, dict)
    assert isinstance(frozen, dict)
    schedule = collection.get("schedule")
    reused_schedule = collection.get("reused_schedule", [])
    if not isinstance(reused_schedule, list):
        raise ValueError("Generation-6 reused collection schedule is invalid")
    combined_schedule = [*reused_schedule, *schedule] if isinstance(
        schedule, list
    ) else []
    if (
        collection.get("episodes") != 12
        or collection.get("stages") != [4, 5, 6]
        or not isinstance(schedule, list)
        or len(combined_schedule) != 12
        or collection.get("new_episodes", len(schedule)) != len(schedule)
        or collection.get("reused_episodes", len(reused_schedule))
        != len(reused_schedule)
        or any(
            not isinstance(row, dict)
            or set(row) != {"episode", "stage", "policy_seed"}
            for row in schedule
        )
        or any(
            not isinstance(row, dict)
            or not {"episode", "stage", "policy_seed"} <= set(row)
            for row in reused_schedule
        )
        or [row.get("episode") for row in combined_schedule]
        != list(range(1, 13))
        or {stage: sum(row["stage"] == stage for row in schedule)
            for stage in (4, 5, 6)} != {
                stage: 4 - sum(
                    row["stage"] == stage for row in reused_schedule
                )
                for stage in (4, 5, 6)
            }
    ):
        raise ValueError("Generation-6 collection schedule is not balanced/frozen")
    if reused_schedule and (
        len(str(collection.get("prior_ledger_sha256", ""))) != 64
        or not isinstance(collection.get("prior_ledger"), str)
        or not str(collection.get("prior_ledger"))
        or any(
            set(row) != {
                "episode", "stage", "policy_seed", "source", "report",
                "report_sha256", "manifest_sha256", "run_sha256",
            }
            for row in reused_schedule
        )
    ):
        raise ValueError("Generation-6 reused collection binding is invalid")
    if latency_audit is not None and (
        not isinstance(latency_audit, dict)
        or set(latency_audit) != {"path", "sha256"}
        or not isinstance(latency_audit.get("path"), str)
        or not latency_audit["path"]
        or len(str(latency_audit.get("sha256", ""))) != 64
    ):
        raise ValueError("Generation-6 latency-tail audit binding is invalid")
    if startup_smoke is not None and (
        not isinstance(startup_smoke, dict)
        or set(startup_smoke) != {"path", "sha256"}
        or not isinstance(startup_smoke.get("path"), str)
        or not startup_smoke["path"]
        or len(str(startup_smoke.get("sha256", ""))) != 64
    ):
        raise ValueError("Generation-6 startup smoke binding is invalid")
    canary_schedule = canary.get("schedule")
    if (
        not isinstance(canary_schedule, list)
        or len(canary_schedule) != 2
        or any(
            not isinstance(row, dict)
            or set(row) != {"trial", "stage", "policy_seed"}
            for row in canary_schedule
        )
        or [row.get("trial") for row in canary_schedule] != [1, 2]
        or any(row["stage"] != 4 for row in canary_schedule)
    ):
        raise ValueError("Generation-6 Wine canary schedule is invalid")
    evaluation_schedule = evaluation.get("schedule")
    expected_pairs = {
        (block, role) for block in range(6)
        for role in ("incumbent", "candidate")
    }
    expected_role_order = [
        role for block in range(6)
        for role in (
            ("incumbent", "candidate")
            if block % 2 == 0 else ("candidate", "incumbent")
        )
    ]
    if (
        not isinstance(evaluation_schedule, list)
        or len(evaluation_schedule) != 12
        or any(
            not isinstance(row, dict)
            or set(row) != {"trial", "block", "role", "policy_seed"}
            for row in evaluation_schedule
        )
        or [row.get("trial") for row in evaluation_schedule]
        != list(range(1, 13))
        or {(row["block"], row["role"]) for row in evaluation_schedule}
        != expected_pairs
        or [row["role"] for row in evaluation_schedule] != expected_role_order
    ):
        raise ValueError("Generation-6 paired Stage-6 schedule is invalid")
    seeds = [
        int(row["policy_seed"])
        for rows in (combined_schedule, canary_schedule, evaluation_schedule)
        for row in rows
    ]
    if len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**64 for seed in seeds):
        raise ValueError("Generation-6 policy seeds are invalid or reused")
    if not all(
        isinstance(offline.get(name), int)
        and 0 <= int(offline[name]) < 2**64
        for name in ("crossfit_seed", "seed_offset", "preflight_policy_seed")
    ):
        raise ValueError("Generation-6 offline seeds are invalid")
    required_capabilities = {
        "complete_stage_observation", "physical_hit_outcome",
        "representation_pretraining", "behavior_state_value",
        "factual_semi_markov_options",
        "recorded_complete_behavior_propensity", "native_safe_candidates",
        "sequential_offline_rl", "natural_rng",
        "generation6_actor_ess_behavior",
    }
    if (
        not isinstance(append.get("capabilities"), list)
        or set(append["capabilities"]) != required_capabilities
        or append.get("base_source_count") != 5
        or not append.get("source_id")
    ):
        raise ValueError("Generation-6 registry append contract is invalid")
    game_cpus = _cpu_set(environment.get("game_cpu_list"))
    controller_cpus = _cpu_set(environment.get("controller_cpu_list"))
    game_nice = environment.get("game_nice")
    controller_nice = environment.get("controller_nice")
    priority_declared = game_nice is not None or controller_nice is not None
    if (
        game_cpus & controller_cpus
        or len(game_cpus | controller_cpus) > 32
        or not (game_cpus | controller_cpus) <= set(os.sched_getaffinity(0))
        or len(str(environment.get("source_game_inventory_sha256", ""))) != 64
    ):
        raise ValueError("Generation-6 Wine CPU partition is invalid")
    if priority_declared:
        try:
            if (
                game_nice is None
                or controller_nice is None
                or validate_nice(int(game_nice)) != int(game_nice)
                or validate_nice(int(controller_nice)) != int(controller_nice)
                or environment.get("scheduler") != "SCHED_OTHER"
            ):
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Generation-6 Wine process priority is invalid"
            ) from error
    if (
        contract.get("maximum_interventions") != 64
        or len(str(contract.get("retail_executable_sha256", ""))) != 64
        or not all(
            isinstance(path, str) and path and isinstance(digest, str)
            and len(digest) == 64
            for path, digest in frozen.items()
        )
    ):
        raise ValueError("Generation-6 fixed execution contract is invalid")


def _contract(path: Path) -> tuple[dict[str, object], str]:
    contract = _object(path)
    digest = _sha256(path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or digest not in ALLOWED_COLLECTION_CONTRACT_SHA256
        or digest not in ALLOWED_AUTONOMOUS_ROUND_CONTRACT_SHA256
        or contract.get("normal_speed") is not True
        or contract.get("natural_rng") is not True
        or contract.get("complete_stage_hit_continuation") is not True
        or contract.get("bomb") != "forbidden"
    ):
        raise ValueError("Generation-6 autonomous round contract is not authorized")
    _validate_contract_shape(contract)
    source_commit = str(contract.get("implementation_commit", ""))
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=REPOSITORY, check=False,
    ).returncode:
        raise ValueError("Generation-6 round implementation commit is not an ancestor")
    for raw, expected in contract["frozen_inputs"].items():
        file = REPOSITORY / str(raw)
        if raw == "config/wine_corpus_registry.json":
            continue
        if not file.is_file() or _sha256(file) != expected:
            raise ValueError(f"Generation-6 round frozen input drifted: {raw}")
    latency_binding = contract.get("latency_tail_audit")
    if isinstance(latency_binding, dict):
        audit_path = (REPOSITORY / str(latency_binding["path"])).resolve()
        if (
            not audit_path.is_relative_to(REPOSITORY)
            or not audit_path.is_file()
            or _sha256(audit_path) != latency_binding["sha256"]
        ):
            raise ValueError("Generation-6 latency-tail audit drifted")
        latency_audit = _object(audit_path)
        gates = latency_audit.get("gates")
        priority = latency_audit.get("priority")
        environment = contract["environment"]
        expected_cpus = sorted(
            _cpu_set(environment["game_cpu_list"])
            | _cpu_set(environment["controller_cpu_list"])
        )
        if (
            latency_audit.get("schema")
            != "generation6-scheduler-tail-latency-audit-v1"
            or latency_audit.get("passed") is not True
            or not isinstance(gates, dict)
            or not gates
            or not all(value is True for value in gates.values())
            or not isinstance(priority, dict)
            or priority.get("scheduler") != "SCHED_OTHER"
            or priority.get("nice") != environment.get("controller_nice")
            or priority.get("cpus") != expected_cpus
        ):
            raise ValueError("Generation-6 latency-tail repair is not proven")
    startup_binding = contract.get("startup_smoke")
    if isinstance(startup_binding, dict):
        startup_path = (REPOSITORY / str(startup_binding["path"])).resolve()
        if (
            not startup_path.is_relative_to(REPOSITORY)
            or not startup_path.is_file()
            or _sha256(startup_path) != startup_binding["sha256"]
            or not _startup_smoke_passed(
                _object(startup_path), contract["environment"]
            )
        ):
            raise ValueError("Generation-6 repaired Wine startup smoke failed")
    if _sha256(REGISTRY) != contract["frozen_inputs"][
        "config/wine_corpus_registry.json"
    ]:
        current = _object(REGISTRY)
        sources = current.get("sources")
        if (
            not isinstance(sources, list)
            or len(sources) != int(contract["registry_append"][
                "base_source_count"
            ]) + 1
            or sources[-1].get("id") != contract["registry_append"]["source_id"]
            or sources[-1].get("root") != contract["registry_append"]["root"]
        ):
            raise ValueError("Wine registry drift is not the frozen round append")
        load_wine_corpus_registry(REGISTRY, repository=REPOSITORY)
    return contract, digest


def _collection_state(
    *, path: Path, contract_path: Path, contract_sha: str, policy_seed: int,
) -> None:
    actor_state = export_state(
        candidate_path=BASE_CANDIDATE,
        qualification_path=QUALIFICATION,
        deployable_audit_path=DEPLOYABLE_AUDIT,
        native_scorer_path=WINE_SCORER,
        canary_contract_path=BASE_CANARY,
        mode="shadow",
        policy_seed=policy_seed,
    )
    value = {
        "schema": COLLECTION_STATE_SCHEMA,
        "collection_contract_sha256": contract_sha,
        "policy_seed": policy_seed,
        "option_horizon_frames": OPTION_HORIZON_FRAMES,
        "mixture": {
            "actor": ACTOR_MASS,
            "uniform": UNIFORM_MASS,
            "inverse_ess": ESS_MASS,
        },
        "actor_state": actor_state,
    }
    if path.is_file() and _object(path) != value:
        raise ValueError(f"collection behavior state drifted: {path}")
    if not path.exists():
        _atomic_json(path, value)


def _clean_trace(report: dict[str, object]) -> bool:
    trace = report.get("trace")
    events = trace.get("event_counts") if isinstance(trace, dict) else None
    return bool(
        isinstance(events, dict)
        and not any(int(events.get(name, 0)) for name in INFRA_EVENTS)
    )


def _priority_arguments(environment: dict[str, object]) -> dict[str, int]:
    if environment.get("game_nice") is None:
        return {}
    return {
        "game_nice": int(environment["game_nice"]),
        "controller_nice": int(environment["controller_nice"]),
    }


def _priority_passed(
    report: dict[str, object], environment: dict[str, object]
) -> bool:
    if environment.get("game_nice") is None:
        return True
    for role in ("game", "controller"):
        cpu_list = str(environment[f"{role}_cpu_list"])
        nice = int(environment[f"{role}_nice"])
        attestation = report.get(f"{role}_priority_attestation")
        expected_cpus = list(parse_cpu_list(cpu_list))
        if (
            report.get(f"{role}_nice") != nice
            or not isinstance(attestation, dict)
            or attestation.get("schema")
            != "bounded-wine-process-priority-v1"
            or attestation.get("authority")
            != "linux-setpriority-and-sched-setaffinity"
            or attestation.get("scheduler") != "SCHED_OTHER"
            or attestation.get("nice") != nice
            or attestation.get("effective_nice") != nice
            or attestation.get("cpus") != expected_cpus
            or attestation.get("effective_cpus") != expected_cpus
            or attestation.get("uid") != attestation.get("target_uid")
            or attestation.get("gid") != attestation.get("target_gid")
        ):
            return False
    return True


def _startup_smoke_passed(
    report: dict[str, object], environment: dict[str, object]
) -> bool:
    trace = report.get("trace")
    game_attestation = report.get("game_priority_attestation")
    return bool(
        report.get("error") is None
        and report.get("gdb_normalized") is True
        and report.get("controller_returncode") == 0
        and report.get("immutable_policy_state_equal") is True
        and report.get("evaluation_mode") == "hit-continuation-benchmark"
        and report.get("complete_stage_training_corpus_root") is None
        and report.get("first_failure_corpus_root") is None
        and report.get("option_smoke_corpus_root") is None
        and isinstance(report.get("seconds"), (int, float))
        and 0 < float(report["seconds"]) <= 1.0
        and isinstance(trace, dict)
        and trace.get("corpus_run_ids") == []
        and trace.get("decisions") == 0
        and report.get("leftover_prefix_processes") == []
        and isinstance(report.get("game_host_pid"), int)
        and isinstance(report.get("game_process_pid"), int)
        and report["game_host_pid"] != report["game_process_pid"]
        and isinstance(game_attestation, dict)
        and game_attestation.get("pid") == report["game_process_pid"]
        and _priority_passed(report, environment)
    )


def _audit_collection(
    *, report: dict[str, object], artifact: Path, run_dir: Path,
    state_path: Path, environment: dict[str, object],
) -> dict[str, object]:
    completion = report.get("controller_completion")
    status = _last_policy_status(artifact / "trace.jsonl")
    metrics = status.get("metrics")
    actor = metrics.get("actor") if isinstance(metrics, dict) else None
    selected = metrics.get("selected") if isinstance(metrics, dict) else None
    rows, option_report = load_audited_option_episode(run_dir)
    gates = {
        "natural_rng_complete_stage_training": (
            report.get("evaluation_mode") == "natural-rng-complete-stage-training"
            and report.get("diagnostic_rng_seed") is None
        ),
        "complete_stage_and_hit_accounting": (
            report.get("controller_returncode") == 0
            and isinstance(completion, dict)
            and completion.get("practice_stage_completed") is True
            and int(completion.get("physical_hits", -1))
            == int(option_report["physical_hits"])
        ),
        "immutable_behavior_state": (
            report.get("immutable_policy_state_equal") is True
            and report.get("policy_state_sha256_before") == _sha256(state_path)
        ),
        "collection_policy_loaded_once": (
            status.get("policy_id") == COLLECTION_POLICY_NAME
            and status.get("reload_failures") == 0
            and status.get("last_error") is None
        ),
        "complete_propensity_options": bool(rows)
        and all(
            len(row.behavior_probabilities) == len(row.legal_actions)
            and min(row.behavior_probabilities) + 1e-12
            >= UNIFORM_MASS / len(row.legal_actions)
            for row in rows
        ),
        "native_safe_actions_only": (
            isinstance(selected, dict)
            and bool(selected)
            and set(selected) <= set(ACTION_NAMES)
        ),
        "zero_bomb": not isinstance(selected, dict) or "bomb" not in selected,
        "actor_runtime_below_4_ms": (
            isinstance(actor, dict)
            and isinstance(actor.get("latency_p95_ms"), (int, float))
            and float(actor["latency_p95_ms"]) < 4.0
            and actor.get("deadline_misses") == 0
        ),
        "zero_infrastructure_events": _clean_trace(report),
        "zero_leftover_prefix_processes": not report.get("leftover_prefix_processes"),
        "bounded_process_priority": _priority_passed(report, environment),
    }
    return {
        "report_sha256": _sha256(artifact / "report.json"),
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
        "run_sha256": _sha256(run_dir / "run.json"),
        "physical_hits": int(completion["physical_hits"]),
        "options": len(rows),
        "actor_proposals": int(metrics.get("actor_proposals", 0)),
        "non_actor_assignments": int(metrics.get("non_actor_assignments", 0)),
        "minimum_probability": float(metrics.get("minimum_probability", 0.0)),
        "actor_latency_p95_ms": float(actor["latency_p95_ms"]),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _recover_accepted_collection(
    *, accepted_episode: Path, artifact: Path, worker: dict[str, object],
    stage: int, environment: dict[str, object],
) -> tuple[dict[str, object], Path] | None:
    """Recover the atomic move -> ledger crash window without resampling."""
    if not accepted_episode.exists():
        return None
    manifests = sorted(accepted_episode.rglob("manifest.json"))
    if not manifests and not any(accepted_episode.iterdir()):
        return None
    if len(manifests) != 1:
        raise RuntimeError(
            f"accepted episode has an invalid run count: {accepted_episode}"
        )
    report, run_dir = _validate_complete_run(
        artifact_dir=artifact, worker=worker, stage=stage, rng_seed=None,
        corpus_root=accepted_episode,
        game_cpu_list=str(environment["game_cpu_list"]),
        controller_cpu_list=str(environment["controller_cpu_list"]),
        **_priority_arguments(environment),
    )
    if run_dir is None or run_dir != manifests[0].parent:
        raise RuntimeError("accepted episode/report binding differs")
    return report, run_dir


def _materialize_reused_collection(
    *, contract: dict[str, object], accepted_root: Path,
) -> list[dict[str, object]]:
    """Hard-link the complete passing prefix from an immutable failed round.

    Reuse is selected only by the prior machine ledger's conjunctive gates.
    It cannot skip a passing row or inspect HIT, Stage location, or outcome.
    The first failed row and everything after it remain excluded.
    """
    collection = contract["collection"]
    reused = collection.get("reused_schedule", [])
    if not isinstance(reused, list) or not reused:
        return []
    ledger_path = (REPOSITORY / str(collection["prior_ledger"])).resolve()
    if (
        not ledger_path.is_relative_to(REPOSITORY)
        or _sha256(ledger_path) != collection["prior_ledger_sha256"]
    ):
        raise ValueError("prior Generation-6 collection ledger differs")
    ledger = _object(ledger_path)
    prior_rows = ledger.get("collection")
    if (
        ledger.get("schema") != SCHEMA
        or ledger.get("status") != "invalid"
        or ledger.get("decision") != "collection-audit-failed"
        or not isinstance(prior_rows, list)
    ):
        raise ValueError("prior Generation-6 collection is not reusable")
    passing_prefix = []
    for row in prior_rows:
        if not isinstance(row, dict) or row.get("passed") is not True:
            break
        passing_prefix.append(row)
    if len(passing_prefix) != len(reused):
        raise ValueError("reused collection is not the complete passing prefix")

    result = []
    staging = accepted_root.parent / "reuse-staging"
    staging.mkdir(parents=True, exist_ok=True)
    for specification, prior in zip(reused, passing_prefix, strict=True):
        expected = {
            name: specification[name]
            for name in (
                "episode", "stage", "policy_seed", "report_sha256",
                "manifest_sha256", "run_sha256",
            )
        }
        if any(prior.get(name) != value for name, value in expected.items()):
            raise ValueError("reused collection row differs from prior ledger")
        source = (REPOSITORY / str(specification["source"])).resolve()
        report = (REPOSITORY / str(specification["report"])).resolve()
        if (
            not source.is_relative_to(REPOSITORY)
            or not report.is_relative_to(REPOSITORY)
            or Path(str(prior.get("corpus_run_dir", ""))).resolve() != source
            or not source.is_dir()
            or not report.is_file()
            or _sha256(report) != specification["report_sha256"]
            or _sha256(source / "manifest.json")
            != specification["manifest_sha256"]
            or _sha256(source / "run.json") != specification["run_sha256"]
        ):
            raise ValueError("reused collection corpus binding differs")
        _validate_run(source, transition_schema=TRANSITION_SCHEMA)
        destination = (
            accepted_root / f"episode-{int(specification['episode']):02d}"
            / source.name
        )
        if not destination.exists():
            with tempfile.TemporaryDirectory(
                prefix=f"episode-{int(specification['episode']):02d}-",
                dir=staging,
            ) as temporary:
                staged = Path(temporary) / source.name
                shutil.copytree(source, staged, copy_function=os.link)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, destination)
        if (
            _sha256(destination / "manifest.json")
            != specification["manifest_sha256"]
            or _sha256(destination / "run.json") != specification["run_sha256"]
        ):
            raise ValueError("materialized reused collection differs")
        _validate_run(destination, transition_schema=TRANSITION_SCHEMA)
        row = dict(prior)
        row.update({
            "corpus_run_dir": str(destination),
            "reused_from": str(source),
            "reused_under_successor_contract": True,
        })
        result.append(row)
    return result


def _append_registry(
    *, contract: dict[str, object], accepted_root: Path,
) -> tuple[dict[str, object], str]:
    specification = contract["registry_append"]
    source = build_wine_corpus_source(
        repository=REPOSITORY,
        root=accepted_root,
        source_id=str(specification["source_id"]),
        access="training",
        capabilities=tuple(map(str, specification["capabilities"])),
        transition_schema=TRANSITION_SCHEMA,
        executable_sha256=str(contract["retail_executable_sha256"]),
    )
    if source["expected_clean_complete_runs"] != int(
        contract["collection"]["episodes"]
    ):
        raise ValueError("new registry source does not contain the frozen episode count")
    registry = _object(REGISTRY)
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise TypeError("Wine registry sources are absent")
    existing = next(
        (row for row in sources if row.get("id") == source["id"]), None
    )
    if existing is None:
        sources.append(source)
        _atomic_json(REGISTRY, registry)
    elif existing != source:
        raise ValueError("existing autonomous round registry row drifted")
    load_wine_corpus_registry(REGISTRY, repository=REPOSITORY)
    return source, _sha256(REGISTRY)


def _offline(
    *, root: Path, contract_path: Path, contract: dict[str, object],
    contract_sha: str,
) -> tuple[Path | None, Path | None]:
    offline = root / "offline"
    synthetic = offline / "synthetic.json"
    if not synthetic.is_file():
        _run([
            sys.executable, str(REPOSITORY / "scripts/smoke_generation6_iql_actor.py"),
            "--output", str(synthetic), "--threads", "8",
        ])
    crossfit = offline / "crossfit.json"
    if not crossfit.is_file():
        _run([
            sys.executable,
            str(REPOSITORY / "scripts/develop_generation6_crossfit_actor.py"),
            "--output", str(crossfit), "--registry", str(REGISTRY),
            "--threads", "32", "--folds", "5",
            "--seed", str(contract["offline"]["crossfit_seed"]),
        ])
    candidate = offline / "candidate.json"
    if not candidate.is_file():
        fit_command = [
            sys.executable, str(REPOSITORY / "scripts/fit_generation6_candidate.py"),
            "--output", str(candidate), "--registry", str(REGISTRY),
            "--threads", "32", "--seed-offset",
            str(contract["offline"]["seed_offset"]),
            "--round-contract-sha256", contract_sha,
            "--native-scorer", str(HOST_SCORER),
        ]
        fit_checkpoint = candidate.with_name(candidate.stem + ".fit.json")
        if fit_checkpoint.is_file():
            fit_command.extend(("--resume-fit", str(fit_checkpoint)))
        completed = subprocess.run(fit_command, cwd=REPOSITORY, check=False)
        report = _object(candidate) if candidate.is_file() else None
        if not _fit_report_matches_returncode(completed.returncode, report):
            raise RuntimeError("Generation-6 production fit crashed")
    linux_state = offline / "preflight-linux-state.json"
    windows_state = offline / "preflight-windows-state.json"
    if not linux_state.is_file():
        _atomic_json(linux_state, export_round_preflight_state(
            candidate_path=candidate, registry_path=REGISTRY,
            scorer_path=HOST_SCORER, contract_path=contract_path,
            policy_seed=int(contract["offline"]["preflight_policy_seed"]),
        ))
    if not windows_state.is_file():
        _atomic_json(windows_state, export_round_preflight_state(
            candidate_path=candidate, registry_path=REGISTRY,
            scorer_path=WINE_SCORER, contract_path=contract_path,
            policy_seed=int(contract["offline"]["preflight_policy_seed"]),
        ))
    online = offline / "online-preflight.json"
    candidate_passed = _object(candidate).get("passed") is True
    if not online.is_file() and not candidate_passed:
        _atomic_json(online, {
            "schema": "autonomous-generation-6-online-policy-preflight-v1",
            "evidence_eligible": False,
            "passed": False,
            "candidate_sha256": _sha256(candidate),
            "skipped": "production-candidate-rejected",
        })
    elif not online.is_file():
        _run([
            sys.executable,
            str(REPOSITORY / "scripts/smoke_generation6_online_policy.py"),
            "--candidate", str(candidate),
            "--linux-state", str(linux_state),
            "--windows-state", str(windows_state),
            "--linux-library", str(HOST_SCORER),
            "--windows-library", str(WINE_SCORER),
            "--windows-python", str(WINDOWS_PYTHON),
            "--output", str(online), "--contexts", "64",
            "--repetitions", "1200",
        ])
    audit = offline / "audit.json"
    if not audit.is_file():
        completed = subprocess.run([
            sys.executable,
            str(REPOSITORY / "scripts/audit_generation6_round_offline.py"),
            "--synthetic", str(synthetic), "--crossfit", str(crossfit),
            "--candidate", str(candidate), "--registry", str(REGISTRY),
            "--online", str(online),
            "--contract", str(contract_path), "--output", str(audit),
        ], cwd=REPOSITORY, check=False)
        if completed.returncode not in (0, 1):
            raise RuntimeError("Generation-6 round offline audit crashed")
    return (candidate, audit) if _object(audit).get("passed") is True else (None, audit)


def _audit_evaluation(
    *, report: dict[str, object], artifact: Path, state_path: Path,
    role: str, environment: dict[str, object],
) -> dict[str, object]:
    completion = report.get("controller_completion")
    status = _last_policy_status(artifact / "trace.jsonl")
    metrics = status.get("metrics")
    selected = metrics.get("selected") if isinstance(metrics, dict) else None
    mode = "active" if role == "candidate" else "shadow"
    gates = {
        "natural_rng_hit_continuation": (
            report.get("evaluation_mode") == "hit-continuation-benchmark"
            and report.get("diagnostic_rng_seed") is None
        ),
        "complete_stage": (
            report.get("controller_returncode") == 0
            and isinstance(completion, dict)
            and completion.get("practice_stage_completed") is True
        ),
        "immutable_state": (
            report.get("immutable_policy_state_equal") is True
            and report.get("policy_state_sha256_before") == _sha256(state_path)
        ),
        "expected_policy": (
            status.get("policy_id") == f"{ACTOR_POLICY_NAME}-{mode}"
            and metrics.get("mode") == mode
            and status.get("last_error") is None
        ),
        "native_safe_and_zero_bomb": (
            isinstance(selected, dict) and bool(selected)
            and set(selected) <= set(ACTION_NAMES) and "bomb" not in selected
        ),
        "shadow_zero_interventions": role != "incumbent"
        or metrics.get("interventions") == 0,
        "candidate_budget_clean": role != "candidate" or (
            int(metrics.get("interventions", -1))
            <= int(metrics.get("intervention_budget", -2)) <= 64
            and metrics.get("budget_abstentions") == 0
        ),
        "latency_and_deadline": (
            float(metrics.get("latency_p95_ms", float("inf"))) < 4.0
            and metrics.get("deadline_misses") == 0
        ),
        "zero_infrastructure_events": _clean_trace(report),
        "zero_corpus": not report.get("trace", {}).get("corpus_run_ids"),
        "zero_leftover_prefix_processes": not report.get("leftover_prefix_processes"),
        "bounded_process_priority": _priority_passed(report, environment),
    }
    return {
        "role": role,
        "report_sha256": _sha256(artifact / "report.json"),
        "physical_hits": int(completion["physical_hits"]),
        "proposals": int(metrics.get("proposals", 0)),
        "interventions": int(metrics.get("interventions", 0)),
        "latency_p95_ms": float(metrics["latency_p95_ms"]),
        "deadline_misses": int(metrics.get("deadline_misses", 0)),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _paired_verdict(rows: list[dict[str, object]]) -> dict[str, object]:
    blocks: dict[int, dict[str, int]] = {}
    for row in rows:
        blocks.setdefault(int(row["block"]), {})[str(row["role"])] = int(
            row["physical_hits"]
        )
    valid_shape = (
        set(blocks) == set(range(6))
        and all(set(blocks[index]) == {"incumbent", "candidate"}
                for index in blocks)
    )
    incumbent = (
        [blocks[index]["incumbent"] for index in range(6)]
        if valid_shape else []
    )
    candidate = (
        [blocks[index]["candidate"] for index in range(6)]
        if valid_shape else []
    )
    differences = [left - right for left, right in zip(incumbent, candidate)]
    candidate_total = sum(candidate)
    incumbent_total = sum(incumbent)
    exercised = sum(
        int(row["interventions"]) > 0 for row in rows
        if row["role"] == "candidate"
    )
    no_worse = sum(value >= 0 for value in differences)
    valid = (
        valid_shape and len(rows) == 12 and all(row["passed"] for row in rows)
    )
    positive = (
        valid and exercised >= 4 and candidate_total < incumbent_total
        and no_worse >= 4
    )
    return {
        "incumbent_hits": incumbent,
        "candidate_hits": candidate,
        "block_effects": differences,
        "incumbent_total_hits": incumbent_total,
        "candidate_total_hits": candidate_total,
        "effect_hits": incumbent_total - candidate_total,
        "candidate_exercised_stages": exercised,
        "candidate_no_worse_blocks": no_worse,
        "verdict": (
            "effective-learning-signal" if positive
            else "no-effective-learning-signal" if valid
            else "invalid"
        ),
        "promotion_eligible": False,
    }


def run(args: argparse.Namespace) -> int:
    contract_path = args.contract.resolve()
    contract, contract_sha = _contract(contract_path)
    enforce_training_cpu_affinity(32)
    root = args.output_root.resolve()
    state_path = root / "generation.json"
    state: dict[str, Any]
    if state_path.is_file():
        state = _object(state_path)
        if state.get("schema") != SCHEMA or state.get("contract_sha256") != contract_sha:
            raise ValueError("Generation-6 autonomous round ledger drifted")
        if state.get("status") == "complete":
            print(json.dumps(state.get("decision"), sort_keys=True))
            return 0
        if state.get("status") == "invalid":
            print(json.dumps(state.get("decision"), sort_keys=True))
            return 1
    else:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPOSITORY, text=True,
        )
        if dirty.strip():
            raise RuntimeError("autonomous round requires a clean tracked checkout")
        state = {
            "schema": SCHEMA,
            "contract_sha256": contract_sha,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
            ).strip(),
            "status": "collecting",
            "collection": [],
            "registry": None,
            "offline": None,
            "canary": [],
            "evaluation": [],
            "decision": None,
        }
        _atomic_json(state_path, state)
    collection_rows = state.get("collection")
    if not isinstance(collection_rows, list):
        raise ValueError("Generation-6 collection ledger is invalid")
    schedule = contract["collection"]["schedule"]
    reused_schedule = contract["collection"].get("reused_schedule", [])
    combined_schedule = [*reused_schedule, *schedule]
    if len(collection_rows) > len(combined_schedule) or any(
        not isinstance(actual, dict)
        or int(actual.get("episode", -1)) != int(expected["episode"])
        or int(actual.get("stage", -1)) != int(expected["stage"])
        or int(actual.get("policy_seed", -1)) != int(expected["policy_seed"])
        or actual.get("passed") is not True
        for actual, expected in zip(
            collection_rows, combined_schedule, strict=False
        )
    ):
        raise ValueError("Generation-6 collection ledger/schedule differs")
    registered = state.get("registry")
    if registered is not None and (
        not isinstance(registered, dict)
        or registered.get("sha256") != _sha256(REGISTRY)
    ):
        raise ValueError("Generation-6 ledger registry binding differs")
    environment = contract["environment"]
    worker = prepare_wine_worker(
        root=root / "workers",
        source_game_dir=REPOSITORY / str(environment["source_game_dir"]),
        worker=0,
        directory=str(environment["worker_directory"]),
        display=str(environment["display"]),
        source_inventory_sha256=str(
            environment["source_game_inventory_sha256"]
        ),
    )
    accepted = root / "accepted-corpus"
    reused_rows = _materialize_reused_collection(
        contract=contract, accepted_root=accepted
    )
    if reused_rows:
        if not state["collection"]:
            state["collection"] = reused_rows
            _atomic_json(state_path, state)
        elif state["collection"][:len(reused_rows)] != reused_rows:
            raise ValueError("Generation-6 reused collection ledger differs")
    new_completed = len(state["collection"]) - len(reused_rows)
    if not 0 <= new_completed <= len(schedule):
        raise ValueError("Generation-6 successor collection progress differs")
    for row in schedule[new_completed:]:
        episode = int(row["episode"])
        behavior = root / "behavior" / f"episode-{episode:02d}.json"
        _collection_state(
            path=behavior, contract_path=contract_path,
            contract_sha=contract_sha, policy_seed=int(row["policy_seed"]),
        )
        artifact = root / "collection" / f"episode-{episode:02d}-stage-{row['stage']}"
        accepted_episode = accepted / f"episode-{episode:02d}"
        recovered = _recover_accepted_collection(
            accepted_episode=accepted_episode, artifact=artifact,
            worker=worker, stage=int(row["stage"]), environment=environment,
        )
        if recovered is None:
            report, run_dir = complete_run(
                artifact_dir=artifact, worker=worker, stage=int(row["stage"]),
                policy_plugin=COLLECTION_PLUGIN, policy_state=behavior,
                scorer=WINE_SCORER, rng_seed=None,
                corpus_root=root / "spool" / f"episode-{episode:02d}",
                game_cpu_list=str(environment["game_cpu_list"]),
                controller_cpu_list=str(environment["controller_cpu_list"]),
                **_priority_arguments(environment),
            )
            assert run_dir is not None
            destination = accepted_episode / run_dir.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(run_dir), destination)
        else:
            report, destination = recovered
        _validate_run(destination, transition_schema=TRANSITION_SCHEMA)
        audited = _audit_collection(
            report=report, artifact=artifact, run_dir=destination,
            state_path=behavior, environment=environment,
        )
        audited.update({
            "episode": episode, "stage": int(row["stage"]),
            "policy_seed": int(row["policy_seed"]),
            "corpus_run_dir": str(destination),
        })
        state["collection"].append(audited)
        _atomic_json(state_path, state)
        print(json.dumps({"collection_completed": audited}, sort_keys=True), flush=True)
        if not audited["passed"]:
            state.update({"status": "invalid", "decision": "collection-audit-failed"})
            _atomic_json(state_path, state)
            return 1
    source, registry_sha = _append_registry(
        contract=contract, accepted_root=accepted
    )
    state["registry"] = {"source": source, "sha256": registry_sha}
    state["status"] = "offline"
    _atomic_json(state_path, state)
    candidate, audit = _offline(
        root=root, contract_path=contract_path, contract=contract,
        contract_sha=contract_sha,
    )
    state["offline"] = {
        "audit": str(audit), "audit_sha256": _sha256(audit),
        "passed": candidate is not None,
    }
    if candidate is None:
        state.update({"status": "complete", "decision": {
            "verdict": "offline-rejected",
            "reason": "predeclared synthetic/crossfit/native smoke failed",
        }})
        _atomic_json(state_path, state)
        print(json.dumps(state["decision"], sort_keys=True))
        return 0
    states = root / "policy-states"
    canary_schedule = contract["canary"]["schedule"]
    state["status"] = "canary"
    _atomic_json(state_path, state)
    for row in canary_schedule[len(state["canary"]):]:
        trial = int(row["trial"])
        policy_state = states / f"canary-{trial:02d}.json"
        if not policy_state.is_file():
            _atomic_json(policy_state, export_round_state(
                candidate_path=candidate, smoke_path=audit,
                registry_path=REGISTRY, scorer_path=WINE_SCORER,
                contract_path=contract_path, mode="active",
                policy_seed=int(row["policy_seed"]),
            ))
        artifact = root / "canary" / f"trial-{trial:02d}"
        report, run_dir = complete_run(
            artifact_dir=artifact, worker=worker, stage=int(row["stage"]),
            policy_plugin=ACTOR_PLUGIN, policy_state=policy_state,
            scorer=WINE_SCORER, rng_seed=None, corpus_root=None,
            game_cpu_list=str(environment["game_cpu_list"]),
            controller_cpu_list=str(environment["controller_cpu_list"]),
            **_priority_arguments(environment),
        )
        assert run_dir is None
        audited = _audit_evaluation(
            report=report, artifact=artifact, state_path=policy_state,
            role="candidate", environment=environment,
        )
        audited["trial"] = trial
        state["canary"].append(audited)
        _atomic_json(state_path, state)
        if not audited["passed"]:
            break
    canary_valid = (
        len(state["canary"]) == len(canary_schedule)
        and all(row["passed"] for row in state["canary"])
        and any(int(row["interventions"]) > 0 for row in state["canary"])
    )
    if not canary_valid:
        state.update({"status": "complete", "decision": {
            "verdict": "canary-rejected",
            "reason": "runtime gate or treatment exposure failed",
        }})
        _atomic_json(state_path, state)
        print(json.dumps(state["decision"], sort_keys=True))
        return 0
    evaluation = contract["evaluation"]["schedule"]
    state["status"] = "evaluation"
    _atomic_json(state_path, state)
    for row in evaluation[len(state["evaluation"]):]:
        trial = int(row["trial"])
        role = str(row["role"])
        policy_state = states / f"evaluation-{trial:02d}-{role}.json"
        if not policy_state.is_file():
            _atomic_json(policy_state, export_round_state(
                candidate_path=candidate, smoke_path=audit,
                registry_path=REGISTRY, scorer_path=WINE_SCORER,
                contract_path=contract_path,
                mode="active" if role == "candidate" else "shadow",
                policy_seed=int(row["policy_seed"]),
            ))
        artifact = root / "evaluation" / f"trial-{trial:02d}-{role}"
        report, run_dir = complete_run(
            artifact_dir=artifact, worker=worker, stage=6,
            policy_plugin=ACTOR_PLUGIN, policy_state=policy_state,
            scorer=WINE_SCORER, rng_seed=None, corpus_root=None,
            game_cpu_list=str(environment["game_cpu_list"]),
            controller_cpu_list=str(environment["controller_cpu_list"]),
            **_priority_arguments(environment),
        )
        assert run_dir is None
        audited = _audit_evaluation(
            report=report, artifact=artifact, state_path=policy_state,
            role=role, environment=environment,
        )
        audited.update({"trial": trial, "block": int(row["block"])})
        state["evaluation"].append(audited)
        _atomic_json(state_path, state)
        print(json.dumps({"evaluation_completed": audited}, sort_keys=True), flush=True)
        if not audited["passed"]:
            break
    decision = _paired_verdict(state["evaluation"])
    state.update({"status": "complete", "decision": decision})
    _atomic_json(state_path, state)
    _atomic_json(root / "result.json", {
        "schema": "autonomous-generation-6-round-result-v1",
        "contract_sha256": contract_sha,
        "training_registry_sha256": registry_sha,
        "collection": state["collection"],
        "offline": state["offline"],
        "canary": state["canary"],
        "evaluation": state["evaluation"],
        "decision": decision,
    })
    print(json.dumps(decision, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY / "artifacts/autonomous-generation-6-round-1",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
