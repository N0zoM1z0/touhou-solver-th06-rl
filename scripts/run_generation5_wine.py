#!/usr/bin/env python3
"""Generic isolated-worker Wine primitive for the Generation-5 curriculum."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_autonomous_learning import _validate_retail_report  # noqa: E402
from scripts.run_autonomous_learning_v3 import (  # noqa: E402
    _archive_incomplete,
    _corpus_runs,
    _trace_is_clean,
)
from th06_rl.advantage_learning import _object, _validate_run  # noqa: E402
from th06_rl.audited_option_loader import load_audited_option_episode  # noqa: E402
from th06_rl.process_priority import parse_cpu_list, validate_nice  # noqa: E402
from th06_rl.sequential_learning import TRANSITION_SCHEMA  # noqa: E402


DIFFICULTY = "lunatic"
MAXIMUM_INFRA_ATTEMPTS = 3


def _validate_complete_run(
    *,
    artifact_dir: Path,
    worker: dict[str, object],
    stage: int,
    rng_seed: int | None,
    corpus_root: Path | None,
    game_cpu_list: str | None = None,
    controller_cpu_list: str | None = None,
    game_nice: int | None = None,
    controller_nice: int | None = None,
) -> tuple[dict[str, object], Path | None]:
    report = _object(artifact_dir / "report.json")
    _validate_retail_report(
        report,
        mode=(
            (
                "fixed-rng-complete-stage-training"
                if rng_seed is not None
                else "natural-rng-complete-stage-training"
            )
            if corpus_root is not None else "hit-continuation-benchmark"
        ),
        diagnostic_rng_seed=rng_seed,
        full_stage=stage,
    )
    executable = Path(str(report.get("retail_executable", ""))).resolve()
    if (
        report.get("difficulty") != DIFFICULTY
        or report.get("practice_stage") != stage
        or report.get("display") != worker["display"]
        or Path(str(report.get("wine_prefix", ""))).resolve()
        != Path(str(worker["wine_prefix"])).resolve()
        or executable.parent != Path(str(worker["game_dir"])).resolve()
        or not _trace_is_clean(report)
    ):
        raise RuntimeError("isolated Wine complete-Stage provenance failed")
    priority_values = (
        ("game", game_cpu_list, game_nice),
        ("controller", controller_cpu_list, controller_nice),
    )
    if any(value is not None for _role, _cpus, value in priority_values):
        if any(
            value is None or cpus is None
            for _role, cpus, value in priority_values
        ):
            raise ValueError("complete-Stage priority contract is incomplete")
        for role, cpu_list, nice in priority_values:
            assert cpu_list is not None and nice is not None
            validate_nice(nice)
            attestation = report.get(f"{role}_priority_attestation")
            expected_cpus = list(parse_cpu_list(cpu_list))
            if (
                report.get(f"{role}_nice") != nice
                or not isinstance(attestation, dict)
                or attestation.get("schema") != "bounded-wine-process-priority-v1"
                or attestation.get("scheduler") != "SCHED_OTHER"
                or attestation.get("nice") != nice
                or attestation.get("effective_nice") != nice
                or attestation.get("cpus") != expected_cpus
                or attestation.get("effective_cpus") != expected_cpus
                or attestation.get("uid") != attestation.get("target_uid")
                or attestation.get("gid") != attestation.get("target_gid")
            ):
                raise RuntimeError(
                    f"isolated Wine {role} priority attestation failed"
                )
    run_dir = None
    if corpus_root is not None:
        trace = report.get("trace")
        run_ids = trace.get("corpus_run_ids") if isinstance(trace, dict) else None
        if not isinstance(run_ids, list) or len(run_ids) != 1:
            raise RuntimeError("complete Stage report must bind one corpus run")
        run_dir = corpus_root / str(run_ids[0])
        _run, manifest = _validate_run(
            run_dir, transition_schema=TRANSITION_SCHEMA
        )
        outcome = manifest["run_outcome"]
        completion = report["controller_completion"]
        if int(outcome["physical_hits"]) != int(completion["physical_hits"]):
            raise RuntimeError("Wine report/corpus physical HIT count differs")
    return report, run_dir


def complete_run(
    *,
    artifact_dir: Path,
    worker: dict[str, object],
    stage: int,
    policy_plugin: Path,
    policy_state: Path,
    scorer: Path | None,
    rng_seed: int | None,
    corpus_root: Path | None,
    game_cpu_list: str | None = None,
    controller_cpu_list: str | None = None,
    game_nice: int | None = None,
    controller_nice: int | None = None,
) -> tuple[dict[str, object], Path | None]:
    """Run or resume one normal-speed original-Wine complete Stage.

    A completed attempt whose strict report/corpus audit fails is never made
    learner-visible.  Preserve its artifact beside the scheduled episode and
    retry the *same* frozen row a bounded number of times.  This is an infra
    retry, not outcome-conditioned sampling: RNG, policy state, worker, stage,
    and scorer remain unchanged.
    """
    if not 1 <= stage <= 6:
        raise ValueError("TH06 Practice stage must be between one and six")
    if (artifact_dir / "report.json").is_file():
        try:
            return _validate_complete_run(
                artifact_dir=artifact_dir,
                worker=worker,
                stage=stage,
                rng_seed=rng_seed,
                corpus_root=corpus_root,
                game_cpu_list=game_cpu_list,
                controller_cpu_list=controller_cpu_list,
                game_nice=game_nice,
                controller_nice=controller_nice,
            )
        except (FileNotFoundError, TypeError, ValueError, RuntimeError):
            _archive_incomplete(artifact_dir)
    elif artifact_dir.exists():
        _archive_incomplete(artifact_dir)
    command = [
        sys.executable,
        str(REPOSITORY / "scripts/run_wine_retail.py"),
        "--game-dir", str(worker["game_dir"]),
        "--wine-prefix", str(worker["wine_prefix"]),
        "--display", str(worker["display"]),
        "--practice-stage", str(stage),
        "--difficulty", DIFFICULTY,
        "--artifact-dir", str(artifact_dir),
        "--policy-plugin", str(policy_plugin),
        "--policy-state", str(policy_state),
        "--immutable-policy",
        "--exploration-rate", "0",
    ]
    if scorer is not None:
        command.extend(("--policy-scorer-library", str(scorer)))
    if game_cpu_list is not None or controller_cpu_list is not None:
        if not game_cpu_list or not controller_cpu_list:
            raise ValueError("Wine CPU partitions must be provided together")
        command.extend((
            "--game-cpu-list", game_cpu_list,
            "--controller-cpu-list", controller_cpu_list,
        ))
    if game_nice is not None or controller_nice is not None:
        if game_nice is None or controller_nice is None:
            raise ValueError("Wine process nice values must be provided together")
        validate_nice(game_nice)
        validate_nice(controller_nice)
        command.extend((
            "--game-nice", str(game_nice),
            "--controller-nice", str(controller_nice),
        ))
    if corpus_root is not None:
        command.extend((
            "--complete-stage-training-corpus-root", str(corpus_root),
        ))
        if rng_seed is not None:
            command.extend(("--diagnostic-rng-seed", hex(rng_seed)))
    last_error: BaseException | None = None
    for attempt in range(1, MAXIMUM_INFRA_ATTEMPTS + 1):
        before = _corpus_runs(corpus_root) if corpus_root is not None else set()
        completed = subprocess.run(command, cwd=REPOSITORY, check=False)
        try:
            report, run_dir = _validate_complete_run(
                artifact_dir=artifact_dir,
                worker=worker,
                stage=stage,
                rng_seed=rng_seed,
                corpus_root=corpus_root,
                game_cpu_list=game_cpu_list,
                controller_cpu_list=controller_cpu_list,
                game_nice=game_nice,
                controller_nice=controller_nice,
            )
            if completed.returncode != int(report["controller_returncode"]):
                raise RuntimeError("outer and recorded Wine return codes differ")
            if corpus_root is not None:
                created = sorted(_corpus_runs(corpus_root) - before)
                if len(created) != 1 or run_dir != corpus_root / created[0]:
                    raise RuntimeError(
                        "complete Stage created/bound the wrong corpus run"
                    )
            return report, run_dir
        except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
            last_error = error
            if attempt == MAXIMUM_INFRA_ATTEMPTS:
                raise
            _archive_incomplete(artifact_dir)
            print(
                "strict Wine attempt audit failed; preserving attempt and "
                f"retrying frozen row ({attempt}/{MAXIMUM_INFRA_ATTEMPTS}): "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
    assert last_error is not None
    raise last_error


def normalized_option_sha256(run_dir: Path) -> str:
    """Hash factual semantics while ignoring per-run identifiers."""
    rows, _report = load_audited_option_episode(run_dir)
    normalized = []
    for row in rows:
        value = asdict(row)
        value.pop("episode_id")
        value.pop("option_id")
        normalized.append(value)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()
