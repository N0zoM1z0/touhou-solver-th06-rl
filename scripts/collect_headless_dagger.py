#!/usr/bin/env python3
"""Roll out a distilled ranker and retain offline-teacher labels on its states."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from train_headless_teacher import Decision, Encoder, candidate_features
from th06_rl.headless import HeadlessClient, HeadlessScope
from th06_rl.headless_corpus import (
    BehaviorDecision,
    CompactHeadlessCorpusWriter,
    NativeOfflineTeacher,
    build_transition,
    compact_candidate_records,
    compact_state_features,
    source_context_id,
)
from th06_rl.headless_geometry import (
    HARD_HORIZON,
    HeadlessAuthorityUnavailable,
    certify_lowered_headless_actions,
    lower_headless_hazards,
)
from th06_rl.native import NativeCertifiedAction, NativeKernel


POLICY_NAME = "distilled-ranker-dagger-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance(binary: Path) -> dict[str, Any]:
    source = binary.resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "clean": not dirty, "binary_sha256": _sha256(binary)}


class DistilledRanker:
    def __init__(self, artifact_path: Path, *, threads: int) -> None:
        import joblib

        self.path = artifact_path.resolve()
        self.threads = threads
        artifact = joblib.load(self.path)
        self.model = artifact["model"]
        self.scope = artifact["scope"]
        self.headless_source = artifact["headless_source"]
        self.encoder = Encoder([])
        self.encoder.categories = {
            name: {value: index for index, value in enumerate(values)}
            for name, values in artifact["categories"].items()
        }

    def rank(
        self,
        observation: dict[str, Any],
        certified: tuple[NativeCertifiedAction, ...],
        *,
        sequence: int,
        seed: int,
    ) -> str:
        state = compact_state_features(observation)
        candidates = tuple(compact_candidate_records(certified, selected="", teacher=""))
        decision = Decision(
            run="dagger-rollout",
            seed=seed,
            sequence=sequence,
            source_context=source_context_id(observation),
            state=state,
            legal_actions=tuple(item.action.name for item in certified),
            candidates=candidates,
            teacher_action="",
            selected_action="",
        )
        features = [candidate_features(decision, candidate) for candidate in candidates]
        scores = self.model.booster_.predict(
            self.encoder.encode(features),
            num_threads=self.threads,
        )
        return str(max(
            zip(candidates, scores, strict=True),
            key=lambda item: (float(item[1]), str(item[0]["action"])),
        )[0]["action"])


def collect(
    *,
    binary: Path,
    game_directory: Path,
    output_root: Path,
    model_path: Path,
    scope: HeadlessScope,
    seed: int,
    max_ticks: int,
    anchor_stride: int,
    teacher_horizon: int,
    threads: int,
) -> Path:
    ranker = DistilledRanker(model_path, threads=threads)
    expected_scope = {
        "difficulty": scope.difficulty,
        "character": scope.character,
        "shot_type": scope.shot_type,
        "stage": scope.stage,
    }
    if ranker.scope != expected_scope:
        raise ValueError("ranker scope does not match requested rollout scope")
    provenance = _source_provenance(binary)
    if ranker.headless_source.get("commit") != provenance["commit"]:
        raise ValueError("ranker and runtime use different headless source revisions")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = output_root / (
        f"{timestamp}-dagger-d{scope.difficulty}-c{scope.character}-"
        f"s{scope.shot_type}-stage{scope.stage}-seed{seed}"
    )
    writer = CompactHeadlessCorpusWriter(run, anchor_stride=anchor_stride)
    kernel = NativeKernel()
    teacher = NativeOfflineTeacher(kernel=kernel, horizon=teacher_horizon)
    termination_reason = "generator-error"
    authority_failure: str | None = None
    terminal: dict[str, Any] | None = None
    try:
        client = HeadlessClient(
            binary=binary,
            game_directory=game_directory,
            scope=scope,
            seed=seed,
            max_ticks=max_ticks,
        )
        try:
            observation = client.start()
            writer.anchor(observation, sequence=0, role="initial", force=True)
            sequence = 0
            while observation.get("terminal_reason") is None:
                try:
                    hard_hazards = lower_headless_hazards(observation, HARD_HORIZON)
                    prepared_hard = kernel.prepare_hazards(hard_hazards)
                    certified = certify_lowered_headless_actions(
                        observation,
                        prepared_hard,
                        kernel=kernel,
                    )
                    if not certified:
                        raise HeadlessAuthorityUnavailable("headless native safe set is empty")
                    teacher_hazards = lower_headless_hazards(observation, teacher_horizon)
                    teacher_decision = teacher.rank(
                        observation,
                        certified,
                        hazards=teacher_hazards,
                    )
                    selected = ranker.rank(
                        observation,
                        certified,
                        sequence=sequence,
                        seed=seed,
                    )
                    issue = certify_lowered_headless_actions(
                        observation,
                        prepared_hard,
                        kernel=kernel,
                    )
                    if selected not in {item.action.name for item in issue}:
                        raise HeadlessAuthorityUnavailable("ranker failed the immediate issue gate")
                except HeadlessAuthorityUnavailable as error:
                    authority_failure = str(error)
                    termination_reason = "authority-failure"
                    writer.anchor(
                        observation,
                        sequence=sequence,
                        role="authority-failure",
                        force=True,
                    )
                    break

                behavior = BehaviorDecision(
                    selected_action=selected,
                    probability=1.0,
                    teacher=teacher_decision,
                    policy=POLICY_NAME,
                )
                next_observation = client.step(selected)
                writer.transition(build_transition(
                    sequence=sequence,
                    observation=observation,
                    next_observation=next_observation,
                    certified=certified,
                    behavior=behavior,
                    epsilon=0.0,
                ))
                sequence += 1
                writer.anchor(next_observation, sequence=sequence, role="periodic")
                observation = next_observation
                if observation.get("terminal_reason") is not None:
                    terminal = observation
                    termination_reason = str(observation["terminal_reason"])
                    writer.anchor(observation, sequence=sequence, role="terminal", force=True)
                    break
        finally:
            client.close()
    except Exception:
        writer.abort()
        raise
    return writer.close({
        "transaction_complete": True,
        "training_eligible": writer.transition_count > 0,
        "scope": expected_scope,
        "initial_seed": seed,
        "behavior_policy": POLICY_NAME,
        "behavior_epsilon": 0.0,
        "teacher_horizon": teacher_horizon,
        "native_gate_horizon": HARD_HORIZON,
        "anchor_stride": anchor_stride,
        "termination_reason": termination_reason,
        "authority_failure": authority_failure,
        "physical_hit": termination_reason == "physical-hit",
        "nmnb_stage_clear": bool(
            terminal is not None
            and termination_reason == "chain-exit-success"
            and int(terminal["deaths"]) == 0
            and int(terminal["bombs_used"]) == 0
        ),
        "effective_transition_ratio": 1.0 if writer.transition_count else 0.0,
        "selected_actions_native_legal_ratio": 1.0 if writer.transition_count else 0.0,
        "source": provenance,
        "ranker": {"path": model_path.name, "sha256": _sha256(model_path)},
    })


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-ticks", type=int, default=1200)
    parser.add_argument("--anchor-stride", type=int, default=120)
    parser.add_argument("--teacher-horizon", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--character", type=int, default=0)
    parser.add_argument("--shot-type", type=int, default=0)
    parser.add_argument("--stage", type=int, default=6)
    parser.add_argument(
        "--binary",
        type=Path,
        default=root / "reference/GensokyoClub-th06-portable/th06",
    )
    parser.add_argument(
        "--game-directory",
        type=Path,
        default=root / "reference/th06-game-original/th06",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "artifacts/headless-corpus-dagger",
    )
    args = parser.parse_args()
    if min(args.max_ticks, args.anchor_stride, args.teacher_horizon) <= 0:
        parser.error("tick, anchor, and teacher bounds must be positive")
    if not 1 <= args.threads <= 12:
        parser.error("threads must be in 1..12 on the shared VPS")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_THREAD_LIMIT"] = str(args.threads)
    manifest = collect(
        binary=args.binary.resolve(),
        game_directory=args.game_directory.resolve(),
        output_root=args.output_root.resolve(),
        model_path=args.model.resolve(),
        scope=HeadlessScope(
            args.difficulty,
            args.character,
            args.shot_type,
            args.stage,
        ),
        seed=args.seed,
        max_ticks=args.max_ticks,
        anchor_stride=args.anchor_stride,
        teacher_horizon=args.teacher_horizon,
        threads=args.threads,
    )
    print(json.dumps({"manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
