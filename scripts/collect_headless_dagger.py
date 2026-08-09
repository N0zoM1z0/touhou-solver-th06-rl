#!/usr/bin/env python3
"""Roll out a distilled ranker and retain offline-teacher labels on its states."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

try:
    from train_headless_teacher import Decision, Encoder, FEATURE_NAMES, candidate_features
except ModuleNotFoundError:  # Imported as scripts.collect_headless_dagger in tests.
    from scripts.train_headless_teacher import (
        Decision,
        Encoder,
        FEATURE_NAMES,
        candidate_features,
    )
from th06_rl.headless import HeadlessClient, HeadlessScope
from th06_rl.headless_corpus import (
    BehaviorDecision,
    CompactHeadlessCorpusWriter,
    NativeOfflineTeacher,
    TeacherDecision,
    build_transition,
    compact_candidate_records,
    compact_state_features,
    source_context_id,
)
from th06_rl.headless_geometry import (
    HARD_HORIZON,
    HEADLESS_DELIVERY_CONTRACT,
    HEADLESS_DELIVERY_DELAYS,
    KINEMATICS,
    HeadlessAuthorityUnavailable,
    action_from_input,
    certify_lowered_headless_actions,
    lower_headless_hard_hazards,
    lower_headless_hazards,
)
from th06_rl.native import NativeCertifiedAction, NativeKernel


POLICY_NAME = "distilled-ranker-dagger-v1"


def _benchmark_release_behavior() -> BehaviorDecision:
    return BehaviorDecision(
        selected_action="stay_fast",
        probability=1.0,
        teacher=TeacherDecision(
            action="stay_fast",
            kind="benchmark-authority-release",
            effort_horizon=0,
            surviving_actions=(),
        ),
        policy="benchmark-authority-release-v1",
    )


def _benchmark_ranker_decision(
    selected_action: str,
    certified: tuple[NativeCertifiedAction, ...],
) -> TeacherDecision:
    """Describe an evaluation-only ranker action without running the teacher."""
    return TeacherDecision(
        action=selected_action,
        kind="benchmark-ranker-only",
        effort_horizon=0,
        surviving_actions=tuple(item.action.name for item in certified),
    )


def source_compatible(
    allowed: list[dict[str, Any]],
    runtime: dict[str, Any],
) -> bool:
    """Require an exact clean commit+binary pair, never a loose revision range."""
    return runtime.get("clean") is True and any(
        source.get("clean") is True
        and source.get("commit") == runtime.get("commit")
        and source.get("binary_sha256") == runtime.get("binary_sha256")
        for source in allowed
    )


def borda_consensus(actions: list[str], member_scores: list[list[float]]) -> str:
    """Select a calibration-free consensus inside one native-safe action set."""
    if (
        not actions
        or not member_scores
        or any(len(scores) != len(actions) for scores in member_scores)
    ):
        raise ValueError("Borda consensus requires complete nonempty member rankings")
    points = {action: 0 for action in actions}
    worst_rank = {action: 0 for action in actions}
    for scores in member_scores:
        ranked = sorted(
            zip(actions, scores, strict=True),
            key=lambda item: (float(item[1]), item[0]),
            reverse=True,
        )
        for rank, (action, _) in enumerate(ranked):
            points[action] += len(actions) - rank - 1
            worst_rank[action] = max(worst_rank[action], rank)
    return max(actions, key=lambda action: (points[action], -worst_rank[action], action))


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
        raw_members = artifact.get("ensemble_members")
        members = raw_members if isinstance(raw_members, list) else [artifact]
        if not members:
            raise ValueError("ranker ensemble has no members")
        self.scope = artifact["scope"]
        self.headless_source = artifact["headless_source"]
        self.compatible_headless_sources = artifact.get(
            "compatible_headless_sources",
            [self.headless_source],
        )
        self.native_delivery_contract = artifact.get(
            "native_delivery_contract",
            "legacy-unspecified-v0",
        )
        self.native_delivery_delays = artifact.get("native_delivery_delays", [])
        if self.native_delivery_contract not in {
            "legacy-unspecified-v0",
            HEADLESS_DELIVERY_CONTRACT,
        } or (
            self.native_delivery_contract == HEADLESS_DELIVERY_CONTRACT
            and self.native_delivery_delays != list(HEADLESS_DELIVERY_DELAYS)
        ):
            raise ValueError("ranker uses an incompatible delivery contract")
        self.members = []
        for member in members:
            model = member["model"]
            stored_features = member.get("feature_names")
            if stored_features is None:
                feature_count = int(model.booster_.num_feature())
                if feature_count > len(FEATURE_NAMES):
                    raise ValueError("ranker uses an unsupported feature schema")
                stored_features = FEATURE_NAMES[:feature_count]
            encoder = Encoder([], feature_names=stored_features)
            encoder.categories = {
                name: {value: index for index, value in enumerate(values)}
                for name, values in member["categories"].items()
            }
            self.members.append((model, encoder))

    def rank(
        self,
        observation: dict[str, Any],
        certified: tuple[NativeCertifiedAction, ...],
        *,
        sequence: int,
        seed: int,
        profiles=(),
    ) -> str:
        state = compact_state_features(observation)
        candidates = tuple(compact_candidate_records(
            certified,
            selected="",
            teacher="",
            profiles=profiles,
        ))
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
        actions = [str(candidate["action"]) for candidate in candidates]
        member_scores = [
            [float(score) for score in model.booster_.predict(
                encoder.encode(features),
                num_threads=self.threads,
            )]
            for model, encoder in self.members
        ]
        return borda_consensus(actions, member_scores)


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
    continue_after_hit: bool = False,
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
    if not source_compatible(ranker.compatible_headless_sources, provenance):
        raise ValueError("ranker and runtime do not share an exact compatible source build")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = output_root / (
        f"{timestamp}-dagger-d{scope.difficulty}-c{scope.character}-"
        f"s{scope.shot_type}-stage{scope.stage}-seed{seed}"
    )
    writer = CompactHeadlessCorpusWriter(run, anchor_stride=anchor_stride)
    model_sha256 = _sha256(model_path)
    (run / "run-intent.json").write_text(json.dumps({
        "schema": "th06-rl-headless-ranker-run-intent-v1",
        "scope": expected_scope,
        "initial_seed": seed,
        "continue_after_hit": continue_after_hit,
        "ranker": {"path": model_path.name, "sha256": model_sha256},
        "source": provenance,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    kernel = NativeKernel()
    teacher = (
        None
        if continue_after_hit
        else NativeOfflineTeacher(kernel=kernel, horizon=teacher_horizon)
    )
    termination_reason = "generator-error"
    authority_failure: str | None = None
    authority_failure_events = 0
    authority_failure_reasons: Counter[str] = Counter()
    benchmark_forced_actions = 0
    physical_hits = 0
    physical_hit_ticks: list[int] = []
    terminal: dict[str, Any] | None = None
    try:
        client = HeadlessClient(
            binary=binary,
            game_directory=game_directory,
            scope=scope,
            seed=seed,
            max_ticks=max_ticks,
            continue_after_hit=continue_after_hit,
        )
        try:
            observation = client.start()
            writer.anchor(observation, sequence=0, role="initial", force=True)
            sequence = 0
            while observation.get("terminal_reason") is None:
                benchmark_forced = False
                try:
                    hard_hazards = lower_headless_hard_hazards(
                        observation,
                        HARD_HORIZON,
                    )
                    prepared_hard = kernel.prepare_hazards(hard_hazards)
                    certified = certify_lowered_headless_actions(
                        observation,
                        prepared_hard,
                        kernel=kernel,
                    )
                    if not certified:
                        raise HeadlessAuthorityUnavailable("headless native safe set is empty")
                    profile_hazards = lower_headless_hazards(
                        observation,
                        teacher_horizon,
                    )
                    player = observation["player"]
                    profiles = kernel.profile_actions(
                        x=float(player["x"]),
                        y=float(player["y"]),
                        half_width=float(player["half_width"]),
                        half_height=float(player["half_height"]),
                        kinematics=KINEMATICS,
                        current_action=action_from_input(int(observation["input"])),
                        hazards=profile_hazards,
                        candidates=tuple(item.action for item in certified),
                    )
                    selected = ranker.rank(
                        observation,
                        certified,
                        sequence=sequence,
                        seed=seed,
                        profiles=profiles,
                    )
                    if teacher is None:
                        teacher_decision = _benchmark_ranker_decision(selected, certified)
                    else:
                        teacher_decision = teacher.rank(
                            observation,
                            certified,
                            hazards=profile_hazards,
                        )
                    issue = certify_lowered_headless_actions(
                        observation,
                        prepared_hard,
                        kernel=kernel,
                    )
                    if selected not in {item.action.name for item in issue}:
                        raise HeadlessAuthorityUnavailable("ranker failed the immediate issue gate")
                except HeadlessAuthorityUnavailable as error:
                    if continue_after_hit:
                        reason = str(error)
                        authority_failure_events += 1
                        authority_failure_reasons[reason] += 1
                        benchmark_forced_actions += 1
                        benchmark_forced = True
                        certified = ()
                        profiles = ()
                        behavior = _benchmark_release_behavior()
                    else:
                        authority_failure = str(error)
                        termination_reason = "authority-failure"
                        writer.anchor(
                            observation,
                            sequence=sequence,
                            role="authority-failure",
                            force=True,
                        )
                        break
                else:
                    behavior = BehaviorDecision(
                        selected_action=selected,
                        probability=1.0,
                        teacher=teacher_decision,
                        policy=POLICY_NAME,
                    )
                next_observation = client.step(behavior.selected_action)
                transition = build_transition(
                    sequence=sequence,
                    observation=observation,
                    next_observation=next_observation,
                    certified=certified,
                    behavior=behavior,
                    epsilon=0.0,
                    profiles=profiles,
                    benchmark_forced_action=benchmark_forced,
                )
                deaths_delta = max(int(transition["outcome_terms"]["deaths_delta"]), 0)
                physical_hits += deaths_delta
                physical_hit_ticks.extend([int(transition["next_tick"])] * deaths_delta)
                if transition["outcome_terms"]["bombs_used_delta"] != 0:
                    raise HeadlessAuthorityUnavailable("Bomb use appeared in ranker rollout")
                writer.transition(transition)
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
        "training_eligible": writer.transition_count > 0 and not continue_after_hit,
        "scope": expected_scope,
        "initial_seed": seed,
        "behavior_policy": POLICY_NAME,
        "behavior_epsilon": 0.0,
        "teacher_horizon": teacher_horizon,
        "teacher_labels_recorded": teacher is not None,
        "native_gate_horizon": HARD_HORIZON,
        "native_delivery_contract": HEADLESS_DELIVERY_CONTRACT,
        "native_delivery_delays": list(HEADLESS_DELIVERY_DELAYS),
        "anchor_stride": anchor_stride,
        "termination_reason": termination_reason,
        "authority_failure": authority_failure,
        "authority_failure_events": authority_failure_events,
        "authority_failure_reasons": dict(sorted(authority_failure_reasons.items())),
        "benchmark_forced_actions": benchmark_forced_actions,
        "continue_after_hit": continue_after_hit,
        "physical_hit": physical_hits > 0,
        "physical_hits": physical_hits,
        "physical_hit_ticks": physical_hit_ticks,
        "nmnb_stage_clear": bool(
            terminal is not None
            and termination_reason in {"chain-exit-success", "stage-clear-success"}
            and int(terminal["deaths"]) == 0
            and int(terminal["bombs_used"]) == 0
            and authority_failure_events == 0
            and benchmark_forced_actions == 0
        ),
        "effective_transition_ratio": 1.0 if writer.transition_count else 0.0,
        "selected_actions_native_legal_ratio": 1.0 if writer.transition_count else 0.0,
        "source": provenance,
        "ranker": {"path": model_path.name, "sha256": model_sha256},
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
    parser.add_argument("--continue-after-hit", action="store_true")
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
    if args.max_ticks < 0 or min(args.anchor_stride, args.teacher_horizon) <= 0:
        parser.error("max ticks must be nonnegative; anchor and teacher bounds must be positive")
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
        continue_after_hit=args.continue_after_hit,
    )
    print(json.dumps({"manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
