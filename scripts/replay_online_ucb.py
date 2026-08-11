#!/usr/bin/env python3
"""Hot-start the active online UCB from faithful physical corpus actions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

try:
    import msgspec
except ImportError:  # pragma: no cover - slower portable fallback
    msgspec = None

from th06_rl.policies.adaptive import AdaptivePolicy
from th06_rl.policy_api import PolicyContext, PolicyFailureEvent, PolicyOutcome


if msgspec is not None:
    class _Scope(msgspec.Struct):
        difficulty: int
        character: int
        shot_type: int
        stage: int
        phase_id: str


    class _FrameSnapshot(msgspec.Struct):
        frame: int
        x: float
        y: float
        current_power: int = 0
        live_bullet_count: int = 0
        laser_count: int = 0


    class _FrameDecision(msgspec.Struct):
        current_action: str | None = None
        hard_actions: tuple[
            tuple[str, float | None, float, float], ...
        ] = ()
        baseline_action: str | None = None
        locally_admissible_actions: tuple[str, ...] = ()
        proposed_action: str | None = None
        published_action: str | None = None


    class _FrameRow(msgspec.Struct):
        sequence: int
        scope: _Scope
        snapshot: _FrameSnapshot
        decision: _FrameDecision


    class _Outcome(msgspec.Struct):
        elapsed_frames: int
        life_lost: bool
        bomb_used: bool
        control_dead_end: bool
        authority_lost: bool
        phase_changed: bool
        hard_count_after: int
        player_x_after: float
        player_y_after: float


    class _ReplayContext(msgspec.Struct):
        current_action: str | None = None
        hard_admissible_actions: tuple[str, ...] = ()
        phase_elapsed_frames: int = 0
        player_x: float = 0.0
        player_y: float = 0.0
        power: int = 0
        bullet_count: int = 0
        laser_count: int = 0
        hard_action_count: int = 0


    class _TransitionRow(msgspec.Struct):
        sequence: int
        snapshot_ref: str
        next_snapshot_ref: str
        scope: _Scope
        next_scope: _Scope
        legal_actions: tuple[str, ...]
        baseline_action: str | None
        proposed_action: str | None
        published_action: str | None
        outcome_terms: _Outcome
        learning_eligible: bool
        policy_context: _ReplayContext | None = None


    _FRAME_DECODER = msgspec.json.Decoder(_FrameRow)
    _TRANSITION_DECODER = msgspec.json.Decoder(_TransitionRow)
else:
    _FRAME_DECODER = None
    _TRANSITION_DECODER = None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_paths(
    run_dir: Path,
    manifest: dict[str, object],
    stream: str,
) -> list[Path]:
    rows = sorted(
        (
            item for item in manifest.get("shards", ())
            if isinstance(item, dict) and item.get("stream") == stream
        ),
        key=lambda item: int(item["first_sequence"]),
    )
    paths = [run_dir / str(item["path"]) for item in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest references missing shards: {missing}")
    return paths


def _rows(paths: list[Path], decoder):
    for path in paths:
        with gzip.open(path, "rb") as source:
            for line in source:
                yield decoder.decode(line) if decoder is not None else json.loads(line)


def _value(row, name: str):
    return row[name] if isinstance(row, dict) else getattr(row, name)


def _scope_tuple(scope) -> tuple[int, int, int, int]:
    return tuple(int(_value(scope, name)) for name in (
        "difficulty", "character", "shot_type", "stage"
    ))


def _snapshot_frame(reference: str) -> int:
    marker = reference.rsplit(":f", 1)
    if len(marker) != 2:
        raise ValueError(f"snapshot reference has no frame: {reference}")
    return int(marker[1])


def _selected_runs(args: argparse.Namespace):
    selected = []
    excluded: dict[str, int] = {}
    for run_dir in sorted(args.corpus_root.iterdir()):
        run_path = run_dir / "run.json"
        manifest_path = run_dir / "manifest.json"
        if not run_dir.is_dir() or not run_path.is_file():
            continue
        if args.run_id and run_dir.name not in args.run_id:
            excluded["other-run-id"] = excluded.get("other-run-id", 0) + 1
            continue
        if not manifest_path.is_file():
            excluded["missing-manifest"] = excluded.get("missing-manifest", 0) + 1
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = run.get("metadata", {})
        expected = (
            args.difficulty,
            args.character,
            args.shot_type,
            args.stage,
        )
        actual = tuple(int(metadata.get(name, -1)) for name in (
            "difficulty", "character", "shot_type", "stage"
        ))
        if actual != expected:
            excluded["other-scope"] = excluded.get("other-scope", 0) + 1
            continue
        if not (
            manifest.get("complete") is True
            and manifest.get("stage_trajectory_complete") is True
            and manifest.get("run_outcome", {}).get("stage_completed") is True
        ):
            excluded["incomplete-stage"] = excluded.get("incomplete-stage", 0) + 1
            continue
        code_commit = str(metadata.get("code_commit", ""))
        if args.code_commit and not code_commit.startswith(args.code_commit):
            excluded["other-code"] = excluded.get("other-code", 0) + 1
            continue
        native = str(metadata.get("native_kernel_sha256", ""))
        if args.native_kernel_sha256 and native != args.native_kernel_sha256:
            excluded["other-native"] = excluded.get("other-native", 0) + 1
            continue
        selected.append((run_dir, run, manifest))
    return selected, excluded


def _context_from_frame(
    row,
    transition,
    phase_start: int,
) -> PolicyContext:
    snapshot = _value(row, "snapshot")
    decision = _value(row, "decision")
    scope = _value(row, "scope")
    hard = tuple(str(item[0]) for item in _value(decision, "hard_actions"))
    return PolicyContext(
        frame=int(_value(snapshot, "frame")),
        scope=_scope_tuple(scope),
        source_context=str(_value(scope, "phase_id")),
        baseline_action=str(_value(decision, "baseline_action")),
        locally_admissible_actions=tuple(
            str(item) for item in _value(transition, "legal_actions")
        ),
        player_x=float(_value(snapshot, "x")),
        player_y=float(_value(snapshot, "y")),
        power=int(_value(snapshot, "current_power")),
        bullet_count=int(_value(snapshot, "live_bullet_count")),
        laser_count=int(_value(snapshot, "laser_count")),
        hard_action_count=len(hard),
        exploration_rate=0.0,
        current_action=str(_value(decision, "current_action")),
        hard_admissible_actions=hard,
        phase_elapsed_frames=max(0, int(_value(snapshot, "frame")) - phase_start),
    )


def _context_from_transition(row) -> PolicyContext:
    replay = _value(row, "policy_context")
    scope = _value(row, "scope")
    if replay is None:
        raise ValueError("transition has no compact policy context")
    return PolicyContext(
        frame=_snapshot_frame(str(_value(row, "snapshot_ref"))),
        scope=_scope_tuple(scope),
        source_context=str(_value(scope, "phase_id")),
        baseline_action=str(_value(row, "baseline_action")),
        locally_admissible_actions=tuple(
            str(item) for item in _value(row, "legal_actions")
        ),
        player_x=float(_value(replay, "player_x")),
        player_y=float(_value(replay, "player_y")),
        power=int(_value(replay, "power")),
        bullet_count=int(_value(replay, "bullet_count")),
        laser_count=int(_value(replay, "laser_count")),
        hard_action_count=int(_value(replay, "hard_action_count")),
        exploration_rate=0.0,
        current_action=str(_value(replay, "current_action")),
        hard_admissible_actions=tuple(
            str(item) for item in _value(replay, "hard_admissible_actions")
        ),
        phase_elapsed_frames=int(_value(replay, "phase_elapsed_frames")),
    )


def _replay_run(
    policy: AdaptivePolicy,
    run_dir: Path,
    run: dict[str, object],
    manifest: dict[str, object],
) -> dict[str, int]:
    transition_paths = _stream_paths(run_dir, manifest, "transitions")
    compact = run.get("schemas", {}).get("transition") in (
        "th06-rl-transition-v5",
        "th06-rl-transition-v6",
    )
    frame_rows = (
        None
        if compact
        else iter(_rows(
            _stream_paths(run_dir, manifest, "frames"),
            _FRAME_DECODER,
        ))
    )
    phase = None
    phase_start = None
    decisions = 0
    trained = 0
    hits = 0
    policy.reset_credit_episode()
    for transition in _rows(transition_paths, _TRANSITION_DECODER):
        frame_row = None if frame_rows is None else next(frame_rows)
        if frame_row is not None:
            if int(_value(frame_row, "sequence")) != int(
                _value(transition, "sequence")
            ):
                raise ValueError(f"frame/transition sequence mismatch in {run_dir}")
            snapshot = _value(frame_row, "snapshot")
            scope = _value(frame_row, "scope")
            current_phase = str(_value(scope, "phase_id"))
            frame = int(_value(snapshot, "frame"))
            if current_phase != phase:
                phase = current_phase
                phase_start = frame
            assert phase_start is not None
            context = _context_from_frame(frame_row, transition, phase_start)
        else:
            context = _context_from_transition(transition)

        proposed = _value(transition, "proposed_action")
        published = _value(transition, "published_action")
        outcome = _value(transition, "outcome_terms")
        if proposed is not None:
            action = str(proposed)
            policy.replay_logged_decision(context, action)
            decisions += 1
            eligible = bool(_value(transition, "learning_eligible"))
            policy.observe(PolicyOutcome(
                frame=context.frame,
                scope=context.scope,
                source_context=context.source_context,
                action=action,
                published=published == proposed,
                elapsed_frames=int(_value(outcome, "elapsed_frames")),
                life_lost=bool(_value(outcome, "life_lost")),
                bomb_used=bool(_value(outcome, "bomb_used")),
                control_dead_end=bool(_value(outcome, "control_dead_end")),
                authority_lost=bool(_value(outcome, "authority_lost")),
                phase_changed=bool(_value(outcome, "phase_changed")),
                next_hard_action_count=int(_value(outcome, "hard_count_after")),
                next_player_x=float(_value(outcome, "player_x_after")),
                next_player_y=float(_value(outcome, "player_y_after")),
                learning_eligible=eligible,
            ))
            trained += int(eligible and published == proposed)
        if bool(_value(outcome, "life_lost")):
            next_scope = _value(transition, "next_scope")
            policy.observe_failure(PolicyFailureEvent(
                frame=_snapshot_frame(str(_value(transition, "next_snapshot_ref"))),
                scope=_scope_tuple(next_scope),
                source_context=str(_value(next_scope, "phase_id")),
                kind="physical-hit",
            ))
            hits += 1
    policy.reset_credit_episode()
    return {"decisions": decisions, "trained": trained, "hits": hits}


def replay(args: argparse.Namespace) -> dict[str, object]:
    selected, excluded = _selected_runs(args)
    if not selected:
        raise RuntimeError("no complete corpus runs matched the requested scope")
    if args.output.exists() and not args.replace:
        raise FileExistsError(
            f"refusing to replace existing checkpoint without --replace: {args.output}"
        )
    policy = AdaptivePolicy()
    started = time.monotonic()
    totals = {"decisions": 0, "trained": 0, "hits": 0}
    run_rows = []
    for index, (run_dir, run, manifest) in enumerate(selected, 1):
        row = _replay_run(policy, run_dir, run, manifest)
        run_rows.append({"run_id": run_dir.name, **row})
        for name in totals:
            totals[name] += row[name]
        print(
            f"[{index}/{len(selected)}] {run_dir.name} "
            f"trained={row['trained']} hits={row['hits']}",
            flush=True,
        )
    _atomic_json(args.output, policy.export_state())
    report = {
        "schema": "th06-rl-online-ucb-corpus-replay-v1",
        "scope": {
            "difficulty": args.difficulty,
            "character": args.character,
            "shot_type": args.shot_type,
            "stage": args.stage,
        },
        "filters": {
            "code_commit": args.code_commit,
            "native_kernel_sha256": args.native_kernel_sha256,
            "run_ids": args.run_id,
            "complete_stage_only": True,
        },
        "selected_runs": len(selected),
        "excluded_runs": excluded,
        "totals": totals,
        "policy_metrics": policy.metrics(),
        "checkpoint": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": _sha256(args.output),
        },
        "elapsed_seconds": time.monotonic() - started,
        "runs": run_rows,
    }
    if args.report is not None:
        _atomic_json(args.report, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=Path("artifacts/corpus"))
    parser.add_argument("--difficulty", type=int, required=True)
    parser.add_argument("--character", type=int, default=0)
    parser.add_argument("--shot-type", type=int, default=0)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--native-kernel-sha256", default=None)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = replay(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
