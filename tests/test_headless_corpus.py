from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.audit_headless_corpus import audit_run
from th06_rl.headless_corpus import (
    CompactHeadlessCorpusWriter,
    EpsilonTeacherBehavior,
    TeacherDecision,
    build_transition,
    canonical_observation_sha256,
    source_context_id,
)
from th06_rl.native import ACTIONS, NativeCertifiedAction


BY_NAME = {action.name: action for action in ACTIONS}


def observation(*, tick: int = 10, terminal: str | None = None) -> dict[str, object]:
    return {
        "schema": "th06-headless-observation-v2",
        "tick": tick,
        "terminal_reason": terminal,
        "scope": {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 6},
        "initial_seed": 7,
        "game_frame": tick,
        "input": 0x04,
        "player": {
            "x": 192.0,
            "y": 384.0,
            "state": 0,
            "half_width": 1.25,
            "half_height": 1.25,
            "focused": True,
        },
        "lives": 2,
        "bombs": 3,
        "score": 1000,
        "power": 64,
        "rank": 32,
        "deaths": int(terminal == "physical-hit"),
        "bombs_used": 0,
        "graze": 4,
        "source_context": {
            "timeline_time": tick,
            "next": {"time": 440, "arg0": 1, "opcode": 0, "size": 28},
        },
        "bullets": [],
        "lasers": [],
        "enemies": [],
    }


def certified() -> tuple[NativeCertifiedAction, ...]:
    return (
        NativeCertifiedAction(BY_NAME["stay"], 10.0, 192.0, 384.0),
        NativeCertifiedAction(BY_NAME["left"], 8.0, 184.0, 384.0),
    )


def test_source_context_uses_automatic_timeline_or_boss_identity() -> None:
    value = observation()
    assert source_context_id(value) == "timeline:440/0/1"

    value["enemies"] = [{"slot": 3, "boss": True, "ecl_sub": 17}]
    assert source_context_id(value) == "boss:3/17"


def test_epsilon_teacher_logs_the_marginal_behavior_probability() -> None:
    teacher = TeacherDecision("stay", "test", 12, ("stay", "left"))
    greedy = EpsilonTeacherBehavior(epsilon=0.0, seed=1).select(teacher, certified())
    assert greedy.selected_action == "stay"
    assert greedy.probability == 1.0

    exploring = EpsilonTeacherBehavior(epsilon=1.0, seed=2).select(teacher, certified())
    assert exploring.probability == 0.5


def test_transition_has_a_factual_successor_and_native_legal_action() -> None:
    current = observation()
    following = observation(tick=11, terminal="physical-hit")
    behavior = EpsilonTeacherBehavior(epsilon=0.0, seed=1).select(
        TeacherDecision("stay", "test", 4, ("stay",)),
        certified(),
    )

    transition = build_transition(
        sequence=0,
        observation=current,
        next_observation=following,
        certified=certified(),
        behavior=behavior,
        epsilon=0.0,
    )

    assert transition["behavior"]["selected_action"] in transition["legal_actions"]
    assert transition["observation_sha256"] == canonical_observation_sha256(current)
    assert transition["next_observation_sha256"] == canonical_observation_sha256(following)
    assert transition["outcome_terms"]["physical_hit"] is True


def test_compact_writer_commits_gzip_shards_and_manifest(tmp_path: Path) -> None:
    writer = CompactHeadlessCorpusWriter(tmp_path / "run", anchor_stride=2)
    current = observation()
    following = observation(tick=11, terminal="physical-hit")
    writer.anchor(current, sequence=0, role="initial", force=True)
    writer.anchor(following, sequence=1, role="periodic")
    writer.anchor(following, sequence=1, role="terminal", force=True)
    writer.transition({"schema": "test", "sequence": 0})
    manifest_path = writer.close({"transaction_complete": True})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["transition_count"] == 1
    assert manifest["anchor_count"] == 2
    with gzip.open(tmp_path / "run/transitions.jsonl.gz", "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["sequence"] == 0
    with gzip.open(tmp_path / "run/anchors.jsonl.gz", "rt", encoding="utf-8") as stream:
        assert len(stream.readlines()) == 2


def test_independent_auditor_rechecks_digest_chain_and_file_hashes(tmp_path: Path) -> None:
    run = tmp_path / "audited"
    writer = CompactHeadlessCorpusWriter(run, anchor_stride=120)
    current = observation()
    following = observation(tick=11, terminal="physical-hit")
    behavior = EpsilonTeacherBehavior(epsilon=0.0, seed=1).select(
        TeacherDecision("stay", "test", 4, ("stay",)),
        certified(),
    )
    transition = build_transition(
        sequence=0,
        observation=current,
        next_observation=following,
        certified=certified(),
        behavior=behavior,
        epsilon=0.0,
    )
    writer.anchor(current, sequence=0, role="initial", force=True)
    writer.transition(transition)
    writer.anchor(following, sequence=1, role="terminal", force=True)
    writer.close({"termination_reason": "physical-hit"})

    result = audit_run(run)
    assert result["valid"] is True
    assert result["factual_successor_rows"] == 1
    assert result["native_legal_rows"] == 1

    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["transitions"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    damaged = audit_run(run)
    assert damaged["valid"] is False
    assert "transitions SHA-256 mismatch" in damaged["errors"]
