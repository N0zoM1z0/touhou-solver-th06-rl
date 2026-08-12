from __future__ import annotations

from th06_rl.advantage_learning import (
    POPULATION_MEMBERS,
    OptionStep,
    audit_wine_option_smoke,
    doubly_robust_advantages,
    fit_dr_option_advantage,
    run_causal_recovery_smoke,
)
from th06_rl.learning_features import tree_feature_names
from th06_rl.offline import ACTION_NAMES
from th06_rl.th06.learning_adapter import ACTION_FEATURE_NAMES, OBSERVATION_FEATURE_NAMES


def _episodes(names: tuple[str, ...]) -> list[OptionStep]:
    width = len(tree_feature_names(OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES))
    result = []
    for episode_index, episode in enumerate(names):
        for action_index, action in enumerate(ACTION_NAMES):
            factual = [0.0] * width
            factual[0] = episode_index / max(1, len(names) - 1)
            factual[len(OBSERVATION_FEATURE_NAMES)] = action_index / len(ACTION_NAMES)
            baseline = "stay"
            legal = (baseline,) if action == baseline else (baseline, action)
            candidates = []
            for candidate_index, _candidate in enumerate(legal):
                vector = factual.copy()
                vector[-1] = candidate_index
                candidates.append(tuple(vector))
            result.append(OptionStep(
                episode_id=episode,
                option_id=f"{episode}:{action}",
                sequence=action_index,
                frame=action_index * 8,
                action=action,
                baseline_action=baseline,
                behavior_probability=0.1 if action != baseline else 1.0,
                vector=tuple(factual),
                legal_actions=legal,
                candidate_vectors=tuple(candidates),
                option_hit_cost=float(action_index == episode_index),
                duration_frames=8,
                return_to_go=float(action_index == episode_index),
                termination_reason="horizon",
            ))
    return result


def test_multi_action_dr_is_relative_to_incumbent() -> None:
    advantages = doubly_robust_advantages(
        5.0,
        [2.0, 3.0, 4.0],
        factual_index=1,
        factual_probability=0.25,
        baseline_index=0,
    )
    assert advantages == [0.0, 9.0, 2.0]


def test_dr_baseline_correction_is_shared_by_every_contrast() -> None:
    advantages = doubly_robust_advantages(
        5.0,
        [2.0, 3.0],
        factual_index=0,
        factual_probability=0.5,
        baseline_index=0,
    )
    assert advantages == [0.0, -5.0]


def test_generation3_fit_crossfits_whole_episodes_and_keeps_population() -> None:
    state = fit_dr_option_advantage(
        _episodes(tuple(f"train-{index}" for index in range(9))),
        _episodes(("validation-a", "validation-b", "validation-c")),
        nuisance_trees=2,
        population_trees=2,
        seed=11,
        threads=1,
        native_scorer_sha256="a" * 64,
    )
    assert state["authorization"]["fit_eligible"] is True
    assert len(state["models"]) == POPULATION_MEMBERS
    report = state["fit_report"]
    heldout = [
        episode
        for fold in report["crossfit_folds"]
        for episode in fold["heldout_episodes"]
    ]
    assert sorted(heldout) == [f"train-{index}" for index in range(9)]
    assert report["baseline_identity_max_error"] == 0.0


def test_causal_smoke_recovers_effect_instead_of_state_risk() -> None:
    report = run_causal_recovery_smoke(threads=1)
    assert report["passed"] is True
    assert all(report["gates"].values())


def test_short_wine_smoke_audits_options_without_becoming_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    import th06_rl.advantage_learning as module

    outcome = {
        "background_reactivations": 0,
        "capture_failures": 0,
        "corpus_failures": 0,
        "infrastructure_failures": 0,
        "trace_failures": 0,
        "corpus_failure": None,
    }
    run = {"schemas": {"transition": "th06-rl-transition-v7"}}
    manifest = {
        "complete": True,
        "dropped_records": 0,
        "run_outcome": outcome,
        "summary": {"reason_counts": {"input-lease": 3}},
    }
    monkeypatch.setattr(
        module,
        "_object",
        lambda path: run if path.name == "run.json" else manifest,
    )
    rows = []
    for index in range(32):
        candidate = index == 0
        action = "left" if candidate else "stay"
        probability = 0.05 if candidate else 0.95
        rows.append({
            "policy_id": "safe-option-exploration-v1",
            "legal_actions": ["stay", "left"],
            "baseline_action": "stay",
            "published_action": action,
            "behavior_probability": probability,
            "option": {
                "option_id": f"option-{index}",
                "boundary": True,
                "intent": action,
                "boundary_probability": probability,
                "conditional_probability": probability,
                "elapsed_frames_at_decision": 1,
                "termination_reason": None,
            },
        })
    rows.insert(1, {
        "policy_id": "safe-option-exploration-v1",
        "legal_actions": ["stay", "left"],
        "baseline_action": "stay",
        "published_action": "left",
        "behavior_probability": 1.0,
        "option": {
            "option_id": "option-0",
            "boundary": False,
            "intent": "left",
            "boundary_probability": 0.05,
            "conditional_probability": 1.0,
            "elapsed_frames_at_decision": 8,
            "termination_reason": "horizon",
        },
    })
    monkeypatch.setattr(module, "_rows", lambda *_args: iter(rows))

    report = audit_wine_option_smoke(tmp_path)
    assert report["passed"] is True
    assert report["evidence_eligible"] is False
