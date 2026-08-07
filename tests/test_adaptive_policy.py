from __future__ import annotations

from th06_rl.policies.adaptive import (
    HIT_CREDIT_HORIZON_FRAMES,
    PACKED_STATE_SCHEMA,
    AdaptivePolicy,
    unpack_state,
)
from th06_rl.policy_api import PolicyContext, PolicyFailureEvent, PolicyOutcome


def context(
    *,
    phase: str = "timeline:test",
    stage: int = 4,
    frame: int = 100,
) -> PolicyContext:
    return PolicyContext(
        frame=frame,
        scope=(2, 0, 0, stage),
        source_context=phase,
        baseline_action="stay",
        locally_admissible_actions=("stay", "left"),
        player_x=192.0,
        player_y=384.0,
        power=64,
        bullet_count=32,
        laser_count=0,
        hard_action_count=12,
        exploration_rate=0.0,
        current_action="stay",
        hard_admissible_actions=("stay", "left"),
        phase_elapsed_frames=60,
    )


def test_zero_exploration_preserves_reactive_baseline() -> None:
    policy = AdaptivePolicy()
    decision = policy.decide(context())
    assert decision.action == "stay"
    assert decision.behavior_probability == 1.0


def test_unpublished_stale_choice_does_not_train() -> None:
    policy = AdaptivePolicy()
    decision = policy.decide(context())
    policy.observe(PolicyOutcome(
        frame=100,
        scope=(2, 0, 0, 4),
        source_context="timeline:test",
        action=decision.action,
        published=False,
        elapsed_frames=0,
        life_lost=False,
        bomb_used=False,
        control_dead_end=False,
        authority_lost=False,
        phase_changed=False,
        next_hard_action_count=12,
        next_player_x=192.0,
        next_player_y=384.0,
    ))
    assert not policy.trials
    assert not policy.pending_keys


def test_source_phase_and_stage_are_nonsharing_keys() -> None:
    policy = AdaptivePolicy()
    first = policy._context_key(context(phase="boss:0:sub1:nonspell", stage=4))
    second = policy._context_key(context(phase="boss:0:sub2:spell", stage=4))
    other_stage = policy._context_key(
        context(phase="boss:0:sub1:nonspell", stage=1)
    )
    assert len({first, second, other_stage}) == 3


def test_fine_context_separates_clock_control_and_hard_geometry() -> None:
    policy = AdaptivePolicy()
    base = context()
    later = PolicyContext(**{
        **base.__dict__,
        "phase_elapsed_frames": 90,
        "current_action": "left",
        "hard_admissible_actions": ("stay",),
    })

    assert policy._context_key(base) == policy._context_key(later)
    assert policy._middle_context_key(base) != policy._middle_context_key(later)
    assert policy._fine_context_key(base) != policy._fine_context_key(later)


def test_middle_context_reuses_different_exact_frontiers() -> None:
    policy = AdaptivePolicy()
    base = context()
    changed_frontier = PolicyContext(**{
        **base.__dict__,
        "hard_admissible_actions": ("stay",),
        "locally_admissible_actions": ("stay",),
    })

    assert (
        policy._middle_context_key(base)
        == policy._middle_context_key(changed_frontier)
    )
    assert policy._fine_context_key(base) != policy._fine_context_key(
        changed_frontier
    )


def test_unseen_fine_state_uses_middle_backoff() -> None:
    policy = AdaptivePolicy()
    value = context()
    middle = policy._middle_context_key(value)
    left = policy._action_key(middle, "left")
    policy.middle_trials[left] = 10
    policy.middle_reward_sum[left] = 20.0

    assert policy.decide(value).action == "left"


def test_physical_hit_is_consumed_as_negative_learning_feedback() -> None:
    policy = AdaptivePolicy()
    decision = policy.decide(context())

    policy.observe(PolicyOutcome(
        frame=100,
        scope=(2, 0, 0, 4),
        source_context="timeline:test",
        action=decision.action,
        published=True,
        elapsed_frames=1,
        life_lost=True,
        bomb_used=False,
        control_dead_end=False,
        authority_lost=False,
        phase_changed=False,
        next_hard_action_count=0,
        next_player_x=192.0,
        next_player_y=384.0,
    ))
    policy.observe_failure(PolicyFailureEvent(
        frame=101,
        scope=(2, 0, 0, 4),
        source_context="timeline:test",
        kind="physical-hit",
    ))

    assert sum(policy.trials.values()) == 1
    assert sum(policy.middle_trials.values()) == 1
    assert sum(policy.fine_trials.values()) == 1
    assert sum(policy.reward_sum.values()) < 0.0
    assert not policy.pending_keys
    assert policy.credited_hit_events == 1
    assert policy.credited_hit_actions == 1


def test_delayed_physical_hit_credits_recent_same_phase_actions() -> None:
    policy = AdaptivePolicy()
    for frame in (100, 101, 102):
        decision = policy.decide(context(frame=frame))
        policy.observe(PolicyOutcome(
            frame=frame,
            scope=(2, 0, 0, 4),
            source_context="timeline:test",
            action=decision.action,
            published=True,
            elapsed_frames=1,
            life_lost=False,
            bomb_used=False,
            control_dead_end=False,
            authority_lost=False,
            phase_changed=False,
            next_hard_action_count=12,
            next_player_x=192.0,
            next_player_y=384.0,
        ))

    policy.observe_failure(PolicyFailureEvent(
        frame=107,
        scope=(2, 0, 0, 4),
        source_context="timeline:test",
        kind="physical-hit",
    ))

    assert sum(policy.trials.values()) == 3
    assert sum(policy.reward_sum.values()) < 0.0
    assert policy.credited_hit_events == 1
    assert policy.credited_hit_actions == 3
    assert not policy.credit_trace


def test_hit_credit_does_not_cross_source_phase() -> None:
    policy = AdaptivePolicy()
    decision = policy.decide(context(phase="boss:sub32", frame=100))
    policy.observe(PolicyOutcome(
        frame=100,
        scope=(2, 0, 0, 4),
        source_context="boss:sub32",
        action=decision.action,
        published=True,
        elapsed_frames=1,
        life_lost=False,
        bomb_used=False,
        control_dead_end=False,
        authority_lost=False,
        phase_changed=False,
        next_hard_action_count=12,
        next_player_x=192.0,
        next_player_y=384.0,
    ))
    reward_before = sum(policy.reward_sum.values())

    policy.observe_failure(PolicyFailureEvent(
        frame=105,
        scope=(2, 0, 0, 4),
        source_context="boss:sub33",
        kind="physical-hit",
    ))

    assert sum(policy.reward_sum.values()) == reward_before
    assert policy.uncredited_hit_events == 1
    assert not policy.credit_trace


def test_hit_credit_expires_after_bounded_horizon() -> None:
    policy = AdaptivePolicy()
    decision = policy.decide(context(frame=100))
    policy.observe(PolicyOutcome(
        frame=100,
        scope=(2, 0, 0, 4),
        source_context="timeline:test",
        action=decision.action,
        published=True,
        elapsed_frames=1,
        life_lost=False,
        bomb_used=False,
        control_dead_end=False,
        authority_lost=False,
        phase_changed=False,
        next_hard_action_count=12,
        next_player_x=192.0,
        next_player_y=384.0,
    ))
    reward_before = sum(policy.reward_sum.values())

    policy.observe_failure(PolicyFailureEvent(
        frame=100 + HIT_CREDIT_HORIZON_FRAMES + 1,
        scope=(2, 0, 0, 4),
        source_context="timeline:test",
        kind="physical-hit",
    ))

    assert sum(policy.reward_sum.values()) == reward_before
    assert policy.uncredited_hit_events == 1


def test_latency_contaminated_outcome_is_retained_but_not_trained_online() -> None:
    policy = AdaptivePolicy()
    decision = policy.decide(context())
    policy.observe(PolicyOutcome(
        frame=100,
        scope=(2, 0, 0, 4),
        source_context="timeline:test",
        action=decision.action,
        published=True,
        elapsed_frames=3,
        life_lost=True,
        bomb_used=False,
        control_dead_end=False,
        authority_lost=False,
        phase_changed=False,
        next_hard_action_count=0,
        next_player_x=192.0,
        next_player_y=384.0,
        learning_eligible=False,
    ))

    assert not policy.trials
    assert not policy.pending_keys


def test_hot_reload_accepts_pre_filter_outcome_shape() -> None:
    policy = AdaptivePolicy()
    decision = policy.decide(context())

    class LegacyOutcome:
        frame = 100
        action = decision.action
        published = True
        life_lost = False
        bomb_used = False
        control_dead_end = False
        authority_lost = False
        next_hard_action_count = 12
        next_player_x = 192.0
        next_player_y = 384.0
        phase_changed = False

    policy.observe(LegacyOutcome())

    assert sum(policy.trials.values()) == 1
    assert not policy.pending_keys


def test_restart_checkpoint_is_compact_and_lossless() -> None:
    policy = AdaptivePolicy()
    decision = policy.decide(context())
    policy.observe(PolicyOutcome(
        frame=100,
        scope=(2, 0, 0, 4),
        source_context="timeline:test",
        action=decision.action,
        published=True,
        elapsed_frames=1,
        life_lost=False,
        bomb_used=False,
        control_dead_end=False,
        authority_lost=False,
        phase_changed=False,
        next_hard_action_count=12,
        next_player_x=192.0,
        next_player_y=384.0,
    ))

    packed = policy.export_state()
    assert packed["schema"] == PACKED_STATE_SCHEMA
    raw = unpack_state(packed)
    assert "fine_opportunities" not in raw
    assert "fine_selected" not in raw
    restored = AdaptivePolicy()
    restored.import_state(packed)

    assert raw["trials"] == dict(policy.trials)
    assert restored.trials == policy.trials
    assert restored.reward_sum == policy.reward_sum
    assert restored.middle_trials == policy.middle_trials
    assert restored.middle_reward_sum == policy.middle_reward_sum
    assert restored.fine_trials == policy.fine_trials
    assert restored.fine_reward_sum == policy.fine_reward_sum


def test_old_reward_checkpoint_is_not_mixed_into_hit_credit_v2() -> None:
    policy = AdaptivePolicy()
    key = policy._action_key(policy._context_key(context()), "stay")
    policy.import_state({
        "schema": "th06-rl-online-ucb-v1",
        "reward_version": "survival-reserve-v1",
        "decisions": 12,
        "trials": {key: 7},
        "reward_sum": {key: 8.75},
    })

    assert not policy.trials
    assert not policy.reward_sum
    assert not policy.fine_trials
    assert policy.decide(context()).action == "stay"


def test_old_hierarchical_checkpoint_is_not_mixed_into_hit_credit_v2() -> None:
    policy = AdaptivePolicy()
    fine_key = policy._action_key(
        policy._fine_context_key(context()),
        "stay",
    )
    policy.import_state({
        "schema": "th06-rl-online-hierarchical-ucb-v2",
        "reward_version": "survival-reserve-v1",
        "decisions": 9,
        "fine_trials": {fine_key: 5},
        "fine_reward_sum": {fine_key: 6.25},
    })

    assert not policy.fine_trials
    assert not policy.fine_reward_sum
    assert not policy.middle_trials
    assert not policy.middle_reward_sum
