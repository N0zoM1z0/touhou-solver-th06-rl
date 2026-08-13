import pytest

from th06_rl.generation7.policy_distribution import (
    ResidualStochasticPolicy,
    reference_probabilities,
)


def test_reference_policy_is_incumbent_uniform_mixture() -> None:
    assert reference_probabilities(
        ("left", "stay", "right"), "stay", epsilon=0.3
    ) == pytest.approx((0.1, 0.8, 0.1))


def test_one_policy_object_produces_complete_distribution_and_direct_sample() -> None:
    policy = ResidualStochasticPolicy(
        epsilon=0.3,
        temperature=1.0,
        maximum_log_tilt=2.0,
    )
    kwargs = {
        "safe_actions": ("left", "stay", "right"),
        "baseline_action": "stay",
        "logits": (2.0, 0.0, -1.0),
        "statistically_supported": (True, True, False),
        "forecast_risky": (False, False, False),
    }
    fitted = policy.distribution(**kwargs)
    shadow = policy.distribution(**kwargs)
    deployment = policy.distribution(**kwargs)
    assert fitted == shadow == deployment
    assert fitted.probability("left") > 0.1
    assert fitted.probability("right") > 0.0
    assert fitted.probability("right") / fitted.probability("stay") == (
        pytest.approx(0.1 / 0.8)
    )
    assert fitted.sample(0.0) in kwargs["safe_actions"]
    assert fitted.sample(0.999999) in kwargs["safe_actions"]


def test_statistical_and_forecast_masks_do_not_change_physical_authority() -> None:
    policy = ResidualStochasticPolicy(0.2, 1.0, 2.0)
    decision = policy.distribution(
        safe_actions=("stay", "up"),
        baseline_action="stay",
        logits=(0.0, 100.0),
        statistically_supported=(True, False),
        forecast_risky=(False, True),
    )
    assert decision.native_collision_safe == (True, True)
    assert decision.statistically_supported == (True, False)
    assert decision.forecast_risky == (False, True)
    assert decision.probabilities == pytest.approx((0.9, 0.1))
