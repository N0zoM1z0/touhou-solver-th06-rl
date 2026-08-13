import math

import pytest

from th06_rl.generation7.objectives import (
    awr_weight,
    extreme_logit_smoke,
    proper_actor_loss,
    weighted_negative_log_likelihood,
)


def test_proper_actor_loss_has_nonnegative_finite_lower_bound() -> None:
    reference = (0.8, 0.1, 0.1)
    for scale in (0.0, 10.0, 1000.0):
        loss = proper_actor_loss(
            (scale, -scale, 0.0),
            factual_index=0,
            weight=2.0,
            reference=reference,
            kl_coefficient=0.1,
        )
        assert math.isfinite(loss)
        assert loss >= 0.0


def test_extreme_logit_smoke_rejects_factual_probability_suppression() -> None:
    report = extreme_logit_smoke()
    assert report["passes"] is True
    assert report["proper_losses"] == sorted(report["proper_losses"])


def test_actor_weights_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_negative_log_likelihood((0.0, 0.0), factual_index=0, weight=-1.0)
    assert 0.0 < awr_weight(-100.0, temperature=1.0, maximum_weight=20.0) <= 20.0
    assert awr_weight(100.0, temperature=1.0, maximum_weight=20.0) == 20.0
