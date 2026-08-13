import numpy as np

from th06_rl.generation7.offline_dataset import proximal_targets


def test_proximal_targets_are_factual_forward_hit_sums() -> None:
    costs = np.asarray([1, 0, 2, 0, 3])
    assert proximal_targets(costs, 1).tolist() == [1, 0, 2, 0, 3]
    assert proximal_targets(costs, 3).tolist() == [3, 2, 5, 3, 3]
