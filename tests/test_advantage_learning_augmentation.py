from __future__ import annotations

from th06_rl import advantage_learning
from th06_rl.advantage_learning import _augment_steps
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.sequential_learning import _causal_episodes


def test_augmentation_encodes_shared_hazards_once_per_option(monkeypatch) -> None:
    sample = _causal_episodes(count=1, options=1)[0]
    calls = []
    encoding = tuple(
        float(index)
        for index, _ in enumerate(advantage_learning.hazard_codebook_feature_names())
    )

    def encode_once(primitives, artifact):
        calls.append((primitives, artifact))
        return encoding

    monkeypatch.setattr(advantage_learning, "encode_hazard_set", encode_once)
    augmented = _augment_steps([sample], {"id": "fixture"})[0]

    assert len(calls) == 1
    assert calls[0] == (sample.hazard_primitives, {"id": "fixture"})
    suffix = (*encoding, *sample.history_features)
    assert augmented.vector == (*sample.vector, *suffix)
    assert augmented.candidate_vectors == tuple(
        (*vector, *suffix) for vector in sample.candidate_vectors
    )
    assert len(sample.history_features) == len(HISTORY_FEATURE_NAMES)
