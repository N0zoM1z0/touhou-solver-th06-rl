import pickle

import numpy as np

from th06_rl.generation7.linear_models import (
    MINIMUM_FEATURE_SCALE,
    ridge_pipeline,
)


def test_ridge_scaler_does_not_amplify_near_constant_feature() -> None:
    rows = np.asarray([[0.0], [1e-12], [2e-12]])
    model = ridge_pipeline(alpha=1.0).fit(rows, np.asarray([0.0, 1.0, 2.0]))
    scaler = model.steps[0][1]
    assert scaler.scale_[0] == MINIMUM_FEATURE_SCALE
    assert abs(float(model.predict(np.asarray([[1.0]]))[0])) < 10.0
    restored = pickle.loads(pickle.dumps(model))
    assert restored.predict(np.asarray([[1.0]])) == model.predict(np.asarray([[1.0]]))
