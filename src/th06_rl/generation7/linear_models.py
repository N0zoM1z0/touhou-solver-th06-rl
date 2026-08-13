"""Numerically bounded scaling contracts for Generation-7 linear models."""

from __future__ import annotations


MINIMUM_FEATURE_SCALE = 1e-3


def _sklearn_types():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return Ridge, make_pipeline, StandardScaler


try:
    _Ridge, _make_pipeline, _StandardScaler = _sklearn_types()
except ModuleNotFoundError:  # Core installs may inspect contracts without offline extras.
    _Ridge = _make_pipeline = _StandardScaler = None


if _StandardScaler is not None:
    class FlooredStandardScaler(_StandardScaler):
        """StandardScaler with a stable, serializable minimum train scale."""

        def __init__(self, *, minimum_scale: float = MINIMUM_FEATURE_SCALE) -> None:
            super().__init__()
            self.minimum_scale = minimum_scale

        def fit(self, values, target=None, sample_weight=None):
            import numpy as np

            if self.minimum_scale <= 0.0:
                raise ValueError("minimum feature scale must be positive")
            result = super().fit(values, target, sample_weight=sample_weight)
            self.scale_ = np.maximum(self.scale_, self.minimum_scale)
            return result
else:
    class FlooredStandardScaler:  # pragma: no cover - offline call fails below
        pass


def ridge_pipeline(*, alpha: float, minimum_scale: float = MINIMUM_FEATURE_SCALE):
    """Build Ridge with a train-fold-only standardizer whose scale cannot vanish."""
    if alpha <= 0.0 or minimum_scale <= 0.0:
        raise ValueError("bounded Ridge configuration is invalid")
    if _Ridge is None or _make_pipeline is None:
        raise ModuleNotFoundError("install the offline extra for scikit-learn")
    return _make_pipeline(
        FlooredStandardScaler(minimum_scale=minimum_scale),
        _Ridge(alpha=alpha),
    )
