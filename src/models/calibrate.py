"""Probability calibration of an already-fitted model.

Calibration answers: "when the model says 20%, do about 20% of such borrowers
actually have payment difficulties?" It changes the probability values but
(for sigmoid) not the ranking, so discrimination metrics barely move.

Leakage safety: the fitted model is wrapped in sklearn's FrozenEstimator, so it is
never refitted; only the small calibration map is learned, on the calibration
split, which was used neither for training nor for model selection.

- sigmoid (Platt scaling): fits p' = 1 / (1 + exp(a*s + b)) on the model score s.
  Two parameters, smooth, hard to overfit; assumes a sigmoid-shaped distortion.
- isotonic: fits any non-decreasing step function. More flexible, needs more
  data, can overfit and creates ties (which can change ROC-AUC slightly).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

METHODS = ("none", "sigmoid", "isotonic")


def fit_calibrated(model: Any, X_cal: pd.DataFrame, y_cal: pd.Series, method: str) -> Any:
    """Return a model whose predict_proba gives calibrated probabilities.

    method="none" returns the original model unchanged.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "none":
        return model
    if len(set(pd.Series(y_cal).unique())) < 2:
        raise ValueError("calibration data must contain both classes")
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_cal, y_cal)


def base_model(model: Any) -> Any:
    """The underlying fitted pipeline (needed e.g. for SHAP), calibrated or not."""
    if isinstance(model, CalibratedClassifierCV):
        frozen = model.calibrated_classifiers_[0].estimator
        return frozen.estimator if isinstance(frozen, FrozenEstimator) else frozen
    return model


def choose_method(results: pd.DataFrame, metric: str = "brier") -> str:
    """Pick the method with the lowest validation `metric` (ties: lower log loss)."""
    ranked = results.sort_values([metric, "log_loss"], ascending=True)
    return str(ranked.iloc[0]["method"])
