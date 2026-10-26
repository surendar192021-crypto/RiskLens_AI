import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.models.calibrate import base_model, choose_method, fit_calibrated
from src.models.evaluate import expected_calibration_error, paired_bootstrap_diff
from src.models.transformers import logistic_regression_pipeline

LR_PARAMS = {"C": 1.0, "max_iter": 500, "clip_lower": 0.005, "clip_upper": 0.995,
             "min_category_frequency": 0.001}


def data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 1.5 * x)))).astype(int)
    return pd.DataFrame({"x": x, "g": pd.Categorical(["a"] * n)}), pd.Series(y)


@pytest.fixture(scope="module")
def miscalibrated():
    """A model trained with class weights, so its probabilities are far too high."""
    X_tr, y_tr = data(5000, 0)
    model = logistic_regression_pipeline(["x"], ["g"], LR_PARAMS, "balanced", 0).fit(X_tr, y_tr)
    return model


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_fixes_inflated_probabilities(miscalibrated, method):
    X_cal, y_cal = data(5000, 1)
    X_ev, y_ev = data(20000, 2)
    raw = miscalibrated.predict_proba(X_ev)[:, 1]
    cal = fit_calibrated(miscalibrated, X_cal, y_cal, method).predict_proba(X_ev)[:, 1]
    assert ((cal >= 0) & (cal <= 1)).all()
    assert brier_score_loss(y_ev, cal) < brier_score_loss(y_ev, raw)
    assert expected_calibration_error(y_ev, cal, strategy="quantile") < 0.02
    assert abs(cal.mean() - y_ev.mean()) < 0.01


def test_sigmoid_preserves_ranking(miscalibrated):
    X_cal, y_cal = data(5000, 1)
    X_ev, y_ev = data(5000, 3)
    raw = miscalibrated.predict_proba(X_ev)[:, 1]
    cal = fit_calibrated(miscalibrated, X_cal, y_cal, "sigmoid").predict_proba(X_ev)[:, 1]
    assert roc_auc_score(y_ev, cal) == pytest.approx(roc_auc_score(y_ev, raw), abs=1e-9)


def test_calibration_does_not_refit_the_model(miscalibrated):
    coef_before = miscalibrated.named_steps["model"].coef_.copy()
    X_cal, y_cal = data(3000, 4)
    cal = fit_calibrated(miscalibrated, X_cal, y_cal, "isotonic")
    np.testing.assert_array_equal(miscalibrated.named_steps["model"].coef_, coef_before)
    assert base_model(cal) is miscalibrated


def test_none_returns_model_and_bad_inputs_rejected(miscalibrated):
    X, y = data(100, 5)
    assert fit_calibrated(miscalibrated, X, y, "none") is miscalibrated
    assert base_model(miscalibrated) is miscalibrated
    with pytest.raises(ValueError, match="method"):
        fit_calibrated(miscalibrated, X, y, "beta")
    with pytest.raises(ValueError, match="both classes"):
        fit_calibrated(miscalibrated, X, pd.Series(np.zeros(len(X), dtype=int)), "sigmoid")


def test_choose_method_lowest_brier_then_log_loss():
    res = pd.DataFrame(
        {"method": ["none", "sigmoid", "isotonic"], "brier": [0.07, 0.065, 0.065],
         "log_loss": [0.25, 0.24, 0.23]}
    )
    assert choose_method(res) == "isotonic"


def test_paired_bootstrap_diff_detects_better_model():
    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.1).astype(int)
    good = np.clip(0.1 + 0.3 * y + rng.normal(0, 0.1, n), 0, 1)
    weak = np.clip(0.1 + 0.05 * y + rng.normal(0, 0.1, n), 0, 1)
    r = paired_bootstrap_diff(y, good, weak, roc_auc_score, n_resamples=200)
    assert r["difference"] > 0 and r["ci_low"] > 0
    assert r["share_resamples_a_better"] == 1.0
    same = paired_bootstrap_diff(y, good, good, roc_auc_score, n_resamples=50)
    assert same["difference"] == 0 and same["ci_low"] == same["ci_high"] == 0
