import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.models.evaluate import (
    bootstrap_ci,
    check_probabilities,
    evaluate_probabilities,
    expected_calibration_error,
    threshold_metrics,
)
from src.models.predict import load_model, predict_proba, save_model
from src.models.thresholds import max_f1_threshold, min_cost_threshold, recall_target_threshold
from src.models.transformers import (
    FeatureFrame,
    QuantileClipper,
    logistic_regression_pipeline,
    random_forest_pipeline,
)


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x2[rng.random(n) < 0.2] = np.nan
    grp = rng.choice(["a", "b", "c"], size=n)
    logit = -2.5 + 1.2 * x1 + 0.8 * (grp == "c")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2, "grp": pd.Categorical(grp), "unused": 1})
    return X, pd.Series(y)


# --- metrics --------------------------------------------------------------------

def test_check_probabilities_rejects_bad_values():
    for bad in ([0.1, 1.2], [0.1, np.nan], [-0.01]):
        with pytest.raises(ValueError):
            check_probabilities(bad)


def test_ece_near_zero_for_calibrated_probabilities():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert expected_calibration_error(y, p, strategy="uniform") < 0.01
    assert expected_calibration_error(y, p, strategy="quantile") < 0.01


def test_ece_detects_overconfidence():
    y = np.array([0, 1] * 500)
    p = np.where(y == 1, 0.9, 0.1)  # perfectly separated but...
    p_bad = np.full(len(y), 0.9)  # ...this one says 90% for everyone, truth is 50%
    assert expected_calibration_error(y, p_bad) == pytest.approx(0.4)
    assert expected_calibration_error(y, p) == pytest.approx(0.1)


def test_threshold_metrics_confusion_matrix():
    y = [1, 1, 0, 0, 0]
    p = [0.9, 0.2, 0.8, 0.1, 0.05]
    m = threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(3 / 5)  # (tp + tn) / n


def test_accuracy_table_shows_why_accuracy_misleads():
    from src.models.accuracy_report import BASELINE, accuracy_table
    y = np.array([0] * 90 + [1] * 10)
    rng = np.random.default_rng(0)
    timid = np.where(y == 1, 0.4, 0.1) + rng.uniform(0, 0.05, 100)  # never reaches 0.5
    t = accuracy_table(y, {"timid": timid}, champion="timid").set_index("model")
    assert t.loc[BASELINE, "accuracy_at_0_5"] == pytest.approx(0.9)
    # flags nobody at 0.5 -> exactly the baseline accuracy, zero recall, despite perfect ranking
    assert t.loc["timid", "accuracy_at_0_5"] == pytest.approx(0.9)
    assert t.loc["timid", "recall_at_0_5"] == 0.0 and t.loc["timid", "roc_auc"] == 1.0
    assert t.loc["timid", "recall_at_threshold"] == 1.0 and bool(t.loc["timid", "chosen"])


def test_evaluate_probabilities_keys_and_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    r = evaluate_probabilities(y, p)
    assert r["roc_auc"] == 1.0 and r["pr_auc"] == 1.0
    assert r["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_bootstrap_ci_contains_point_estimate():
    X, y = synthetic(4000)
    p = 1 / (1 + np.exp(-(-2.5 + 1.2 * X["x1"].to_numpy())))
    point = roc_auc_score(y, p)
    low, high = bootstrap_ci(y, p, roc_auc_score, n_resamples=200)
    assert low < point < high and high - low < 0.1


# --- thresholds -----------------------------------------------------------------

def test_max_f1_threshold_on_separable_data():
    y = [0, 0, 0, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8]
    t, f1 = max_f1_threshold(y, p)
    assert 0.3 < t <= 0.7 and f1 == pytest.approx(1.0)


def test_min_cost_threshold_follows_costs():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.35, 0.6, 0.4, 0.9])
    t_cheap_fn, _ = min_cost_threshold(y, p, cost_fn=1, cost_fp=10)
    t_dear_fn, _ = min_cost_threshold(y, p, cost_fn=10, cost_fp=1)
    assert t_dear_fn < t_cheap_fn  # expensive misses -> flag more borrowers


def test_recall_target_threshold():
    y = np.array([1, 1, 1, 1, 0])
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.5])
    assert recall_target_threshold(y, p, 0.5) == pytest.approx(0.8)
    assert recall_target_threshold(y, p, 1.0) == pytest.approx(0.2)


# --- transformers and pipelines -------------------------------------------------

def test_feature_frame_orders_columns_and_handles_unseen_category():
    X, _ = synthetic(100)
    ff = FeatureFrame(["x2", "x1"], ["grp"]).fit(X)
    new = X.head(3).copy()
    new["grp"] = ["a", "zzz", None]
    out = ff.transform(new[["grp", "x1", "x2", "unused"]])
    assert list(out.columns) == ["x2", "x1", "grp"]
    assert out["grp"].iloc[0] == "a" and pd.isna(out["grp"].iloc[1])


def test_feature_frame_missing_column_is_informative():
    X, _ = synthetic(50)
    ff = FeatureFrame(["x1", "x2"], ["grp"]).fit(X)
    with pytest.raises(ValueError, match="x2"):
        ff.transform(X.drop(columns="x2"))


def test_quantile_clipper_uses_training_bounds_and_keeps_nan():
    train = np.array([[0.0], [1.0], [2.0], [3.0], [100.0], [np.nan]])
    clip = QuantileClipper(0.0, 0.75).fit(train)
    out = clip.transform(np.array([[1000.0], [np.nan], [-5.0]]))
    assert out[0, 0] == pytest.approx(np.nanquantile(train, 0.75))
    assert np.isnan(out[1, 0]) and out[2, 0] == 0.0


LR_PARAMS = {"C": 1.0, "max_iter": 500, "clip_lower": 0.005, "clip_upper": 0.995,
             "min_category_frequency": 0.001}
RF_PARAMS = {"n_estimators": 20, "min_samples_leaf": 10, "max_features": "sqrt"}


@pytest.mark.parametrize(
    "builder, params",
    [(logistic_regression_pipeline, LR_PARAMS), (random_forest_pipeline, RF_PARAMS)],
)
def test_pipelines_learn_signal_and_return_valid_probabilities(builder, params, tmp_path):
    X, y = synthetic(4000)
    pipe = builder(["x1", "x2"], ["grp"], params, None, 0).fit(X.iloc[:3000], y.iloc[:3000])
    p = predict_proba(pipe, X.iloc[3000:])
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(y.iloc[3000:], p) > 0.7

    # same predictions after save/load and with columns shuffled
    save_model(pipe, {"name": "t"}, tmp_path, "m")
    loaded, meta = load_model(tmp_path / "m.joblib")
    shuffled = X.iloc[3000:][["unused", "grp", "x2", "x1"]]
    np.testing.assert_allclose(predict_proba(loaded, shuffled), p)
    assert meta == {"name": "t"}


def test_predict_proba_rejects_empty_and_non_frames():
    X, y = synthetic(500)
    pipe = logistic_regression_pipeline(["x1", "x2"], ["grp"], LR_PARAMS, None, 0).fit(X, y)
    with pytest.raises(ValueError, match="no rows"):
        predict_proba(pipe, X.head(0))
    with pytest.raises(TypeError):
        predict_proba(pipe, X.to_numpy())
