import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.explainability.counterfactuals import (
    amount_bounds,
    check_locked,
    lever_grid,
    make_candidates,
    search,
)
from src.explainability.shap_explainer import ShapExplainer, format_value, plain_english
from src.models.transformers import FeatureFrame

NUMERIC = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL",
           "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_ANNUITY_RATIO",
           "GOODS_CREDIT_RATIO", "EXT_SOURCE_MEAN"]
CATEGORICAL = ["NAME_CONTRACT_TYPE"]


def synthetic(n=6000, seed=0):
    """Risk rises with the annuity-to-income ratio and falls with the external score."""
    rng = np.random.default_rng(seed)
    income = rng.uniform(50_000, 300_000, n)
    credit = rng.uniform(100_000, 1_000_000, n)
    term = rng.uniform(10, 40, n)
    annuity = credit / term
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(n), "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": credit * 0.9, "AMT_INCOME_TOTAL": income,
        "EXT_SOURCE_MEAN": rng.uniform(0, 1, n),
        "NAME_CONTRACT_TYPE": pd.Categorical(rng.choice(["Cash", "Revolving"], n)),
    })
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    logit = -3 + 6 * df["ANNUITY_INCOME_RATIO"] - 2 * df["EXT_SOURCE_MEAN"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df, pd.Series(y)


@pytest.fixture(scope="module")
def fitted():
    df, y = synthetic()
    frame = FeatureFrame(NUMERIC, CATEGORICAL).fit(df)
    model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, verbose=-1, random_state=0)
    model.fit(frame.transform(df), y)
    return Pipeline([("features", frame), ("model", model)]), df


# --- SHAP -----------------------------------------------------------------------

def test_shap_values_add_up_and_match_lightgbm(fitted):
    pipe, df = fitted
    ex = ShapExplainer(pipe)
    exp = ex.explain(df.head(200))
    X = pipe.named_steps["features"].transform(df.head(200))
    contrib = pipe.named_steps["model"].booster_.predict(X, pred_contrib=True)
    np.testing.assert_allclose(exp.values, contrib[:, :-1], atol=1e-9)
    np.testing.assert_allclose(exp.base_values, contrib[:, -1], atol=1e-9)


def test_global_importance_finds_true_drivers(fitted):
    pipe, df = fitted
    g = ShapExplainer(pipe).global_importance(df.head(2000))
    top3 = set(g["feature"].head(3))
    assert "EXT_SOURCE_MEAN" in top3
    assert top3 & {"ANNUITY_INCOME_RATIO", "AMT_ANNUITY", "CREDIT_ANNUITY_RATIO",
                   "AMT_INCOME_TOTAL"}
    assert g["share_of_total"].sum() == pytest.approx(1.0)
    assert list(g["shap_rank"]) == list(range(1, len(g) + 1))


def test_borrower_explanation_consistent_with_prediction(fitted):
    pipe, df = fitted
    r = ShapExplainer(pipe).explain_borrower(df.iloc[[5]])
    c = r["contributions"]
    assert r["base_value"] + c["shap"].sum() == pytest.approx(r["raw_score"])
    assert 1 / (1 + np.exp(-r["raw_score"])) == pytest.approx(r["probability"], rel=1e-9)
    assert c["shap"].abs().is_monotonic_decreasing
    text = plain_english(r, portfolio_mean=0.08, threshold=0.16)
    assert "not proven causes" in text and f"{r['probability']:.1%}" in text


def test_explain_borrower_requires_one_row(fitted):
    pipe, df = fitted
    with pytest.raises(ValueError, match="one row"):
        ShapExplainer(pipe).explain_borrower(df.head(2))


def test_format_value():
    assert format_value(np.nan) == "missing"
    assert format_value(1234567.8) == "1,234,568"
    assert format_value(0.12345) == "0.123"
    assert format_value("Cash") == "Cash"


# --- Counterfactuals --------------------------------------------------------------

GRID = {"credit_factor": {"min": 0.5, "max": 1.0, "steps": 11},
        "term_factor": {"min": 0.75, "max": 1.5, "steps": 7}}


def test_candidates_recompute_ratios_and_lock_everything_else(fitted):
    _, df = fitted
    row = df.iloc[[0]]
    cf, tf = lever_grid(GRID)
    cand = make_candidates(row, cf, tf)
    assert len(cand) == len(cf) * len(tf)
    np.testing.assert_allclose(cand["CREDIT_INCOME_RATIO"],
                               cand["AMT_CREDIT"] / cand["AMT_INCOME_TOTAL"])
    np.testing.assert_allclose(cand["ANNUITY_INCOME_RATIO"],
                               cand["AMT_ANNUITY"] / cand["AMT_INCOME_TOTAL"])
    np.testing.assert_allclose(cand["AMT_ANNUITY"],
                               row["AMT_ANNUITY"].iloc[0] * cand["_credit_factor"]
                               / cand["_term_factor"])
    check_locked(row, cand.drop(columns=["_credit_factor", "_term_factor"]))


def test_check_locked_detects_changes(fitted):
    _, df = fitted
    row = df.iloc[[0]]
    cand = pd.concat([row] * 2, ignore_index=True)
    cand.loc[1, "EXT_SOURCE_MEAN"] = 0.99
    with pytest.raises(AssertionError, match="EXT_SOURCE_MEAN"):
        check_locked(row, cand)


def test_lever_grid_rejects_borrowing_more():
    with pytest.raises(ValueError, match="borrowing more"):
        lever_grid({"credit_factor": {"min": 0.5, "max": 1.2, "steps": 3},
                    "term_factor": {"min": 1, "max": 1, "steps": 1}})


def test_search_finds_lower_risk_terms(fitted):
    pipe, df = fitted
    cf, tf = lever_grid(GRID)
    p = pipe.predict_proba(df)[:, 1]
    i = int(np.argmax(np.where(df["EXT_SOURCE_MEAN"] > 0.7, p, -1)))  # risky, but good score
    row = df.iloc[[i]]
    target = float(p[i]) * 0.8
    r = search(pipe, row, target, cf, tf)
    assert r.status == "found"
    assert r.best["probability"] < target <= r.original_probability
    assert r.best["credit_factor"] <= 1.0
    assert set(r.changes["feature"]) >= {"AMT_CREDIT", "AMT_ANNUITY"}
    assert "hypothetical" in r.note


def test_search_reports_already_below_and_not_found(fitted):
    pipe, df = fitted
    cf, tf = lever_grid(GRID)
    row = df.iloc[[0]]
    p0 = float(pipe.predict_proba(row)[:, 1][0])
    assert search(pipe, row, min(p0 + 0.01, 0.99), cf, tf).status == "already_below_target"
    r = search(pipe, row, 1e-6, cf, tf)  # impossible target
    assert r.status == "not_found" and r.best is None
    assert r.best_achievable_probability <= p0


def test_plausibility_bounds_exclude_out_of_range_amounts(fitted):
    pipe, df = fitted
    cf, tf = lever_grid(GRID)
    bounds = amount_bounds(df, (0.0, 1.0))
    row = df.iloc[[int(df["AMT_CREDIT"].idxmin())]]  # smallest loan: any cut leaves the range
    p0 = float(pipe.predict_proba(row)[:, 1][0])
    r = search(pipe, row, max(p0 - 1e-9, 1e-6), cf, tf, bounds)
    if r.status == "found":
        assert r.best["credit_factor"] == 1.0
