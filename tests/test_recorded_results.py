"""Audit tests: the recorded Phase 6 and Phase 10 results must follow from the deployed
model and the saved data. Skipped until the pipeline outputs exist.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.config import load_config, project_path
from src.explainability.counterfactuals import lever_grid
from src.explainability.shap_explainer import ShapExplainer
from src.models.predict import load_model, predict_proba
from src.monitoring.drift import psi_numeric

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
TABLES = project_path(CFG, "tables")
RESULTS = project_path(CFG, "experiment_results")
NEEDED = ["models/current_model.json", "data/processed/modelling_train.parquet",
          "data/processed/portfolio_test.parquet",
          "reports/experiment_results/phase6_summary.json",
          "reports/experiment_results/phase10_monitoring_summary.json"]
pytestmark = pytest.mark.skipif(not all((ROOT / f).exists() for f in NEEDED),
                                reason="pipeline outputs not built")


@pytest.fixture(scope="module")
def deployed():
    models = project_path(CFG, "models")
    current = json.loads((models / "current_model.json").read_text(encoding="utf-8"))
    model, _ = load_model(models / current["model_file"])
    return model, current


@pytest.fixture(scope="module")
def splits():
    from src.models.train import load_modelling_data
    data = load_modelling_data(CFG)
    df, target = data["df"], data["features"]["target"]
    return {s: df[df["split"] == s].drop(columns=["split", target]).reset_index(drop=True)
            for s in ["train", "test"]}


def phase6():
    return json.loads((RESULTS / "phase6_summary.json").read_text(encoding="utf-8"))


def test_shap_is_exact_for_the_deployed_model(deployed, splits):
    model, _ = deployed
    rows = splits["test"].sample(200, random_state=0)
    ex = ShapExplainer(model)
    assert ex.tree is model.named_steps["model"]  # explains the deployed object itself
    exp = ex.explain(rows)
    contrib = ex.tree.booster_.predict(ex.frame.transform(rows), pred_contrib=True)
    np.testing.assert_allclose(exp.values, contrib[:, :-1], atol=1e-10)
    raw = exp.values.sum(axis=1) + exp.base_values
    np.testing.assert_allclose(1 / (1 + np.exp(-raw)), predict_proba(model, rows), atol=1e-9)


def test_recorded_spearman_follows_from_saved_importances():
    g = pd.read_csv(TABLES / "shap_global_importance_test.csv")
    rho = spearmanr(g["mean_abs_shap"], g["gain_share"]).statistic
    assert rho == pytest.approx(phase6()["spearman_shap_vs_gain"], abs=1e-12)
    # Unused features are zero in both measures; they must be exactly zero in SHAP too.
    unused = g["gain_share"] == 0
    assert (g.loc[unused, "mean_abs_shap"] == 0).all()


def test_counterfactual_summary_follows_from_per_borrower_table(deployed):
    _, current = deployed
    thr = float(current["threshold"])
    cf = phase6()["counterfactuals"]
    ev = pd.read_csv(TABLES / "counterfactual_evaluation_test.csv")
    assert len(ev) == cf["sample_size"] == CFG["counterfactuals"]["evaluation_sample"]
    assert ev["borrower_id"].is_unique
    assert (ev["original_probability"] >= thr).all()  # every sampled borrower was flagged
    assert (ev["status"] == "found").mean() == pytest.approx(cf["found_rate"])
    found = ev[ev["status"] == "found"]
    assert (found["cf_probability"] < thr).all()  # every reported change crosses the threshold
    assert (ev.loc[ev["status"] == "not_found", "best_achievable_probability"] >= thr).all()
    credit, term = lever_grid(CFG["counterfactuals"])
    assert found["credit_factor"].isin(credit).all() and found["term_factor"].isin(term).all()


def test_recorded_score_psi_uses_train_reference_and_portfolio(deployed, splits):
    model, _ = deployed
    port = pd.read_parquet(project_path(CFG, "processed_data") / "portfolio_test.parquet")
    p_train = predict_proba(model, splits["train"])
    p_port = predict_proba(model, port)
    psi, _ = psi_numeric(pd.Series(p_train), pd.Series(p_port), CFG["monitoring"]["psi_bins"])
    mon = json.loads((RESULTS / "phase10_monitoring_summary.json").read_text(encoding="utf-8"))
    assert psi == pytest.approx(mon["drift"]["prediction_psi"], abs=1e-12)
