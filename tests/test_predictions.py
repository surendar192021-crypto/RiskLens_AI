"""Checks on the real saved Phase 4 models.

Skipped until `python -m scripts.train_models` has run.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, project_path
from src.models.predict import load_model, predict_proba

CFG = load_config()
MODELS = project_path(CFG, "models")
PROCESSED = project_path(CFG, "processed_data")
POINTER = MODELS / "latest_phase4.json"

pytestmark = pytest.mark.skipif(
    not POINTER.exists() or not (PROCESSED / "portfolio_test.parquet").exists(),
    reason="Phase 4 models or processed data not available",
)


@pytest.fixture(scope="module")
def setup():
    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    run_dir = MODELS / pointer["run_id"]
    features = json.loads((PROCESSED / "feature_list.json").read_text(encoding="utf-8"))
    portfolio = pd.read_parquet(PROCESSED / "portfolio_test.parquet").head(500)
    return pointer, run_dir, features, portfolio


@pytest.mark.parametrize("role", ["champion", "challenger"])
def test_saved_model_scores_unlabelled_portfolio(setup, role):
    pointer, run_dir, features, portfolio = setup
    model, meta = load_model(run_dir / f"{pointer[role]}.joblib")
    p = predict_proba(model, portfolio)
    assert len(p) == len(portfolio)
    assert ((p >= 0) & (p <= 1)).all()
    # trained on exactly the documented feature list, in order
    assert meta["numeric"] == features["numeric"]
    assert meta["categorical"] == features["categorical"]


def test_prediction_is_deterministic_and_order_independent(setup):
    pointer, run_dir, _, portfolio = setup
    model, _ = load_model(run_dir / f"{pointer['champion']}.joblib")
    p1 = predict_proba(model, portfolio)
    p2 = predict_proba(model, portfolio[portfolio.columns[::-1]])
    np.testing.assert_allclose(p1, p2)


def test_missing_feature_gives_clear_error(setup):
    pointer, run_dir, features, portfolio = setup
    model, _ = load_model(run_dir / f"{pointer['champion']}.joblib")
    with pytest.raises(ValueError, match="missing"):
        predict_proba(model, portfolio.drop(columns=[features["numeric"][0]]))


def test_target_column_not_needed_for_scoring(setup):
    _, _, _, portfolio = setup
    assert "TARGET" not in portfolio.columns  # scoring works on unlabelled data above
