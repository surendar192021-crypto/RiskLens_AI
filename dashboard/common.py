"""Shared data layer for the RiskLens AI dashboard.

The dashboard never trains anything. It loads the deployed model and the saved
pipeline outputs once (cached), and computes everything else live with the same
`src/` functions used by the analysis scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config, project_path  # noqa: E402
from src.explainability.counterfactuals import amount_bounds  # noqa: E402
from src.explainability.shap_explainer import ShapExplainer  # noqa: E402
from src.models.predict import load_model, predict_proba  # noqa: E402
from src.risk.borrower_risk import assign_risk_band  # noqa: E402
from src.risk.early_warning import (  # noqa: E402
    LEVELS,
    evaluate_flags,
    final_level,
    pd_level,
    resolve_levels,
)
from src.risk.portfolio_analytics import add_age_band  # noqa: E402

CFG = load_config()
MODELS = project_path(CFG, "models")
PROCESSED = project_path(CFG, "processed_data")
TABLES = project_path(CFG, "tables")
FIGURES = project_path(CFG, "figures")
RESULTS = project_path(CFG, "experiment_results")

REQUIRED = {
    "models/current_model.json": "python -m scripts.calibrate_and_evaluate",
    "data/processed/portfolio_test.parquet": "python -m scripts.build_dataset",
    "data/processed/alerts/current_alerts.parquet": "python -m scripts.run_early_warning",
    "reports/experiment_results/phase10_monitoring_summary.json":
        "python -m scripts.run_monitoring",
}

DISCLAIMER = ("Decision support only. Probabilities estimate early payment difficulty "
              "(TARGET) for this sample; they are not guarantees, credit decisions or "
              "regulatory risk measures.")
LEVEL_COLOURS = {"green": "#2E8B57", "yellow": "#D4A017", "orange": "#E07B39", "red": "#C0392B"}


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"RiskLens AI | {title}", layout="wide")
    st.title(title)
    missing = [(f, cmd) for f, cmd in REQUIRED.items() if not (ROOT / f).exists()]
    if missing:
        st.error("Pipeline outputs are missing. Run these commands from the project root first:")
        for f, cmd in missing:
            st.code(cmd, language="bash")
            st.caption(f"missing: {f}")
        st.stop()


def money(x: float) -> str:
    """Dataset currency units (not stated in the data)."""
    if abs(x) >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.1f}M"
    return f"{x:,.0f}"


@st.cache_resource(show_spinner="Loading deployed model...")
def model_and_meta() -> tuple[Any, dict[str, Any]]:
    current = json.loads((MODELS / "current_model.json").read_text(encoding="utf-8"))
    model, _ = load_model(MODELS / current["model_file"])
    return model, current


@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def explainer() -> ShapExplainer:
    return ShapExplainer(model_and_meta()[0])


@st.cache_data(show_spinner="Loading portfolio...")
def portfolio() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "portfolio_test.parquet")
    return add_age_band(df, CFG["portfolio"]["age_bands"])


@st.cache_data(show_spinner="Scoring portfolio...")
def scored() -> pd.DataFrame:
    """Every portfolio borrower scored by the deployed model, with band and alert level."""
    model, current = model_and_meta()
    port = portfolio()
    p = predict_proba(model, port)
    ew = CFG["early_warning"]
    thresholds = resolve_levels(ew["levels"], current["threshold"])
    flags = evaluate_flags(port, ew["flags"])
    p_idx = pd_level(p, thresholds)
    lvl = final_level(p_idx, flags.any(axis=1).to_numpy(), ew["flag_escalation"])
    bands = CFG["risk_bands"]
    return pd.DataFrame({
        "SK_ID_CURR": port["SK_ID_CURR"].to_numpy(),
        "probability": p,
        "risk_band": pd.Categorical(assign_risk_band(p, bands["edges"], bands["labels"]),
                                    categories=bands["labels"], ordered=True),
        "alert_level": pd.Categorical(np.array(LEVELS)[lvl], categories=LEVELS, ordered=True),
        "n_flags": flags.sum(axis=1).to_numpy(),
        "ead": port[CFG["portfolio"]["exposure_column"]].to_numpy(dtype="float64"),
    })


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


@st.cache_data
def summary(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@st.cache_data
def alerts(name: str = "current_alerts.parquet") -> pd.DataFrame:
    path = PROCESSED / "alerts" / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner="Loading training ranges...")
def counterfactual_bounds() -> dict[str, tuple]:
    """Plausible loan amounts from the TRAINING split (same rule as Phase 6)."""
    cols = ["SK_ID_CURR", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    df = pd.read_parquet(PROCESSED / "modelling_train.parquet", columns=cols)
    splits = pd.read_parquet(PROCESSED / "splits.parquet")
    train = df[df["SK_ID_CURR"].isin(splits.loc[splits["split"] == "train", "SK_ID_CURR"])]
    return amount_bounds(train, tuple(CFG["counterfactuals"]["plausibility_quantiles"]))


def figure(name: str) -> Path | None:
    path = FIGURES / name
    return path if path.exists() else None
