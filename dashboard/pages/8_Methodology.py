"""Methodology and model information: data, model, metrics, assumptions, limitations, status."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from common import CFG, figure, model_and_meta, page_setup, summary, table  # noqa: E402

page_setup("Methodology and Model Information")
_, current = model_and_meta()
audit = summary("audit_summary.json")
build = summary("build_summary.json")

st.header("Dataset")
t = audit["target"]
st.markdown(
    f"- **Source:** Home Credit Default Risk (Kaggle), 9 tables. Every number on this "
    f"dashboard is computed from these files.\n"
    f"- **Labelled applicants:** {t['n']:,}, of which {t['n_positive']:,} "
    f"({t['positive_rate']:.2%}) have TARGET = 1.\n"
    f"- **TARGET definition:** \"{audit['target_definition']}\"\n"
    f"- **Current portfolio:** application_test, {build['rows']['test']:,} unlabelled "
    f"applicants.\n"
    f"- **Features:** {build['n_features']['numeric']} numeric and "
    f"{build['n_features']['categorical']} categorical, from the application plus aggregated "
    f"bureau, previous-application, instalment, POS and card history (one row per applicant).")
st.dataframe(pd.DataFrame([{"split": k, **v} for k, v in build["splits"].items()]),
             hide_index=True)
st.caption("Stratified random split, seed 42. There are no dates, so no temporal split is "
           "possible.")

st.header("Model")
fm = table("final_test_metrics.csv")
st.markdown(
    f"- **Deployed:** `{current['model_version']}`.\n"
    f"- **Calibration:** {current['calibration_method']}. Sigmoid and isotonic calibration "
    f"did not improve on the raw model on validation.\n"
    f"- **Operating threshold:** {current['threshold']:.4f} "
    f"({current['threshold_policy']}, chosen on {current['threshold_chosen_on']}).")
st.dataframe(fm[["model", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "brier",
                 "ece_quantile"]], hide_index=True, width="stretch")
st.dataframe(table("calibration_comparison_validation.csv")[
    ["method", "roc_auc", "pr_auc", "brier", "log_loss", "ece_quantile", "selected"]],
    hide_index=True)

st.header("Temporal analysis (previous loans only)")
st.markdown("Survival analysis of time to the first 30+ day late payment on applicants' "
            "**previous** Home Credit loans. The current loan has no follow-up, so no "
            "time-to-default is shown for current borrowers.")
km = table("survival_km_horizons.csv")
st.dataframe(km[["group", "loans", "events", "S(90d)", "S(365d)", "S(730d)"]],
             hide_index=True, width="stretch")
path = figure("phase7_km_by_target.png")
if path:
    st.image(str(path), width=650)

st.header("Assumptions")
st.dataframe(pd.DataFrame([
    {"item": "LGD", "value": f"{CFG['expected_loss']['lgd']:.0%}",
     "status": "assumed: no recovery data"},
    {"item": "EAD", "value": CFG["portfolio"]["exposure_column"],
     "status": "proxy: credit amount, fully outstanding"},
    {"item": "Default correlation rho", "value": str(CFG["monte_carlo"]["rho"]),
     "status": "assumed: no data to estimate it"},
    {"item": "Risk bands", "value": str(CFG["risk_bands"]["edges"]),
     "status": "provisional presentation bands"},
    {"item": "Alert levels", "value": str(CFG["early_warning"]["levels"]),
     "status": "derived from validation results"},
    {"item": "PSI thresholds", "value": str(CFG["monitoring"]["psi_thresholds"]),
     "status": "industry rule of thumb"},
]), hide_index=True, width="stretch")

st.header("Limitations")
st.markdown(
    "- TARGET means early payment difficulty in a sample, not legal default.\n"
    "- Random split: performance on future applicants is unverified. The current portfolio "
    "shows real input drift, and has no outcome labels.\n"
    "- SHAP and counterfactuals describe the model, not causes. Stress responses to income and "
    "instalment shocks are not economically credible with this model.\n"
    "- Expected loss and VaR depend on assumed LGD and correlation; they are not regulatory "
    "measures.\n"
    "- Fairness: the model is calibrated within groups but error rates differ by gender and "
    "age. No fairness or compliance claim is made.")

st.header("Implementation status")
st.dataframe(pd.DataFrame([
    ("Data audit and feature pipeline", "Implemented and tested"),
    ("4 models, calibration, threshold, test evaluation", "Implemented; results verified"),
    ("SHAP and constrained counterfactuals", "Implemented; SHAP verified exact"),
    ("Survival analysis", "Previous loans only (current loans: not supported by the data)"),
    ("Early warning", "Implemented; deterioration alerts are synthetic demo only"),
    ("Portfolio analytics, expected loss", "Implemented; EL is assumption-based"),
    ("Monte Carlo, stress tests, migration", "Implemented; rho and LGD assumed"),
    ("Drift, fairness, registry, audit log", "Implemented; offline monitoring"),
    ("Optional FastAPI", "Not implemented"),
], columns=["component", "status"]), hide_index=True, width="stretch")
st.caption("Full documentation: the docs/ folder in the project.")
