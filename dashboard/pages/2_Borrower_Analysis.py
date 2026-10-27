"""Individual borrower: score, alert, SHAP explanation and counterfactual analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from common import (  # noqa: E402
    CFG,
    DISCLAIMER,
    counterfactual_bounds,
    explainer,
    lever_text,
    model_and_meta,
    money,
    page_setup,
    portfolio,
    scored,
)

from src.explainability.counterfactuals import lever_grid, search  # noqa: E402
from src.explainability.shap_explainer import format_value, plain_english  # noqa: E402
from src.risk.early_warning import evaluate_flags, generate_alerts, resolve_levels  # noqa: E402

page_setup("Individual Borrower Analysis")
model, current = model_and_meta()
s, port = scored(), portfolio()
threshold = float(current["threshold"])
st.caption(DISCLAIMER)

ranked = s.sort_values("probability", ascending=False)
examples = {
    "Highest PD in portfolio": int(ranked["SK_ID_CURR"].iloc[0]),
    "Just above review threshold": int(
        s.loc[s["probability"] >= threshold].nsmallest(1, "probability")["SK_ID_CURR"].iloc[0]),
    "Median PD": int(s.iloc[(s["probability"] - s["probability"].median()).abs().argmin()]
                     ["SK_ID_CURR"]),
}
c1, c2 = st.columns([1, 2])
pick = c1.selectbox("Quick pick", ["(enter an ID)"] + list(examples))
default = examples.get(pick, int(ranked["SK_ID_CURR"].iloc[0]))
bid = c2.number_input("Borrower ID (SK_ID_CURR)", value=default, step=1, format="%d")
if bid not in set(port["SK_ID_CURR"]):
    st.warning(f"Borrower {bid} is not in the portfolio (application_test).")
    st.stop()

row = port[port["SK_ID_CURR"] == bid]
srow = s[s["SK_ID_CURR"] == bid].iloc[0]
result = explainer().explain_borrower(row)

c = st.columns(4)
c[0].metric("Raw model probability", f"{1 / (1 + np.exp(-result['raw_score'])):.1%}",
            help="sigmoid(raw score) of the underlying LightGBM model, before any calibration.")
c[1].metric("Deployed (calibrated) PD", f"{result['probability']:.1%}",
            help=f"Calibration method: {current['calibration_method']}. With 'none' the two "
                 "values are equal; Phase 5 found no calibrator improved on the raw model.")
c[2].metric("Risk band (provisional)", str(srow["risk_band"]))
c[3].metric("Alert level", str(srow["alert_level"]).upper())
ew = CFG["early_warning"]
one = pd.DataFrame({"SK_ID_CURR": [bid], "probability": [result["probability"]],
                    "risk_band": [str(srow["risk_band"])]})
alert = generate_alerts(one, evaluate_flags(row.reset_index(drop=True), ew["flags"]),
                        resolve_levels(ew["levels"], threshold), ew, current["model_version"],
                        "application date (day 0)")
if len(alert):
    st.warning(f"**Triggers:** {alert['triggers'].iloc[0]}  \n"
               f"**Recommended action:** {alert['recommended_action'].iloc[0]}")
else:
    st.success("No alert: PD below the yellow (watch) threshold.")
st.caption(f"Model version `{current['model_version']}`. Observation time: application date "
           "(the dataset has no calendar dates).")

st.subheader("Why does the model give this estimate? (SHAP)")
contrib = result["contributions"].head(12).iloc[::-1]
fig = go.Figure(go.Bar(
    x=contrib["shap"], y=[f"{f} = {format_value(v)}" for f, v in
                          zip(contrib["feature"], contrib["value"], strict=True)],
    orientation="h", marker_color=np.where(contrib["shap"] > 0, "#C0392B", "#2E86C1")))
fig.update_layout(height=430, xaxis_title="Contribution to model score (log-odds)",
                  margin={"l": 10, "r": 10, "t": 10, "b": 10})
st.plotly_chart(fig, width="stretch")
st.text(plain_english(result, float(s["probability"].mean()), threshold))
st.caption(f"Base value {result['base_value']:.3f} log-odds + contributions = raw score "
           f"{result['raw_score']:.3f} (checked exactly). Contributions describe the model, "
           "not causes.")

with st.expander("Borrower's main financial features"):
    cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_ANNUITY_RATIO", "AGE_YEARS",
            "EMPLOYED_YEARS", "NAME_INCOME_TYPE", "NAME_CONTRACT_TYPE", "EXT_SOURCE_MEAN",
            "BUREAU_COUNT", "BUREAU_ACTIVE_DEBT_RATIO", "PREV_REFUSAL_RATE", "INST_LATE_SHARE"]
    st.dataframe(pd.DataFrame({"feature": cols,
                               "value": [format_value(row[c].iloc[0]) for c in cols]}),
                 hide_index=True, width="stretch")

st.subheader("Counterfactual: which loan terms would bring the estimate below the threshold?")
st.caption(f"{lever_text()}; everything else is locked. Each candidate is scored by the "
           "deployed model.")
if result["probability"] < threshold:
    st.info("This borrower is already below the review threshold.")
elif st.button("Search for a counterfactual"):
    cf = search(model, row, threshold, *lever_grid(CFG["counterfactuals"]),
                counterfactual_bounds())
    if cf.status == "found":
        st.success(f"Found: loan amount x{cf.best['credit_factor']:.2f}, repayment period "
                   f"x{cf.best['term_factor']:.3f} -> PD {cf.best['probability']:.1%} "
                   f"(from {cf.original_probability:.1%}).")
        ch = cf.changes.copy()
        for col in ("original", "counterfactual"):
            ch[col] = [money(v) if f.startswith("AMT") else f"{v:.3f}"
                       for f, v in zip(ch["feature"], ch[col], strict=True)]
        st.dataframe(ch, hide_index=True, width="stretch")
    else:
        st.error(f"No feasible counterfactual within the allowed changes. Lowest reachable PD: "
                 f"{cf.best_achievable_probability:.1%} (threshold {threshold:.1%}).")
    st.caption(cf.note)

st.subheader("Time-to-default (survival)")
st.info("Not available for current borrowers: the dataset has no event times or follow-up "
        "for the current loan, so no survival curve is shown. A survival study of applicants' "
        "*previous* loans is summarised on the Methodology page.")
