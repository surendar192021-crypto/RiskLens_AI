"""Model explainability: global SHAP, SHAP vs split gain, counterfactual feasibility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from common import figure, page_setup, summary, table  # noqa: E402

page_setup("Model Explainability")
st.caption("SHAP values describe how the deployed model uses each feature (log-odds "
           "contributions, verified exact against LightGBM). They are not causal effects.")
g = table("shap_global_importance_test.csv")
ph6 = summary("phase6_summary.json")

top_n = st.slider("Features shown", 5, 40, 20)
top = g.head(top_n).iloc[::-1]
fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h",
             hover_data=["share_of_total", "gain_rank"])
fig.update_layout(height=26 * top_n + 80, xaxis_title="Mean |SHAP| (log-odds), 5,000 test "
                  "borrowers", yaxis_title=None)
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("SHAP rank vs split-gain rank")
    fig = px.scatter(g, x="gain_rank", y="shap_rank", hover_name="feature", log_x=True,
                     log_y=True)
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Spearman correlation of importances: {ph6['spearman_shap_vs_gain']:.2f}. "
               "Exception: CODE_GENDER ranks 5th by SHAP but 45th by gain; it is examined "
               "on the Monitoring & Fairness page.")
with right:
    st.subheader("Beeswarm (direction of effects)")
    path = figure("phase6_shap_beeswarm.png")
    if path:
        st.image(str(path), width="stretch")

st.subheader("Constrained counterfactuals: how often can loan terms alone help?")
cf = ph6["counterfactuals"]
c = st.columns(4)
c[0].metric("Flagged test borrowers sampled", f"{cf['sample_size']:,}")
c[1].metric("Counterfactual found", f"{cf['found_rate']:.0%}")
c[2].metric("Median loan factor (found)", f"{cf['median_credit_factor_found']:.2f}")
c[3].metric("Median period factor (found)", f"{cf['median_term_factor_found']:.3f}")
st.dataframe(pd.DataFrame(cf["by_original_pd"]), hide_index=True, width="stretch")
st.markdown(
    "- Only the loan amount (down to 50%) and repayment period (x0.75-1.5) may change; "
    "everything else is locked and checked.\n"
    "- Above about 40% PD no allowed change reaches the threshold: risk is driven by locked "
    "factors (external scores, credit history).\n"
    "- Some solutions shorten the loan (larger instalments). That is a learned association, "
    "not advice. Results are model-based hypotheticals, not offers or guarantees.\n"
    "- Try it for any borrower on the **Borrower Analysis** page.")
