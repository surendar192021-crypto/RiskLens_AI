"""Model explainability: global SHAP, SHAP vs split gain, counterfactual feasibility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from common import figure, lever_text, page_setup, summary, table  # noqa: E402

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
    used = g[g["gain_share"] > 0]
    rho_used = used["mean_abs_shap"].rank().corr(used["gain_share"].rank())
    top20 = g.head(20)
    rho_top = top20["mean_abs_shap"].rank().corr(top20["gain_share"].rank())
    gender = g.set_index("feature").loc["CODE_GENDER"]
    st.caption(f"Spearman correlation of importances: {ph6['spearman_shap_vs_gain']:.2f} over "
               f"all {len(g)} features. {len(g) - len(used)} features are never used by the "
               f"model (zero in both measures), which inflates this; over the {len(used)} used "
               f"features it is {rho_used:.2f}, and within the SHAP top 20 only "
               f"{rho_top:.2f}. Exception: CODE_GENDER ranks {gender['shap_rank']} by SHAP but "
               f"{gender['gain_rank']} by gain; it is examined on the Monitoring & Fairness "
               "page.")
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
groups = pd.DataFrame(cf["by_original_pd"])
st.dataframe(groups, hide_index=True, width="stretch")
ev = table("counterfactual_evaluation_test.csv")
found = ev[ev["status"] == "found"]
shorter = int((found["term_factor"] < 1).sum())
none_found = groups.loc[groups["found_rate"] == 0, "pd_group"].tolist()
lines = [f"- {lever_text()}; everything else is locked and checked."]
if none_found:
    lines.append(f"- In the PD group(s) {', '.join(none_found)} no allowed change reaches the "
                 "threshold: risk is driven by locked factors (external scores, credit "
                 "history).")
lines.append(f"- {shorter} of {len(found)} solutions shorten the repayment period (larger "
             "instalments). That is a learned association, not advice. Results are "
             "model-based hypotheticals, not offers or guarantees.")
lines.append("- Try it for any borrower on the **Borrower Analysis** page.")
st.markdown("\n".join(lines))
