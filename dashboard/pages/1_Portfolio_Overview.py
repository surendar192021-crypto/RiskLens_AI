"""Executive / Portfolio Overview page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from common import CFG, DISCLAIMER, LEVEL_COLOURS, alerts, model_and_meta, money, page_setup, scored  # noqa: E402,E501

from src.risk.expected_loss import expected_loss  # noqa: E402
from src.risk.portfolio_analytics import portfolio_summary  # noqa: E402

page_setup("Executive / Portfolio Overview")
_, current = model_and_meta()
s = scored()
lgd = float(CFG["expected_loss"]["lgd"])
threshold = float(current["threshold"])
summ = portfolio_summary(s["probability"], s["ead"], threshold, s["risk_band"].astype(str))
el = expected_loss(s["probability"], lgd, s["ead"]).sum()

st.caption(f"Model `{current['model_version']}` | portfolio = application_test "
           f"({summ['borrowers']:,} unlabelled applicants) | {DISCLAIMER}")

info = {"delta_color": "off", "delta_arrow": "off"}
c = st.columns(4)
c[0].metric("Borrowers", f"{summ['borrowers']:,}")
c[1].metric("Mean PD", f"{summ['mean_pd']:.2%}")
c[2].metric("Expected cases (sum of PD)", f"{summ['expected_difficulties']:,.0f}",
            help="Expected number of payment-difficulty cases: an expectation, not a forecast "
                 "of the exact count. 95% range if borrowers were independent: "
                 f"{summ['low_if_independent_difficulties']:,.0f}-"
                 f"{summ['high_if_independent_difficulties']:,.0f}; real defaults are "
                 "correlated, so the true range is wider.")
c[3].metric(f"Above threshold ({threshold:.1%})", f"{summ['above_threshold_borrowers']:,}",
            f"{summ['above_threshold_share']:.1%} of borrowers", **info,
            help="Borrowers with PD at or above the operating threshold chosen on validation.")
c = st.columns(4)
c[0].metric("High or Critical band", f"{summ['high_or_critical_borrowers']:,}",
            f"{summ['high_or_critical_share']:.2%} of borrowers", **info)
c[1].metric("Exposure", money(summ["total_exposure"]),
            help="Sum of AMT_CREDIT (EAD proxy), dataset currency units.")
c[2].metric("Exposure-weighted PD", f"{summ['exposure_weighted_pd']:.2%}")
c[3].metric("Expected loss", money(el), f"LGD {lgd:.0%} assumed; "
            f"{el / summ['total_exposure']:.2%} of exposure", **info)

left, right = st.columns(2)
with left:
    st.subheader("Risk bands")
    bands = (s.groupby("risk_band", observed=False)
             .agg(borrowers=("probability", "size"), expected=("probability", "sum"))
             .reset_index())
    bands["share_of_expected_cases"] = bands["expected"] / bands["expected"].sum()
    fig = px.bar(bands, x="risk_band", y="borrowers", text="borrowers", log_y=True,
                 hover_data={"share_of_expected_cases": ":.1%"})
    fig.update_layout(height=340, xaxis_title=None, yaxis_title="Borrowers (log scale)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Bands are provisional presentation categories (config `risk_bands`), not "
               "validated banking grades. Most expected cases sit in 'Very Low' because it "
               f"holds most borrowers ({bands.loc[0, 'share_of_expected_cases']:.0%} of "
               "expected cases).")
with right:
    st.subheader("Early-warning levels")
    lv = s["alert_level"].value_counts().reindex(list(LEVEL_COLOURS)).reset_index()
    lv.columns = ["level", "borrowers"]
    fig = px.bar(lv, x="level", y="borrowers", text="borrowers", color="level",
                 color_discrete_map=LEVEL_COLOURS)
    fig.update_layout(height=340, showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    a = alerts()
    st.caption(f"{len(a):,} active alerts in the latest early-warning run. Alerts recommend a "
               "review; they are not lending decisions.")

st.subheader("Distribution of predicted probabilities")
fig = px.histogram(s, x="probability", nbins=100, log_y=True)
fig.add_vline(x=threshold, line_dash="dash", line_color="#C0392B",
              annotation_text=f"review threshold {threshold:.1%}")
fig.update_layout(height=320, xaxis_title="Predicted probability of payment difficulty",
                  yaxis_title="Borrowers (log scale)")
st.plotly_chart(fig, width="stretch")
st.info("Use the pages on the left for borrower explanations, alerts, segment analysis, "
        "simulation and stress tests, model explainability, monitoring and fairness, and the "
        "full methodology with assumptions and limitations.")
