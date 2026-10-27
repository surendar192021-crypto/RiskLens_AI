"""Portfolio analytics: filters, segments, concentration and expected loss."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from common import CFG, model_and_meta, money, page_setup, portfolio, scored, table  # noqa: E402

from src.risk.expected_loss import expected_loss, lgd_sensitivity  # noqa: E402
from src.risk.portfolio_analytics import (  # noqa: E402
    apply_filters,
    lorenz_curve,
    portfolio_summary,
    segment_table,
    top_share,
)

page_setup("Portfolio Analytics")
_, current = model_and_meta()
s, port = scored(), portfolio()
threshold = float(current["threshold"])

with st.sidebar:
    st.header("Filters")
    filters = {}
    for col in ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "AGE_BAND", "REGION_RATING_CLIENT"]:
        options = sorted(port[col].dropna().astype(str).unique())
        chosen = st.multiselect(col, options)
        if chosen:
            filters[col] = chosen
    pd_range = st.slider("PD range", 0.0, 1.0, (0.0, 1.0), 0.01)
    lgd = st.slider("LGD (ASSUMPTION)", 0.0, 1.0, float(CFG["expected_loss"]["lgd"]), 0.05)

port_view = port.assign(_pd=s["probability"].to_numpy())
port_view[list(filters)] = port_view[list(filters)].astype("object").astype(str)
mask = apply_filters(port_view, {**filters, "_pd": pd_range}).to_numpy()
fs, fp = s[mask], port[mask]
if mask.sum() == 0:
    st.warning("No borrowers match these filters.")
    st.stop()

summ = portfolio_summary(fs["probability"], fs["ead"], threshold)
el = expected_loss(fs["probability"], lgd, fs["ead"])
c = st.columns(5)
c[0].metric("Borrowers", f"{summ['borrowers']:,}", f"{mask.mean():.1%} of portfolio",
            delta_color="off", delta_arrow="off")
c[1].metric("Mean PD", f"{summ['mean_pd']:.2%}")
c[2].metric("Expected cases (sum of PD)", f"{summ['expected_difficulties']:,.0f}")
c[3].metric("Exposure / exposure-weighted PD", money(summ["total_exposure"]),
            f"{summ['exposure_weighted_pd']:.2%}", delta_color="off", delta_arrow="off")
c[4].metric(f"Expected loss, LGD {lgd:.0%} (assumed)", money(el.sum()))
st.caption("Borrower counts, expected cases (sum of PD), exposure and exposure-weighted PD are "
           "different quantities. The sum of PD is an expectation, not an exact future count.")

st.subheader("Risk by segment")
seg_var = st.selectbox("Segment by", CFG["portfolio"]["segments"])
seg = segment_table(fp, seg_var, fs["probability"], fs["ead"],
                    max_levels=CFG["portfolio"]["max_segment_levels"])
left, right = st.columns(2)
fig = px.bar(seg.sort_values("segment"), x="segment", y="mean_pd", text_auto=".1%",
             hover_data=["borrowers", "expected_difficulties", "exposure_weighted_pd"])
fig.add_hline(y=summ["mean_pd"], line_dash="dash", annotation_text="filtered mean PD")
fig.update_layout(height=360, yaxis_tickformat=".0%", xaxis_title=None)
left.plotly_chart(fig, width="stretch")
fig = px.bar(seg.sort_values("segment"), x="segment",
             y=["borrower_share", "share_of_expected_difficulties", "exposure_share"],
             barmode="group")
fig.update_layout(height=360, yaxis_tickformat=".0%", xaxis_title=None, legend_title=None)
right.plotly_chart(fig, width="stretch")
st.dataframe(seg, hide_index=True, width="stretch")

st.subheader("Concentration and expected-loss sensitivity")
left, right = st.columns(2)
lc = lorenz_curve(el)
fig = px.line(lc, x="share_of_borrowers", y="share_of_total")
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="evenly spread")
fig.update_layout(height=340, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                  xaxis_title="Share of borrowers (largest EL first)",
                  yaxis_title="Share of expected loss", showlegend=False)
left.plotly_chart(fig, width="stretch")
left.caption(f"Top 10% of borrowers hold {top_share(el, 0.10):.1%} of expected loss.")
sens = lgd_sensitivity(fs["probability"], fs["ead"], CFG["expected_loss"]["lgd_sensitivity"])
sens["portfolio_el"] = sens["portfolio_el"].map(money)
right.dataframe(sens, hide_index=True, width="stretch")
right.caption("EL = PD x LGD x EAD. PD is early payment difficulty, LGD is assumed, EAD is the "
              "credit amount assumed fully outstanding. Illustrative, not regulatory capital.")

with st.expander("Do aggregated PDs match reality? (labelled test split, Phase 8)"):
    chk = table("portfolio_segment_check_test.csv")
    st.dataframe(chk[chk["variable"] == seg_var][["segment", "borrowers",
                                                  "expected_difficulties",
                                                  "observed_difficulties", "relative_error"]],
                 hide_index=True, width="stretch")
