"""Early Warning Center: current alerts, history, rule evidence and the synthetic demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from common import CFG, LEVEL_COLOURS, alerts, model_and_meta, page_setup, table  # noqa: E402

from src.risk.early_warning import resolve_levels  # noqa: E402

page_setup("Early Warning Center")
a = alerts()
hist = alerts("alert_history.parquet")
escalation = CFG["early_warning"]["flag_escalation"]
lv_cut = resolve_levels(CFG["early_warning"]["levels"], model_and_meta()[1]["threshold"])
st.caption("Alerts flag borrowers for REVIEW. They are not credit decisions. Levels come from "
           f"the deployed model's PD (yellow {lv_cut['yellow']:.1%}, orange "
           f"{lv_cut['orange']:.1%} = operating threshold, red {lv_cut['red']:.0%}); "
           f"red flags are shown as context (escalation: {escalation}).")

c = st.columns(4)
for i, lv in enumerate(["red", "orange", "yellow"]):
    c[i].metric(f"{lv.capitalize()} alerts", f"{int((a['level'] == lv).sum()):,}")
c[3].metric("Alerts with red flags", f"{int((a['n_flags'] > 0).sum()):,}")

tab1, tab2, tab3, tab4 = st.tabs(["Current alerts", "Alert history", "Rule evidence",
                                  "SIMULATED deterioration"])
with tab1:
    f1, f2, f3 = st.columns([1, 1, 2])
    levels = f1.multiselect("Level", ["red", "orange", "yellow"], default=["red", "orange"])
    only_flags = f2.checkbox("Only with red flags")
    text = f3.text_input("Search triggers or drivers", "")
    view = a[a["level"].isin(levels)]
    if only_flags:
        view = view[view["n_flags"] > 0]
    if text:
        mask = (view["triggers"].str.contains(text, case=False)
                | view["main_drivers"].str.contains(text, case=False))
        view = view[mask]
    sort = st.selectbox("Sort by", ["probability", "severity", "n_flags"])
    view = view.sort_values(sort, ascending=False)
    st.write(f"{len(view):,} alerts shown")
    st.dataframe(
        view[["SK_ID_CURR", "level", "probability", "risk_band", "triggers", "main_drivers",
              "recommended_action", "observation_time", "model_version"]],
        hide_index=True, width="stretch", height=420,
        column_config={"probability": st.column_config.NumberColumn(format="%.3f")})
with tab2:
    if hist.empty:
        st.info("No alert history yet.")
    else:
        st.write("De-duplicated history: an alert that recurs is not added twice; alerts that "
                 "disappear or change are closed, not deleted.")
        st.dataframe(hist.groupby(["status", "level"]).size().rename("alerts").reset_index(),
                     hide_index=True)
        st.dataframe(hist[["SK_ID_CURR", "level", "status", "first_seen", "last_seen",
                           "closed_at", "triggers"]].head(500), hide_index=True,
                     width="stretch")
with tab3:
    st.markdown("**Alert levels on the labelled test split** (measured once, nothing tuned "
                "on it). Observed difficulty rates rise with the level.")
    lt = table("early_warning_levels_test.csv")
    lt = lt[lt["rule"] == "pd_only"]
    fig = px.bar(lt, x="level", y="target_rate", color="level",
                 color_discrete_map=LEVEL_COLOURS, text_auto=".1%")
    fig.update_layout(height=320, showlegend=False, yaxis_tickformat=".0%",
                      yaxis_title="Observed difficulty rate")
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Red flags on the validation split** (lift = rate if flagged / base rate).")
    st.dataframe(table("early_warning_flags_validation.csv"), hide_index=True,
                 width="stretch")
with tab4:
    sim = alerts("SIMULATED_deterioration.parquet")
    st.error("SYNTHETIC DATA. The dataset has no repeated observations of borrowers, so real "
             "deterioration cannot be observed. This demo applies a hypothetical income shock "
             "and is never mixed into the real alert history.")
    if sim.empty:
        st.info("No simulated deterioration file.")
    else:
        st.caption(sim["scenario"].iloc[0])
        st.dataframe(sim["change"].value_counts().rename("borrowers").reset_index(),
                     hide_index=True)
        st.dataframe(pd.DataFrame(sim).head(200), hide_index=True, width="stretch")
