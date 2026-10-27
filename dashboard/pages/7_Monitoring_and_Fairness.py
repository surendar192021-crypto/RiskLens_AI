"""Monitoring and fairness: data quality, drift, performance, fairness, registry, audit log."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from common import MODELS, page_setup, summary, table  # noqa: E402

from src.monitoring.audit_log import AuditLog  # noqa: E402
from src.monitoring.model_registry import ModelRegistry  # noqa: E402

page_setup("Model Monitoring and Fairness")
ph10 = summary("phase10_monitoring_summary.json")
st.caption("Offline comparison: the dataset has no dates, so this is not live production "
           "monitoring. Reference = training split; comparison = current portfolio.")

tabs = st.tabs(["Drift", "Performance", "Fairness", "Registry & audit log"])
with tabs[0]:
    d = ph10["drift"]
    c = st.columns(4)
    lv = d["feature_levels"]
    c[0].metric("Features: stable / moderate / major",
                f"{lv.get('stable', 0)} / {lv.get('moderate', 0)} / {lv.get('major', 0)}")
    c[1].metric("Control (train vs test) major", d["control_levels"].get("major", 0))
    c[2].metric("Model-score PSI", f"{d['prediction_psi']:.4f}", d["prediction_psi_level"],
                delta_color="off", delta_arrow="off")
    c[3].metric("Review recommended", "YES" if d["review"]["review_recommended"] else "no")
    st.write(d["review"]["action"])
    schema = ph10["schema"]
    if schema["ok"]:
        st.success("Schema check passed: all model features present with matching types.")
    else:
        st.error(f"Schema problems: {schema}")
    df = table("drift_features_portfolio.csv")
    top = df.head(15).iloc[::-1]
    fig = px.bar(top, x=["psi", "psi_control"], y="feature", orientation="h", barmode="group")
    fig.add_vline(x=0.10, line_dash="dot")
    fig.add_vline(x=0.25, line_dash="dot")
    fig.update_layout(height=520, xaxis_title="PSI (dotted: 0.10 and 0.25 rule of thumb)",
                      yaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df[["feature", "type", "psi", "psi_control", "level", "ks_statistic",
                     "missing_reference", "missing_new", "shap_rank"]],
                 hide_index=True, width="stretch", height=300)
    st.dataframe(table("drift_risk_bands.csv"), hide_index=True)
with tabs[1]:
    perf = ph10["performance"]
    rows = [{"split": k, **{m: perf[k][m] for m in ["roc_auc", "pr_auc", "brier",
                                                     "ece_quantile"]}}
            for k in ["validation", "test"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.warning(f"Portfolio performance: {perf['portfolio']}. Data drift does not show whether "
               "the model has become less accurate; that needs outcome labels.")
with tabs[2]:
    st.caption("Labelled test split, deployed model at the operating threshold. Disparities "
               "are measured; this does not establish that the model is fair, or any legal "
               "compliance.")
    fm = table("fairness_metrics_test.csv")
    attr = st.radio("Attribute", sorted(fm["attribute"].unique()), horizontal=True)
    f = fm[(fm["attribute"] == attr) & fm["reliable"]]
    metric = st.selectbox("Metric", ["flag_rate", "tpr", "fpr"])
    fig = px.bar(f, x="group", y=metric, error_y=f[f"{metric}_ci_high"] - f[metric],
                 error_y_minus=f[metric] - f[f"{metric}_ci_low"], text_auto=".1%")
    fig.update_layout(height=340, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(fm[fm["attribute"] == attr][["group", "n", "base_rate", "mean_pd",
                                              "calibration_gap", "flag_rate", "tpr", "fpr",
                                              "precision", "auc", "reliable"]],
                 hide_index=True, width="stretch")
    st.markdown("**Differences vs the largest group (95% CI)**")
    diff = table("fairness_differences_test.csv")
    st.dataframe(diff[diff["attribute"] == attr], hide_index=True, width="stretch")
    st.markdown("**Does the model use gender directly?** Flip test (F <-> M, everything else "
                "unchanged):")
    st.dataframe(table("fairness_gender_flip_test.csv"), hide_index=True)
    st.markdown("**Experiment, not deployed: LightGBM retrained without CODE_GENDER**")
    st.dataframe(table("fairness_without_gender_test.csv")[
        ["model", "group", "n", "mean_pd", "calibration_gap", "flag_rate", "tpr", "fpr"]],
        hide_index=True, width="stretch")
    wg = ph10["fairness"]["without_gender"]
    st.caption(f"Test ROC-AUC: deployed {wg['deployed_test']['roc_auc']:.4f}, without gender "
               f"{wg['test']['roc_auc']:.4f}. Removing the variable narrows error-rate gaps "
               "but worsens calibration for men: a trade-off for decision-makers.")
with tabs[3]:
    reg = ModelRegistry(MODELS / "registry.json", MODELS)
    entries = reg.load()
    st.dataframe(pd.DataFrame([{k: e.get(k) for k in ["model_version", "status", "phase",
                                                     "calibration", "threshold", "created_at"]}
                               for e in entries]), hide_index=True, width="stretch")
    if st.button("Verify model files against registered SHA-256"):
        st.dataframe(pd.DataFrame([{"model_version": e["model_version"],
                                    "file_matches_registry": reg.verify(e["model_version"])}
                                   for e in entries]), hide_index=True)
    log = AuditLog(MODELS / "audit_log.jsonl")
    check = log.verify()
    (st.success if check["valid"] else st.error)(f"Audit log chain check: {check}")
    st.dataframe(pd.DataFrame([{k: e[k] for k in ["seq", "timestamp", "event", "actor"]}
                               for e in log.entries()]), hide_index=True)
