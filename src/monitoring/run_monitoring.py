"""Phase 10: drift monitoring, fairness audit, model registry and audit log.

- Drift: reference = TRAIN split; comparison = portfolio (application_test).
  Control = TRAIN vs TEST split (same population: shows what pure sampling noise looks like).
- Performance: only measurable where outcomes exist (validation vs test). The portfolio
  has no outcomes, so performance drift there is reported as not measurable.
- Fairness: labelled TEST split, deployed model and operating threshold.
  Experiments (not deployed): flip CODE_GENDER; retrain LightGBM without CODE_GENDER.
- Registry/audit: models/registry.json, models/audit_log.jsonl.

Run from the project root:  python -m scripts.run_monitoring
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config, project_path
from src.models.evaluate import evaluate_probabilities
from src.models.predict import load_model, predict_proba, save_model
from src.models.thresholds import max_f1_threshold
from src.models.train import boosting_pipeline, load_modelling_data
from src.monitoring.audit_log import AuditLog
from src.monitoring.drift import (
    classify_psi,
    feature_drift,
    psi_numeric,
    review_recommendation,
    schema_check,
)
from src.monitoring.fairness import difference_vs_reference, group_metrics, parity_ratios
from src.monitoring.model_registry import ModelRegistry
from src.risk.borrower_risk import band_distribution
from src.risk.portfolio_analytics import add_age_band

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _save(fig: Any, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT))


def bootstrap_registry(registry: ModelRegistry, current: dict[str, Any], models_dir: Path,
                       tables: Path) -> list[dict[str, Any]]:
    """Register the models created in Phases 4-5 with their recorded metrics."""
    p4 = json.loads((models_dir / "latest_phase4.json").read_text(encoding="utf-8"))
    val = pd.read_csv(tables / "model_comparison_validation.csv").set_index("experiment")
    test = pd.read_csv(tables / "final_test_metrics.csv").set_index("model")
    entries = []

    def metrics(exp: str) -> dict[str, Any]:
        out = {"validation": {k: float(val.loc[exp, k]) for k in ["roc_auc", "pr_auc", "brier"]}}
        label = f"{exp} (uncalibrated)"
        if label in test.index:
            out["test"] = {k: float(test.loc[label, k]) for k in ["roc_auc", "pr_auc", "brier"]}
        return out

    for exp in val.index:
        role = ("baseline" if exp == "logistic_regression"
                else "experiment")
        if exp in (p4["champion"], p4["challenger"]):
            continue  # registered below from their Phase 5 final files
        entries.append(registry.register({
            "model_version": f"{exp}-{p4['run_id']}", "model_file": f"{p4['run_id']}/{exp}.joblib",
            "status": role, "created_at": p4["run_id"].split("_", 1)[1],
            "experiment": exp, "phase": 4, "metrics": metrics(exp),
            "trained_on": "train split", "calibration": "none",
        }))
    entries.append(registry.register({
        "model_version": f"{p4['challenger']}-none-{current['model_file'].split('/')[0]}",
        "model_file": current["challenger_file"], "status": "challenger",
        "created_at": current["model_file"].split("/")[0].split("_", 1)[1],
        "experiment": p4["challenger"], "phase": 5, "metrics": metrics(p4["challenger"]),
        "trained_on": "train split", "calibration": current["calibration_method"],
    }))
    entries.append(registry.register({
        "model_version": current["model_version"], "model_file": current["model_file"],
        "status": "deployed", "created_at": current["model_file"].split("/")[0].split("_", 1)[1],
        "experiment": current["experiment"], "phase": 5, "metrics": metrics(current["experiment"]),
        "trained_on": "train split", "calibration": current["calibration_method"],
        "threshold": current["threshold"], "threshold_policy": current["threshold_policy"],
    }))
    return entries


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    mon, fair = cfg["monitoring"], cfg["fairness"]
    models_dir = project_path(cfg, "models")
    processed, tables = project_path(cfg, "processed_data"), project_path(cfg, "tables")
    figures = project_path(cfg, "figures")
    current = json.loads((models_dir / "current_model.json").read_text(encoding="utf-8"))
    model, _ = load_model(models_dir / current["model_file"])
    threshold = float(current["threshold"])
    registry = ModelRegistry(models_dir / "registry.json", models_dir)
    audit = AuditLog(models_dir / "audit_log.jsonl")
    run_id = f"phase10_{datetime.now():%Y%m%d_%H%M%S}"

    # 0. Registry
    log("Model registry")
    reg_entries = bootstrap_registry(registry, current, models_dir, tables)
    audit.log("registry_bootstrap", {"run_id": run_id, "registered": len(reg_entries),
                                     "deployed": current["model_version"]})
    integrity = {e["model_version"]: registry.verify(e["model_version"]) for e in registry.load()}

    data = load_modelling_data(cfg)
    df, feats = data["df"], data["features"]
    target, numeric, categorical = feats["target"], feats["numeric"], feats["categorical"]
    split = {s: df[df["split"] == s].drop(columns="split").reset_index(drop=True)
             for s in ["train", "validation", "test"]}
    portfolio = pd.read_parquet(processed / "portfolio_test.parquet")

    # 1. Drift
    log("Feature drift: train vs portfolio, and train vs test (control)")
    schema = schema_check(split["train"], portfolio, numeric + categorical)
    drift = feature_drift(split["train"], portfolio, numeric, categorical, mon["psi_bins"],
                          mon["psi_thresholds"])
    control = feature_drift(split["train"], split["test"], numeric, categorical,
                            mon["psi_bins"], mon["psi_thresholds"])
    drift = drift.merge(control[["feature", "psi"]].rename(columns={"psi": "psi_control"}),
                        on="feature")
    shap_rank = pd.read_csv(tables / "shap_global_importance_test.csv")[["feature", "shap_rank"]]
    drift = drift.merge(shap_rank, on="feature", how="left")
    drift.to_csv(tables / "drift_features_portfolio.csv", index=False)
    important = shap_rank.nsmallest(mon["important_features"], "shap_rank")["feature"].tolist()

    p = {s: predict_proba(model, split[s].drop(columns=[target])) for s in split}
    p["portfolio"] = predict_proba(model, portfolio)
    pred_psi = psi_numeric(pd.Series(p["train"]), pd.Series(p["portfolio"]), mon["psi_bins"])[0]
    pred_psi_control = psi_numeric(pd.Series(p["train"]), pd.Series(p["test"]),
                                   mon["psi_bins"])[0]
    bands = cfg["risk_bands"]
    band_cmp = band_distribution(p["train"], bands["edges"], bands["labels"])[["band", "share"]]
    band_cmp = band_cmp.rename(columns={"share": "train_share"})
    band_cmp["portfolio_share"] = band_distribution(p["portfolio"], bands["edges"],
                                                    bands["labels"])["share"].to_numpy()
    band_cmp.to_csv(tables / "drift_risk_bands.csv", index=False)
    review = review_recommendation(drift, pred_psi, important, mon["review_rule"])
    level_counts = drift["level"].value_counts().to_dict()
    control_counts = control["level"].value_counts().to_dict()
    audit.log("drift_check", {"run_id": run_id, "prediction_psi": pred_psi,
                              "feature_levels": level_counts,
                              "review_recommended": review["review_recommended"]})

    # 2. Performance stability where labels exist
    perf = {s: evaluate_probabilities(split[s][target], p[s]) for s in ["validation", "test"]}

    # 3. Fairness on the labelled test split
    log("Fairness audit")
    test = add_age_band(split["test"], cfg["portfolio"]["age_bands"])
    y = test[target].to_numpy()
    fair_tables, diffs, ratios = [], [], {}
    for attr in fair["groups"]:
        m = group_metrics(y, p["test"], test[attr], threshold, fair["min_group_size"])
        m.insert(0, "attribute", attr)
        ref = m.loc[m["n"].idxmax(), "group"]
        d = difference_vs_reference(m, ref)
        d.insert(0, "attribute", attr)
        fair_tables.append(m)
        diffs.append(d)
        ratios[attr] = parity_ratios(m)
    fair_df = pd.concat(fair_tables, ignore_index=True)
    fair_df.to_csv(tables / "fairness_metrics_test.csv", index=False)
    diff_df = pd.concat(diffs, ignore_index=True)
    diff_df.to_csv(tables / "fairness_differences_test.csv", index=False)

    # 3a. Flip test: how much does the deployed model depend on CODE_GENDER directly?
    flipped = split["test"].drop(columns=[target]).copy()
    g = flipped["CODE_GENDER"].astype("object")
    flipped["CODE_GENDER"] = g.map({"F": "M", "M": "F"}).where(g.notna())
    p_flip = predict_proba(model, flipped)
    delta = p_flip - p["test"]
    flags_before, flags_after = p["test"] >= threshold, p_flip >= threshold
    flip_rows = []
    for grp in ["F", "M"]:
        mk = (g == grp).to_numpy()
        flip_rows.append({
            "original_gender": grp, "borrowers": int(mk.sum()),
            "mean_pd_change_pp": float(delta[mk].mean() * 100),
            "mean_abs_pd_change_pp": float(np.abs(delta[mk]).mean() * 100),
            "newly_flagged": int((~flags_before & flags_after & mk).sum()),
            "no_longer_flagged": int((flags_before & ~flags_after & mk).sum()),
        })
    flip_df = pd.DataFrame(flip_rows)
    flip_df.to_csv(tables / "fairness_gender_flip_test.csv", index=False)

    # 3b. Retrain without the sensitive attribute (comparison only, NOT deployed)
    drop = fair["retrain_without"]
    log(f"Retraining LightGBM without {drop} (experiment)")
    num2 = [c for c in numeric if c not in drop]
    cat2 = [c for c in categorical if c not in drop]
    lgb_params = cfg["modelling"]["params"]["lightgbm"]
    tr, va = split["train"], split["validation"]
    exp_model, best = boosting_pipeline(
        "lightgbm", num2, cat2, lgb_params, 1.0, cfg["project"]["random_seed"],
        cfg["modelling"]["early_stopping_rounds"], tr.drop(columns=[target]), tr[target],
        va.drop(columns=[target]), va[target])
    exp_name = f"lightgbm_without_{'_'.join(drop)}"
    exp_dir = models_dir / run_id
    save_model(exp_model, {"experiment": exp_name, "dropped": drop, "best_iteration": best,
                           "numeric": num2, "categorical": cat2, "status": "experiment"},
               exp_dir, exp_name)
    p_exp_val = predict_proba(exp_model, va.drop(columns=[target]))
    t_exp, _ = max_f1_threshold(va[target], p_exp_val)
    p_exp = predict_proba(exp_model, split["test"].drop(columns=[target]))
    exp_perf = evaluate_probabilities(y, p_exp)
    exp_fair = group_metrics(y, p_exp, test["CODE_GENDER"], t_exp, fair["min_group_size"])
    exp_fair.insert(0, "model", exp_name)
    dep_fair = fair_df[fair_df["attribute"] == "CODE_GENDER"].drop(columns="attribute").copy()
    dep_fair.insert(0, "model", "deployed")
    pd.concat([dep_fair, exp_fair], ignore_index=True).to_csv(
        tables / "fairness_without_gender_test.csv", index=False)
    registry.register({
        "model_version": f"{exp_name}-{run_id}", "model_file": f"{run_id}/{exp_name}.joblib",
        "status": "experiment", "created_at": run_id.split("_", 1)[1], "experiment": exp_name,
        "phase": 10, "trained_on": "train split", "calibration": "none",
        "threshold": t_exp, "metrics": {"test": {k: exp_perf[k] for k in
                                                 ["roc_auc", "pr_auc", "brier"]}},
    })
    audit.log("fairness_audit", {"run_id": run_id, "groups": fair["groups"],
                                 "parity_ratios": ratios, "experiment": exp_name})

    # 4. Figures
    figs = []
    top = drift.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["psi"], color="#C44E52", label="train vs portfolio")
    ax.barh(top["feature"], top["psi_control"], color="#4C72B0", label="train vs test (control)")
    for v in mon["psi_thresholds"].values():
        ax.axvline(v, color="k", ls=":", lw=0.8)
    ax.set(xlabel="PSI", title="Largest feature shifts (dotted: 0.10 and 0.25 rule of thumb)")
    ax.legend(fontsize=8)
    figs.append(_save(fig, figures / "phase10_drift_top_features.png"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    age = fair_df[fair_df["attribute"] == "AGE_BAND"]
    for ax, metric in zip(axes, ["flag_rate", "tpr", "fpr"], strict=True):
        err = [age[metric] - age[f"{metric}_ci_low"], age[f"{metric}_ci_high"] - age[metric]]
        ax.bar(age["group"], age[metric], yerr=err, color="#4C72B0", capsize=3)
        ax.set(title=f"{metric} by age band (test, 95% CI)")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    figs.append(_save(fig, figures / "phase10_fairness_age.png"))

    chain = audit.verify()
    summary = {
        "run_id": run_id, "model_version": current["model_version"],
        "registry": {"entries": len(registry.load()), "integrity": integrity,
                     "deployed": registry.deployed()["model_version"]},
        "audit_log": chain,
        "schema": schema,
        "drift": {"feature_levels": level_counts, "control_levels": control_counts,
                  "prediction_psi": pred_psi,
                  "prediction_psi_level": classify_psi(pred_psi, mon["psi_thresholds"]),
                  "prediction_psi_control": pred_psi_control,
                  "mean_pd_train": float(p["train"].mean()),
                  "mean_pd_portfolio": float(p["portfolio"].mean()),
                  "top_features": drift.head(10)[["feature", "psi", "psi_control", "level",
                                                  "shap_rank"]].to_dict("records"),
                  "review": review},
        "performance": {"validation": perf["validation"], "test": perf["test"],
                        "portfolio": "not measurable: no outcome labels"},
        "fairness": {"metrics": fair_df.to_dict("records"), "parity_ratios": ratios,
                     "gender_flip": flip_df.to_dict("records"),
                     "flags_changed_by_flip": int((flags_before != flags_after).sum()),
                     "without_gender": {"name": exp_name, "best_iteration": best,
                                        "threshold": t_exp, "test": exp_perf,
                                        "deployed_test": perf["test"]}},
        "figures": figs, "runtime_seconds": round(time.time() - start, 1),
    }
    (project_path(cfg, "experiment_results") / "phase10_monitoring_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), "utf-8")
    log(f"Done in {summary['runtime_seconds']} s; audit chain {chain}")
    return summary


def main() -> int:
    run(load_config())
    return 0
