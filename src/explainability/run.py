"""Phase 6 report: global and local SHAP plus a counterfactual feasibility study,
all for the deployed model (models/current_model.json) on the TEST split.

Nothing here changes the model or chooses anything; it only explains it.

Outputs:
- reports/tables/shap_global_importance_test.csv, shap_example_borrowers.csv,
  counterfactual_evaluation_test.csv, counterfactual_summary_test.csv
- reports/figures/phase6_shap_beeswarm.png, phase6_shap_bar.png,
  phase6_waterfall_<case>.png
- reports/experiment_results/phase6_summary.json

Run from the project root:  python -m scripts.run_explainability
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src.config import PROJECT_ROOT, load_config, project_path
from src.explainability.counterfactuals import amount_bounds, lever_grid, search
from src.explainability.shap_explainer import ShapExplainer, numeric_view, plain_english
from src.models.predict import load_model, predict_proba
from src.models.train import load_modelling_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _save(path: Any) -> str:
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return str(path.relative_to(PROJECT_ROOT))


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    seed = cfg["project"]["random_seed"]
    ecfg, ccfg = cfg["explainability"], cfg["counterfactuals"]
    models_dir = project_path(cfg, "models")
    tables, figures = project_path(cfg, "tables"), project_path(cfg, "figures")
    current = json.loads((models_dir / "current_model.json").read_text(encoding="utf-8"))
    model, _ = load_model(models_dir / current["model_file"])
    threshold = float(current["threshold"])
    log(f"Explaining {current['model_version']} (threshold {threshold:.4f})")

    data = load_modelling_data(cfg)
    df, feats = data["df"], data["features"]
    target, id_col = feats["target"], feats["id"]
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    X_test = test.drop(columns=[target])
    p_test = predict_proba(model, X_test)
    portfolio_mean = float(p_test.mean())
    explainer = ShapExplainer(model)

    # 1. Global SHAP
    log("Global SHAP")
    sample = X_test.sample(n=min(ecfg["shap_sample_size"], len(X_test)), random_state=seed)
    g = explainer.global_importance(sample)
    g.to_csv(tables / "shap_global_importance_test.csv", index=False)
    rho = spearmanr(g["mean_abs_shap"], g["gain_share"]).statistic if "gain_share" in g else None
    exp = explainer.explain(sample)
    shap.plots.beeswarm(numeric_view(exp), max_display=ecfg["top_n"], show=False)
    plt.title(f"SHAP values, {len(sample):,} test borrowers (log-odds contribution)")
    fig_paths = [_save(figures / "phase6_shap_beeswarm.png")]
    top = g.head(ecfg["top_n"]).iloc[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    plt.xlabel("Mean |SHAP| (log-odds)")
    plt.title("Global feature importance (test sample)")
    fig_paths.append(_save(figures / "phase6_shap_bar.png"))

    # 2. Example borrowers (deterministic choices)
    log("Local explanations")
    order = np.argsort(p_test)
    above = np.where(p_test >= threshold)[0]
    cases = {
        "highest_risk": int(order[-1]),
        "just_above_threshold": int(above[np.argmin(p_test[above])]),
        "median_risk": int(order[len(order) // 2]),
    }
    cf_grid = lever_grid(ccfg)
    bounds = amount_bounds(train, tuple(ccfg["plausibility_quantiles"]))
    examples, example_rows = {}, []
    for case, i in cases.items():
        row = X_test.iloc[[i]]
        r = explainer.explain_borrower(row)
        shap.plots.waterfall(r["explanation"], max_display=12, show=False)
        plt.title(f"{case.replace('_', ' ')}: borrower {int(row[id_col].iloc[0])}, "
                  f"PD {r['probability']:.1%}")
        fig_paths.append(_save(figures / f"phase6_waterfall_{case}.png"))
        cf = search(model, row, threshold, *cf_grid, bounds)
        examples[case] = {
            "borrower_id": int(row[id_col].iloc[0]),
            "observed_target": int(test[target].iloc[i]),
            "probability": r["probability"],
            "text": plain_english(r, portfolio_mean, threshold),
            "top_contributions": r["contributions"].head(8).astype({"value": str})
            .to_dict("records"),
            "counterfactual": {"status": cf.status, "best": cf.best,
                               "best_achievable_probability": cf.best_achievable_probability,
                               "changes": cf.changes.to_dict("records")},
        }
        for rec in r["contributions"].head(8).itertuples():
            example_rows.append({"case": case, "borrower_id": examples[case]["borrower_id"],
                                 "probability": r["probability"], "feature": rec.feature,
                                 "value": str(rec.value), "shap": rec.shap})
    pd.DataFrame(example_rows).to_csv(tables / "shap_example_borrowers.csv", index=False)

    # 3. Counterfactual feasibility on flagged test borrowers
    log("Counterfactual feasibility study")
    rng = np.random.default_rng(seed)
    n = min(ccfg["evaluation_sample"], len(above))
    chosen = rng.choice(above, size=n, replace=False)
    rows = []
    for i in chosen:
        row = X_test.iloc[[i]]
        r = search(model, row, threshold, *cf_grid, bounds)
        rows.append({
            "borrower_id": int(row[id_col].iloc[0]), "observed_target": int(test[target].iloc[i]),
            "original_probability": r.original_probability, "status": r.status,
            "best_achievable_probability": r.best_achievable_probability,
            "credit_factor": r.best["credit_factor"] if r.best else None,
            "term_factor": r.best["term_factor"] if r.best else None,
            "cf_probability": r.best["probability"] if r.best else None,
            "has_annuity": bool(pd.notna(row["AMT_ANNUITY"].iloc[0])),
        })
    ev = pd.DataFrame(rows)
    ev.to_csv(tables / "counterfactual_evaluation_test.csv", index=False)
    ev["pd_group"] = pd.cut(ev["original_probability"], [threshold, 0.25, 0.4, 1.0],
                            right=False, include_lowest=True)
    by_group = ev.groupby("pd_group", observed=True).agg(
        borrowers=("status", "size"),
        found_rate=("status", lambda s: (s == "found").mean()),
        median_best_achievable=("best_achievable_probability", "median"),
    ).reset_index()
    by_group["pd_group"] = by_group["pd_group"].astype(str)
    found = ev[ev["status"] == "found"]

    def stat(fn: Any) -> float | None:
        return float(fn(found)) if len(found) else None

    summary_cf = {
        "sample_size": int(n), "flagged_test_borrowers": int(len(above)),
        "found_rate": float((ev["status"] == "found").mean()),
        "found_with_term_only": stat(lambda f: (f["credit_factor"] == 1.0).mean()),
        "found_needing_credit_cut": stat(lambda f: (f["credit_factor"] < 1.0).mean()),
        "median_credit_factor_found": stat(lambda f: f["credit_factor"].median()),
        "median_term_factor_found": stat(lambda f: f["term_factor"].median()),
        "median_probability_drop_found": stat(
            lambda f: (f["original_probability"] - f["cf_probability"]).median()),
        "median_best_achievable_not_found": float(
            ev.loc[ev["status"] == "not_found", "best_achievable_probability"].median())
        if (ev["status"] == "not_found").any() else None,
        "by_original_pd": by_group.to_dict("records"),
    }
    pd.DataFrame([{k: v for k, v in summary_cf.items() if k != "by_original_pd"}]).to_csv(
        tables / "counterfactual_summary_test.csv", index=False)

    summary = {
        "model_version": current["model_version"], "threshold": threshold,
        "portfolio_mean_pd_test": portfolio_mean, "base_value_log_odds": explainer.base_value,
        "shap_sample_size": int(len(sample)),
        "top_features": g.head(15)[["feature", "mean_abs_shap", "share_of_total",
                                    "gain_rank"]].to_dict("records"),
        "spearman_shap_vs_gain": rho, "examples": examples,
        "counterfactuals": summary_cf, "bounds": bounds, "figures": fig_paths,
        "runtime_seconds": round(time.time() - start, 1),
    }
    out = project_path(cfg, "experiment_results")
    (out / "phase6_summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    log(f"Done in {summary['runtime_seconds']} s")
    return summary


def main() -> int:
    run(load_config())
    return 0
