"""Phase 7b: score the current portfolio and generate early-warning alerts.

"Current portfolio" = application_test (48,744 unlabelled applicants). The rules are
also measured on labelled data: red flags on the VALIDATION split, final alert
levels on the TEST split (nothing is tuned on test).

Outputs:
- data/processed/scored_portfolio.parquet          every portfolio borrower's score
- data/processed/alerts/current_alerts.parquet     this run's alerts
- data/processed/alerts/alert_history.parquet      de-duplicated history (AlertStore)
- data/processed/alerts/SIMULATED_deterioration.parquet  synthetic demo, labelled as such
- reports/tables/early_warning_flags_validation.csv, early_warning_levels_test.csv,
  early_warning_portfolio_summary.csv
- reports/experiment_results/phase7_early_warning_summary.json

Run from the project root:  python -m scripts.run_early_warning
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.explainability.shap_explainer import ShapExplainer
from src.models.predict import load_model, predict_proba
from src.models.train import load_modelling_data
from src.risk.borrower_risk import assign_risk_band, score_portfolio
from src.risk.early_warning import (
    LEVELS,
    AlertStore,
    compare_snapshots,
    evaluate_flags,
    final_level,
    generate_alerts,
    pd_level,
    resolve_levels,
    top_drivers_text,
)
from src.simulation.scenarios import scale_income

SIM_SHARE, SIM_INCOME_FACTOR = 0.10, 0.80  # synthetic demo: 10% of borrowers lose 20% income


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def flag_table(df: pd.DataFrame, flags: pd.DataFrame, y: np.ndarray, p_idx: np.ndarray,
               texts: dict[str, str]) -> pd.DataFrame:
    base = y.mean()
    rows = []
    for name in flags.columns:
        f = flags[name].to_numpy()
        green = p_idx == 0
        rows.append({
            "flag": name, "description": texts[name],
            "prevalence": float(f.mean()), "borrowers": int(f.sum()),
            "target_rate_flagged": float(y[f].mean()) if f.any() else None,
            "lift_vs_base": float(y[f].mean() / base) if f.any() else None,
            "green_by_pd_and_flagged": int((f & green).sum()),
            "target_rate_green_flagged": float(y[f & green].mean()) if (f & green).any() else None,
            "target_rate_green_unflagged": float(y[~f & green].mean()),
        })
    return pd.DataFrame(rows)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    ew, bands, seed = cfg["early_warning"], cfg["risk_bands"], cfg["project"]["random_seed"]
    models_dir = project_path(cfg, "models")
    processed, tables = project_path(cfg, "processed_data"), project_path(cfg, "tables")
    current = json.loads((models_dir / "current_model.json").read_text(encoding="utf-8"))
    model, _ = load_model(models_dir / current["model_file"])
    thresholds = resolve_levels(ew["levels"], current["threshold"])
    texts = {k: v["text"] for k, v in ew["flags"].items()}
    run_time = datetime.now().isoformat(timespec="seconds")
    as_of = f"application date (day 0 of the dataset); scored {run_time}"
    log(f"Model {current['model_version']}; thresholds {thresholds}")

    # 1. Measure flags (validation) and final levels (test) on labelled data
    data = load_modelling_data(cfg)
    df, target = data["df"], data["features"]["target"]
    evaluation = {}
    for split in ("validation", "test"):
        d = df[df["split"] == split].reset_index(drop=True)
        y = d[target].to_numpy()
        p = predict_proba(model, d.drop(columns=[target, "split"]))
        flags = evaluate_flags(d, ew["flags"])
        p_idx = pd_level(p, thresholds)
        lvl = final_level(p_idx, flags.any(axis=1).to_numpy(), ew["flag_escalation"])
        evaluation[split] = {"y": y, "p_idx": p_idx, "lvl": lvl, "flags": flags, "d": d}
    v = evaluation["validation"]
    flags_val = flag_table(v["d"], v["flags"], v["y"], v["p_idx"], texts)
    flags_val.to_csv(tables / "early_warning_flags_validation.csv", index=False)

    t = evaluation["test"]
    level_rows = []
    for i, name in enumerate(LEVELS):
        for kind, arr in (("pd_only", t["p_idx"]), ("pd_plus_flags", t["lvl"])):
            m = arr == i
            at_or_above = arr >= i
            level_rows.append({
                "level": name, "rule": kind, "borrowers": int(m.sum()), "share": float(m.mean()),
                "target_rate": float(t["y"][m].mean()) if m.any() else None,
                "share_of_all_difficulties_at_or_above": float(
                    t["y"][at_or_above].sum() / t["y"].sum()),
                "share_of_borrowers_at_or_above": float(at_or_above.mean()),
            })
    levels_test = pd.DataFrame(level_rows)
    levels_test.to_csv(tables / "early_warning_levels_test.csv", index=False)

    # 2. Score the current (unlabelled) portfolio and raise alerts
    log("Scoring portfolio")
    portfolio = pd.read_parquet(processed / "portfolio_test.parquet")
    scored = score_portfolio(model, portfolio, current, bands, run_time)
    scored.to_parquet(processed / "scored_portfolio.parquet", index=False)
    flags = evaluate_flags(portfolio, ew["flags"])
    p_idx = pd_level(scored["probability"], thresholds)
    lvl = final_level(p_idx, flags.any(axis=1).to_numpy(), ew["flag_escalation"])
    alerted = np.flatnonzero(lvl > 0)
    log(f"Explaining {len(alerted):,} alerted borrowers")
    drivers = pd.Series([""] * len(portfolio))
    if len(alerted):
        exp = ShapExplainer(model).explain(portfolio.iloc[alerted])
        drivers.iloc[alerted] = top_drivers_text(exp.values, list(exp.feature_names), 3)
    alerts = generate_alerts(scored, flags, thresholds, ew, current["model_version"], as_of,
                             drivers=drivers)
    alert_dir = processed / "alerts"
    alert_dir.mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(alert_dir / "current_alerts.parquet", index=False)
    store_result = AlertStore(alert_dir / "alert_history.parquet").upsert(alerts, run_time)
    summary_rows = []
    for name in LEVELS:
        m = alerts["level"] == name
        summary_rows.append({
            "level": name,
            "borrowers": int(m.sum()) if name != "green" else int((lvl == 0).sum()),
            "share_of_portfolio": float((lvl == LEVELS.index(name)).mean()),
            "mean_probability": float(scored["probability"][lvl == LEVELS.index(name)].mean()),
            "with_red_flags": int((alerts.loc[m, "n_flags"] > 0).sum()) if name != "green"
            else int(flags.any(axis=1).to_numpy()[lvl == 0].sum()),
        })
    port_summary = pd.DataFrame(summary_rows)
    port_summary.to_csv(tables / "early_warning_portfolio_summary.csv", index=False)

    # 3. SYNTHETIC deterioration demo (never mixed into real alert history)
    log("Synthetic deterioration demo")
    rng = np.random.default_rng(seed)
    shocked = rng.random(len(portfolio)) < SIM_SHARE
    factor = np.where(shocked, SIM_INCOME_FACTOR, 1.0)
    sim = scale_income(portfolio, factor)
    p_sim = predict_proba(model, sim)
    lvl_sim = final_level(pd_level(p_sim, thresholds),
                          evaluate_flags(sim, ew["flags"]).any(axis=1).to_numpy(),
                          ew["flag_escalation"])
    ids = portfolio["SK_ID_CURR"]
    before = pd.DataFrame({"SK_ID_CURR": ids, "level": np.array(LEVELS)[lvl],
                           "probability": scored["probability"]})
    after = pd.DataFrame({"SK_ID_CURR": ids, "level": np.array(LEVELS)[lvl_sim],
                          "probability": p_sim})
    det = compare_snapshots(before, after)
    det.insert(0, "data_type", "SYNTHETIC")
    det["scenario"] = (f"SIMULATED: {SIM_SHARE:.0%} of borrowers' income x {SIM_INCOME_FACTOR} "
                       f"(seed {seed}); not observed behaviour")
    det.to_parquet(alert_dir / "SIMULATED_deterioration.parquet", index=False)
    sim_summary = {
        "data_type": "SYNTHETIC", "shocked_borrowers": int(shocked.sum()),
        "level_worsened": int(len(det)),
        "worsened_among_shocked": int(det["SK_ID_CURR"].isin(ids[shocked]).sum()),
        "worsened_among_unshocked": int((~det["SK_ID_CURR"].isin(ids[shocked])).sum()),
        "transitions": det["change"].value_counts().to_dict(),
        "mean_pd_change_shocked": float((p_sim - scored["probability"].to_numpy())[shocked].mean()),
    }

    summary = {
        "model_version": current["model_version"], "thresholds": thresholds,
        "flag_escalation": ew["flag_escalation"], "run_time": run_time,
        "flags_validation": flags_val.to_dict("records"),
        "levels_test": levels_test.to_dict("records"),
        "portfolio": {"borrowers": int(len(portfolio)), "alerts": int(len(alerts)),
                      "levels": port_summary.to_dict("records"),
                      "risk_bands": assign_risk_band(scored["probability"], bands["edges"],
                                                     bands["labels"]).value_counts().to_dict()},
        "alert_store": store_result, "simulation": sim_summary,
        "runtime_seconds": round(time.time() - start, 1),
    }
    out = project_path(cfg, "experiment_results")
    (out / "phase7_early_warning_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), "utf-8")
    log(f"Alerts: {len(alerts):,}; store {store_result}; done in {summary['runtime_seconds']} s")
    return summary


def main() -> int:
    run(load_config())
    return 0
