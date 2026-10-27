"""Phase 7a: survival study of previous Home Credit loans of TRAINING-split applicants.

Outputs:
- reports/tables/survival_km_horizons.csv, survival_cox_summary.csv, survival_ph_test.csv
- reports/figures/phase7_km_overall.png, phase7_km_by_target.png, phase7_cox_hazard_ratios.png
- reports/experiment_results/phase7_survival_summary.json

Run from the project root:  python -m scripts.run_survival
"""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index

from src.config import PROJECT_ROOT, load_config, project_path
from src.data.load_data import read_table
from src.survival.survival_analysis import (
    add_loan_covariates,
    cox_design,
    instalment_level,
    km_at_horizons,
    loan_survival_table,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _save(fig: Any, path: Any) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT))


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    scfg = cfg["survival"]
    seed = cfg["project"]["random_seed"]
    horizons = scfg["horizons_days"]
    tables, figures = project_path(cfg, "tables"), project_path(cfg, "figures")
    processed = project_path(cfg, "processed_data")

    splits = pd.read_parquet(processed / "splits.parquet")
    labels = pd.read_parquet(processed / "modelling_train.parquet",
                             columns=["SK_ID_CURR", "TARGET"])
    train_ids = splits.loc[splits["split"] == "train", "SK_ID_CURR"]
    target = labels.set_index("SK_ID_CURR")["TARGET"]

    log("Building loan-level survival table")
    inst = read_table("installments_payments", cfg)
    per = instalment_level(inst)
    del inst
    gc.collect()
    loans = loan_survival_table(per, dpd=scfg["dpd_event"])
    n_all_loans = int(per["SK_ID_PREV"].nunique())
    del per
    gc.collect()
    loans = loans[loans["SK_ID_CURR"].isin(train_ids)].reset_index(drop=True)
    prev = read_table("previous_application", cfg,
                      usecols=["SK_ID_PREV", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "CNT_PAYMENT",
                               "NAME_YIELD_GROUP"])
    loans = add_loan_covariates(loans, prev)
    loans["applicant_target"] = loans["SK_ID_CURR"].map(target).astype("int8")
    describe = {
        "loans_with_instalments_all_applicants": n_all_loans,
        "loans_in_study": int(len(loans)),
        "applicants_in_study": int(loans["SK_ID_CURR"].nunique()),
        "events": int(loans["event"].sum()),
        "event_share": float(loans["event"].mean()),
        "median_follow_up_days": float(loans["duration"].median()),
        "loans_per_applicant_median": float(loans.groupby("SK_ID_CURR").size().median()),
    }
    log(f"{describe['loans_in_study']:,} loans, {describe['events']:,} events")

    # Kaplan-Meier
    log("Kaplan-Meier")
    km_rows, curves = [], {}
    groups: dict[str, pd.Series] = {"all loans": pd.Series(True, index=loans.index)}
    for ct in ["Cash loans", "Consumer loans", "Revolving loans"]:
        groups[f"contract: {ct}"] = loans["NAME_CONTRACT_TYPE"] == ct
    groups["applicant TARGET = 0"] = loans["applicant_target"] == 0
    groups["applicant TARGET = 1"] = loans["applicant_target"] == 1
    for name, mask in groups.items():
        d = loans[mask]
        kmf = KaplanMeierFitter(label=name).fit(d["duration"], d["event"])
        curves[name] = kmf
        s = km_at_horizons(kmf, horizons)
        ci = kmf.confidence_interval_survival_function_
        row = {"group": name, "loans": int(len(d)), "events": int(d["event"].sum())}
        for h in horizons:
            row[f"S({h}d)"] = s[h]
            idx = ci.index.searchsorted(h, side="right") - 1
            row[f"S({h}d)_ci_low"] = float(ci.iloc[max(idx, 0), 0])
            row[f"S({h}d)_ci_high"] = float(ci.iloc[max(idx, 0), 1])
        km_rows.append(row)
    km_df = pd.DataFrame(km_rows)
    km_df.to_csv(tables / "survival_km_horizons.csv", index=False)
    lr = logrank_test(
        loans.loc[groups["applicant TARGET = 0"], "duration"],
        loans.loc[groups["applicant TARGET = 1"], "duration"],
        loans.loc[groups["applicant TARGET = 0"], "event"],
        loans.loc[groups["applicant TARGET = 1"], "event"],
    )

    figs = []
    xmax = 1460
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ["all loans", "contract: Cash loans", "contract: Consumer loans",
                 "contract: Revolving loans"]:
        curves[name].plot_survival_function(ax=ax, ci_show=True)
    ax.set(xlim=(0, xmax), xlabel="Days since first scheduled instalment",
           ylabel="Share with no 30+ day late payment yet",
           title="Previous Home Credit loans (training-split applicants)")
    figs.append(_save(fig, figures / "phase7_km_overall.png"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ["applicant TARGET = 0", "applicant TARGET = 1"]:
        curves[name].plot_survival_function(ax=ax, ci_show=True)
    ax.set(xlim=(0, xmax), xlabel="Days since first scheduled instalment",
           ylabel="Share with no 30+ day late payment yet",
           title=f"Previous loans by applicant's current TARGET (log-rank p = {lr.p_value:.1e})")
    figs.append(_save(fig, figures / "phase7_km_by_target.png"))

    # Cox proportional hazards on a sample of applicants (all their loans kept together)
    log("Cox proportional hazards")
    design = cox_design(loans)
    rng = np.random.default_rng(seed)
    applicants = design["SK_ID_CURR"].unique()
    rng.shuffle(applicants)
    per_app = design.groupby("SK_ID_CURR").size()
    cum = per_app.loc[applicants].cumsum().to_numpy()
    n_fit = int(np.searchsorted(cum, scfg["cox_sample_loans"]))
    fit_ids, eval_ids = set(applicants[:n_fit]), set(applicants[n_fit:2 * n_fit])
    fit_df = design[design["SK_ID_CURR"].isin(fit_ids)]
    eval_df = design[design["SK_ID_CURR"].isin(eval_ids)]
    cph = CoxPHFitter()
    cph.fit(fit_df, duration_col="duration", event_col="event", cluster_col="SK_ID_CURR",
            robust=True)
    cox = cph.summary[["coef", "exp(coef)", "se(coef)", "exp(coef) lower 95%",
                       "exp(coef) upper 95%", "p"]].reset_index()
    cox.to_csv(tables / "survival_cox_summary.csv", index=False)
    eval_c = concordance_index(eval_df["duration"],
                               -cph.predict_partial_hazard(eval_df).to_numpy(),
                               eval_df["event"])
    # lifelines' PH test must use the model's own training data, and is slow on 200k
    # rows, so the same specification is refitted on a random subsample for the test.
    ph_sample = fit_df.sample(n=min(scfg["ph_test_sample_loans"], len(fit_df)), random_state=seed)
    ph_data = ph_sample.drop(columns="SK_ID_CURR")
    cph_ph = CoxPHFitter().fit(ph_data, duration_col="duration", event_col="event")
    ph = proportional_hazard_test(cph_ph, ph_data, time_transform="rank")
    ph_df = ph.summary.reset_index().rename(columns={"index": "covariate"})
    ph_df.to_csv(tables / "survival_ph_test.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    cph.plot(hazard_ratios=True, ax=ax)
    ax.set_title("Cox hazard ratios, 95% CI (cluster-robust by applicant)")
    figs.append(_save(fig, figures / "phase7_cox_hazard_ratios.png"))

    summary = {
        "design": {"unit": "previous Home Credit loan", "event": f"first instalment paid more "
                   f"than {scfg['dpd_event']} days late", "population": "training-split applicants",
                   "time_origin": "first scheduled instalment"},
        "describe": describe, "km": km_rows,
        "logrank_target": {"statistic": float(lr.test_statistic), "p_value": float(lr.p_value)},
        "cox": {"fit_loans": int(len(fit_df)), "fit_applicants": len(fit_ids),
                "eval_loans": int(len(eval_df)),
                "concordance_train": float(cph.concordance_index_),
                "concordance_held_out": float(eval_c),
                "coefficients": cox.to_dict("records")},
        "ph_test": ph_df.to_dict("records"), "figures": figs,
        "runtime_seconds": round(time.time() - start, 1),
    }
    out = project_path(cfg, "experiment_results")
    (out / "phase7_survival_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                      "utf-8")
    log(f"Done in {summary['runtime_seconds']} s")
    return summary


def main() -> int:
    run(load_config())
    return 0
