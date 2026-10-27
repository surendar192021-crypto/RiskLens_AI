"""Phase 8: portfolio analytics and expected loss.

- Portfolio = application_test (48,744 unlabelled applicants), scored by the deployed model.
- Reality check = labelled TEST split: do summed PDs (and PD x EAD) match what happened,
  overall and per segment? (No model or threshold is chosen here.)

Outputs:
- reports/tables/portfolio_summary.csv, portfolio_bands.csv, portfolio_segments.csv,
  portfolio_segment_check_test.csv, el_lgd_sensitivity.csv, el_by_band.csv
- reports/figures/phase8_segments.png, phase8_el_lorenz.png
- data/processed/portfolio_el.parquet (borrower-level EL for the dashboard)
- reports/experiment_results/phase8_portfolio_summary.json

Run from the project root:  python -m scripts.run_portfolio
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config, project_path
from src.models.predict import load_model, predict_proba
from src.models.train import load_modelling_data
from src.risk.borrower_risk import score_portfolio
from src.risk.expected_loss import el_by_group, expected_loss, lgd_sensitivity
from src.risk.portfolio_analytics import (
    add_age_band,
    expected_count_interval,
    herfindahl,
    lorenz_curve,
    portfolio_summary,
    segment_table,
    top_share,
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
    pcfg, elcfg, bands = cfg["portfolio"], cfg["expected_loss"], cfg["risk_bands"]
    ead_col, lgd = pcfg["exposure_column"], float(elcfg["lgd"])
    models_dir = project_path(cfg, "models")
    processed, tables = project_path(cfg, "processed_data"), project_path(cfg, "tables")
    figures = project_path(cfg, "figures")
    current = json.loads((models_dir / "current_model.json").read_text(encoding="utf-8"))
    model, _ = load_model(models_dir / current["model_file"])
    threshold = float(current["threshold"])
    run_time = datetime.now().isoformat(timespec="seconds")

    # 1. Portfolio (unlabelled)
    log("Scoring portfolio")
    port = add_age_band(pd.read_parquet(processed / "portfolio_test.parquet"), pcfg["age_bands"])
    scored = score_portfolio(model, port, current, bands, run_time)
    p, ead = scored["probability"].to_numpy(), port[ead_col].to_numpy(dtype="float64")
    band = scored["risk_band"].astype(str).to_numpy()
    summary = portfolio_summary(p, ead, threshold, band)
    el = expected_loss(p, lgd, ead)
    summary.update({"lgd_assumed": lgd, "portfolio_el": float(el.sum()),
                    "el_rate_of_exposure": float(el.sum() / ead.sum())})
    pd.DataFrame([summary]).to_csv(tables / "portfolio_summary.csv", index=False)

    band_df = segment_table(pd.DataFrame({"risk_band": band}), "risk_band", p, ead,
                            max_levels=len(bands["labels"]))
    band_df = band_df.merge(el_by_group(band, p, lgd, ead)[["group", "el", "el_share"]],
                            left_on="segment", right_on="group").drop(columns="group")
    order = {lab: i for i, lab in enumerate(bands["labels"])}
    band_df = band_df.sort_values("segment", key=lambda s: s.map(order)).reset_index(drop=True)
    band_df.to_csv(tables / "portfolio_bands.csv", index=False)
    el_by_group(np.where(p >= threshold, "at/above threshold", "below threshold"), p, lgd,
                ead).to_csv(tables / "el_by_threshold.csv", index=False)

    seg_frames = [segment_table(port, s, p, ead, max_levels=pcfg["max_segment_levels"])
                  for s in pcfg["segments"]]
    segs = pd.concat(seg_frames, ignore_index=True)
    segs.to_csv(tables / "portfolio_segments.csv", index=False)
    hhi = {s: herfindahl(f["exposure_share"].to_numpy() / f["exposure_share"].sum())
           for s, f in zip(pcfg["segments"], seg_frames, strict=True)}

    sens = lgd_sensitivity(p, ead, elcfg["lgd_sensitivity"])
    sens.to_csv(tables / "el_lgd_sensitivity.csv", index=False)
    lor = lorenz_curve(el)
    conc = {
        "el_share_top_1pct": top_share(el, 0.01), "el_share_top_10pct": top_share(el, 0.10),
        "exposure_share_top_10pct": top_share(ead, 0.10),
        "expected_difficulties_share_top_10pct": top_share(p, 0.10),
        "hhi_exposure_by_segment": hhi,
    }
    pd.DataFrame({"SK_ID_CURR": port["SK_ID_CURR"], "probability": p, "risk_band": band,
                  "ead": ead, "lgd_assumed": lgd, "expected_loss": el}).to_parquet(
        processed / "portfolio_el.parquet", index=False)

    # 2. Reality check on the labelled test split
    log("Checking aggregates against observed outcomes (test split)")
    data = load_modelling_data(cfg)
    df, target = data["df"], data["features"]["target"]
    test = add_age_band(df[df["split"] == "test"].reset_index(drop=True), pcfg["age_bands"])
    yt = test[target].to_numpy()
    pt = predict_proba(model, test.drop(columns=[target, "split"]))
    et = test[ead_col].to_numpy(dtype="float64")
    interval = expected_count_interval(pt)
    check_overall = {
        "borrowers": int(len(yt)), "observed_difficulties": int(yt.sum()),
        **{f"{k}_difficulties": v for k, v in interval.items()},
        "observed_inside_independent_range": bool(
            interval["low_if_independent"] <= yt.sum() <= interval["high_if_independent"]),
        "expected_exposure_with_difficulty": float((pt * et).sum()),
        "observed_exposure_with_difficulty": float((yt * et).sum()),
    }
    check = pd.concat([segment_table(test, s, pt, et, y=yt, max_levels=pcfg["max_segment_levels"])
                       for s in pcfg["segments"]], ignore_index=True)
    check["relative_error"] = check["expected_minus_observed"] / check["observed_difficulties"]
    check.to_csv(tables / "portfolio_segment_check_test.csv", index=False)
    big = check[check["borrowers"] >= 1000]
    check_overall["segments_with_1000plus_borrowers"] = int(len(big))
    check_overall["max_abs_relative_error_1000plus"] = float(big["relative_error"].abs().max())

    # 3. Figures
    figs = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, var in zip(axes, ["NAME_INCOME_TYPE", "AGE_BAND"], strict=True):
        f = segs[segs["variable"] == var].sort_values("segment")
        ax.bar(f["segment"].astype(str), f["mean_pd"], color="#4C72B0")
        ax.axhline(summary["mean_pd"], color="k", ls="--", lw=0.8, label="portfolio mean PD")
        ax.set(title=f"Mean PD by {var}", ylabel="Mean estimated PD")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.legend(fontsize=7)
    figs.append(_save(fig, figures / "phase8_segments.png"))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(lor["share_of_borrowers"], lor["share_of_total"], label="expected loss")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="evenly spread")
    ax.set(xlabel="Share of borrowers (largest EL first)", ylabel="Share of portfolio EL",
           title=f"Concentration of expected loss (LGD = {lgd:.0%}, assumed)")
    ax.legend()
    figs.append(_save(fig, figures / "phase8_el_lorenz.png"))

    out = {
        "model_version": current["model_version"], "run_time": run_time,
        "ead_definition": f"{ead_col} (credit amount of the application; currency unstated; "
                          "assumed fully outstanding at default)",
        "lgd_definition": f"assumed constant {lgd} (no observed recoveries)",
        "summary": summary, "bands": band_df.to_dict("records"),
        "lgd_sensitivity": sens.to_dict("records"), "concentration": conc,
        "reality_check_test": check_overall, "figures": figs,
        "runtime_seconds": round(time.time() - start, 1),
    }
    (project_path(cfg, "experiment_results") / "phase8_portfolio_summary.json").write_text(
        json.dumps(out, indent=2, default=str), "utf-8")
    log(f"Done in {out['runtime_seconds']} s")
    return out


def main() -> int:
    run(load_config())
    return 0
