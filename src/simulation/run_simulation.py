"""Phase 9a: Monte Carlo loss distribution and stress scenarios for the portfolio.

Portfolio = application_test (48,744 unlabelled applicants), deployed model,
EAD = AMT_CREDIT, LGD and rho = ASSUMPTIONS from config.

Outputs:
- reports/tables/mc_baseline.csv, mc_rho_sensitivity.csv, stress_results.csv
- reports/figures/phase9_loss_distribution.png, phase9_stress.png
- data/processed/mc_baseline_losses.parquet (simulated losses, for the dashboard)
- reports/experiment_results/phase9_simulation_summary.json

Run from the project root:  python -m scripts.run_simulation
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
from src.risk.borrower_risk import assign_risk_band
from src.simulation.monte_carlo import loss_statistics, resolve_rho, simulate_losses
from src.simulation.stress_testing import apply_scenario, check_scenario_output

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
    mc, st, bands = cfg["monte_carlo"], cfg["stress_testing"], cfg["risk_bands"]
    lgd = float(cfg["expected_loss"]["lgd"])
    ead_col = cfg["portfolio"]["exposure_column"]
    levels = mc["confidence_levels"]
    models_dir = project_path(cfg, "models")
    processed, tables = project_path(cfg, "processed_data"), project_path(cfg, "tables")
    figures = project_path(cfg, "figures")
    current = json.loads((models_dir / "current_model.json").read_text(encoding="utf-8"))
    model, _ = load_model(models_dir / current["model_file"])
    threshold = float(current["threshold"])

    port = pd.read_parquet(processed / "portfolio_test.parquet")
    p = predict_proba(model, port)
    ead = port[ead_col].to_numpy(dtype="float64")
    el = float((p * lgd * ead).sum())

    def mc_run(pd_: np.ndarray, rho: Any, n: int, lgd_: float) -> tuple[dict, dict]:
        sim = simulate_losses(pd_, ead, lgd_, rho, n, mc["seed"], mc["chunk_size"])
        stats = loss_statistics(sim["losses"], float((pd_ * lgd_ * ead).sum()), levels)
        stats["mean_defaults"] = float(sim["defaults"].mean())
        stats["defaults_p99"] = float(np.quantile(sim["defaults"], 0.99))
        return stats, sim

    # 1. Baseline Monte Carlo
    log(f"Baseline Monte Carlo: {mc['n_simulations']} simulations, rho = {mc['rho']}")
    base, sim = mc_run(p, mc["rho"], mc["n_simulations"], lgd)
    base["mean_vs_analytic_in_se"] = (base["mean_loss"] - el) / base["mc_standard_error"]
    base["expected_defaults_analytic"] = float(p.sum())
    pd.DataFrame([base]).to_csv(tables / "mc_baseline.csv", index=False)
    pd.DataFrame({"simulation": np.arange(len(sim["losses"])), "loss": sim["losses"],
                  "defaults": sim["defaults"], "factor_z": sim["factor"]}).to_parquet(
        processed / "mc_baseline_losses.parquet", index=False)

    # 2. Correlation sensitivity
    rows = []
    for rho in mc["rho_sensitivity"]:
        log(f"rho sensitivity: {rho}")
        stats, _ = mc_run(p, rho, mc["n_simulations"], lgd)
        r = resolve_rho(rho, p)
        rows.append({"rho": str(rho), "mean_rho": float(np.mean(r)), **stats})
    rho_df = pd.DataFrame(rows)
    rho_df.to_csv(tables / "mc_rho_sensitivity.csv", index=False)

    # 3. Stress scenarios (feature perturbation, rescored by the model)
    stress_rows, stressed_p = [], {}
    for name, spec in st["scenarios"].items():
        log(f"Stress scenario: {name}")
        s_df = apply_scenario(port, spec, seed=mc["seed"])
        check_scenario_output(port, s_df)
        ps = predict_proba(model, s_df.drop(columns=["_job_loss"], errors="ignore"))
        stressed_p[name] = ps
        lgd_s = float(spec.get("lgd", lgd))
        stats, _ = mc_run(ps, mc["rho"], st["n_simulations"], lgd_s)
        b = pd.Series(assign_risk_band(ps, bands["edges"], bands["labels"])).value_counts()
        delta = ps - p
        stress_rows.append({
            "scenario": name, "spec": json.dumps(spec), "lgd": lgd_s,
            "mean_pd": float(ps.mean()), "expected_difficulties": float(ps.sum()),
            "above_threshold": int((ps >= threshold).sum()),
            "moderate_or_worse": int(b.get("Moderate", 0) + b.get("High", 0)
                                     + b.get("Critical", 0)),
            "high_or_critical": int(b.get("High", 0) + b.get("Critical", 0)),
            "mean_pd_change_pp": float(delta.mean() * 100),
            "share_pd_up_1pp": float((delta > 0.01).mean()),
            "share_pd_down": float((delta < 0).mean()),
            "analytic_el": stats["analytic_el"], "mc_mean_loss": stats["mean_loss"],
            **{k: stats[k] for k in stats if k.startswith(("var_", "es_"))},
            "job_loss_borrowers": int(s_df["_job_loss"].sum()) if "_job_loss" in s_df else 0,
        })
    stress = pd.DataFrame(stress_rows)
    base_row = stress.set_index("scenario").loc["baseline"]
    for col in ["expected_difficulties", "above_threshold", "analytic_el", "var_99", "var_99.9"]:
        stress[f"{col}_vs_baseline_pct"] = (stress[col] / base_row[col] - 1) * 100
    stress.to_csv(tables / "stress_results.csv", index=False)

    # 4. Figures
    figs = []
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(sim["losses"] / 1e6, bins=80, color="#4C72B0")
    for key, ls in [("var_99", "--"), ("var_99.9", ":")]:
        ax.axvline(base[key] / 1e6, color="#C44E52", ls=ls, label=f"{key} = {base[key]/1e6:,.0f}M")
    ax.axvline(el / 1e6, color="k", lw=1, label=f"analytic EL = {el/1e6:,.0f}M")
    ax.set(xlabel="Simulated portfolio loss (millions, dataset currency units)",
           ylabel="Simulations",
           title=f"Loss distribution: {mc['n_simulations']:,} simulations, "
                 f"rho = {mc['rho']} and LGD = {lgd:.0%} (both assumed)")
    ax.legend(fontsize=8)
    figs.append(_save(fig, figures / "phase9_loss_distribution.png"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].barh(stress["scenario"], stress["expected_difficulties_vs_baseline_pct"],
                 color="#4C72B0")
    axes[0].set(xlabel="Change in expected difficulties vs baseline (%)",
                title="Model response to each scenario")
    axes[1].barh(stress["scenario"], stress["var_99.9"] / 1e6, color="#C44E52", label="VaR 99.9%")
    axes[1].barh(stress["scenario"], stress["analytic_el"] / 1e6, color="#4C72B0", label="EL")
    axes[1].set(xlabel="Millions (dataset units)", title="EL and VaR 99.9% by scenario")
    axes[1].legend(fontsize=8)
    figs.append(_save(fig, figures / "phase9_stress.png"))

    summary = {
        "model_version": current["model_version"], "portfolio_borrowers": int(len(p)),
        "assumptions": {"rho": mc["rho"], "lgd": lgd, "ead": ead_col,
                        "n_simulations": mc["n_simulations"], "seed": mc["seed"],
                        "chunk_size": mc["chunk_size"],
                        "horizon": "that of TARGET (early payment difficulty), not one year"},
        "baseline": base, "rho_sensitivity": rho_df.to_dict("records"),
        "stress": stress.to_dict("records"), "figures": figs,
        "runtime_seconds": round(time.time() - start, 1),
    }
    (project_path(cfg, "experiment_results") / "phase9_simulation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), "utf-8")
    log(f"Done in {summary['runtime_seconds']} s")
    return summary


def main() -> int:
    run(load_config())
    return 0
