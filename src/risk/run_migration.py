"""Phase 9b: observed delinquency-state migration of bureau credits
(training-split applicants only).

Outputs:
- reports/tables/migration_counts_1m.csv, migration_matrix_1m.csv,
  migration_matrix_12m_markov.csv, migration_matrix_12m_observed.csv
- reports/figures/phase9_migration_1m.png
- reports/experiment_results/phase9_migration_summary.json

Run from the project root:  python -m scripts.run_migration
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
from src.data.load_data import read_table
from src.risk.rating_migration import (
    matrix_power,
    transition_counts,
    transition_matrix,
    wilson_interval,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    mcfg = cfg["rating_migration"]
    states, k = mcfg["states"], mcfg["horizon_check_months"]
    tables, figures = project_path(cfg, "tables"), project_path(cfg, "figures")
    splits = pd.read_parquet(project_path(cfg, "processed_data") / "splits.parquet")
    train_ids = set(splits.loc[splits["split"] == "train", "SK_ID_CURR"])

    log("Loading bureau data")
    bureau = read_table("bureau", cfg, usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
    keep = set(bureau.loc[bureau["SK_ID_CURR"].isin(train_ids), "SK_ID_BUREAU"])
    bb = read_table("bureau_balance", cfg)
    bb = bb[bb["SK_ID_BUREAU"].isin(keep)].reset_index(drop=True)
    log(f"{len(bb):,} monthly records for {bb['SK_ID_BUREAU'].nunique():,} credits")

    counts1 = transition_counts(bb, states, 1)
    P1 = transition_matrix(counts1)
    counts_k = transition_counts(bb, states, k)
    Pk_obs = transition_matrix(counts_k)
    Pk_markov = matrix_power(P1, k)
    counts1.to_csv(tables / "migration_counts_1m.csv")
    P1.to_csv(tables / "migration_matrix_1m.csv")
    Pk_markov.to_csv(tables / f"migration_matrix_{k}m_markov.csv")
    Pk_obs.to_csv(tables / f"migration_matrix_{k}m_observed.csv")

    diag = np.diag(counts1.to_numpy())
    totals = counts1.sum(axis=1).to_numpy()
    lo, hi = wilson_interval(diag, totals)
    stay = pd.DataFrame({"state": states, "transitions_observed": totals,
                         "stay_probability": diag / np.where(totals > 0, totals, np.nan),
                         "ci_low": lo, "ci_high": hi})
    delinquent = ["1", "2", "3", "4", "5"]
    worsen = {s: float(P1.loc[s, [t for t in delinquent if t > s]].sum())
              for s in ["0", "1", "2", "3", "4"]}
    cure = {s: float(P1.loc[s, "0"]) for s in delinquent}
    markov_gap = (Pk_markov - Pk_obs).abs()
    gap_rows = {s: float(markov_gap.loc[s].max()) for s in states}

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(P1.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(len(states)), yticks=range(len(states)), xticklabels=states,
           yticklabels=states, xlabel="Status next month", ylabel="Status this month",
           title="Observed 1-month transitions, bureau credits")
    for i in range(len(states)):
        for j in range(len(states)):
            v = P1.iloc[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures / "phase9_migration_1m.png", dpi=150)
    plt.close(fig)

    summary = {
        "population": "bureau credits of training-split applicants (bureau_balance)",
        "monthly_records": int(len(bb)), "credits": int(bb["SK_ID_BUREAU"].nunique()),
        "one_month_transitions": int(counts1.to_numpy().sum()),
        "stay_probabilities": stay.to_dict("records"),
        "prob_worsen_next_month": worsen, "prob_cure_to_0_next_month": cure,
        "markov_check": {"horizon_months": k,
                         "max_abs_gap_by_from_state": gap_rows},
        "figure": str((figures / "phase9_migration_1m.png").relative_to(PROJECT_ROOT)),
        "runtime_seconds": round(time.time() - start, 1),
    }
    (project_path(cfg, "experiment_results") / "phase9_migration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), "utf-8")
    log(f"Done in {summary['runtime_seconds']} s")
    return summary


def main() -> int:
    run(load_config())
    return 0
