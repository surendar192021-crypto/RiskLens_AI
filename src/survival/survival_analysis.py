"""Survival analysis of PREVIOUS Home Credit loans (the current loan has no follow-up).

Study design (docs/survival_analysis.md):
- Unit:      one previous loan (SK_ID_PREV) of an applicant.
- Start:     the loan's first scheduled instalment (NUM_INSTALMENT_NUMBER == 1 must be
             observed, so loans that began before the data window are excluded).
- Event:     first instalment paid more than `dpd` days after its due date. The event
             time is that instalment's due day + dpd (when it became dpd days overdue).
- Censoring: loans without the event are censored at the due day of their last paid
             instalment (loan finished, or data ends at the current application date).
- Instalments with no recorded payment are not treated as events (their timing is
             unknown and many are simply due just before the data ends).

Survival probability S(t) here = probability a previous loan has NOT yet had a
30+ day late payment t days after its first instalment. It is NOT the probability
that the current applicant defaults, and NOT the model's PD.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.validate_data import require_columns

INST_KEYS = ["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]


def instalment_level(inst: pd.DataFrame) -> pd.DataFrame:
    """One row per scheduled instalment (partial payments combined)."""
    require_columns(inst, INST_KEYS + ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"],
                    "installments_payments")
    return (
        inst.groupby(INST_KEYS, sort=False)
        .agg(due_day=("DAYS_INSTALMENT", "min"), paid_day=("DAYS_ENTRY_PAYMENT", "max"))
        .reset_index()
    )


def loan_survival_table(per_inst: pd.DataFrame, dpd: int = 30) -> pd.DataFrame:
    """One row per previous loan: duration (days), event (0/1)."""
    require_columns(per_inst, INST_KEYS + ["due_day", "paid_day"], "instalment table")
    if dpd < 0:
        raise ValueError("dpd must be non-negative")
    starts = per_inst.groupby("SK_ID_PREV").agg(
        SK_ID_CURR=("SK_ID_CURR", "first"),
        first_number=("NUM_INSTALMENT_NUMBER", "min"),
        start_day=("due_day", "min"),
    )
    paid = per_inst[per_inst["paid_day"].notna()].copy()
    paid["late"] = (paid["paid_day"] - paid["due_day"]) > dpd
    last_paid = paid.groupby("SK_ID_PREV")["due_day"].max().rename("last_paid_due")
    first_late = (
        paid[paid["late"]].groupby("SK_ID_PREV")["due_day"].min().rename("first_late_due")
    )
    loans = starts.join(last_paid, how="inner").join(first_late, how="left")
    loans = loans[loans["first_number"] == 1]  # start of the schedule must be observed
    event = loans["first_late_due"].notna()
    duration = np.where(
        event,
        loans["first_late_due"] + dpd - loans["start_day"],
        loans["last_paid_due"] - loans["start_day"],
    )
    out = pd.DataFrame(
        {"SK_ID_CURR": loans["SK_ID_CURR"].to_numpy(), "duration": duration.astype("float64"),
         "event": event.astype("int8").to_numpy()},
        index=loans.index,
    ).reset_index()
    return out[out["duration"] >= 0].reset_index(drop=True)


def add_loan_covariates(loans: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """Attach characteristics of each previous loan, known when it was granted."""
    cols = ["SK_ID_PREV", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "CNT_PAYMENT", "NAME_YIELD_GROUP"]
    require_columns(prev, cols, "previous_application")
    p = prev[cols].copy()
    for c in ["NAME_CONTRACT_TYPE", "NAME_YIELD_GROUP"]:
        p[c] = p[c].astype("object").where(p[c].astype("object") != "XNA")
    p["LOG_CREDIT"] = np.log1p(p["AMT_CREDIT"].clip(lower=0))
    return loans.merge(p.drop(columns="AMT_CREDIT"), on="SK_ID_PREV", how="left",
                       validate="one_to_one")


def km_at_horizons(kmf: Any, horizons: list[int]) -> dict[int, float]:
    return {h: float(kmf.survival_function_at_times(h).iloc[0]) for h in horizons}


def cox_design(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric design matrix for Cox regression (complete cases, reference levels dropped)."""
    d = df.dropna(subset=["NAME_CONTRACT_TYPE", "LOG_CREDIT", "CNT_PAYMENT",
                          "NAME_YIELD_GROUP"]).copy()
    X = pd.get_dummies(d[["NAME_CONTRACT_TYPE", "NAME_YIELD_GROUP"]], drop_first=True,
                       dtype="float64")
    X["LOG_CREDIT"] = d["LOG_CREDIT"]
    X["CNT_PAYMENT"] = d["CNT_PAYMENT"]
    return pd.concat([d[["SK_ID_CURR", "duration", "event"]], X], axis=1)
