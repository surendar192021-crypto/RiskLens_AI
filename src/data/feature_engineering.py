"""Feature construction: application-level ratios and one-row-per-applicant
aggregates of each history table.

Rules followed here (see docs/dataset_audit_findings.md):
- No function uses TARGET, so aggregates can be built on train and test together.
- Every aggregate returns exactly one row per SK_ID_CURR, so joins cannot multiply rows.
- Divisions go through `safe_divide`: a zero or missing denominator gives missing, never inf.
- All history records are dated before the application (verified in the audit).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validate_data import DAYS_PLACEHOLDER, require_columns

ID = "SK_ID_CURR"
DAYS_PER_YEAR = 365.25
RECENT_DAYS = 365  # "last year" window for day-based histories
RECENT_MONTHS = 12  # MONTHS_BALANCE >= -12 for month-based histories
EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# Plain-English meaning of every engineered feature (used for docs/data_dictionary.md).
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # application
    "DAYS_EMPLOYED_PLACEHOLDER": "1 if DAYS_EMPLOYED held the 365243 placeholder (almost all pensioners)",
    "CREDIT_INCOME_RATIO": "AMT_CREDIT / AMT_INCOME_TOTAL",
    "ANNUITY_INCOME_RATIO": "AMT_ANNUITY / AMT_INCOME_TOTAL (repayment burden)",
    "CREDIT_ANNUITY_RATIO": "AMT_CREDIT / AMT_ANNUITY (rough proxy for loan length)",
    "GOODS_CREDIT_RATIO": "AMT_GOODS_PRICE / AMT_CREDIT",
    "AGE_YEARS": "Age at application in years (-DAYS_BIRTH / 365.25)",
    "EMPLOYED_YEARS": "Current employment length in years (missing for placeholder rows)",
    "EMPLOYED_TO_AGE_RATIO": "EMPLOYED_YEARS / AGE_YEARS",
    "INCOME_PER_FAMILY_MEMBER": "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS",
    "EXT_SOURCE_MEAN": "Mean of available EXT_SOURCE_1-3",
    "EXT_SOURCE_MIN": "Minimum of available EXT_SOURCE_1-3",
    "EXT_SOURCE_N_MISSING": "How many of EXT_SOURCE_1-3 are missing (0-3)",
    "DOCUMENT_COUNT": "Number of FLAG_DOCUMENT_* flags set",
    # source indicators
    "HAS_BUREAU": "1 if the applicant has any credit bureau record",
    "HAS_PREV": "1 if the applicant has any previous Home Credit application",
    "HAS_INST": "1 if the applicant has any previous instalment record",
    "HAS_POS": "1 if the applicant has any previous POS/cash loan monthly record",
    "HAS_CC": "1 if the applicant has any previous credit card monthly record",
    # bureau
    "BUREAU_COUNT": "Number of credit bureau credits",
    "BUREAU_ACTIVE_COUNT": "Number of active bureau credits",
    "BUREAU_CLOSED_COUNT": "Number of closed bureau credits",
    "BUREAU_BAD_DEBT_OR_SOLD_COUNT": "Number of bureau credits marked 'Bad debt' or 'Sold'",
    "BUREAU_DAYS_CREDIT_MAX": "Days before application of the most recent bureau credit",
    "BUREAU_DAYS_CREDIT_MIN": "Days before application of the oldest bureau credit",
    "BUREAU_ACTIVE_DEBT_SUM": "Total current debt on active bureau credits (currency 1 only)",
    "BUREAU_ACTIVE_CREDIT_SUM": "Total credit amount of active bureau credits (currency 1 only)",
    "BUREAU_ACTIVE_DEBT_RATIO": "BUREAU_ACTIVE_DEBT_SUM / BUREAU_ACTIVE_CREDIT_SUM",
    "BUREAU_OVERDUE_SUM": "Total currently overdue amount on bureau credits (currency 1 only)",
    "BUREAU_MAX_OVERDUE_MAX": "Largest overdue amount ever recorded on a bureau credit (currency 1 only)",
    "BUREAU_PROLONG_SUM": "Total number of times bureau credits were prolonged",
    "BUREAU_BB_WORST_STATUS_12M": "Worst monthly DPD bucket (0-5) on any bureau credit, MONTHS_BALANCE >= -12",
    "BUREAU_BB_DPD_CREDIT_COUNT": "Number of bureau credits with any month in DPD bucket 1-5",
    # previous applications
    "PREV_COUNT": "Number of previous Home Credit applications",
    "PREV_APPROVED_COUNT": "Number of previous applications approved",
    "PREV_REFUSED_COUNT": "Number of previous applications refused",
    "PREV_REFUSED_LAST_YEAR_COUNT": "Previous applications refused within 365 days before this application",
    "PREV_APPROVAL_RATE": "PREV_APPROVED_COUNT / PREV_COUNT",
    "PREV_REFUSAL_RATE": "PREV_REFUSED_COUNT / PREV_COUNT",
    "PREV_AMT_CREDIT_MEAN": "Mean final credit amount of previous applications",
    "PREV_AMT_ANNUITY_MEAN": "Mean annuity of previous applications",
    "PREV_CREDIT_TO_APPLICATION_MEAN": "Mean of (granted AMT_CREDIT / requested AMT_APPLICATION)",
    "PREV_DAYS_DECISION_MAX": "Days before application of the most recent previous decision",
    "PREV_CNT_PAYMENT_MEAN": "Mean planned number of payments on previous credits",
    # instalments
    "INST_COUNT": "Number of scheduled previous instalments (partial payments combined)",
    "INST_UNPAID_COUNT": "Scheduled instalments with no recorded payment",
    "INST_LATE_SHARE": "Share of paid instalments paid after the due day",
    "INST_LATE30_SHARE": "Share of paid instalments paid more than 30 days late",
    "INST_DAYS_LATE_MEAN": "Mean days late per paid instalment (early = 0)",
    "INST_DAYS_LATE_MAX": "Maximum days late on any instalment",
    "INST_UNDERPAID_SHARE": "Share of instalments where total paid < amount due (by more than 0.01)",
    "INST_SHORTFALL_SUM": "Total unpaid amount across instalments",
    "INST_LATE_SHARE_LAST_YEAR": "INST_LATE_SHARE for instalments due within 365 days before application",
    # POS / cash
    "POS_MONTHS_COUNT": "Number of monthly POS/cash loan records",
    "POS_LOAN_COUNT": "Number of distinct previous POS/cash loans",
    "POS_DPD_MAX": "Maximum days past due on POS/cash loans",
    "POS_DPD_DEF_MAX": "Maximum days past due (with tolerance) on POS/cash loans",
    "POS_DPD_MONTH_SHARE": "Share of POS/cash months with DPD > 0",
    "POS_DPD_MAX_LAST_YEAR": "Maximum POS/cash DPD with MONTHS_BALANCE >= -12",
    # credit card
    "CC_MONTHS_COUNT": "Number of monthly credit card records",
    "CC_UTILISATION_MEAN": "Mean of balance / credit limit (months with limit > 0)",
    "CC_UTILISATION_MAX": "Maximum of balance / credit limit",
    "CC_UTILISATION_MEAN_LAST_YEAR": "Mean utilisation with MONTHS_BALANCE >= -12",
    "CC_DPD_MAX": "Maximum days past due on credit cards",
    "CC_DPD_MONTH_SHARE": "Share of credit card months with DPD > 0",
    "CC_DRAWINGS_MEAN": "Mean monthly drawn amount on credit cards",
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, with missing (not inf) where the denominator is 0 or missing."""
    num = pd.to_numeric(numerator).astype("float64")
    den = pd.to_numeric(denominator).astype("float64")
    valid = den.notna() & (den != 0)
    out = num / den.where(valid)
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Application table
# ---------------------------------------------------------------------------

LOAN_RATIO_FEATURES = [
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_ANNUITY_RATIO", "GOODS_CREDIT_RATIO",
]


def loan_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """(Re)compute the ratios that depend on the loan amounts, in place.

    The single definition used both when building the dataset and when
    counterfactual or stress scenarios change AMT_CREDIT / AMT_ANNUITY /
    AMT_GOODS_PRICE, so derived features can never fall out of sync.
    """
    df["CREDIT_INCOME_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_ANNUITY_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["GOODS_CREDIT_RATIO"] = safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    return df


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio and life-cycle features. Expects `clean_application` to have run.

    DAYS_BIRTH and DAYS_EMPLOYED are replaced by AGE_YEARS and EMPLOYED_YEARS
    (same information, readable units).
    """
    require_columns(
        df,
        ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH",
         "DAYS_EMPLOYED", "CNT_FAM_MEMBERS", *EXT_SOURCES],
        "application table",
    )
    if (df["DAYS_EMPLOYED"] == DAYS_PLACEHOLDER).any():
        raise ValueError("DAYS_EMPLOYED still contains the placeholder: run clean_application first")
    out = loan_ratio_features(df.copy())
    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / DAYS_PER_YEAR
    out["EMPLOYED_YEARS"] = -out["DAYS_EMPLOYED"] / DAYS_PER_YEAR
    out["EMPLOYED_TO_AGE_RATIO"] = safe_divide(out["EMPLOYED_YEARS"], out["AGE_YEARS"])
    out["INCOME_PER_FAMILY_MEMBER"] = safe_divide(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])
    ext = out[EXT_SOURCES]
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    out["EXT_SOURCE_MIN"] = ext.min(axis=1)
    out["EXT_SOURCE_N_MISSING"] = ext.isna().sum(axis=1).astype("int8")
    doc_cols = [c for c in out.columns if c.startswith("FLAG_DOCUMENT_")]
    out["DOCUMENT_COUNT"] = out[doc_cols].sum(axis=1).astype("int16") if doc_cols else 0
    return out.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])


def redundant_building_columns(columns: list[str]) -> list[str]:
    """`_MEDI` / `_MODE` building columns that duplicate an existing `_AVG` column."""
    cols = set(columns)
    return sorted(
        c for c in columns
        if c.endswith(("_MEDI", "_MODE")) and f"{c[:-5]}_AVG" in cols
    )


# ---------------------------------------------------------------------------
# Credit bureau
# ---------------------------------------------------------------------------

BUREAU_STATUS_TO_DPD = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}  # C, X -> missing
BUREAU_AMOUNTS = [
    "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
    "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM_LIMIT", "AMT_ANNUITY",
]


def bureau_balance_per_credit(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_BUREAU: worst recent DPD bucket and whether any DPD was ever recorded."""
    require_columns(bureau_balance, ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"], "bureau_balance")
    dpd = bureau_balance["STATUS"].map(BUREAU_STATUS_TO_DPD).astype("float64")
    recent = bureau_balance["MONTHS_BALANCE"] >= -RECENT_MONTHS
    tmp = pd.DataFrame(
        {"SK_ID_BUREAU": bureau_balance["SK_ID_BUREAU"], "dpd": dpd, "dpd_recent": dpd.where(recent)}
    )
    g = tmp.groupby("SK_ID_BUREAU")
    return pd.DataFrame(
        {
            "BB_WORST_STATUS_12M": g["dpd_recent"].max(),
            "BB_EVER_DPD": (g["dpd"].max() > 0).astype("float64"),
        }
    )


def aggregate_bureau(
    bureau: pd.DataFrame, bureau_balance: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per SK_ID_CURR from bureau (+ bureau_balance)."""
    require_columns(
        bureau, [ID, "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_CURRENCY", "DAYS_CREDIT"], "bureau"
    )
    b = bureau.copy()
    if "DAYS_CREDIT_UPDATE" in b:  # defined as days BEFORE application; 17 positive rows are errors
        b["DAYS_CREDIT_UPDATE"] = b["DAYS_CREDIT_UPDATE"].where(b["DAYS_CREDIT_UPDATE"] <= 0)
    other_currency = b["CREDIT_CURRENCY"] != "currency 1"  # unknown exchange rates
    for col in BUREAU_AMOUNTS:
        if col in b:
            b[col] = b[col].where(~other_currency)
    active = b["CREDIT_ACTIVE"] == "Active"
    b["_active"] = active.astype("int8")
    b["_closed"] = (b["CREDIT_ACTIVE"] == "Closed").astype("int8")
    b["_bad"] = b["CREDIT_ACTIVE"].isin(["Bad debt", "Sold"]).astype("int8")
    b["_active_debt"] = b["AMT_CREDIT_SUM_DEBT"].where(active)
    b["_active_credit"] = b["AMT_CREDIT_SUM"].where(active)
    if bureau_balance is not None:
        bb = bureau_balance_per_credit(bureau_balance)
        b = b.merge(bb, left_on="SK_ID_BUREAU", right_index=True, how="left")

    g = b.groupby(ID)
    out = pd.DataFrame(
        {
            "BUREAU_COUNT": g.size(),
            "BUREAU_ACTIVE_COUNT": g["_active"].sum(),
            "BUREAU_CLOSED_COUNT": g["_closed"].sum(),
            "BUREAU_BAD_DEBT_OR_SOLD_COUNT": g["_bad"].sum(),
            "BUREAU_DAYS_CREDIT_MAX": g["DAYS_CREDIT"].max(),
            "BUREAU_DAYS_CREDIT_MIN": g["DAYS_CREDIT"].min(),
            "BUREAU_ACTIVE_DEBT_SUM": g["_active_debt"].sum(min_count=1),
            "BUREAU_ACTIVE_CREDIT_SUM": g["_active_credit"].sum(min_count=1),
            "BUREAU_OVERDUE_SUM": g["AMT_CREDIT_SUM_OVERDUE"].sum(min_count=1),
            "BUREAU_MAX_OVERDUE_MAX": g["AMT_CREDIT_MAX_OVERDUE"].max(),
            "BUREAU_PROLONG_SUM": g["CNT_CREDIT_PROLONG"].sum(min_count=1),
        }
    )
    out["BUREAU_ACTIVE_DEBT_RATIO"] = safe_divide(
        out["BUREAU_ACTIVE_DEBT_SUM"], out["BUREAU_ACTIVE_CREDIT_SUM"]
    )
    if bureau_balance is not None:
        out["BUREAU_BB_WORST_STATUS_12M"] = g["BB_WORST_STATUS_12M"].max()
        out["BUREAU_BB_DPD_CREDIT_COUNT"] = g["BB_EVER_DPD"].sum()
    return out


# ---------------------------------------------------------------------------
# Previous Home Credit applications
# ---------------------------------------------------------------------------

def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        prev,
        [ID, "NAME_CONTRACT_STATUS", "AMT_CREDIT", "AMT_APPLICATION", "AMT_ANNUITY",
         "DAYS_DECISION", "CNT_PAYMENT"],
        "previous_application",
    )
    p = pd.DataFrame({ID: prev[ID]})
    approved = prev["NAME_CONTRACT_STATUS"] == "Approved"
    refused = prev["NAME_CONTRACT_STATUS"] == "Refused"
    p["approved"] = approved.astype("int8")
    p["refused"] = refused.astype("int8")
    p["refused_recent"] = (refused & (prev["DAYS_DECISION"] >= -RECENT_DAYS)).astype("int8")
    p["credit_to_app"] = safe_divide(prev["AMT_CREDIT"], prev["AMT_APPLICATION"])
    for col in ["AMT_CREDIT", "AMT_ANNUITY", "DAYS_DECISION", "CNT_PAYMENT"]:
        p[col] = prev[col]
    g = p.groupby(ID)
    out = pd.DataFrame(
        {
            "PREV_COUNT": g.size(),
            "PREV_APPROVED_COUNT": g["approved"].sum(),
            "PREV_REFUSED_COUNT": g["refused"].sum(),
            "PREV_REFUSED_LAST_YEAR_COUNT": g["refused_recent"].sum(),
            "PREV_AMT_CREDIT_MEAN": g["AMT_CREDIT"].mean(),
            "PREV_AMT_ANNUITY_MEAN": g["AMT_ANNUITY"].mean(),
            "PREV_CREDIT_TO_APPLICATION_MEAN": g["credit_to_app"].mean(),
            "PREV_DAYS_DECISION_MAX": g["DAYS_DECISION"].max(),
            "PREV_CNT_PAYMENT_MEAN": g["CNT_PAYMENT"].mean(),
        }
    )
    out["PREV_APPROVAL_RATE"] = safe_divide(out["PREV_APPROVED_COUNT"], out["PREV_COUNT"])
    out["PREV_REFUSAL_RATE"] = safe_divide(out["PREV_REFUSED_COUNT"], out["PREV_COUNT"])
    return out


# ---------------------------------------------------------------------------
# Instalment payments of previous credits
# ---------------------------------------------------------------------------

def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """Partial payments of the same instalment are combined first, so lateness is
    measured per scheduled instalment (date of the last payment vs due date)."""
    keys = [ID, "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]
    require_columns(
        inst, keys + ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
        "installments_payments",
    )
    per = (
        inst.groupby(keys, sort=False)
        .agg(
            due_day=("DAYS_INSTALMENT", "min"),
            amount_due=("AMT_INSTALMENT", "max"),
            amount_paid=("AMT_PAYMENT", "sum"),
            paid_day=("DAYS_ENTRY_PAYMENT", "max"),
        )
        .reset_index()
    )
    unpaid = per["paid_day"].isna()
    days_late = (per["paid_day"] - per["due_day"]).clip(lower=0)  # missing if unpaid
    shortfall = (per["amount_due"] - per["amount_paid"]).clip(lower=0)
    recent = per["due_day"] >= -RECENT_DAYS
    late = (days_late > 0).astype("float64").where(~unpaid)
    tmp = pd.DataFrame(
        {
            ID: per[ID],
            "unpaid": unpaid.astype("int8"),
            "late": late,
            "late30": (days_late > 30).astype("float64").where(~unpaid),
            "days_late": days_late,
            "underpaid": (shortfall > 0.01).astype("float64"),
            "shortfall": shortfall,
            "late_recent": late.where(recent),
        }
    )
    g = tmp.groupby(ID)
    return pd.DataFrame(
        {
            "INST_COUNT": g.size(),
            "INST_UNPAID_COUNT": g["unpaid"].sum(),
            "INST_LATE_SHARE": g["late"].mean(),
            "INST_LATE30_SHARE": g["late30"].mean(),
            "INST_DAYS_LATE_MEAN": g["days_late"].mean(),
            "INST_DAYS_LATE_MAX": g["days_late"].max(),
            "INST_UNDERPAID_SHARE": g["underpaid"].mean(),
            "INST_SHORTFALL_SUM": g["shortfall"].sum(),
            "INST_LATE_SHARE_LAST_YEAR": g["late_recent"].mean(),
        }
    )


# ---------------------------------------------------------------------------
# POS / cash loan monthly balances
# ---------------------------------------------------------------------------

def aggregate_pos(pos: pd.DataFrame) -> pd.DataFrame:
    require_columns(pos, [ID, "SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF"], "POS_CASH")
    recent = pos["MONTHS_BALANCE"] >= -RECENT_MONTHS
    tmp = pd.DataFrame(
        {
            ID: pos[ID],
            "SK_ID_PREV": pos["SK_ID_PREV"],
            "dpd": pos["SK_DPD"],
            "dpd_def": pos["SK_DPD_DEF"],
            "dpd_pos": (pos["SK_DPD"] > 0).astype("float64"),
            "dpd_recent": pos["SK_DPD"].where(recent),
        }
    )
    g = tmp.groupby(ID)
    return pd.DataFrame(
        {
            "POS_MONTHS_COUNT": g.size(),
            "POS_LOAN_COUNT": g["SK_ID_PREV"].nunique(),
            "POS_DPD_MAX": g["dpd"].max(),
            "POS_DPD_DEF_MAX": g["dpd_def"].max(),
            "POS_DPD_MONTH_SHARE": g["dpd_pos"].mean(),
            "POS_DPD_MAX_LAST_YEAR": g["dpd_recent"].max(),
        }
    )


# ---------------------------------------------------------------------------
# Credit card monthly balances
# ---------------------------------------------------------------------------

def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        cc,
        [ID, "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD",
         "AMT_DRAWINGS_CURRENT"],
        "credit_card_balance",
    )
    util = safe_divide(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    recent = cc["MONTHS_BALANCE"] >= -RECENT_MONTHS
    tmp = pd.DataFrame(
        {
            ID: cc[ID],
            "util": util,
            "util_recent": util.where(recent),
            "dpd": cc["SK_DPD"],
            "dpd_pos": (cc["SK_DPD"] > 0).astype("float64"),
            "drawings": cc["AMT_DRAWINGS_CURRENT"],
        }
    )
    g = tmp.groupby(ID)
    return pd.DataFrame(
        {
            "CC_MONTHS_COUNT": g.size(),
            "CC_UTILISATION_MEAN": g["util"].mean(),
            "CC_UTILISATION_MAX": g["util"].max(),
            "CC_UTILISATION_MEAN_LAST_YEAR": g["util_recent"].mean(),
            "CC_DPD_MAX": g["dpd"].max(),
            "CC_DPD_MONTH_SHARE": g["dpd_pos"].mean(),
            "CC_DRAWINGS_MEAN": g["drawings"].mean(),
        }
    )


def join_aggregate(
    base: pd.DataFrame, agg: pd.DataFrame, flag: str
) -> pd.DataFrame:
    """Left-join one aggregate onto the applicant table and add a HAS_<source> flag.

    Count columns (*_COUNT) become 0 for applicants without records (a true count);
    every other aggregate stays missing, because "no history" is not "zero".
    """
    if not agg.index.is_unique:
        raise ValueError(f"{flag}: aggregate has duplicate {ID} values")
    n_before = len(base)
    out = base.merge(agg, left_on=ID, right_index=True, how="left")
    if len(out) != n_before:
        raise AssertionError(f"{flag}: join changed row count {n_before} -> {len(out)}")
    out[flag] = out[ID].isin(agg.index).astype("int8")
    for col in agg.columns:
        if col.endswith("_COUNT"):
            out[col] = out[col].fillna(0)
    return out
