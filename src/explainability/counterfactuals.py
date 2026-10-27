"""Constrained counterfactual search: "what change to THIS loan's terms would bring
the model's estimate below the target?"

Design (deliberately simple and transparent, instead of a generic optimiser):
- Only two levers, both about the loan being applied for:
    credit_factor: borrow less    -> AMT_CREDIT and AMT_GOODS_PRICE x factor (factor <= 1)
    term_factor:   repay over a different period -> AMT_ANNUITY x credit_factor / term_factor
  The annuity rule ignores interest (the dataset gives no rate), so longer terms
  look somewhat cheaper than they would be.
- Everything else is locked: age, income, employment, external scores, all credit
  history. Locked columns are checked to be unchanged.
- Derived ratios are recomputed with the same function used to build the dataset.
- New amounts must stay inside the range seen in training (plausibility bounds).
- Every candidate is scored through the deployed model, never approximated.
- Among candidates that reach the target, the one with the smallest total relative
  change |1 - credit_factor| + |1 - term_factor| is reported.

Every result is a MODEL-BASED HYPOTHETICAL: it says what the model would estimate,
not what would happen to the borrower, and not that a lender would offer those terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_engineering import LOAN_RATIO_FEATURES, loan_ratio_features
from src.models.predict import predict_proba

LEVER_AMOUNTS = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
CHANGEABLE = set(LEVER_AMOUNTS) | set(LOAN_RATIO_FEATURES)
DISCLAIMER = (
    "Model-based hypothetical scenario: shows how the model's estimate would change if the "
    "loan terms were different. It is not a causal prediction, a guarantee, or a credit offer."
)


@dataclass
class CounterfactualResult:
    status: str  # already_below_target | found | not_found
    original_probability: float
    target: float
    best_achievable_probability: float
    best: dict[str, Any] | None = None
    changes: pd.DataFrame = field(default_factory=pd.DataFrame)
    feasible_candidates: int = 0
    note: str = DISCLAIMER


def amount_bounds(train: pd.DataFrame, quantiles: tuple[float, float]) -> dict[str, tuple]:
    """Plausible range of each loan amount, from the TRAINING split."""
    return {c: tuple(float(v) for v in train[c].quantile(list(quantiles))) for c in LEVER_AMOUNTS}


def lever_grid(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    def grid(g: dict[str, Any]) -> np.ndarray:
        if not 0 < g["min"] <= g["max"] or g["steps"] < 1:
            raise ValueError(f"invalid lever grid {g}")
        return np.round(np.linspace(g["min"], g["max"], g["steps"]), 6)

    cf, tf = grid(cfg["credit_factor"]), grid(cfg["term_factor"])
    if cf.max() > 1.0:
        raise ValueError("credit_factor above 1 would mean borrowing more")
    return cf, tf


def make_candidates(
    row: pd.DataFrame, credit_factors: np.ndarray, term_factors: np.ndarray
) -> pd.DataFrame:
    """All lever combinations for one borrower, with ratios recomputed."""
    if len(row) != 1:
        raise ValueError("make_candidates expects exactly one row")
    has_annuity = pd.notna(row["AMT_ANNUITY"].iloc[0])
    tfs = term_factors if has_annuity else np.array([1.0])  # no annuity -> term lever unusable
    cf, tf = np.meshgrid(credit_factors, tfs, indexing="ij")
    cf, tf = cf.ravel(), tf.ravel()
    cand = pd.concat([row] * len(cf), ignore_index=True)
    base = row.iloc[0]
    cand["AMT_CREDIT"] = base["AMT_CREDIT"] * cf
    cand["AMT_GOODS_PRICE"] = base["AMT_GOODS_PRICE"] * cf
    cand["AMT_ANNUITY"] = base["AMT_ANNUITY"] * cf / tf
    loan_ratio_features(cand)
    cand["_credit_factor"] = cf
    cand["_term_factor"] = tf
    return cand


def check_locked(row: pd.DataFrame, cand: pd.DataFrame) -> None:
    """Fail if any column outside the loan terms and their ratios was changed."""
    locked = [c for c in row.columns if c not in CHANGEABLE]
    a = cand[locked].reset_index(drop=True)
    b = pd.concat([row[locked]] * len(cand), ignore_index=True)
    same = (a == b) | (a.isna() & b.isna())
    if not same.all().all():
        bad = same.columns[~same.all()].tolist()
        raise AssertionError(f"locked features changed: {bad}")


def search(
    model: Any, row: pd.DataFrame, target: float, credit_factors: np.ndarray,
    term_factors: np.ndarray, bounds: dict[str, tuple] | None = None,
) -> CounterfactualResult:
    if not 0 < target < 1:
        raise ValueError("target must be a probability in (0, 1)")
    p0 = float(predict_proba(model, row)[0])
    if p0 < target:
        return CounterfactualResult("already_below_target", p0, target, p0)

    cand = make_candidates(row, credit_factors, term_factors)
    check_locked(row, cand)
    plausible = pd.Series(True, index=cand.index)
    for col, (lo, hi) in (bounds or {}).items():
        v = cand[col]
        plausible &= v.isna() | v.between(lo, hi)
    cand = cand[plausible]
    if cand.empty:
        return CounterfactualResult("not_found", p0, target, p0)

    cand = cand.assign(probability=predict_proba(model, cand.drop(columns=["_credit_factor",
                                                                            "_term_factor"])))
    cand["change_cost"] = (1 - cand["_credit_factor"]).abs() + (cand["_term_factor"] - 1).abs()
    best_p = float(cand["probability"].min())
    hits = cand[cand["probability"] < target]
    if hits.empty:
        return CounterfactualResult("not_found", p0, target, best_p,
                                    feasible_candidates=len(cand))

    best = hits.sort_values(["change_cost", "probability"]).iloc[0]
    changed = [c for c in [*LEVER_AMOUNTS, *LOAN_RATIO_FEATURES] if c in row.columns]
    changes = pd.DataFrame(
        {"feature": changed, "original": [row.iloc[0][c] for c in changed],
         "counterfactual": [best[c] for c in changed]}
    )
    return CounterfactualResult(
        status="found", original_probability=p0, target=target,
        best_achievable_probability=best_p,
        best={"credit_factor": float(best["_credit_factor"]),
              "term_factor": float(best["_term_factor"]),
              "probability": float(best["probability"]),
              "change_cost": float(best["change_cost"])},
        changes=changes, feasible_candidates=int(len(hits)),
    )
