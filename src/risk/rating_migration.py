"""Delinquency-state migration of credit bureau loans (bureau_balance).

States (monthly STATUS): 0 = no days past due, 1 = 1-30 DPD, 2 = 31-60, 3 = 61-90,
4 = 91-120, 5 = 120+ DPD or sold/written off, C = closed, X = status unknown.

These are OBSERVED monthly states of applicants' loans at other lenders before their
Home Credit application. They are not ratings assigned by RiskLens, and not the
current Home Credit portfolio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validate_data import require_columns


def transition_counts(bb: pd.DataFrame, states: list[str], step: int = 1) -> pd.DataFrame:
    """Counts of state(m) -> state(m + step) for the same credit, over consecutive records."""
    require_columns(bb, ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"], "bureau_balance")
    if step < 1:
        raise ValueError("step must be >= 1")
    # integer state codes keep memory small on tens of millions of rows
    cat = bb["STATUS"].astype("category")
    lookup = np.array([states.index(str(c)) if str(c) in states else -1
                       for c in cat.cat.categories] + [-1])  # last entry: missing (code -1)
    d = pd.DataFrame({"id": bb["SK_ID_BUREAU"].to_numpy(dtype="int64"),
                      "m": bb["MONTHS_BALANCE"].to_numpy(dtype="int64"),
                      "s": lookup[cat.cat.codes.to_numpy()]})
    d = d[d["s"] >= 0]
    later = d.rename(columns={"s": "to"})
    later["m"] = later["m"] - step
    pairs = d.merge(later, on=["id", "m"], how="inner")
    k = len(states)
    flat = np.bincount(pairs["s"].to_numpy() * k + pairs["to"].to_numpy(), minlength=k * k)
    return pd.DataFrame(flat.reshape(k, k), index=pd.Index(states, name="from"),
                        columns=pd.Index(states, name="to"))


def transition_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalised probabilities; rows with no observations stay missing."""
    totals = counts.sum(axis=1)
    return counts.div(totals.where(totals > 0), axis=0)


def matrix_power(P: pd.DataFrame, k: int) -> pd.DataFrame:
    """k-step probabilities under a time-homogeneous Markov ASSUMPTION."""
    if P.isna().any().any():
        raise ValueError("matrix has rows without observations")
    return pd.DataFrame(np.linalg.matrix_power(P.to_numpy(), k), index=P.index,
                        columns=P.columns)


def wilson_interval(successes: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion (works near 0 and 1)."""
    successes, n = np.asarray(successes, float), np.asarray(n, float)
    p = np.divide(successes, n, out=np.full_like(n, np.nan), where=n > 0)
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half
