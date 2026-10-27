"""Feature-level scenario transformations shared by stress tests and simulation demos.

Each function changes a raw amount and recomputes every feature derived from it,
using the same formulas as the dataset build, so scenarios stay internally consistent.
These are hypothetical perturbations, not observed borrower behaviour.
"""

from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import loan_ratio_features, safe_divide


def scale_income(df: pd.DataFrame, factor: float | pd.Series) -> pd.DataFrame:
    """Multiply AMT_INCOME_TOTAL by `factor` (scalar or per-row) and recompute
    the income-based ratios. Returns a copy."""
    if (pd.Series(factor) <= 0).any():
        raise ValueError("income factor must be positive")
    out = df.copy()
    out["AMT_INCOME_TOTAL"] = out["AMT_INCOME_TOTAL"] * factor
    loan_ratio_features(out)
    if "INCOME_PER_FAMILY_MEMBER" in out:
        out["INCOME_PER_FAMILY_MEMBER"] = safe_divide(out["AMT_INCOME_TOTAL"],
                                                      out["CNT_FAM_MEMBERS"])
    return out
