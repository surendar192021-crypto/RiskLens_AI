"""Row-level cleaning decided in the Phase 2 audit (docs/dataset_audit_findings.md).

These steps use no target information and no statistics from other rows, so they
can be applied to train and test alike without leakage. Anything that is *fitted*
(imputation, scaling, encoding) belongs in the model pipeline instead (Phase 4).
"""

from __future__ import annotations

import pandas as pd

from src.data.validate_data import DAYS_PLACEHOLDER, require_columns

# Codes that mean "unknown" in specific columns (audit section 5).
MISSING_CODES: dict[str, list[str]] = {
    "CODE_GENDER": ["XNA"],
    "NAME_FAMILY_STATUS": ["Unknown"],
}


def set_missing_codes(series: pd.Series, codes: list[str]) -> pd.Series:
    """Replace the given codes with missing; drop them from the categories if categorical."""
    out = series.where(~series.isin(codes))
    if isinstance(out.dtype, pd.CategoricalDtype):
        out = out.cat.remove_unused_categories()
    return out


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Clean application_train / application_test rows.

    - DAYS_EMPLOYED == 365243 (placeholder, 99.96% pensioners) -> missing,
      plus the flag DAYS_EMPLOYED_PLACEHOLDER, because the placeholder itself is informative.
    - CODE_GENDER 'XNA' and NAME_FAMILY_STATUS 'Unknown' -> missing.
    """
    require_columns(df, ["DAYS_EMPLOYED"], "application table")
    out = df.copy()
    placeholder = out["DAYS_EMPLOYED"] == DAYS_PLACEHOLDER
    out["DAYS_EMPLOYED_PLACEHOLDER"] = placeholder.astype("int8")
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].where(~placeholder).astype("float64")
    for col, codes in MISSING_CODES.items():
        if col in out.columns:
            out[col] = set_missing_codes(out[col], codes)
    return out
