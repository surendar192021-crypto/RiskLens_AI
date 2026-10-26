"""Pure functions that check and summarise tables. Used by the audit and by later pipeline steps.

None of these functions modify their input.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

TEMPORAL_PREFIXES = ("DAYS_", "MONTHS_")
# Home Credit uses 365243 days (~1000 years) as a placeholder in some DAYS_ columns.
DAYS_PLACEHOLDER = 365243


class SchemaError(ValueError):
    """Raised when a table does not have the structure the pipeline expects."""


def require_columns(df: pd.DataFrame, columns: list[str], table: str = "table") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{table} is missing required columns: {missing}")


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing count and missing percentage (highest first)."""
    n = len(df)
    missing = df.isna().sum()
    out = pd.DataFrame(
        {
            "column": missing.index,
            "dtype": df.dtypes.astype(str).to_numpy(),
            "n_missing": missing.to_numpy(),
            "pct_missing": missing.to_numpy() / n * 100 if n else 0.0,
        }
    )
    return out.sort_values(["pct_missing", "column"], ascending=[False, True]).reset_index(
        drop=True
    )


def key_uniqueness(df: pd.DataFrame, key: str) -> dict[str, Any]:
    """Is `key` a valid primary key (no nulls, no duplicates)?"""
    require_columns(df, [key])
    s = df[key]
    n_null = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    n_duplicates = int(len(s) - n_null - n_unique)
    return {
        "key": key,
        "n_rows": len(s),
        "n_null": n_null,
        "n_unique": n_unique,
        "n_duplicates": n_duplicates,
        "is_primary_key": n_null == 0 and n_duplicates == 0,
    }


def key_relationship(
    child_keys: pd.Series, parent_keys: pd.Series, child: str, parent: str
) -> dict[str, Any]:
    """How a child table's foreign key relates to a parent table's key.

    Reports orphan keys (child keys with no parent), how many parent keys have
    at least one child row, and how many child rows each key has (one-to-many size).
    """
    child_unique = pd.Index(child_keys.dropna().unique())
    parent_unique = pd.Index(parent_keys.dropna().unique())
    in_parent = child_unique.isin(parent_unique)
    rows_per_key = child_keys.value_counts()
    return {
        "child": child,
        "parent": parent,
        "key": child_keys.name,
        "child_rows": len(child_keys),
        "child_null_keys": int(child_keys.isna().sum()),
        "child_unique_keys": len(child_unique),
        "orphan_child_keys": int((~in_parent).sum()),
        "pct_child_keys_in_parent": float(in_parent.mean() * 100) if len(child_unique) else 0.0,
        "parent_unique_keys": len(parent_unique),
        "pct_parent_with_child": float(parent_unique.isin(child_unique).mean() * 100)
        if len(parent_unique)
        else 0.0,
        "rows_per_key_mean": float(rows_per_key.mean()) if len(rows_per_key) else 0.0,
        "rows_per_key_median": float(rows_per_key.median()) if len(rows_per_key) else 0.0,
        "rows_per_key_max": int(rows_per_key.max()) if len(rows_per_key) else 0,
    }


def temporal_summary(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Range of every DAYS_/MONTHS_ column, relative to the application date (day 0).

    Positive values are dates AFTER the application: allowed for planned dates,
    a leakage warning for observed events. The 365243 placeholder is counted
    separately and excluded from min/max.
    """
    rows = []
    for col in df.columns:
        if not col.startswith(TEMPORAL_PREFIXES):
            continue
        s = df[col]
        placeholder = s == DAYS_PLACEHOLDER
        valid = s[~placeholder].dropna()
        rows.append(
            {
                "table": table,
                "column": col,
                "n_missing": int(s.isna().sum()),
                "n_placeholder_365243": int(placeholder.sum()),
                "min": valid.min() if len(valid) else None,
                "max": valid.max() if len(valid) else None,
                "n_positive": int((valid > 0).sum()),
            }
        )
    columns = ["table", "column", "n_missing", "n_placeholder_365243", "min", "max", "n_positive"]
    return pd.DataFrame(rows, columns=columns)


def target_distribution(target: pd.Series) -> dict[str, Any]:
    """Class counts for a binary 0/1 target. Fails loudly on anything else."""
    if target.isna().any():
        raise SchemaError(f"target has {int(target.isna().sum())} missing values")
    values = set(pd.unique(target))
    if not values <= {0, 1}:
        raise SchemaError(f"target must be binary 0/1, found values {sorted(values)}")
    n_pos = int((target == 1).sum())
    n_neg = int((target == 0).sum())
    return {
        "n": len(target),
        "n_negative": n_neg,
        "n_positive": n_pos,
        "positive_rate": n_pos / len(target) if len(target) else 0.0,
        "negatives_per_positive": n_neg / n_pos if n_pos else float("inf"),
    }


def categorical_summary(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Cardinality and most common values of every non-numeric column."""
    rows = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        shares = s.value_counts(normalize=True, dropna=True).head(top_n)
        top = "; ".join(f"{v} ({p:.1%})" for v, p in shares.items())
        rows.append(
            {
                "column": col,
                "n_unique": int(s.nunique(dropna=True)),
                "n_missing": int(s.isna().sum()),
                "top_values": top,
            }
        )
    return pd.DataFrame(rows, columns=["column", "n_unique", "n_missing", "top_values"])


def numeric_profile(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Min, 1st/50th/99th percentile and max: a first look at outliers."""
    require_columns(df, columns)
    rows = []
    for col in columns:
        s = df[col]
        q = s.quantile([0.0, 0.01, 0.5, 0.99, 1.0])
        rows.append(
            {
                "column": col,
                "n_missing": int(s.isna().sum()),
                "min": q.loc[0.0],
                "p01": q.loc[0.01],
                "median": q.loc[0.5],
                "p99": q.loc[0.99],
                "max": q.loc[1.0],
            }
        )
    return pd.DataFrame(rows)
