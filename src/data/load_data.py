"""Read raw Home Credit tables. Raw files are only ever read, never written.

Memory strategy for a 16 GB laptop:
- integers are downcast (lossless, e.g. int64 -> int32);
- text columns with few distinct values become `category` (lossless);
- floats stay float64 by default so audit statistics are exact. Pass
  `downcast_floats=True` for modelling, where float32 precision is enough.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config, project_path

TABLE_FILES: dict[str, str] = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "sample_submission": "sample_submission.csv",
}
COLUMN_DESCRIPTION_FILE = "HomeCredit_columns_description.csv"


def raw_path(table: str, cfg: dict[str, Any] | None = None) -> Path:
    """Absolute path of a raw table, by its short name (see TABLE_FILES)."""
    if table not in TABLE_FILES:
        raise KeyError(f"Unknown table {table!r}. Known tables: {sorted(TABLE_FILES)}")
    cfg = cfg or load_config()
    return project_path(cfg, "raw_data") / TABLE_FILES[table]


def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def optimise_dtypes(
    df: pd.DataFrame, downcast_floats: bool = False, max_category_ratio: float = 0.5
) -> pd.DataFrame:
    """Shrink memory use in place without changing any values (unless floats are downcast)."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            if downcast_floats:
                df[col] = pd.to_numeric(s, downcast="float")
        elif _is_text(s) and s.nunique(dropna=True) <= max_category_ratio * max(len(s), 1):
            df[col] = s.astype("category")
    return df


def read_table(
    table: str,
    cfg: dict[str, Any] | None = None,
    usecols: list[str] | None = None,
    nrows: int | None = None,
    downcast_floats: bool = False,
) -> pd.DataFrame:
    """Read one raw table with memory-efficient dtypes."""
    path = raw_path(table, cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the Kaggle Home Credit CSVs in data/raw (see README)."
        )
    df = pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)
    return optimise_dtypes(df, downcast_floats=downcast_floats)


def read_column_descriptions(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """The official column description file (it is not UTF-8 encoded)."""
    cfg = cfg or load_config()
    path = project_path(cfg, "raw_data") / COLUMN_DESCRIPTION_FILE
    df = pd.read_csv(path, encoding="latin-1")
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Checksum used to record exactly which version of each file was analysed."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
