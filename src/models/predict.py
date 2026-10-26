"""The single scoring path. Dashboard, stress tests, counterfactuals and the API
all call `predict_proba` here, so every consumer uses the same preprocessing,
feature order and model.

Model files are joblib pickles. Only load artifacts this project created:
unpickling a file from an untrusted source can run arbitrary code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.evaluate import check_probabilities


def save_model(pipeline: Any, metadata: dict[str, Any], directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.joblib"
    joblib.dump(pipeline, path, compress=3)
    with (directory / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return path


def load_model(path: Path) -> tuple[Any, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return joblib.load(path), metadata


def predict_proba(pipeline: Any, df: pd.DataFrame) -> np.ndarray:
    """Probability of TARGET = 1 for each row. Input must contain the training features
    (extra columns are ignored; missing ones raise a clear error)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("predict_proba expects a pandas DataFrame")
    if len(df) == 0:
        raise ValueError("no rows to score")
    p = pipeline.predict_proba(df)[:, 1]
    return check_probabilities(p)
