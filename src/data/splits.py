"""Reproducible stratified train / validation / calibration / test split.

Why four parts (all from application_train, the only labelled data):
- train:        fit models
- validation:   compare models, tune hyper-parameters, choose thresholds
- calibration:  fit the probability calibrator only (Phase 5)
- test:         used once, for the final reported numbers
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SPLIT_NAMES, validate_splits


def make_splits(
    ids: pd.Series, target: pd.Series, splits: dict[str, Any], seed: int
) -> pd.DataFrame:
    """Assign each ID to one split, stratified on the target.

    Returns a DataFrame with columns [<id name>, "split"], in the original ID order.
    """
    validate_splits(splits)
    if len(ids) != len(target):
        raise ValueError("ids and target must have the same length")
    if ids.duplicated().any():
        raise ValueError("ids must be unique")
    ids = ids.reset_index(drop=True)
    y = target.reset_index(drop=True)
    idx = pd.Series(range(len(ids)))

    # Peel off test, then calibration, then validation; the rest is train.
    remaining, assigned = idx, {}
    remaining_share = 1.0
    for name in ("test", "calibration", "validation"):
        share = splits[name] / remaining_share
        remaining, taken = train_test_split(
            remaining, test_size=share, stratify=y[remaining], random_state=seed
        )
        assigned[name] = taken
        remaining_share -= splits[name]
    assigned["train"] = remaining

    label = pd.Series(index=idx, dtype="object")
    for name in SPLIT_NAMES:
        label[assigned[name]] = name
    return pd.DataFrame({ids.name or "id": ids, "split": label.to_numpy()})
