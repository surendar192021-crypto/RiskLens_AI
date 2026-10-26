"""Turning a probability into a presentation risk band.

Bands come from config `risk_bands` and are PROVISIONAL presentation categories,
not validated banking grades. Intervals are [lower, upper), the last includes 1.0,
so every probability in [0, 1] gets exactly one band.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import validate_risk_bands
from src.models.evaluate import check_probabilities


def assign_risk_band(p: Any, edges: list[float], labels: list[str]) -> pd.Categorical:
    validate_risk_bands({"edges": edges, "labels": labels})
    p = check_probabilities(p)
    idx = np.searchsorted(np.asarray(edges[1:-1]), p, side="right")
    return pd.Categorical.from_codes(idx, categories=labels, ordered=True)


def band_distribution(
    p: Any, edges: list[float], labels: list[str], y: Any = None
) -> pd.DataFrame:
    """Borrower count and share per band, plus the observed TARGET rate if labels are given."""
    bands = assign_risk_band(p, edges, labels)
    df = pd.DataFrame({"band": bands, "p": check_probabilities(p)})
    if y is not None:
        df["y"] = np.asarray(y)
    g = df.groupby("band", observed=False)
    out = pd.DataFrame(
        {
            "lower": edges[:-1],
            "upper": edges[1:],
            "borrowers": g.size().to_numpy(),
            "share": (g.size() / len(df)).to_numpy(),
            "mean_predicted": g["p"].mean().to_numpy(),
        },
        index=pd.Index(labels, name="band"),
    )
    if y is not None:
        out["observed_rate"] = g["y"].mean().to_numpy()
    return out.reset_index()
