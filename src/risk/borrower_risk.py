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


def score_portfolio(
    model: Any, df: pd.DataFrame, current: dict[str, Any], bands: dict[str, Any],
    scored_at: str, id_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """Borrower-level scores through the single scoring path."""
    from src.models.predict import predict_proba

    p = predict_proba(model, df)
    return pd.DataFrame({
        id_col: df[id_col].to_numpy(),
        "probability": p,
        "risk_band": assign_risk_band(p, bands["edges"], bands["labels"]),
        "above_threshold": p >= current["threshold"],
        "model_version": current["model_version"],
        "calibration": current["calibration_method"],
        "scored_at": scored_at,
    })


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
