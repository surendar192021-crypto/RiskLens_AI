"""Data and prediction drift between a reference sample and a comparison sample.

Population Stability Index (PSI): bin the reference distribution (deciles for
numeric features, categories for categorical ones, plus an explicit "missing" bin),
then PSI = sum over bins of (new% - ref%) * ln(new% / ref%). 0 = identical.
It measures the SIZE of a shift, not its statistical significance; with tens of
thousands of rows even tiny shifts are "significant", so PSI is the main measure
and KS / chi-square p-values are reported only as secondary information.

Data drift (inputs change) is not the same as performance drift (the model gets
worse). Performance drift needs outcome labels for the new data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-4  # floor for empty bins so ln() stays finite


def _psi_from_shares(ref: np.ndarray, new: np.ndarray) -> float:
    ref = np.clip(ref, EPS, None)
    new = np.clip(new, EPS, None)
    return float(np.sum((new - ref) * np.log(new / ref)))


def psi_numeric(ref: pd.Series, new: pd.Series, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    r = pd.to_numeric(ref, errors="coerce").to_numpy(dtype="float64")
    n = pd.to_numeric(new, errors="coerce").to_numpy(dtype="float64")
    finite = r[~np.isnan(r)]
    edges = (np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1))[1:-1])
             if len(finite) else np.array([]))
    k = len(edges) + 1

    def shares(x: np.ndarray) -> np.ndarray:
        idx = np.where(np.isnan(x), k, np.searchsorted(edges, x, side="right"))
        return np.bincount(idx, minlength=k + 1) / max(len(x), 1)

    sr, sn = shares(r), shares(n)
    labels = [f"<= {edges[0]:.4g}"] if len(edges) else ["all"]
    labels += [f"({a:.4g}, {b:.4g}]" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    if len(edges):
        labels.append(f"> {edges[-1]:.4g}")
    labels.append("missing")
    table = pd.DataFrame({"bin": labels, "reference_share": sr, "new_share": sn})
    return _psi_from_shares(sr, sn), table


def psi_categorical(ref: pd.Series, new: pd.Series) -> tuple[float, pd.DataFrame]:
    r = ref.astype("object").where(ref.notna(), "<missing>")
    n = new.astype("object").where(new.notna(), "<missing>")
    cats = sorted(set(r.unique()) | set(n.unique()), key=str)
    sr = r.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    sn = n.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    table = pd.DataFrame({"bin": [str(c) for c in cats], "reference_share": sr, "new_share": sn})
    return _psi_from_shares(sr, sn), table


def classify_psi(value: float, thresholds: dict[str, float]) -> str:
    if value >= thresholds["major"]:
        return "major"
    if value >= thresholds["moderate"]:
        return "moderate"
    return "stable"


def feature_drift(
    ref: pd.DataFrame, new: pd.DataFrame, numeric: list[str], categorical: list[str],
    n_bins: int, thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for col in numeric + categorical:
        if col in numeric:
            value, _ = psi_numeric(ref[col], new[col], n_bins)
            a = ref[col].dropna().to_numpy(dtype="float64")
            b = new[col].dropna().to_numpy(dtype="float64")
            ks = ks_2samp(a, b) if len(a) and len(b) else None
            ks_stat, ks_p = (float(ks.statistic), float(ks.pvalue)) if ks else (None, None)
        else:
            value, _ = psi_categorical(ref[col], new[col])
            ks_stat = ks_p = None
        rows.append({
            "feature": col, "type": "numeric" if col in numeric else "categorical",
            "psi": value, "level": classify_psi(value, thresholds),
            "ks_statistic": ks_stat, "ks_pvalue": ks_p,
            "missing_reference": float(ref[col].isna().mean()),
            "missing_new": float(new[col].isna().mean()),
        })
    out = pd.DataFrame(rows)
    out["missing_change_pp"] = (out["missing_new"] - out["missing_reference"]) * 100
    return out.sort_values("psi", ascending=False, ignore_index=True)


def schema_check(ref: pd.DataFrame, new: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    missing = [c for c in features if c not in new.columns]
    dtype_changes = {
        c: (str(ref[c].dtype), str(new[c].dtype))
        for c in features if c in new.columns and c in ref.columns
        and pd.api.types.is_numeric_dtype(ref[c]) != pd.api.types.is_numeric_dtype(new[c])
    }
    return {"missing_features": missing, "type_changes": dtype_changes,
            "ok": not missing and not dtype_changes}


def review_recommendation(
    drift: pd.DataFrame, prediction_psi: float, important: list[str], rule: dict[str, Any],
) -> dict[str, Any]:
    """A REVIEW flag with reasons. Never triggers retraining by itself."""
    major_important = drift[(drift["level"] == "major") & drift["feature"].isin(important)]
    reasons = []
    if prediction_psi >= rule["prediction_psi"]:
        reasons.append(f"model-score PSI {prediction_psi:.3f} >= {rule['prediction_psi']}")
    if len(major_important) >= rule["major_drift_important_features"]:
        reasons.append(f"{len(major_important)} important features with major drift: "
                       f"{', '.join(major_important['feature'])}")
    return {"review_recommended": bool(reasons), "reasons": reasons,
            "major_drift_important_features": major_important["feature"].tolist(),
            "action": ("Review recommended: check data pipeline and population change; obtain "
                       "outcome labels before deciding on retraining (no automatic retrain)."
                       if reasons else "No review triggered by the configured rule.")}
