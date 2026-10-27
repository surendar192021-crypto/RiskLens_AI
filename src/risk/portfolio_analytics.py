"""Portfolio-level aggregation of borrower-level PDs.

Quantities are kept distinct on purpose:
- borrower count / share:     how many borrowers
- mean PD:                    average estimated probability per borrower
- expected difficulties:      sum of PDs = expected NUMBER of payment-difficulty cases.
                              It is an expectation, not a forecast of the exact count.
- exposure:                   sum of the EAD proxy (AMT_CREDIT)
- exposure-weighted PD:       sum(PD x EAD) / sum(EAD): the PD of an average unit of money lent

The 95% range for the expected count assumes borrowers are INDEPENDENT. Real defaults are
correlated (common economic shocks), so the true uncertainty is wider; see Phase 9.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.evaluate import check_probabilities


def _arrays(p: Any, exposure: Any) -> tuple[np.ndarray, np.ndarray]:
    p = check_probabilities(p)
    e = np.asarray(exposure, dtype="float64")
    if e.shape != p.shape:
        raise ValueError("probabilities and exposure must have the same length")
    if not np.isfinite(e).all() or (e < 0).any():
        raise ValueError("exposure must be finite and non-negative")
    return p, e


def expected_count_interval(p: Any, z: float = 1.96) -> dict[str, float]:
    """Expected number of difficulties and a normal-approximation range ASSUMING independence."""
    p = check_probabilities(p)
    mean = float(p.sum())
    sd = float(np.sqrt((p * (1 - p)).sum()))
    return {"expected": mean, "sd_if_independent": sd,
            "low_if_independent": max(mean - z * sd, 0.0), "high_if_independent": mean + z * sd}


def portfolio_summary(p: Any, exposure: Any, threshold: float, bands: Any = None,
                      high_bands: tuple[str, ...] = ("High", "Critical")) -> dict[str, Any]:
    p, e = _arrays(p, exposure)
    above = p >= threshold
    out = {
        "borrowers": int(len(p)),
        "mean_pd": float(p.mean()),
        "median_pd": float(np.median(p)),
        **{f"{k}_difficulties": v for k, v in expected_count_interval(p).items()},
        "above_threshold_borrowers": int(above.sum()),
        "above_threshold_share": float(above.mean()),
        "total_exposure": float(e.sum()),
        "exposure_weighted_pd": float((p * e).sum() / e.sum()) if e.sum() > 0 else float("nan"),
        "exposure_above_threshold": float(e[above].sum()),
        "exposure_above_threshold_share": float(e[above].sum() / e.sum()) if e.sum() else 0.0,
    }
    if bands is not None:
        b = np.asarray(pd.Series(bands).astype(str))
        hb = np.isin(b, list(high_bands))
        out["high_or_critical_borrowers"] = int(hb.sum())
        out["high_or_critical_share"] = float(hb.mean())
        out["critical_borrowers"] = int((b == "Critical").sum())
    return out


def group_levels(s: pd.Series, max_levels: int) -> pd.Series:
    """Keep the most common levels; group the rest (and missing) explicitly."""
    s = s.astype("object")
    top = s.value_counts().index[:max_levels]
    out = s.where(s.isin(top), "Other")
    return out.where(s.notna(), "Missing")


def add_age_band(df: pd.DataFrame, edges: list[int]) -> pd.DataFrame:
    out = df.copy()
    out["AGE_BAND"] = pd.cut(out["AGE_YEARS"], edges, right=False).astype(str)
    return out


def segment_table(df: pd.DataFrame, segment: str, p: Any, exposure: Any,
                  y: Any = None, max_levels: int = 12) -> pd.DataFrame:
    """One row per segment level, with every aggregate kept separate."""
    p, e = _arrays(p, exposure)
    seg = group_levels(df[segment], max_levels).to_numpy()
    t = pd.DataFrame({"segment": seg, "p": p, "e": e, "pe": p * e})
    if y is not None:
        t["y"] = np.asarray(y, dtype="float64")
    g = t.groupby("segment")
    out = pd.DataFrame({
        "borrowers": g.size(),
        "mean_pd": g["p"].mean(),
        "expected_difficulties": g["p"].sum(),
        "exposure": g["e"].sum(),
    })
    out["borrower_share"] = out["borrowers"] / len(t)
    out["share_of_expected_difficulties"] = out["expected_difficulties"] / p.sum()
    out["exposure_share"] = out["exposure"] / e.sum()
    out["exposure_weighted_pd"] = g["pe"].sum() / out["exposure"].where(out["exposure"] > 0)
    # > 1: the segment holds more of the expected difficulties than of the borrowers
    out["risk_concentration_ratio"] = out["share_of_expected_difficulties"] / out["borrower_share"]
    if y is not None:
        out["observed_difficulties"] = g["y"].sum()
        out["observed_rate"] = g["y"].mean()
        out["expected_minus_observed"] = out["expected_difficulties"] - out["observed_difficulties"]
    out.insert(0, "variable", segment)
    return out.reset_index().sort_values("expected_difficulties", ascending=False,
                                         ignore_index=True)


def herfindahl(shares: Any) -> float:
    """Herfindahl-Hirschman index of shares (1/n = evenly spread, 1 = all in one group)."""
    s = np.asarray(shares, dtype="float64")
    if (s < 0).any() or not np.isclose(s.sum(), 1.0):
        raise ValueError("shares must be non-negative and sum to 1")
    return float((s ** 2).sum())


def top_share(values: Any, fraction: float) -> float:
    """Share of the total held by the top `fraction` of borrowers ranked by value."""
    v = np.sort(np.asarray(values, dtype="float64"))[::-1]
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    k = max(int(np.ceil(fraction * len(v))), 1)
    return float(v[:k].sum() / v.sum()) if v.sum() > 0 else 0.0


def lorenz_curve(values: Any, points: int = 101) -> pd.DataFrame:
    """Cumulative share of total held by the top x% of borrowers (riskiest first)."""
    v = np.sort(np.asarray(values, dtype="float64"))[::-1]
    cum = np.concatenate([[0.0], np.cumsum(v)]) / v.sum()
    x = np.linspace(0, 1, points)
    return pd.DataFrame({"share_of_borrowers": x,
                         "share_of_total": np.interp(x * len(v), np.arange(len(v) + 1), cum)})


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.Series:
    """Boolean mask for dashboard filters: {column: list of allowed values} or
    {column: (low, high)} for numeric ranges (inclusive). Unknown columns raise."""
    mask = pd.Series(True, index=df.index)
    for col, cond in filters.items():
        if col not in df.columns:
            raise KeyError(f"cannot filter on unknown column {col!r}")
        if isinstance(cond, tuple) and len(cond) == 2:
            mask &= df[col].between(cond[0], cond[1])
        else:
            mask &= df[col].astype("object").isin(list(cond))
    return mask
