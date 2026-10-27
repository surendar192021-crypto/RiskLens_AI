"""Expected loss: EL = PD x LGD x EAD per borrower; portfolio EL = sum.

Inputs in this project (docs/portfolio_risk.md):
- PD:  the deployed model's probability of TARGET = 1 (early payment difficulty).
       This is NOT a regulatory 12-month default probability; treating every payment
       difficulty as a default makes EL an upper-end illustration.
- LGD: an ASSUMPTION (config expected_loss.lgd); the dataset has no recovery data.
- EAD: proxy = AMT_CREDIT of the application, assumed fully outstanding at default
       (no amortisation); currency unstated, so amounts are in dataset units.

Outputs are assumption-based illustrations. They are not regulatory capital,
provisions, or audited loss estimates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.evaluate import check_probabilities


def _check_lgd(lgd: Any) -> np.ndarray:
    lgd = np.asarray(lgd, dtype="float64")
    if not np.isfinite(lgd).all() or (lgd < 0).any() or (lgd > 1).any():
        raise ValueError("LGD must lie in [0, 1]")
    return lgd


def _check_ead(ead: Any) -> np.ndarray:
    ead = np.asarray(ead, dtype="float64")
    if not np.isfinite(ead).all() or (ead < 0).any():
        raise ValueError("EAD must be finite and non-negative")
    return ead


def expected_loss(pd_: Any, lgd: Any, ead: Any) -> np.ndarray:
    """Borrower-level EL. LGD may be a scalar or one value per borrower."""
    p = check_probabilities(pd_)
    lgd_a, ead_a = _check_lgd(lgd), _check_ead(ead)
    if ead_a.shape != p.shape:
        raise ValueError("PD and EAD must have the same length")
    if lgd_a.ndim and lgd_a.shape != p.shape:
        raise ValueError("LGD must be a scalar or have the same length as PD")
    return p * lgd_a * ead_a


def lgd_sensitivity(pd_: Any, ead: Any, lgd_values: list[float]) -> pd.DataFrame:
    """Portfolio EL for several LGD assumptions (EL is linear in LGD)."""
    total_ead = float(_check_ead(ead).sum())
    rows = []
    for lgd in lgd_values:
        el = float(expected_loss(pd_, lgd, ead).sum())
        rows.append({"lgd": float(lgd), "portfolio_el": el,
                     "el_rate_of_exposure": el / total_ead if total_ead else float("nan")})
    return pd.DataFrame(rows)


def el_by_group(groups: Any, pd_: Any, lgd: float, ead: Any) -> pd.DataFrame:
    el = expected_loss(pd_, lgd, ead)
    t = pd.DataFrame({"group": np.asarray(groups).astype(str), "el": el,
                      "ead": _check_ead(ead)})
    g = t.groupby("group", sort=False)
    out = pd.DataFrame({"borrowers": g.size(), "exposure": g["ead"].sum(), "el": g["el"].sum()})
    out["el_share"] = out["el"] / out["el"].sum()
    out["el_rate_of_exposure"] = out["el"] / out["exposure"].where(out["exposure"] > 0)
    return out.reset_index()
