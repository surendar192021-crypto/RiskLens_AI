"""Monte Carlo portfolio loss simulation with a Vasicek one-factor Gaussian model.

Model (Vasicek 2002; Gordy 2003): borrower i defaults in a simulation when
    sqrt(rho_i) * Z + sqrt(1 - rho_i) * eps_i  <  Phi^-1(PD_i)
with Z ~ N(0, 1) shared by all borrowers (the "economy") and eps_i ~ N(0, 1) independent.
Given Z, defaults are independent with conditional probability
    PD_i(Z) = Phi((Phi^-1(PD_i) - sqrt(rho_i) * Z) / sqrt(1 - rho_i)),
and E[PD_i(Z)] = PD_i, so the model keeps each borrower's PD while making defaults
move together. rho = 0 gives independent defaults.

Loss in one simulation = sum over defaulted borrowers of LGD_i * EAD_i.

Assumptions here: rho and LGD are assumed (docs/simulation_and_stress.md); the
"horizon" is that of the PD (TARGET = early payment difficulty), not one year.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

from src.models.evaluate import check_probabilities

_EPS = 1e-12


def basel_other_retail_rho(pd_: Any) -> np.ndarray:
    """Basel II correlation function for 'other retail' exposures (0.03 to 0.16)."""
    p = check_probabilities(pd_)
    w = (1 - np.exp(-35 * p)) / (1 - np.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


def resolve_rho(rho: Any, pd_: Any) -> np.ndarray:
    if isinstance(rho, str):
        if rho != "basel_other_retail":
            raise ValueError(f"unknown rho option {rho!r}")
        return basel_other_retail_rho(pd_)
    r = np.asarray(rho, dtype="float64")
    if not np.isfinite(r).all() or (r < 0).any() or (r >= 1).any():
        raise ValueError("rho must lie in [0, 1)")
    return r


def simulate_losses(
    pd_: Any, ead: Any, lgd: Any, rho: Any, n_sims: int, seed: int, chunk_size: int = 250,
) -> dict[str, np.ndarray]:
    """Simulated portfolio losses and default counts (one value per simulation)."""
    p = check_probabilities(pd_)
    ead = np.asarray(ead, dtype="float64")
    lgd = np.broadcast_to(np.asarray(lgd, dtype="float64"), p.shape)
    if ead.shape != p.shape or not np.isfinite(ead).all() or (ead < 0).any():
        raise ValueError("EAD must be finite, non-negative and match PD")
    if (lgd < 0).any() or (lgd > 1).any():
        raise ValueError("LGD must lie in [0, 1]")
    if n_sims < 1 or chunk_size < 1:
        raise ValueError("n_sims and chunk_size must be positive")
    r = np.broadcast_to(resolve_rho(rho, p), p.shape)
    threshold = ndtri(np.clip(p, _EPS, 1 - _EPS))
    a, b = np.sqrt(r), np.sqrt(1 - r)
    loss_per_default = lgd * ead

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    losses = np.empty(n_sims)
    defaults = np.empty(n_sims, dtype=np.int64)
    for start in range(0, n_sims, chunk_size):
        zc = z[start:start + chunk_size]
        cond = ndtr((threshold - a * zc[:, None]) / b)          # (chunk, borrowers)
        d = rng.random(cond.shape) < cond
        losses[start:start + len(zc)] = d @ loss_per_default
        defaults[start:start + len(zc)] = d.sum(axis=1)
    return {"losses": losses, "defaults": defaults, "factor": z}


def loss_statistics(
    losses: np.ndarray, analytic_el: float, levels: list[float]
) -> dict[str, float]:
    """Summary of a simulated loss distribution.

    VaR_a = a-quantile of loss. Expected shortfall ES_a = mean loss at or beyond VaR_a.
    Unexpected loss (definition used here) UL_a = VaR_a - analytic EL.
    """
    losses = np.asarray(losses, dtype="float64")
    out = {
        "n_simulations": int(len(losses)),
        "analytic_el": float(analytic_el),
        "mean_loss": float(losses.mean()),
        "mc_standard_error": float(losses.std(ddof=1) / np.sqrt(len(losses))),
        "median_loss": float(np.median(losses)),
        "sd_loss": float(losses.std(ddof=1)),
    }
    for a in levels:
        var = float(np.quantile(losses, a))
        key = f"{a * 100:g}"
        out[f"var_{key}"] = var
        out[f"es_{key}"] = float(losses[losses >= var].mean())
        out[f"ul_{key}"] = var - float(analytic_el)
    return out


def vasicek_loss_quantile(pd_: float, rho: float, alpha: float) -> float:
    """Large-portfolio limit (Vasicek): alpha-quantile of the default FRACTION of a
    homogeneous portfolio with PD pd_ and correlation rho."""
    return float(ndtr((ndtri(pd_) + np.sqrt(rho) * ndtri(alpha)) / np.sqrt(1 - rho)))
