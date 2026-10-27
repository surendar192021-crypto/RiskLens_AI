"""Stress testing by FEATURE PERTURBATION.

A scenario changes borrowers' raw inputs, recomputes every derived feature with the
same functions used to build the dataset, and rescores everyone with the deployed
model. Probabilities are never multiplied directly.

What this is: a hypothetical "what would the model say if..." experiment.
What it is not: a macroeconomic stress model calibrated to economic data (none is
available), or a forecast. A model's response to a shock is only as credible as the
model's learned relationship with that feature.

Scenario keys (all optional): income_cut, annuity_rise, ext_source_cut,
job_loss_share + income_after_job_loss, lgd (portfolio LGD override).
Order of application: job loss, then income cut, then annuity rise, then external scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_engineering import (
    EXT_SOURCES,
    employment_features,
    ext_source_features,
    loan_ratio_features,
)
from src.simulation.scenarios import scale_income

KNOWN_KEYS = {"income_cut", "annuity_rise", "ext_source_cut", "job_loss_share",
              "income_after_job_loss", "lgd"}
EMPLOYED_TYPES = ["Working", "Commercial associate", "State servant"]


def _fraction(spec: dict[str, Any], key: str, upper: float = 1.0) -> float:
    v = float(spec[key])
    if not 0 <= v < upper:
        raise ValueError(f"{key} must be in [0, {upper}), got {v}")
    return v


def validate_scenario(spec: dict[str, Any]) -> None:
    unknown = set(spec) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
    for key in ("income_cut", "ext_source_cut", "job_loss_share"):
        if key in spec:
            _fraction(spec, key)
    if "annuity_rise" in spec:
        _fraction(spec, "annuity_rise", upper=5.0)
    if "job_loss_share" in spec and "income_after_job_loss" not in spec:
        raise ValueError("job_loss_share needs income_after_job_loss")
    if "income_after_job_loss" in spec and not 0 < float(spec["income_after_job_loss"]) <= 1:
        raise ValueError("income_after_job_loss must be in (0, 1]")
    if "lgd" in spec and not 0 <= float(spec["lgd"]) <= 1:
        raise ValueError("lgd must be in [0, 1]")


def apply_scenario(df: pd.DataFrame, spec: dict[str, Any], seed: int = 42) -> pd.DataFrame:
    """Return a stressed copy of the borrower table (the input is not modified)."""
    validate_scenario(spec)
    out = df.copy()
    if spec.get("job_loss_share", 0) > 0:
        employed = out["NAME_INCOME_TYPE"].astype("object").isin(EMPLOYED_TYPES).to_numpy()
        rng = np.random.default_rng(seed)
        lose = employed & (rng.random(len(out)) < float(spec["job_loss_share"]))
        factor = np.where(lose, float(spec["income_after_job_loss"]), 1.0)
        out = scale_income(out, factor)
        # a new job (or none) means current-job tenure restarts at zero
        out.loc[lose, "EMPLOYED_YEARS"] = 0.0
        employment_features(out)
        out["_job_loss"] = lose
    if spec.get("income_cut", 0) > 0:
        out = scale_income(out, 1 - float(spec["income_cut"]))
    if spec.get("annuity_rise", 0) > 0:
        out["AMT_ANNUITY"] = out["AMT_ANNUITY"] * (1 + float(spec["annuity_rise"]))
        loan_ratio_features(out)
    if spec.get("ext_source_cut", 0) > 0:
        for c in EXT_SOURCES:
            out[c] = (out[c] * (1 - float(spec["ext_source_cut"]))).clip(0, 1)
        ext_source_features(out)
    return out


def check_scenario_output(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Sanity checks: same rows, valid ranges, no infinities in numeric features."""
    if len(before) != len(after) or not before["SK_ID_CURR"].equals(after["SK_ID_CURR"]):
        raise AssertionError("scenario changed the set or order of borrowers")
    if (after["AMT_INCOME_TOTAL"] <= 0).any():
        raise AssertionError("non-positive income after scenario")
    ext = after[EXT_SOURCES]
    if ((ext < 0) | (ext > 1)).any().any():
        raise AssertionError("EXT_SOURCE outside [0, 1] after scenario")
    num = after.select_dtypes("number").drop(columns=["_job_loss"], errors="ignore")
    if np.isinf(num.to_numpy(dtype="float64")).any():
        raise AssertionError("infinite values after scenario")
