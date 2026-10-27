"""SHAP explanations for the deployed tree model.

What SHAP gives here: for one borrower, how much each feature moved the model's
raw score (log-odds of payment difficulty) away from the average raw score.
Contributions add up exactly: base value + sum(contributions) = raw score.

What SHAP does NOT give: causal effects. A large contribution says the model
relied on that feature for this borrower; it does not say that changing the
feature in real life would change the borrower's real risk.

If the deployed model is calibrated, the explanation is of the underlying
model's raw score (calibration is a monotone map applied afterwards).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from src.models.calibrate import base_model
from src.models.predict import predict_proba


def _sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype="float64")))


class ShapExplainer:
    def __init__(self, model: Any):
        self.model = model  # deployed model (possibly calibrated): used for probabilities
        self.pipeline = base_model(model)
        steps = getattr(self.pipeline, "named_steps", {})
        if "features" not in steps or "model" not in steps:
            raise TypeError("expected a Pipeline with 'features' and 'model' steps")
        self.frame = steps["features"]
        self.tree = steps["model"]
        self.explainer = shap.TreeExplainer(self.tree)  # exact TreeSHAP, raw (log-odds) output

    def explain(self, df: pd.DataFrame) -> shap.Explanation:
        X = self.frame.transform(df)
        exp = self.explainer(X)
        raw = self.tree.predict(X, raw_score=True)
        err = np.abs(exp.values.sum(axis=1) + exp.base_values - raw).max()
        if err > 1e-6:
            raise AssertionError(f"SHAP values do not add up to the model output (error {err})")
        return exp

    @property
    def base_value(self) -> float:
        return float(np.ravel(self.explainer.expected_value)[0])

    def global_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per feature, next to LightGBM's own split-gain importance."""
        exp = self.explain(df)
        mean_abs = np.abs(exp.values).mean(axis=0)
        mean_signed = exp.values.mean(axis=0)
        out = pd.DataFrame(
            {"feature": exp.feature_names, "mean_abs_shap": mean_abs,
             "mean_shap": mean_signed, "share_of_total": mean_abs / mean_abs.sum()}
        )
        if hasattr(self.tree, "booster_"):
            gain = self.tree.booster_.feature_importance(importance_type="gain")
            names = self.tree.booster_.feature_name()
            gain_s = pd.Series(gain / gain.sum(), index=names)
            out["gain_share"] = out["feature"].map(gain_s).to_numpy()
            out["gain_rank"] = out["gain_share"].rank(ascending=False, method="min").astype(int)
        out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        out.insert(0, "shap_rank", np.arange(1, len(out) + 1))
        return out

    def explain_borrower(self, row: pd.DataFrame) -> dict[str, Any]:
        """Full explanation of one borrower (a one-row DataFrame)."""
        if len(row) != 1:
            raise ValueError("explain_borrower expects exactly one row")
        exp = self.explain(row)
        contrib = pd.DataFrame(
            {"feature": exp.feature_names, "value": list(exp.data[0]), "shap": exp.values[0]}
        )
        contrib["abs_shap"] = contrib["shap"].abs()
        contrib = contrib.sort_values("abs_shap", ascending=False).reset_index(drop=True)
        raw_score = float(exp.base_values[0] + exp.values[0].sum())
        return {
            "probability": float(predict_proba(self.model, row)[0]),
            "raw_score": raw_score,
            "base_value": float(exp.base_values[0]),
            "base_probability": float(_sigmoid(exp.base_values[0])),
            "contributions": contrib.drop(columns="abs_shap"),
            "explanation": exp[0],
        }


def format_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"
    return str(v)


def plain_english(
    result: dict[str, Any], portfolio_mean: float, threshold: float | None = None, n: int = 3
) -> str:
    """Short, non-causal summary of a borrower explanation."""
    c = result["contributions"]
    up = c[c["shap"] > 0].head(n)
    down = c[c["shap"] < 0].head(n)

    def items(df: pd.DataFrame) -> str:
        if df.empty:
            return "none"
        return "; ".join(f"{r.feature} = {format_value(r.value)} ({r.shap:+.2f})"
                         for r in df.itertuples())

    p = result["probability"]
    rel = "above" if p > portfolio_mean else "below"
    text = (f"Estimated probability of payment difficulty: {p:.1%} "
            f"({rel} the portfolio average of {portfolio_mean:.1%}")
    if threshold is not None:
        text += f"; {'at or above' if p >= threshold else 'below'} the review threshold "
        text += f"of {threshold:.1%}"
    text += ").\n"
    text += f"Features that raised the model's estimate most (log-odds): {items(up)}.\n"
    text += f"Features that lowered it most (log-odds): {items(down)}.\n"
    text += ("These are contributions to the model's output for this borrower, "
             "not proven causes of payment difficulty.")
    return text


def numeric_view(exp: shap.Explanation) -> shap.Explanation:
    """Copy with categorical values replaced by codes, so beeswarm plots can colour them."""
    data = pd.DataFrame(exp.data, columns=exp.feature_names)
    for c in data.columns:
        num = pd.to_numeric(data[c], errors="coerce")
        if num.notna().sum() < data[c].notna().sum():  # text values present -> categorical
            codes = data[c].astype("category").cat.codes.astype("float64")
            num = codes.where(codes >= 0)
        data[c] = num.astype("float64")
    return shap.Explanation(
        values=exp.values, base_values=exp.base_values,
        data=data.to_numpy(dtype="float64"), feature_names=exp.feature_names,
    )
