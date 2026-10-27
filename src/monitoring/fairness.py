"""Group-wise fairness metrics for a scored, LABELLED sample.

For each group: size, observed difficulty rate, mean PD, calibration gap
(mean PD - observed rate), flag rate at the operating threshold (selection rate),
true-positive rate (share of actual difficulty cases flagged), false-positive rate
(share of non-difficulty cases flagged), precision, and ROC-AUC within the group.

Several fairness criteria exist and they generally cannot all hold at once when
base rates differ (Kleinberg et al. 2017; Chouldechova 2017). These numbers describe
disparities; they do not by themselves establish that the model is fair or unfair,
and they say nothing about legal compliance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.evaluate import check_probabilities
from src.risk.rating_migration import wilson_interval


def group_metrics(y: Any, p: Any, groups: Any, threshold: float,
                  min_size: int = 500) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = check_probabilities(p)
    g = pd.Series(groups).astype("object").where(pd.Series(groups).notna(), "Missing")
    g = g.astype(str).to_numpy()
    flag = p >= threshold
    rows = []
    for name in sorted(set(g)):
        m = g == name
        yy, pp, ff = y[m], p[m], flag[m]
        tp, fn = int((ff & (yy == 1)).sum()), int((~ff & (yy == 1)).sum())
        fp, tn = int((ff & (yy == 0)).sum()), int((~ff & (yy == 0)).sum())
        rows.append({
            "group": name, "n": int(m.sum()), "positives": int(yy.sum()),
            "base_rate": float(yy.mean()), "mean_pd": float(pp.mean()),
            "calibration_gap": float(pp.mean() - yy.mean()),
            "flagged": int(ff.sum()), "flag_rate": float(ff.mean()),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "tpr": tp / (tp + fn) if tp + fn else np.nan,
            "fpr": fp / (fp + tn) if fp + tn else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "auc": float(roc_auc_score(yy, pp)) if 0 < yy.sum() < len(yy) else np.nan,
            "reliable": bool(m.sum() >= min_size),
        })
    out = pd.DataFrame(rows)
    for rate, k, n in [("flag_rate", "flagged", "n"), ("tpr", "tp", "positives")]:
        lo, hi = wilson_interval(out[k].to_numpy(), out[n].to_numpy())
        out[f"{rate}_ci_low"], out[f"{rate}_ci_high"] = lo, hi
    lo, hi = wilson_interval(out["fp"].to_numpy(), (out["fp"] + out["tn"]).to_numpy())
    out["fpr_ci_low"], out["fpr_ci_high"] = lo, hi
    return out


def _count_and_total(row: pd.Series, k: str, n_col: str | None) -> tuple[float, float]:
    """(successes, trials) for a rate; FPR's trials are the actual negatives (fp + tn)."""
    n = row["fp"] + row["tn"] if n_col is None else row[n_col]
    return float(row[k]), float(n)


def difference_vs_reference(metrics: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Each group's rate minus the reference group's, with a 95% normal-approximation CI
    for a difference of two independent proportions."""
    m = metrics.set_index("group")
    if reference not in m.index:
        raise KeyError(f"reference group {reference!r} not found")
    specs = {"flag_rate": ("flagged", "n"), "tpr": ("tp", "positives"),
             "fpr": ("fp", None), "base_rate": ("positives", "n")}
    rows = []
    for grp in m.index:
        if grp == reference:
            continue
        for metric, (k, n_col) in specs.items():
            k1, n1 = _count_and_total(m.loc[grp], k, n_col)
            k0, n0 = _count_and_total(m.loc[reference], k, n_col)
            if n1 == 0 or n0 == 0:
                continue
            p1, p0 = k1 / n1, k0 / n0
            se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
            rows.append({"group": grp, "reference": reference, "metric": metric,
                         "group_value": p1, "reference_value": p0, "difference": p1 - p0,
                         "ci_low": p1 - p0 - 1.96 * se, "ci_high": p1 - p0 + 1.96 * se})
    return pd.DataFrame(rows)


def parity_ratios(metrics: pd.DataFrame, reliable_only: bool = True) -> dict[str, float]:
    """min/max across groups for flag rate, TPR and FPR (1 = perfect parity)."""
    m = metrics[metrics["reliable"]] if reliable_only else metrics
    out = {}
    for col in ("flag_rate", "tpr", "fpr", "precision"):
        v = m[col].dropna()
        out[col] = float(v.min() / v.max()) if len(v) and v.max() > 0 else np.nan
    return out
