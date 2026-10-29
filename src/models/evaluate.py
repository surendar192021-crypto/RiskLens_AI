"""Evaluation metrics for predicted default probabilities.

Discrimination (does the model rank risky borrowers higher?): ROC-AUC, PR-AUC.
Probability quality (are the numbers themselves right?): Brier score, log loss, ECE.
Decision quality at a threshold: precision, recall, F1, confusion matrix.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def check_probabilities(p: Any) -> np.ndarray:
    """Return p as a float array; raise if any value is non-finite or outside [0, 1]."""
    p = np.asarray(p, dtype="float64")
    if p.ndim != 1:
        raise ValueError("probabilities must be a 1-D array")
    if not np.isfinite(p).all():
        raise ValueError("probabilities contain NaN or infinite values")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return p


def _check_labels(y: Any) -> np.ndarray:
    y = np.asarray(y)
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0/1")
    return y.astype(int)


def expected_calibration_error(
    y: Any, p: Any, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """Weighted mean |observed default rate - mean predicted probability| over bins.

    strategy="uniform": equal-width bins on [0, 1] (the usual definition).
    strategy="quantile": equal-count bins, more informative when most
    probabilities are small, as here.
    """
    y, p = _check_labels(y), check_probabilities(p)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    bins = np.digitize(p, edges[1:-1], right=True)
    ece = 0.0
    for b in np.unique(bins):
        mask = bins == b
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def threshold_metrics(y: Any, p: Any, threshold: float) -> dict[str, Any]:
    """Classification metrics when borrowers with p >= threshold are flagged."""
    y, p = _check_labels(y), check_probabilities(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "flagged_rate": float(pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_probabilities(y: Any, p: Any) -> dict[str, float]:
    """Threshold-free metrics."""
    y, p = _check_labels(y), check_probabilities(p)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "ece_uniform": expected_calibration_error(y, p, 10, "uniform"),
        "ece_quantile": expected_calibration_error(y, p, 10, "quantile"),
    }


def bootstrap_ci(
    y: Any, p: Any, metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 300, seed: int = 42, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a metric on a fixed evaluation set.

    Measures uncertainty from the finite evaluation sample only (not from retraining).
    """
    y, p = _check_labels(y), check_probabilities(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        values.append(metric(y[idx], p[idx]))
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def paired_bootstrap_diff(
    y: Any, p_a: Any, p_b: Any, metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 300, seed: int = 42, alpha: float = 0.05,
) -> dict[str, float]:
    """metric(A) - metric(B) on the same borrowers, with a paired bootstrap CI.

    Both models are scored on the SAME resampled borrowers each time, so the
    interval reflects the uncertainty of the difference itself; it is narrower
    and more appropriate than comparing two separate intervals.
    """
    y = _check_labels(y)
    p_a, p_b = check_probabilities(p_a), check_probabilities(p_b)
    if not len(y) == len(p_a) == len(p_b):
        raise ValueError("y, p_a and p_b must have the same length")
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "difference": float(metric(y, p_a) - metric(y, p_b)),
        "ci_low": float(low),
        "ci_high": float(high),
        "share_resamples_a_better": float(np.mean(np.asarray(diffs) > 0)),
    }
