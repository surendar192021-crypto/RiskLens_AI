"""Choosing a classification threshold on validation data.

A threshold turns a probability into a yes/no flag. It is a business choice that
trades missed defaulters (false negatives) against wrongly flagged good borrowers
(false positives), so 0.5 is not assumed. Thresholds must be chosen on validation
data and then applied unchanged to test data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.models.evaluate import _check_labels, check_probabilities


def max_f1_threshold(y: Any, p: Any) -> tuple[float, float]:
    """Threshold that maximises F1 (equal weight on precision and recall)."""
    y, p = _check_labels(y), check_probabilities(p)
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def min_cost_threshold(
    y: Any, p: Any, cost_fn: float, cost_fp: float, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Threshold minimising cost_fn * FN + cost_fp * FP (average cost per borrower)."""
    if cost_fn < 0 or cost_fp < 0:
        raise ValueError("costs must be non-negative")
    y, p = _check_labels(y), check_probabilities(p)
    grid = np.linspace(0.0, 1.0, 1001) if grid is None else np.asarray(grid)
    best_t, best_cost = 0.0, np.inf
    for t in grid:
        pred = p >= t
        fn = np.sum((~pred) & (y == 1))
        fp = np.sum(pred & (y == 0))
        cost = (cost_fn * fn + cost_fp * fp) / len(y)
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost


def recall_target_threshold(y: Any, p: Any, target_recall: float) -> float:
    """Highest threshold that still catches at least `target_recall` of actual positives."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    y, p = _check_labels(y), check_probabilities(p)
    positives = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target_recall * len(positives)))
    return float(positives[k - 1])
