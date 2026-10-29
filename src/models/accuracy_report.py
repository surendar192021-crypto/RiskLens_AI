"""Accuracy of every model, for reference.

Accuracy is NOT used to choose models here. With 8% positives, predicting "no payment
difficulty" for everyone is already about 92% accurate while catching nobody, so
accuracy hides the thing a lender cares about. It is reported so that the question
"what is each model's accuracy?" has a measured answer next to that baseline.

Two tables, both computed from saved outputs (nothing is retrained or re-scored):
- validation: all six Phase 4 experiments, from models/<run>/validation_predictions.parquet,
  at the default 0.5 cut-off and at each model's own max-F1 threshold
  (chosen on the same validation data, so slightly optimistic)
- test: the three Phase 5 final models, from the confusion counts already saved in
  reports/tables/final_test_metrics.csv (threshold chosen on validation)

Run from the project root:  python -m scripts.accuracy_report
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.models.evaluate import evaluate_probabilities, threshold_metrics
from src.models.thresholds import max_f1_threshold

BASELINE = "baseline: predict no difficulty for everyone"


def accuracy_table(y: Any, preds: dict[str, np.ndarray], champion: str | None = None
                   ) -> pd.DataFrame:
    """One row per model plus the all-negative baseline."""
    y = np.asarray(y).astype(int)
    rows = [{"model": BASELINE, "accuracy_at_0_5": 1 - y.mean(), "recall_at_0_5": 0.0,
             "flagged_at_0_5": 0.0, "accuracy_at_threshold": 1 - y.mean(),
             "recall_at_threshold": 0.0}]
    for name, p in preds.items():
        t, _ = max_f1_threshold(y, p)
        at5, att = threshold_metrics(y, p, 0.5), threshold_metrics(y, p, t)
        m = evaluate_probabilities(y, p)
        rows.append({
            "model": name, "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
            "accuracy_at_0_5": at5["accuracy"], "recall_at_0_5": at5["recall"],
            "flagged_at_0_5": at5["flagged_rate"], "threshold_max_f1": t,
            "accuracy_at_threshold": att["accuracy"], "precision_at_threshold": att["precision"],
            "recall_at_threshold": att["recall"], "f1_at_threshold": att["f1"],
        })
    out = pd.DataFrame(rows)
    out["chosen"] = out["model"] == champion
    order = ["model", "roc_auc", "pr_auc", "accuracy_at_0_5", "recall_at_0_5", "flagged_at_0_5",
             "threshold_max_f1", "accuracy_at_threshold", "precision_at_threshold",
             "recall_at_threshold", "f1_at_threshold", "chosen"]
    return out[order]


def test_accuracy_from_saved(final_metrics: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of the final models on test, from their saved confusion counts."""
    c = final_metrics
    n = c["tp_at_t"] + c["tn_at_t"] + c["fp_at_t"] + c["fn_at_t"]
    positives = c["tp_at_t"] + c["fn_at_t"]
    return pd.DataFrame({
        "model": c["model"], "threshold_from_validation": c["threshold_from_validation"],
        "accuracy_at_threshold": (c["tp_at_t"] + c["tn_at_t"]) / n,
        "baseline_accuracy": 1 - positives / n,
        "precision_at_threshold": c["precision_at_t"], "recall_at_threshold": c["recall_at_t"],
        "roc_auc": c["roc_auc"], "pr_auc": c["pr_auc"],
    })


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    models_dir, tables = project_path(cfg, "models"), project_path(cfg, "tables")
    pointer = json.loads((models_dir / "latest_phase4.json").read_text(encoding="utf-8"))
    val = pd.read_parquet(models_dir / pointer["run_id"] / "validation_predictions.parquet")
    target = cfg["data"]["target_column"]
    preds = {c: val[c].to_numpy() for c in val.columns if c not in (target, "SK_ID_CURR")}
    val_table = accuracy_table(val[target], preds, pointer["champion"])
    val_table.to_csv(tables / "model_accuracy_validation.csv", index=False)
    test_table = test_accuracy_from_saved(pd.read_csv(tables / "final_test_metrics.csv"))
    test_table.to_csv(tables / "model_accuracy_test.csv", index=False)
    return {"validation": val_table, "test": test_table}


def main() -> int:
    out = run(load_config())
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out["validation"].round(4).to_string(index=False))
        print()
        print(out["test"].round(4).to_string(index=False))
    return 0
