"""Phase 4: train every configured model on the TRAIN split and compare on VALIDATION.

The calibration and test splits are not touched here.

Outputs:
- models/<run_id>/<experiment>.joblib (+ .json metadata) and validation_predictions.parquet
- models/latest_phase4.json                       pointer to the run, champion and challenger
- reports/tables/model_comparison_validation.csv
- reports/figures/phase4_{roc,pr,reliability}_validation.png
- reports/experiment_results/<run_id>_summary.json
- one row per experiment appended to docs/experiment_log.md

Run from the project root:  python -m scripts.train_models
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT, load_config, project_path
from src.models.evaluate import bootstrap_ci, evaluate_probabilities, threshold_metrics
from src.models.predict import predict_proba, save_model
from src.models.thresholds import max_f1_threshold
from src.models.transformers import (
    FeatureFrame,
    logistic_regression_pipeline,
    random_forest_pipeline,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_modelling_data(cfg: dict[str, Any]) -> dict[str, Any]:
    processed = project_path(cfg, "processed_data")
    df = pd.read_parquet(processed / "modelling_train.parquet")
    splits = pd.read_parquet(processed / "splits.parquet")
    features = json.loads((processed / "feature_list.json").read_text(encoding="utf-8"))
    df = df.merge(splits, on=features["id"], how="left", validate="one_to_one")
    if df["split"].isna().any():
        raise ValueError("some applicants have no split assignment; rebuild the dataset")
    return {"df": df, "features": features}


def boosting_pipeline(
    family: str, numeric: list[str], categorical: list[str], params: dict[str, Any],
    pos_weight: float, seed: int, early_stopping: int,
    X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
) -> tuple[Pipeline, int]:
    """Fit XGBoost / LightGBM with early stopping on validation log-loss.

    Both use pandas categorical columns natively and handle missing values themselves.
    Validation is used only to decide when to stop adding trees (model selection),
    never to fit the trees.
    """
    frame = FeatureFrame(numeric, categorical).fit(X_tr)
    Xt, Xv = frame.transform(X_tr), frame.transform(X_val)
    if family == "xgboost":
        model = xgb.XGBClassifier(
            **params, tree_method="hist", enable_categorical=True, eval_metric="logloss",
            early_stopping_rounds=early_stopping, scale_pos_weight=pos_weight,
            random_state=seed, n_jobs=-1,
        )
        model.fit(Xt, y_tr, eval_set=[(Xv, y_val)], verbose=False)
        best = int(model.best_iteration) + 1
    elif family == "lightgbm":
        model = lgb.LGBMClassifier(
            **params, scale_pos_weight=pos_weight, random_state=seed, n_jobs=-1, verbose=-1
        )
        model.fit(
            Xt, y_tr, eval_set=[(Xv, y_val)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        best = int(model.best_iteration_)
    else:
        raise ValueError(f"unknown boosting family {family!r}")
    return Pipeline([("features", frame), ("model", model)]), best


def select_champion(results: pd.DataFrame, metric: str) -> tuple[str, str]:
    """Champion = best `metric` (ties: lower Brier). Challenger = best of a different family."""
    ranked = results.sort_values([metric, "brier"], ascending=[False, True])
    champion = ranked.iloc[0]
    others = ranked[ranked["family"] != champion["family"]]
    return str(champion["experiment"]), str(others.iloc[0]["experiment"])


def plot_curves(preds: pd.DataFrame, y: np.ndarray, figures: Path) -> list[str]:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    figures.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in preds.columns:
        fpr, tpr, _ = roc_curve(y, preds[name])
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, preds[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves (validation split)")
    ax.legend(fontsize=7)
    paths.append(_save(fig, figures / "phase4_roc_validation.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in preds.columns:
        prec, rec, _ = precision_recall_curve(y, preds[name])
        ax.plot(rec, prec, label=f"{name} ({average_precision_score(y, preds[name]):.3f})")
    ax.axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"base rate {y.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (validation split)")
    ax.legend(fontsize=7)
    paths.append(_save(fig, figures / "phase4_pr_validation.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in preds.columns:
        frac, mean_p = calibration_curve(y, preds[name], n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", ms=3, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed TARGET rate",
           title="Reliability before calibration (validation, 10 equal-count bins)")
    ax.legend(fontsize=7)
    paths.append(_save(fig, figures / "phase4_reliability_validation.png"))
    return paths


def _save(fig: Any, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT))


def append_experiment_log(rows: list[dict[str, Any]]) -> None:
    path = PROJECT_ROOT / "docs" / "experiment_log.md"
    text = path.read_text(encoding="utf-8").replace("_No experiments run yet._\n", "")
    lines = [
        f"| {r['date']} | {r['run_id']} | 4 | {r['dataset_version']} | {r['n_features']} features "
        f"| train → validation | {r['experiment']} ({r['params']}) | {r['seed']} "
        f"| {r['imbalance']} "
        f"| none | max-F1 on validation | ROC-AUC {r['roc_auc']:.4f}, PR-AUC {r['pr_auc']:.4f}, "
        f"Brier {r['brier']:.4f}, ECE(q) {r['ece_quantile']:.4f} | {r['notes']} |"
        for r in rows
    ]
    path.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def train_all(cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    seed = cfg["project"]["random_seed"]
    mcfg = cfg["modelling"]
    run_id = f"phase4_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = project_path(cfg, "models") / run_id

    log("Loading modelling data")
    data = load_modelling_data(cfg)
    df, feats = data["df"], data["features"]
    numeric, categorical, target = feats["numeric"], feats["categorical"], feats["target"]
    tr, val = df[df["split"] == "train"], df[df["split"] == "validation"]
    X_tr, y_tr = tr.drop(columns=[target, "split"]), tr[target].astype(int)
    X_val, y_val = val.drop(columns=[target, "split"]), val[target].astype(int)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    log(f"train {len(tr):,} rows, validation {len(val):,} rows, "
        f"{len(numeric)} numeric + {len(categorical)} categorical features, "
        f"positive weight if used = {pos_weight:.2f}")

    results, preds, log_rows = [], {}, []
    for exp in mcfg["experiments"]:
        name, family, weighted = exp["name"], exp["family"], exp["weighted"]
        params = mcfg["params"][family]
        log(f"Training {name}")
        t0 = time.time()
        extra: dict[str, Any] = {}
        if family == "logistic_regression":
            pipe = logistic_regression_pipeline(
                numeric, categorical, params, "balanced" if weighted else None, seed
            ).fit(X_tr, y_tr)
            extra["n_iter"] = int(pipe.named_steps["model"].n_iter_[0])
        elif family == "random_forest":
            pipe = random_forest_pipeline(
                numeric, categorical, params, "balanced_subsample" if weighted else None, seed
            ).fit(X_tr, y_tr)
        else:
            pipe, best = boosting_pipeline(
                family, numeric, categorical, params, pos_weight if weighted else 1.0, seed,
                mcfg["early_stopping_rounds"], X_tr, y_tr, X_val, y_val,
            )
            extra["best_iteration"] = best
        fit_seconds = time.time() - t0

        p_val = predict_proba(pipe, X_val)
        preds[name] = p_val
        metrics = evaluate_probabilities(y_val, p_val)
        n_boot = mcfg["bootstrap_resamples"]
        metrics["roc_auc_ci"] = bootstrap_ci(y_val, p_val, roc_auc_score, n_boot, seed)
        metrics["pr_auc_ci"] = bootstrap_ci(y_val, p_val, average_precision_score, n_boot, seed)
        t_f1, _ = max_f1_threshold(y_val, p_val)
        at_t = threshold_metrics(y_val, p_val, t_f1)
        row = {
            "experiment": name, "family": family, "weighted": weighted,
            **{k: v for k, v in metrics.items() if not k.endswith("_ci")},
            "roc_auc_ci_low": metrics["roc_auc_ci"][0],
            "roc_auc_ci_high": metrics["roc_auc_ci"][1],
            "pr_auc_ci_low": metrics["pr_auc_ci"][0],
            "pr_auc_ci_high": metrics["pr_auc_ci"][1],
            "f1_threshold": at_t["threshold"], "precision_at_t": at_t["precision"],
            "recall_at_t": at_t["recall"], "f1_at_t": at_t["f1"],
            "flagged_rate_at_t": at_t["flagged_rate"],
            "tn": at_t["tn"], "fp": at_t["fp"], "fn": at_t["fn"], "tp": at_t["tp"],
            "fit_seconds": round(fit_seconds, 1), **extra,
        }
        results.append(row)
        log(f"  {name}: ROC-AUC {row['roc_auc']:.4f}  PR-AUC {row['pr_auc']:.4f}  "
            f"Brier {row['brier']:.4f}  ({fit_seconds:.0f} s)")

        save_model(
            pipe,
            {"run_id": run_id, "experiment": name, "family": family, "weighted": weighted,
             "params": params, "seed": seed, "trained_on": "train split",
             "dataset_version": feats["dataset_version"], "numeric": numeric,
             "categorical": categorical, "validation_metrics": row,
             "created_at": datetime.now().isoformat(timespec="seconds")},
            run_dir, name,
        )
        log_rows.append({
            "date": datetime.now().strftime("%Y-%m-%d"), "run_id": run_id,
            "dataset_version": feats["dataset_version"],
            "n_features": len(numeric) + len(categorical), "experiment": name,
            "params": ", ".join(f"{k}={v}" for k, v in params.items())
            + (f", trees={extra['best_iteration']}" if "best_iteration" in extra else ""),
            "seed": seed,
            "imbalance": "class weights" if weighted else "none",
            "notes": "validation used for early stopping" if "best_iteration" in extra else "",
            **row,
        })

    results_df = pd.DataFrame(results)
    champion, challenger = select_champion(results_df, mcfg["selection_metric"])
    log(f"Champion: {champion}; challenger: {challenger}")

    pred_df = pd.DataFrame(preds)
    pd.concat(
        [X_val[[feats["id"]]].reset_index(drop=True), y_val.reset_index(drop=True), pred_df],
        axis=1,
    ).to_parquet(run_dir / "validation_predictions.parquet", index=False)
    results_df.to_csv(project_path(cfg, "tables") / "model_comparison_validation.csv", index=False)
    figures = plot_curves(pred_df, y_val.to_numpy(), project_path(cfg, "figures"))
    append_experiment_log(log_rows)

    summary = {
        "run_id": run_id, "champion": champion, "challenger": challenger,
        "selection_metric": mcfg["selection_metric"], "results": results,
        "figures": figures, "positive_weight": pos_weight,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "xgboost": xgb.__version__, "lightgbm": lgb.__version__},
        "runtime_seconds": round(time.time() - start, 1),
    }
    out = project_path(cfg, "experiment_results")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    (project_path(cfg, "models") / "latest_phase4.json").write_text(
        json.dumps({"run_id": run_id, "champion": champion, "challenger": challenger}, indent=2),
        "utf-8",
    )
    log(f"Done in {summary['runtime_seconds']} s")
    return summary


def main() -> int:
    train_all(load_config())
    return 0
