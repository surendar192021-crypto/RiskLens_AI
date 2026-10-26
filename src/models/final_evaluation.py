"""Phase 5: calibrate, choose a threshold, then evaluate ONCE on the test split.

Data use (no split is used for two purposes):
- calibration split: fits the calibration maps only
- validation split:  chooses the calibration method and the threshold
- test split:        final reported numbers; nothing is chosen on it

Outputs:
- models/<run_id>/{champion,challenger}_final.joblib (+ .json metadata)
- models/current_model.json         the model the rest of RiskLens uses
- reports/tables/calibration_comparison_validation.csv, threshold_options_validation.csv,
  final_test_metrics.csv, paired_differences_test.csv, risk_band_distribution_test.csv
- reports/figures/phase5_*.png
- reports/experiment_results/<run_id>_summary.json, rows in docs/experiment_log.md

Run from the project root:  python -m scripts.calibrate_and_evaluate
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import PROJECT_ROOT, load_config, project_path
from src.models.calibrate import METHODS, choose_method, fit_calibrated
from src.models.evaluate import (
    bootstrap_ci,
    evaluate_probabilities,
    paired_bootstrap_diff,
    threshold_metrics,
)
from src.models.predict import load_model, predict_proba, save_model
from src.models.thresholds import max_f1_threshold, min_cost_threshold, recall_target_threshold
from src.models.train import load_modelling_data
from src.risk.borrower_risk import band_distribution

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

BASELINE = "logistic_regression"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def threshold_options(y: Any, p: Any, tcfg: dict[str, Any]) -> pd.DataFrame:
    """The three threshold policies, all computed on validation data."""
    t_f1, _ = max_f1_threshold(y, p)
    t_cost, _ = min_cost_threshold(y, p, tcfg["cost_fn"], tcfg["cost_fp"])
    t_rec = recall_target_threshold(y, p, tcfg["recall_target"])
    rows = []
    for policy, t, rule in [
        ("max_f1", t_f1, "maximise F1"),
        ("min_cost", t_cost, f"minimise {tcfg['cost_fn']}*FN + {tcfg['cost_fp']}*FP"),
        ("recall_target", t_rec, f"recall >= {tcfg['recall_target']}"),
    ]:
        rows.append({"policy": policy, "rule": rule, **threshold_metrics(y, p, t)})
    return pd.DataFrame(rows)


def reliability_plot(curves: dict[str, np.ndarray], y: np.ndarray, title: str, path: Any) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in curves.items():
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", ms=3, label=name)
    top = max(float(np.max([np.quantile(p, 0.995) for p in curves.values()])), 0.05)
    ax.plot([0, top], [0, top], "k--", lw=0.8, label="perfect calibration")
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="Mean predicted probability",
           ylabel="Observed TARGET rate", title=title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT))


def histogram_plot(p: np.ndarray, edges: list[float], threshold: float, path: Any) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(p, bins=100, range=(0, 1), color="#4C72B0")
    for e in edges[1:-1]:
        ax.axvline(e, color="grey", ls=":", lw=1)
    ax.axvline(threshold, color="#C44E52", lw=1.5, label=f"operating threshold {threshold:.3f}")
    ax.set(xlabel="Predicted probability of payment difficulty (calibrated champion)",
           ylabel="Borrowers (test split)", yscale="log",
           title="Test-split PD distribution; dotted lines = provisional risk-band edges")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT))


def append_experiment_log(rows: list[dict[str, Any]]) -> None:
    path = PROJECT_ROOT / "docs" / "experiment_log.md"
    text = path.read_text(encoding="utf-8").rstrip("\n")
    lines = [
        f"| {r['date']} | {r['run_id']} | 5 | {r['dataset_version']} | 155 features "
        f"| {r['split']} | {r['model']} | {r['seed']} | none | {r['calibration']} "
        f"| {r['threshold_rule']} | ROC-AUC {r['roc_auc']:.4f}, PR-AUC {r['pr_auc']:.4f}, "
        f"Brier {r['brier']:.4f}, ECE(q) {r['ece_quantile']:.4f}{r['extra']} | {r['notes']} |"
        for r in rows
    ]
    path.write_text(text + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    seed = cfg["project"]["random_seed"]
    n_boot = cfg["modelling"]["bootstrap_resamples"]
    tcfg, ccfg, bands = cfg["thresholds"], cfg["calibration"], cfg["risk_bands"]
    run_id = f"phase5_{datetime.now():%Y%m%d_%H%M%S}"
    models_dir = project_path(cfg, "models")
    run_dir = models_dir / run_id
    tables, figures = project_path(cfg, "tables"), project_path(cfg, "figures")
    figures.mkdir(parents=True, exist_ok=True)

    pointer = json.loads((models_dir / "latest_phase4.json").read_text(encoding="utf-8"))
    p4_dir = models_dir / pointer["run_id"]
    names = {"champion": pointer["champion"], "challenger": pointer["challenger"],
             "baseline": BASELINE}
    log(f"Phase 4 run {pointer['run_id']}: {names}")

    data = load_modelling_data(cfg)
    df, feats = data["df"], data["features"]
    target = feats["target"]
    part = {s: df[df["split"] == s] for s in ["calibration", "validation", "test"]}
    X = {s: d.drop(columns=[target, "split"]) for s, d in part.items()}
    y = {s: d[target].astype(int).to_numpy() for s, d in part.items()}
    raw = {role: load_model(p4_dir / f"{name}.joblib")[0] for role, name in names.items()}

    # 1. Calibration methods for the champion, compared on validation
    log("Comparing calibration methods (fit: calibration split, compare: validation)")
    fitted, val_p, rows = {}, {}, []
    for method in ccfg["methods"]:
        if method not in METHODS:
            raise ValueError(f"unknown calibration method {method!r}")
        fitted[method] = fit_calibrated(raw["champion"], X["calibration"], y["calibration"], method)
        val_p[method] = predict_proba(fitted[method], X["validation"])
        rows.append({"method": method, **evaluate_probabilities(y["validation"], val_p[method])})
    cal_df = pd.DataFrame(rows)
    method = choose_method(cal_df, ccfg["selection_metric"])
    cal_df["selected"] = cal_df["method"] == method
    cal_df.to_csv(tables / "calibration_comparison_validation.csv", index=False)
    log(f"Selected calibration: {method}")

    final = {
        "champion": fitted[method],
        "challenger": fit_calibrated(raw["challenger"], X["calibration"], y["calibration"], method),
    }

    # 2. Threshold policies on validation (final champion probabilities)
    thr_df = threshold_options(y["validation"], val_p[method], tcfg)
    thr_df.to_csv(tables / "threshold_options_validation.csv", index=False)
    chosen = thr_df.set_index("policy").loc[tcfg["policy"]]
    threshold = float(chosen["threshold"])
    log(f"Operating threshold ({tcfg['policy']}, validation): {threshold:.4f}")

    # 3. ONE evaluation on the test split
    log("Final evaluation on the test split")
    def label(role: str, m: str) -> str:
        return f"{names[role]} ({'uncalibrated' if m == 'none' else m})"

    candidates = {label("baseline", "none"): raw["baseline"]}
    for role in ("challenger", "champion"):
        candidates[label(role, "none")] = raw[role]
        if method != "none":  # avoid reporting the same model twice
            candidates[label(role, method)] = final[role]
    test_p, test_rows = {}, []
    for model_name, model in candidates.items():
        p_val = predict_proba(model, X["validation"])
        t_model, _ = max_f1_threshold(y["validation"], p_val)  # each model's own validation choice
        p = predict_proba(model, X["test"])
        test_p[model_name] = p
        m = evaluate_probabilities(y["test"], p)
        roc_ci = bootstrap_ci(y["test"], p, roc_auc_score, n_boot, seed)
        pr_ci = bootstrap_ci(y["test"], p, average_precision_score, n_boot, seed)
        at = threshold_metrics(y["test"], p, t_model)
        test_rows.append({
            "model": model_name, **m,
            "roc_auc_ci_low": roc_ci[0], "roc_auc_ci_high": roc_ci[1],
            "pr_auc_ci_low": pr_ci[0], "pr_auc_ci_high": pr_ci[1],
            "threshold_from_validation": t_model, **{f"{k}_at_t": v for k, v in at.items()
                                                     if k != "threshold"},
        })
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(tables / "final_test_metrics.csv", index=False)

    champ_label = label("champion", method)
    pairs = [(champ_label, label("baseline", "none")), (champ_label, label("challenger", method))]
    if method != "none":
        pairs.append((champ_label, label("champion", "none")))
    diff_rows = []
    for a, b in pairs:
        for metric_name, fn in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score),
                                ("brier", brier_score_loss)]:
            r = paired_bootstrap_diff(y["test"], test_p[a], test_p[b], fn, n_boot, seed)
            diff_rows.append({"model_a": a, "model_b": b, "metric": metric_name, **r})
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(tables / "paired_differences_test.csv", index=False)

    op_test = threshold_metrics(y["test"], test_p[champ_label], threshold)
    band_df = band_distribution(test_p[champ_label], bands["edges"], bands["labels"], y["test"])
    band_df.to_csv(tables / "risk_band_distribution_test.csv", index=False)

    # 4. Figures
    fig_paths = [
        reliability_plot(
            {f"{names['champion']} {m}": val_p[m] for m in ccfg["methods"]}, y["validation"],
            "Champion calibration methods (validation, equal-count bins)",
            figures / "phase5_reliability_validation.png"),
        reliability_plot(
            {k: v for k, v in test_p.items()}, y["test"],
            "Final models on the test split (equal-count bins)",
            figures / "phase5_reliability_test.png"),
        histogram_plot(test_p[champ_label], bands["edges"], threshold,
                       figures / "phase5_pd_histogram_test.png"),
    ]

    # 5. Save final models and the pointer used by the rest of RiskLens
    common = {"run_id": run_id, "phase4_run": pointer["run_id"], "seed": seed,
              "calibration_method": method, "calibration_fit_on": "calibration split",
              "dataset_version": feats["dataset_version"], "numeric": feats["numeric"],
              "categorical": feats["categorical"],
              "created_at": datetime.now().isoformat(timespec="seconds")}
    for role in ("champion", "challenger"):
        save_model(final[role], {**common, "role": role, "experiment": names[role]},
                   run_dir, f"{role}_final")
    current = {
        "model_version": f"{names['champion']}-{method}-{run_id}",
        "model_file": f"{run_id}/champion_final.joblib",
        "challenger_file": f"{run_id}/challenger_final.joblib",
        "experiment": names["champion"],
        "calibration_method": method,
        "threshold": threshold,
        "threshold_policy": tcfg["policy"],
        "threshold_chosen_on": "validation split",
        "test_metrics": {k: test_df.set_index("model").loc[champ_label, k]
                         for k in ["roc_auc", "pr_auc", "brier", "ece_quantile"]},
    }
    (models_dir / "current_model.json").write_text(json.dumps(current, indent=2), "utf-8")

    # 6. Experiment log and summary
    today = datetime.now().strftime("%Y-%m-%d")
    log_rows = []
    for r in cal_df.itertuples(index=False):
        log_rows.append({
            "date": today, "run_id": run_id, "dataset_version": feats["dataset_version"],
            "split": "calibration → validation", "model": names["champion"], "seed": seed,
            "calibration": r.method, "threshold_rule": "n/a", "roc_auc": r.roc_auc,
            "pr_auc": r.pr_auc, "brier": r.brier, "ece_quantile": r.ece_quantile, "extra": "",
            "notes": "selected" if r.method == method else "",
        })
    for r in test_df.itertuples(index=False):
        log_rows.append({
            "date": today, "run_id": run_id, "dataset_version": feats["dataset_version"],
            "split": "TEST (final)", "model": r.model, "seed": seed,
            "calibration": r.model.split("(")[-1].rstrip(")").replace("uncalibrated", "none"),
            "threshold_rule": f"max-F1 on validation (t={r.threshold_from_validation:.3f})",
            "roc_auc": r.roc_auc, "pr_auc": r.pr_auc, "brier": r.brier,
            "ece_quantile": r.ece_quantile,
            "extra": f", precision {r.precision_at_t:.3f}, recall {r.recall_at_t:.3f}",
            "notes": "deployed champion" if r.model == champ_label else "",
        })
    append_experiment_log(log_rows)

    summary = {
        "run_id": run_id, "phase4_run": pointer["run_id"], "models": names,
        "calibration_selected": method, "calibration_comparison": cal_df.to_dict("records"),
        "threshold_policy": tcfg["policy"], "threshold": threshold,
        "threshold_options_validation": thr_df.to_dict("records"),
        "test_metrics": test_df.to_dict("records"),
        "paired_differences_test": diff_df.to_dict("records"),
        "operating_point_test": op_test, "risk_bands_test": band_df.to_dict("records"),
        "figures": fig_paths, "runtime_seconds": round(time.time() - start, 1),
    }
    out = project_path(cfg, "experiment_results")
    (out / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    log(f"Done in {summary['runtime_seconds']} s")
    return summary


def main() -> int:
    run(load_config())
    return 0
