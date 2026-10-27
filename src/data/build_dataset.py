"""Build the modelling table: one row per applicant, application features plus
aggregated history, and the reproducible split assignment.

Outputs (data/processed/, git-ignored):
- modelling_train.parquet   labelled applicants (application_train) with TARGET
- portfolio_test.parquet    unlabelled applicants (application_test), no TARGET
- splits.parquet            SK_ID_CURR -> train / validation / calibration / test
- feature_list.json         numeric and categorical feature names, excluded columns
Also: reports/tables/feature_summary.csv, docs/data_dictionary.md,
reports/experiment_results/build_summary.json.

Run from the project root:  python -m scripts.build_dataset
"""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import PROJECT_ROOT, load_config, project_path
from src.data.feature_engineering import (
    FEATURE_DESCRIPTIONS,
    ID,
    add_application_features,
    aggregate_bureau,
    aggregate_credit_card,
    aggregate_installments,
    aggregate_pos,
    aggregate_previous,
    join_aggregate,
    redundant_building_columns,
)
from src.data.load_data import optimise_dtypes, read_column_descriptions, read_table
from src.data.preprocess import clean_application
from src.data.splits import make_splits
from src.data.validate_data import require_columns

DATASET_VERSION = "homecredit_v1"
PREFIX_SOURCES = {
    "BUREAU_": "bureau + bureau_balance",
    "PREV_": "previous_application",
    "INST_": "installments_payments",
    "POS_": "POS_CASH_balance",
    "CC_": "credit_card_balance",
    "HAS_": "indicator",
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def feature_columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Numeric and categorical model inputs. ID and TARGET are never features."""
    excluded = {ID, target}
    feats = [c for c in df.columns if c not in excluded]
    numeric = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feats if c not in numeric]
    return numeric, categorical


def feature_source(name: str) -> str:
    for prefix, source in PREFIX_SOURCES.items():
        if name.startswith(prefix):
            return source
    return "application (engineered)" if name in FEATURE_DESCRIPTIONS else "application (raw)"


def univariate_auc(x: pd.Series, y: pd.Series) -> float | None:
    """ROC-AUC of one numeric feature on its non-missing rows (0.5 = no signal)."""
    mask = x.notna()
    if mask.sum() < 100 or y[mask].nunique() < 2 or x[mask].nunique() < 2:
        return None
    return float(roc_auc_score(y[mask], x[mask]))


def build(cfg: dict[str, Any], out_dir: Path | None = None,
          side_outputs: bool = True) -> dict[str, Any]:
    """Build the modelling table.

    out_dir: where the parquet files and feature list go (default data/processed).
    side_outputs: also write the feature summary, data dictionary and build summary.
    The reproducibility check uses a temporary out_dir with side_outputs=False, so it
    never overwrites the project's files.
    """
    start = time.time()
    target = cfg["data"]["target_column"]
    seed = cfg["project"]["random_seed"]
    summary: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
    }

    # 1. Applications (train + test together: cleaning and features use no target)
    log("Loading applications")
    train = read_table("application_train", cfg)
    test = read_table("application_test", cfg)
    require_columns(train, [ID, target], "application_train")
    n_train, n_test = len(train), len(test)
    apps = pd.concat([train, test], ignore_index=True)
    is_train = pd.Series([True] * n_train + [False] * n_test)
    del train, test
    apps = optimise_dtypes(apps)  # re-align categories across train and test

    # 2. Split assignment (labelled rows only)
    splits = make_splits(
        apps.loc[is_train, ID], apps.loc[is_train, target].astype(int), cfg["splits"], seed
    )
    train_ids = set(splits.loc[splits["split"] == "train", ID])
    summary["splits"] = {
        name: {"rows": int((splits["split"] == name).sum()),
               "positive_rate": float(
                   apps.loc[is_train, target].to_numpy()[(splits["split"] == name).to_numpy()]
                   .mean())}
        for name in ["train", "validation", "calibration", "test"]
    }

    # 3. Cleaning and application features
    log("Cleaning and application features")
    apps = clean_application(apps)
    apps = add_application_features(apps)
    dropped = redundant_building_columns(list(apps.columns))
    on_train = apps[ID].isin(train_ids)
    corr = {
        c: float(apps.loc[on_train, c].corr(apps.loc[on_train, f"{c[:-5]}_AVG"])) for c in dropped
    }
    summary["dropped_building_columns"] = {
        "n": len(dropped),
        "min_corr_with_avg_on_train": min(corr.values()) if corr else None,
        "columns": dropped,
    }
    apps = apps.drop(columns=dropped)

    # 4. History tables, one at a time to limit memory
    steps = [
        ("HAS_BUREAU", lambda: aggregate_bureau(
            read_table("bureau", cfg), read_table("bureau_balance", cfg))),
        ("HAS_PREV", lambda: aggregate_previous(read_table("previous_application", cfg))),
        ("HAS_INST", lambda: aggregate_installments(read_table("installments_payments", cfg))),
        ("HAS_POS", lambda: aggregate_pos(read_table("POS_CASH_balance", cfg))),
        ("HAS_CC", lambda: aggregate_credit_card(read_table("credit_card_balance", cfg))),
    ]
    joins = []
    for flag, make in steps:
        log(f"Aggregating for {flag}")
        agg = make()
        before = len(apps)
        apps = join_aggregate(apps, agg, flag)
        joins.append({"source": flag, "aggregate_rows": len(agg), "features": agg.shape[1],
                      "rows_before": before, "rows_after": len(apps),
                      "pct_train_with_records": float(apps.loc[is_train, flag].mean() * 100)})
        del agg
        gc.collect()
    summary["joins"] = joins

    # 5. Final checks
    if len(apps) != n_train + n_test or not apps[ID].is_unique:
        raise AssertionError("row count or ID uniqueness changed during the build")
    numeric, categorical = feature_columns(apps, target)
    if ID in numeric + categorical or target in numeric + categorical:
        raise AssertionError("ID or TARGET leaked into the feature list")
    inf_cols = [c for c in numeric if np.isinf(apps[c].to_numpy(dtype="float64")).any()]
    if inf_cols:
        raise AssertionError(f"infinite values in {inf_cols}")
    summary["rows"] = {"train": n_train, "test": n_test}
    summary["n_features"] = {"numeric": len(numeric), "categorical": len(categorical)}

    # 6. Save
    log("Saving")
    out_dir = Path(out_dir) if out_dir is not None else project_path(cfg, "processed_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    modelling = apps.loc[is_train].reset_index(drop=True)
    modelling[target] = modelling[target].astype("int8")
    portfolio = apps.loc[~is_train].drop(columns=[target]).reset_index(drop=True)
    modelling.to_parquet(out_dir / "modelling_train.parquet", index=False)
    portfolio.to_parquet(out_dir / "portfolio_test.parquet", index=False)
    splits.to_parquet(out_dir / "splits.parquet", index=False)
    with (out_dir / "feature_list.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset_version": DATASET_VERSION, "id": ID, "target": target,
                   "numeric": numeric, "categorical": categorical,
                   "excluded": [ID, target]}, f, indent=2)

    if not side_outputs:
        summary["runtime_seconds"] = round(time.time() - start, 1)
        log(f"Done in {summary['runtime_seconds']} s (no side outputs)")
        return summary

    # 7. Feature summary on the TRAINING split only (validation/test untouched)
    log("Feature summary (training split only)")
    tr = modelling[modelling[ID].isin(train_ids)]
    y = tr[target]
    desc = read_column_descriptions(cfg)
    app_desc = (
        desc[desc["Table"].str.startswith("application")].drop_duplicates("Row")
        .set_index("Row")["Description"]
    )
    rows = []
    for col in numeric + categorical:
        auc = univariate_auc(tr[col], y) if col in numeric else None
        rows.append({
            "feature": col,
            "source": feature_source(col),
            "type": "numeric" if col in numeric else "categorical",
            "pct_missing_train": float(tr[col].isna().mean() * 100),
            "univariate_auc_train": auc,
            "abs_auc_signal": abs(auc - 0.5) if auc is not None else None,
            "description": FEATURE_DESCRIPTIONS.get(col, app_desc.get(col, "")),
        })
    feat = pd.DataFrame(rows)
    feat.to_csv(project_path(cfg, "tables") / "feature_summary.csv", index=False)
    summary["top_univariate_features"] = (
        feat.dropna(subset=["abs_auc_signal"]).nlargest(15, "abs_auc_signal")
        [["feature", "univariate_auc_train"]].to_dict("records")
    )
    write_data_dictionary(feat, summary)

    summary["runtime_seconds"] = round(time.time() - start, 1)
    results = project_path(cfg, "experiment_results")
    results.mkdir(parents=True, exist_ok=True)
    with (results / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Done in {summary['runtime_seconds']} s")
    return summary


def write_data_dictionary(feat: pd.DataFrame, summary: dict[str, Any]) -> None:
    def cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v).replace("|", "/")

    lines = [
        "# Data dictionary (auto-generated)\n",
        f"Generated {summary['built_at']} by `python -m scripts.build_dataset` "
        f"(dataset version `{summary['dataset_version']}`). Do not edit by hand.\n",
        "- **pct_missing_train** and **univariate_auc_train** are computed on the training split "
        "only. Validation, calibration and test rows are not used.\n"
        "- Univariate AUC is the ROC-AUC of the single feature on its non-missing rows. 0.5 means "
        "no ranking signal on its own; below 0.5 means higher values go with *lower* TARGET rate. "
        "It is a screening statistic, not a model result.\n"
        "- `*_COUNT` history features are 0 when the applicant has no records. Other history "
        "features stay missing (see the `HAS_*` flags).\n",
        "| feature | source | type | pct_missing_train | univariate_auc_train | description |",
        "|---|---|---|---|---|---|",
    ]
    for r in feat.itertuples(index=False):
        lines.append(
            f"| {r.feature} | {r.source} | {r.type} | {cell(r.pct_missing_train)} | "
            f"{cell(r.univariate_auc_train)} | {cell(r.description)} |"
        )
    (PROJECT_ROOT / "docs" / "data_dictionary.md").write_text("\n".join(lines) + "\n", "utf-8")


def main() -> int:
    build(load_config())
    return 0
