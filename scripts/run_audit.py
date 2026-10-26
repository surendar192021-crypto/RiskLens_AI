"""Phase 2 dataset audit. Reads every raw table once and writes:

- reports/tables/audit_*.csv                 detailed tables
- reports/experiment_results/audit_summary.json  machine-readable summary
- docs/dataset_audit.md                      generated report (do not edit by hand)

Run from the project root:  python -m scripts.run_audit
Takes a few minutes; loads one table at a time to stay within laptop memory.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config, project_path
from src.data.load_data import (
    COLUMN_DESCRIPTION_FILE,
    TABLE_FILES,
    file_sha256,
    raw_path,
    read_column_descriptions,
    read_table,
)
from src.data.validate_data import (
    DAYS_PLACEHOLDER,
    categorical_summary,
    key_relationship,
    key_uniqueness,
    missingness_table,
    numeric_profile,
    target_distribution,
    temporal_summary,
)

# Primary key of each table (None = no single-column key expected).
PRIMARY_KEYS = {
    "application_train": "SK_ID_CURR",
    "application_test": "SK_ID_CURR",
    "bureau": "SK_ID_BUREAU",
    "bureau_balance": None,
    "previous_application": "SK_ID_PREV",
    "POS_CASH_balance": None,
    "installments_payments": None,
    "credit_card_balance": None,
    "sample_submission": "SK_ID_CURR",
}
# Categorical columns worth a value count in each side table.
SIDE_TABLE_CATEGORIES = {
    "bureau": ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"],
    "bureau_balance": ["STATUS"],
    "previous_application": ["NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE"],
    "POS_CASH_balance": ["NAME_CONTRACT_STATUS"],
    "credit_card_balance": ["NAME_CONTRACT_STATUS"],
}
AMOUNT_COLUMNS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]
GROUP_ATTRIBUTES = ["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def to_builtin(obj: Any) -> Any:
    """Make numpy/pandas values JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NA or (isinstance(obj, float) and np.isnan(obj)):
        return None
    return obj


def fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.3f}".rstrip("0").rstrip(".")
    return str(v).replace("|", "/")


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_none_\n"
    df = df.head(max_rows) if max_rows else df
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def value_shares(s: pd.Series) -> pd.DataFrame:
    counts = s.value_counts(dropna=False)
    counts = counts[counts > 0]  # categorical dtype also lists unused categories
    return pd.DataFrame(
        {"value": counts.index.astype(str), "count": counts.to_numpy(),
         "pct": counts.to_numpy() / len(s) * 100}
    )


def main() -> int:
    start = time.time()
    cfg = load_config()
    tables_dir = project_path(cfg, "tables")
    results_dir = project_path(cfg, "experiment_results")
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    id_col = cfg["data"]["id_column"]
    target_col = cfg["data"]["target_column"]

    summary: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds")}

    # ---- 1. Files and checksums -------------------------------------------------
    log("Checksumming files")
    files = []
    for name in list(TABLE_FILES) + ["column_descriptions"]:
        path = (
            project_path(cfg, "raw_data") / COLUMN_DESCRIPTION_FILE
            if name == "column_descriptions"
            else raw_path(name, cfg)
        )
        files.append(
            {"file": path.name, "present": path.exists(),
             "size_mb": path.stat().st_size / 1e6 if path.exists() else None,
             "sha256": file_sha256(path) if path.exists() else None}
        )
    files_df = pd.DataFrame(files)
    summary["files"] = files
    if not files_df["present"].all():
        print("Missing files:", list(files_df.loc[~files_df["present"], "file"]))
        return 1

    # ---- 2. Column descriptions --------------------------------------------------
    desc = read_column_descriptions(cfg)
    target_desc = desc.loc[desc["Row"] == target_col, "Description"]
    summary["target_definition"] = target_desc.iloc[0] if len(target_desc) else None

    overview, temporal_frames, relationships, key_checks = [], [], [], []
    side_counts: dict[str, dict[str, pd.DataFrame]] = {}
    keys: dict[str, pd.Series] = {}
    sections: dict[str, Any] = {}

    def record_overview(name: str, df: pd.DataFrame) -> None:
        miss = missingness_table(df)
        miss.to_csv(tables_dir / f"audit_missingness_{name}.csv", index=False)
        n_num = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
        overview.append(
            {"table": name, "rows": len(df), "columns": df.shape[1],
             "numeric_cols": n_num, "non_numeric_cols": df.shape[1] - n_num,
             "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
             "cols_with_missing": int((miss["n_missing"] > 0).sum()),
             "cols_over_50pct_missing": int((miss["pct_missing"] > 50).sum()),
             "pct_cells_missing": float(df.isna().to_numpy().mean() * 100)}
        )
        temporal_frames.append(temporal_summary(df, name))
        pk = PRIMARY_KEYS.get(name)
        if pk:
            key_checks.append({"table": name, **key_uniqueness(df, pk)})

    # ---- 3. application_train ----------------------------------------------------
    log("Auditing application_train")
    train = read_table("application_train", cfg)
    record_overview("application_train", train)
    summary["target"] = target_distribution(train[target_col])
    keys["train_ids"] = train[id_col].copy()
    train_cols = list(train.columns)

    dup_rows = int(train.drop(columns=[id_col]).duplicated().sum())
    placeholder = train["DAYS_EMPLOYED"] == DAYS_PLACEHOLDER
    emp = {
        "n_placeholder": int(placeholder.sum()),
        "pct_placeholder": float(placeholder.mean() * 100),
        "income_type_of_placeholder": value_shares(train.loc[placeholder, "NAME_INCOME_TYPE"])
        .head(5).to_dict("records"),
        "target_rate_placeholder": float(train.loc[placeholder, target_col].mean()),
        "target_rate_other": float(train.loc[~placeholder, target_col].mean()),
    }
    age_years = -train["DAYS_BIRTH"] / 365.25
    groups = {}
    for col in GROUP_ATTRIBUTES:
        g = train.groupby(col, observed=True, dropna=False)[target_col].agg(["size", "mean"])
        g = g.rename(columns={"size": "count", "mean": "target_rate"}).reset_index()
        g[col] = g[col].astype(str)
        groups[col] = g.sort_values("count", ascending=False)
    age_band = pd.cut(age_years, [0, 25, 35, 45, 55, 65, 120], right=False)
    g = train.groupby(age_band, observed=True)[target_col].agg(["size", "mean"]).reset_index()
    g.columns = ["AGE_BAND_YEARS", "count", "target_rate"]
    g["AGE_BAND_YEARS"] = g["AGE_BAND_YEARS"].astype(str)
    groups["AGE_BAND_YEARS"] = g

    cat_train = categorical_summary(train)
    cat_train.to_csv(tables_dir / "audit_categorical_application_train.csv", index=False)
    num_train = numeric_profile(train, AMOUNT_COLUMNS)
    num_train.to_csv(tables_dir / "audit_numeric_application_train.csv", index=False)
    ratio = train["AMT_CREDIT"] / train["AMT_INCOME_TOTAL"]
    sections["train"] = {
        "duplicate_rows_excluding_id": dup_rows,
        "days_employed": emp,
        "age_years_min": float(age_years.min()),
        "age_years_max": float(age_years.max()),
        "credit_to_income_median": float(ratio.median()),
        "credit_to_income_p99": float(ratio.quantile(0.99)),
        "n_ext_source_all_missing": int(
            train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().all(axis=1).sum()
        ),
    }
    del train
    gc.collect()

    # ---- 4. application_test -----------------------------------------------------
    log("Auditing application_test")
    test = read_table("application_test", cfg)
    record_overview("application_test", test)
    keys["test_ids"] = test[id_col].copy()
    sections["schema"] = {
        "in_train_not_test": [c for c in train_cols if c not in test.columns],
        "in_test_not_train": [c for c in test.columns if c not in train_cols],
        "train_test_id_overlap": int(keys["train_ids"].isin(keys["test_ids"]).sum()),
    }
    del test
    gc.collect()
    all_app_ids = pd.concat([keys["train_ids"], keys["test_ids"]], ignore_index=True)

    sub = read_table("sample_submission", cfg)
    record_overview("sample_submission", sub)
    sections["sample_submission_ids_match_test"] = bool(
        set(sub[id_col]) == set(keys["test_ids"])
    )
    del sub

    # ---- 5. Side tables ----------------------------------------------------------
    def train_coverage(child_ids: pd.Series) -> float:
        return float(keys["train_ids"].isin(child_ids.unique()).mean() * 100)

    coverage = {}
    side_tables = [
        "bureau", "bureau_balance", "previous_application",
        "POS_CASH_balance", "installments_payments", "credit_card_balance",
    ]
    for name in side_tables:
        log(f"Auditing {name}")
        df = read_table(name, cfg)
        record_overview(name, df)
        side_counts[name] = {
            c: value_shares(df[c]) for c in SIDE_TABLE_CATEGORIES.get(name, []) if c in df
        }
        if id_col in df:
            relationships.append(key_relationship(df[id_col], all_app_ids, name, "applications"))
            coverage[name] = train_coverage(df[id_col])
        if name == "bureau":
            keys["bureau_ids"] = df["SK_ID_BUREAU"].copy()
        if name == "bureau_balance":
            relationships.append(
                key_relationship(df["SK_ID_BUREAU"], keys["bureau_ids"], name, "bureau")
            )
            months = df.groupby("SK_ID_BUREAU").size()
            sections["bureau_balance_credits_with_2plus_months"] = int((months >= 2).sum())
            sections["bureau_balance_credits"] = len(months)
        if name == "previous_application":
            keys["prev_ids"] = df["SK_ID_PREV"].copy()
        if "SK_ID_PREV" in df and name != "previous_application":
            relationships.append(
                key_relationship(df["SK_ID_PREV"], keys["prev_ids"], name, "previous_application")
            )
        del df
        gc.collect()

    # ---- 6. Save tables and summary ----------------------------------------------
    overview_df = pd.DataFrame(overview)
    temporal_df = pd.concat(temporal_frames, ignore_index=True)
    rel_df = pd.DataFrame(relationships)
    keys_df = pd.DataFrame(key_checks)
    files_df.to_csv(tables_dir / "audit_files.csv", index=False)
    overview_df.to_csv(tables_dir / "audit_tables_overview.csv", index=False)
    temporal_df.to_csv(tables_dir / "audit_temporal.csv", index=False)
    rel_df.to_csv(tables_dir / "audit_relationships.csv", index=False)
    keys_df.to_csv(tables_dir / "audit_primary_keys.csv", index=False)
    for col, g in groups.items():
        g.to_csv(tables_dir / f"audit_group_{col}.csv", index=False)

    summary.update(
        {
            "tables": overview,
            "primary_keys": key_checks,
            "relationships": relationships,
            "train_coverage_pct": coverage,
            "temporal": temporal_df.to_dict("records"),
            "sections": sections,
        }
    )
    with (results_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_builtin(summary), f, indent=2)

    # ---- 7. Markdown report --------------------------------------------------------
    t = summary["target"]
    files_view = files_df.assign(sha256=files_df["sha256"].str[:12]).drop(columns="present")
    cov_df = pd.DataFrame(
        [{"table": k, "pct_train_applicants_with_records": v} for k, v in coverage.items()]
    )
    rel_view = rel_df[
        ["child", "parent", "key", "child_rows", "child_unique_keys", "orphan_child_keys",
         "pct_child_keys_in_parent", "pct_parent_with_child", "rows_per_key_median",
         "rows_per_key_max"]
    ]
    temporal_view = temporal_df.drop(columns="n_missing")
    miss_train = pd.read_csv(tables_dir / "audit_missingness_application_train.csv")
    emp_view = pd.DataFrame(emp["income_type_of_placeholder"])
    tr = sections["train"]
    bb_two_plus = sections["bureau_balance_credits_with_2plus_months"]

    md = [
        "# Dataset audit (auto-generated)\n",
        f"Generated {summary['generated_at']} by `python -m scripts.run_audit`. "
        "Every number below comes from running that script on the files in `data/raw/`. "
        "Do not edit by hand; interpretation lives in `docs/dataset_audit_findings.md`.\n",
        "## 1. Files\n", md_table(files_view),
        "## 2. Tables\n", md_table(overview_df),
        "## 3. Target\n",
        f"Official description of `{target_col}`: \"{summary['target_definition']}\"\n",
        f"- Rows: {t['n']:,}\n- TARGET=1: {t['n_positive']:,} ({t['positive_rate']:.2%})\n"
        f"- TARGET=0: {t['n_negative']:,}\n"
        f"- Negatives per positive: {t['negatives_per_positive']:.2f}\n",
        "## 4. Train vs test schema\n",
        f"- Columns in train but not test: {sections['schema']['in_train_not_test']}\n"
        f"- Columns in test but not train: {sections['schema']['in_test_not_train']}\n"
        f"- SK_ID_CURR values shared by train and test: "
        f"{sections['schema']['train_test_id_overlap']}\n"
        f"- sample_submission IDs identical to application_test IDs: "
        f"{fmt(sections['sample_submission_ids_match_test'])}\n",
        "## 5. Keys and relationships\n", "### Primary keys\n", md_table(keys_df),
        "### Foreign keys\n", md_table(rel_view),
        "### Coverage of training applicants\n", md_table(cov_df),
        f"\nbureau_balance: {sections['bureau_balance_credits']:,} bureau credits have monthly "
        f"records; {bb_two_plus:,} have 2 or more months.\n",
        "## 6. Missing values\n",
        "### application_train: 25 most-missing columns\n", md_table(miss_train, 25),
        "## 7. Temporal fields (days/months relative to the application, day 0)\n",
        md_table(temporal_view),
        "## 8. Data-quality checks on application_train\n",
        f"- Duplicate rows (ignoring SK_ID_CURR): {tr['duplicate_rows_excluding_id']}\n"
        f"- DAYS_EMPLOYED = {DAYS_PLACEHOLDER} placeholder: {emp['n_placeholder']:,} rows "
        f"({emp['pct_placeholder']:.2f}%). TARGET rate {emp['target_rate_placeholder']:.2%} "
        f"vs {emp['target_rate_other']:.2%} for other rows.\n"
        f"- Age range: {sections['train']['age_years_min']:.1f} to "
        f"{sections['train']['age_years_max']:.1f} years\n"
        f"- AMT_CREDIT / AMT_INCOME_TOTAL: median {tr['credit_to_income_median']:.2f}, "
        f"99th percentile {sections['train']['credit_to_income_p99']:.2f}\n"
        f"- Rows with all three EXT_SOURCE scores missing: "
        f"{sections['train']['n_ext_source_all_missing']:,}\n",
        "\nIncome type of DAYS_EMPLOYED placeholder rows:\n", md_table(emp_view),
        "### Numeric profile\n", md_table(num_train),
        "### Categorical columns\n", md_table(cat_train),
        "## 9. Candidate group attributes (counts and observed TARGET rate)\n",
        "These are descriptive base rates, not model results or fairness findings.\n",
    ]
    for col, g in groups.items():
        md += [f"### {col}\n", md_table(g)]
    md.append("## 10. Side-table categorical values\n")
    for name, counts in side_counts.items():
        for col, g in counts.items():
            md += [f"### {name}.{col}\n", md_table(g, 15)]
    md.append(f"\n_Audit runtime: {time.time() - start:.0f} s._\n")

    out = PROJECT_ROOT / "docs" / "dataset_audit.md"
    out.write_text("\n".join(md), encoding="utf-8")
    log(f"Done in {time.time() - start:.0f} s. Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
