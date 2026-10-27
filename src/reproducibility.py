"""Phase 12: end-to-end reproducibility and verification check.

Re-derives the project's key results from the raw data and saved configuration
and compares them with what was recorded. Nothing in data/, models/ or reports/
is overwritten except this check's own outputs:
- reports/tables/reproducibility_check.csv
- reports/experiment_results/phase12_reproducibility.json
and one `reproducibility_check` entry in the audit log.

Run from the project root:  python -m scripts.verify_reproducibility
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.data.build_dataset import build
from src.data.load_data import COLUMN_DESCRIPTION_FILE, TABLE_FILES, file_sha256, raw_path
from src.models.evaluate import evaluate_probabilities
from src.models.predict import load_model, predict_proba
from src.models.train import boosting_pipeline, load_modelling_data
from src.monitoring.audit_log import AuditLog
from src.monitoring.model_registry import ModelRegistry
from src.risk.expected_loss import expected_loss
from src.simulation.monte_carlo import loss_statistics, simulate_losses


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, area: str, check: str, recorded: Any, reproduced: Any, passed: bool,
            note: str = "") -> None:
        self.rows.append({"area": area, "check": check, "recorded": recorded,
                          "reproduced": reproduced, "passed": bool(passed), "note": note})
        log(f"{'PASS' if passed else 'FAIL'}  {area}: {check}")

    def close(self, area: str, check: str, recorded: float, reproduced: float,
              tol: float) -> None:
        self.add(area, check, recorded, reproduced, abs(recorded - reproduced) <= tol,
                 f"tolerance {tol:g}")


def compare_frames(new: pd.DataFrame, old: pd.DataFrame) -> tuple[bool, str]:
    """Strict equality first; if only column TYPES differ (e.g. text stored differently by
    another pandas version), compare values with text and categories as strings."""
    shape = f"{len(new):,} rows x {new.shape[1]} columns"
    try:
        pd.testing.assert_frame_equal(new, old)
        return True, f"{shape}: identical, including types"
    except AssertionError:
        pass
    if list(new.columns) != list(old.columns) or len(new) != len(old):
        return False, "different columns or row count"

    def as_values(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if not pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].astype("object").where(out[col].notna(), None).astype(str)
        return out

    try:
        pd.testing.assert_frame_equal(as_values(new), as_values(old), check_dtype=False)
        changed = [c for c in new.columns if str(new[c].dtype) != str(old[c].dtype)]
        return True, (f"{shape}: all values identical; storage type differs for "
                      f"{len(changed)} text columns (pandas version)")
    except AssertionError as exc:
        return False, str(exc)[:200]


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    c = Checks()
    models_dir = project_path(cfg, "models")
    processed, tables = project_path(cfg, "processed_data"), project_path(cfg, "tables")
    results = project_path(cfg, "experiment_results")

    # 1. Raw data unchanged since the audit
    audit = json.loads((results / "audit_summary.json").read_text(encoding="utf-8"))
    recorded = {f["file"]: f["sha256"] for f in audit["files"]}
    for name in list(TABLE_FILES) + ["column_descriptions"]:
        path = (project_path(cfg, "raw_data") / COLUMN_DESCRIPTION_FILE
                if name == "column_descriptions" else raw_path(name, cfg))
        sha = file_sha256(path)
        c.add("raw data", f"{path.name} SHA-256 matches audit", recorded[path.name][:12],
              sha[:12], sha == recorded[path.name])

    # 2. Rebuild the modelling table from raw data into a temporary folder
    log("Rebuilding modelling table from raw data (temporary folder)")
    with tempfile.TemporaryDirectory() as tmp:
        build(cfg, out_dir=Path(tmp), side_outputs=False)
        for f in ["modelling_train.parquet", "portfolio_test.parquet", "splits.parquet"]:
            new, old = pd.read_parquet(Path(tmp) / f), pd.read_parquet(processed / f)
            same, note = compare_frames(new, old)
            c.add("dataset", f"rebuilt {f} matches saved", "saved", "rebuilt", same, note)
        same_feats = (json.loads((Path(tmp) / "feature_list.json").read_text())
                      == json.loads((processed / "feature_list.json").read_text()))
        c.add("dataset", "feature list identical", "saved", "rebuilt", same_feats)

    # 3. Deployed model: file integrity and recorded test metrics
    current = json.loads((models_dir / "current_model.json").read_text(encoding="utf-8"))
    registry = ModelRegistry(models_dir / "registry.json", models_dir)
    c.add("model", "deployed model file matches registry SHA-256", "registered", "on disk",
          registry.verify(current["model_version"]))
    model, _ = load_model(models_dir / current["model_file"])
    data = load_modelling_data(cfg)
    df, feats = data["df"], data["features"]
    target = feats["target"]
    test = df[df["split"] == "test"]
    p_test = predict_proba(model, test.drop(columns=[target, "split"]))
    m = evaluate_probabilities(test[target], p_test)
    final = pd.read_csv(tables / "final_test_metrics.csv").set_index("model")
    label = f"{current['experiment']} (uncalibrated)" if current["calibration_method"] == "none" \
        else f"{current['experiment']} ({current['calibration_method']})"
    for k in ["roc_auc", "pr_auc", "brier", "ece_quantile"]:
        c.close("model", f"test {k} of deployed model", float(final.loc[label, k]), m[k], 1e-12)

    # 4. Retrain the champion from scratch with the recorded settings
    log("Retraining the champion from scratch")
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"]
    retrained, best = boosting_pipeline(
        "lightgbm", feats["numeric"], feats["categorical"], cfg["modelling"]["params"]["lightgbm"],
        1.0, cfg["project"]["random_seed"], cfg["modelling"]["early_stopping_rounds"],
        tr.drop(columns=[target, "split"]), tr[target], va.drop(columns=[target, "split"]),
        va[target])
    p4 = json.loads((models_dir / "latest_phase4.json").read_text(encoding="utf-8"))
    p4_meta = json.loads((models_dir / p4["run_id"] / f"{p4['champion']}.json").read_text())
    rec_best = int(p4_meta["validation_metrics"]["best_iteration"])
    c.add("retraining", "same number of trees (early stopping)", rec_best, best, rec_best == best)
    p_re = predict_proba(retrained, test.drop(columns=[target, "split"]))
    diff = float(np.abs(p_re - p_test).max())
    c.add("retraining", "retrained test predictions equal deployed (max abs diff)", 0.0, diff,
          diff <= 1e-9, "tolerance 1e-9")

    # 5. Portfolio aggregates and expected loss
    port = pd.read_parquet(processed / "portfolio_test.parquet")
    p_port = predict_proba(model, port)
    ead = port[cfg["portfolio"]["exposure_column"]].to_numpy(dtype="float64")
    lgd = float(cfg["expected_loss"]["lgd"])
    ph8 = json.loads((results / "phase8_portfolio_summary.json").read_text())["summary"]
    c.close("portfolio", "expected difficulties (sum of PD)", ph8["expected_difficulties"],
            float(p_port.sum()), 1e-6)
    c.close("portfolio", "portfolio expected loss", ph8["portfolio_el"],
            float(expected_loss(p_port, lgd, ead).sum()), 1e-3)

    # 6. Monte Carlo baseline with the recorded seed
    log("Re-running baseline Monte Carlo")
    mc = cfg["monte_carlo"]
    sim = simulate_losses(p_port, ead, lgd, mc["rho"], mc["n_simulations"], mc["seed"],
                          mc["chunk_size"])
    stats = loss_statistics(sim["losses"], float((p_port * lgd * ead).sum()),
                            mc["confidence_levels"])
    ph9 = json.loads((results / "phase9_simulation_summary.json").read_text())["baseline"]
    for k in ["mean_loss", "var_99", "var_99.9", "es_99"]:
        c.close("simulation", f"Monte Carlo {k}", ph9[k], stats[k], 1e-3)

    # 7. Audit trail
    chain = AuditLog(models_dir / "audit_log.jsonl").verify()
    c.add("audit", "audit-log hash chain valid", True, chain["valid"], chain["valid"])

    out = pd.DataFrame(c.rows)
    out.to_csv(tables / "reproducibility_check.csv", index=False)
    summary = {"checked_at": datetime.now().isoformat(timespec="seconds"),
               "checks": len(out), "passed": int(out["passed"].sum()),
               "failed": out.loc[~out["passed"], ["area", "check"]].to_dict("records"),
               "runtime_seconds": round(time.time() - start, 1)}
    (results / "phase12_reproducibility.json").write_text(json.dumps(summary, indent=2,
                                                                     default=str), "utf-8")
    AuditLog(models_dir / "audit_log.jsonl").log(
        "reproducibility_check", {"checks": summary["checks"], "passed": summary["passed"]})
    log(f"{summary['passed']}/{summary['checks']} checks passed in {summary['runtime_seconds']} s")
    return summary


def main() -> int:
    s = run(load_config())
    return 0 if s["passed"] == s["checks"] else 1
