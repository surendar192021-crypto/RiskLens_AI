import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.monitoring.audit_log import AuditLog
from src.monitoring.drift import (
    classify_psi,
    feature_drift,
    psi_categorical,
    psi_numeric,
    review_recommendation,
    schema_check,
)
from src.monitoring.fairness import difference_vs_reference, group_metrics, parity_ratios
from src.monitoring.model_registry import ModelRegistry

TH = {"moderate": 0.10, "major": 0.25}
RNG = np.random.default_rng(0)


# --- drift ------------------------------------------------------------------------

def test_psi_zero_for_same_distribution_and_small_for_resample():
    x = pd.Series(RNG.normal(size=50_000))
    assert psi_numeric(x, x)[0] == pytest.approx(0.0, abs=1e-12)
    assert psi_numeric(x, pd.Series(RNG.normal(size=50_000)))[0] < 0.01


def test_psi_detects_shift_and_matches_hand_calculation():
    ref = pd.Series(RNG.normal(size=50_000))
    assert psi_numeric(ref, ref + 1.0)[0] > 0.25
    # categorical by hand: 50/50 -> 80/20
    v, _ = psi_categorical(pd.Series(["a", "b"] * 50), pd.Series(["a"] * 80 + ["b"] * 20))
    assert v == pytest.approx((0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5))


def test_psi_counts_missingness_and_new_categories():
    ref = pd.Series([1.0, 2.0, 3.0, 4.0] * 250)
    new = pd.Series([1.0, 2.0, np.nan, np.nan] * 250)
    v, table = psi_numeric(ref, new, n_bins=4)
    assert table.loc[table["bin"] == "missing", "new_share"].iloc[0] == pytest.approx(0.5)
    assert v > 0.25
    v2, t2 = psi_categorical(pd.Series(["a"] * 100), pd.Series(["a"] * 90 + ["new"] * 10))
    assert "new" in set(t2["bin"]) and v2 > 0


@pytest.mark.parametrize("v, level", [(0.0, "stable"), (0.0999, "stable"), (0.1, "moderate"),
                                      (0.2499, "moderate"), (0.25, "major"), (3.0, "major")])
def test_classify_psi(v, level):
    assert classify_psi(v, TH) == level


def test_feature_drift_table_and_review_rule():
    ref = pd.DataFrame({"x": RNG.normal(size=5000), "y": RNG.normal(size=5000),
                        "c": RNG.choice(["a", "b"], 5000)})
    new = pd.DataFrame({"x": RNG.normal(size=5000) + 2, "y": RNG.normal(size=5000),
                        "c": RNG.choice(["a", "b"], 5000, p=[0.95, 0.05])})
    d = feature_drift(ref, new, ["x", "y"], ["c"], 10, TH).set_index("feature")
    assert d.loc["x", "level"] == "major" and d.loc["y", "level"] == "stable"
    assert d.loc["c", "level"] == "major" and d.loc["x", "ks_statistic"] > 0.5
    rule = {"prediction_psi": 0.10, "major_drift_important_features": 2}
    r = review_recommendation(d.reset_index(), 0.02, ["x", "c"], rule)
    assert r["review_recommended"] and "retrain" in r["action"]
    assert not review_recommendation(d.reset_index(), 0.02, ["y"], rule)["review_recommended"]
    assert review_recommendation(d.reset_index(), 0.2, ["y"], rule)["review_recommended"]


def test_schema_check():
    ref = pd.DataFrame({"a": [1.0], "b": ["x"]})
    assert schema_check(ref, ref, ["a", "b"])["ok"]
    bad = schema_check(ref, pd.DataFrame({"a": ["text"]}), ["a", "b"])
    assert bad["missing_features"] == ["b"] and "a" in bad["type_changes"] and not bad["ok"]


# --- fairness ---------------------------------------------------------------------

def test_group_metrics_by_hand():
    y = [1, 1, 0, 0, 1, 0, 0, 0]
    p = [0.9, 0.1, 0.8, 0.1, 0.7, 0.6, 0.2, 0.1]
    g = ["A", "A", "A", "A", "B", "B", "B", None]
    m = group_metrics(y, p, g, threshold=0.5, min_size=4).set_index("group")
    a = m.loc["A"]
    assert (a["tp"], a["fn"], a["fp"], a["tn"]) == (1, 1, 1, 1)
    assert a["tpr"] == 0.5 and a["fpr"] == 0.5 and a["flag_rate"] == 0.5
    assert a["calibration_gap"] == pytest.approx(np.mean([0.9, 0.1, 0.8, 0.1]) - 0.5)
    assert m.loc["B", "tpr"] == 1.0 and not m.loc["B", "reliable"]
    assert m.loc["Missing", "n"] == 1
    assert a["tpr_ci_low"] < 0.5 < a["tpr_ci_high"]


def test_differences_and_parity():
    rng = np.random.default_rng(1)
    n = 20_000
    g = rng.choice(["F", "M"], n)
    y = (rng.random(n) < np.where(g == "M", 0.10, 0.07)).astype(int)
    p = np.clip(np.where(y == 1, 0.3, 0.05) + rng.normal(0, 0.05, n), 0, 1)
    m = group_metrics(y, p, g, 0.16)
    d = difference_vs_reference(m, "F").set_index("metric")
    assert d.loc["base_rate", "ci_low"] > 0  # M really has a higher base rate here
    assert d.loc["tpr", "ci_low"] < 0 < d.loc["tpr", "ci_high"]  # same TPR by construction
    r = parity_ratios(m)
    assert 0 < r["flag_rate"] <= 1 and r["tpr"] > 0.9
    with pytest.raises(KeyError):
        difference_vs_reference(m, "X")


# --- registry and audit log -------------------------------------------------------

def test_registry_hash_deploy_and_verify(tmp_path):
    joblib.dump({"m": 1}, tmp_path / "a.joblib")
    joblib.dump({"m": 2}, tmp_path / "b.joblib")
    reg = ModelRegistry(tmp_path / "registry.json", tmp_path)
    reg.register({"model_version": "a", "model_file": "a.joblib", "status": "deployed",
                  "created_at": "t"})
    reg.register({"model_version": "b", "model_file": "b.joblib", "status": "deployed",
                  "created_at": "t"})
    assert reg.deployed()["model_version"] == "b"
    assert {e["model_version"]: e["status"] for e in reg.load()}["a"] == "archived"
    assert reg.verify("b")
    joblib.dump({"m": "tampered"}, tmp_path / "b.joblib")
    assert not reg.verify("b")
    with pytest.raises(ValueError):
        reg.register({"model_version": "c", "model_file": "a.joblib", "status": "live",
                      "created_at": "t"})
    with pytest.raises(FileNotFoundError):
        reg.register({"model_version": "d", "model_file": "nope.joblib", "status": "experiment",
                      "created_at": "t"})


def test_audit_log_chain_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(3):
        log.log("event", {"i": i})
    assert log.verify() == {"valid": True, "entries": 3}
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    e = json.loads(lines[1])
    e["details"]["i"] = 99
    lines[1] = json.dumps(e)
    (tmp_path / "audit.jsonl").write_text("\n".join(lines) + "\n")
    assert log.verify() == {"valid": False, "broken_at_seq": 2}
    with pytest.raises(ValueError):
        log.log("", {})
