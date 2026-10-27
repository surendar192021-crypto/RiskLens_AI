import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.risk.early_warning import (
    AlertStore,
    compare_snapshots,
    evaluate_flags,
    final_level,
    generate_alerts,
    pd_level,
    resolve_levels,
    top_drivers_text,
)
from src.simulation.scenarios import scale_income

EW = load_config()["early_warning"]
T = resolve_levels(EW["levels"], 0.16)


def test_resolve_levels_uses_operating_threshold_and_checks_order():
    assert T == {"yellow": 0.105, "orange": 0.16, "red": 0.40}
    with pytest.raises(ValueError, match="increase"):
        resolve_levels({"yellow": 0.2, "orange": "operating_threshold", "red": 0.4}, 0.16)


@pytest.mark.parametrize(
    "p, level", [(0.0, 0), (0.1049, 0), (0.105, 1), (0.1599, 1), (0.16, 2), (0.3999, 2),
                 (0.40, 3), (1.0, 3)],
)
def test_pd_level_boundaries(p, level):
    assert pd_level([p], T)[0] == level


def test_flags_ignore_missing_values():
    df = pd.DataFrame({f["feature"]: [np.nan, 1e9] for f in EW["flags"].values()})
    flags = evaluate_flags(df, EW["flags"])
    assert not flags.iloc[0].any() and flags.iloc[1].all()


def test_flag_with_unknown_feature_fails():
    with pytest.raises(KeyError, match="not in data"):
        evaluate_flags(pd.DataFrame({"x": [1]}), EW["flags"])


def test_escalation_one_step_never_to_red():
    pd_idx = np.array([0, 1, 2, 3, 0])
    flag = np.array([True, True, True, True, False])
    assert list(final_level(pd_idx, flag, True)) == [1, 2, 2, 3, 0]
    assert list(final_level(pd_idx, flag, False)) == [0, 1, 2, 3, 0]


def scored_frame():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4],
        "probability": [0.05, 0.12, 0.30, 0.55],
        "risk_band": ["Very Low", "Very Low", "Low", "Moderate"],
    })


def flag_frame(first_flagged=True):
    f = pd.DataFrame(False, index=range(4), columns=list(EW["flags"]))
    f.iloc[0, 0] = first_flagged
    return f


EW_ESC = {**EW, "flag_escalation": True}


def test_config_default_does_not_escalate():
    a = generate_alerts(scored_frame(), flag_frame(), T, EW, "v1", "t")
    assert EW["flag_escalation"] is False
    assert 1 not in set(a["SK_ID_CURR"])  # flagged but green -> no alert


def test_generate_alerts_levels_triggers_and_actions():
    a = generate_alerts(scored_frame(), flag_frame(), T, EW_ESC, "v1", "2026-10-27",
                        drivers=pd.Series(["d1", "d2", "d3", "d4"]))
    by_id = a.set_index("SK_ID_CURR")
    assert set(by_id.index) == {1, 2, 3, 4}  # borrower 1 is green by PD but flagged
    assert by_id.loc[1, "level"] == "yellow" and "raised one step" in by_id.loc[1, "triggers"]
    assert by_id.loc[2, "level"] == "yellow" and by_id.loc[3, "level"] == "orange"
    assert by_id.loc[4, "level"] == "red"
    assert by_id.loc[4, "recommended_action"] == EW["actions"]["red"]
    assert by_id.loc[4, "main_drivers"] == "d4"
    assert list(a["severity"]) == sorted(a["severity"], reverse=True)
    assert a["alert_key"].is_unique


def test_green_unflagged_borrowers_get_no_alert():
    a = generate_alerts(scored_frame(), flag_frame(False), T, EW_ESC, "v1", "t")
    assert 1 not in set(a["SK_ID_CURR"])


def test_alert_store_deduplicates_and_closes(tmp_path):
    store = AlertStore(tmp_path / "alerts.parquet")
    a1 = generate_alerts(scored_frame(), flag_frame(), T, EW_ESC, "v1", "t1")
    r1 = store.upsert(a1, "run1")
    assert r1 == {"new": 4, "unchanged": 0, "closed": 0, "active_total": 4}
    r2 = store.upsert(a1, "run2")  # identical run -> no duplicates
    assert r2 == {"new": 0, "unchanged": 4, "closed": 0, "active_total": 4}
    a3 = generate_alerts(scored_frame(), flag_frame(False), T, EW_ESC, "v1", "t3")
    r3 = store.upsert(a3, "run3")  # borrower 1 no longer flagged -> closed
    assert r3["closed"] == 1 and r3["new"] == 0 and r3["active_total"] == 3
    hist = store.load()
    assert len(hist) == 4 and (hist["status"] == "closed").sum() == 1


def test_compare_snapshots_reports_only_worsening():
    before = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "level": ["green", "orange", "yellow"],
                           "probability": [0.05, 0.2, 0.11]})
    after = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "level": ["yellow", "yellow", "yellow"],
                          "probability": [0.11, 0.12, 0.11]})
    out = compare_snapshots(before, after)
    assert list(out["SK_ID_CURR"]) == [1] and out.loc[0, "change"] == "green -> yellow"


def test_top_drivers_only_positive():
    v = np.array([[0.5, -1.0, 0.2], [-0.1, -0.2, -0.3]])
    assert top_drivers_text(v, ["a", "b", "c"], 2) == ["a (+0.50), c (+0.20)", ""]


def test_scale_income_recomputes_ratios():
    df = pd.DataFrame({"AMT_INCOME_TOTAL": [100.0], "AMT_CREDIT": [300.0], "AMT_ANNUITY": [30.0],
                       "AMT_GOODS_PRICE": [270.0], "CNT_FAM_MEMBERS": [2.0],
                       "INCOME_PER_FAMILY_MEMBER": [50.0]})
    out = scale_income(df, 0.5)
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(6.0)
    assert out.loc[0, "ANNUITY_INCOME_RATIO"] == pytest.approx(0.6)
    assert out.loc[0, "INCOME_PER_FAMILY_MEMBER"] == pytest.approx(25.0)
    assert df.loc[0, "AMT_INCOME_TOTAL"] == 100.0
    with pytest.raises(ValueError):
        scale_income(df, 0)
