import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    add_application_features,
    aggregate_bureau,
    aggregate_credit_card,
    aggregate_installments,
    aggregate_pos,
    aggregate_previous,
    bureau_balance_per_credit,
    join_aggregate,
    redundant_building_columns,
    safe_divide,
)
from src.data.preprocess import clean_application
from src.data.splits import make_splits
from src.data.validate_data import DAYS_PLACEHOLDER


def app_rows(**overrides):
    base = {
        "SK_ID_CURR": [1, 2],
        "AMT_CREDIT": [100_000.0, 200_000.0],
        "AMT_INCOME_TOTAL": [50_000.0, 0.0],
        "AMT_ANNUITY": [10_000.0, np.nan],
        "AMT_GOODS_PRICE": [90_000.0, 200_000.0],
        "DAYS_BIRTH": [-36525, -18262],
        "DAYS_EMPLOYED": [-3652, DAYS_PLACEHOLDER],
        "CNT_FAM_MEMBERS": [2.0, 0.0],
        "EXT_SOURCE_1": [0.2, np.nan],
        "EXT_SOURCE_2": [0.4, np.nan],
        "EXT_SOURCE_3": [np.nan, np.nan],
        "FLAG_DOCUMENT_3": [1, 0],
        "FLAG_DOCUMENT_5": [1, 1],
        "CODE_GENDER": pd.Categorical(["F", "XNA"]),
        "NAME_FAMILY_STATUS": ["Married", "Unknown"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_safe_divide_handles_zero_and_missing():
    out = safe_divide(pd.Series([1.0, 1.0, np.nan, 4.0]), pd.Series([2.0, 0.0, 1.0, np.nan]))
    assert out.iloc[0] == 0.5
    assert out.iloc[1:].isna().all()
    assert not np.isinf(out).any()


def test_clean_application_placeholder_and_codes():
    raw = app_rows()
    out = clean_application(raw)
    assert list(out["DAYS_EMPLOYED_PLACEHOLDER"]) == [0, 1]
    assert out.loc[0, "DAYS_EMPLOYED"] == -3652 and np.isnan(out.loc[1, "DAYS_EMPLOYED"])
    assert pd.isna(out.loc[1, "CODE_GENDER"]) and "XNA" not in out["CODE_GENDER"].cat.categories
    assert pd.isna(out.loc[1, "NAME_FAMILY_STATUS"])
    assert raw.loc[1, "DAYS_EMPLOYED"] == DAYS_PLACEHOLDER  # input not modified


def test_application_features_values():
    out = add_application_features(clean_application(app_rows()))
    r0, r1 = out.iloc[0], out.iloc[1]
    assert r0["CREDIT_INCOME_RATIO"] == pytest.approx(2.0)
    assert r0["ANNUITY_INCOME_RATIO"] == pytest.approx(0.2)
    assert r0["CREDIT_ANNUITY_RATIO"] == pytest.approx(10.0)
    assert r0["AGE_YEARS"] == pytest.approx(100.0)
    assert r0["EMPLOYED_YEARS"] == pytest.approx(3652 / 365.25)
    assert r0["EXT_SOURCE_MEAN"] == pytest.approx(0.3) and r0["EXT_SOURCE_N_MISSING"] == 1
    assert r0["DOCUMENT_COUNT"] == 2
    # zero income / zero family / missing annuity -> missing, not inf
    assert np.isnan(r1["CREDIT_INCOME_RATIO"]) and np.isnan(r1["INCOME_PER_FAMILY_MEMBER"])
    assert np.isnan(r1["CREDIT_ANNUITY_RATIO"]) and np.isnan(r1["EMPLOYED_YEARS"])
    assert r1["EXT_SOURCE_N_MISSING"] == 3 and np.isnan(r1["EXT_SOURCE_MEAN"])
    assert "DAYS_BIRTH" not in out and "DAYS_EMPLOYED" not in out


def test_application_features_require_cleaning_first():
    with pytest.raises(ValueError, match="clean_application"):
        add_application_features(app_rows())


def test_redundant_building_columns():
    cols = ["APARTMENTS_AVG", "APARTMENTS_MEDI", "APARTMENTS_MODE", "TOTALAREA_MODE",
            "HOUSETYPE_MODE"]
    assert redundant_building_columns(cols) == ["APARTMENTS_MEDI", "APARTMENTS_MODE"]


def test_bureau_balance_worst_recent_status():
    bb = pd.DataFrame(
        {"SK_ID_BUREAU": [10, 10, 10, 11, 11],
         "MONTHS_BALANCE": [-1, -5, -30, -2, -3],
         "STATUS": ["0", "2", "5", "C", "X"]}
    )
    out = bureau_balance_per_credit(bb)
    assert out.loc[10, "BB_WORST_STATUS_12M"] == 2  # the 5 is older than 12 months
    assert out.loc[10, "BB_EVER_DPD"] == 1
    assert np.isnan(out.loc[11, "BB_WORST_STATUS_12M"]) and out.loc[11, "BB_EVER_DPD"] == 0


def test_aggregate_bureau_counts_currency_and_ratio():
    bureau = pd.DataFrame(
        {"SK_ID_CURR": [1, 1, 1, 2],
         "SK_ID_BUREAU": [10, 11, 12, 13],
         "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Sold"],
         "CREDIT_CURRENCY": ["currency 1", "currency 1", "currency 2", "currency 1"],
         "DAYS_CREDIT": [-100, -900, -50, -400],
         "DAYS_CREDIT_UPDATE": [-5, -10, 3, -1],
         "AMT_CREDIT_SUM": [1000.0, 500.0, 9999.0, 300.0],
         "AMT_CREDIT_SUM_DEBT": [400.0, 0.0, 9999.0, 300.0],
         "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 0.0, 50.0],
         "AMT_CREDIT_MAX_OVERDUE": [np.nan, 10.0, np.nan, 50.0],
         "CNT_CREDIT_PROLONG": [0, 1, 0, 0]}
    )
    out = aggregate_bureau(bureau)
    assert out.index.is_unique and set(out.index) == {1, 2}
    a = out.loc[1]
    assert (a["BUREAU_COUNT"], a["BUREAU_ACTIVE_COUNT"], a["BUREAU_CLOSED_COUNT"]) == (3, 2, 1)
    assert a["BUREAU_ACTIVE_CREDIT_SUM"] == 1000.0  # currency 2 credit excluded from sums
    assert a["BUREAU_ACTIVE_DEBT_RATIO"] == pytest.approx(0.4)
    assert a["BUREAU_DAYS_CREDIT_MAX"] == -50
    assert out.loc[2, "BUREAU_BAD_DEBT_OR_SOLD_COUNT"] == 1
    assert bureau.loc[2, "AMT_CREDIT_SUM"] == 9999.0  # input not modified


def test_aggregate_previous_rates():
    prev = pd.DataFrame(
        {"SK_ID_CURR": [1, 1, 1, 1],
         "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Refused", "Canceled"],
         "AMT_CREDIT": [100.0, 0.0, 0.0, 0.0],
         "AMT_APPLICATION": [200.0, 50.0, 0.0, 10.0],
         "AMT_ANNUITY": [10.0, np.nan, np.nan, np.nan],
         "DAYS_DECISION": [-1000, -100, -500, -20],
         "CNT_PAYMENT": [12.0, np.nan, np.nan, np.nan]}
    )
    r = aggregate_previous(prev).loc[1]
    assert (r["PREV_COUNT"], r["PREV_APPROVED_COUNT"], r["PREV_REFUSED_COUNT"]) == (4, 1, 2)
    assert r["PREV_REFUSED_LAST_YEAR_COUNT"] == 1
    assert r["PREV_REFUSAL_RATE"] == pytest.approx(0.5)
    assert r["PREV_DAYS_DECISION_MAX"] == -20
    # 0/0 application ratio is ignored, not inf: mean of 0.5, 0, 0
    assert r["PREV_CREDIT_TO_APPLICATION_MEAN"] == pytest.approx(0.5 / 3)


def test_aggregate_installments_combines_partial_payments():
    inst = pd.DataFrame(
        {"SK_ID_CURR": [1, 1, 1, 1],
         "SK_ID_PREV": [7, 7, 7, 7],
         "NUM_INSTALMENT_VERSION": [1, 1, 1, 1],
         "NUM_INSTALMENT_NUMBER": [1, 1, 2, 3],
         "DAYS_INSTALMENT": [-100, -100, -70, -40],
         # instalment 1 paid in two parts, the last 40 days late; 2 early; 3 never paid
         "DAYS_ENTRY_PAYMENT": [-105, -60, -75, np.nan],
         "AMT_INSTALMENT": [100.0, 100.0, 100.0, 100.0],
         "AMT_PAYMENT": [60.0, 40.0, 90.0, np.nan]}
    )
    r = aggregate_installments(inst).loc[1]
    assert r["INST_COUNT"] == 3 and r["INST_UNPAID_COUNT"] == 1
    assert r["INST_LATE_SHARE"] == pytest.approx(0.5)  # of the 2 paid instalments
    assert r["INST_LATE30_SHARE"] == pytest.approx(0.5)
    assert r["INST_DAYS_LATE_MAX"] == 40
    assert r["INST_SHORTFALL_SUM"] == pytest.approx(10.0 + 100.0)
    assert r["INST_UNDERPAID_SHARE"] == pytest.approx(2 / 3)


def test_aggregate_pos_and_credit_card():
    pos = pd.DataFrame(
        {"SK_ID_CURR": [1, 1, 1], "SK_ID_PREV": [5, 5, 6], "MONTHS_BALANCE": [-1, -20, -3],
         "SK_DPD": [0, 30, 2], "SK_DPD_DEF": [0, 30, 0]}
    )
    p = aggregate_pos(pos).loc[1]
    assert (p["POS_LOAN_COUNT"], p["POS_DPD_MAX"], p["POS_DPD_MAX_LAST_YEAR"]) == (2, 30, 2)
    assert p["POS_DPD_MONTH_SHARE"] == pytest.approx(2 / 3)

    cc = pd.DataFrame(
        {"SK_ID_CURR": [1, 1, 1], "MONTHS_BALANCE": [-1, -2, -30],
         "AMT_BALANCE": [50.0, 100.0, 10.0], "AMT_CREDIT_LIMIT_ACTUAL": [100.0, 0.0, 100.0],
         "SK_DPD": [0, 0, 5], "AMT_DRAWINGS_CURRENT": [10.0, 0.0, 20.0]}
    )
    c = aggregate_credit_card(cc).loc[1]
    assert c["CC_UTILISATION_MAX"] == pytest.approx(0.5)  # zero-limit month ignored
    assert c["CC_UTILISATION_MEAN_LAST_YEAR"] == pytest.approx(0.5)
    assert c["CC_DPD_MAX"] == 5


def test_join_aggregate_keeps_rows_and_fills_counts_only():
    base = pd.DataFrame({"SK_ID_CURR": [1, 2, 3]})
    agg = pd.DataFrame(
        {"X_COUNT": [4], "X_MEAN": [1.5]}, index=pd.Index([2], name="SK_ID_CURR")
    )
    out = join_aggregate(base, agg, "HAS_X")
    assert len(out) == 3
    assert list(out["HAS_X"]) == [0, 1, 0]
    assert list(out["X_COUNT"]) == [0, 4, 0]
    assert out["X_MEAN"].isna().sum() == 2


def test_join_aggregate_rejects_duplicate_keys():
    agg = pd.DataFrame({"X_COUNT": [1, 2]}, index=pd.Index([1, 1], name="SK_ID_CURR"))
    with pytest.raises(ValueError, match="duplicate"):
        join_aggregate(pd.DataFrame({"SK_ID_CURR": [1]}), agg, "HAS_X")


SPLITS = {"train": 0.60, "validation": 0.15, "calibration": 0.10, "test": 0.15}


def synthetic_target(n=20_000, rate=0.08, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(range(n), name="SK_ID_CURR"), pd.Series((rng.random(n) < rate).astype(int))


def test_make_splits_sizes_stratification_and_disjointness():
    ids, y = synthetic_target()
    s = make_splits(ids, y, SPLITS, seed=42)
    assert len(s) == len(ids) and s["SK_ID_CURR"].is_unique
    assert set(s["split"]) == set(SPLITS)
    shares = s["split"].value_counts(normalize=True)
    for name, share in SPLITS.items():
        assert shares[name] == pytest.approx(share, abs=0.002)
    rates = y.groupby(s["split"]).mean()
    assert (rates - y.mean()).abs().max() < 0.002


def test_make_splits_is_reproducible():
    ids, y = synthetic_target()
    a = make_splits(ids, y, SPLITS, seed=42)
    b = make_splits(ids, y, SPLITS, seed=42)
    c = make_splits(ids, y, SPLITS, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a["split"].equals(c["split"])
