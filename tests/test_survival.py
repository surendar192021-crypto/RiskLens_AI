import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter

from src.survival.survival_analysis import (
    add_loan_covariates,
    cox_design,
    instalment_level,
    km_at_horizons,
    loan_survival_table,
)


def inst_rows(rows):
    cols = ["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    return pd.DataFrame(rows, columns=cols)


def test_event_and_censoring_times():
    raw = inst_rows([
        # loan 10: due -300, -270, -240; second instalment paid 45 days late -> event
        (1, 10, 1, 1, -300, -300, 100, 100),
        (1, 10, 1, 2, -270, -225, 100, 100),
        (1, 10, 1, 3, -240, -240, 100, 100),
        # loan 20: always on time -> censored at last paid due day
        (1, 20, 1, 1, -200, -201, 50, 50),
        (1, 20, 1, 2, -170, -170, 50, 50),
        # loan 30: schedule starts at instalment 5 (left-truncated) -> excluded
        (2, 30, 1, 5, -100, -100, 10, 10),
        # loan 40: partial payments; final part 10 days late -> not an event at dpd=30
        (2, 40, 1, 1, -90, -95, 70, 30),
        (2, 40, 1, 1, -90, -80, 70, 40),
        # loan 50: second instalment never paid -> not an event; censored at first
        (3, 50, 1, 1, -60, -60, 20, 20),
        (3, 50, 1, 2, -30, np.nan, 20, np.nan),
    ])
    t = loan_survival_table(instalment_level(raw), dpd=30).set_index("SK_ID_PREV")
    assert set(t.index) == {10, 20, 40, 50}
    assert (t.loc[10, "event"], t.loc[10, "duration"]) == (1, (-270 + 30) - (-300))
    assert (t.loc[20, "event"], t.loc[20, "duration"]) == (0, 30)
    assert (t.loc[40, "event"], t.loc[40, "duration"]) == (0, 0)
    assert (t.loc[50, "event"], t.loc[50, "duration"]) == (0, 0)


def test_dpd_parameter_changes_event_definition():
    raw = inst_rows([(1, 10, 1, 1, -100, -90, 10, 10), (1, 10, 1, 2, -70, -70, 10, 10)])
    per = instalment_level(raw)
    assert loan_survival_table(per, dpd=30)["event"].iloc[0] == 0
    assert loan_survival_table(per, dpd=5)["event"].iloc[0] == 1
    with pytest.raises(ValueError):
        loan_survival_table(per, dpd=-1)


def test_km_matches_hand_calculation():
    # durations 1..4, events at 1 and 3: S(1)=3/4, S(3)=3/4*1/2
    kmf = KaplanMeierFitter().fit([1, 2, 3, 4], [1, 0, 1, 0])
    s = km_at_horizons(kmf, [1, 3])
    assert s[1] == pytest.approx(0.75) and s[3] == pytest.approx(0.375)


def test_covariates_and_design_matrix():
    loans = pd.DataFrame({"SK_ID_PREV": [1, 2, 3], "SK_ID_CURR": [9, 9, 8],
                          "duration": [10.0, 20.0, 30.0], "event": [1, 0, 0]})
    prev = pd.DataFrame({
        "SK_ID_PREV": [1, 2, 3],
        "NAME_CONTRACT_TYPE": ["Cash loans", "Consumer loans", "XNA"],
        "AMT_CREDIT": [1000.0, 0.0, 50.0], "CNT_PAYMENT": [12.0, 6.0, 24.0],
        "NAME_YIELD_GROUP": ["high", "low_normal", "middle"],
    })
    d = add_loan_covariates(loans, prev)
    assert pd.isna(d.loc[2, "NAME_CONTRACT_TYPE"])  # XNA -> missing
    assert d.loc[0, "LOG_CREDIT"] == pytest.approx(np.log1p(1000))
    X = cox_design(d)
    assert len(X) == 2  # incomplete row dropped
    assert {"duration", "event", "LOG_CREDIT", "CNT_PAYMENT"} <= set(X.columns)
    assert not X.drop(columns="SK_ID_CURR").isna().any().any()
