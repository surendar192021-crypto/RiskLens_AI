import numpy as np
import pandas as pd
import pytest

from src.risk.expected_loss import el_by_group, expected_loss, lgd_sensitivity
from src.risk.portfolio_analytics import (
    add_age_band,
    apply_filters,
    expected_count_interval,
    group_levels,
    herfindahl,
    lorenz_curve,
    portfolio_summary,
    segment_table,
    top_share,
)

P = np.array([0.1, 0.2, 0.5, 0.05])
EAD = np.array([1000.0, 2000.0, 500.0, 4000.0])


# --- expected loss --------------------------------------------------------------

def test_expected_loss_hand_calculation():
    el = expected_loss(P, 0.45, EAD)
    np.testing.assert_allclose(el, [45.0, 180.0, 112.5, 90.0])
    assert el.sum() == pytest.approx(427.5)


def test_expected_loss_per_borrower_lgd_and_units():
    el = expected_loss(P, np.array([0.0, 1.0, 0.5, 0.5]), EAD)
    np.testing.assert_allclose(el, [0.0, 400.0, 125.0, 100.0])
    # EL scales with EAD units (e.g. thousands)
    in_thousands = expected_loss(P, 0.45, EAD / 1000) * 1000
    np.testing.assert_allclose(in_thousands, expected_loss(P, 0.45, EAD))


@pytest.mark.parametrize(
    "p, lgd, ead",
    [([1.1], 0.45, [1.0]), ([0.1], 1.2, [1.0]), ([0.1], -0.1, [1.0]), ([0.1], 0.45, [-5.0]),
     ([0.1], 0.45, [np.nan]), ([0.1, 0.2], 0.45, [1.0]), ([0.1, 0.2], [0.4, 0.5, 0.6], [1, 1])],
)
def test_expected_loss_rejects_invalid_inputs(p, lgd, ead):
    with pytest.raises(ValueError):
        expected_loss(p, lgd, ead)


def test_lgd_sensitivity_is_linear():
    t = lgd_sensitivity(P, EAD, [0.25, 0.5, 1.0])
    assert t["portfolio_el"].iloc[1] == pytest.approx(2 * t["portfolio_el"].iloc[0])
    assert t["portfolio_el"].iloc[2] == pytest.approx((P * EAD).sum())
    assert t["el_rate_of_exposure"].iloc[2] == pytest.approx((P * EAD).sum() / EAD.sum())


def test_el_by_group_sums_to_total():
    g = el_by_group(["a", "a", "b", "b"], P, 0.45, EAD)
    assert g["el"].sum() == pytest.approx(expected_loss(P, 0.45, EAD).sum())
    assert g["el_share"].sum() == pytest.approx(1.0)


# --- portfolio aggregation ------------------------------------------------------

def test_portfolio_summary_keeps_quantities_distinct():
    s = portfolio_summary(P, EAD, threshold=0.2, bands=["Very Low", "Low", "Moderate", "Critical"])
    assert s["borrowers"] == 4
    assert s["expected_difficulties"] == pytest.approx(0.85)  # sum of PD, not a count
    assert s["mean_pd"] == pytest.approx(0.2125)
    assert s["exposure_weighted_pd"] == pytest.approx((P * EAD).sum() / EAD.sum())
    assert s["above_threshold_borrowers"] == 2
    assert s["exposure_above_threshold"] == 2500.0
    assert s["high_or_critical_borrowers"] == 1 and s["critical_borrowers"] == 1


def test_expected_count_interval_independence():
    p = np.full(10_000, 0.1)
    r = expected_count_interval(p)
    assert r["expected"] == pytest.approx(1000)
    assert r["sd_if_independent"] == pytest.approx(np.sqrt(10_000 * 0.09))
    assert r["low_if_independent"] < 1000 < r["high_if_independent"]


def test_segment_table_adds_up_to_portfolio():
    df = pd.DataFrame({"seg": ["a", "a", "b", None]})
    y = [0, 1, 1, 0]
    t = segment_table(df, "seg", P, EAD, y=y).set_index("segment")
    assert t["borrowers"].sum() == 4 and "Missing" in t.index
    assert t["expected_difficulties"].sum() == pytest.approx(P.sum())
    assert t["exposure"].sum() == pytest.approx(EAD.sum())
    assert t["share_of_expected_difficulties"].sum() == pytest.approx(1.0)
    assert t.loc["a", "observed_difficulties"] == 1
    assert t.loc["a", "exposure_weighted_pd"] == pytest.approx((0.1 * 1000 + 0.2 * 2000) / 3000)
    assert t.loc["a", "risk_concentration_ratio"] == pytest.approx((0.3 / 0.85) / 0.5)


def test_group_levels_keeps_top_and_marks_other_and_missing():
    s = pd.Series(["x"] * 5 + ["y"] * 3 + ["z"] + [None])
    out = group_levels(s, 2)
    assert out.value_counts().to_dict() == {"x": 5, "y": 3, "Other": 1, "Missing": 1}


def test_age_band():
    out = add_age_band(pd.DataFrame({"AGE_YEARS": [24.9, 25.0, 64.99]}), [18, 25, 65, 120])
    assert list(out["AGE_BAND"]) == ["[18, 25)", "[25, 65)", "[25, 65)"]


def test_concentration_measures():
    assert herfindahl([0.25] * 4) == pytest.approx(0.25)
    assert herfindahl([1.0, 0.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        herfindahl([0.5, 0.6])
    assert top_share([10, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0.1) == pytest.approx(10 / 19)
    lc = lorenz_curve([1, 1, 1, 1])
    assert lc["share_of_total"].iloc[50] == pytest.approx(0.5)
    assert lc["share_of_total"].iloc[-1] == pytest.approx(1.0)


def test_apply_filters():
    df = pd.DataFrame({"t": ["Cash", "Revolving", "Cash"], "age": [30, 50, 70]})
    assert list(apply_filters(df, {"t": ["Cash"], "age": (25, 60)})) == [True, False, False]
    with pytest.raises(KeyError):
        apply_filters(df, {"nope": ["x"]})
