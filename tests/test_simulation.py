import numpy as np
import pandas as pd
import pytest

from src.risk.rating_migration import (
    matrix_power,
    transition_counts,
    transition_matrix,
    wilson_interval,
)
from src.simulation.monte_carlo import (
    basel_other_retail_rho,
    loss_statistics,
    resolve_rho,
    simulate_losses,
    vasicek_loss_quantile,
)
from src.simulation.stress_testing import (
    apply_scenario,
    check_scenario_output,
    validate_scenario,
)

RNG = np.random.default_rng(0)
P = RNG.uniform(0.01, 0.3, 2000)
EAD = RNG.uniform(1e4, 1e6, 2000)


# --- Monte Carlo ----------------------------------------------------------------

def test_mean_loss_matches_analytic_expected_loss():
    sim = simulate_losses(P, EAD, 0.45, 0.10, n_sims=4000, seed=1)
    el = (P * 0.45 * EAD).sum()
    stats = loss_statistics(sim["losses"], el, [0.99])
    assert abs(stats["mean_loss"] - el) < 4 * stats["mc_standard_error"]


def test_independent_case_matches_poisson_binomial_variance():
    sim = simulate_losses(P, EAD, 1.0, 0.0, n_sims=6000, seed=2)
    assert sim["defaults"].mean() == pytest.approx(P.sum(), rel=0.01)
    assert sim["defaults"].var() == pytest.approx((P * (1 - P)).sum(), rel=0.08)


def test_large_homogeneous_portfolio_matches_vasicek_formula():
    n, pd_, rho = 20_000, 0.05, 0.15
    sim = simulate_losses(np.full(n, pd_), np.ones(n), 1.0, rho, n_sims=6000, seed=3)
    for alpha in (0.95, 0.99):
        assert np.quantile(sim["defaults"] / n, alpha) == pytest.approx(
            vasicek_loss_quantile(pd_, rho, alpha), rel=0.06)


def test_correlation_widens_tail_but_keeps_mean():
    lo = simulate_losses(P, EAD, 0.45, 0.0, n_sims=3000, seed=4)["losses"]
    hi = simulate_losses(P, EAD, 0.45, 0.25, n_sims=3000, seed=4)["losses"]
    assert lo.mean() == pytest.approx(hi.mean(), rel=0.03)
    assert np.quantile(hi, 0.99) > 1.5 * np.quantile(lo, 0.99)


def test_reproducible_with_seed():
    a = simulate_losses(P, EAD, 0.45, 0.1, n_sims=500, seed=7)["losses"]
    b = simulate_losses(P, EAD, 0.45, 0.1, n_sims=500, seed=7)["losses"]
    c = simulate_losses(P, EAD, 0.45, 0.1, n_sims=500, seed=8)["losses"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_loss_statistics_definitions():
    losses = np.arange(1, 1001, dtype=float)
    s = loss_statistics(losses, analytic_el=400.0, levels=[0.99])
    assert s["var_99"] == pytest.approx(np.quantile(losses, 0.99))
    assert s["es_99"] >= s["var_99"]
    assert s["ul_99"] == pytest.approx(s["var_99"] - 400.0)


def test_basel_rho_range_and_invalid_inputs():
    r = basel_other_retail_rho([0.0001, 0.05, 0.99])
    assert r[0] == pytest.approx(0.16, abs=1e-3) and r[-1] == pytest.approx(0.03, abs=1e-6)
    assert ((r >= 0.03) & (r <= 0.16)).all()
    for bad in (1.0, -0.1, "unknown"):
        with pytest.raises(ValueError):
            resolve_rho(bad, [0.1])
    with pytest.raises(ValueError):
        simulate_losses([0.1], [-1.0], 0.45, 0.1, 10, 0)
    with pytest.raises(ValueError):
        simulate_losses([0.1], [1.0], 1.5, 0.1, 10, 0)


# --- Stress scenarios -------------------------------------------------------------

def borrowers():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4],
        "NAME_INCOME_TYPE": ["Working", "Pensioner", "Working", "State servant"],
        "AMT_INCOME_TOTAL": [100.0, 80.0, 200.0, 150.0],
        "AMT_CREDIT": [300.0, 160.0, 400.0, 600.0],
        "AMT_ANNUITY": [30.0, 16.0, 40.0, 60.0],
        "AMT_GOODS_PRICE": [270.0, 150.0, 380.0, 500.0],
        "CNT_FAM_MEMBERS": [2.0, 1.0, 3.0, 2.0],
        "INCOME_PER_FAMILY_MEMBER": [50.0, 80.0, 66.7, 75.0],
        "EXT_SOURCE_1": [0.5, np.nan, 0.9, 0.2],
        "EXT_SOURCE_2": [0.6, 0.4, 0.95, 0.3],
        "EXT_SOURCE_3": [0.7, 0.2, np.nan, 0.4],
        "AGE_YEARS": [30.0, 70.0, 40.0, 50.0],
        "EMPLOYED_YEARS": [5.0, np.nan, 10.0, 20.0],
    })


def test_income_and_annuity_shocks_recompute_ratios():
    b = borrowers()
    s = apply_scenario(b, {"income_cut": 0.2, "annuity_rise": 0.1})
    np.testing.assert_allclose(s["AMT_INCOME_TOTAL"], b["AMT_INCOME_TOTAL"] * 0.8)
    np.testing.assert_allclose(s["AMT_ANNUITY"], b["AMT_ANNUITY"] * 1.1)
    np.testing.assert_allclose(s["ANNUITY_INCOME_RATIO"], s["AMT_ANNUITY"] / s["AMT_INCOME_TOTAL"])
    np.testing.assert_allclose(s["CREDIT_ANNUITY_RATIO"], s["AMT_CREDIT"] / s["AMT_ANNUITY"])
    check_scenario_output(b, s)
    assert b["AMT_INCOME_TOTAL"].iloc[0] == 100.0  # input untouched


def test_ext_source_shock_stays_in_range_and_updates_summaries():
    s = apply_scenario(borrowers(), {"ext_source_cut": 0.1})
    assert s.loc[0, "EXT_SOURCE_1"] == pytest.approx(0.45)
    assert s.loc[0, "EXT_SOURCE_MEAN"] == pytest.approx(np.mean([0.45, 0.54, 0.63]))
    assert s.loc[1, "EXT_SOURCE_MIN"] == pytest.approx(0.18)
    assert pd.isna(s.loc[1, "EXT_SOURCE_1"])


def test_job_loss_hits_only_employed_and_is_seeded():
    spec = {"job_loss_share": 0.999, "income_after_job_loss": 0.5}
    s = apply_scenario(borrowers(), spec, seed=0)
    assert not s.loc[1, "_job_loss"]  # pensioner unaffected
    hit = s["_job_loss"]
    assert (s.loc[hit, "EMPLOYED_YEARS"] == 0).all()
    np.testing.assert_allclose(s.loc[hit, "AMT_INCOME_TOTAL"],
                               borrowers().loc[hit, "AMT_INCOME_TOTAL"] * 0.5)
    pd.testing.assert_frame_equal(s, apply_scenario(borrowers(), spec, seed=0))


def test_baseline_scenario_changes_nothing():
    b = borrowers()
    pd.testing.assert_frame_equal(apply_scenario(b, {}), b)


@pytest.mark.parametrize("spec", [{"income_cut": 1.0}, {"income_cut": -0.1}, {"typo": 0.1},
                                  {"job_loss_share": 0.1}, {"lgd": 1.5},
                                  {"job_loss_share": 0.1, "income_after_job_loss": 0}])
def test_invalid_scenarios_rejected(spec):
    with pytest.raises(ValueError):
        validate_scenario(spec)


# --- Migration --------------------------------------------------------------------

STATES = ["0", "1", "C"]


def test_transition_counts_use_consecutive_months_only():
    bb = pd.DataFrame({
        "SK_ID_BUREAU": [1, 1, 1, 1, 2, 2],
        "MONTHS_BALANCE": [-4, -3, -2, 0, -1, 0],  # gap between -2 and 0 for credit 1
        "STATUS": ["0", "1", "0", "C", "0", "0"],
    })
    c = transition_counts(bb, STATES)
    assert c.loc["0", "1"] == 1 and c.loc["1", "0"] == 1 and c.loc["0", "0"] == 1
    assert c.to_numpy().sum() == 3  # the -2 -> 0 jump is not a one-month transition
    c2 = transition_counts(bb, STATES, step=2)
    assert c2.loc["1", "C"] == 0 and c2.loc["0", "0"] == 1


def test_matrix_rows_sum_to_one_and_power():
    counts = pd.DataFrame([[8, 2, 0], [5, 5, 0], [0, 0, 4]], index=STATES, columns=STATES)
    P = transition_matrix(counts)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    P2 = matrix_power(P, 2)
    np.testing.assert_allclose(P2.to_numpy(), P.to_numpy() @ P.to_numpy())
    empty = transition_matrix(pd.DataFrame([[1, 0], [0, 0]], index=["a", "b"],
                                           columns=["a", "b"]))
    with pytest.raises(ValueError):
        matrix_power(empty, 2)


def test_wilson_interval_contains_estimate():
    lo, hi = wilson_interval(np.array([0, 50, 100]), np.array([100, 100, 100]))
    assert lo[0] == 0 or lo[0] < 1e-12
    assert lo[1] < 0.5 < hi[1] and hi[2] == pytest.approx(1.0)
