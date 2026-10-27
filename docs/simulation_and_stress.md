# Monte Carlo simulation, stress testing and delinquency migration (Phase 9)

Runs:
- `python -m scripts.run_simulation` (about 80 s)
- `python -m scripts.run_migration` (about 15 s)

Code: `src/simulation/monte_carlo.py`, `stress_testing.py` and `src/risk/rating_migration.py`.
Settings: `config.yaml → monte_carlo, stress_testing, rating_migration`. Numbers come from
`reports/experiment_results/phase9_*_summary.json` and `reports/tables/mc_*.csv`,
`stress_results.csv`, `migration_*.csv`.

**Portfolio:** the 48,744 `application_test` borrowers, deployed model.
**EAD:** AMT_CREDIT. **LGD:** 45% (**assumed**). Amounts are in dataset currency units.

## 1. Monte Carlo loss distribution

**Model.** Vasicek one-factor Gaussian model (Vasicek, 2002; Gordy, 2003). Borrower *i*
defaults when √ρ·Z + √(1−ρ)·εᵢ < Φ⁻¹(PDᵢ):
- **Z** is one common "economy" factor shared by all borrowers.
- **εᵢ** is each borrower's own independent noise.

Given Z, defaults are independent with probability Φ((Φ⁻¹(PDᵢ) − √ρ·Z)/√(1−ρ)). Its average over
Z is exactly PDᵢ, so the model **keeps every borrower's PD but makes defaults move together**.

| Assumption | Value | Status |
|---|---|---|
| Asset correlation ρ | 0.10 | **Assumed.** No data can estimate it. It lies inside the Basel II "other retail" range (0.03–0.16), used only as a reference |
| LGD | 45% | **Assumed** (Phase 8) |
| Horizon | That of `TARGET` (early payment difficulty) | Not a one-year default horizon |
| Simulations / seed | 5,000 / 42, in batches of 250 | Reproducible: same settings, same numbers (unit-tested) |

**Verification:**

| Check | Result |
|---|---|
| Simulated mean loss vs analytic EL = Σ PD·LGD·EAD (portfolio) | 809.1M vs 800.9M, a gap of 1.3 Monte Carlo standard errors ✓ |
| ρ = 0: variance of the default count vs the formula for independent borrowers, Σp(1−p) | Matches within 8% (unit test) ✓ |
| Large uniform portfolio: simulated tail quantiles vs Vasicek's closed form | Within 6% at 95% and 99% (unit test) ✓ |
| More correlation widens the tail but leaves the mean unchanged | ✓ (unit test and the table below) |

**Baseline results (ρ = 0.10):**

| Measure | Value |
|---|---:|
| Analytic EL | 800.9M |
| Mean simulated loss | 809.1M |
| Median simulated loss | 717.8M (right-skewed: most years are mild, a few are severe) |
| VaR 95% / 99% / 99.9% | 1.67B / 2.23B / 2.90B |
| Expected shortfall 99% (average loss beyond VaR 99%) | 2.53B |
| Unexpected loss, defined here as VaR 99.9% − EL | 2.09B |
| Mean simulated difficulty cases / 99th percentile | 3,697 / 9,985 |

Figure: `reports/figures/phase9_loss_distribution.png`.

**Sensitivity to the assumed correlation** (5,000 simulations each):

| ρ | Mean loss | Loss standard deviation | VaR 99% | VaR 99.9% | 99th percentile of difficulty cases |
|---|---:|---:|---:|---:|---:|
| 0 (independent) | 800.8M | 14.5M | 0.84B | 0.84B | 3,788 |
| 0.03 | 805.6M | 234.6M | 1.47B | 1.72B | 6,656 |
| Basel "other retail" formula (mean ρ = 0.064) | 806.6M | 292.2M | 1.72B | 2.13B | 7,666 |
| **0.10 (baseline)** | 809.1M | 441.7M | 2.23B | 2.90B | 9,985 |
| 0.16 | 811.3M | 573.5M | 2.76B | 3.77B | 12,382 |
| 0.25 | 814.0M | 744.3M | 3.52B | 4.98B | 15,771 |

**Finding:**
- **The mean barely moves** (all within simulation error of 800.9M), as theory says it should.
- **The tail is set almost entirely by the assumed correlation.** VaR 99.9% goes from 0.84B to
  4.98B. With independent borrowers the portfolio looks almost riskless beyond its mean; with
  realistic correlation, a bad year can cost 3–6 times the expected loss.
- **So VaR and unexpected loss here are not estimates of real risk.** They are *conditional on
  ρ*, and they illustrate why correlation, not PD accuracy, dominates tail risk. This answers
  **RQ6**: the simulation represents loss uncertainty transparently, but its tail is only as
  credible as the correlation assumption.

**Not claimed:** these figures are not regulatory capital, and not the Basel IRB capital formula.

## 2. Stress testing (feature perturbation, rescored by the model)

Each scenario changes borrowers' **raw inputs**, recomputes every derived feature with the
dataset's own functions, checks validity (same borrowers, positive incomes, external scores
in [0, 1], no infinities), and **rescores everyone with the deployed model**. PDs are never
multiplied directly.

**These are hypothetical experiments, not calibrated macroeconomic stress tests.** No
macroeconomic data is available to calibrate them.

| Scenario | Change to inputs |
|---|---|
| income_shock | Every income × 0.8 |
| repayment_burden | Every annuity × 1.15 (e.g. a rate rise) |
| external_score_shock | EXT_SOURCE_1–3 × 0.9 |
| employment_shock | 10% of employed borrowers (3,877) lose their job: income × 0.5, job tenure reset to 0 |
| combined_adverse | All of the above, plus LGD 55% instead of 45% |

| Scenario | Mean PD | Expected difficulties (change) | Above 16.2% threshold (change) | Borrowers with PD +1pp or more | Borrowers whose PD *fell* | EL (change) | VaR 99.9% (change) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 7.51% | 3,659 | 5,582 | — | — | 800.9M | 2.81B |
| income_shock | 7.66% | 3,731 (+2.0%) | 5,781 (+3.6%) | 4.5% | **32.1%** | 816.6M (+2.0%) | +1.0% |
| repayment_burden | 7.46% | 3,635 (**−0.7%**) | 5,431 (**−2.7%**) | 8.9% | **38.4%** | 797.9M (−0.4%) | +0.5% |
| external_score_shock | 8.72% | 4,249 (+16.1%) | 7,040 (+26.1%) | 41.7% | 3.2% | 938.8M (+17.2%) | +12.3% |
| employment_shock | 7.63% | 3,721 (+1.7%) | 5,746 (+2.9%) | 4.0% | 0.4% | 815.2M (+1.8%) | +1.1% |
| combined_adverse | 9.02% | 4,398 (+20.2%) | 7,337 (+31.4%) | 52.0% | 9.8% | 1,189.8M (+48.6%) | +41.3% |

Stress VaR uses 2,000 simulations with ρ = 0.10. Every scenario uses the same random draws, so
the *differences* between scenarios are reliable. Absolute levels carry about ±3% simulation
error: the baseline's 2,000-run mean sits about 2.4 standard errors above its analytic EL.
Figure: `reports/figures/phase9_stress.png`.

**Findings:**
1. **External scores dominate the model's response.** A 10% fall in external credit scores
   raises expected difficulties by 16% and flagged borrowers by 26%. That fits Phase 6, where
   these were the strongest drivers.
2. **Income and employment shocks barely move the model** (+2%). Income-based features are not
   among its main drivers (Phase 6, Phase 7 synthetic demo).
3. **Two scenarios produce economically implausible responses:**
   - **Income cut:** a 20% income cut *lowers* the PD of 32% of borrowers.
   - **Higher instalments:** a 15% rise *lowers* the portfolio's average PD. The model associates
     larger instalments relative to the loan (shorter loans, lower `CREDIT_ANNUITY_RATIO`) with
     lower risk; the counterfactuals in Phase 6 showed the same pattern.

   These are **learned associations in a non-causal model, not effects of the shocks.**
   **Conclusion: feature-perturbation stress testing with this model is not credible for income-
   and instalment-based shocks.** Their results must not be reported as "the portfolio is
   resilient to income shocks".
4. **The combined scenario's EL increase (+48.6%) comes mostly from the LGD assumption** (+22%
   from 45% to 55%), with the rest from the model's response to the external-score shock.

**What would make stress testing credible (not done here):**
- **Monotonic constraints:** retrain with constraints such as PD never falling when income falls
  (LightGBM `monotone_constraints`).
- **Macro calibration:** link scenarios to macroeconomic data through a validated satellite model.

## 3. Delinquency-state migration (observed data)

**Data:**
- **What:** the monthly status of bureau credits (`bureau_balance`) of training-split applicants.
- **Size:** 8,784,979 monthly records, 313,007 credits, 8,471,972 one-month transitions.
- **States:** 0 = not overdue; 1–5 = 1–30, 31–60, 61–90, 91–120, 120+ days past due (5 also
  covers sold or written off); C = closed; X = status unknown.

**These are observed transitions of credits at other lenders before the Home Credit
application.** They are not RiskLens ratings, and not the current portfolio.

**One-month transition probabilities** (rows = this month, columns = next month; full matrix in
`migration_matrix_1m.csv`, figure `phase9_migration_1m.png`):

| From | Transitions observed | Stay | Back to 0 | Worse | Closed |
|---|---:|---:|---:|---:|---:|
| 0 | 2,694,569 | 0.859 [0.858, 0.859] | — | 0.021 | 0.051 |
| 1 (1–30 DPD) | 90,022 | 0.348 [0.345, 0.351] | 0.466 | 0.063 | 0.091 |
| 2 (31–60) | 9,138 | 0.175 [0.167, 0.183] | 0.248 | 0.314 | 0.072 |
| 3 (61–90) | 3,530 | 0.108 [0.098, 0.119] | 0.132 | 0.561 | 0.068 |
| 4 (91–120) | 2,229 | 0.059 [0.050, 0.069] | 0.071 | 0.687 | 0.078 |
| 5 (120+) | 23,817 | 0.915 [0.911, 0.918] | 0.018 | — | 0.028 |
| C (closed) | 4,019,517 | 1.000 | 0 | 0 | — |

The bracketed intervals are 95% Wilson intervals. Every row sums to 1, and every row has
at least 2,229 observations.

**Findings:**
- **Delinquency snowballs.** The chance of getting worse next month rises from 2% (not overdue)
  to 69% (91–120 days).
- **Recovery is common early and rare late.** 47% of credits 1–30 days overdue return to 0 next
  month, against 1.8% of those 120+ days overdue.
- **Closed is absorbing:** 1 exit in 4.0 million.

**The "same behaviour every month" (time-homogeneous Markov) assumption fails.** Multiplying the
1-month matrix 12 times (P¹²) gives probabilities that differ from the 12-month transitions
actually observed by up to:
- **0.23 from state 5:** P¹² gives 35% still at 120+ days after a year; 58% were observed.
- **0.14 from state 0:** P¹² understates closures (0.41 vs 0.51) and overstates unknown status
  (0.21 vs 0.06).
- **0.44 from X**, which looks like a reporting artefact.

So delinquency has *memory*: credits that have been severely overdue stay that way longer than
a memoryless chain predicts. **The 1-month matrix is a sound description, but projecting it
forward with P^k is not valid.** Multi-period risk would need duration-dependent or
semi-Markov models.

## 4. Limitations

- **The Monte Carlo tail depends on an assumed ρ.** A single factor captures common shocks but
  not sector or regional clustering.
- **LGD is assumed constant across borrowers,** and it is uncorrelated with the factor here. In
  reality, LGD rises in downturns; the combined scenario's 55% only crudely reflects that.
- **The stress results are only as sound as the model's learned associations**, which are not
  causal and not monotone (section 2).
- **The migration matrix describes external bureau credits in the past.** It is not a rating
  system and not a forecast.

## References

- Vasicek, O. (2002). The distribution of loan portfolio value. *Risk*, 15(12), 160–162.
- Gordy, M. B. (2003). A risk-factor model foundation for ratings-based bank capital rules.
  *Journal of Financial Intermediation*, 12(3), 199–232.
- Basel Committee on Banking Supervision (2006). *International Convergence of Capital
  Measurement and Capital Standards: A Revised Framework* (Basel II), retail correlation and LGD
  parameters, used only as reference values.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
  *Journal of the American Statistical Association*, 22(158), 209–212.
