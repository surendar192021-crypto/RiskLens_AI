# Portfolio risk analytics and expected loss (Phase 8)

Run: `python -m scripts.run_portfolio` (about 3 s). Code: `src/risk/portfolio_analytics.py` and
`src/risk/expected_loss.py`. Settings: `config.yaml → portfolio, expected_loss`. Numbers come from
`reports/experiment_results/phase8_portfolio_summary.json` and `reports/tables/portfolio_*.csv`,
`el_*.csv`.

**Portfolio** = the 48,744 unlabelled applicants in `application_test`, scored by the deployed
model (`lightgbm-none-phase5_20261027_004048`). **Amounts** are in the dataset's own currency
units, which are not stated.

## 1. Definitions (kept separate on purpose)

| Quantity | Meaning | It is NOT |
|---|---|---|
| Borrowers / share | How many borrowers | — |
| Mean PD | Average estimated probability of payment difficulty | A default rate |
| Expected difficulties = Σ PD | Expected *number* of payment-difficulty cases | A forecast of the exact count |
| Exposure (EAD proxy) | Σ AMT_CREDIT, the credit amount applied for | Outstanding balance at default |
| Exposure-weighted PD | Σ(PD × EAD) / Σ EAD: PD of an average unit of money lent | — |
| Risk concentration ratio | Share of expected difficulties ÷ share of borrowers | — |
| Expected loss | Σ PD × LGD × EAD, with LGD **assumed** | Regulatory capital, a provision, or an audited loss |

The "95% range" for expected difficulties uses a normal approximation that **assumes
independent borrowers**. Real defaults are correlated, so the true uncertainty is wider; Phase 9
models that.

## 2. Portfolio summary

| Measure | Value |
|---|---:|
| Borrowers | 48,744 |
| Mean PD / median PD | 7.51% / 4.47% |
| Expected difficulties (Σ PD) | 3,659 (range if independent: 3,551–3,768) |
| At or above the 16.2% review threshold | 5,582 borrowers (11.5%), holding 10.3% of exposure |
| High or Critical band (PD ≥ 60%) | 45 borrowers (0.09%) |
| Total exposure | 25.19 billion |
| Exposure-weighted PD | 7.07% |

**Exposure-weighted PD is below mean PD** (7.07% vs 7.51%), so larger loans go, on average, to
lower-PD borrowers.

**Risk bands:**

| Band | Borrowers | Share | Mean PD | Share of expected difficulties | Concentration ratio | Share of EL |
|---|---:|---:|---:|---:|---:|---:|
| Very Low | 44,875 | 92.1% | 5.6% | 68.2% | 0.74 | 69.9% |
| Low | 3,297 | 6.8% | 26.9% | 24.3% | 3.58 | 23.0% |
| Moderate | 527 | 1.1% | 46.9% | 6.7% | 6.24 | 6.4% |
| High | 44 | 0.09% | 65.1% | 0.8% | 8.68 | 0.7% |
| Critical | 1 | 0.002% | 80.9% | 0.02% | — | 0.02% |

**Observation:** most expected difficulties (68%) and most EL (70%) sit in the **Very Low**
band, simply because it holds 92% of borrowers. A portfolio view that watches only High and
Critical borrowers would miss nearly all of the expected losses.

## 3. Segments (selected)

| Segment | Borrowers | Mean PD | Exposure-weighted PD | Concentration ratio |
|---|---:|---:|---:|---:|
| Age 18–25 | 1,880 | 12.4% | 12.7% | 1.65 |
| Age 25–35 | 11,288 | 9.6% | 9.2% | 1.28 |
| Age 65+ | 1,234 | 3.7% | 4.0% | 0.49 |
| Income: working | 24,533 | 8.8% | 8.2% | 1.17 |
| Income: pensioner | 9,273 | 5.2% | 5.3% | 0.69 |
| Region rating 3 (lowest) | 7,624 | 9.8% | 9.4% | 1.31 |
| Region rating 1 (best) | 5,764 | 4.4% | 4.1% | 0.58 |
| Higher education | 12,516 | 5.3% | 4.9% | 0.71 |

The full table covers 7 segment variables, with rare levels grouped as "Other" and missing values
shown as "Missing". These are **descriptive model outputs**: they show where the model puts
risk, not why. Age and some other variables raise fairness questions, which Phase 10 addresses.

**Exposure concentration across segments** (Herfindahl index; 1 = all exposure in one group):
contract type 0.99, housing type 0.82, region rating 0.53, education 0.52, income type 0.34,
age band 0.24, organisation type 0.17.

**Contract-type composition differs from the labelled data.** Revolving loans are **0.9%** of
this portfolio (439 borrowers) but **9.5%** of the labelled test split (4,375 of 46,127). So
`application_test` is not a simple random draw from the same population as the labelled data.
That is a population difference, and Phase 10 (drift) will measure it properly.

## 4. Do the aggregates match reality? (labelled test split, 46,127 borrowers)

| Check | Expected (from PDs) | Observed |
|---|---:|---:|
| Number of difficulty cases | 3,704 (range if independent: 3,597–3,812) | **3,724** |
| Credit amount of borrowers with difficulties (Σ PD × EAD vs Σ y × EAD) | 2.058 billion | 2.075 billion (−0.8%) |
| Segments with 1,000+ borrowers (30 of them): largest relative error | — | 13.6% |

**Finding:** summed PDs reproduce the observed number of difficulty cases, both overall and
exposure-weighted. The largest segment error (organisation type "School": 76 expected vs 88
observed) is within about 1.4 standard deviations of chance variation for a count that size.

**The model slightly underestimates some groups** (pensioners −5%, age 65+ −11%) **and
overestimates others** (commercial associates +7%, higher education +7%). These are small
errors, but systematic, and they belong in the fairness review.

So aggregating the model's PDs is **empirically supported** for counts and exposure on data like
the training population. It is not verified for a population that has shifted (see section 3).

## 5. Expected loss (assumption-based)

**Inputs:**
- **PD:** the model probability of `TARGET` = 1. This means *early payment difficulty*, not a
  12-month regulatory default. Treating every payment difficulty as a default with loss LGD makes
  this EL an **upper-end illustration**.
- **LGD:** **assumed 45%**. That is the Basel II foundation-IRB supervisory value for senior
  unsecured claims, used here only as a reference point; the dataset has no recovery data.
- **EAD:** AMT_CREDIT, assumed fully outstanding at default (no amortisation). For the few
  revolving loans this treats the credit limit as fully drawn.

**Result (LGD 45%): portfolio EL = 800.9 million, 3.18% of exposure.**

**Sensitivity to the LGD assumption.** EL is linear in LGD, so this assumption alone moves EL by
a factor of 3.4 across a plausible range:

| LGD (assumed) | Portfolio EL | EL as share of exposure |
|---:|---:|---:|
| 25% | 444.9 million | 1.77% |
| 45% | 800.9 million | 3.18% |
| 65% | 1,156.8 million | 4.59% |
| 85% | 1,512.8 million | 6.01% |

**Concentration:**

| Group | Share of EL |
|---|---:|
| Top 1% of borrowers by EL | 8.4% |
| Top 10% of borrowers by EL | 40.7% |
| Borrowers at or above the review threshold (11.5% of borrowers) | 38.1% |

For comparison, the top 10% hold 26.2% of exposure. Figure: `reports/figures/phase8_el_lorenz.png`.

**Not done, deliberately:**
- **Unexpected loss, VaR and loss distributions:** these need a model of correlated defaults
  (Phase 9).
- **Regulatory capital:** not computed or claimed.

## 6. Limitations

- **EL is only as good as its weakest input.** Here that is LGD, which is entirely assumed.
- **EAD ignores amortisation, fees and interest,** and the currency is unknown.
- **PD measures early payment difficulty, not default.** Its horizon and definition differ from
  regulatory PD.
- **Independence:** the 95% range for expected difficulties assumes it, which understates real
  uncertainty.
- **Segment results describe the model's view.** Their agreement with outcomes is verified only
  on the labelled test split.
