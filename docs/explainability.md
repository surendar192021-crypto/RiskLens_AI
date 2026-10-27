# Explainability: SHAP and constrained counterfactuals (Phase 6)

Run: `python -m scripts.run_explainability` (about 2 minutes). Model explained:
`lightgbm-none-phase5_20261027_004048` from `models/current_model.json`. All numbers are on the
**test split**, and come from `reports/experiment_results/phase6_summary.json`,
`reports/tables/shap_global_importance_test.csv` and
`reports/tables/counterfactual_evaluation_test.csv`. This phase explains the model; it doesn't
change it or choose anything.

## 1. How the SHAP explanations work, and how they were verified

**What SHAP is.** For each borrower, SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017)
splits the model's raw score into one contribution per feature. The raw score is in
*log-odds*, i.e. log(p / (1 − p)). The split satisfies:

> base value + Σ feature contributions = the model's raw score for that borrower

**Algorithm.** For tree models we use **TreeSHAP** (Lundberg et al., 2020), which computes these
values exactly rather than approximately.

**Base value.** −2.912 log-odds, i.e. 5.2%. It is the average raw score, which differs from the
average probability (8.0%) because averaging log-odds is not the same as averaging probabilities.

**Verification:**
- **Additivity is checked on every call.** `ShapExplainer.explain` raises an error if base value +
  contributions differs from the model's raw score by more than 1e-6. The observed error is about 1e-14.
- **Results match LightGBM's own implementation exactly.** The `shap` values equal LightGBM's
  built-in TreeSHAP output (`pred_contrib=True`); the maximum difference is 0.0, checked on the
  real model and in `tests/test_explainability.py`.
- **The explainer recovers known drivers.** On synthetic data with known risk drivers, the true
  drivers rank in the top 3 (unit test).

**Important: contributions are not causes.** A large contribution means the model *relied on*
that feature for this borrower. It does not mean that changing the feature would change the
borrower's real-world risk.

## 2. Global importance (5,000 random test borrowers)

| Rank | Feature | Mean \|SHAP\| (log-odds) | Share of total | Rank by split gain |
|---:|---|---:|---:|---:|
| 1 | EXT_SOURCE_MEAN | 0.285 | 8.9% | 1 |
| 2 | EXT_SOURCE_3 | 0.117 | 3.6% | 2 |
| 3 | GOODS_CREDIT_RATIO | 0.111 | 3.5% | 7 |
| 4 | PREV_CREDIT_TO_APPLICATION_MEAN | 0.094 | 2.9% | 9 |
| 5 | CODE_GENDER | 0.093 | 2.9% | 45 |
| 6 | ORGANIZATION_TYPE | 0.092 | 2.9% | 4 |
| 7 | PREV_CNT_PAYMENT_MEAN | 0.092 | 2.9% | 19 |
| 8 | EXT_SOURCE_1 | 0.091 | 2.8% | 8 |
| 9 | CREDIT_ANNUITY_RATIO | 0.090 | 2.8% | 6 |
| 10 | EXT_SOURCE_2 | 0.088 | 2.7% | 5 |

The full ranking is in the CSV. Figures: `reports/figures/phase6_shap_beeswarm.png` and
`phase6_shap_bar.png`.

**How to read the beeswarm colours:** for categorical features, colour shows the category's
alphabetical code, not a size. For `CODE_GENDER`, blue is F and red is M. In the plot, M pushes
estimates up and F pushes them down, in line with the observed base rates (10.1% vs 7.0%, Phase 2).

**Observed:**
- **The external credit scores dominate.** Their mean is by far the strongest single driver, and
  all five EXT_SOURCE features are in the top 11. This matches the Phase 3 univariate screening.
- **Influence is spread out.** The top 10 features account for only 36% of total |SHAP|, so the
  model draws on many features, including history from four of the five side tables.
- **SHAP and split-gain broadly agree on which features matter, less on their order.** Their
  Spearman rank correlation across all 155 features is 0.95. This figure is flattered by the 24
  features the model never splits on: they are exactly zero in both measures, so they agree
  trivially. Over the 131 used features it is 0.92; within the SHAP top 20 it is only 0.51.
  (Recomputed in the Phase 12 audit; stable at 0.950–0.951 across bootstrap resamples of the
  5,000-borrower SHAP sample.)
- **Exception: `CODE_GENDER`** ranks 5th by SHAP but 45th by gain. It is used in few splits, but
  those splits shift many borrowers' scores. **This is a substantive finding for the fairness
  audit (Phase 10).** The model gives gender direct weight. Whether that is acceptable is a policy
  and legal question this project cannot settle; we will measure its effect on group-level error
  rates.
- **The external scores can't be explained further.** The data documentation doesn't say how
  `EXT_SOURCE_*` are built, so the model's strongest driver is itself a black box.

This answers **RQ3**: SHAP gives exact, verified global and per-borrower explanations of the
deployed model's output. Their meaning is limited to "what the model relied on".

## 3. Example borrower explanations (test split, chosen by rule, not by hand)

| Case | Borrower | PD | Actual TARGET | Top drivers of the model's estimate (log-odds) |
|---|---:|---:|---:|---|
| Highest PD in test | 164614 | 84.4% | 1 | ↑ BUREAU_OVERDUE_SUM = 436,460 (+0.86), EXT_SOURCE_MEAN = 0.115 (+0.82), EXT_SOURCE_MIN = 0.0005 (+0.35) |
| Lowest PD at or above the 16.2% threshold | 383485 | 16.2% | 0 | ↑ EXT_SOURCE_MEAN = 0.371 (+0.33), GOODS_CREDIT_RATIO = 0.716 (+0.31), BUREAU_ACTIVE_DEBT_RATIO = 0.969 (+0.26); ↓ OWN_CAR_AGE = 8 (−0.14) |
| Median PD | 216125 | 4.7% | 0 | ↓ EXT_SOURCE_MEAN = 0.594 (−0.21), OWN_CAR_AGE = 5 (−0.14), higher education (−0.14) |

Each case has a waterfall chart (`reports/figures/phase6_waterfall_<case>.png`) and a
plain-English summary generated by `plain_english()`. The summary always ends: *"These are
contributions to the model's output for this borrower, not proven causes of payment difficulty."*

## 4. Constrained counterfactuals

**Question:** what change to *this loan's terms* would bring the model's estimate below the
16.2% review threshold?

**Constraints** (`src/explainability/counterfactuals.py`, config `counterfactuals`):

| Allowed to change | How | Range |
|---|---|---|
| Loan amount | AMT_CREDIT and AMT_GOODS_PRICE × credit factor | 0.50–1.00 (borrow up to 50% less; never more) |
| Repayment period | AMT_ANNUITY × credit factor ÷ term factor | 0.75–1.50 × the original period |
| Derived ratios | Recomputed with the same `loan_ratio_features()` used to build the dataset | — |

- **Locked:** everything else, including age, income, employment, external scores, gender and all
  credit history. Each search checks that every locked column is unchanged.
- **Plausibility:** new amounts must stay within the training split's 0.5–99.5% range.
- **Scoring:** every candidate (11 × 7 grid) is scored by the deployed model.
- **Choice:** the reported counterfactual is the one with the smallest total relative change.
- **Approximation:** the annuity rule ignores interest, because the dataset gives no rate, so
  longer periods look somewhat cheaper than they would really be.
- **Why not DiCE or another generic optimiser:** two transparent levers with explicit bounds are
  easier to defend and to verify. A generic optimiser would need custom constraints for locked
  features and derived ratios anyway.

**Feasibility study.** 500 randomly chosen test borrowers at or above the threshold, out of 5,973
flagged in total:

| Original PD | Borrowers | Counterfactual found | Median lowest reachable PD |
|---|---:|---:|---:|
| 16.2–25% | 271 | 87.1% | 12.8% |
| 25–40% | 161 | 21.1% | 20.0% |
| 40% and above | 68 | 0% | 32.2% |
| **All** | **500** | **54.0%** | — |

Among the 270 borrowers with a counterfactual:
- **Change needed:** 45.6% need only a different repayment period; 54.4% also need a smaller loan.
- **Typical change:** the median is a 5% smaller loan (factor 0.95) and a 12.5% longer period
  (factor 1.125), lowering the estimate by a median of 5.0 percentage points.
- **Outcomes differ by group:** the actual TARGET rate was 23.3% among borrowers with a
  counterfactual, against 34.3% among those without one.

**Interpretation and caveats:**
1. **Loan terms only help near the threshold.** Above about 40% PD, no allowed change to the loan
   reaches the threshold: the estimate is driven by locked factors such as external scores and
   credit history. This answers **RQ4**: constrained counterfactuals are feasible for about half
   of flagged borrowers, almost all of them close to the threshold.
2. **The model's associations are not always intuitive.** For 36 of the 270 borrowers (13%;
   18 with the period change alone, 18 combined with a smaller loan), the smallest change was a
   *shorter* repayment period, which means larger instalments. (An earlier version of this
   document said 12; the audit recount from `counterfactual_evaluation_test.csv` gives 36.) The model
   associates shorter loans (lower `CREDIT_ANNUITY_RATIO`, 9th by SHAP) with lower risk. That is a
   learned association in this sample, possibly reflecting who is offered short loans, **not**
   evidence that higher instalments make a borrower safer. Counterfactuals expose such patterns;
   they don't justify them.
3. **These are model-based hypotheticals.** They are not offers, and not predictions of what
   would happen to the borrower. Every result carries that note in code.

## 5. Limitations

- **SHAP explains the model, not reality.** Correlated features (such as the five EXT_SOURCE
  variables) share credit in ways that depend on the model's splits.
- **Explanations are in log-odds.** One feature's effect on the *probability* depends on the
  other features, so per-feature probability effects aren't given.
- **Approximations in the counterfactuals:** no interest rate; goods price is scaled with the loan
  amount; only two levers; a discrete grid, so finer changes could exist between grid points.
- **Sample size:** the feasibility study uses 500 of 5,973 flagged test borrowers. Rates for the
  smaller groups (e.g. 68 borrowers at 40%+) are imprecise.

## References

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
  *Advances in Neural Information Processing Systems 30*.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global
  understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
- Wachter, S., Mittelstadt, B., & Russell, C. (2018). Counterfactual explanations without opening
  the black box: automated decisions and the GDPR. *Harvard Journal of Law & Technology*, 31(2).
