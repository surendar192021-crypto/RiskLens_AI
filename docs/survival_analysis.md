# Survival analysis of previous loans (Phase 7a)

Run: `python -m scripts.run_survival` (about 21 minutes, mostly the cluster-robust Cox fit). Code:
`src/survival/`. Numbers come from `reports/experiment_results/phase7_survival_summary.json`,
`reports/tables/survival_km_horizons.csv`, `survival_cox_summary.csv` and `survival_ph_test.csv`.

## 1. Why this study, and not time-to-default for current applicants

The Phase 2 audit confirmed that **the current loan has no event time and no follow-up**:
`TARGET` covers "the first Y instalments", and Y is not disclosed. Survival analysis on `TARGET`
would need invented event dates, so it is not done.

`installments_payments` does record, for **previous** Home Credit loans, when each instalment
was due and when it was paid. That supports a genuine time-to-event study with right-censoring.

| Design element | Choice |
|---|---|
| Unit | One previous Home Credit loan (`SK_ID_PREV`) |
| Population | Previous loans of **training-split** applicants only. The test split is never used |
| Time origin | The loan's first scheduled instalment. Loans whose instalment 1 is not in the data (schedule started before the 8-year window) are excluded, to avoid left truncation |
| Event | First instalment paid **more than 30 days late**. Event time = its due day + 30 |
| Censoring | No event: censored at the due day of the last paid instalment (the loan finished, or the data ends at the current application) |
| Unpaid instalments | Not counted as events: their timing is unknown, and many are due just before the data ends |
| Dependence | Applicants have several loans (median 2), so Cox standard errors are cluster-robust by applicant |

S(t) here means the **probability that a previous loan has not yet had a 30+ day late payment, t
days after its first instalment**. It is not the current applicant's default probability and not
the model's PD.

## 2. Study population

| | Value |
|---|---:|
| Previous loans with instalment records (all applicants) | 997,752 |
| Loans in the study (training applicants, schedule start observed, at least one paid instalment) | 495,681 |
| Applicants | 173,858 |
| Events (first 30+ day late payment) | 11,325 (2.3% of loans) |
| Median follow-up | 240 days |

## 3. Kaplan-Meier estimates (share with no 30+ day late payment yet)

| Group | Loans | Events | S(90 d) | S(180 d) | S(365 d) | S(730 d) [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| All loans | 495,681 | 11,325 | 0.997 | 0.988 | 0.961 | 0.937 [0.936, 0.939] |
| Cash loans | 147,079 | 2,182 | 0.998 | 0.993 | 0.982 | 0.964 [0.962, 0.966] |
| Consumer loans | 313,337 | 7,640 | 0.997 | 0.985 | 0.934 | 0.803 [0.779, 0.825] |
| Revolving loans | 31,976 | 1,298 | 0.996 | 0.991 | 0.981 | 0.962 [0.959, 0.964] |
| Applicant's current TARGET = 0 | 457,587 | 10,046 | 0.997 | 0.988 | 0.962 | 0.940 [0.938, 0.942] |
| Applicant's current TARGET = 1 | 38,094 | 1,279 | 0.996 | 0.983 | 0.944 | 0.906 [0.898, 0.913] |

Figures: `reports/figures/phase7_km_overall.png` and `phase7_km_by_target.png`.

**Observed:**
- **Late payments build up gradually.** The share without a 30+ day late payment falls from
  99.7% at 90 days to 93.7% at two years.
- **Consumer loans** fall fastest after the first year. Their two-year estimate is imprecise
  (wide CI), because few consumer loans last that long, so few remain under observation.
- **Applicants who later had current-loan difficulties (`TARGET` = 1)** had previous loans that
  became 30+ days late sooner: at two years, 90.6% had no late payment against 94.0%. The
  log-rank test strongly rejects equal curves (statistic 232.7, p ≈ 1.6 × 10⁻⁵²).
- **Caution about the test:** loans of the same applicant are not independent, so this p-value
  is somewhat overstated. The size of the gap is the more useful information.

## 4. Cox proportional-hazards model

- **Data:** fitted on 199,999 loans (74,961 applicants, kept together by applicant).
- **Covariates:** characteristics of each previous loan known when it was granted: contract
  type, interest-rate group, log credit amount, planned number of payments.
- **Reference levels:** cash loans and the "high" interest-rate group. Revolving loans drop out
  because their rate group and planned payments are recorded as unknown.

| Covariate | Hazard ratio [95% CI, cluster-robust] | p |
|---|---|---:|
| Consumer loan (vs cash) | 1.16 [1.06, 1.27] | 0.0009 |
| Interest group "low_action" (vs high) | 0.33 [0.27, 0.39] | <0.001 |
| Interest group "low_normal" (vs high) | 0.61 [0.56, 0.67] | <0.001 |
| Interest group "middle" (vs high) | 0.67 [0.62, 0.72] | <0.001 |
| log(credit amount), per unit | 0.71 [0.68, 0.75] | <0.001 |
| Planned number of payments, per payment | 0.89 [0.87, 0.90] | <0.001 |

Figure: `reports/figures/phase7_cox_hazard_ratios.png`.

**Discrimination:**
- **Concordance:** 0.729 on the fitting loans and **0.728 on 200,880 loans of different
  applicants**, so there is no sign of over-fitting.
- **Meaning:** the model ranks loans by time-to-late-payment far better than chance (0.5), using
  only four loan characteristics.

**Interpretation (associations, not causes):**
- **Interest rate:** high-rate loans have the highest hazard. The bank's pricing presumably
  already reflects perceived risk, so the rate group partly *encodes* the bank's own risk
  assessment.
- **Loan size:** larger loans have lower hazards, perhaps because larger loans go to applicants
  who passed stricter checks. This is selection, not an effect of lending more.

## 5. The proportional-hazards assumption: partly violated

The test uses scaled Schoenfeld residuals on rank-transformed time, refitted on 20,000 loans:

| Covariate | Test statistic | p |
|---|---:|---:|
| Planned number of payments | 23.2 | <0.001 |
| Consumer loan | 8.2 | 0.004 |
| log(credit amount) | 0.12 | 0.73 |
| Interest groups (3 terms) | 0.13–1.54 | 0.21–0.72 |

**Finding:** the assumption holds for credit amount and interest group. It is **violated for
planned number of payments and contract type**, whose effects change over time. For example, a
short loan cannot have a late payment after it ends, so loan length interacts with time itself.
Their hazard ratios above are best read as *averages over follow-up time*.

**Possible remedies (not implemented):**
- stratify the model by contract type
- add an interaction between time and the number of payments
- report hazard ratios within time windows

## 6. What this answers, and what it doesn't

This answers **revised RQ5** for the data available:
- Time to first serious late payment on previous loans can be estimated properly, with censoring.
- It differs clearly by loan type and pricing.
- It is shorter for applicants who later had difficulties on their current loan.

So **pre-application repayment timing carries real risk information.** The model already uses it
through the instalment features (Phase 3).

**What it does not answer:**
- **The current loan's time-to-default:** that is unsupported by this dataset.
- **Survival curves for the borrowers on the dashboard:** the dashboard must not show any.
  Section 1 explains why.

## 7. Limitations

- **Selection:** previous loans appear only for applicants who came back to apply again, so they
  are not a random sample of Home Credit loans.
- **Censoring:** it is assumed to be independent of risk. Loans that end early, for example by
  early repayment, may differ from those that run their full term.
- **Event definition:** "30+ days late on one instalment" is a delinquency measure, not legal
  default. Unpaid instalments are excluded from events.
- **Time-varying effects:** the proportional-hazards assumption fails for two covariates (section 5).
- **Statistical caution:** the log-rank p-value ignores clustering by applicant, while the Cox
  intervals do account for it.

## References

- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations.
  *Journal of the American Statistical Association*, 53(282), 457–481.
- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical
  Society: Series B*, 34(2), 187–220.
- Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on
  weighted residuals. *Biometrika*, 81(3), 515–526.
- Davidson-Pilon, C. (2019). lifelines: survival analysis in Python. *Journal of Open Source
  Software*, 4(40), 1317.
