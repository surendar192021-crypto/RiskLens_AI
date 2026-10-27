# RiskLens AI: Loan Portfolio Risk Monitoring and Early Warning with Explainable Machine Learning

*Final project report. Integrated M.Sc. Computational Statistics and Data Analytics.*
*Every number in this report was produced by the project's code on the Home Credit Default Risk
data; the phase documents in `docs/` give full detail.*

## Abstract

RiskLens AI is a reproducible decision-support pipeline for consumer credit risk. It links
borrower-level default prediction to probability calibration, explanation, counterfactual
analysis, early warning, portfolio aggregation, loss simulation, drift monitoring and a fairness
audit, all on the Home Credit Default Risk data.

**Data and model.** The data has 307,511 labelled applicants (8.07% with early payment
difficulty), 48,744 unlabelled applicants, and six history tables. A gradient-boosted tree model
(LightGBM) reaches **test ROC-AUC 0.788 and PR-AUC 0.290**, against 0.774 and 0.268 for
logistic regression; the paired-bootstrap intervals exclude zero. The model's probabilities were
already well calibrated (ECE about 0.002 on test), so neither sigmoid nor isotonic calibration
was adopted.

**Explanations.** SHAP values were verified exact against LightGBM's own TreeSHAP. Constrained
counterfactuals, which may change only the loan's own terms, reached the review threshold for 54%
of flagged borrowers, and for none above 40% PD.

**Portfolio and simulation.** Summed PDs reproduced the observed number of difficulty cases on
the test split (3,704 expected vs 3,724 observed). Expected loss and Monte Carlo VaR are reported
as assumption-based, because LGD and default correlation are not observable here. The simulated
tail varies sixfold across plausible correlations.

**Negative findings, reported as results:**
- class weighting distorted probabilities
- the Markov assumption for delinquency migration failed
- the proportional-hazards assumption failed for two covariates
- the model responded implausibly to income and instalment stress
- error rates differ by gender and age even though the model is calibrated within groups

The project claims no new algorithm. Its contribution is an integrated, tested, reproducible
pipeline with explicit assumptions.

## 1. Introduction

**Beyond "who will default".** Credit-risk models are usually evaluated as classifiers. Risk
management also asks:
- **Reliability:** how reliable are the probabilities?
- **Explanation:** why is this borrower risky?
- **Hypotheticals:** what would change the assessment?
- **Portfolio:** what does borrower-level risk mean for the portfolio?
- **Downturns:** how bad could losses get?
- **Model health:** is the model still valid, and does it treat groups differently?

RiskLens AI builds and evaluates each of these on one real dataset. It separates what the data
supports from what has to be assumed or simulated.

**Not a lending authority.** All outputs are decision support. The prediction target is early
payment difficulty in a sample, not legal default.

## 2. Literature survey

- **Credit scoring benchmarks.** Statistical scoring is long established (Hand & Henley, 1997).
  Large benchmarks find ensembles, especially boosted trees, usually beat logistic regression,
  but the margins vary by dataset (Lessmann et al., 2015). The tree ensembles used here are
  random forests (Breiman, 2001), XGBoost (Chen & Guestrin, 2016) and LightGBM (Ke et al., 2017).
- **Evaluation under imbalance.** With rare positives, precision-recall analysis is more
  informative than ROC alone (Saito & Rehmsmeier, 2015). The Brier score (Brier, 1950) and
  expected calibration error (Guo et al., 2017) measure probability quality. Resampling such as
  SMOTE (Chawla et al., 2002) is common but changes the class balance the model learns.
- **Calibration.** Platt scaling (Platt, 1999) and isotonic regression (Zadrozny & Elkan, 2002)
  are standard. Niculescu-Mizil and Caruana (2005) show that boosted models can be miscalibrated,
  but that the benefit depends on the model and the loss it optimises.
- **Explainability.** SHAP provides additive feature attributions (Lundberg & Lee, 2017), with
  an exact polynomial-time algorithm for tree ensembles (Lundberg et al., 2020). Counterfactual
  explanations ask what minimal change would alter a decision (Wachter et al., 2018), and DiCE
  generates diverse counterfactuals (Mothilal et al., 2020). Actionability and plausibility
  constraints are central to both.
- **Survival analysis in credit.** Banasik, Crook and Thomas (1999) argued for modelling *when*
  borrowers default. The tools used here are the Kaplan–Meier estimator (Kaplan & Meier, 1958),
  the Cox model (Cox, 1972) and its proportional-hazards diagnostics (Grambsch & Therneau, 1994).
- **Portfolio credit risk.** The Vasicek one-factor model (Vasicek, 2002) underlies the Basel
  IRB formula (Gordy, 2003; BCBS, 2006). Markov chains are the classical model of rating
  migration (Jarrow, Lando & Turnbull, 1997). Population stability monitoring is standard
  scorecard practice (Siddiqi, 2006).
- **Fairness.** Equal error rates and calibration within groups are distinct criteria that
  generally cannot both hold when base rates differ (Kleinberg et al., 2017; Chouldechova, 2017).
  Hardt et al. (2016) formalise "equality of opportunity".

## 3. Research gap and contribution

Studies of the Home Credit data (and student projects in general) typically optimise one
discrimination metric. The established methods above are rarely combined in one pipeline,
evaluated against observed outcomes, with assumptions made explicit.

**The contribution here is integration and rigour, not a new method:**
- a single scoring path that all components share
- leakage-safe data splits, each with one purpose
- verification of each component against theory or outcomes
- reproducibility checks
- explicit reporting of what the data cannot support

## 4. Research questions (as revised after the data audit)

- **RQ1:** How do logistic regression, random forest and gradient boosting compare?
- **RQ2:** Does probability calibration improve the reliability of predicted PDs?
- **RQ3:** Can SHAP explain global and individual predictions faithfully?
- **RQ4:** Can constrained counterfactuals find feasible changes that alter predictions?
- **RQ5 (revised):** The data has no follow-up of current loans, so this became: what temporal
  risk patterns can be estimated from applicants' *pre-application* repayment histories?
- **RQ6:** How does an explicit portfolio simulation represent loss uncertainty and stress?

## 5. Data

**Home Credit Default Risk (Kaggle).** All 10 files were verified by checksum.

| Table | Rows |
|---|---:|
| application_train | 307,511 |
| application_test | 48,744 |
| bureau | 1,716,428 |
| bureau_balance | 27,299,925 |
| previous_application | 1,670,214 |
| POS_CASH_balance | 10,001,358 |
| installments_payments | 13,605,401 |
| credit_card_balance | 3,840,312 |

**What the audit established** (`docs/dataset_audit.md`, `docs/dataset_audit_findings.md`):
- **Clean keys.**
- **TARGET:** 8.07% positive.
- **Dates:** all relative to the application, with no post-application events (17 impossible
  values were set to missing).
- **Placeholders:** 18% of `DAYS_EMPLOYED` holds the 365243 placeholder, almost all for
  pensioners.
- **What is missing:** absolute dates, follow-up of the current loan, recoveries.

## 6. Methodology

1. **Features (155).** Application ratios (affordability, age, employment, external-score
   summaries), plus one-row-per-applicant aggregates of bureau, previous-application, instalment,
   POS and credit-card history with `HAS_*` indicators. Derived features come from shared
   functions, reused by the counterfactuals and stress tests.
2. **Splits.** Stratified 60/15/10/15 (train, validation, calibration, test), seed 42. Each split
   has one purpose.
3. **Models.** Logistic regression (with training-split clipping, imputation and scaling, and
   one-hot categories), random forest, XGBoost and LightGBM (with early stopping on validation).
   Class weighting was tested as an alternative. All share a schema-guard first step.
4. **Selection and calibration.** The champion is the one with the highest validation PR-AUC.
   Sigmoid and isotonic calibration are fitted on the calibration split on a frozen model and
   compared on validation. The threshold policy (max-F1) is chosen on validation.
5. **Explanation.** TreeSHAP with an additivity check on every call. Counterfactuals use a grid
   over the loan amount (×0.5–1.0) and repayment period (×0.75–1.5): all other features are
   locked, and plausibility bounds come from the training split.
6. **Temporal.** Kaplan–Meier and cluster-robust Cox models of time to the first 30+ day late
   payment on previous loans. The Markov transition matrix of the bureau's monthly overdue status
   is checked against observed 12-month transitions.
7. **Early warning.** PD levels derived from validation results; factual red flags; alert
   records with SHAP drivers; a de-duplicating alert history.
8. **Portfolio.** Counts, sum of PD, exposure and exposure-weighted PD kept distinct. EL =
   PD × LGD × EAD, with LGD assumed and EAD a proxy.
9. **Simulation.** Vasicek one-factor Monte Carlo, with VaR, ES and sensitivity to ρ. Stress
   tests perturb features and rescore through the model.
10. **Monitoring.** PSI per feature and on the model's scores, against a control comparison.
    Group fairness metrics with confidence intervals; a gender flip test; retraining without
    gender. A model registry with checksums, and a hash-chained audit log.

## 7. Experimental design and reproducibility

**Fixed settings.** Seed 42 and every parameter in `config/config.yaml`. Every run is logged in
`docs/experiment_log.md`, including superseded runs and the reason they were superseded.

**Tests.** The pytest suite covers:
- data validation, features and splits
- metrics, thresholds and calibration
- risk bands and expected loss
- the simulation, including its analytic checks
- drift, fairness, the registry and the audit log
- the dashboard, run headlessly

**End-to-end check.** `scripts/verify_reproducibility.py` re-derives the main results from the
raw data. Its outcome is in §7.1.

### 7.1 Reproducibility result

Run on 27 Oct 2026 (127 s): **28 of 28 checks passed** (`reports/tables/reproducibility_check.csv`).

| Area | What was re-derived | Result |
|---|---|---|
| Raw data | SHA-256 of all 10 files vs the Phase 2 audit | All identical |
| Dataset | Full rebuild from raw files into a temporary folder | Modelling table (307,511 × 157), portfolio (48,744 × 156), splits and feature list identical, including column types |
| Model | Deployed model file vs registry hash; test metrics recomputed | Identical (differences below 1e-12) |
| Retraining | Champion retrained from scratch with the recorded settings | Same 596 trees; test predictions identical (maximum difference 0.0) |
| Portfolio | Sum of PD and expected loss recomputed | Identical |
| Simulation | Baseline Monte Carlo re-run with seed 42 | Mean, VaR 99%, VaR 99.9% and ES 99% identical |
| Audit | Hash chain of the audit log | Valid |

**Note.** The verification first uncovered an environment problem. Installing `lifelines`
(which requires pandas < 3) had downgraded pandas from 3.1.0 to 2.3.3, and under 2.3.3 the
dataset build failed on a merge of compact integer keys. That was fixed (keys compared as
int64), a regression test that fails without the fix was added, and the full check above was
then run under the locked environment.

## 8. Results

### RQ1: Model comparison (test split, evaluated once)

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier |
|---|---|---|---:|
| Logistic regression | 0.774 [0.766, 0.783] | 0.268 [0.255, 0.281] | 0.0666 |
| XGBoost | 0.789 [0.782, 0.798] | 0.291 [0.276, 0.306] | 0.0655 |
| **LightGBM (deployed)** | **0.788 [0.781, 0.797]** | **0.290 [0.277, 0.305]** | **0.0655** |

**Paired differences.** LightGBM − logistic regression: ROC-AUC +0.014 [0.011, 0.017], PR-AUC
+0.023 [0.017, 0.030]. LightGBM − XGBoost: indistinguishable.

**Class weighting.** It did not improve ranking, and it inflated mean PD to 0.36–0.40 against a
base rate of 0.081.

**Operating point.** At the validation-chosen threshold of 0.162, the model flags 12.9% of test
borrowers, with precision 0.276 and recall 0.443.

### RQ2: Calibration

The raw LightGBM was already well calibrated:
- **Validation:** Brier 0.06646 raw, 0.06649 sigmoid, 0.06653 isotonic; quantile ECE 0.004.
- **Choice:** no calibrator was adopted.
- **Within bands:** predicted and observed rates match closely, for example 5.7% vs 5.7% in
  "Very Low" and 27.4% vs 27.8% in "Low".

So calibration did not change reliability here, because the unweighted log-loss-trained model
was already calibrated. Calibration does correct a miscalibrated model, as the tests show with
weighted models.

### RQ3: SHAP

**Verified exact:** the maximum difference from LightGBM's own TreeSHAP is 0, and the additivity
error is about 1e-14.

**Drivers:**
- **External scores dominate:** their mean has 8.9% of total |SHAP|.
- **Influence is spread:** the top 10 features hold 36%.
- **SHAP and split gain agree:** Spearman 0.95.
- **Exception:** `CODE_GENDER` ranks 5th by SHAP but 45th by gain.

Plain-English explanations state that contributions are not causes.

### RQ4: Counterfactuals

Of 500 flagged test borrowers, a feasible loan-term change reaching the threshold exists for
**54%**:
- **By original PD:** 87% for PD 16–25%, 21% for 25–40%, and 0% at or above 40%.
- **Typical change:** a 5% smaller loan and a 12.5% longer period.
- **A non-intuitive pattern:** some solutions shorten the loan, a learned association.

### RQ5: Temporal risk patterns

**Kaplan–Meier (495,681 previous loans, 11,325 events).** 93.7% had no 30+ day late payment two
years after the first instalment. Previous loans of applicants with current `TARGET` = 1 fared
worse: 90.6% vs 94.0% (log-rank p ≈ 10⁻⁵²).

**Cox model (199,999 loans).**
- **Held-out concordance:** 0.728.
- **Associations:** higher interest-rate groups, consumer loans, smaller loans and fewer planned
  payments go with earlier late payment.
- **Assumption failure:** the proportional-hazards assumption fails for planned payments and
  contract type.

**Delinquency migration (8.5M monthly transitions).**
- **Snowballing:** the chance of getting worse next month rises from 2% to 69% as delinquency
  deepens.
- **Markov failure:** projecting the 1-month matrix 12 months ahead understates how persistent
  120+ days overdue is (35% vs an observed 58%), so the time-homogeneous Markov assumption is
  rejected.

### RQ6: Portfolio simulation and stress

**Portfolio** (48,744 borrowers): mean PD 7.51%; expected difficulties 3,659; EL 800.9M at an
assumed 45% LGD (445M–1,513M for LGD 25–85%).

**Monte Carlo (ρ = 0.10, assumed).**
- **Check:** mean 809.1M against analytic 800.9M (1.3 standard errors).
- **Tail:** VaR 99.9% is 2.90B. It ranges 0.84B–4.98B for ρ 0–0.25.

**Stress tests.**
- **External scores −10%:** expected difficulties +16%.
- **Income −20%:** +2%, and 32% of borrowers' PD *falls*.
- **Instalments +15%:** −0.7%.
- **Combined, with LGD 55%:** EL +49%.

The tail risk is therefore set by the correlation assumption, and stress responses inherit the
model's non-causal associations.

### Early warning, monitoring and fairness

**Early warning.**
- **Levels on test:** observed difficulty rates of 4.0%, 13.8%, 24.5% and 48.6% for
  green, yellow, orange and red.
- **Red flags:** they add little beyond PD, so escalation is off by default.
- **Portfolio:** 10,642 alerts.

**Drift.**
- **Control comparison:** stable (largest PSI 0.0015).
- **Portfolio vs training:** 5 major-drift features, while the model's scores are stable
  (PSI 0.004).
- **Performance:** cannot be measured, because there are no labels.

**Fairness.**
- **Calibrated within groups:** gaps below 0.5 pp.
- **Gender:** flag rate M 18.0% vs F 10.3%; TPR gap +12.7 pp [9.5, 15.9].
- **Age:** flag rate ranges 27.2% (under 25) to 1.0% (65+).
- **Flip test:** changing gender alone changes 1,520 test flags.
- **Retraining without gender:** ROC-AUC 0.787 vs 0.788; the error-rate gaps narrow but men's
  calibration worsens.

## 9. Discussion

- **Gradient boosting helps, modestly.** The ROC-AUC gain over logistic regression is real but
  moderate (+0.014), in line with benchmark studies that find dataset-dependent gains.
- **The most useful findings are about limits:**
  - **Correlation:** without data on default correlation, VaR is an assumption.
  - **Causality:** a model fitted for prediction is not a causal model, so feature-perturbation
    stress tests and counterfactuals can mislead.
  - **Markov chains:** they understate how persistent delinquency is.
  - **Fairness:** a calibrated model can still produce unequal error rates.
- **Every one of these was measured, not assumed.** Several would have been hidden by reporting
  a single AUC.

## 10. Conclusions

- **Prediction:** LightGBM is the best-performing and best-calibrated model here.
- **Calibration:** it was unnecessary for the unweighted model.
- **Explanations:** they are exact and verified.
- **Counterfactuals:** they help only near the threshold.
- **Temporal analysis:** pre-application repayment timing carries risk information.
- **Portfolio aggregates:** they match observed outcomes.
- **Loss tails and stress results:** they are only as credible as their assumptions and the
  model's learned associations.

RiskLens AI shows how to deliver these analyses with their assumptions visible and their limits
measured.

## 11. Limitations and future work

All limitations are listed in `docs/assumptions_and_limitations.md`.

**Main limitations:**
- **The target:** early payment difficulty, not legal default.
- **No time ordering:** no temporal validation, no current-loan survival analysis, no real
  deterioration monitoring.
- **Missing inputs:** LGD and correlation are assumed.
- **A non-causal, non-monotone model.**
- **One seed and split, and no hyper-parameter search.**

**Future work:**
- **Data:** time-stamped loan-performance data, for temporal validation and real survival
  analysis.
- **Model:** monotonic constraints, and uncertainty across seeds.
- **Loss parameters:** recovery data for LGD, and macro-calibrated stress tests.
- **Fairness:** mitigation, decided with policy owners.
- **Integration:** the optional API.

## References

- Banasik, J., Crook, J. N., & Thomas, L. C. (1999). Not if but when will borrowers default.
  *Journal of the Operational Research Society*, 50(12), 1185–1190.
- Basel Committee on Banking Supervision (2006). *International Convergence of Capital
  Measurement and Capital Standards: A Revised Framework*. Bank for International Settlements.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly
  Weather Review*, 78(1), 1–3.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic
  minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Chen, T., & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *Proceedings of the
  22nd ACM SIGKDD Conference*, 785–794.
- Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data*, 5(2), 153–163.
- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical
  Society: Series B*, 34(2), 187–220.
- Davidson-Pilon, C. (2019). lifelines: survival analysis in Python. *Journal of Open Source
  Software*, 4(40), 1317.
- Gordy, M. B. (2003). A risk-factor model foundation for ratings-based bank capital rules.
  *Journal of Financial Intermediation*, 12(3), 199–232.
- Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on
  weighted residuals. *Biometrika*, 81(3), 515–526.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural
  networks. *Proceedings of the 34th ICML*.
- Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit
  scoring: a review. *Journal of the Royal Statistical Society: Series A*, 160(3), 523–541.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning.
  *Advances in Neural Information Processing Systems 29*.
- Jarrow, R. A., Lando, D., & Turnbull, S. M. (1997). A Markov model for the term structure of
  credit risk spreads. *Review of Financial Studies*, 10(2), 481–523.
- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations.
  *Journal of the American Statistical Association*, 53(282), 457–481.
- Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: a highly efficient gradient boosting
  decision tree. *Advances in Neural Information Processing Systems 30*.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). Inherent trade-offs in the fair
  determination of risk scores. *Proceedings of ITCS 2017*.
- Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art
  classification algorithms for credit scoring: an update of research. *European Journal of
  Operational Research*, 247(1), 124–136.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
  *Advances in Neural Information Processing Systems 30*.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global
  understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
- Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through
  diverse counterfactual explanations. *Proceedings of FAT\* 2020*, 607–617.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised
  learning. *Proceedings of the 22nd ICML*, 625–632.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to
  regularized likelihood methods. In *Advances in Large Margin Classifiers*, MIT Press.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC
  plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
- Vasicek, O. (2002). The distribution of loan portfolio value. *Risk*, 15(12), 160–162.
- Wachter, S., Mittelstadt, B., & Russell, C. (2018). Counterfactual explanations without opening
  the black box. *Harvard Journal of Law & Technology*, 31(2), 841–887.
- Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass
  probability estimates. *Proceedings of the 8th ACM SIGKDD Conference*, 694–699.

*Note: these are established, real publications. Before submission, confirm the exact volume
and page details against the publishers' records or Google Scholar. They were written from
knowledge, not checked online during this project.*
