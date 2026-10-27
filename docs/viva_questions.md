# Viva preparation: likely questions and evidence-based answers

Every answer points to where the evidence is. Numbers are from the project's own runs.

## Data and design

**Q1. What exactly does your model predict?**
The probability that `TARGET` = 1: per the dataset's documentation, a client who paid more than
X days late on at least one of the first Y instalments. X and Y are not disclosed. So this is
early payment difficulty in a sample, not legal default, and the sample's 8.07% rate isn't a real
portfolio's rate. (`docs/dataset_audit_findings.md` §2)

**Q2. Why a random split instead of a time-based one?**
The data has no calendar dates; every date is relative to the application. A time-based split is
impossible, so performance on future applicants is untested. Phase 10 also shows that the
unlabelled portfolio differs from the training data. (§7 of the audit findings; `docs/monitoring_and_fairness.md`)

**Q3. How did you prevent data leakage?**
- **Four splits, each with one job:** train fits, validation selects, calibration calibrates,
  test is used once.
- **No target-based features,** and the build checks that TARGET and the ID never enter the
  feature list (tested).
- **Every fitted step stays inside the training pipeline.**
- **No post-application events:** the audit showed none in the history tables, and 17
  impossible `DAYS_CREDIT_UPDATE` values were set to missing.

(`docs/dataset_audit_findings.md` §4)

**Q4. How did you join seven tables without multiplying rows?**
Every history table is aggregated to one row per applicant before joining, and a row-count
assertion runs after every join. 356,255 rows went in and 356,255 came out each time.
(`docs/feature_engineering.md`)

## Modelling

**Q5. Why LightGBM?**
- **Ranking:** best validation PR-AUC (0.268), against 0.246 for logistic regression.
- **Tie with XGBoost:** statistically tied with XGBoost (0.268, overlapping intervals). Faster
  training (21 s vs 119 s) and native SHAP support decided it.
- **Test confirmation:** on the test split it beats logistic regression on every metric. The
  paired-bootstrap intervals exclude zero: ROC-AUC +0.014 [0.011, 0.017].

(`docs/model_training.md`, `docs/calibration_and_evaluation.md`)

**Q6. On test, XGBoost was slightly better. Why not switch?**
The champion was fixed on validation before the test split was opened. Switching would be model
selection on test data, which biases the final estimate. The difference is also within noise
(paired PR-AUC difference −0.001 [−0.005, +0.003]).

**Q7. How did you handle the 11:1 class imbalance?**
- **Measured, not assumed:** class weighting left ranking unchanged (ROC-AUC 0.782 vs 0.784)
  but inflated mean PD from 8% to 36–40%.
- **No weighting, no SMOTE.** The imbalance is handled through PR-AUC-based selection and the
  threshold.

(`docs/model_training.md` §3)

**Q8. Why isn't accuracy reported?**
Predicting "no difficulty" for everyone scores 91.9%. We report ROC-AUC, PR-AUC, Brier score,
log loss and ECE, plus precision, recall and F1 at a validation-chosen threshold.

**Q9. Did calibration help?**
No. Sigmoid and isotonic calibration did not beat the raw model on validation (Brier 0.06646
raw, 0.06649 sigmoid, 0.06653 isotonic). The unweighted model optimises log-loss and was
already calibrated (ECE about 0.004). The tests show the calibration code does fix a deliberately
miscalibrated model, so the method works; it just isn't needed here. (`docs/calibration_and_evaluation.md` §2)

**Q10. What is the difference between discrimination and calibration?**
Discrimination is ranking: does a borrower with difficulties score higher than one without?
That's ROC-AUC and PR-AUC. Calibration is whether the numbers are right: do 20% of borrowers
given 20% actually have difficulties? That's the Brier score, ECE and reliability curves. A
model can rank well and still be badly calibrated; the weighted models were an example.

## Explainability

**Q11. How do you know your SHAP values are correct?**
- **Additivity is checked on every call:** base value + contributions = the model's raw score,
  with an error of about 1e-14.
- **The values equal LightGBM's own TreeSHAP** (`pred_contrib`) exactly, with a maximum
  difference of 0.0.

(`docs/explainability.md` §1)

**Q12. Does a large SHAP value mean the feature *causes* default?**
No. It shows what the model relied on. Phase 9 shows that the model's associations can be
non-causal; for example, a 20% income cut lowers the PD for 32% of borrowers.

**Q13. How are your counterfactuals constrained?**
- **What can change:** only the loan amount (down to 50%) and the repayment period (×0.75–1.5).
- **What is locked:** every other feature, checked in code.
- **Consistency:** ratios are recomputed with the dataset's own formulas, and new amounts must
  stay within the training range.
- **Scoring:** every candidate is scored by the deployed model.
- **Result:** feasible for 54% of 500 flagged borrowers, and 0% above 40% PD.

## Temporal, portfolio and simulation

**Q14. Where is your time-to-default analysis?**
The current loan has no event time or follow-up, so it isn't possible, and inventing dates would
fabricate results. Instead we studied time to the first 30+ day late payment on 495,681
*previous* loans. Its held-out Cox concordance is 0.728; the proportional-hazards assumption is
violated for two covariates, and we say so. (`docs/survival_analysis.md`)

**Q15. Is the sum of PDs the number of defaults you expect?**
It is the expected *number*, not a forecast of the exact count. On the test split it was 3,704
expected against 3,724 observed. The ±108 range assumes independence; correlation widens it.
(`docs/portfolio_risk.md` §4)

**Q16. Where does your LGD come from?**
It is assumed: there is no recovery data. 45% is the Basel II foundation-IRB reference value for
senior unsecured claims. EL is linear in LGD, so 25–85% gives 445M to 1,513M, and the documents
show this sensitivity. (`docs/portfolio_risk.md` §5)

**Q17. Why does your VaR change so much?**
It depends on the assumed correlation ρ. VaR 99.9% is 0.84B with ρ = 0 and 4.98B with ρ = 0.25,
while the mean stays at about 0.80B. Without data to estimate ρ, VaR is conditional on that
assumption. That is the lesson. (`docs/simulation_and_stress.md` §1)

**Q18. How did you check the Monte Carlo was right?**
- **Mean vs analytic EL:** within 1.3 standard errors.
- **Independence case:** the variance matches Σp(1−p) for ρ = 0.
- **Vasicek closed form:** a large uniform portfolio's tail quantiles match it within 6%.
- **Reproducibility:** the same seed gives identical results.

**Q19. Are your stress tests credible?**
- **Partly:** the external-score shock (+16% expected difficulties) behaves sensibly.
- **Income and instalment shocks do not:** the model lowers some PDs when income falls. We
  report that explicitly rather than calling the portfolio resilient.
- **The remedy:** monotonic constraints or a macro-calibrated satellite model (future work).

## Monitoring, fairness and engineering

**Q20. Has the population drifted?**
- **Control first:** train vs test is stable on all 155 features (largest PSI 0.0015), so the
  shifts below are real.
- **Portfolio vs training:** 5 features show major drift. `CREDIT_ANNUITY_RATIO` has PSI 1.17,
  and bureau monthly-status data is missing for 76% of training applicants vs 27% of the
  portfolio.
- **Model scores:** stable (PSI 0.004). That does not prove performance is unchanged; there are
  no labels.

**Q21. Is the model fair?**
We don't claim it is.
- **Calibrated within every group** (gaps under 0.5 pp).
- **Error rates differ:** men are flagged at 18.0% vs women 10.3%, and the TPR gap is +12.7 pp
  [9.5, 15.9].
- **Why both happen:** with different base rates (10.2% vs 7.0%), calibration and equal error
  rates can't both hold (Kleinberg et al., 2017; Chouldechova, 2017).
- **Direct use of gender:** the flip test shows the model uses it (1,520 flag changes).
- **Removing gender:** it barely changes accuracy and narrows the gaps, but worsens calibration
  for men. That trade-off is a policy decision.

**Q22. How can someone reproduce your results?**
Run `python -m scripts.verify_reproducibility`. It checks raw-file checksums, rebuilds the
dataset from raw data, retrains the champion from scratch, and recomputes the test metrics,
portfolio figures and Monte Carlo results. It compares all of them with the recorded values.
(`docs/final_report.md` §7)

**Q23. What is your actual contribution? Is it novel?**
No new algorithm is claimed. The contribution is an integrated, tested and reproducible pipeline
that links prediction to calibration, explanation, monitoring and portfolio risk on real data,
with every assumption made explicit. It also reports several negative or limiting findings
honestly: calibration was unnecessary; class weighting hurt the probabilities; the Markov and
proportional-hazards assumptions failed; stress responses were implausible.

**Q24. What would you do with more time or data?**
- **Time-stamped data with loan performance:** for temporal validation, real survival analysis
  and real drift monitoring.
- **Recovery data:** for LGD.
- **A more robust model:** monotonic constraints, plus a seed and hyper-parameter study.
- **Fairness:** mitigation, decided together with policy owners.
- **The optional API:** for integration with other systems.
