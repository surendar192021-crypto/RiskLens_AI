# Drift monitoring, fairness audit, model registry and audit log (Phase 10)

Run: `python -m scripts.run_monitoring` (about 50 s). Code: `src/monitoring/`. Settings:
`config.yaml → monitoring, fairness`. Numbers come from
`reports/experiment_results/phase10_monitoring_summary.json` and `reports/tables/drift_*.csv`,
`fairness_*.csv`.

## 1. Drift: is the current portfolio like the training data?

**This is an offline comparison, not live production monitoring.** The dataset has no dates,
so "current" here means `application_test`.

| Comparison | Reference | Compared with | Purpose |
|---|---|---|---|
| Main | Training split (184,505) | Portfolio (`application_test`, 48,744) | Has the population shifted? |
| Control | Training split | Test split (46,127) | Same population: shows what pure sampling noise looks like |

**Measure.** The Population Stability Index (PSI) bins the reference distribution (deciles for
numeric features, categories for categorical ones, plus an explicit "missing" bin):

> PSI = Σ (new% − ref%) · ln(new% / ref%)

0 means identical distributions. The thresholds are the common credit-scorecard rule of thumb
(Siddiqi, 2006): **below 0.10 stable, 0.10–0.25 moderate, above 0.25 major**. This is a
convention, not a statistical test. The KS statistic and missing-value rates are also reported.
P-values are not used to decide anything, because with this many rows even trivial shifts are
"significant".

**Results:**

| | Stable | Moderate | Major | Largest PSI |
|---|---:|---:|---:|---:|
| Control (train vs test) | 155 | 0 | 0 | 0.0015 |
| Main (train vs portfolio) | 147 | 3 | 5 | 1.17 |

**Largest shifts (main comparison):**

| Feature | PSI | Control PSI | SHAP rank (Phase 6) | What changed |
|---|---:|---:|---:|---|
| CREDIT_ANNUITY_RATIO | 1.17 | 0.0002 | 9 | Loan-length proxy distributed very differently (KS 0.32) |
| BUREAU_BB_WORST_STATUS_12M | 1.05 | 0.0000 | 60 | **Missing for 76% of training applicants vs 27% of the portfolio.** Bureau monthly-status data is filled in very differently |
| AMT_REQ_CREDIT_BUREAU_MON | 0.43 | 0.0001 | 116 | Credit-bureau enquiry counts |
| BUREAU_BB_DPD_CREDIT_COUNT | 0.41 | 0.0000 | 122 | Follows from the bureau-balance coverage change |
| AMT_REQ_CREDIT_BUREAU_QRT | 0.28 | 0.0000 | 54 | Credit-bureau enquiry counts |
| NAME_CONTRACT_TYPE | 0.21 | 0.0000 | 102 | Revolving loans: 9.5% of labelled data vs 0.9% of the portfolio (Phase 8) |
| DOCUMENT_COUNT | 0.15 | 0.0000 | 45 | |
| CREDIT_INCOME_RATIO | 0.11 | 0.0002 | 59 | |

Figure: `reports/figures/phase10_drift_top_features.png`.

**Model scores:**
- **PSI of the scores:** 0.0036 (stable). The control comparison gives 0.0003. Reference =
  the deployed model's scores on the 184,505 training applicants; comparison = its scores on
  the 48,744 portfolio applicants (10 decile bins of the reference). Training-set scores are
  in-sample (train ROC-AUC 0.868 vs test 0.788), so the audit also used the held-out
  validation split as reference: PSI 0.0040, the same conclusion. Train vs validation scores
  give 0.0003, so in-sample fitting barely changes the score distribution.
- **Mean PD:** 8.07% on training data vs 7.51% on the portfolio.
- **Risk bands:** the portfolio has slightly fewer Moderate and High borrowers (1.1% and 0.09%,
  vs 1.6% and 0.25% in training).

**Review rule (set in config before this run):** flag for review if score PSI ≥ 0.10, or if 3 or
more of the 20 most important features show major drift. **Not triggered**: score PSI is 0.004,
and only one important feature (CREDIT_ANNUITY_RATIO) has major drift.

**Interpretation:**
- **The portfolio is clearly a different population on several inputs.** The control comparison
  shows sampling noise is about 1,000 times smaller than these shifts, so they are real.
- **The model's scores are stable**, so the input shifts mostly offset each other or touch
  features with little influence on the model.
- **Score stability does not prove performance is unchanged.** The model's 9th most important
  feature has PSI 1.17, and the bureau data is filled in very differently.
- **The rule's cut-offs were fixed in advance.** They are deliberately *not* tightened after
  seeing these results, because that would be tuning the monitoring to the outcome. Even so, a
  human reviewer should note these shifts.
- **No retraining happens automatically**, whatever the drift result.

**Performance drift is not measurable on the portfolio**, because it has no outcome labels.
On labelled data the performance is stable across splits: validation ROC-AUC 0.784 and PR-AUC
0.268; test 0.788 and 0.290. A real deployment would need outcome labels as they arrive before
it could judge whether the model has degraded.

## 2. Fairness audit (labelled test split, deployed model, 16.2% threshold)

**Groups:**
- **Gender (`CODE_GENDER`):** F and M. Two borrowers have missing gender; they are reported but
  too few to interpret.
- **Age band:** derived from `AGE_YEARS`.

**What these attributes mean here:** they are present in the data and relevant to fairness. The
analysis **measures** disparities. It does not establish whether using them is lawful or
appropriate, which is a policy and legal question.

**Gender:**

| Group | n | Observed rate | Mean PD | Calibration gap | Flagged | TPR [95% CI] | FPR | Precision | AUC |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| F | 30,390 | 6.97% | 7.10% | +0.13 pp | 10.3% | 38.8% [36.7, 40.9] | 8.2% | 26.2% | 0.780 |
| M | 15,735 | 10.20% | 9.83% | −0.37 pp | 18.0% | 51.5% [49.1, 54.0] | 14.2% | 29.2% | 0.788 |

**Differences, M − F, with 95% CI:**
- flag rate +7.7 pp [7.0, 8.4]
- TPR +12.7 pp [9.5, 15.9]
- FPR +6.0 pp [5.4, 6.7]
- observed difficulty rate +3.2 pp [2.7, 3.8]

**Parity ratios (lowest ÷ highest):** flag rate 0.57, TPR 0.75, FPR 0.58, precision 0.90.

**Age:**

| Age band | n | Observed rate | Mean PD | Flagged | TPR [95% CI] | FPR | AUC |
|---|---:|---:|---:|---:|---|---:|---:|
| 18–25 | 1,752 | 12.8% | 12.7% | 27.2% | 54.2% [47.7, 60.6] | 23.2% | 0.725 |
| 25–35 | 10,767 | 10.6% | 10.5% | 20.1% | 52.7% [49.8, 55.6] | 16.2% | 0.782 |
| 35–45 | 12,609 | 8.0% | 8.3% | 13.5% | 46.0% [42.9, 49.0] | 10.7% | 0.793 |
| 45–55 | 10,577 | 7.3% | 7.2% | 10.3% | 40.9% [37.4, 44.4] | 7.9% | 0.796 |
| 55–65 | 9,179 | 5.7% | 5.5% | 5.8% | 27.0% [23.3, 30.9] | 4.5% | 0.750 |
| 65+ | 1,243 | 3.9% | 3.4% | 1.0% | 6.3% [2.2, 16.8] | 0.8% | 0.770 |

Figure: `reports/figures/phase10_fairness_age.png`.

**Findings:**
1. **The model is calibrated within groups.** In every group with 500+ borrowers, mean PD is
   within 0.5 pp of the observed rate.
2. **Error rates are not equal.** Men and younger applicants are flagged far more often. That
   comes with higher TPR (more of their difficulty cases caught), but also higher FPR (more of
   their non-difficulty cases flagged). When base rates differ, as here, a calibrated model at a
   single threshold *must* produce unequal error rates: calibration and error-rate parity cannot
   both hold (Kleinberg et al., 2017; Chouldechova, 2017). Choosing which criterion matters is a
   policy decision.
3. **Older applicants are almost never flagged.** Only 6% of difficulty cases among those 65+
   are caught (n = 48 cases, so the CI is wide), and the model slightly *under*-predicts their
   risk (−0.4 pp; Phase 8 saw the same).
4. **The model ranks less well within the youngest and the 55–65 groups** (AUC 0.725 and 0.750,
   against about 0.79 for the middle groups).
5. **On the "four-fifths" convention** (ratio at least 0.8, shown for reference only; it is not
   a legal test here): flag-rate ratios are 0.57 for gender and 0.04 for age. Much of the age
   ratio reflects real differences in observed difficulty rates (12.8% vs 3.9%).

**Does the model use gender directly?**

**Flip test:** change each test borrower's gender (F ↔ M), keep everything else, and rescore.

| Original gender | Mean PD change | Newly flagged | No longer flagged |
|---|---:|---:|---:|
| F → M | +1.35 pp | 897 | 42 |
| M → F | −1.22 pp | 33 | 548 |

**1,520 flag decisions (3.3% of the test split) change because of the gender value alone.**
That confirms the Phase 6 SHAP finding (gender ranks 5th).

**Experiment: retrain without `CODE_GENDER`.** Same parameters and early stopping (786 trees);
threshold re-chosen on validation (0.157). **Not deployed.**

| Test split | Deployed (with gender) | Without gender |
|---|---:|---:|
| ROC-AUC | 0.7877 | 0.7869 |
| PR-AUC | 0.2902 | 0.2878 |
| Brier | 0.0655 | 0.0657 |
| Flag rate F / M | 10.3% / 18.0% | 11.5% / 17.7% |
| TPR F / M | 38.8% / 51.5% | 41.0% / 50.8% |
| Calibration gap F / M | +0.13 / −0.37 pp | +0.35 / **−0.86 pp** |

**Interpretation:**
- **Accuracy:** removing gender costs very little (ROC-AUC −0.001). These differences are well
  within the test confidence intervals (about ±0.008), and no paired test was run.
- **Gaps narrow but remain:** the TPR gap falls from 12.7 to 9.8 pp. Other features act as
  proxies, and the base rates genuinely differ.
- **Calibration by gender gets worse:** without the variable, the model under-predicts men's
  risk by 0.86 pp. Removing a sensitive feature is not automatically "fairer". It moves the
  disparity from error rates to calibration.

**Decision:** the deployed model is unchanged. The audit documents the trade-off; choosing
between models is for a decision-maker with legal and policy input. **This audit does not
declare the model fair, and it does not establish legal or regulatory compliance.**

## 3. Model registry and audit log

**Registry: `models/registry.json`.**
- **What each entry records:** version, file, status, phase, training data, calibration,
  threshold, validation and test metrics, and the file's **SHA-256**.
- **Deployment rule:** registering a new deployed model automatically archives the previous one.
- **Current entries:** 7 in total.
  - deployed: `lightgbm-none-phase5_20261027_004048`
  - challenger: XGBoost
  - baseline: logistic regression
  - experiments: random forest, the two weighted models, and LightGBM without gender
- **Integrity check:** every registered file matches its recorded hash
  (`ModelRegistry.verify`).

**Audit log: `models/audit_log.jsonl`.**
- **Format:** append-only. Each entry stores the hash of the previous one, so editing or deleting
  a past entry breaks the chain, and `AuditLog.verify` reports where.
- **Events logged this run:** `registry_bootstrap`, `drift_check`, `fairness_audit`.
- **Chain status:** valid.
- **Limitation:** tampering is *detectable*, not *prevented*. Anyone who can write the file could
  rebuild the whole chain.

Both files live in `models/`, which is not in Git, alongside the model files they describe.

## 4. Limitations

- **Drift is an offline comparison** between two samples that were never time-ordered. The PSI
  cut-offs are conventions.
- **Fairness is measured on a test split from the training population.** On the shifted
  portfolio the error rates are unknown, because there are no labels.
- **Only two attributes were audited.** Others, such as family status or region, could be
  examined the same way.
- **Group differences use normal-approximation intervals**, which treat groups as independent
  samples.

## References

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit
  Scoring*. Wiley.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). Inherent trade-offs in the fair
  determination of risk scores. *Proceedings of Innovations in Theoretical Computer Science (ITCS)*.
- Chouldechova, A. (2017). Fair prediction with disparate impact: a study of bias in recidivism
  prediction instruments. *Big Data*, 5(2), 153–163.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning.
  *Advances in Neural Information Processing Systems 29*.
