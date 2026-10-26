# Probability calibration and final test evaluation (Phase 5)

Run `phase5_20261027_004048` (`python -m scripts.calibrate_and_evaluate`, about 1 minute).
The numbers come from `reports/tables/calibration_comparison_validation.csv`,
`threshold_options_validation.csv`, `final_test_metrics.csv`, `paired_differences_test.csv`
and `risk_band_distribution_test.csv`.

## 1. Protocol (each split used for one purpose only)

| Split | Rows | Used for |
|---|---:|---|
| calibration | 30,752 | Fitting the sigmoid and isotonic maps (the model is frozen, never refitted) |
| validation | 46,127 | Choosing the calibration method and the operating threshold |
| test | 46,127 | The final numbers below. Opened once; nothing was chosen on it |

Two background terms:
- **Discrimination** asks whether the model ranks borrowers who have difficulties above those
  who don't (ROC-AUC, PR-AUC).
- **Calibration** asks whether the probabilities are numerically right: of the borrowers given
  20%, do about 20% have difficulties? (Brier score, ECE, reliability diagrams.)

A model can be strong on one and weak on the other, so the two are evaluated separately.

## 2. Calibration result: none was selected

| Method (LightGBM, validation) | ROC-AUC | PR-AUC | Brier | Log loss | ECE (equal-count) | ECE (equal-width) |
|---|---:|---:|---:|---:|---:|---:|
| **none** | 0.7842 | 0.2681 | **0.06646** | **0.23871** | 0.0043 | 0.0028 |
| sigmoid (Platt) | 0.7842 | 0.2681 | 0.06649 | 0.23876 | 0.0049 | 0.0037 |
| isotonic | 0.7838 | 0.2581 | 0.06653 | 0.23892 | 0.0046 | 0.0042 |

**Observed:** neither calibration method improved any probability metric on validation. Isotonic
regression also lowered PR-AUC (0.268 → 0.258), because its step function creates tied scores.

**Interpretation:** the unweighted LightGBM is trained to minimise log-loss, a "proper" scoring
rule that rewards correct probabilities. Here that already left it well calibrated (ECE about
0.004). A calibrator fitted on 30,752 rows can only add noise when there is so little
miscalibration to fix.

**Decision:** the deployed model is the **uncalibrated** LightGBM. The comparison stays in the
pipeline and is re-run whenever the model changes.

**Contrast:** the class-weighted models from Phase 4 had a mean PD of about 0.36–0.40 against a
true rate of 0.081. The unit tests show calibration correcting exactly that kind of distortion
(`tests/test_calibration.py`), so the method works; it just isn't needed for this model.

This answers **RQ2** for this dataset: calibration did not measurably change reliability,
because the unweighted gradient-boosting model was already well calibrated.

## 3. Operating threshold (chosen on validation)

| Policy | Threshold | Precision | Recall | Share flagged |
|---|---:|---:|---:|---:|
| **max F1 (selected, config `thresholds.policy`)** | **0.162** | 0.270 | 0.429 | 12.8% |
| min cost, FN = 5 × FP (**assumed** cost ratio) | 0.161 | 0.269 | 0.432 | 13.0% |
| recall ≥ 0.60 | 0.105 | 0.210 | 0.600 | 23.0% |

The threshold is a *policy* choice, not a property of the model. Catching 60% of difficulty
cases instead of 43% means flagging almost twice as many borrowers. The 5:1 cost ratio is an
illustrative assumption, not a real bank figure.

## 4. Final test results

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier | ECE (equal-count) | Mean PD |
|---|---|---|---:|---:|---:|
| logistic_regression (baseline) | 0.774 [0.766, 0.783] | 0.268 [0.255, 0.281] | 0.0666 | 0.0022 | 0.0808 |
| xgboost (challenger) | 0.789 [0.782, 0.798] | 0.291 [0.276, 0.306] | 0.0655 | 0.0032 | 0.0799 |
| **lightgbm (champion, deployed)** | **0.788 [0.781, 0.797]** | **0.290 [0.277, 0.305]** | **0.0655** | **0.0022** | 0.0803 |

The test base rate is 0.0807.

**At the validation-chosen threshold of 0.162, the champion on test:**
- precision 0.276, recall 0.443, F1 0.340
- 12.9% of borrowers flagged
- TP 1,649 · FP 4,324 · FN 2,075 · TN 38,079

**Paired bootstrap differences on test** (same borrowers, 300 resamples, 95% CI):

| Comparison | ROC-AUC difference | PR-AUC difference | Brier difference |
|---|---|---|---|
| LightGBM − logistic regression | +0.014 [+0.011, +0.017] | +0.023 [+0.017, +0.030] | −0.0011 [−0.0014, −0.0009] |
| LightGBM − XGBoost | −0.001 [−0.003, +0.001] | −0.001 [−0.005, +0.003] | +0.0001 [−0.0001, +0.0002] |

**Findings:**
1. **The champion beats the baseline on every metric, and the result is robust.** All three
   paired intervals exclude zero, and LightGBM was better in 100% of resamples. This answers
   **RQ1**: gradient boosting outperforms logistic regression and the random forest here, but
   the margin is moderate (ROC-AUC +0.014), not dramatic.
2. **LightGBM and XGBoost can't be told apart.** Every interval contains zero. On test,
   XGBoost is very slightly ahead. The champion was fixed on validation before the test split
   was opened, and switching now would mean choosing on test data, so LightGBM stays.
3. **Test scores are a little higher than validation scores** (PR-AUC 0.290 vs 0.268). The
   intervals overlap, which is consistent with sampling variation between two random splits of
   46k borrowers. There's no sign of over-fitting to validation.

## 5. Provisional risk bands on the test split (champion)

| Band | PD range | Borrowers | Share | Mean predicted PD | Observed TARGET rate |
|---|---|---:|---:|---:|---:|
| Very Low | [0, 0.2) | 41,926 | 90.9% | 0.057 | 0.057 |
| Low | [0.2, 0.4) | 3,430 | 7.4% | 0.274 | 0.278 |
| Moderate | [0.4, 0.6) | 658 | 1.4% | 0.472 | 0.451 |
| High | [0.6, 0.8) | 110 | 0.2% | 0.663 | 0.682 |
| Critical | [0.8, 1.0] | 3 | 0.01% | 0.834 | 3 of 3 |

**Observed:**
- In every band with enough borrowers, the mean predicted PD matches the observed rate within
  about 2 percentage points. This is further evidence that the probabilities are calibrated.
- As expected from the Phase 2 audit, **90.9% of borrowers fall in "Very Low"**.
- The "Critical" band has only 3 borrowers, too few for any conclusion.

**Implication (to decide before the dashboard phase):**
- The fixed 20%-wide bands are easy to explain, but they barely separate this portfolio.
- The alternative is data-driven bands, such as PD quantiles or multiples of the base rate.
  These would spread borrowers more evenly but are less intuitive.
- The bands are configurable either way.

## 6. Deployed model

`models/current_model.json` points to the model that every later phase uses:
- **Model:** `lightgbm`, uncalibrated, from Phase 4 run `phase4_20261027_002624`.
- **Threshold:** 0.1623 (max-F1, chosen on validation).
- **Version string:** `lightgbm-none-phase5_20261027_004048`.

## 7. Limitations

- **One random split and one seed.** The results describe this sample, and its 8.07% rate is
  not a real portfolio's default rate.
- **No time ordering.** The split is random, so performance on *future* applicants is untested
  (see the Phase 2 audit).
- **Probabilities are calibrated on average and within the bands with enough data.** At the
  rare high end (PD above 0.6; only 113 test borrowers) the estimates are much less certain.
- **The threshold metrics depend on the assumed policy** (max-F1). A real lender would set the
  policy from its own costs and capacity.
