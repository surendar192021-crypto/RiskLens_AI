# Model training and comparison (Phase 4)

Run `phase4_20261027_002624` (`python -m scripts.train_models`, about 5.5 minutes). Every number
below comes from that run: `reports/tables/model_comparison_validation.csv` and
`reports/experiment_results/phase4_20261027_002624_summary.json`. **All results are on the
validation split (46,127 applicants, 3,724 positives).** The calibration and test splits were not
used in this phase.

## 1. Set-up

- **Data:** the training split is used for fitting (184,505 applicants, 155 features). Validation is
  used for comparison and for early stopping of the boosted models.
- **Shared schema guard:** every model starts with the same `FeatureFrame` step, which fixes the
  feature order and the category levels. Scoring goes through `src/models/predict.py`.
- **Preprocessing per model:**
  - Logistic regression: clip numeric values to the training 0.5–99.5% quantiles, fill missing
    values with the training median and add missing-value indicators, standardise, one-hot encode
    the categoricals (rare levels grouped).
  - Random forest: ordinal-encoded categoricals, missing values handled natively.
  - XGBoost and LightGBM: native categorical and missing-value support.
- **Hyper-parameters:** fixed, reasonable settings in `config/config.yaml`, plus early stopping
  for the boosted models. **There was no hyper-parameter search**; see the limitations below.
- **Confidence intervals:** 95% bootstrap intervals (300 resamples) of validation ROC-AUC and
  PR-AUC. They reflect sampling noise in the validation set only.

## 2. Results (validation)

| Experiment | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier | Log loss | ECE (10 equal-count bins) | Mean predicted | Fit time |
|---|---|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.766 [0.759, 0.774] | 0.246 [0.234, 0.259] | 0.0676 | 0.244 | 0.0040 | 0.081 | 15 s |
| logistic_regression_weighted | 0.767 [0.759, 0.775] | 0.242 [0.230, 0.256] | 0.1942 | 0.572 | 0.3235 | 0.404 | 24 s |
| random_forest | 0.764 [0.756, 0.772] | 0.243 [0.231, 0.256] | 0.0681 | 0.246 | 0.0104 | 0.081 | 82 s |
| xgboost (721 trees) | 0.783 [0.776, 0.791] | 0.268 [0.255, 0.283] | 0.0665 | 0.239 | 0.0033 | 0.080 | 119 s |
| **lightgbm (596 trees)** | **0.784 [0.776, 0.791]** | **0.268 [0.255, 0.282]** | **0.0665** | **0.239** | 0.0043 | 0.080 | 21 s |
| lightgbm_weighted (657 trees) | 0.782 [0.774, 0.789] | 0.267 [0.253, 0.282] | 0.1652 | 0.498 | 0.2762 | 0.357 | 19 s |

For reference, the validation base rate is 0.0807. A model that always predicts the base rate
has a PR-AUC of about 0.081 and a Brier score of about 0.074.

**At the max-F1 threshold, chosen on validation** (an optimistic, same-data choice, used only
for comparison):

| Experiment | Threshold | Precision | Recall | F1 | Share of borrowers flagged |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.146 | 0.240 | 0.454 | 0.314 | 15.3% |
| random_forest | 0.157 | 0.258 | 0.388 | 0.310 | 12.2% |
| xgboost | 0.166 | 0.276 | 0.426 | 0.335 | 12.4% |
| lightgbm | 0.162 | 0.270 | 0.429 | 0.331 | 12.8% |

Figures: `reports/figures/phase4_roc_validation.png`, `phase4_pr_validation.png` and
`phase4_reliability_validation.png`.

## 3. Findings (observed), with interpretation

1. **Boosted trees rank best.** XGBoost and LightGBM reach a PR-AUC of 0.268, against 0.246
   for logistic regression and 0.243 for the random forest. Their ROC-AUC is 0.783–0.784, against
   0.764–0.766. The intervals overlap only slightly. Because all models are scored on the same
   applicants, a paired bootstrap of the *difference* would be a stronger test; it is planned for
   the final test-set comparison.
2. **XGBoost and LightGBM are statistically indistinguishable here.** Their PR-AUC differs by
   0.0004, and their confidence intervals almost coincide.
3. **Class weighting did not help.** It left ranking unchanged or slightly worse (ROC-AUC
   0.767 vs 0.766 for logistic regression, 0.782 vs 0.784 for LightGBM). It also badly distorted
   the probabilities: mean predicted PD rose to 0.40 and 0.36, against a true rate of 0.081, and
   the Brier score roughly tripled. At the max-F1 threshold, weighted and unweighted LightGBM
   catch exactly the same number of defaulters (1,597); only the threshold moves.
   **Decision:** no class weighting. The imbalance is handled through threshold choice instead.
   No resampling (e.g. SMOTE) is used: it would distort probabilities in the same way, and it
   fits poorly with missing values and categorical features.
4. **The unweighted models are already well calibrated on validation.** Their quantile ECE is
   0.0033–0.0043, and 0.0104 for the random forest. The models that minimise log-loss produce
   probabilities whose average (0.080) matches the base rate. Phase 5 must check honestly whether
   calibration adds anything. A small or zero gain is a valid result.
5. **A stopping-rule problem was found and fixed.** In the first run (marked SUPERSEDED in the
   experiment log), weighted LightGBM stopped after 2 trees: weighting raises its log-loss from
   the start, so early stopping fired at once. Weighted models now stop on validation ROC-AUC.
   All six experiments were re-run, and the unweighted results came out identical.

### Accuracy, for reference only

Accuracy is not used to choose a model, because the data is imbalanced: 8.07% of applicants
have TARGET = 1, so predicting "no difficulty" for everyone is already **91.93%** accurate while
catching nobody. The table below answers "what is each model's accuracy?" next to that
baseline. It is computed from the saved validation predictions (46,127 applicants) by
`python -m scripts.accuracy_report`, which writes `reports/tables/model_accuracy_validation.csv`.

| Model | ROC-AUC | PR-AUC | Accuracy at 0.5 | Recall at 0.5 | Accuracy at own max-F1 threshold | Recall there |
|---|---|---|---|---|---|---|
| Baseline: predict no difficulty | — | — | 91.93% | 0% | 91.93% | 0% |
| Logistic regression | 0.766 | 0.246 | 91.97% | 3.0% | 83.95% (t = 0.146) | 45.4% |
| Logistic regression, weighted | 0.767 | 0.242 | 70.61% | 69.2% | 82.45% (t = 0.638) | 49.3% |
| Random forest | 0.764 | 0.243 | 91.93% | 0.0% | 86.02% (t = 0.157) | 38.8% |
| XGBoost | 0.783 | 0.268 | 91.98% | 4.4% | 86.37% (t = 0.166) | 42.6% |
| **LightGBM (chosen)** | **0.784** | **0.268** | **92.00%** | **4.5%** | **86.03% (t = 0.162)** | **42.9%** |
| LightGBM, weighted | 0.782 | 0.267 | 75.19% | 65.5% | 86.18% (t = 0.666) | 42.9% |

How to read it:
- **At the default 0.5 cut-off,** every unweighted model is within 0.1 points of the baseline,
  and it catches 0–4.5% of the applicants who had difficulty. The random forest flags nobody
  at all and still scores 91.93%. High accuracy here means almost nothing.
- **At a useful threshold,** accuracy falls to 82–86% because the model now flags borrowers
  for review, and some of them turn out fine. That is the intended trade: about 43% of
  difficulty cases are caught.
- **The weighted models** look worse at 0.5 only because their probabilities are inflated
  (about 5×), so 0.5 means something different for them.
- **On the test split** (`reports/tables/model_accuracy_test.csv`, from the saved confusion
  counts, threshold chosen on validation): LightGBM 86.13%, XGBoost 86.55%, logistic
  regression 84.34%, against a 91.93% baseline.

## 4. Champion and challenger

- **Champion: LightGBM.** It has the highest validation PR-AUC under the configured rule.
  Because its lead over XGBoost is within noise, the deciding practical reasons are:
  - it is equal on Brier score and log loss
  - it trains in 21 s against 119 s
  - SHAP supports it natively (Phase 6)
- **Why not accuracy:** see the table above. Every model "wins" on accuracy at 0.5 by catching
  almost no one; PR-AUC rewards the model that ranks the rare difficulty cases highest.
- **Why not the random forest or logistic regression:** both rank borrowers worse (ROC-AUC
  0.764 and 0.766 vs 0.784; PR-AUC 0.243 and 0.246 vs 0.268).
- **Why not the weighted versions:** same ranking, but probabilities several times too high
  (Brier 0.165–0.194 vs 0.067), which breaks expected loss and the risk bands.
- **Challenger: XGBoost.** It is the best model from a different family and stays available for
  the final comparison.
- **Baseline: logistic regression.** It is kept as the interpretable reference point.

## 5. Limitations

- **No hyper-parameter search.** Tuning could change the gaps between models, but small,
  measured differences matter more than a heavily tuned leaderboard here.
- **Validation is slightly optimistic for the boosted models,** because it also chose their
  number of trees. The untouched test split will give the unbiased final numbers (Phase 5).
- **Thresholds above were chosen on the same validation data they are reported on.** The
  operational threshold will be chosen on validation and reported on test.
- **One random split and one seed.** Variation across seeds was not measured.
- **Sample-specific calibration.** All probabilities are calibrated to this sample's 8.07%
  difficulty rate, not to any real portfolio.
