# Feature engineering and modelling table (Phase 3)

Built by `python -m scripts.build_dataset` (about 75 s). Code: `src/data/preprocess.py`,
`src/data/feature_engineering.py`, `src/data/splits.py`, `src/data/build_dataset.py`.
Numbers below come from the run on 27 Oct 2026 (`reports/experiment_results/build_summary.json`).

## 1. Pipeline

```
application_train + application_test (356,255 rows)
  → clean_application        placeholders → missing + flag; XNA/Unknown → missing
  → add_application_features ratios, age/employment years, EXT_SOURCE summaries
  → drop 28 redundant building columns (_MEDI/_MODE duplicates of _AVG)
  → join 5 history aggregates (each exactly one row per SK_ID_CURR)
  → split: modelling_train (307,511, with TARGET) / portfolio_test (48,744, no TARGET)
```

Cleaning and aggregation use **no target information and no statistics fitted across rows**,
so it is safe to run them on train and test together. Everything that is *fitted* (imputation,
scaling, encoding, clipping outliers) happens inside the model pipeline in Phase 4, on the
training split only.

## 2. Observed build results

| Check | Result |
|---|---|
| Rows after every join | 356,255 (unchanged) |
| Features | 155 (139 numeric, 16 categorical); ID and TARGET excluded and checked by test |
| Infinite values | none (checked during build and by test) |
| Dropped building columns | 28; minimum correlation with their `_AVG` column on the training split: 0.963 |

| History source | Applicants with aggregate rows | Training applicants covered | Features |
|---|---:|---:|---:|
| bureau + bureau_balance | 305,811 | 85.7% | 14 |
| previous_application | 338,857 | 94.6% | 11 |
| installments_payments | 339,587 | 94.8% | 9 |
| POS_CASH_balance | 337,252 | 94.1% | 6 |
| credit_card_balance | 103,558 | 28.3% | 7 |

## 3. Splits (stratified on TARGET, seed 42)

| Split | Rows | TARGET rate | Use |
|---|---:|---:|---|
| train | 184,505 | 8.072% | fit models |
| validation | 46,127 | 8.073% | compare models, tune, choose thresholds |
| calibration | 30,752 | 8.074% | fit the probability calibrator only |
| test | 46,127 | 8.073% | final evaluation, once |

## 4. Design decisions

- **Partial payments combined.** Several `installments_payments` rows can belong to one scheduled
  instalment. They're combined first: the amounts paid are summed, and lateness is measured from
  the last payment. Lateness and underpayment are then counted once per instalment.
- **Counts vs other aggregates.** `*_COUNT` features are 0 for applicants with no records,
  because 0 is the true count. All other history features stay missing, and the `HAS_*` flags
  say whether the source had any records. This refines the Phase 2 note, which said "missing
  aggregates are left missing".
- **Recent windows.** "Last year" means `DAYS_* >= -365`, or `MONTHS_BALANCE >= -12` for
  monthly tables. These windows are fixed choices, not tuned.
- **Bureau amounts.** Only "currency 1" rows go into amount sums. Credits in other currencies
  still count towards the credit counts.
- **Bureau statuses.** Status C (closed) and X (unknown) are not DPD buckets, so they are
  excluded from the worst-status feature.
- **Raw application columns are kept** alongside the engineered ones. Phase 4 can test whether a
  smaller feature set does as well.

## 5. Univariate screening (training split only)

ROC-AUC of each numeric feature on its own, using its non-missing training rows. Values far from
0.5 in either direction show stronger ranking signal. **This is a screening statistic, not a
model result.** Full table: `reports/tables/feature_summary.csv` and
[data_dictionary.md](data_dictionary.md).

| Feature | Univariate AUC |
|---|---:|
| EXT_SOURCE_MEAN | 0.285 |
| EXT_SOURCE_MIN | 0.311 |
| EXT_SOURCE_3 | 0.320 |
| EXT_SOURCE_1 | 0.333 |
| EXT_SOURCE_2 | 0.347 |
| CC_UTILISATION_MEAN_LAST_YEAR | 0.635 |
| CC_UTILISATION_MEAN | 0.625 |
| AGE_YEARS | 0.416 |
| EMPLOYED_YEARS | 0.416 |
| BUREAU_ACTIVE_DEBT_RATIO | 0.587 |

Interpretation, with limitations:
- **External scores:** higher values go with a lower TARGET rate (AUC below 0.5). They are the
  strongest single signals. The data doesn't say how they are built, which limits how far they
  can be explained.
- **Card utilisation:** its AUC is measured only on the 28% of applicants who have card history,
  so it is not directly comparable with features that are available for everyone.
- **Univariate is not multivariate:** a feature with a weak AUC on its own can still help a
  model in combination with others.

## 6. Limitations

- **Pre-application history only.** The history features describe behaviour before the
  application. They cannot capture what happens to the current loan.
- **Approximate loan-length proxy.** The unit of `AMT_ANNUITY` isn't documented, so
  `CREDIT_ANNUITY_RATIO` is a rough proxy for loan length, not a number of months.
- **Shared aggregates are not leakage.** The history aggregates for train and test are built in
  one pass. That doesn't leak labels, since no target is used. It does mean both share the same
  feature definitions, which is what we want for drift comparison later.
