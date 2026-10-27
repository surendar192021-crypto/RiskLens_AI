# Early-warning engine (Phase 7b)

Run: `python -m scripts.run_early_warning` (about 15 s). Code: `src/risk/early_warning.py`, with
settings in `config.yaml → early_warning`. Numbers come from
`reports/experiment_results/phase7_early_warning_summary.json` and `reports/tables/early_warning_*.csv`.

## 1. What an alert is

An alert says **"this borrower should be reviewed"**. It is never a lending decision. Every
alert contains:
- borrower ID
- PD
- risk band
- alert level
- triggering conditions, in plain words
- the top 3 SHAP features pushing that borrower's estimate up
- observation time
- model version
- recommended review action
- a stable `alert_key`

**Levels** come from the deployed model's probability. Each threshold traces back to an
earlier, measured result:

| Level | PD at or above | Where the threshold comes from |
|---|---:|---|
| Yellow (watch) | 10.5% | Validation threshold that catches 60% of difficulty cases (Phase 5) |
| Orange (manual review) | 16.2% | Operating threshold, max-F1 on validation (`models/current_model.json`) |
| Red (senior review) | 40% | Observed difficulty rate 45–68% in the PD ≥ 40% bands (Phase 5). No change to loan terms reaches the threshold there (Phase 6) |

**Red flags** are factual signals from the borrower's pre-application history. A missing value
never raises a flag. 30+ days past due is the usual delinquency cut-off:
- an amount currently overdue on a credit bureau loan
- 31+ days past due on a bureau loan in the last 12 months
- 30+ days past due on a previous POS/cash loan in the last 12 months
- 2 or more refused applications in the last year
- average credit-card utilisation above 90% in the last year

**Observation time:** the dataset has no calendar dates. Every alert says "application date
(day 0 of the dataset)" plus the scoring timestamp, rather than pretending to a real date.

## 2. Are the rules informative? (measured on labelled data)

**Red flags, validation split.** The base difficulty rate is 8.07%.

| Flag | Prevalence | Difficulty rate if flagged | Lift | Rate among PD-green borrowers: flagged vs not |
|---|---:|---:|---:|---|
| Bureau amount overdue now | 1.1% | 17.8% | 2.20 | 7.1% vs 4.2% |
| Bureau 31+ DPD, last 12 months | 0.8% | 14.1% | 1.75 | 8.6% vs 4.2% |
| POS 30+ DPD, last 12 months | 0.3% | 17.0% | 2.11 | 6.4% vs 4.2% |
| 2+ refusals, last year | 7.3% | 13.5% | 1.67 | 5.8% vs 4.1% |
| Card utilisation above 90% | 2.6% | 16.6% | 2.06 | 6.2% vs 4.2% |

**Finding:** every flag marks riskier borrowers (lift 1.7–2.2). But *among borrowers the model
already rates green*, flagged borrowers have a difficulty rate of only 5.8–8.6%, around the
average. The model already uses the same history, so most of each flag's signal is in the PD.

**Decision:** flags are **shown as context** on every alert, but by default they **do not raise
the level** (`flag_escalation: false`). Escalation stays available as a configurable option.

**Alert levels, test split** (evaluated once; nothing was tuned on test):

| Level (PD only, the default) | Share of borrowers | Difficulty rate | Share of all difficulty cases at this level or above |
|---|---:|---:|---:|
| Green | 76.9% | 4.0% | 100% |
| Yellow | 10.2% | 13.8% | 61.6% |
| Orange | 11.3% | 24.5% | 44.3% |
| Red | 1.7% | 48.6% | 10.1% |

- **The rates rise steadily with the level:** 4% → 14% → 24% → 49%, so the levels are
  meaningful.
- **Escalation on the test split, for comparison:** yellow-or-above would rise from 23.1% to
  29.6% of borrowers, but catch only 65.4% of difficulty cases instead of 61.6%. That confirms
  the validation-based decision.
- **Capacity trade-off:** reviewing every orange and red borrower means 13% of applicants and
  catches 44% of difficulty cases.

## 3. Current portfolio (application_test, 48,744 unlabelled applicants)

| Level | Borrowers | Share | Mean PD | With at least 1 red flag |
|---|---:|---:|---:|---:|
| Green (no alert) | 38,102 | 78.2% | 4.0% | 2,998 |
| Yellow | 5,060 | 10.4% | 13.0% | 783 |
| Orange | 5,010 | 10.3% | 23.8% | 1,147 |
| Red | 572 | 1.2% | 48.3% | 230 |

**Alerts raised: 10,642.**

- **Storage:** alerts are kept in `data/processed/alerts/current_alerts.parquet`, and
  `scored_portfolio.parquet` holds every borrower's score for the dashboard.
- **Example (red):** borrower 215910, PD 80.9%. Trigger: "PD 80.9% ≥ red threshold 40.0%".
  Drivers: EXT_SOURCE_MEAN (+0.74), EXT_SOURCE_3 (+0.47), CREDIT_ANNUITY_RATIO (+0.33).
  Action: senior credit review, where the model output alone is not a decision.

## 4. Alert history and de-duplication

`AlertStore` keeps `data/processed/alerts/alert_history.parquet`. The `alert_key` is a hash of
borrower, level, triggers and model version, and history is handled as follows:
- **Same alert:** if the same alert recurs, it is not added again; its `last_seen` date is
  updated.
- **Resolved alert:** if a borrower's alert disappears or changes, the old entry is closed
  (`closed_at`) and any new one is added.

**Observed on real data:**
- **Same rules run twice:** the second run added 0 new alerts and left 13,640 unchanged.
- **Escalation switched off:** 3,781 escalation-only alerts were closed, and 783 borrowers moved
  to a lower level, which counted as new alerts. The history records the rule change rather
  than overwriting it.

## 5. Deterioration monitoring: SIMULATION ONLY

A real deterioration alert ("this borrower moved from yellow to orange") needs repeated scoring
of the same borrowers over time. **Home Credit has no such snapshots, so no real deterioration
monitoring is possible.**

**What is implemented instead:**
- **`compare_snapshots()`** is ready for real future snapshots, and unit-tested.
- **A synthetic demonstration** is stored separately as
  `data/processed/alerts/SIMULATED_deterioration.parquet`. Every row is marked
  `data_type = SYNTHETIC`, and none of it goes into the real alert history.

**SYNTHETIC scenario:** 10% of portfolio borrowers (4,807, chosen at random with seed 42) have
their income cut by 20%. The income ratios are recomputed, then everyone is rescored.

**Result:**
- **Level changes:** only 65 borrowers worsened (28 green → yellow, 33 yellow → orange,
  4 orange → red), all of them among the shocked borrowers.
- **Average PD change:** +0.15 percentage points for the shocked borrowers.

**Interpretation:** this model is **not very sensitive to income alone**. Income-based features
are not among its main drivers (Phase 6); external scores and credit history are. This matters
for stress testing (Phase 9): an income shock pushed through this model will move portfolio
risk only a little. That is a property of the model, not evidence about real borrowers.

## 6. Limitations

- **The thresholds and flags are defensible project settings, not validated bank policy.** The
  red threshold and flag cut-offs are conventions, supported by the measurements above.
- **Each alert is a single snapshot taken at the application date.** There is no time dimension.
- **The levels inherit the model's limitations** (see `docs/calibration_and_evaluation.md`).
- **The recommended actions are generic review steps.** A real lender would define them in its
  credit policy.
