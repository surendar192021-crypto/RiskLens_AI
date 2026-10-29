# RiskLens AI

**AI-driven loan portfolio risk monitoring and early warning with explainable machine learning.**
An academic decision-support project on the Home Credit Default Risk data. It is **not** a lending
authority: probabilities estimate *early payment difficulty* in a sample, and no output is a
credit decision, a guarantee or a regulatory risk measure.

## What it does

| Question | Component | Status |
|---|---|---|
| Who is risky, and how risky? | LightGBM PD model (test ROC-AUC 0.788, PR-AUC 0.290), risk bands | Measured |
| How reliable are the probabilities? | Calibration study (none needed; ECE about 0.002 on test) | Measured |
| Why is a borrower risky? | SHAP, verified exact; plain-English summaries | Measured |
| What could change the prediction? | Constrained counterfactuals (loan terms only) | Model-based hypotheticals |
| When does risk materialise? | Survival analysis of *previous* loans; delinquency migration | Measured (current loans: not supported by the data) |
| Who needs review? | Early-warning alerts with triggers, drivers and history | Rule-based on measured PD; deterioration = synthetic demo |
| What is happening to the portfolio? | Segments, concentration, sum of PD checked against outcomes | Measured |
| How much could be lost? | Expected loss, Monte Carlo VaR/ES, stress tests | **Assumption-based** (LGD, ρ) |
| Is the model still reliable? Is it fair? | PSI drift, fairness by gender and age, registry, audit log | Measured (offline) |

The full list of what is measured, simulated, assumed or unsupported is in
[docs/assumptions_and_limitations.md](docs/assumptions_and_limitations.md).

## Quick start (Windows PowerShell)

```powershell
cd C:\Users\suren\Documents\RiskLens_AI
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m scripts.smoke_test
```

If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or
call `.\.venv\Scripts\python.exe` directly.

**Data.** Download "Home Credit Default Risk" from Kaggle: join the competition, accept its
rules, then use Data → Download All. Unzip the CSVs into `data/raw/` and check them with
`python -m scripts.list_raw_files`. Raw files are never modified and are not committed to Git.

**Run the pipeline** (in this order, about 35 minutes in total on a 16 GB laptop):

| Step | Command | Time |
|---|---|---|
| Dataset audit | `python -m scripts.run_audit` | 1 min |
| Modelling table and splits | `python -m scripts.build_dataset` | 1.5 min |
| Train and compare 6 models | `python -m scripts.train_models` | 5.5 min |
| Calibration, threshold, test evaluation | `python -m scripts.calibrate_and_evaluate` | 1 min |
| Accuracy of all models (reference only) | `python -m scripts.accuracy_report` | 5 s |
| SHAP and counterfactuals | `python -m scripts.run_explainability` | 2 min |
| Early-warning alerts | `python -m scripts.run_early_warning` | 15 s |
| Survival study (previous loans) | `python -m scripts.run_survival` | 21 min |
| Portfolio and expected loss | `python -m scripts.run_portfolio` | 5 s |
| Monte Carlo and stress tests | `python -m scripts.run_simulation` | 1.5 min |
| Delinquency migration | `python -m scripts.run_migration` | 15 s |
| Drift, fairness, registry, audit log | `python -m scripts.run_monitoring` | 1 min |
| Reproducibility check | `python -m scripts.verify_reproducibility` | 2 min |

Then run the dashboard and the tests:

```powershell
streamlit run dashboard/app.py
python -m pytest
python -m ruff check .
```

## Verified status

- **Tests:** the pytest suite passes. It covers data, features, models, calibration, risk, the
  simulation's analytic checks, monitoring and the dashboard.
- **Lint:** clean.
- **Reproducibility:** 28 of 28 checks pass.
  - **Raw data:** the files match the audit checksums.
  - **Dataset rebuild:** identical to the saved one.
  - **Model rebuild:** retraining gives identical test predictions.
  - **Recorded results:** test metrics, portfolio figures and Monte Carlo results reproduce exactly.

  See [docs/final_report.md](docs/final_report.md) §7.1.

| Phase | Document |
|---|---|
| 0–2 Requirements, setup, dataset audit | [dataset_audit.md](docs/dataset_audit.md), [dataset_audit_findings.md](docs/dataset_audit_findings.md) |
| 3 Features and modelling table | [feature_engineering.md](docs/feature_engineering.md), [data_dictionary.md](docs/data_dictionary.md) |
| 4 Model comparison | [model_training.md](docs/model_training.md) |
| 5 Calibration and test evaluation | [calibration_and_evaluation.md](docs/calibration_and_evaluation.md) |
| 6 SHAP and counterfactuals | [explainability.md](docs/explainability.md) |
| 7 Survival analysis and early warning | [survival_analysis.md](docs/survival_analysis.md), [early_warning.md](docs/early_warning.md) |
| 8 Portfolio and expected loss | [portfolio_risk.md](docs/portfolio_risk.md) |
| 9 Simulation, stress testing, migration | [simulation_and_stress.md](docs/simulation_and_stress.md) |
| 10 Monitoring and fairness | [monitoring_and_fairness.md](docs/monitoring_and_fairness.md) |
| 11 Dashboard | [dashboard.md](docs/dashboard.md) |
| 12 Delivery | [final_report.md](docs/final_report.md), [architecture.md](docs/architecture.md), [viva_questions.md](docs/viva_questions.md), [presentation_outline.md](docs/presentation_outline.md), [experiment_log.md](docs/experiment_log.md) |

## Layout

```
config/config.yaml   every threshold, assumption and parameter (validated)
src/                 pipeline logic: data, models, explainability, survival, risk,
                     simulation, monitoring, reproducibility
scripts/             command-line entry points (python -m scripts.<name>)
dashboard/           Streamlit app (app.py + 8 pages)
tests/               pytest suite
docs/                methodology, results, assumptions, report, viva preparation
reports/             generated tables and figures (committed)
data/, models/       local only (git-ignored): raw and processed data, model files,
                     registry, audit log
```

## Not included

- **The optional FastAPI service** was not built.
- **The planned Jupyter notebooks** were replaced by tested modules and scripts.
- **Time-to-default for current borrowers** and **real deterioration monitoring** are not
  supported by this dataset (it has no follow-up data); this is documented rather than simulated
  as if real.
