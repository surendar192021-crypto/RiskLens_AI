# Architecture (as built)

```
config/config.yaml ── every threshold, assumption, seed and parameter (validated by src/config.py)

data/raw/ (Kaggle CSVs, read-only, checksummed)
   │  scripts.run_audit          → docs/dataset_audit.md, reports/tables/audit_*.csv
   ▼
src/data/       load → clean → features → one-row-per-applicant aggregates → splits
   │  scripts.build_dataset      → data/processed/{modelling_train, portfolio_test, splits}.parquet
   ▼
src/models/     FeatureFrame schema guard → LR / RF / XGBoost / LightGBM
   │  scripts.train_models       → models/phase4_*/ , comparison on validation
   │  scripts.calibrate_and_evaluate → calibration choice, threshold, ONE test evaluation
   │                                 → models/current_model.json  (the deployed model)
   ▼
src/models/predict.py  ← the single scoring path used by every consumer below
   ├─ src/explainability/  SHAP (verified exact) and constrained counterfactuals
   ├─ src/risk/            risk bands, early warning + alert store, portfolio analytics,
   │                       expected loss, delinquency migration
   ├─ src/survival/        Kaplan-Meier and Cox for previous loans
   ├─ src/simulation/      scenarios, stress tests, Vasicek Monte Carlo
   └─ src/monitoring/      drift (PSI), fairness, model registry, hash-chained audit log
   ▼
dashboard/      Streamlit (8 pages), cached, reads models/ + data/processed/ + reports/
tests/          pytest: unit tests on synthetic data, plus checks on real outputs
src/reproducibility.py   end-to-end re-derivation of key results (Phase 12)
```

## Principles

1. **One scoring path.** `src/models/predict.py::predict_proba` is used by the evaluation
   scripts, the explanations, counterfactuals, stress tests, early warning, the portfolio
   analytics and the dashboard. The first pipeline step, `FeatureFrame`, fixes the feature order
   and the category levels, and fails clearly if a feature is missing.
2. **Shared feature formulas.** Derived features are recomputed by the functions used to build
   the dataset (`loan_ratio_features`, `ext_source_features`, `employment_features`).
   Counterfactuals and stress scenarios therefore can't drift out of sync with training.
3. **One purpose per data split.** Train fits the models. Validation selects models, stops
   boosting and chooses thresholds. Calibration fits the calibrators. Test is used once, for the
   final numbers.
4. **Configuration, not constants.** Every assumption (LGD, ρ, thresholds, bands, flag rules,
   PSI cut-offs) is in `config.yaml`, with its source noted.
5. **Outputs can be traced.**
   - Every script writes a JSON summary and CSV tables.
   - Experiment runs are logged in `docs/experiment_log.md`.
   - Models are registered with checksums in `models/registry.json`.
   - Events go into a hash-chained `models/audit_log.jsonl`.

## Changes from the Phase 0 plan

| Planned | As built | Reason |
|---|---|---|
| `notebooks/01..08` | Not created; logic lives in `src/` plus `scripts/` | Easier to test and reproduce; the results are in `docs/` |
| `src/models/train.py` only | Added `transformers.py`, `calibrate.py`, `final_evaluation.py` | Saved models need stable import paths; Phase 5 needed its own step |
| `src/data/splits.py` | Added | The split logic is tested on its own |
| `dashboard/app.py` + `pages/` | `app.py` is a navigation router (`st.navigation`) | Proper page titles and grouping |
| `api/main.py` (optional) | Not implemented | Kept optional by the plan; the dashboard covers the use cases |
| `src/reproducibility.py` | Added | Phase 12 verification |

## Environment

- **Python and core packages:** Python 3.13.4 in `.venv`. Exact versions are in
  `requirements-lock.txt`.
- **pandas 2.3.3:** `lifelines` requires pandas < 3. Installing it in Phase 7 downgraded pandas
  from 3.1.0, which ran Phases 1–6. Phase 12 found the dataset build no longer worked under
  pandas 2.3.3, fixed it (`_merge_on_index`), and added a regression test.
