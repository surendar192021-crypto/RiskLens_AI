# RiskLens AI dashboard (Phase 11)

## Running it

From the project root, with the virtual environment active:

```powershell
streamlit run dashboard/app.py
```

Then open http://localhost:8501. The first page load takes a few seconds while the model and
portfolio are loaded and scored. After that they are cached.

**Prerequisite:** the pipeline outputs must exist. If any are missing, each page lists the
command that creates them. A full rebuild, in order:

```powershell
python -m scripts.build_dataset
python -m scripts.train_models
python -m scripts.calibrate_and_evaluate
python -m scripts.run_explainability
python -m scripts.run_early_warning
python -m scripts.run_survival
python -m scripts.run_portfolio
python -m scripts.run_simulation
python -m scripts.run_migration
python -m scripts.run_monitoring
```

## Design rules

- **No training and no hard-coded results.** The dashboard loads the deployed model
  (`models/current_model.json`) and computes everything with the same `src/` functions the
  analysis scripts use: scoring, alert levels, SHAP, counterfactuals, segment tables, expected
  loss, Monte Carlo and stress scenarios. The results it reads from files instead are the saved
  outputs of studies that are too slow to run interactively: global SHAP, drift, fairness and
  survival.
- **Caching.** The model and SHAP explainer use `st.cache_resource`; the portfolio, scores and
  tables use `st.cache_data`. Moving a slider never reloads the model or rescores the whole
  portfolio unless the inputs to that calculation change.
- **Labelling.** Assumption-based numbers (LGD, ρ, stress scenarios) and synthetic data (the
  deterioration demo) are marked as such on the page where they appear. No page shows a
  survival curve for current borrowers.

## Pages

| Page | What it shows | Computed live or saved |
|---|---|---|
| Portfolio Overview | KPIs, risk bands, alert levels, PD distribution | Live |
| Portfolio Analytics | Filters, segment tables, concentration, EL with an LGD slider, check against observed outcomes | Live (the outcome check is saved, Phase 8) |
| Simulation & Stress Testing | User-chosen simulations, seed, ρ, LGD and shocks; baseline vs stressed loss distributions, VaR and ES; a warning when the model's response is implausible | Live |
| Borrower Analysis | PD, band, alert and triggers, SHAP chart and plain-English summary, counterfactual search on request | Live |
| Early Warning Center | Alerts with filters, history, rule evidence, SYNTHETIC deterioration tab | Saved alerts (Phase 7) and saved evaluations |
| Model Explainability | Global SHAP, SHAP vs split gain, beeswarm, counterfactual feasibility | Saved (Phase 6) |
| Monitoring & Fairness | Drift, schema, performance, fairness, flip test, without-gender experiment, registry with live hash check, audit-log chain check | Saved (Phase 10); hash and chain checks live |
| Methodology & Model Info | Data, model, metrics, calibration, temporal study, assumptions, limitations, implementation status | Saved |

## Verification

`tests/test_dashboard.py` uses Streamlit's `AppTest` to run every page headlessly. It checks:
- **All 8 pages run without exceptions**, and every page appears in the navigation.
- **The entry point opens the overview.**
- **The overview's numbers are right:** borrowers, mean PD, expected cases and flagged count
  equal an independent recalculation with `src/` functions.
- **The simulation page completes a 500-run simulation** from its form.
- **The borrower page's counterfactual button works**, and its result carries the "hypothetical"
  label.

**Manual browser check (27 Oct 2026, 1280 × 900):**
- **Overview:** shows 48,744 borrowers, 7.51% mean PD, 3,659 expected cases, 5,582 above the
  threshold and 800.9M EL, all matching Phase 8.
- **Borrower page:** borrower 215910 shows PD 80.9% and RED, matching Phase 7.
- **Simulation page:** the default run reproduces Phase 9 (baseline VaR 99.9% = 2.90B; stressed
  EL 1.19B).
- **Errors:** no browser console errors and no server errors.

## Known limitations

- **Single user, local only.** There is no authentication, so it is not suitable for real
  borrower data in this form.
- **Some pages show saved study results.** The Explainability, Monitoring and Methodology pages
  show the outputs of the latest pipeline runs; re-run the scripts to refresh them.
- **Counterfactual search takes a moment.** It scores 77 candidates through the model, about
  1 second.
