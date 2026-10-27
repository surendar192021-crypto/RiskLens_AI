# Presentation outline (about 15 slides, 15–20 minutes)

Each slide lists its message, then the evidence or figure to show. All figures are in
`reports/figures/`.

1. **Title.** RiskLens AI: loan portfolio risk monitoring and early warning with explainable ML.
   Name, programme, date.
2. **Problem.** Most credit projects stop at "who defaults". Risk management also needs *how
   reliable*, *why*, *what if* and *what is happening to the portfolio*. The research questions
   RQ1–RQ6 are shown as six short lines.
3. **Data.** Home Credit, 9 tables, 307,511 labelled applicants (8.07% with payment difficulty).
   What the data can and cannot support, using the feasibility table from
   `docs/dataset_audit_findings.md` §8.
4. **Pipeline.** The architecture diagram (`docs/architecture.md`). One scoring path, four
   single-purpose splits, everything configurable.
5. **Features.** 155 features from 7 tables, built without multiplying rows. Univariate
   screening: the external scores dominate.
6. **Model comparison (RQ1).** `phase4_pr_validation.png`, then the test table: LightGBM 0.788 /
   0.290 vs logistic regression 0.774 / 0.268, with paired confidence intervals.
7. **Calibration (RQ2).** `phase5_reliability_validation.png`. "None" was the right answer.
   Contrast it with weighted models inflating PD 5×.
8. **Risk bands and thresholds.** `phase5_pd_histogram_test.png` and the band table. 91% of
   borrowers fall in "Very Low", and predicted rates match observed rates in each band.
9. **Explainability (RQ3).** `phase6_shap_beeswarm.png` plus one waterfall chart. SHAP is
   verified exact. Gender ranks 5th.
10. **Counterfactuals (RQ4).** The feasibility-by-PD table: 87% / 21% / 0%. The shorter-loan
    pattern is a learned association, not advice.
11. **Temporal (RQ5).** `phase7_km_by_target.png`; Cox concordance 0.728; the PH violation is
    stated.
12. **Early warning and portfolio.** Alert levels on test (4% / 14% / 24% / 49% observed rates).
    Expected vs observed cases: 3,704 vs 3,724. EL with its LGD sensitivity.
13. **Simulation and stress (RQ6).** `phase9_loss_distribution.png` and the ρ sensitivity
    (VaR 0.84B → 4.98B). Stress results, including the implausible responses.
14. **Monitoring and fairness.** `phase10_drift_top_features.png`. Error rates by gender and age;
    the flip test; the without-gender trade-off.
15. **Live demo.** The dashboard: overview, then a borrower page (SHAP and counterfactual), then
    the simulation page.
16. **Contributions, limitations, future work.** Be honest: no novel algorithm, and several
    negative findings. The key limitations come from `docs/assumptions_and_limitations.md`.

**Demo tip:** start the dashboard (`streamlit run dashboard/app.py`) before the talk; the first
load takes a few seconds. Have borrower 215910 (PD 80.9%, red) ready, and use "Just above review
threshold" for the counterfactual.
