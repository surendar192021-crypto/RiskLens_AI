# Assumptions and limitations (final, Phase 12)

The single place that lists what RiskLens AI assumes and what it cannot do. Each item is labelled:
- **Decision:** a project choice, with its reason.
- **Assumption:** a value not observed in the data.
- **Limitation:** something the data or methods do not support.

The phase documents give the evidence for each item.

## 1. What each output is

| Output | Status | Basis |
|---|---|---|
| Borrower PD, risk band, model metrics, calibration | **Measured.** Test metrics come from a single evaluation on held-out data | Phase 4–5 |
| SHAP explanations | **Measured and verified exact.** Describes the model, not causes | Phase 6 |
| Counterfactuals | **Model-based hypotheticals** | Phase 6 |
| Survival curves and Cox model | **Measured,** for *previous* loans only | Phase 7 |
| Early-warning alerts | **Rule-based on measured PD.** Rules evaluated on labelled data | Phase 7 |
| Deterioration alerts | **SYNTHETIC demo only** | Phase 7 |
| Portfolio aggregates (sum of PD, segments) | **Measured.** Checked against observed outcomes on the test split | Phase 8 |
| Expected loss | **Assumption-based** (LGD assumed, EAD proxy) | Phase 8 |
| Monte Carlo VaR and ES | **Assumption-based** (ρ and LGD assumed) | Phase 9 |
| Stress tests | **Hypothetical feature perturbations.** Two scenarios are not economically credible | Phase 9 |
| Delinquency migration matrix | **Measured** (bureau credits). Projecting it forward is invalid | Phase 9 |
| Drift | **Measured** as an offline comparison. Not live monitoring | Phase 10 |
| Fairness metrics | **Measured** on the test split. No fairness or compliance claim | Phase 10 |

## 2. Decisions

| Decision | Reason |
|---|---|
| Stratified random 60/15/10/15 split, seed 42 | No dates exist, so a temporal split is impossible |
| No class weighting, no resampling | Weighting gave no ranking gain and inflated PDs about 5× (Phase 4) |
| Champion = LightGBM | Highest validation PR-AUC; tied with XGBoost within noise, so faster training and native SHAP decided it |
| No calibration | Sigmoid and isotonic did not improve validation Brier or ECE (Phase 5) |
| Operating threshold 0.162 (max-F1, validation) | A policy choice. The alternatives (cost-based, recall target) are documented |
| Count features 0 when no history; other aggregates missing, with `HAS_*` flags | Zero is a true count; "no history" is not zero debt |
| Flag escalation off | On validation, flagged PD-green borrowers had a near-average difficulty rate (Phase 7) |
| Deployed model keeps `CODE_GENDER` | Removing it trades error-rate gaps for worse calibration. That is left to a decision-maker (Phase 10) |
| Thresholds and rules never tuned on the test split | Test results stay unbiased |

## 3. Assumptions (not observed in the data)

| Assumption | Value | Effect if wrong |
|---|---|---|
| LGD | 45% (Basel II foundation-IRB senior unsecured reference) | EL is linear in LGD: 25–85% gives 445M–1,513M |
| EAD | AMT_CREDIT, fully outstanding at default | Overstates exposure for amortising loans |
| Asset correlation ρ | 0.10 | VaR 99.9% ranges 0.84B–4.98B for ρ 0–0.25 |
| PD horizon | That of TARGET (first instalments) | Not a one-year PD |
| Risk bands | 0/20/40/60/80/100% | Presentation only; 92% of borrowers fall in "Very Low" |
| Alert levels | yellow 10.5%, orange 16.2%, red 40% | Derived from validation results; still a policy choice |
| Cost ratio for the min-cost threshold | FN = 5 × FP | Illustrative only |
| Counterfactual annuity rule | No interest rate | Longer terms look cheaper than they are |
| PSI thresholds | 0.10 / 0.25 | An industry rule of thumb, not a test |

## 4. Limitations

**Data**
- **TARGET is "payment difficulty in the first Y instalments".** X and Y are undisclosed. It is
  not legal default, and the sample's 8.07% rate is not a real portfolio's rate.
- **No calendar dates.** Performance on *future* applicants is untested.
- **No follow-up of the current loan.** So there is no survival analysis or real deterioration
  monitoring for current borrowers.
- **No recoveries, so no observed LGD.** The currency of the amounts is not stated.
- **`application_test` differs from the training population** on several inputs (Phase 10) and
  has no labels. Performance on it cannot be measured.

**Model**
- **Associations are non-causal and not monotone.** A 20% income cut lowers the PD for 32% of
  borrowers. This invalidates income and instalment stress tests and makes some counterfactuals
  counter-intuitive.
- **The strongest drivers are opaque.** The external scores' construction is not documented.
- **No hyper-parameter search, one seed, one split.**
- **Gender has direct weight** (flip test: 1,520 test flag decisions change).

**Methods**
- **The time-homogeneous Markov assumption fails** for the delinquency matrix.
- **The Cox model's proportional-hazards assumption fails** for two covariates.
- **The single-factor Gaussian model** ignores sector and regional clustering.
- **Normal-approximation intervals** for fairness differences, and independence assumed in the
  expected-count range.

**Engineering**
- **The dashboard** is single-user and local, with no authentication.
- **The audit log** detects tampering but cannot prevent it.
- **The optional FastAPI was not implemented.**
- **The Jupyter notebooks in the original plan were not created.** All logic lives in tested
  `src/` modules and scripts instead.
