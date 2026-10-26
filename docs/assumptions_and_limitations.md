# Assumptions and limitations

Each item is labelled **Decision** (a project choice), **Assumption** (a value not observed in the data)
or **Expected limitation** (to be confirmed by the Phase 2 audit).

## Decisions (Phase 0)
- Stratified random split of `application_train`: 60% train / 15% validation / 10% calibration / 15% test, seed 42.
  There are no absolute application dates, so a temporal split is not possible. Random splits likely overstate
  real-world performance.
- Risk bands 0–20–40–60–80–100% are provisional presentation bands, not banking standards (see `config/config.yaml`).
- One scoring function (`src/models/predict.py`, Phase 4) is used by the dashboard, stress tests, counterfactuals and API.

## Expected limitations (to verify in Phase 2)
- `TARGET` is documented as "payment difficulties", not legal default.
- No event time or censoring for current loans: survival analysis on the main target is not supported.
- No repeated observations after origination: deterioration monitoring can only be simulated, and is labelled as such.
- No observed LGD; `AMT_CREDIT` is at best an origination-time proxy for EAD. Expected loss will be assumption-based.
- No data to estimate default correlation: the Monte Carlo correlation is an assumption.
- No chronology or labels in `application_test`: drift checks are an offline comparison only.
