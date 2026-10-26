# RiskLens AI

AI-driven loan portfolio risk monitoring and early warning with explainable machine learning.
This is an academic decision-support project. It is **not** a lending authority and does not
guarantee future defaults.

## Status

| Phase | Status |
|---|---|
| 0. Requirements and feasibility | Done (plan agreed) |
| 1. Environment and project structure | Done (all 10 dataset files in `data/raw/`) |
| 2. Dataset audit | Done: `python -m scripts.run_audit` → [docs/dataset_audit.md](docs/dataset_audit.md), findings in [docs/dataset_audit_findings.md](docs/dataset_audit_findings.md) |
| 3. Feature engineering and modelling table | Done: `python -m scripts.build_dataset` → `data/processed/`, [docs/data_dictionary.md](docs/data_dictionary.md), [docs/feature_engineering.md](docs/feature_engineering.md) |
| 4. Model training and comparison | Done: `python -m scripts.train_models` → champion LightGBM, challenger XGBoost; [docs/model_training.md](docs/model_training.md) |
| 5. Calibration and final test evaluation | Done: `python -m scripts.calibrate_and_evaluate` → deployed model in `models/current_model.json` (test ROC-AUC 0.788, PR-AUC 0.290); [docs/calibration_and_evaluation.md](docs/calibration_and_evaluation.md) |
| 6+ | Not started |

## Setup (Windows PowerShell)

```powershell
cd C:\Users\suren\Documents\RiskLens_AI
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m scripts.smoke_test
python -m pytest
```

If `Activate.ps1` is blocked, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or skip activation and call
`.\.venv\Scripts\python.exe` directly.

## Data

Home Credit Default Risk (Kaggle competition). Download it from the competition's Data tab after
accepting the rules, then unzip the CSVs into `data/raw/`. Raw files are never modified and are
not committed to Git. Check them with `python -m scripts.list_raw_files`.

## Layout

- `config/config.yaml`: every threshold and assumption (seed, splits, risk bands, ...)
- `src/`: all pipeline logic (importable and tested)
- `scripts/`: command-line entry points
- `notebooks/`: exploration and presentation only
- `tests/`: pytest suite
- `docs/`: methodology, assumptions, experiment log
- `models/`, `data/`: local only (git-ignored)
