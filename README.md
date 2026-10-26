# RiskLens AI

AI-driven loan portfolio risk monitoring and early warning with explainable machine learning.
This is an academic decision-support project. It is **not** a lending authority and does not
guarantee future defaults.

## Status

| Phase | Status |
|---|---|
| 0. Requirements and feasibility | Done (plan agreed) |
| 1. Environment and project structure | In progress (dataset not yet in `data/raw/`) |
| 2+ | Not started |

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
