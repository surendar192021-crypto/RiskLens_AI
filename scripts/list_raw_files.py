"""List files in data/raw with sizes and header column counts (no full load).

Run from the project root:  python -m scripts.list_raw_files
"""

import csv
import sys

from src.config import load_config, project_path

EXPECTED = [
    "application_train.csv", "application_test.csv", "bureau.csv", "bureau_balance.csv",
    "previous_application.csv", "installments_payments.csv", "POS_CASH_balance.csv",
    "credit_card_balance.csv", "sample_submission.csv", "HomeCredit_columns_description.csv",
]


def main() -> int:
    raw = project_path(load_config(), "raw_data")
    files = sorted(p for p in raw.iterdir() if p.is_file() and p.name != ".gitkeep")
    if not files:
        print(f"No data files in {raw}. Download the dataset first.")
        return 1
    print(f"{'file':<38}{'size (MB)':>11}{'columns':>9}")
    for p in files:
        cols = ""
        if p.suffix.lower() == ".csv":
            # HomeCredit_columns_description.csv is not UTF-8, so read leniently.
            with p.open(encoding="utf-8", errors="replace", newline="") as f:
                cols = len(next(csv.reader(f)))
        print(f"{p.name:<38}{p.stat().st_size / 1e6:>11.1f}{cols:>9}")
    names = {p.name for p in files}
    missing = [n for n in EXPECTED if n not in names]
    print("All expected files present." if not missing else f"Missing: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
