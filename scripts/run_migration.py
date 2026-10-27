"""Delinquency migration of bureau credits.  Run:  python -m scripts.run_migration"""

import sys

from src.risk.run_migration import main

if __name__ == "__main__":
    sys.exit(main())
