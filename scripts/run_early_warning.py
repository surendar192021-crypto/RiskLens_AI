"""Score the portfolio and raise early-warning alerts.  Run:  python -m scripts.run_early_warning"""

import sys

from src.risk.run_early_warning import main

if __name__ == "__main__":
    sys.exit(main())
