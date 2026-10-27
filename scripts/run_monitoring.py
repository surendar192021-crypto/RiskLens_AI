"""Drift, fairness, registry and audit log.  Run:  python -m scripts.run_monitoring"""

import sys

from src.monitoring.run_monitoring import main

if __name__ == "__main__":
    sys.exit(main())
