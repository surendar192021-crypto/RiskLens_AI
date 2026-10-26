"""Calibrate the champion, choose the threshold, evaluate once on test.

Run:  python -m scripts.calibrate_and_evaluate
"""

import sys

from src.models.final_evaluation import main

if __name__ == "__main__":
    sys.exit(main())
