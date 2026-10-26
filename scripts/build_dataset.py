"""Build the modelling table and splits.  Run:  python -m scripts.build_dataset"""

import sys

from src.data.build_dataset import main

if __name__ == "__main__":
    sys.exit(main())
