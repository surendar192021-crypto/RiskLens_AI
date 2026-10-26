"""Train and compare all configured models.  Run:  python -m scripts.train_models"""

import sys

from src.models.train import main

if __name__ == "__main__":
    sys.exit(main())
