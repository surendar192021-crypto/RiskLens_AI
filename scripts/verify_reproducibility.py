"""End-to-end reproducibility check.  Run:  python -m scripts.verify_reproducibility

Exit code 0 only if every check passes.
"""

import sys

from src.reproducibility import main

if __name__ == "__main__":
    sys.exit(main())
