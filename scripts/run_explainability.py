"""Global/local SHAP and counterfactual study.  Run:  python -m scripts.run_explainability"""

import sys

from src.explainability.run import main

if __name__ == "__main__":
    sys.exit(main())
