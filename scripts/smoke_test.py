"""Phase 1 smoke test: confirm every core package imports and does basic work.

Run from the project root:  python -m scripts.smoke_test
"""

import importlib
import sys

PACKAGES = [
    "numpy", "pandas", "scipy", "pyarrow", "sklearn", "lightgbm", "xgboost",
    "shap", "plotly", "matplotlib", "streamlit", "yaml", "joblib", "pytest",
]


def main() -> int:
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    failed = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  OK   {name:<11} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # report every failure, not just the first
            print(f"  FAIL {name:<11} {exc}")
            failed.append(name)

    if not failed:
        # Tiny end-to-end check: fit a LightGBM model on synthetic data and explain it.
        import lightgbm as lgb
        import numpy as np
        import shap

        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 4))
        y = (X[:, 0] + rng.normal(scale=0.5, size=500) > 0).astype(int)
        model = lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y)
        p = model.predict_proba(X)[:, 1]
        sv = shap.TreeExplainer(model).shap_values(X[:5])
        assert ((p >= 0) & (p <= 1)).all()
        print(f"  OK   synthetic LightGBM + SHAP check (SHAP array shape {np.shape(sv)})")

    from src.config import load_config

    cfg = load_config()
    print(f"  OK   config loaded (seed={cfg['project']['random_seed']})")

    print("SMOKE TEST PASSED" if not failed else f"SMOKE TEST FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
